# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from vLLM v0.30.0's DFlash2 speculator
# (vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py).

"""ForeSpark's drafter in vLLM: under Fore-D, or at fixed k for STS collection."""

import os
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

from ..config import CPU_RANKING_ENV, ForeDConfig, STS_COLLECT_ENV


def create_speculator(vllm_config: VllmConfig, device: torch.device):
    """vLLM's dispatch for `DFlash2ConfDraftModel` (see patches/vllm.patch)."""
    if ForeDConfig.from_env().enabled:
        from ..fore_d.runtime.speculator import ForeDSpeculator

        return ForeDSpeculator(vllm_config, device)
    if vllm_config.speculative_config.enable_adaptive_verification:
        raise ValueError(
            "ForeSpark verifies adaptively under Fore-D; fixed-k serving is for STS collection"
        )
    return DFlash2ConfSpeculator(vllm_config, device)


class DFlash2ConfSpeculator(DFlash2Speculator):
    """Stock DFlash2 drafting at fixed k; with STS collection, the confidence
    logits of every block are recorded (eager only). ForeDSpeculator extends it."""

    _speculator_name = "DFlash2Conf"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # Read by the model runner: `enable_adaptive_verification` creates the
        # manager, `draft_token_confidence_probs` feeds `record_confidences`.
        self.enable_adaptive_verification = (
            self.speculative_config.enable_adaptive_verification
        )
        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32
        )
        self.collect_confidence = bool(os.environ.get(STS_COLLECT_ENV))
        if self.collect_confidence and self.enable_adaptive_verification:
            raise ValueError(
                "confidence collection needs untrimmed blocks: disable adaptive verification"
            )
        if os.environ.get(CPU_RANKING_ENV):
            from .cpu_ranking import HostRankingVerificationManager

            self.verification_manager_cls = HostRankingVerificationManager

    def load_draft_model(
        self, target_model: torch.nn.Module, target_attn_layer_names: set[str]
    ):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        if (
            self.enable_adaptive_verification or self.collect_confidence
        ) and model.model.confidence_head is None:
            raise ValueError("this dflash2_conf checkpoint has no confidence head")
        return model

    def _predecessor_ids(self, num_reqs: int, anchor_ids: torch.Tensor) -> torch.Tensor:
        """The token each draft position was conditioned on: the walk's previous
        choice, or the anchor for position 0. [num_reqs, K]"""
        previous = self.draft_tokens[:num_reqs, : self.num_speculative_steps - 1]
        return torch.cat(
            [anchor_ids.view(num_reqs, 1).to(previous.dtype), previous], dim=1
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if not self.collect_confidence:
            return super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
        # Upstream DFlash2Speculator._generate_draft with the confidence logits
        # recorded; the hidden states and anchors they need are locals there.
        last_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        steps = self.num_speculative_steps
        hidden = last_hidden[self.sample_indices[: num_reqs * steps]].view(
            num_reqs, steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(num_reqs, steps, self.selector_top_k)
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_ids = self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]
        scores = self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden, anchor_ids
        )
        self._sample_path(candidate_ids, scores, num_reqs)
        if self.draft_logits is not None:
            self._cache_draft_logits(candidate_ids, num_reqs * steps)
        from . import sts

        if not sts._is_rank_zero():
            return
        # After the walk: the head conditions on the predecessor it chose.
        slots = self.sample_idx_mapping[: num_reqs * steps].view(num_reqs, steps)[:, 0]
        logits = self.model.confidence_logits(
            hidden, self._predecessor_ids(num_reqs, anchor_ids)
        )
        sts.collector(steps, self.max_num_reqs).record_proposal(slots, logits)
