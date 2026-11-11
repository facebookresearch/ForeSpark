# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from vLLM v0.30.0's adaptive verification
# (vllm/v1/worker/gpu/spec_decode/adaptive_verification.py).

"""Host-side verification ranking: adaptive verification on targets whose
backends plan from host lengths (the GLM models' sparse MLA, GLM-5.3-Flash's
KDA), where upstream refuses adaptive verification outright.

The per-request prefixes are ranked on the host, so the CPU partition is exact
and no device/host mismatch is left to refuse. They are ranked from upstream's
one-step lagged host snapshot of the confidences, not the current ones: under
async scheduling the host runs a step ahead of the GPU, so reading the current
confidences is a pipeline stall, not a copy (it took 5.1 ms of a 5.3 ms
get_num_tokens; lagged, the cost vanished). The lag costs ordering quality
only; verification keeps it lossless. Costs one ranking per step on the host.

Enabled by FORESPARK_CPU_RANKING (models.yaml `cpu_ranking`).
"""

import dataclasses
import sys

import numpy as np
import torch
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)


def prefix_widths(survival: np.ndarray, valid: np.ndarray, budget: int) -> np.ndarray:
    """Per-request prefix lengths of the global top-`budget` survival slots.

    Stable sort, so ties go to earlier requests/positions and the admitted set
    stays a prefix of every row (rows are nonincreasing).
    """
    if budget <= 0:
        return np.zeros(survival.shape[0], dtype=np.int32)
    winners = np.argsort(-np.where(valid, survival, -np.inf).ravel(), kind="stable")[
        :budget
    ]
    return np.bincount(
        winners // survival.shape[1], minlength=survival.shape[0]
    ).astype(np.int32)


class HostRankingVerificationManager(AdaptiveVerificationManager):
    compact_batch_takes_req_ids = True  # read by the patched model runner

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._widths: dict[str, int] | None = None

    def get_num_tokens(self, num_tokens_per_req, draft_tokens) -> int:
        num_tokens = super().get_num_tokens(num_tokens_per_req, draft_tokens)
        req_ids = list(num_tokens_per_req)
        slots = np.array(
            [self.req_states.req_id_to_index[r] for r in req_ids], dtype=np.int64
        )
        drafts = np.array(
            [len(draft_tokens.get(r, ())) for r in req_ids], dtype=np.int64
        )
        # The snapshot upstream already keeps on the host to choose the budget.
        lagged = self._stale_confidences[self._stale_idx].np[slots]
        survival = np.cumprod(lagged.astype(np.float64), axis=1)
        valid = np.arange(self.num_speculative_steps)[None, :] < drafts[:, None]
        widths = prefix_widths(survival, valid, self._batch_budget[2])
        self._widths = dict(zip(req_ids, widths.tolist(), strict=True))
        return num_tokens

    def compact_batch(
        self, num_draft_tokens_per_req, num_scheduled_tokens, cu_num_logits_np, req_ids
    ):
        budget = self._batch_budget[2]
        if budget == int(num_draft_tokens_per_req.sum()):
            return num_scheduled_tokens, cu_num_logits_np
        widths = np.array(
            [self._widths[r] for r in req_ids], dtype=num_scheduled_tokens.dtype
        )
        cu_num_logits_np = np.zeros_like(cu_num_logits_np)
        np.cumsum(
            widths + self.num_bonus_tokens, out=cu_num_logits_np[1 : len(req_ids) + 1]
        )
        return (
            num_scheduled_tokens - num_draft_tokens_per_req + widths,
            cu_num_logits_np,
        )

    def reallocate_drafts(self, req_ids, idx_mapping):
        (_, non_draft_per_req, budget), self._batch_budget = self._batch_budget, None
        widths, self._widths = self._widths, None
        num_reqs = idx_mapping.shape[0]
        admitted = np.array([widths[r] for r in req_ids], dtype=np.int32)
        non_draft = np.array([non_draft_per_req[r] for r in req_ids], dtype=np.int32)
        capacities = self._batch_draft_capacity[:num_reqs]
        async_tensor_h2d(admitted, out=capacities)
        non_draft_gpu = self._num_non_draft_tokens[:num_reqs]
        async_tensor_h2d(non_draft, out=non_draft_gpu)
        self._cu_num_logits[:1].zero_()
        torch.cumsum(
            capacities + self.num_bonus_tokens,
            dim=0,
            out=self._cu_num_logits[1 : num_reqs + 1],
        )
        self.query_start_loc[:1].zero_()
        torch.cumsum(
            capacities + non_draft_gpu,
            dim=0,
            out=self.query_start_loc[1 : num_reqs + 1],
        )
        self.query_start_loc[num_reqs + 1 :].fill_(int(non_draft.sum()) + budget)
        return self._cu_num_logits[: num_reqs + 1], self.query_start_loc, budget


def relax_gates() -> None:
    """Upstream's adaptive-verification gates, lifted for host-ranked widths.

    * Every backend may see device/host query lengths that differ: the host
      ranking leaves no mismatch.
    * The manager's ALWAYS-cudagraph gate: the GLM backends capture a uniform
      decode shape and replay ragged batches from the device query_start_loc,
      a level the enum lacks. Claimed at the gate only: on the builder it
      changes how vLLM resolves the cudagraph mode.
    * GDN (GLM-5.3-Flash's KDA) reads a capture batch's draft counts off its
      query lengths: adaptive verification captures N requests x 1 token, so it
      recorded the non-spec decode kernels, which replayed verify rows then ran
      (garbage recurrent state). Promise K drafts per row: the spec kernels it
      captures take per-request lengths from the device cu_seqlens.
    """
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.worker.gpu.attn_utils import AttentionCGSupport
    from vllm.v1.worker.gpu.spec_decode import adaptive_verification as av

    AttentionBackend.supports_device_cpu_query_lens_mismatch = classmethod(
        lambda cls: True
    )

    def forced(info):
        return dataclasses.replace(info, min_cg_support=AttentionCGSupport.ALWAYS)

    create = av.maybe_create_adaptive_verification_manager

    def create_forced(*, attn_cg_support, **kwargs):
        support = av.get_attn_cg_support
        av.get_attn_cg_support = lambda *a, **k: forced(support(*a, **k))
        try:
            return create(attn_cg_support=forced(attn_cg_support), **kwargs)
        finally:
            av.get_attn_cg_support = support

    av.maybe_create_adaptive_verification_manager = create_forced
    if (runner := sys.modules.get("vllm.v1.worker.gpu.model_runner")) is not None:
        runner.maybe_create_adaptive_verification_manager = (
            create_forced  # imported by name
        )

    gdn_capture = GDNAttentionMetadataBuilder.build_for_cudagraph_capture

    def gdn_build_for_cudagraph_capture(self, common_attn_metadata):
        if not self.use_spec_decode:
            return gdn_capture(self, common_attn_metadata)
        m = common_attn_metadata
        assert m.num_reqs <= self.decode_cudagraph_max_bs
        assert m.num_actual_tokens <= self.decode_cudagraph_max_bs
        num_accepted_tokens = torch.diff(m.query_start_loc)
        drafts = torch.full_like(m.query_start_loc_cpu[1:], self.num_spec)
        return self.build(0, m, num_accepted_tokens, drafts)

    GDNAttentionMetadataBuilder.build_for_cudagraph_capture = (
        gdn_build_for_cudagraph_capture
    )


def install() -> None:
    """Use the host manager for DSpark too (ForeSpark's speculators pick theirs
    themselves), and let every backend through."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    DSparkSpeculator.verification_manager_cls = HostRankingVerificationManager
    relax_gates()
