# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from vLLM v0.30.0's DFlash2 drafter
# (vllm/model_executor/models/qwen3_dflash2.py).

"""Serving model: DFlash2 + confidence head (+ the width-conditioned Fore-D head).

Registered as `DFlash2ConfDraftModel` by `registry.py`. The heads hang off the
inner model because `DFlashQwen3ForCausalLM.load_weights` prefixes non-lm_head
keys with `model.`, which is where a speculators checkpoint puts them.
"""

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
    DFlashGroupedConv,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..config import STS_ENV
from ..fore_d.kernels.grouped_conv import grouped_conv
from ..serving.sts import load_temperatures
from .heads import confidence_features, confidence_input_dim, selector_context


def _fp32_linear(
    in_features: int, out_features: int, bias: bool, prefix: str
) -> ReplicatedLinear:
    # fp32 because Fore-D and adaptive verification rank products of these probabilities batch-wide.
    return ReplicatedLinear(
        in_features,
        out_features,
        bias=bias,
        return_bias=False,
        params_dtype=torch.float32,
        prefix=prefix,
    )


class ConfidenceHead(nn.Module):
    def __init__(self, input_dim: int, prefix: str) -> None:
        super().__init__()
        self.proj = _fp32_linear(input_dim, 1, True, maybe_prefix(prefix, "proj"))

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.proj(confidence_features(hidden, context).float()).squeeze(-1)


class ForeDHead(nn.Module):
    """Serving-side twin of `heads.ForeDHead` (same parameter names)."""

    def __init__(
        self, hidden_size: int, rank: int, num_positions: int, prefix: str
    ) -> None:
        super().__init__()
        self.context_projection = _fp32_linear(
            hidden_size, rank, False, maybe_prefix(prefix, "context_projection")
        )
        self.anchor_projection = _fp32_linear(
            hidden_size, rank, False, maybe_prefix(prefix, "anchor_projection")
        )
        self.fusion_bias = nn.Parameter(torch.zeros(rank, dtype=torch.float32))
        self.output_projection = _fp32_linear(
            rank, num_positions, True, maybe_prefix(prefix, "output_projection")
        )
        self.bf16_projections = False

    def use_bf16_projections(self) -> None:
        """Run the two hidden-size projections as bf16 GEMMs accumulating in fp32.

        Their inputs (a normalized hidden state and a token embedding) are bf16
        already; fp32 weights only buy a non-tensor-core GEMM (~16 us each).
        """
        for name in ("context", "anchor"):
            weight = getattr(self, f"{name}_projection").weight.detach()
            self.register_buffer(
                f"{name}_weight_bf16", weight.to(torch.bfloat16), persistent=False
            )
        self.bf16_projections = True

    def forward(
        self, context: torch.Tensor, anchor_embeddings: torch.Tensor
    ) -> torch.Tensor:
        if self.bf16_projections:
            bf16, fp32 = torch.bfloat16, torch.float32
            fused = torch.addmm(
                self.fusion_bias,
                context.to(bf16),
                self.context_weight_bf16.t(),
                out_dtype=fp32,
            )
            fused = torch.addmm(
                fused,
                anchor_embeddings.to(bf16),
                self.anchor_weight_bf16.t(),
                out_dtype=fp32,
            )
        else:
            fused = (
                self.context_projection(context.float())
                + self.anchor_projection(anchor_embeddings.float())
                + self.fusion_bias
            )
        return self.output_projection(F.silu(fused))


class PackedGroupedConv(DFlashGroupedConv):
    """Upstream's conv with the in-request row position read from a buffer.

    Upstream derives it as `row % block_size`, which only holds when every
    request has exactly block_size rows. Packed Fore-D rows carry their position
    in `row_position` (written by the packed-prep kernel), so a request's first
    row never reads its neighbour's last. Identical math otherwise, and
    uncompiled it runs as one fused kernel per call (fore_d/kernels/grouped_conv.py).
    """

    row_position: torch.Tensor

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        if (
            not torch.compiler.is_compiling()
        ):  # compiled, Inductor fuses the eager form below
            return grouped_conv(
                hidden_states,
                delta,
                self.base_kernel[side],
                self.group_size,
                self.row_position,
            )
        blocks = hidden_states.unflatten(-1, (self.num_groups, self.group_size))
        base = self.base_kernel[side].view(
            1, self.taps, self.num_groups, self.group_size
        )
        coefficients = base + delta.unsqueeze(-1)
        output = coefficients[:, 0] * blocks
        position = self.row_position[: hidden_states.shape[0]]
        for tap in range(1, self.taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
        return output.flatten(-2)


class DFlash2ConfModel(DFlash2Qwen3Model):
    def __init__(
        self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        draft = self.config.dflash_config
        hidden = self.config.hidden_size
        self.confidence_head = None
        if draft.get("enable_confidence_head", False):
            self.confidence_head = ConfidenceHead(
                confidence_input_dim(hidden, int(draft["selector_rank"])),
                prefix=maybe_prefix(prefix, "confidence_head"),
            )
        self.fore_d_predictor = None
        if draft.get("enable_fore_d_predictor", False):
            k = vllm_config.speculative_config.num_speculative_tokens
            self.fore_d_predictor = ForeDHead(
                hidden,
                int(draft.get("fore_d_rank", 256)),
                k * (k + 1) // 2,  # one row per width d = 1..K
                prefix=maybe_prefix(prefix, "fore_d_predictor"),
            )

    def use_packed_rows(self, row_position: torch.Tensor) -> None:
        """Switch every conv to packed layout, driven by `row_position`."""
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                conv.__class__ = PackedGroupedConv
                conv.register_buffer("row_position", row_position, persistent=False)


class DFlash2ConfForCausalLM(DFlash2Qwen3ForCausalLM):
    model_cls = DFlash2ConfModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        path = os.environ.get(STS_ENV)
        if path and self.model.confidence_head is not None:
            positions = vllm_config.speculative_config.num_speculative_tokens
            device = next(self.parameters()).device
            temperatures = torch.tensor(
                load_temperatures(path, positions), device=device
            )
            # A buffer, so it follows the module and needs no copy inside a graph.
            self.register_buffer("sts_temperatures", temperatures, persistent=False)

    def confidence_logits(
        self, hidden: torch.Tensor, predecessor_ids: torch.Tensor
    ) -> torch.Tensor:
        """Uncalibrated per-position confidence logits (what calibration fits)."""
        selector = self.model.candidate_selector
        context = selector_context(
            selector.hidden_projection(hidden),
            selector.predecessor_codebook[predecessor_ids],
        )
        return self.model.confidence_head(hidden, context)

    def compute_confidence(
        self,
        hidden: torch.Tensor,
        predecessor_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calibrated acceptance probability: [..., K] for whole blocks, or one per
        packed row (Fore-D) with `positions`, each row's draft position."""
        logits = self.confidence_logits(hidden, predecessor_ids)
        temperatures = getattr(self, "sts_temperatures", None)
        if temperatures is not None:
            logits = logits / (
                temperatures if positions is None else temperatures[positions]
            )
        return torch.sigmoid(logits)

    def compute_fore_d_logits(
        self, combined_context: torch.Tensor, anchor_ids: torch.Tensor
    ) -> torch.Tensor:
        """Pre-draft conditional acceptance logits, [requests, K(K+1)/2]: width d's
        row at offset d(d-1)/2.

        `combined_context` is the fc-projected target hidden state of the last
        accepted token; training applies the same `hidden_norm` to it.
        """
        return self.model.fore_d_predictor(
            self.model.hidden_norm(combined_context),
            self.model.embed_input_ids(anchor_ids),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load, then refuse a checkpoint missing a head its config promises."""
        seen: set[str] = set()
        fore_d = self.model.fore_d_predictor

        def tracked():
            for name, weight in weights:
                for head in ("confidence_head.", "fore_d_predictor."):
                    if head in name:
                        seen.add(name[name.index(head) :])
                if "fore_d_predictor." in name and fore_d is None:
                    continue  # a head we do not serve
                yield name, weight

        loaded = super().load_weights(tracked())
        for head_name in ("confidence_head", "fore_d_predictor"):
            head = getattr(self.model, head_name)
            if head is None:
                continue
            missing = {f"{head_name}.{p}" for p in head.state_dict()} - seen
            if missing:
                raise ValueError(
                    f"checkpoint promises {head_name} but lacks {sorted(missing)}"
                )
        return loaded


EntryClass = DFlash2ConfForCausalLM
