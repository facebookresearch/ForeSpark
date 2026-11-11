# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from vLLM v0.30.0's DSpark drafter
# (vllm/model_executor/models/qwen3_dspark.py).

"""Upstream DSpark with temperature-scaled confidence (registered by `sts.install`)."""

import os

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkForCausalLM

from ..config import STS_ENV
from .sts import load_temperatures


class StsQwen3DSparkForCausalLM(Qwen3DSparkForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        positions = vllm_config.speculative_config.num_speculative_tokens
        param = next(self.parameters())
        temperatures = load_temperatures(os.environ[STS_ENV], positions)
        self.register_buffer(
            "sts_temperatures",
            torch.tensor(temperatures, dtype=param.dtype, device=param.device),
            persistent=False,
        )

    def compute_confidence(
        self, head_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        # Same shape discipline as upstream: this runs inside the captured graph.
        logits = self.model.confidence_head(head_hidden, markov_embed)
        positions = self.sts_temperatures.numel()
        return torch.sigmoid(
            (logits.view(-1, positions) / self.sts_temperatures).view_as(logits)
        )
