# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""One verification decision on every tensor-parallel rank.

Adaptive verification (and Fore-D, whose manager extends it) decides
each step's verify budget on the host and each request's share of it on the
device, every TP rank on its own, from the drafter's confidences. The ranks'
drafters do not produce bitwise-identical confidences, and a near-tie then
flips on some ranks only: two ranks verified 7 drafts where the other six
verified 6, so the ranks ran different batch shapes through the same
collectives and the server died a step later (other runs returned garbage
instead; Fore-D's plan flipped the same way and hung). The confidences
differ on every step; the decisions flip only at near-ties. Rank 0's draft
tokens and confidences are
broadcast before anything reads them, so every rank verifies the same tokens
with the same budget; Fore-D broadcasts its head's logits too
(fore_d/runtime/speculator.py). Rank 0's sampler verdict is broadcast as well:
rejection sampling above temperature 0 reads the drafter's per-rank
probabilities. Collectives of num_reqs x (K + 1) elements or fewer, through the
graph-safe NCCL communicator.

Enabled by FORESPARK_TP_CONSISTENT_VERIFY (the harness sets it at TP > 1 for every
speculative arm).
"""

import os

import torch

from ..config import TP_CONSISTENT_VERIFY_ENV


def rank0_broadcaster():
    """In place: rank 0's tensor on every TP rank (None at TP1 or when off)."""
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    if group.world_size == 1 or not os.environ.get(TP_CONSISTENT_VERIFY_ENV):
        return None
    return lambda tensor: group.device_communicator.broadcast(tensor, 0)


def broadcast_drafts(
    broadcast,
    confidence_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_reqs: int,
) -> None:
    """Overwrite this step's confidences and draft-token rows with rank 0's."""
    if broadcast is None or num_reqs == 0:
        return
    broadcast(confidence_probs[:num_reqs])
    slots = idx_mapping[:num_reqs]
    rows = draft_tokens[slots]
    broadcast(rows)
    draft_tokens[slots] = rows


def broadcast_in_place(broadcast, tensor: torch.Tensor) -> None:
    if tensor.is_contiguous():
        broadcast(tensor)
    else:  # NCCL reads numel elements from data_ptr
        staged = tensor.contiguous()
        broadcast(staged)
        tensor.copy_(staged)


def consistent_propose(propose, broadcast):
    """A speculator's propose whose draft tokens are rank 0's."""

    def wrapper(*args, **kwargs):
        tokens = propose(*args, **kwargs)
        broadcast_in_place(broadcast, tokens)
        return tokens

    return wrapper


def broadcast_verdict(broadcast, outputs, num_draft_tokens: int):
    """Rank 0's sampled tokens and accept/reject counts on a verify step."""
    sampler_output, num_sampled, num_rejected = outputs
    if broadcast is not None and num_draft_tokens:
        for tensor in (sampler_output.sampled_token_ids, num_sampled, num_rejected):
            if tensor is not None:
                broadcast_in_place(broadcast, tensor)
    return outputs


def _patch_sampling(runner_cls) -> None:
    """The runner's sample, with rank 0's verdict (patched once, in workers)."""
    if getattr(runner_cls.sample, "_tp_consistent", False):
        return
    sample = runner_cls.sample

    def sample_consistent(self, hidden_states, input_batch, grammar_output):
        if not hasattr(self, "_tp_broadcast"):
            self._tp_broadcast = rank0_broadcaster()
        outputs = sample(self, hidden_states, input_batch, grammar_output)
        return broadcast_verdict(
            self._tp_broadcast, outputs, input_batch.num_draft_tokens
        )

    sample_consistent._tp_consistent = True
    runner_cls.sample = sample_consistent


def install() -> None:
    import sys

    from vllm.v1.worker.gpu import spec_decode
    from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
        AdaptiveVerificationManager,
    )

    create = spec_decode.init_speculator

    def init_speculator(vllm_config, device):
        # Called by the runner's __init__, so its module is loaded: no runner
        # import at plugin load, where non-worker processes would pay for it.
        _patch_sampling(sys.modules["vllm.v1.worker.gpu.model_runner"].GPUModelRunner)
        speculator = create(vllm_config, device)
        if speculator is not None and (broadcast := rank0_broadcaster()) is not None:
            speculator.propose = consistent_propose(speculator.propose, broadcast)
        return speculator

    spec_decode.init_speculator = init_speculator
    if (runner := sys.modules.get("vllm.v1.worker.gpu.model_runner")) is not None:
        runner.init_speculator = init_speculator  # imported by name

    record = AdaptiveVerificationManager.record_confidences

    def record_confidences(self, confidence_probs, input_batch):
        if not hasattr(self, "_tp_broadcast"):
            self._tp_broadcast = rank0_broadcaster()
        broadcast_drafts(
            self._tp_broadcast, confidence_probs, self.req_states.draft_tokens, input_batch.idx_mapping,
            input_batch.num_reqs,
        )  # fmt: skip
        return record(self, confidence_probs, input_batch)

    AdaptiveVerificationManager.record_confidences = record_confidences
