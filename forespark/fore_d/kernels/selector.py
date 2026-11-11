# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from vLLM v0.30.0's DFlash2 speculator kernels
# (vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py).

"""DFlash2's candidate selector over packed sample rows.

Upstream scores a dense [requests, K, top_k, top_k] edge tensor and walks K
steps per request. Here samples are packed request-major (request i owns
d_i consecutive rows), so the edge scores are the same `_score_edges` math with
the predecessor taken from the previous sample row (or the anchor token at a
request's first row), and the walk stops at d_i.

`sliced_top_k` is upstream's LM-head top-k without the [samples, vocab] logits
block, the largest allocation in the draft's CUDA-graph pool.
"""

import torch
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.logits_processor import _topk
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

LOGITS_BLOCK_BYTES = 256 << 20  # past this, the LM head runs in vocab slices


def sliced_top_k(
    processor, lm_head, hidden: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`processor.get_top_k_tokens(lm_head, hidden, k)`, the logits at most
    LOGITS_BLOCK_BYTES at a time: a running top-k over vocab slices, merged
    in vocab order with the same deterministic top-k, so ties resolve as in
    one pass. (ids, values), values float32 after scale and soft cap."""
    shard = lm_head.shard_indices
    vocab = (
        lm_head.weight.shape[0] - shard.num_org_vocab_padding
    )  # padding rows never qualify
    # Slices as wide as the budget allows: every slice block then has the size
    # of the largest unsliced one, so all tail graphs share one pool block.
    width = LOGITS_BLOCK_BYTES // (hidden.shape[0] * hidden.element_size())
    values = ids = None
    for start in range(0, vocab, width):
        stop = min(start + width, vocab)
        # The last slice can be narrower than k (a few rows at thousands of
        # samples), and FlashInfer's top-k then returns indices past it.
        v, i = _topk(F.linear(hidden, lm_head.weight[start:stop]), min(k, stop - start))
        i = i.to(torch.int64) + start
        if values is not None:
            v, i = torch.cat([values, v], dim=-1), torch.cat([ids, i], dim=-1)
            v, keep = _topk(v, k)
            i = i.gather(-1, keep)
        values, ids = v, i
    ids = ids + shard.org_vocab_start_index
    if lm_head.tp_size > 1:
        values = tensor_model_parallel_all_gather(values, dim=-1)
        ids = tensor_model_parallel_all_gather(ids, dim=-1)
        values, keep = _topk(values, k)
        ids = ids.gather(-1, keep)
    values = values.float()
    if processor.scale != 1.0:
        values = values * processor.scale
    if processor.soft_cap is not None:
        values = torch.tanh(values / processor.soft_cap) * processor.soft_cap
    return ids, values


def logits_fit(lm_head, hidden: torch.Tensor) -> bool:
    return (
        hidden.shape[0] * lm_head.weight.shape[0] * hidden.element_size()
        <= LOGITS_BLOCK_BYTES
    )


def packed_edge_scores(
    selector,  # upstream CandidateSelector (codebooks + hidden_projection)
    candidate_ids: torch.Tensor,  # [S, k]
    unary_logits: torch.Tensor,  # [S, k]
    hidden: torch.Tensor,  # [S, H]
    first_step: torch.Tensor,  # [S] bool: row is its request's first draft position
    anchor_ids: torch.Tensor,  # [S] the request's anchor token (used where first_step)
) -> torch.Tensor:
    """scores[s, p, c] = unary[s, c] + <E_pred[prev_p] * W h_s, E_succ[cand_c]>."""
    projected = selector.hidden_projection(hidden)
    previous = torch.where(
        first_step[:, None],
        anchor_ids[:, None].expand_as(candidate_ids),
        torch.roll(candidate_ids, 1, dims=0),
    )
    predecessors = selector.predecessor_codebook[previous]
    successors = selector.successor_codebook[candidate_ids]
    return unary_logits[:, None, :] + torch.einsum(
        "spr,scr->spc", predecessors * projected[:, None, :], successors
    )


@triton.jit
def _packed_walk_kernel(
    scores_ptr, candidate_ptr, sample_pos_ptr, sample_slot_ptr, widths_ptr, sample_start_ptr,
    num_reqs_ptr, temperature_ptr, seeds_ptr, tokens_ptr, token_stride, realized_ptr,
    num_steps: tl.constexpr, top_k: tl.constexpr, BLOCK_K: tl.constexpr,
    SAMPLE_PROBABILISTIC: tl.constexpr, USE_FP64: tl.constexpr,
):  # fmt: skip
    """One program per request: walk its d_i packed rows (upstream walk, bounded)."""
    req = tl.program_id(0)
    if req >= tl.load(num_reqs_ptr):
        return
    width = tl.load(widths_ptr + req)
    start = tl.load(sample_start_ptr + req)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    previous = 0
    for step in range(num_steps):
        if step < width:
            row = start + step
            slot = tl.load(sample_slot_ptr + row)
            temperature = tl.load(temperature_ptr + slot)
            seed = tl.load(seeds_ptr + slot)
            scores = tl.load(
                scores_ptr + (row * top_k + previous) * top_k + offsets,
                mask=mask,
                other=float("-inf"),
            )
            scores = scores.to(tl.float64 if USE_FP64 else tl.float32)
            candidates = tl.load(
                candidate_ptr + row * top_k + offsets, mask=mask, other=0
            )
            # The draw is keyed by the position before the sampled token.
            _, index = gumbel_noised_argmax(
                scores, candidates, mask, seed, tl.load(sample_pos_ptr + row) - 1,
                temperature if SAMPLE_PROBABILISTIC else 0.0, IS_DRAFTING=True, USE_FP64=USE_FP64,
            )  # fmt: skip
            tl.store(realized_ptr + row * top_k + offsets, scores, mask=mask)
            tl.store(
                tokens_ptr + req * token_stride + step,
                tl.load(candidate_ptr + row * top_k + index),
            )
            previous = index


def packed_walk(
    scores: torch.Tensor,  # [S, k, k]
    candidate_ids: torch.Tensor,  # [S, k]
    sample_pos: torch.Tensor,
    sample_slot: torch.Tensor,
    widths: torch.Tensor,
    sample_start: torch.Tensor,
    num_reqs: torch.Tensor,  # 0-d device int: the walk grid is static for graphs
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    tokens: torch.Tensor,  # [max_reqs, K] output (row stride may exceed K)
    realized: torch.Tensor,  # [S, k] realized scores (for the draft-logit cache)
    max_reqs: int,
    probabilistic: bool,
    use_fp64: bool,
) -> None:
    top_k = candidate_ids.shape[-1]
    _packed_walk_kernel[(max_reqs,)](
        scores.contiguous(), candidate_ids.contiguous(), sample_pos, sample_slot, widths, sample_start,
        num_reqs, temperature, seeds, tokens, tokens.stride(0), realized,
        num_steps=tokens.shape[1], top_k=top_k, BLOCK_K=triton.next_power_of_2(top_k),
        SAMPLE_PROBABILISTIC=probabilistic, USE_FP64=use_fp64, num_warps=1,
    )  # fmt: skip


@triton.jit
def _packed_cache_kernel(
    draft_logits_ptr, cached_ptr, candidate_ptr, scores_ptr, sample_slot_ptr, sample_step_ptr,
    stride0, stride1, num_steps: tl.constexpr, top_k: tl.constexpr, BLOCK_K: tl.constexpr,
):  # fmt: skip
    """Upstream `_cache_draft_logits_kernel`, with (slot, step) read per sample."""
    sample = tl.program_id(0)
    slot = tl.load(sample_slot_ptr + sample)
    step = tl.load(sample_step_ptr + sample)
    offsets = tl.arange(0, BLOCK_K)
    mask = (slot >= 0) & (offsets < top_k)
    cache = (slot * num_steps + step) * top_k
    base = draft_logits_ptr + slot * stride0 + step * stride1
    tl.store(
        base + tl.load(cached_ptr + cache + offsets, mask=mask),
        float("-inf"),
        mask=mask,
    )
    ids = tl.load(candidate_ptr + sample * top_k + offsets, mask=mask)
    tl.store(
        base + ids, tl.load(scores_ptr + sample * top_k + offsets, mask=mask), mask=mask
    )
    tl.store(cached_ptr + cache + offsets, ids, mask=mask)


def packed_cache_draft_logits(
    draft_logits: torch.Tensor,  # [max_reqs, K, vocab]
    cached_candidates: torch.Tensor,  # [max_reqs, K, top_k]
    candidate_ids: torch.Tensor,  # [S, top_k]
    realized: torch.Tensor,  # [>=S, top_k]
    sample_slot: torch.Tensor,
    sample_step: torch.Tensor,  # [S]
) -> None:
    """Keep the proposal distribution the rejection sampler checks (T > 0)."""
    top_k = candidate_ids.shape[-1]
    _packed_cache_kernel[(candidate_ids.shape[0],)](
        draft_logits, cached_candidates, candidate_ids.contiguous(), realized, sample_slot, sample_step,
        draft_logits.stride(0), draft_logits.stride(1), num_steps=cached_candidates.shape[1],
        top_k=top_k, BLOCK_K=triton.next_power_of_2(top_k), num_warps=1,
    )  # fmt: skip
