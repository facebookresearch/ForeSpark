# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Portions adapted from vLLM v0.30.0's DFlash input preparation
# (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py).

"""Packed draft inputs: request i gets 1 + d_i contiguous query rows.

Upstream `_prepare_dflash_inputs_kernel` lays out exactly 1 + K rows per
request, in one program per request that walks its context rows and then its
query rows. Here the two halves are separate kernels, so each has a grid that
does not depend on the step's longest prefill chunk:

  queries  one program per request, 1 + d_i rows at per-request prefix offsets
               rows      [a_0 m m ... | a_1 m ... | ... | padding]   a = anchor (bonus token),
               start_i   = sum_{r<i} (1 + d_r)                       m = mask token
               samples   the d_i mask rows of each request, packed request-major
           plus `row_position` (0 for an anchor, j for the j-th mask) so the
           drafter's two-tap convolution never reads across requests.
  context  one lane per target token: its position and KV slot, pad slots
           past the step's T tokens, so a graph captured for `size` rows
           writes nothing stray.

Both read the request count n (and D, T) from device memory: the query half
lives in the pre-draft CUDA graph (keyed by requests), the context half in the
context graph (keyed by target tokens).
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.cp_utils import cp_local_slot


@triton.jit
def _query_prepare_kernel(
    # outputs
    out_input_ids_ptr, out_positions_ptr, out_query_start_loc_ptr, out_seq_lens_ptr, out_query_slots_ptr,
    out_row_position_ptr, out_sample_rows_ptr, out_sample_pos_ptr, out_sample_slot_ptr,
    out_sample_step_ptr, out_sample_flat_ptr, out_sample_pred_ptr, out_sample_anchor_ptr, out_tokens_ptr,
    out_temperature_ptr, out_seeds_ptr,
    # per-request layout (device, from the width allocation)
    widths_ptr, row_start_ptr, sample_start_ptr,
    # target batch
    target_positions_ptr, target_query_start_loc_ptr, idx_mapping_ptr,
    last_sampled_ptr, next_prefill_tokens_ptr, num_sampled_ptr, num_rejected_ptr,
    temperature_ptr, seeds_ptr, block_table_ptr, block_table_stride, n_ptr,
    # scalars
    mask_token_id, block_size, max_model_len, cp_rank,
    K: tl.constexpr, PAD: tl.constexpr, CP_SIZE: tl.constexpr, CP_INTERLEAVE: tl.constexpr, QBLOCK: tl.constexpr,
):  # fmt: skip
    req = tl.program_id(0)
    if req >= tl.load(n_ptr):
        return
    slot = tl.load(idx_mapping_ptr + req)
    width = tl.load(widths_ptr + req)
    row_start = tl.load(row_start_ptr + req)
    sample_start = tl.load(sample_start_ptr + req)

    ctx_start = tl.load(target_query_start_loc_ptr + req)
    num_valid_ctx = (
        tl.load(target_query_start_loc_ptr + req + 1)
        - ctx_start
        - tl.load(num_rejected_ptr + req)
    )
    if tl.load(num_sampled_ptr + req) > 0:
        bonus = tl.load(last_sampled_ptr + slot).to(tl.int32)
    else:  # chunked prefill: the next prefill token is the anchor
        bonus = tl.load(next_prefill_tokens_ptr + slot).to(tl.int32)
    last_pos = tl.load(target_positions_ptr + ctx_start + num_valid_ctx - 1)

    # Query rows: the anchor then `width` mask rows, at packed offset row_start.
    q = tl.arange(0, QBLOCK)
    is_query = q <= width
    q = tl.where(is_query, q, 0)
    pos = last_pos + 1 + q
    q_block = tl.load(
        block_table_ptr
        + req * block_table_stride
        + tl.minimum(pos // (block_size * CP_SIZE), block_table_stride - 1),
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_slot = cp_local_slot(
        pos, q_block, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD
    )
    q_slot = tl.where(
        is_query & (q_block != 0), q_slot, PAD
    )  # block 0 is the null block
    row = row_start + q
    tl.store(
        out_input_ids_ptr + row, tl.where(q == 0, bonus, mask_token_id), mask=is_query
    )
    tl.store(out_positions_ptr + row, tl.minimum(pos, max_model_len - 1), mask=is_query)
    tl.store(out_query_slots_ptr + row, q_slot, mask=is_query)
    tl.store(out_row_position_ptr + row, q, mask=is_query)

    # Samples: mask row q predicts the token at its own position (DFlash).
    # `tokens` is [requests + 1, 1 + K]: the anchor, then the walk's choices, so
    # a sample's predecessor token sits right before its own column.
    is_sample = is_query & (q >= 1)
    sample = sample_start + q - 1
    step = q - 1
    tl.store(out_sample_rows_ptr + sample, row, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample, pos, mask=is_sample)
    tl.store(out_sample_slot_ptr + sample, slot, mask=is_sample)
    tl.store(out_sample_step_ptr + sample, step, mask=is_sample)
    tl.store(out_sample_flat_ptr + sample, req * K + step, mask=is_sample)
    tl.store(out_sample_pred_ptr + sample, req * (K + 1) + step, mask=is_sample)
    tl.store(out_sample_anchor_ptr + sample, bonus, mask=is_sample)

    tl.store(out_tokens_ptr + req * (K + 1), bonus)
    tl.store(out_query_start_loc_ptr + req, row_start)
    tl.store(out_seq_lens_ptr + req, tl.minimum(last_pos + 2 + width, max_model_len))
    tl.store(out_temperature_ptr + slot, tl.load(temperature_ptr + slot))
    tl.store(out_seeds_ptr + slot, tl.load(seeds_ptr + slot))


@triton.jit
def _context_prepare_kernel(
    out_context_positions_ptr, out_context_slots_ptr,
    target_positions_ptr, target_query_start_loc_ptr, num_rejected_ptr,
    block_table_ptr, block_table_stride, n_ptr, size, block_size, cp_rank,
    PAD: tl.constexpr, CP_SIZE: tl.constexpr, CP_INTERLEAVE: tl.constexpr, SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
):  # fmt: skip
    """Row t < T = qsl[n]: position and slot of target token t (rejected ones
    write nothing: pad slot); T <= t < size: pad."""
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.load(n_ptr)
    is_ctx = t < tl.load(target_query_start_loc_ptr + n)
    # Token t's request: the last r with qsl[r] <= t, by binary search.
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.zeros([BLOCK], dtype=tl.int32) + n - 1
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi + 1) // 2
        up = tl.load(target_query_start_loc_ptr + mid, mask=is_ctx, other=0) <= t
        lo = tl.where(up, mid, lo)
        hi = tl.where(up, hi, mid - 1)
    ctx_end = tl.load(target_query_start_loc_ptr + lo + 1, mask=is_ctx, other=0)
    is_valid = is_ctx & (
        t < ctx_end - tl.load(num_rejected_ptr + lo, mask=is_ctx, other=0)
    )
    pos = tl.load(target_positions_ptr + t, mask=is_valid, other=0)
    block = tl.load(
        block_table_ptr
        + lo * block_table_stride
        + tl.minimum(pos // (block_size * CP_SIZE), block_table_stride - 1),
        mask=is_valid,
        other=0,
    ).to(tl.int64)
    slot = cp_local_slot(pos, block, block_size, cp_rank, CP_SIZE, CP_INTERLEAVE, PAD)
    slot = tl.where(is_valid & (block != 0), slot, PAD)
    in_range = t < size
    tl.store(out_context_positions_ptr + t, pos, mask=in_range)
    tl.store(out_context_slots_ptr + t, slot, mask=in_range)


@triton.jit
def _pad_kernel(
    out_query_start_loc_ptr, out_seq_lens_ptr, out_query_slots_ptr, out_row_position_ptr,
    out_sample_rows_ptr, out_sample_slot_ptr, out_sample_step_ptr, out_sample_flat_ptr, out_sample_pred_ptr,
    out_sample_anchor_ptr, n_ptr, draft_total_ptr, max_num_reqs, max_rows, max_samples,
    K: tl.constexpr, PAD: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """Make padding inert for graph replay: no KV writes, no sample scatter."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    num_reqs = tl.load(n_ptr)
    total_samples = tl.load(draft_total_ptr)
    total_rows = num_reqs + total_samples
    req_pad = (i >= num_reqs) & (i <= max_num_reqs)
    tl.store(out_query_start_loc_ptr + i, total_rows, mask=req_pad)
    tl.store(out_seq_lens_ptr + i, 0, mask=req_pad & (i < max_num_reqs))
    row_pad = (i >= total_rows) & (i < max_rows)
    tl.store(out_query_slots_ptr + i, PAD, mask=row_pad)
    tl.store(out_row_position_ptr + i, 0, mask=row_pad)
    sample_pad = (i >= total_samples) & (i < max_samples)
    tl.store(out_sample_rows_ptr + i, 0, mask=sample_pad)
    tl.store(out_sample_slot_ptr + i, -1, mask=sample_pad)
    tl.store(out_sample_step_ptr + i, 0, mask=sample_pad)
    # Padding samples read and write the spare request row.
    tl.store(out_sample_flat_ptr + i, max_num_reqs * K, mask=sample_pad)
    tl.store(out_sample_pred_ptr + i, max_num_reqs * (K + 1), mask=sample_pad)
    tl.store(out_sample_anchor_ptr + i, 0, mask=sample_pad)


def prepare_queries(
    b,  # Fore-D speculator: its packed buffers (see runtime/speculator.py)
    n: torch.Tensor,  # 0-d int32 on device: requests
    draft_total: torch.Tensor,  # 0-d int32 on device: D
    grid_reqs: int,  # >= n
    target_positions: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    query_slots: torch.Tensor,
    block_table: torch.Tensor,
    block_tables,  # vLLM BlockTables (block size, CP geometry)
    group: int,
    mask_token_id: int,
    max_model_len: int,
) -> None:
    """Every packed query-side input for this step, padding made inert.

    `b.widths`, `b.row_start` (exclusive cumsum of widths + 1) and
    `b.sample_start` (of widths) are already on device (kernels/select.py).
    """
    _query_prepare_kernel[(grid_reqs,)](
        b.input_ids, b.positions, b.query_start_loc, b.seq_lens, query_slots,
        b.row_position, b.sample_rows, b.sample_pos, b.sample_slot, b.sample_step,
        b.sample_flat, b.sample_pred, b.sample_anchor, b.tokens, b.temperature, b.seeds,
        b.widths, b.row_start, b.sample_start,
        target_positions, target_query_start_loc, idx_mapping,
        last_sampled, next_prefill_tokens, num_sampled, num_rejected,
        temperature, seeds, block_table, block_table.stride(0), n,
        mask_token_id, block_tables.kernel_block_sizes[group], max_model_len, block_tables.cp_rank,
        K=b.max_width, PAD=PAD_SLOT_ID, CP_SIZE=block_tables.cp_size, CP_INTERLEAVE=block_tables.cp_interleave,
        QBLOCK=triton.next_power_of_2(b.max_width + 1),
    )  # fmt: skip
    extent = max(b.max_rows, b.max_num_reqs + 1)
    _pad_kernel[(triton.cdiv(extent, 1024),)](
        b.query_start_loc, b.seq_lens, query_slots, b.row_position,
        b.sample_rows, b.sample_slot, b.sample_step, b.sample_flat, b.sample_pred, b.sample_anchor,
        n, draft_total, b.max_num_reqs, b.max_rows, b.max_samples,
        K=b.max_width, PAD=PAD_SLOT_ID, BLOCK=1024,
    )  # fmt: skip


def prepare_context(
    n: torch.Tensor,  # 0-d int32 on device: requests
    size: int,  # rows to write: >= the step's target tokens T (the rest get pad slots)
    target_positions: torch.Tensor,
    target_query_start_loc: torch.Tensor,
    num_rejected: torch.Tensor,
    context_positions: torch.Tensor,
    context_slots: torch.Tensor,
    block_table: torch.Tensor,
    block_tables,
    group: int,
    max_num_reqs: int,
) -> None:
    """Position and KV slot of every target token, for the context K/V write."""
    block = min(1024, triton.next_power_of_2(size))
    _context_prepare_kernel[(triton.cdiv(size, block),)](
        context_positions, context_slots, target_positions, target_query_start_loc, num_rejected,
        block_table, block_table.stride(0), n, size, block_tables.kernel_block_sizes[group], block_tables.cp_rank,
        PAD=PAD_SLOT_ID, CP_SIZE=block_tables.cp_size, CP_INTERLEAVE=block_tables.cp_interleave,
        SEARCH_STEPS=max(1, max_num_reqs.bit_length()), BLOCK=block,
    )  # fmt: skip
