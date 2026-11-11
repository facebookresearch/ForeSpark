# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Top-T prefix selection: the device half of both Fore-D allocations.

Widths (pre-draft) and verify lengths (post-draft) are one rule: order every
eligible position of the batch by its score (descending, ties by flat index)
and keep the first T. Score rows are nonincreasing, so the kept set is a
prefix of every row and the answer is a per-request count.

The kernels all read n and T from device memory so they can sit inside a CUDA
graph:

  width gains (pre-draft) the Fore-D head's triangular logits -> F[i, d] =
             expected accepted drafts at width d -> the increments of F's
             upper concave envelope: nonincreasing (the drafter is non-causal,
             so its draft changes with its width), and a top-T over them is
             the best allocation of the relaxed problem
  survival   (post-draft) confidences p[i, j] -> s[i, j] = prod_{k<=j} p[i, k];
             limits = min of caps
  rank       rank of each eligible position among all eligible positions
             (all-pairs compare; exact stable-sort order, no atomics)
  finalize   counts[i] = #{j: rank < T}; exclusive offsets of counts + extras,
             padded past n with the total (what varlen metadata expects)

This replaces a torch chain of ~15 small kernels (argsort, scatter, cumsum,
reductions) per allocation.
"""

import torch
from vllm.triton_utils import tl, triton

RANK_BLOCK_I = 64
RANK_BLOCK_J = 256


@triton.jit
def _survival_kernel(
    src_ptr, src_stride, gather_ptr, n_ptr, cap_a_ptr, cap_b_ptr, gather_cap_b_ptr, out_ptr, limit_ptr,
    K: tl.constexpr, K_PAD: tl.constexpr, BLOCK_R: tl.constexpr, GATHER: tl.constexpr, CAP_B: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n = tl.load(n_ptr)
    live = r < n
    c = tl.arange(0, K_PAD)
    src_row = (
        tl.load(gather_ptr + r, mask=live, other=0).to(tl.int64)
        if GATHER
        else r.to(tl.int64)
    )
    mask = live[:, None] & (c[None, :] < K)
    p = tl.load(
        src_ptr + src_row[:, None] * src_stride + c[None, :], mask=mask, other=1.0
    ).to(tl.float32)
    s = tl.cumprod(p, axis=1)
    tl.store(out_ptr + r[:, None] * K_PAD + c[None, :], s, mask=r[:, None] >= 0)
    limit = tl.load(cap_a_ptr + r, mask=live, other=0)
    if CAP_B:
        slot = tl.load(gather_cap_b_ptr + r, mask=live, other=0)
        limit = tl.minimum(limit, tl.load(cap_b_ptr + slot, mask=live, other=0))
    tl.store(limit_ptr + r, tl.where(live, limit, 0))


@triton.jit
def _width_gains_kernel(
    src_ptr, src_stride, scale_ptr, bias_ptr, n_ptr, cap_ptr,
    out_ptr, limit_ptr, rows_out_ptr, rows_index_ptr, rows_dummy,
    K: tl.constexpr, K_PAD: tl.constexpr, BLOCK_R: tl.constexpr, WRITE_ROWS: tl.constexpr,
):  # fmt: skip
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    n = tl.load(n_ptr)
    live = r < n
    c = tl.arange(0, K_PAD)
    limit = tl.load(cap_ptr + r, mask=live, other=0)
    row = src_ptr + r.to(tl.int64) * src_stride
    # Column d - 1: F at width d, from width d's row (at offset d(d-1)/2).
    expected = tl.zeros([BLOCK_R, K_PAD], dtype=tl.float32)
    for d in tl.static_range(1, K + 1):
        survive = tl.full([BLOCK_R], 1.0, dtype=tl.float32)
        total = tl.zeros([BLOCK_R], dtype=tl.float32)
        for j in tl.static_range(d):
            k = d * (d - 1) // 2 + j
            logit = tl.load(row + k, mask=live, other=0.0).to(tl.float32)
            survive *= tl.sigmoid(
                logit * tl.load(scale_ptr + k) + tl.load(bias_ptr + k)
            )
            total += survive
        expected = tl.where(c[None, :] == d - 1, total[:, None], expected)
    # Upper concave envelope of (0, 0), (1, F_1), ..., (limit, F_limit): at each
    # width the best chord over widths a < d < b <= limit, or F_d itself.
    width = (c + 1).to(tl.float32)[None, :]
    hull = expected
    for a in tl.static_range(0, K - 1):
        f_a = tl.sum(tl.where(c[None, :] == a - 1, expected, 0.0), axis=1)  # F_0 = 0
        for b in tl.static_range(a + 2, K + 1):
            f_b = tl.sum(tl.where(c[None, :] == b - 1, expected, 0.0), axis=1)
            chord = f_a[:, None] + (f_b - f_a)[:, None] * (width - a) / (b - a)
            inside = (width > a) & (width < b) & (b <= limit)[:, None]
            hull = tl.where(inside, tl.maximum(hull, chord), hull)
    # Increments h(d) - h(d - 1), made exactly nonincreasing against rounding.
    gains = tl.zeros([BLOCK_R, K_PAD], dtype=tl.float32)
    previous = tl.zeros([BLOCK_R], dtype=tl.float32)
    smallest = tl.full([BLOCK_R], float("inf"), dtype=tl.float32)
    for d in tl.static_range(1, K + 1):
        current = tl.sum(tl.where(c[None, :] == d - 1, hull, 0.0), axis=1)
        smallest = tl.minimum(smallest, current - previous)
        gains = tl.where(c[None, :] == d - 1, smallest[:, None], gains)
        previous = current
    tl.store(out_ptr + r[:, None] * K_PAD + c[None, :], gains, mask=r[:, None] >= 0)
    tl.store(limit_ptr + r, tl.where(live, limit, 0))
    if WRITE_ROWS:
        # Per-slot copy for the host planner; padding rows go to a dummy row.
        slot = tl.where(
            live, tl.load(rows_index_ptr + r, mask=live, other=0), rows_dummy
        ).to(tl.int64)
        tl.store(
            rows_out_ptr + slot[:, None] * K + c[None, :], gains, mask=c[None, :] < K
        )


@triton.jit
def _rank_kernel(
    score_ptr, limit_ptr, n_ptr, total_ptr, selected_ptr,
    K_PAD: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):  # fmt: skip
    i = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
    size = tl.load(n_ptr) * K_PAD
    if tl.program_id(0) * BLOCK_I >= size:
        return
    row_i = i // K_PAD
    live_i = i < size
    valid_i = live_i & ((i % K_PAD) < tl.load(limit_ptr + row_i, mask=live_i, other=0))
    v_i = tl.load(score_ptr + i, mask=valid_i, other=-1.0)
    rank = tl.zeros([BLOCK_I], dtype=tl.int32)
    for start in range(0, size, BLOCK_J):
        j = start + tl.arange(0, BLOCK_J)
        live_j = j < size
        valid_j = live_j & (
            (j % K_PAD) < tl.load(limit_ptr + j // K_PAD, mask=live_j, other=0)
        )
        v_j = tl.load(score_ptr + j, mask=valid_j, other=-1.0)
        ahead = (v_j[None, :] > v_i[:, None]) | (
            (v_j[None, :] == v_i[:, None]) & (j[None, :] < i[:, None])
        )
        rank += tl.sum((ahead & valid_j[None, :]).to(tl.int32), axis=1)
    keep = valid_i & (rank < tl.load(total_ptr))
    tl.store(selected_ptr + i, keep.to(tl.int8), mask=live_i)


@triton.jit
def _finalize_kernel(
    selected_ptr, n_ptr, counts_ptr, slot_counts_ptr, slot_ptr,
    offsets_a_ptr, extra_a, offsets_b_ptr, extra_b_ptr, extra_b_scalar, max_rows,
    K_PAD: tl.constexpr, MAX_R: tl.constexpr,
    SLOT_COUNTS: tl.constexpr, OFFSETS_B: tl.constexpr, EXTRA_B_ARRAY: tl.constexpr,
):  # fmt: skip
    r = tl.arange(0, MAX_R)
    n = tl.load(n_ptr)
    live = r < n
    c = tl.arange(0, K_PAD)
    flags = tl.load(
        selected_ptr + r[:, None] * K_PAD + c[None, :], mask=live[:, None], other=0
    )
    counts = tl.sum(flags.to(tl.int32), axis=1)
    fits = r < max_rows
    tl.store(counts_ptr + r, counts, mask=fits)
    if SLOT_COUNTS:
        slot = tl.load(slot_ptr + r, mask=live, other=0)
        tl.store(slot_counts_ptr + slot, counts, mask=live)
    # offsets[0] = 0, offsets[i + 1] = offsets[i] + counts[i] + extra[i]; past n
    # every entry is the total, so padded requests are empty.
    step_a = tl.where(live, counts + extra_a, 0)
    tl.store(offsets_a_ptr + r + 1, tl.cumsum(step_a, axis=0), mask=fits)
    tl.store(offsets_a_ptr + r, 0, mask=r == 0)
    if OFFSETS_B:
        if EXTRA_B_ARRAY:
            extra_b = tl.load(extra_b_ptr + r, mask=live, other=0)
        else:
            extra_b = extra_b_scalar
        step_b = tl.where(live, counts + extra_b, 0)
        tl.store(offsets_b_ptr + r + 1, tl.cumsum(step_b, axis=0), mask=fits)
        tl.store(offsets_b_ptr + r, 0, mask=r == 0)


class PrefixSelect:
    """Scratch buffers plus the three launches, for rows up to `max_rows` of width K."""

    def __init__(self, max_rows: int, width: int, device: torch.device) -> None:
        self.max_rows, self.width = max_rows, width
        self.k_pad = triton.next_power_of_2(width)
        self.max_r = triton.next_power_of_2(max_rows)
        self.scores = torch.zeros(
            self.max_r * self.k_pad, dtype=torch.float32, device=device
        )
        self.limits = torch.zeros(self.max_r, dtype=torch.int32, device=device)
        self.selected = torch.zeros(
            self.max_r * self.k_pad, dtype=torch.int8, device=device
        )
        self._dummy = torch.zeros(1, dtype=torch.int32, device=device)

    def survival(
        self,
        src: torch.Tensor,  # [>=n, K] probabilities
        n: torch.Tensor,  # 0-d int32 on device
        cap: torch.Tensor,  # [>=n] int32: per-row limit
        gather: torch.Tensor | None = None,  # [>=n] rows of src to read (else 0..n-1)
        cap_b: torch.Tensor
        | None = None,  # per-slot second limit, indexed by cap_b_index
        cap_b_index: torch.Tensor | None = None,
    ) -> None:
        d = self._dummy
        block_r = min(16, self.max_r)
        _survival_kernel[(self.max_r // block_r,)](
            src, src.stride(0), gather if gather is not None else d, n,
            cap, cap_b if cap_b is not None else d, cap_b_index if cap_b_index is not None else d,
            self.scores, self.limits, K=self.width, K_PAD=self.k_pad, BLOCK_R=block_r,
            GATHER=gather is not None, CAP_B=cap_b is not None,
        )  # fmt: skip

    def width_gains(
        self,
        logits: torch.Tensor,  # [>=n, K(K+1)/2] triangular Fore-D logits, width d at offset d(d-1)/2
        n: torch.Tensor,
        cap: torch.Tensor,  # [>=n] int32: widest legal draft per row
        calibration: tuple[torch.Tensor, torch.Tensor],  # ([K(K+1)/2] scale, bias)
        rows_out: torch.Tensor
        | None = None,  # [num_slots + 1, K]: gains copied per slot
        rows_index: torch.Tensor | None = None,
    ) -> None:
        d = self._dummy
        block_r = min(16, self.max_r)
        _width_gains_kernel[(self.max_r // block_r,)](
            logits, logits.stride(0), *calibration, n, cap, self.scores, self.limits,
            rows_out if rows_out is not None else d, rows_index if rows_index is not None else d,
            rows_out.shape[0] - 1 if rows_out is not None else 0,
            K=self.width, K_PAD=self.k_pad, BLOCK_R=block_r, WRITE_ROWS=rows_out is not None,
        )  # fmt: skip

    def select(
        self,
        n: torch.Tensor,
        total: torch.Tensor,  # 0-d int32 on device: positions to keep
        counts: torch.Tensor,  # [max_rows] int32 out
        offsets_a: torch.Tensor,  # [max_rows + 1] int32 out: cumsum(counts + extra_a)
        extra_a: int,
        offsets_b: torch.Tensor | None = None,  # optional second offsets
        extra_b: torch.Tensor | int = 0,
        slot_counts: torch.Tensor
        | None = None,  # counts scattered to slot_counts[slots[i]]
        slots: torch.Tensor | None = None,
    ) -> None:
        d = self._dummy
        _rank_kernel[(triton.cdiv(self.max_r * self.k_pad, RANK_BLOCK_I),)](
            self.scores, self.limits, n, total, self.selected,
            K_PAD=self.k_pad, BLOCK_I=RANK_BLOCK_I, BLOCK_J=RANK_BLOCK_J,
        )  # fmt: skip
        extra_b_array = isinstance(extra_b, torch.Tensor)
        _finalize_kernel[(1,)](
            self.selected, n, counts, slot_counts if slot_counts is not None else d,
            slots if slots is not None else d, offsets_a, extra_a,
            offsets_b if offsets_b is not None else d, extra_b if extra_b_array else d,
            0 if extra_b_array else int(extra_b), self.max_rows,
            K_PAD=self.k_pad, MAX_R=self.max_r, SLOT_COUNTS=slot_counts is not None,
            OFFSETS_B=offsets_b is not None, EXTRA_B_ARRAY=extra_b_array,
            num_warps=8,
        )  # fmt: skip
