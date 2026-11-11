# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exact top-k indices over a vocabulary of logits (torch.topk: 55 ms for 46k x 155k
bf16 rows on a B200; this: 10 ms), for the narrow re-drafts of the width-conditioned
Fore-D head.

One program per row. The k-th largest of the row's tile maxima is a lower bound on
its k-th largest value (the maxima are distinct entries), so one more pass collects
the few entries at or above it and the k largest are extracted from those; ties go
to the lower index. A row with more than CAP such entries takes k passes over itself.
"""

import torch
import triton
import triton.language as tl

_CAP = 256
_BLOCK = 1024
_INT64_MIN = -9223372036854775807 - 1


@triton.jit
def _keys(values, index):
    # float order as signed int32, then the lower index first among equal values
    bits = values.to(tl.float32).to(tl.int32, bitcast=True)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return (ordered.to(tl.int64) << 32) | (4294967295 - index.to(tl.int64))


@triton.jit
def _topk_rows(x_ptr, stride, vocab, out_ptr, scratch_ptr, maxima_ptr, K: tl.constexpr, CAP: tl.constexpr,
               BLOCK: tl.constexpr, TILES: tl.constexpr, MINIMUM: tl.constexpr):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride
    lanes = tl.arange(0, BLOCK)
    maxima = maxima_ptr + row * TILES
    for start in range(0, vocab, BLOCK):
        offsets = start + lanes
        tile = tl.load(x_row + offsets, mask=offsets < vocab, other=float("-inf")).to(
            tl.float32
        )
        tl.store(maxima + start // BLOCK, tl.max(tile, axis=0))
    tl.debug_barrier()  # other lanes stored the tile maxima read below
    tiles = tl.arange(0, TILES)
    values = tl.load(
        maxima + tiles, mask=tiles < tl.cdiv(vocab, BLOCK), other=float("-inf")
    )
    tau = float("inf")
    for _ in range(K):  # the k-th largest tile maximum
        top = tl.argmax(values, axis=0)
        tau = tl.max(tl.where(tiles == top, values, float("-inf")), axis=0)
        values = tl.where(tiles == top, float("-inf"), values)
    scratch = scratch_ptr + row * CAP
    total = 0
    for start in range(0, vocab, BLOCK):
        offsets = start + lanes
        tile = tl.load(x_row + offsets, mask=offsets < vocab, other=float("-inf")).to(
            tl.float32
        )
        hit = (tile >= tau) & (offsets < vocab)
        hits = tl.sum(hit.to(tl.int32), axis=0)
        if hits > 0:  # most tiles hold none of the few entries >= tau
            slot = total + tl.cumsum(hit.to(tl.int32), axis=0) - 1
            tl.store(scratch + slot, _keys(tile, offsets), mask=hit & (slot < CAP))
        total += hits
    tl.debug_barrier()  # other lanes wrote the slots read below
    rank = tl.arange(0, K)
    chosen = tl.full([K], MINIMUM, tl.int64)
    if total <= CAP:
        cap = tl.arange(0, CAP)
        keys = tl.load(scratch + cap, mask=cap < total, other=MINIMUM)
        for i in range(K):
            top = tl.max(keys, axis=0)
            chosen = tl.where(rank == i, top, chosen)
            keys = tl.where(keys == top, MINIMUM, keys)
    else:
        bound = 9223372036854775807
        for i in range(K):
            top = MINIMUM
            for start in range(0, vocab, BLOCK):
                offsets = start + lanes
                tile = tl.load(
                    x_row + offsets, mask=offsets < vocab, other=float("-inf")
                )
                keys = _keys(tile, offsets)
                keys = tl.where((offsets < vocab) & (keys < bound), keys, MINIMUM)
                top = tl.maximum(top, tl.max(keys, axis=0))
            chosen = tl.where(rank == i, top, chosen)
            bound = top
    tl.store(out_ptr + row * K + rank, 4294967295 - (chosen & 4294967295))


@torch.library.custom_op("forespark::topk_indices", mutates_args=())
def topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k largest entries along the last dimension, largest first."""
    vocab = x.shape[-1]
    if not x.is_cuda or k & (k - 1) or vocab < k * _BLOCK:
        return x.topk(k, dim=-1).indices
    rows = x.reshape(-1, vocab)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    tiles = triton.next_power_of_2(triton.cdiv(vocab, _BLOCK))
    out = torch.empty((rows.shape[0], k), dtype=torch.long, device=x.device)
    scratch = torch.empty((rows.shape[0], _CAP), dtype=torch.long, device=x.device)
    maxima = torch.empty((rows.shape[0], tiles), dtype=torch.float32, device=x.device)
    _topk_rows[(rows.shape[0],)](
        rows, rows.stride(0), vocab, out, scratch, maxima, K=k, CAP=_CAP, BLOCK=_BLOCK, TILES=tiles,
        MINIMUM=_INT64_MIN,
    )  # fmt: skip
    return out.view(*x.shape[:-1], k)


@topk_indices.register_fake
def _(x: torch.Tensor, k: int) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], k), dtype=torch.long)
