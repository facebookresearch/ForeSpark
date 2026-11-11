# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The Fore-D head's inputs: which target row and which anchor token each request uses.

The head reads the target hidden state of the last accepted token (the token
that sampled the anchor) and the anchor itself: the bonus token, or the next
prefill token for a chunked prefill. One program for the batch.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fore_d_inputs_kernel(
    query_start_loc_ptr, num_rejected_ptr, num_sampled_ptr, idx_mapping_ptr,
    last_sampled_ptr, next_prefill_ptr, n_ptr, rows_ptr, anchors_ptr, BLOCK: tl.constexpr,
):  # fmt: skip
    r = tl.arange(0, BLOCK)
    live = r < tl.load(n_ptr)
    start = tl.load(query_start_loc_ptr + r, mask=live, other=0)
    end = tl.load(query_start_loc_ptr + r + 1, mask=live, other=0)
    row = tl.maximum(end - tl.load(num_rejected_ptr + r, mask=live, other=0) - 1, start)
    slot = tl.load(idx_mapping_ptr + r, mask=live, other=0)
    bonus = tl.load(last_sampled_ptr + slot, mask=live, other=0)
    prefill = tl.load(next_prefill_ptr + slot, mask=live, other=0)
    anchor = tl.where(
        tl.load(num_sampled_ptr + r, mask=live, other=0) > 0, bonus, prefill
    )
    tl.store(rows_ptr + r, row, mask=live)
    tl.store(anchors_ptr + r, anchor, mask=live)


def fore_d_inputs(
    query_start_loc: torch.Tensor,  # target batch [n + 1]
    num_rejected: torch.Tensor,
    num_sampled: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled: torch.Tensor,  # [max_reqs] per slot
    next_prefill_tokens: torch.Tensor,  # [(lookahead,) max_reqs] per slot; row 0 used
    n: torch.Tensor,  # 0-d int32 on device
    rows: torch.Tensor,  # [max_reqs] int64 out
    anchors: torch.Tensor,  # [max_reqs] int64 out
) -> None:
    _fore_d_inputs_kernel[(1,)](
        query_start_loc, num_rejected, num_sampled, idx_mapping, last_sampled, next_prefill_tokens,
        n, rows, anchors, BLOCK=triton.next_power_of_2(rows.shape[0]),
    )  # fmt: skip
