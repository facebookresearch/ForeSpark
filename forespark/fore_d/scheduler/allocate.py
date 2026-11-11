# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The device half of the scheduler, in torch: the reference kernels/select.py must match.

Both allocations take a global top-k over per-request rows: draft widths over
the Fore-D head's width gains (hull_gains of width_expected), verify lengths
over realized cumulative confidence. Rows are nonincreasing and ties break by
flat index (earlier request, earlier position), so the winners are always a
prefix of every row -- a legal draft width / verification length.
"""

import torch


def width_expected(
    logits: torch.Tensor,
    scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected accepted drafts per draft width, [rows, K], from a non-causal
    drafter's triangular Fore-D logits [rows, K(K+1)/2] (width d's row at
    offset d(d-1)/2), each an affine-calibrated conditional acceptance logit."""
    logits = logits.float()
    if scale is not None:
        logits = logits * scale + bias
    width = int(round(((8 * logits.shape[1] + 1) ** 0.5 - 1) / 2))
    rows = [
        torch.cumprod(
            torch.sigmoid(logits[:, d * (d - 1) // 2 : d * (d + 1) // 2]), -1
        ).sum(-1)
        for d in range(1, width + 1)
    ]
    return torch.stack(rows, dim=1)


def hull_gains(expected: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Increments of the upper concave envelope of (0, 0), (1, F_1), ...,
    (limit, F_limit) per row, [rows, K]: nonincreasing up to the limit, so a
    top-T over them is a prefix of every row. Past the limit: undefined."""
    rows, width = expected.shape
    f = torch.cat(
        [expected.new_zeros(rows, 1), expected.float()], dim=1
    )  # f[:, d] = F at width d
    hull = f.clone()
    d = torch.arange(width + 1, dtype=torch.float32)
    for a in range(width + 1):
        for b in range(a + 2, width + 1):
            chord = f[:, a : a + 1] + (f[:, b : b + 1] - f[:, a : a + 1]) * (d - a) / (
                b - a
            )
            inside = (d > a) & (d < b) & (b <= limit[:, None])
            hull = torch.where(inside, torch.maximum(hull, chord), hull)
    return torch.cummin(hull[:, 1:] - hull[:, :-1], dim=1).values


def prefix_counts(
    scores: torch.Tensor, limit: torch.Tensor, total: torch.Tensor
) -> torch.Tensor:
    """Per-row count of the `total` best entries, only positions < `limit`.

    scores [R, K] nonincreasing per row; limit [R] int; total 0-d int tensor
    (clipped to the eligible count). Returns [R] int32.
    """
    rows, width = scores.shape
    position = torch.arange(width, device=scores.device)
    valid = position[None, :] < limit[:, None]
    masked = torch.where(valid, scores, -float("inf")).flatten()
    order = torch.argsort(masked, descending=True, stable=True)
    rank = torch.empty_like(order).scatter_(
        0, order, torch.arange(rows * width, device=scores.device)
    )
    return ((rank.view(rows, width) < total) & valid).sum(dim=1, dtype=torch.int32)


def allocate_widths(
    survival_rows: torch.Tensor, legal: torch.Tensor, draft_total: torch.Tensor
) -> torch.Tensor:
    """Draft width per request: the top `draft_total` positions by predicted survival."""
    return prefix_counts(survival_rows, legal, draft_total)


def allocate_verify(
    confidence: torch.Tensor, capacity: torch.Tensor, quota: torch.Tensor
) -> torch.Tensor:
    """Verified prefix per request: the top `quota` drafted positions by realized
    cumulative confidence (DSpark's adaptive-verification rule)."""
    return prefix_counts(torch.cumprod(confidence.float(), dim=-1), capacity, quota)
