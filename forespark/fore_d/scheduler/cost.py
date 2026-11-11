# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Step cost for the joint (D, B) planner.

    T(n, N, D, B) = fixed(n) + backbone(bucket_P(n + D)) + tail(bucket_S(D)) + verify(n, N + B)

MRV2 runs target -> sampling -> draft back to back on one stream, so a step is
the sum of these parts, each measured at startup:
  verify    target forward of n requests carrying N + B tokens, as
            bare(N + B) + [context(n) - bare(n)] * L / (n * L0): the graph for
            that many tokens with no context, plus n requests' context reads.
            Both curves are upstream's profile shape (one token per request),
            with and without context L0 each; a draft position costs a query
            row, not a context read (upstream's single curve prices every token
            as a new request). The reads scale linearly to the batch's actual
            total context L: attention streams the whole KV
  backbone  packed draft backbone graph replay, per row bucket P
  tail      draft tail graph replay (LM head, selector, walk, confidence), per S
  fixed     the rest of a draft step outside the graphs (the Fore-D head, allocation,
            prep, context K/V, metadata), per request count
The planner evaluates bucket boundaries only. A measured table (calibrate_costs.py)
replaces the profile's verify part with target forwards timed on served steps.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def bucket_index(buckets: np.ndarray, value: np.ndarray | int) -> np.ndarray:
    """Index of the smallest captured bucket >= value (clipped to the largest)."""
    return np.minimum(np.searchsorted(buckets, value, side="left"), len(buckets) - 1)


def isotonic(values: np.ndarray) -> np.ndarray:
    """Least-squares nondecreasing fit (pool adjacent violators)."""
    blocks: list[list[float]] = []  # [mean, weight]
    for v in np.asarray(values, dtype=np.float64):
        blocks.append([v, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            (m1, w1), (m2, w2) = blocks.pop(-2), blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    return np.concatenate([np.full(int(w), m) for m, w in blocks])


def extend_sizes(sizes, limit: int, step: int = 64) -> list[int]:
    """`sizes` up to `limit`, then every `step` up to `limit` itself. vLLM
    captures up to 512 tokens by default since v0.30; the scheduler's shapes (and
    the planner's verify totals) reach max_num_seqs * (K+1)."""
    kept = {int(s) for s in sizes if 0 < s <= limit}
    grid = range(max(kept, default=0) + step, limit, step)
    return sorted(kept | set(grid) | {int(limit)})


def _padded(
    x: np.ndarray, xs: np.ndarray, ys: np.ndarray, limit: float = 0.0
) -> np.ndarray:
    """Cost of the smallest profiled size >= x (graphs pad up); linear past the last.

    Past the largest graph (`limit`) nothing pads: interpolate between the
    profiled sizes above it, never across it (upstream's
    build_cost_tables_from_curves, so the scheduler and adaptive verification
    price these steps alike)."""
    x = np.asarray(x, dtype=np.float64)
    y = ys[np.minimum(np.searchsorted(xs, x, side="left"), len(xs) - 1)]
    above = xs > limit
    if limit > 0 and above.any():
        y = np.where(x > limit, np.interp(x, xs[above], ys[above]), y)
    if len(xs) > 1:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        y = np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * slope, y)
    return y


def _extrapolate(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Piecewise-linear through (xs, ys), continued linearly past both ends."""
    y = np.interp(x, xs, ys)
    if len(xs) > 1:
        lo = (ys[1] - ys[0]) / (xs[1] - xs[0])
        hi = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        y = np.where(x < xs[0], ys[0] + (x - xs[0]) * lo, y)
        y = np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * hi, y)
    return np.maximum(y, 0.0)


@dataclass
class MeasuredCosts:
    """Target forwards timed on the served model (calibrate_costs.py), per n.

    The startup profile prices verified drafts as one-token requests, which
    carries per-request work an extra draft does not. Per level: target forward
    by drafts verified per request; linear within a level, linear in log n
    across levels, clamped to the outer levels."""

    requests: np.ndarray  # ascending request counts
    verify_per_request: list[np.ndarray]  # drafts verified per request, ascending
    verify_ms: list[np.ndarray]  # target forward ms at those

    def verify(self, n: int, verified: np.ndarray) -> np.ndarray:
        b = np.asarray(verified, dtype=np.float64) / n
        curves = [
            _extrapolate(b, x, y)
            for x, y in zip(self.verify_per_request, self.verify_ms, strict=True)
        ]
        if len(self.requests) == 1:
            return curves[0]
        levels = np.log(self.requests)
        x = np.clip(np.log(max(n, 1)), levels[0], levels[-1])
        j = min(int(np.searchsorted(levels, x, side="right")) - 1, len(levels) - 2)
        t = (x - levels[j]) / (levels[j + 1] - levels[j])
        return (1 - t) * curves[j] + t * curves[j + 1]

    def to_dict(self) -> dict:
        return {
            "requests": np.asarray(self.requests).tolist(),
            "verify_per_request": [
                np.asarray(v).tolist() for v in self.verify_per_request
            ],
            "verify_ms": [np.asarray(v).tolist() for v in self.verify_ms],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MeasuredCosts":
        if payload.get("draft_ms") or len(payload.get("rest_ms") or []):
            raise ValueError(
                "a table with draft-step costs: ForeSpark's planner reads verify-only tables"
            )
        return cls(
            np.asarray(payload["requests"], dtype=np.float64),
            [np.asarray(v, dtype=np.float64) for v in payload["verify_per_request"]],
            [np.asarray(v, dtype=np.float64) for v in payload["verify_ms"]],
        )

    @classmethod
    def load(cls, path: Path) -> "MeasuredCosts":
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class CostModel:
    row_buckets: np.ndarray  # captured backbone row counts P, ascending
    backbone_ms: np.ndarray  # per row bucket
    sample_buckets: (
        np.ndarray
    )  # captured tail sample-row counts S, ascending (starts at 0)
    tail_ms: np.ndarray  # per sample bucket
    verify_tokens: np.ndarray  # profiled target sizes, ascending
    verify_bare_ms: np.ndarray  # target forward at each size, no context
    verify_context_ms: (
        np.ndarray
    )  # the same, every token a request at the profile context
    fixed_requests: np.ndarray  # request counts, ascending
    fixed_ms: np.ndarray  # per fixed_requests entry
    profile_context: float = (
        0.0  # L0, context tokens per profiled request (0: unknown, no scaling)
    )
    graph_limit: float = 0.0  # the target's largest graph; 0: pad-priced throughout
    # Timed decode steps replace the profile's verify part; their forwards
    # already read a typical context, so the context term drops.
    measured: MeasuredCosts | None = None

    def draft_ms(self, n: int, draft_totals: np.ndarray) -> np.ndarray:
        rows = bucket_index(self.row_buckets, n + draft_totals)
        samples = bucket_index(self.sample_buckets, draft_totals)
        return self.backbone_ms[rows] + self.tail_ms[samples]

    def _verify(self, tokens, curve: np.ndarray) -> np.ndarray:
        return _padded(tokens, self.verify_tokens, curve, self.graph_limit)

    def context_ms(self, n: int, context_tokens: float | None = None) -> float:
        """The target's context reads for n requests holding `context_tokens` in
        total (None: at the profiled L0 each)."""
        if self.measured is not None:
            return 0.0
        context = self._verify(n, self.verify_context_ms)
        ms = float(context - self._verify(n, self.verify_bare_ms))
        if context_tokens is not None and self.profile_context > 0:
            ms *= context_tokens / (n * self.profile_context)
        return max(ms, 0.0)

    def verify_cost(
        self, n: int, tokens: np.ndarray, context_tokens: float | None = None
    ) -> np.ndarray:
        """Target forward of n requests carrying `tokens` in total."""
        bare = self._verify(tokens, self.verify_bare_ms)
        return bare + self.context_ms(n, context_tokens)

    def base_ms(
        self, n: int, non_draft: int, draft_totals: np.ndarray, quotas: np.ndarray
    ) -> np.ndarray:
        """A step without the target's context reads, the one part that
        depends on context length and not on (D, B)."""
        fixed = float(np.interp(n, self.fixed_requests, self.fixed_ms))
        draft = fixed + self.draft_ms(n, draft_totals)
        if self.measured is None:
            return draft + self._verify(non_draft + quotas, self.verify_bare_ms)
        step = draft + self.measured.verify(n, quotas)
        if non_draft > n:  # prefill rows: the profile's price above a decode step
            bare = self.verify_bare_ms
            extra = self._verify(non_draft + quotas, bare)
            step = step + np.maximum(extra - self._verify(n + quotas, bare), 0.0)
        return step

    def step_ms(self, n: int, non_draft: int, draft_totals: np.ndarray, quotas: np.ndarray,
                context_tokens: float | None = None) -> np.ndarray:  # fmt: skip
        return self.base_ms(n, non_draft, draft_totals, quotas) + self.context_ms(
            n, context_tokens
        )

    def to_dict(self) -> dict:
        items = [(k, v) for k, v in self.__dict__.items() if k != "measured"]
        out = {k: np.asarray(v).tolist() for k, v in items}
        if self.measured is not None:
            out["measured"] = self.measured.to_dict()
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> "CostModel":
        payload = dict(payload)
        measured = payload.pop("measured", None)
        fields = {
            k: float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=np.float64)
            for k, v in payload.items()
        }
        measured = MeasuredCosts.from_dict(measured) if measured else None
        return cls(**fields, measured=measured)
