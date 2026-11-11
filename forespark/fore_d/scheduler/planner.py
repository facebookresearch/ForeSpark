# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The host half of the scheduler: choose the step totals (D, B).

Each request's row holds the increments of its expected accepted drafts by
draft width (the upper concave envelope of the Fore-D head's per-width
predictions, so the row is nonincreasing), and one descending sort of all rows
orders every draft position in the batch: the first D entries are a legal set
of per-request widths, and the first B of those are the verification witness.
With R0(B) = sum of the top-B entries (expected accepted drafts) the planner
maximises predicted output rate over the captured bucket grid:

    Theta(D, B) = (n + R0(B)) / T(n, N, D, B),   B <= D <= sum(K_i)

The rows are the previous step's (one step stale, which is what lets the host
decide without waiting on the GPU); the GPU then spends D and B on the
*current* rows (kernels/select.py). Totals are snapped to captured buckets and
filled, since rows inside a bucket cost nothing extra.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .cost import CostModel


@dataclass(frozen=True)
class Plan:
    draft_total: int  # D: draft positions this step (sum of per-request widths)
    verify_quota: int  # B: of those, positions the target verifies next step
    rate: float  # predicted output tokens per ms
    step_ms: float = 0.0  # predicted step time


class JointPlanner:
    def __init__(self, cost: CostModel) -> None:
        self.cost = cost
        self.verify_buckets = cost.verify_tokens.astype(int)  # target graph sizes
        # Everything but the reward and the context reads (a constant per step)
        # depends only on (n, N, legal total), which repeat step after step in
        # steady state: the host cost is one sort.
        self._grids: dict[tuple[int, int, int], tuple] = {}

    def reward_curve(self, rows: np.ndarray, legal: np.ndarray) -> np.ndarray:
        """R0[B] for B = 0..sum(legal): cumulative sum of the sorted row entries."""
        valid = np.arange(rows.shape[1])[None, :] < legal[:, None]
        scores = np.sort(rows[valid])[::-1]
        return np.concatenate(([0.0], np.cumsum(scores)))

    def candidates(
        self, n: int, non_draft: int, legal_total: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """(D, B) pairs worth evaluating: bucket boundaries only.

        D: each captured sample bucket, clipped to the legal total (the GPU
        always fills exactly D). B: for each verify bucket, the largest quota
        that fits it, clipped to D. Ordered by (D, B) ascending, so argmax ties
        resolve to the smallest totals.
        """
        draft = np.unique(np.minimum(self.cost.sample_buckets.astype(int), legal_total))
        quota = np.unique(np.clip(self.verify_buckets - non_draft, 0, legal_total))
        d, b = np.meshgrid(draft, quota, indexing="ij")
        # Clipping B to D repeats (D, D) along a row; duplicates are harmless
        # for the argmax and cheaper than deduplicating.
        return d.ravel(), np.minimum(b, d).ravel()

    def _grid(self, n: int, non_draft: int, legal_total: int) -> tuple:
        key = (n, non_draft, legal_total)
        if key not in self._grids:
            if len(self._grids) > 4096:
                self._grids.clear()
            draft, quota = self.candidates(n, non_draft, legal_total)
            self._grids[key] = (
                draft,
                quota,
                self.cost.base_ms(n, non_draft, draft, quota),
            )
        return self._grids[key]

    def plan(
        self,
        reward: np.ndarray,
        n: int,
        non_draft: int,
        context_tokens: float | None = None,
    ) -> Plan:
        """reward: reward_curve of the stale rows; n requests; non_draft: N
        target rows; context_tokens: the batch's total context length."""
        legal_total = len(reward) - 1
        if legal_total == 0:
            return Plan(0, 0, 0.0)
        draft, quota, base = self._grid(n, non_draft, legal_total)
        step = base + self.cost.context_ms(n, context_tokens)
        rate = (reward[quota] + n) / step
        best = int(np.argmax(rate))
        return Plan(
            int(draft[best]), int(quota[best]), float(rate[best]), float(step[best])
        )

    def replan_quota(self, draft_total: int, reward: np.ndarray, n: int, non_draft: int,
                     context_tokens: float | None = None) -> int:  # fmt: skip
        """B for an already-drafted D with the actual non-draft rows N; `reward`
        is the reward_curve the plan was made from."""
        key = (n, non_draft, -draft_total - 1)  # disjoint from plan()'s keys
        if key not in self._grids:
            quota = np.unique(np.clip(self.verify_buckets - non_draft, 0, draft_total))
            draft = np.full_like(quota, draft_total)
            self._grids[key] = (quota, self.cost.base_ms(n, non_draft, draft, quota))
        # base: the step without its context reads
        quota, base = self._grids[key]
        step = base + self.cost.context_ms(n, context_tokens)
        return int(
            quota[np.argmax((reward[np.minimum(quota, len(reward) - 1)] + n) / step)]
        )


def safe_verify_quota(
    plan: Plan, legal_by_slot: np.ndarray, drafted_slots: np.ndarray, scheduled_by_slot: np.ndarray,
    reset_slots: Iterable[int] = (),
) -> int:  # fmt: skip
    """B for verifying the last proposal, never above what surely survives.

    A request drafted up to its legal width; if it has left (its slot now
    schedules nothing, or holds a new request) or the scheduler scheduled fewer
    drafts than that, the difference may be missing. A slot a new request took
    after the proposal (`reset_slots`) holds none of its drafts even when the
    scheduler gave the newcomer placeholders: a prompt whose every token but the
    last hits the prefix cache is scheduled as a decode row with K placeholders
    in its first step. Bounding by it keeps B <= the drafted positions actually
    present, so the device fills exactly B (equal CPU/device totals).
    """
    if reset_slots := list(reset_slots):
        scheduled_by_slot = scheduled_by_slot.copy()
        scheduled_by_slot[reset_slots] = 0
    missing = np.maximum(
        legal_by_slot[drafted_slots] - scheduled_by_slot[drafted_slots], 0
    ).sum()
    return max(0, min(plan.verify_quota, plan.draft_total - int(missing)))
