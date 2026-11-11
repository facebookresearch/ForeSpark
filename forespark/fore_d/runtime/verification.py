# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptive verification whose batch quota comes from Fore-D's planner.

Upstream chooses the verify quota B from stale confidences with a 1-D search
and spreads it over requests by current confidence. Here B was chosen before
drafting, jointly with D (planner.py); only the per-request spread is left,
and it is upstream's rule -- global top-B of realized cumulative confidence --
bounded by each request's actual draft width.
"""

from collections import defaultdict

import numpy as np
import torch
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)

from ...serving.cpu_ranking import HostRankingVerificationManager, prefix_widths
from .stale import PinnedStaging


class JointVerificationManager(AdaptiveVerificationManager):
    speculator = None  # set by ForeDSpeculator.bind_runner

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per-step host inputs, one H2D copy: [n, B, scheduled..., non-draft...].
        max_reqs = self._confidence_probs.shape[0]
        self._inputs = torch.zeros(
            2 + 2 * max_reqs, dtype=torch.int32, device=self._confidence_probs.device
        )
        self._staging = PinnedStaging(2 + 2 * max_reqs)

    def add_request(self, req_idx: int) -> None:
        super().add_request(req_idx)
        self.speculator.on_new_request(req_idx)

    def batches_to_profile(self, capture_sizes: list[int]):
        """Upstream's profile, then the same shapes with no context.

        vLLM pads every dummy batch to the graph's request count, one token
        each, so a (requests, tokens) grid cannot be profiled directly; the
        with/without-context pair separates token cost from context reads.
        """
        upstream = list(super().batches_to_profile(capture_sizes))
        self._num_upstream = len(upstream)
        self._profile_context = upstream[0]["context_len"] if upstream else 0
        yield from upstream
        for batch in upstream:
            yield {**batch, "context_len": 0}

    def set_initial_cost_curves(self, samples) -> None:
        with_context, bare = (
            samples[: self._num_upstream],
            samples[self._num_upstream :],
        )
        super().set_initial_cost_curves(with_context)

        def curve(group, key, value):
            points = defaultdict(list)
            for s in group:
                points[key(s)].append(value(s))
            return {k: float(np.median(v)) for k, v in sorted(points.items())}

        context = curve(
            with_context, lambda s: s.num_target_tokens, lambda s: s.forward_ms
        )
        no_context = curve(bare, lambda s: s.num_target_tokens, lambda s: s.forward_ms)
        drafted = curve(
            [s for s in with_context if s.full_cudagraph],
            lambda s: s.num_reqs,
            lambda s: s.drafter_ms,
        )
        tokens = sorted(set(context) & set(no_context))
        costs = (
            tokens,
            [no_context[t] for t in tokens],
            [context[t] for t in tokens],
            drafted,
            self._profile_context,
        )
        from vllm.distributed.parallel_state import get_tp_group

        self.speculator.on_step_costs(*get_tp_group().broadcast_object(costs, src=0))

    def get_num_tokens(
        self, num_tokens_per_req: dict[str, int], draft_tokens: dict[str, list[int]]
    ) -> int:
        req_ids = list(num_tokens_per_req)
        scheduled = np.array(
            [len(draft_tokens.get(r, ())) for r in req_ids], dtype=np.int64
        )
        non_draft = {
            r: num_tokens_per_req[r] - int(s)
            for r, s in zip(req_ids, scheduled, strict=True)
        }
        non_draft_total = sum(non_draft.values())
        logit_room = max(
            0, self._max_total_logits - len(req_ids) * self.num_bonus_tokens
        )
        slots = np.fromiter(
            (self.req_states.req_id_to_index[r] for r in req_ids),
            dtype=np.int64,
            count=len(req_ids),
        )
        quota = self.speculator.verify_quota(slots, scheduled, non_draft_total)
        quota = int(min(quota, logit_room, scheduled.sum()))
        self._batch_budget = (
            dict(zip(req_ids, scheduled.tolist(), strict=True)),
            non_draft,
            quota,
        )
        return non_draft_total + quota

    def reallocate_drafts(self, req_ids: list[str], idx_mapping: torch.Tensor):
        (drafts_per_req, non_draft_per_req, quota), self._batch_budget = (
            self._batch_budget,
            None,
        )
        n = idx_mapping.shape[0]
        host = self._staging.next()
        host[:2] = n, quota
        host[2 : 2 + n] = [drafts_per_req[r] for r in req_ids]
        host[2 + n : 2 + 2 * n] = [non_draft_per_req[r] for r in req_ids]
        self._staging.copy_to(self._inputs, 2 + 2 * n)
        num_reqs, total = self._inputs[0], self._inputs[1]
        scheduled, non_draft = self._inputs[2 : 2 + n], self._inputs[2 + n : 2 + 2 * n]
        # Capacity = drafts scheduled AND actually drafted; then the global top-B
        # of realized survival, as upstream, bounded by it.
        select = self.speculator.select
        select.survival(
            self._confidence_probs, num_reqs, scheduled, gather=idx_mapping,
            cap_b=self.speculator.widths_by_slot, cap_b_index=idx_mapping,
        )  # fmt: skip
        select.select(
            num_reqs, total, self._batch_draft_capacity, self._cu_num_logits, self.num_bonus_tokens,
            self.query_start_loc, non_draft,
        )  # fmt: skip
        return self._cu_num_logits[: n + 1], self.query_start_loc, quota


class HostJointVerificationManager(
    HostRankingVerificationManager, JointVerificationManager
):
    """The same quota, spread on the host (FORESPARK_CPU_RANKING): for targets whose
    backends plan from host lengths (the GLM models' sparse MLA), the CPU partition
    must be exact. Upstream's lagged confidence snapshot is ranked within the
    widths the host allocated (the speculator's pending caps);
    HostRankingVerificationManager writes the partition to both sides."""

    def get_num_tokens(self, num_tokens_per_req, draft_tokens) -> int:
        total = JointVerificationManager.get_num_tokens(
            self, num_tokens_per_req, draft_tokens
        )
        drafts, non_draft, quota = self._batch_budget
        req_ids = list(drafts)
        slots = np.array([self.req_states.req_id_to_index[r] for r in req_ids])
        caps = np.array([drafts[r] for r in req_ids])
        if (pending := self.speculator.pending) is not None:
            caps = np.minimum(caps, pending[1][slots])
        # Newcomers hold no drafts, placeholders or not.
        if reset := self.speculator.reset_slots:
            caps[np.isin(slots, list(reset))] = 0
        capped = min(quota, int(caps.sum()))
        confidences = self._stale_confidences[self._stale_idx].np[slots]
        survival = np.cumprod(confidences.astype(np.float64), axis=1)
        valid = np.arange(survival.shape[1])[None, :] < caps[:, None]
        widths = prefix_widths(survival, valid, capped)
        self._widths = dict(zip(req_ids, widths.tolist(), strict=True))
        self._batch_budget = (drafts, non_draft, capped)
        return total - quota + capped
