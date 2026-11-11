# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fore-D's planner against brute force (CPU only, no vLLM)."""

import itertools

import numpy as np
import pytest
import torch
from forespark.fore_d.scheduler.allocate import (
    allocate_verify,
    allocate_widths,
    hull_gains,
    prefix_counts,
    width_expected,
)
from forespark.fore_d.scheduler.cost import (
    CostModel,
    extend_sizes,
    isotonic,
    MeasuredCosts,
)
from forespark.fore_d.scheduler.planner import JointPlanner, Plan, safe_verify_quota


def random_rows(rng, n: int, k: int) -> np.ndarray:
    """Nonincreasing survival rows, with deliberate ties."""
    conditional = rng.choice([0.2, 0.5, 0.8, 1.0], size=(n, k))
    return np.cumprod(conditional, axis=1)


def toy_cost(n_max: int, k: int) -> CostModel:
    rows = np.array([1, 2, 4, 8, 12, 16, 24, 32])
    samples = np.array([0, 1, 2, 4, 6, 8, 12, 16, 24])
    tokens = np.array([8.0, 16.0, 32.0, 64.0])  # graph sizes: steps between them
    return CostModel(
        row_buckets=rows,
        backbone_ms=1.0 + 0.05 * rows,
        sample_buckets=samples,
        tail_ms=0.3 + 0.02 * samples,
        verify_tokens=tokens,
        verify_bare_ms=5.0 + 0.1 * tokens,
        verify_context_ms=5.0
        + 0.1 * tokens
        + 0.3 * tokens,  # 0.3 ms per request's context read
        fixed_requests=np.array([1.0, 8.0]),
        fixed_ms=np.array([0.5, 0.5]),
    )


def test_verify_cost_separates_tokens_from_context_reads():
    cost = toy_cost(8, 3)
    # 8 requests, 16 tokens: the 16-token graph without context, plus 8 context reads.
    assert cost.verify_cost(8, np.array([16]))[0] == pytest.approx(5.0 + 1.6 + 2.4)
    # Tokens pad up to the next profiled graph; past the last size, linear.
    assert cost.verify_cost(8, np.array([17]))[0] == pytest.approx(5.0 + 3.2 + 2.4)
    assert cost.verify_cost(8, np.array([80]))[0] == pytest.approx(5.0 + 8.0 + 2.4)


def test_verify_cost_is_continuous_past_the_graph_limit():
    cost = toy_cost(8, 3)
    cost.graph_limit = 16.0  # 32 and 64 were profiled past the largest graph
    # Up to 16 tokens pad to the next graph; past it nothing is blended across
    # 16, 32..64 interpolates, and past 64 it is linear.
    assert cost.verify_cost(8, np.array([12]))[0] == pytest.approx(5.0 + 1.6 + 2.4)
    assert cost.verify_cost(8, np.array([20]))[0] == pytest.approx(5.0 + 3.2 + 2.4)
    assert cost.verify_cost(8, np.array([48]))[0] == pytest.approx(5.0 + 4.8 + 2.4)
    assert cost.verify_cost(8, np.array([80]))[0] == pytest.approx(5.0 + 8.0 + 2.4)


def test_measured_costs_replace_the_profiles_verify_step():
    measured = MeasuredCosts(
        requests=np.array([8.0, 32.0]),
        verify_per_request=[np.array([1.0, 3.0]), np.array([1.0, 3.0])],
        verify_ms=[np.array([6.0, 8.0]), np.array([10.0, 14.0])],
    )
    # Linear in drafts per request within a level, continued past its points.
    assert measured.verify(8, np.array([16]))[0] == pytest.approx(7.0)
    assert measured.verify(8, np.array([0]))[0] == pytest.approx(5.0)
    # Linear in log n across levels: 16 requests sit halfway between 8 and 32.
    assert measured.verify(16, np.array([32]))[0] == pytest.approx((7.0 + 12.0) / 2)
    assert measured.verify(64, np.array([128]))[0] == pytest.approx(12.0)  # clamped
    cost = toy_cost(8, 3)
    cost.measured = measured
    step = cost.base_ms(8, 8, np.array([56]), np.array([16]))[0]
    profile_draft = 0.5 + float(cost.draft_ms(8, np.array([56]))[0])
    assert step == pytest.approx(
        profile_draft + 7.0
    )  # the profile's draft step + verify
    assert cost.context_ms(8, 10_000.0) == 0.0  # timed forwards read a context
    # Prefill rows add the profile's price above the decode step: 40 tokens pad
    # to its 64-token graph, the decode step's 24 to the 32-token one.
    prefill = cost.base_ms(8, 24, np.array([56]), np.array([16]))[0]
    assert prefill == pytest.approx(step + 3.2)
    copy = CostModel.from_dict(cost.to_dict())
    assert copy.base_ms(8, 8, np.array([56]), np.array([16]))[0] == pytest.approx(step)


def test_calibrate_costs_fits_each_timed_request_count(tmp_path):
    from forespark.train import calibrate_costs

    # Fields: monotonic, reqs, tokens, drafted, full graph, prep, fwd, sample, draft,
    # post, idle.
    rows = []
    for w, forward, draft in ((1, 6.0, 1.0), (2, 7.0, 1.5)):
        rows += [[0, 8, 8 + 8 * w, 8 * w, 1, 0.2, forward, 0.1, draft, 0.05, 0]] * 5
    rows += [[0, 8, 40, 8, 0, 9, 99.0, 9, 9, 9, 0]] * 5  # a prefill step: ignored
    path = tmp_path / "run.steps.bin"
    np.array(rows, dtype=np.float64).tofile(path)
    steps = calibrate_costs.load_steps([path])
    costs = calibrate_costs.fit(steps, min_steps=4)
    assert costs.requests.tolist() == [8.0]
    assert costs.verify_per_request[0].tolist() == [1.0, 2.0]
    assert costs.verify_ms[0].tolist() == [6.0, 7.0]
    cost = toy_cost(8, 3)
    cost.measured = costs
    profile_draft = 0.5 + float(cost.draft_ms(8, np.array([16]))[0])
    step = cost.base_ms(8, 8, np.array([16]), np.array([8]))[0]
    assert step == pytest.approx(profile_draft + 6.0)


def test_calibrate_costs_extends_a_table_past_its_last_level():
    from forespark.train import calibrate_costs

    def table(levels):
        b = [np.array([1.0, 2.0]) for _ in levels]
        ms = [np.array([n / 10, n / 5]) for n in levels]
        return MeasuredCosts(np.array(levels, dtype=float), b, ms)

    base = table([1, 128, 151])
    merged = calibrate_costs.extend(base, table([150, 192, 231]))
    assert merged.requests.tolist() == [1, 128, 151, 192, 231]  # 150 stays base's
    assert merged.verify_ms[2].tolist() == [15.1, 30.2]
    assert merged.verify(192, np.array([192.0]))[0] == pytest.approx(19.2)
    assert merged.verify(256, np.array([256.0]))[0] == pytest.approx(
        23.1
    )  # held at 231
    with pytest.raises(SystemExit, match="no request count above"):
        calibrate_costs.extend(base, table([64, 151]))


def test_measured_costs_refuse_a_table_with_draft_step_costs():
    table = {
        "requests": [8.0],
        "verify_per_request": [[1.0, 2.0]],
        "verify_ms": [[6.0, 7.0]],
    }
    assert MeasuredCosts.from_dict(
        table | {"draft_ms": [], "rest_ms": []}
    ).requests.tolist() == [8.0]
    with pytest.raises(ValueError, match="verify-only"):
        MeasuredCosts.from_dict(table | {"draft_ms": [[1.0, 1.5]], "rest_ms": [0.35]})


def test_extend_sizes_continues_past_the_largest_graph():
    vllm = [1, 2, 4, 8, 16, 256, 512]
    assert extend_sizes(vllm, 4096)[7] == 576
    assert extend_sizes(vllm, 4096)[-3:] == [3968, 4032, 4096]
    assert extend_sizes(vllm, 3584)[-1] == 3584
    assert extend_sizes(vllm, 300) == [1, 2, 4, 8, 16, 256, 300]


def test_a_prefill_past_the_largest_graph_still_verifies():
    cost = toy_cost(8, 3)
    cost.graph_limit = 16.0
    reward = JointPlanner(cost).reward_curve(np.full((4, 3), 0.9), np.full(4, 3))
    graphs_only = JointPlanner(cost)
    graphs_only.verify_buckets = np.array([8, 16])
    assert graphs_only.replan_quota(12, reward, 4, non_draft=20) == 0
    past = JointPlanner(cost)
    past.verify_buckets = np.array(extend_sizes([8, 16], 64, step=8))
    assert past.replan_quota(12, reward, 4, non_draft=20) == 12


def test_context_reads_scale_with_the_batch_context():
    cost = toy_cost(8, 3)
    cost.profile_context = 1000.0
    assert (
        cost.context_ms(8, 8 * 1000)
        == pytest.approx(cost.context_ms(8))
        == pytest.approx(2.4)
    )
    assert cost.context_ms(8, 8 * 3000) == pytest.approx(3 * 2.4)
    assert cost.step_ms(8, 8, np.array([4]), np.array([2]), 8 * 3000)[
        0
    ] == pytest.approx(cost.base_ms(8, 8, np.array([4]), np.array([2]))[0] + 7.2)


@pytest.mark.parametrize("seed", range(10))
def test_longer_context_never_buys_a_cheaper_step(seed):
    # The context reads add the same constant to every (D, B): a larger one
    # makes extra drafts relatively cheaper, never dearer.
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(2, 9)), 7
    cost = toy_cost(n, k)
    cost.profile_context = 1000.0
    planner = JointPlanner(cost)
    reward = planner.reward_curve(random_rows(rng, n, k), np.full(n, k))
    steps = []
    for context in (100, 1000, 10000, 100000):
        plan = planner.plan(reward, n, n, context_tokens=n * context)
        steps.append(
            cost.base_ms(
                n, n, np.array([plan.draft_total]), np.array([plan.verify_quota])
            )[0]
        )
    assert steps == sorted(steps)


def best_reward(rows: np.ndarray, legal: np.ndarray, draft: int, quota: int) -> float:
    """max over width vectors with sum D of the best B-position verification."""
    best = -1.0
    for widths in itertools.product(*(range(w + 1) for w in legal)):
        if sum(widths) != draft:
            continue
        drafted = np.concatenate(
            [rows[i, :w] for i, w in enumerate(widths)] or [np.zeros(0)]
        )
        best = max(best, np.sort(drafted)[::-1][:quota].sum())
    return best


@pytest.mark.parametrize("seed", range(20))
def test_plan_is_the_argmax_over_all_width_vectors(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 4)), 3
    rows, legal = random_rows(rng, n, k), rng.integers(0, k + 1, size=n)
    planner = JointPlanner(toy_cost(8, k))
    planner.verify_buckets = np.array([1, 2, 4, 8, 16, 32, 63])
    plan = planner.plan(planner.reward_curve(rows, legal), n, non_draft=n)
    if legal.sum() == 0:
        assert (plan.draft_total, plan.verify_quota) == (0, 0)
        return
    draft, quota = planner.candidates(n, n, int(legal.sum()))
    rate = [
        (n + best_reward(rows, legal, d, b))
        / planner.cost.step_ms(n, n, np.array([d]), np.array([b]))[0]
        for d, b in zip(draft, quota, strict=True)
    ]
    assert plan.rate == pytest.approx(max(rate) if len(rate) else 0.0)
    assert plan.verify_quota <= plan.draft_total <= legal.sum()


def test_reward_curve_is_sorted_prefix_sum():
    rng = np.random.default_rng(0)
    rows, legal = random_rows(rng, 4, 5), np.array([5, 0, 3, 2])
    reward = JointPlanner(toy_cost(8, 5)).reward_curve(rows, legal)
    for quota in range(legal.sum() + 1):
        assert reward[quota] == pytest.approx(
            best_reward(rows, legal, int(legal.sum()), quota)
        )


@pytest.mark.parametrize("seed", range(10))
def test_device_allocations_are_exact_prefixes(seed):
    rng = np.random.default_rng(seed)
    n, k = 6, 7
    rows = torch.tensor(random_rows(rng, n, k), dtype=torch.float32)
    legal = torch.tensor(rng.integers(0, k + 1, size=n), dtype=torch.int32)
    valid = np.concatenate([rows[i, : int(legal[i])].numpy() for i in range(n)])
    for total in range(int(legal.sum()) + 1):
        widths = allocate_widths(rows, legal, torch.tensor(total))
        assert int(widths.sum()) == total and bool((widths <= legal).all())
        # Prefixes holding exactly the top-`total` valid survivals.
        drafted = np.concatenate(
            [rows[i, : int(w)].numpy() for i, w in enumerate(widths)]
        )
        assert drafted.sum() == pytest.approx(np.sort(valid)[::-1][:total].sum())


def test_verify_allocation_respects_capacity_and_quota():
    confidence = torch.tensor([[0.9, 0.9, 0.9], [0.5, 1.0, 1.0], [1.0, 1.0, 0.1]])
    capacity = torch.tensor([2, 3, 3], dtype=torch.int32)
    lengths = allocate_verify(confidence, capacity, torch.tensor(5))
    assert lengths.tolist() == [
        2,
        1,
        2,
    ]  # survivals .9 .81 | .5 .5 .5 | 1 1 .1 -> top-5 prefixes


def test_safe_quota_accounts_for_departures_and_short_schedules():
    plan = Plan(draft_total=12, verify_quota=10, rate=0.0)
    legal, drafted = (
        np.array([4, 0, 4, 4]),
        np.array([0, 2, 3]),
    )  # slots 0, 2, 3 drafted 4 each

    def quota(scheduled_by_slot):
        return safe_verify_quota(plan, legal, drafted, np.array(scheduled_by_slot))

    assert quota([4, 0, 4, 4]) == 10
    assert quota([4, 0, 4, 0]) == 8  # slot 3's request left with up to 4
    assert quota([1, 0, 4, 4]) == 9  # slot 0 scheduled 3 fewer
    assert quota([4, 7, 4, 0]) == 8  # a new request in slot 1 changes nothing
    # A new request took drafted slot 3 and the scheduler gave it K placeholders (its
    # prompt was all in the prefix cache): slot 3's drafts are gone all the same.
    reset = safe_verify_quota(
        plan, legal, drafted, np.array([4, 0, 4, 15]), reset_slots={3}
    )
    assert reset == 8


def test_host_ranking_gives_a_newcomer_in_a_drafted_slot_nothing_to_verify():
    """On the host path (CPU ranking): a newcomer that took a drafted slot with K
    placeholders is given nothing to verify once its slot is reset."""
    from types import SimpleNamespace

    from forespark.fore_d.runtime.verification import HostJointVerificationManager

    k = 4
    manager = HostJointVerificationManager.__new__(HostJointVerificationManager)
    manager.req_states = SimpleNamespace(req_id_to_index={"a": 0, "b": 1, "new": 3})
    manager._max_total_logits, manager.num_bonus_tokens, manager._stale_idx = (
        1 << 20,
        1,
        0,
    )
    manager._stale_confidences = [
        SimpleNamespace(np=np.full((4, k), 0.9, dtype=np.float32))
    ]
    # "old" drafted 4 in slot 3 and finished; "new" took the slot this step.
    caps_by_slot = np.array([4, 4, 0, 4])
    manager.speculator = SimpleNamespace(
        pending=(Plan(12, 12, 1.0), caps_by_slot, np.array([0, 1, 3]), None),
        reset_slots={3}, verify_quota=lambda slots, scheduled, non_draft: 12,
    )  # fmt: skip
    tokens = {r: 1 + k for r in ("a", "b", "new")}  # a decode row + K placeholders each
    drafts = {r: [-1] * k for r in tokens}
    assert manager.get_num_tokens(tokens, drafts) == 3 + 8
    assert manager._widths == {"a": 4, "b": 4, "new": 0}


def test_isotonic_fit_is_monotone_least_squares():
    assert isotonic(np.array([1.0, 3.0, 2.0, 4.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert isotonic(np.array([5.0, 1.0, 1.0])).tolist() == pytest.approx([7 / 3] * 3)


def best_widths_value(expected: np.ndarray, legal: np.ndarray, draft: int) -> float:
    """Exact max of sum_i F_i(d_i) over integer widths d_i <= legal_i with sum d_i = draft (DP)."""
    best = np.full(draft + 1, -np.inf)
    best[0] = 0.0
    for f, limit in zip(expected, legal, strict=True):
        values = np.concatenate(([0.0], f[:limit]))
        nxt = np.full(draft + 1, -np.inf)
        for width, value in enumerate(values[: draft + 1]):
            nxt[width:] = np.maximum(nxt[width:], best[: draft + 1 - width] + value)
        best = nxt
    return float(best[draft])


def test_width_expected_is_the_survival_sum_of_each_width_row():
    rng = np.random.default_rng(0)
    k = 4
    logits = torch.tensor(rng.normal(size=(3, k * (k + 1) // 2)), dtype=torch.float32)
    expected = width_expected(logits).numpy()
    p = 1 / (1 + np.exp(-logits.numpy()))
    for d in range(1, k + 1):
        row = p[:, d * (d - 1) // 2 : d * (d + 1) // 2]
        assert np.allclose(
            expected[:, d - 1], np.cumprod(row, axis=1).sum(axis=1), atol=1e-6
        )


@pytest.mark.parametrize("seed", range(20))
def test_hull_widths_are_optimal_on_concave_rows_and_bracket_the_optimum_otherwise(
    seed,
):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(1, 5)), 5
    legal = rng.integers(0, k + 1, size=n)
    for concave in (True, False):
        steps = (
            np.sort(rng.uniform(-0.2, 1.0, size=(n, k)), axis=1)[:, ::-1]
            if concave
            else rng.uniform(-0.5, 1.0, size=(n, k))
        )
        expected = np.cumsum(steps, axis=1)
        gains = hull_gains(
            torch.tensor(expected, dtype=torch.float32), torch.tensor(legal)
        )
        g = gains.numpy()
        for i in range(n):
            assert np.all(np.diff(g[i, : legal[i]]) <= 1e-6)  # a prefix of every row
        for draft in range(int(legal.sum()) + 1):
            widths = prefix_counts(
                gains, torch.tensor(legal), torch.tensor(draft)
            ).numpy()
            assert widths.sum() == draft and np.all(widths <= legal)
            realized = sum(
                expected[i, w - 1] if w else 0.0 for i, w in enumerate(widths)
            )
            relaxed = sum(g[i, :w].sum() for i, w in enumerate(widths))
            optimum = best_widths_value(expected, legal, draft)
            if concave:
                assert realized == pytest.approx(optimum, abs=1e-5)
            else:
                assert realized <= optimum + 1e-5 and optimum <= relaxed + 1e-5


def test_fore_d_fit_converges_where_undamped_newton_diverged():
    # Shaped like the Alpaca logs' first positions: logits up to +15 against a
    # true scale of 0.18. An undamped Newton step went to scale ~3e11 here.
    from forespark.train.calibrate_fore_d import fit_affine

    rng = np.random.default_rng(0)
    logits = np.clip(rng.normal(1.6, 2.5, 200_000), -6.0, 15.0)
    labels = (
        rng.random(logits.size) < 1.0 / (1.0 + np.exp(-(0.18 * logits + 0.73)))
    ).astype(np.float64)
    scale, bias = fit_affine(logits, labels)
    assert scale == pytest.approx(0.18, abs=0.01) and bias == pytest.approx(
        0.73, abs=0.02
    )
