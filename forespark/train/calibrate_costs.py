# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fit Fore-D's measured step costs (`--fore-d-costs`) from timed steps.

    ./calibrate.py costs RUN.steps.bin ... --out costs.json
    ./calibrate.py costs WIDE.steps.bin ... --extend costs.json --out costs-ext.json

Inputs are the step timer's files (`eval.py ... --step-timing` writes one beside
each result): the forced-width log runs give, at each concurrency, one verify
point per width (every request drafts w and verifies all of it). Only
full-CUDA-graph steps count: decode-only, so the target's tokens are n + B.
Per request count with enough steps: the target forward by drafts verified per
request (the median per point, made nondecreasing). The planner keeps the
startup profile's draft step: a log run's pre-draft stage runs eagerly.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ..bench.protocol import STEP_TIMING_FIELDS
from ..fore_d.scheduler.cost import isotonic, MeasuredCosts

COLUMN = {name: i for i, name in enumerate(STEP_TIMING_FIELDS)}


def load_steps(paths: list[Path]) -> np.ndarray:
    width = len(STEP_TIMING_FIELDS)
    rows = [np.fromfile(p, dtype=np.float64) for p in paths]
    rows = [r[: len(r) // width * width].reshape(-1, width) for r in rows]
    steps = np.concatenate(rows) if rows else np.zeros((0, width))
    return steps[steps[:, COLUMN["full_cudagraph"]] == 1]


def curve(
    x: np.ndarray, y: np.ndarray, min_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Median y per distinct x (at least `min_points` steps each), nondecreasing."""
    xs = [v for v in np.unique(x) if (x == v).sum() >= min_points]
    if len(xs) < 2:
        return None
    return np.array(xs, dtype=np.float64), isotonic(
        np.array([np.median(y[x == v]) for v in xs])
    )


def fit(steps: np.ndarray, min_steps: int = 30, min_points: int = 3) -> MeasuredCosts:
    n_col = steps[:, COLUMN["num_reqs"]]
    levels, verify_b, verify_ms = [], [], []
    for n in np.unique(n_col):
        s = steps[n_col == n]
        if len(s) < min_steps or n < 1:
            continue
        verify = curve(
            (s[:, COLUMN["num_tokens"]] - n) / n, s[:, COLUMN["forward_ms"]], min_points
        )
        if verify is None:
            continue
        levels.append(float(n))
        verify_b.append(verify[0])
        verify_ms.append(verify[1])
    if not levels:
        raise SystemExit(
            "no request count has two timed verify points: time the forced-width runs"
        )
    return MeasuredCosts(np.array(levels), verify_b, verify_ms)


def extend(base: MeasuredCosts, high: MeasuredCosts) -> MeasuredCosts:
    """`base`'s levels, then `high`'s above its last: a table timed under the final
    servers past the residency a calibration reached (the planner holds larger
    batches at its last level)."""
    keep = [i for i, n in enumerate(high.requests) if n > base.requests[-1]]
    if not keep:
        raise SystemExit(f"no request count above n = {base.requests[-1]:g} to add")
    return MeasuredCosts(
        np.concatenate([base.requests, high.requests[keep]]),
        base.verify_per_request + [high.verify_per_request[i] for i in keep],
        base.verify_ms + [high.verify_ms[i] for i in keep],
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("steps", type=Path, nargs="+", help="step timer files (.steps.bin)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--min-steps", type=int, default=30, help="timed steps a request count needs"
    )
    p.add_argument(
        "--extend",
        type=Path,
        metavar="COSTS",
        help="keep this table's levels and add the fitted ones above its last",
    )
    args = p.parse_args(argv)
    costs = fit(load_steps(args.steps), args.min_steps)
    if args.extend:
        costs = extend(MeasuredCosts.load(args.extend), costs)
    for i, n in enumerate(costs.requests):
        b = np.round(costs.verify_per_request[i], 2).tolist()
        print(
            f"n={int(n):4d}: verify {b} -> {np.round(costs.verify_ms[i], 2).tolist()} ms"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(costs.to_dict()) + "\n")
    print(f"{len(costs.requests)} request counts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
