# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Calibrate the Fore-D head per output from Fore-D's step logs.

Position j of a width-d draft is reached when the target accepted positions
< j, and then accepted or not; the head's output (d, j) predicts that
conditional event. The head is triangular (one row per width d, K(K+1)/2
outputs), so output (d, j) is fit on the drafts that drafted width d: the logs
must hold every width -- forced-width runs (`--drafter-width d --fore-d-log`),
merged by passing several LOG_DIRs. A per-output affine map
sigmoid(scale * logit + bias) is fit by logistic regression on the reached
positions (Newton's method, two parameters each). The planner ranks expected
accepted drafts batch-wide, so calibration moves both which requests get
width and how much the planner drafts.

    ./calibrate.py fore-d LOG_DIR [LOG_DIR ...] --out fore_d.json
    eval.py ... --fore-d-calibration fore_d.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .fore_d_logs import load_drafts


def fit_affine(
    logits: np.ndarray, labels: np.ndarray, iterations: int = 100, ridge: float = 1e-6
):
    """Logistic regression of labels on [logit, 1], by Newton steps halved until the loss falls.

    An undamped step diverged (scale ~1e11) on Alpaca logs whose logits reach
    +15: once every prediction saturates, the Hessian is the ridge alone.
    """
    x = np.stack([logits, np.ones_like(logits)], axis=1)
    w = np.array([1.0, 0.0])

    def loss(v: np.ndarray) -> float:
        z = x @ v
        return float(np.sum(np.logaddexp(0.0, z) - labels * z)) + 0.5 * ridge * float(
            v @ v
        )

    current = loss(w)
    for _ in range(iterations):
        p = 0.5 * (1.0 + np.tanh(0.5 * (x @ w)))
        grad = x.T @ (p - labels) + ridge * w
        hess = x.T @ (x * (p * (1 - p))[:, None]) + ridge * np.eye(2)
        step, shrink = np.linalg.solve(hess, grad), 1.0
        while shrink > 1e-10 and loss(w - shrink * step) > current:
            shrink *= 0.5
        if shrink <= 1e-10:
            break
        w = w - shrink * step
        previous, current = current, loss(w)
        if np.abs(shrink * step).max() < 1e-9 or previous - current <= 1e-12 * max(
            1.0, abs(current)
        ):
            break
    return float(w[0]), float(w[1])


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    return float(sum(
        abs(probs[index == b].mean() - labels[index == b].mean()) * (index == b).mean()
        for b in range(bins) if (index == b).any()
    ))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log_dir", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = [load_drafts(d) for d in args.log_dir]
    drafts = {
        key: np.concatenate([r[key] for r in runs])
        for key in ("fore_d_logits", "accepted", "legal", "confidence")
    }
    logits, accepted, legal = (
        drafts["fore_d_logits"],
        drafts["accepted"],
        drafts["legal"],
    )
    k = drafts["confidence"].shape[1]
    # Output o predicts position j of a width-d draft: rows for widths 1..K.
    outputs = [(d, j) for d in range(1, k + 1) for j in range(d)]
    if logits.shape[1] != len(outputs):
        raise SystemExit(
            f"{logits.shape[1]} Fore-D outputs: expected K(K+1)/2 = {len(outputs)}"
        )
    scale, bias, report = [], [], []
    for o, (d, j) in enumerate(outputs):
        reached = (legal == d) & (accepted >= j)
        x, y = (
            logits[reached, o].astype(np.float64),
            (accepted[reached] > j).astype(np.float64),
        )
        if (
            len(y) < 20 or y.min() == y.max()
        ):  # nothing to fit: this width was not logged
            print(f"width {d} position {j}: {len(y)} samples, left uncalibrated")
            scale.append(1.0)
            bias.append(0.0)
            continue
        s, b = fit_affine(x, y)
        before, after = 1 / (1 + np.exp(-x)), 1 / (1 + np.exp(-(s * x + b)))
        scale.append(s)
        bias.append(b)
        report.append(
            {
                "width": d,
                "position": j,
                "samples": int(reached.sum()),
                "rate": float(y.mean()),
                "mean_before": float(before.mean()),
                "ece_before": ece(before, y),
                "ece_after": ece(after, y),
            }
        )
        print(
            f"width {d} position {j}: n={reached.sum():7d} rate {y.mean():.3f}  pred {before.mean():.3f} -> {after.mean():.3f}"
            f"  ECE {ece(before, y):.3f} -> {ece(after, y):.3f}  (scale {s:.3f}, bias {b:+.3f})"
        )
    args.out.write_text(
        json.dumps({"scale": scale, "bias": bias, "positions": report}, indent=2)
    )
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
