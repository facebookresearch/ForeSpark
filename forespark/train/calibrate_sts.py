# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fit the confidence head's per-position temperatures (the sts.json for --sts).

Adaptive verification compares expected accepted tokens across budgets, so it
needs the magnitude of the survival probabilities, not only their order; raw
heads are overconfident. Sequential temperature scaling (DSpark 3.2.1) fits
position k against the ECE of the calibrated survival product so far.

    ./calibrate.py sts collect qwen14 --draft CKPT --method forespark --out cal/sts-forespark
    ./calibrate.py sts fit --shards 'cal/sts-forespark/raw/sts.*.npz' --out cal/sts-forespark/sts.json

Collection serves the drafter at fixed k with every draft verified, eagerly
(the recorder is Python in the draft path), on held-out Alpaca prompts
(`eval.py acceptance`: its collect.json also holds the drafter's acceptance
length). Fit one file per checkpoint, k and decoding temperature.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..paths import ROOT

ALPACA, ALPACA_REVISION = (
    "tatsu-lab/alpaca",
    "dce01c9b08f87459cf36a430d809084718273017",
)  # in neither training pool
GRID = np.power(
    10.0, np.arange(-1.0, 1.0 + 1e-9, 0.05)
)  # the reference fitter's lattice
NUM_BINS = 15


def ece(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Equal-width binned |confidence - accuracy| weighted by bin population."""
    if not predicted.size:
        return 0.0
    index = np.clip(
        np.digitize(predicted, np.linspace(0, 1, NUM_BINS + 1)[1:-1], right=True),
        0,
        NUM_BINS - 1,
    )
    return float(sum((index == b).mean() * abs(observed[index == b].mean() - predicted[index == b].mean())
                     for b in range(NUM_BINS) if (index == b).any()))  # fmt: skip


def fit(logits: np.ndarray, survived: np.ndarray) -> dict:
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    calibrated = raw = np.ones(len(logits))
    temperatures, before, after = [], [], []
    for k in range(logits.shape[1]):
        column = logits[:, k].astype(np.float64)
        raw = raw * sigmoid(column)
        before.append(ece(raw, survived[:, k]))
        best = min(
            GRID, key=lambda t: ece(calibrated * sigmoid(column / t), survived[:, k])
        )
        temperatures.append(float(best))
        calibrated = calibrated * sigmoid(column / best)
        after.append(ece(calibrated, survived[:, k]))
    return {"temperatures": temperatures, "ece_before": before, "ece_after": after, "num_samples": len(logits),
            "dataset": f"{ALPACA}@{ALPACA_REVISION}:train"}  # fmt: skip


def alpaca_rows(limit: int) -> list[dict]:
    """The first `limit` held-out Alpaca prompts as chat rows (instruction, then input)."""
    from datasets import load_dataset

    rows = []
    for row in load_dataset(ALPACA, revision=ALPACA_REVISION, split="train"):
        prompt = "\n\n".join(
            p
            for p in (row["instruction"].strip(), (row.get("input") or "").strip())
            if p
        )
        if prompt:
            rows.append(
                {
                    "category": "alpaca",
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
        if len(rows) >= limit:
            break
    return rows


def write_prompts(args) -> int:
    rows = alpaca_rows(args.num_prompts)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} prompts from {ALPACA}@{ALPACA_REVISION} -> {args.out}")
    return 0


def collect(args) -> int:
    out = Path(args.out)
    if list((out / "raw").glob("sts.*.npz")):
        raise SystemExit(f"shards already exist under {out / 'raw'}; remove them first")
    rows = alpaca_rows(args.num_prompts)
    out.mkdir(parents=True, exist_ok=True)
    prompts = out / "alpaca.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cmd = [
        sys.executable, str(ROOT / "eval.py"), "acceptance", args.model, "--draft", args.draft,
        "--method", args.method, "--prompts", str(prompts), "--sts-collect", str(out / "raw"),
        "--concurrency", str(args.concurrency), "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature), "--gpus", args.gpus, "--port", str(args.port),
        "--out", str(out / "collect.json"),
        *(["--k", str(args.k)] if args.k else []),
        *(["--draft-attention-backend", args.draft_attention_backend] if args.draft_attention_backend else []),
    ]  # fmt: skip
    if code := subprocess.call(cmd):
        return code
    shards = sorted((out / "raw").glob("sts.*.npz"))
    print(f"{len(shards)} shard(s) under {out / 'raw'}")
    return 0 if shards else 1


def run_fit(args) -> int:
    paths = sorted(Path(args.shards).parent.glob(Path(args.shards).name))
    if not paths:
        raise SystemExit(f"no shards matched {args.shards!r}")
    shards = [np.load(p) for p in paths]
    logits = np.concatenate([s["confidence_logits"] for s in shards])
    survived = np.concatenate([s["survived"] for s in shards])
    result = fit(logits, survived) | {"collect_temperature": args.temperature}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for k, t in enumerate(result["temperatures"]):
        print(
            f"position {k + 1}: T {t:.4f}  ECE {result['ece_before'][k]:.4f} -> {result['ece_after'][k]:.4f}"
        )
    print(f"{len(logits)} blocks -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect", help="serve at fixed k and record Alpaca outcomes")
    c.add_argument("model")
    c.add_argument("--draft", required=True)
    c.add_argument("--method", default="forespark", choices=["forespark", "dspark"])
    c.add_argument("--draft-attention-backend", help="the drafter's, as eval.py's")
    c.add_argument(
        "--k", type=int, help="the drafter's draft length (default: the target's)"
    )
    c.add_argument("--out", required=True, metavar="DIR")
    c.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="fit one file per decoding temperature",
    )
    c.add_argument("--num-prompts", type=int, default=1000)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--concurrency", type=int, default=16)
    c.add_argument("--gpus", default="0")
    c.add_argument("--port", type=int, default=8100)
    f = sub.add_parser("fit", help="temperatures from collected shards")
    f.add_argument("--shards", required=True, help="glob of the collect output")
    f.add_argument("--out", required=True)
    f.add_argument(
        "--temperature", type=float, default=0.0, help="recorded in the file"
    )
    w = sub.add_parser(
        "prompts",
        help="write held-out Alpaca prompts (Fore-D's forced-width calibration runs)",
    )
    w.add_argument("--out", required=True)
    w.add_argument("--num-prompts", type=int, default=4000)
    args = p.parse_args(argv)
    return {"collect": collect, "fit": run_fit, "prompts": write_prompts}[args.command](
        args
    )


if __name__ == "__main__":
    raise SystemExit(main())
