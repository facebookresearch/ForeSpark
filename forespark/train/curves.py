# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Summarise drafter training from the console log train.py tees.

    ./curves.py                                   # every run under runs/
    ./curves.py runs/qwen14-forespark-k15 --watch
    ./curves.py runs/* --plot curves.png

Acceptance (`accept_len`) includes the anchor, as both drafters log it.
"""

import argparse
import re
import sys
import time
from pathlib import Path

from ..paths import RUNS

PAIR = re.compile(r"([A-Za-z_][\w/]*)=(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
ENTRY_START = re.compile(
    r"^\s*(?:\[\d{2}:\d{2}:\d{2}\]\s+)?(?:INFO|WARNING|ERROR|DEBUG|CRITICAL)\b"
)
SUMMARY = (  # (metric, label); val keys carry an _epoch suffix
    ("loss", "loss"),
    ("fore_d_loss", "Fore-D loss"),
    ("fore_d_brier", "Fore-D Brier"),
    ("confidence_loss", "confidence loss"),
)


def parse(path: Path) -> list[dict[str, float]]:
    """One record per logged metric dict; Rich wraps long lines, so rejoin them."""
    records, entry, seen, last_step = [], [], set(), None

    def flush():
        nonlocal last_step
        pairs = PAIR.findall(" ".join(entry))
        if any(k.startswith(("train/", "val/")) for k, _ in pairs):
            record = {k: float(v) for k, v in pairs}
            # Validation records carry only the epoch; place them at the last step.
            if "global_step" in record:
                last_step = record["global_step"]
            elif last_step is not None:
                record["global_step"] = last_step
            records.append(record)
        entry.clear()

    for line in path.read_text(errors="replace").splitlines():
        keys = {k for k, _ in PAIR.findall(line)}
        if ENTRY_START.match(line) or keys & seen:
            flush()
            seen = set()
        entry.append(line)
        seen |= keys
    flush()
    return records


def series(records: list[dict], key: str) -> tuple[list[float], list[float]]:
    points = [
        (r.get("global_step", i), r[key]) for i, r in enumerate(records) if key in r
    ]
    return [x for x, _ in points], [y for _, y in points]


def accept_key(records: list[dict], scope: str) -> str | None:
    key = f"{scope}/accept_len{'_epoch' if scope == 'val' else ''}"
    return key if any(key in r for r in records) else None


def summarise(name: str, records: list[dict]) -> None:
    print(name)
    if not records:
        print("  no metrics yet")
        return
    steps = [r["global_step"] for r in records if "global_step" in r]
    print(f"  step {int(max(steps)) if steps else 0}   records {len(records)}")
    for scope in ("train", "val"):
        suffix = "_epoch" if scope == "val" else ""
        for metric, label in SUMMARY:
            _, values = series(records, f"{scope}/{metric}{suffix}")
            if values:
                print(
                    f"  {scope + ' ' + label:<24}{values[-1]:.4f}  (min {min(values):.4f})"
                )
        if key := accept_key(records, scope):
            _, values = series(records, key)
            print(
                f"  {scope + ' accept+anchor':<24}{values[-1]:.4f}  (best {max(values):.4f})"
            )
        _, predicted = series(records, f"{scope}/fore_d_pred_accept_len{suffix}")
        _, target = series(records, f"{scope}/fore_d_target_accept_len{suffix}")
        if predicted and target:
            print(f"  {scope + ' Fore-D A':<24}{predicted[-1]:.4f} / {target[-1]:.4f}  "
                  f"(bias {predicted[-1] - target[-1]:+.4f})")  # fmt: skip


def plot(runs: dict[str, list[dict]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("train/loss", "train loss"), (None, "accepted length (incl. anchor)"),
              ("train/fore_d_loss", "Fore-D loss"), ("train/fore_d_pred_accept_len", "Fore-D predicted A")]  # fmt: skip
    figure, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4))
    for name, records in runs.items():
        for axis, (key, _) in zip(axes, panels, strict=True):
            key = key or accept_key(records, "train")
            if key and (xs := series(records, key))[0]:
                axis.plot(*xs, label=name, linewidth=1)
    for axis, (_, label) in zip(axes, panels, strict=True):
        axis.set(xlabel="step", ylabel=label)
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out, dpi=140)
    print(f"-> {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="*", type=Path, help="run directories or log files")
    p.add_argument("--plot", type=Path, metavar="PNG")
    p.add_argument("--watch", type=float, nargs="?", const=30.0, metavar="SECONDS")
    args = p.parse_args(argv)
    targets = args.runs or sorted(d for d in RUNS.glob("*") if d.is_dir())
    while True:
        runs = {}
        for target in targets:
            log = target if target.is_file() else target / "logs/train.log"
            if log.is_file():
                runs[target.name if target.is_dir() else log.parent.parent.name] = (
                    parse(log)
                )
            else:
                print(f"no log at {log}", file=sys.stderr)
        if not runs:
            return 1
        for name, records in runs.items():
            summarise(name, records)
        if args.plot:
            plot(runs, args.plot)
        if not args.watch:
            return 0
        time.sleep(args.watch)
        print()
