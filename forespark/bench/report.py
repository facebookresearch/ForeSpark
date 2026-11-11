# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Summarise result files: per-arm tables, or one metric with arms as columns.

A cell is (model, phase, parallel, arm, category, rate or concurrency). Files
tagged with a round (`eval.py --round N`) are samples: a cell averages its
rounds and carries a 95% t-interval (`<metric>_ci`). Otherwise, and within one
round, the newer file wins: reruns are corrections. plot/ draws the figures.
"""

import csv
import json
import math
import statistics
from pathlib import Path

from .protocol import row_warnings

COLUMNS = (
    ("acceptance_length", "accept_len", "{:.3f}"),
    ("output_tokens_per_second", "tok/s", "{:.1f}"),
    ("running_peak", "ran", "{:.0f}"),
    ("median_tpot_ms", "tpot_ms", "{:.1f}"),
    ("latency_p50_ms", "p50_ms", "{:.0f}"),
    ("draft_len_observed", "D/step", "{:.2f}"),
    ("verify_len_observed", "B/step", "{:.2f}"),
)
AVERAGED = (
    "output_tokens_per_second", "per_user_tokens_per_second", "acceptance_length", "median_tpot_ms", "median_ttft_ms",
    "latency_p50_ms", "draft_len_observed", "verify_len_observed", "running_peak", "tpot_p99_ms", "ttft_p99_ms",
    "slo_attainment", "goodput_rps",
)  # fmt: skip
# Two-sided 95% Student-t quantiles for 1..30 degrees of freedom.
T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)  # fmt: skip


def interval(values: list[float]) -> float:
    """Half-width of the 95% t-interval of the mean."""
    if len(values) < 2:
        return 0.0
    quantile = T95[len(values) - 2] if len(values) - 2 < len(T95) else 1.96
    return quantile * statistics.stdev(values) / math.sqrt(len(values))


def aggregate(records: list[dict]) -> dict:
    """One cell from its rounds: every numeric field averaged, `_ci` half-widths
    on the headline ones, every round's warnings."""
    merged = {**max(records, key=lambda r: r["_mtime"]), "rounds": len(records)}
    if len(records) > 1:
        for name in [key for key in merged if not key.startswith("_")]:
            values = [r.get(name) for r in records]
            if all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
            ):
                merged[name] = statistics.fmean(values)
                if name in AVERAGED:
                    merged[f"{name}_ci"] = interval(values)
        merged["warnings"] = sorted({w for r in records for w in r["warnings"]})
    return merged


def load_cells(
    root: Path, model: str | None = None, prefix_caching: str | None = None
) -> dict[tuple, dict]:
    samples: dict[tuple, dict] = {}
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "results" not in data or (model and data.get("model") != model):
            continue
        if prefix_caching and data.get("prefix_caching") != prefix_caching:
            continue
        stamp = path.stat().st_mtime
        for row in data["results"]:
            for part in row.get("by_category") or [row]:
                cell = part.get("category", part.get("rate", part.get("concurrency")))
                key = (
                    data["model"],
                    data["phase"],
                    data.get("parallel", "?"),
                    data["arm"],
                    cell,
                )
                rounds = samples.setdefault(key, {})
                if (previous := rounds.get(data.get("round"))) is not None and previous[
                    "_mtime"
                ] >= stamp:
                    continue
                tpot = part.get("median_tpot_ms")
                rounds[data.get("round")] = {**part, "cell": cell, "_mtime": stamp, "warnings": row_warnings(part),
                                             "per_user_tokens_per_second": 1000.0 / tpot if tpot else None}  # fmt: skip
    cells = {}
    for key, rounds in samples.items():
        tagged = {
            round_: record for round_, record in rounds.items() if round_ is not None
        }
        cells[key] = aggregate(list(tagged.values()) or [rounds[None]]) | {
            "_rounds": tagged
        }
    return cells


def _order(cell):
    return (isinstance(cell, str), cell)


def select_arms(
    cells: dict[tuple, dict], specs: list[str]
) -> tuple[dict[tuple, dict], list[str]]:
    """The arms named in `specs` ("LABEL=NAME": the arm whose name up to its first
    "+" is LABEL, shown as NAME), and the names in spec order."""
    names = dict(spec.split("=", 1) if "=" in spec else (spec, spec) for spec in specs)
    picked, source = {}, {}
    for (model, phase, parallel, arm, cell), record in cells.items():
        name = names.get(arm.split("+", 1)[0])
        if name is None:
            continue
        group = (model, phase, parallel, name)
        if source.setdefault(group, arm) != arm:
            raise SystemExit(f"{source[group]} and {arm} both show as {name}")
        picked[(model, phase, parallel, name, cell)] = record
    return picked, list(dict.fromkeys(names.values()))


def arm_key(order: list[str] | None):
    """Sort key: an arm's place in `order` (select_arms'), then its name."""
    position = {name: i for i, name in enumerate(order or [])}
    return lambda arm: (position.get(arm, len(position)), arm)


def ranked(arms, order: list[str] | None = None) -> list[str]:
    return sorted(arms, key=arm_key(order))


def print_tables(cells: dict[tuple, dict], order: list[str] | None = None) -> None:
    groups: dict[tuple, list] = {}
    for (model, phase, parallel, arm, _), record in cells.items():
        groups.setdefault((model, phase, parallel, arm), []).append(record)

    def value(record: dict, name: str, fmt: str) -> str:
        if record.get(name) is None:
            return "-"
        ci = record.get(f"{name}_ci")
        return fmt.format(record[name]) + (f"±{fmt.format(ci)}" if ci else "")

    key = arm_key(order)
    for (model, phase, parallel, arm), records in sorted(
        groups.items(), key=lambda g: (g[0][:3], key(g[0][3]))
    ):
        print(f"\n{model} / {phase} / {parallel} / {arm}")
        print(
            f"  {'cell':>14}"
            + "".join(f"{label:>16}" for _, label, _ in COLUMNS)
            + f"{'rounds':>8}"
        )
        for record in sorted(records, key=lambda r: _order(r["cell"])):
            values = "".join(
                f"{value(record, name, fmt):>16}" for name, _, fmt in COLUMNS
            )
            print(
                f"{'!' if record['warnings'] else ' '} {str(record['cell']):>14}{values}{record['rounds']:>8}"
            )


def print_comparison(
    cells: dict[tuple, dict],
    metric: str,
    out: Path,
    markdown: bool = False,
    order: list[str] | None = None,
) -> None:
    """One metric, arms as columns; a macro-average over cells every arm has.
    `markdown`: Markdown tables instead of the plain table and `out`'s CSV."""
    grouped: dict[tuple, dict[str, dict]] = {}
    for (model, phase, parallel, arm, cell), record in cells.items():
        if record.get(metric) is not None:
            grouped.setdefault((model, phase, parallel), {}).setdefault(arm, {})[
                cell
            ] = record[metric]
    if markdown:
        _markdown_comparison(grouped, metric, order)
        return
    rows = []
    for (model, phase, parallel), arms in sorted(grouped.items()):
        names = ranked(arms, order)
        width = max(12, min(24, 96 // len(names)))
        print(f"\n{model} / {phase} / {parallel} -- {metric}")
        print(
            f"  {'cell':<14}" + "".join(f"{n[-(width - 2) :]:>{width}}" for n in names)
        )
        keys = sorted({c for by_cell in arms.values() for c in by_cell}, key=_order)
        for cell in keys:
            print(
                f"  {str(cell):<14}"
                + "".join(
                    f"{arms[n][cell]:>{width}.4f}"
                    if cell in arms[n]
                    else f"{'-':>{width}}"
                    for n in names
                )
            )
            rows.append(
                {"model": model, "phase": phase, "cell": cell}
                | {n: arms[n].get(cell) for n in names}
            )
        common = [c for c in keys if all(c in arms[n] for n in names)]
        if len(common) > 1:
            print(
                f"  {'macro-average':<14}"
                + "".join(
                    f"{sum(arms[n][c] for c in common) / len(common):>{width}.4f}"
                    for n in names
                )
            )
        if len(common) > 1 and all(arms[n][c] > 0 for n in names for c in common):
            print(
                f"  {'geomean':<14}"
                + "".join(
                    f"{statistics.geometric_mean(arms[n][c] for c in common):>{width}.4f}"
                    for n in names
                )
            )
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n-> {out}")


def _markdown_comparison(
    grouped: dict[tuple, dict[str, dict]], metric: str, order: list[str] | None = None
) -> None:
    fmt = next((f for key, _, f in COLUMNS if key == metric), "{:.4f}")

    def row(label, values) -> None:
        print(f"| {label} | " + " | ".join(values) + " |")

    for (model, phase, parallel), arms in sorted(grouped.items()):
        names = ranked(arms, order)
        keys = sorted({c for by_cell in arms.values() for c in by_cell}, key=_order)
        print(f"\n## {model} / {phase} / {parallel}: {metric}\n")
        row("category" if all(isinstance(c, str) for c in keys) else "cell", names)
        print("|---|" + "---:|" * len(names))
        for cell in keys:
            values = [arms[n].get(cell) for n in names]
            row(cell, ["-" if v is None else fmt.format(v) for v in values])
        common = [c for c in keys if all(c in arms[n] for n in names)]
        if len(common) > 1:
            means = [statistics.fmean(arms[n][c] for c in common) for n in names]
            row("macro-average", [fmt.format(m) for m in means])
