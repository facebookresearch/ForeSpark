# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SLO attainment under load (DistServe style), two views x the benchmarked models.

Single column, two rows (sub-plots) of one panel per model, each row on one attainment axis (%), all from the
open-loop ShareGPT runs (Poisson arrivals, scripts/bench/slo.sh). (a) Rate sweep: attainment against the request
rate, from just before the first arm slips below full attainment; a run's capacity is where its line crosses the
dashed 90%, and an arrow marks ForeSpark's gain over the best other line's capacity; the caption lists each
column's SLO. (b) SLO scale: attainment at one fixed rate as both SLOs are scaled by x (log; left of 1 = tighter),
recomputed from the run's request records: the rate (in the caption, per column) is the first measured one past
DSpark's capacity (FIXED_RATE overrides it).

    .venv/bin/python plot/slo.py   ->   plot/plots/slo.pdf
"""

from __future__ import annotations

import style as st

# ============================ Layout knobs (all centimetres) =================
PANEL_H_CM = 1.45  # height of each panel's box
LEFT_BAND_CM = 0.82  # left edge -> panels: the y-axis title and its tick numbers
RIGHT_MARGIN_CM = 0.10  # right panel edge -> figure edge
COL_GAP_CM = 0.22  # gap between a row's panels (they share the y-axis)
TITLE_BAND_CM = 0.24  # above each panel: its model identifier
XLABEL_BAND_CM = 0.55  # below each panel: x tick numbers + the x-axis title
CAPTION_GAP_CM = 0.08  # x-axis title -> caption
CAPTION_BAND_CM = 0.26  # below that: the (a)/(b) caption, hung from the band's top
ROW_GAP_CM = 0.12  # caption -> next row's identifiers
BOTTOM_MARGIN_CM = 0.02  # last caption -> figure bottom edge
LEGEND_TOP_GAP_CM = 0.02  # legend top <-> figure top edge
LEGEND_BAND_CM = 0.40  # figure top -> first row's identifiers (legend included)
LEGEND_COLSPACING = 0.7  # gap between legend entries (font-units)
LEGEND_HANDLELENGTH = 1.6  # sample line length (font-units)
LINE_WIDTH = 0.5  # a baseline's line
OURS_LINE_WIDTH = 0.8  # ForeSpark's line
MARKER_SIZE = 2.2  # marker size (points)
MARKER_EDGE_WIDTH = 0.4
REF_LINE_WIDTH = 0.6  # the dashed 90% line
REF_DASH = (0, (3.0, 1.6))
CALLOUT_WIDTH = 0.7  # capacity arrow line width (points)
CALLOUT_HEAD = 2.5  # capacity arrow head size (points)
CALLOUT_LABEL_Y = 102.0  # the gain label's bottom (attainment %), above every curve
PANEL_ID_PAD_CM = 0.04  # gap between a panel and its identifier above it
YLABEL_PAD_CM = 0.05  # gap between the y-axis title and its tick numbers
XLABEL_PAD_CM = 0.02  # gap between the x tick numbers and the x-axis title
TICK_PAD_CM = 0.04  # gap between a tick number and the axis
TICK_LEN_CM = 0.07  # major tick length
TICK_MINOR_LEN_CM = 0.04  # minor tick length
TICK_WIDTH_CM = 0.02  # tick thickness
SLO_FLOOR = 90.0  # the dashed attainment line (%), and where capacity is read
FULL_PCT = (
    99.0  # rate panels start one measured rate before the first arm drops below this
)
Y_TOP = 120.0  # axis top (%): headroom above 100 for the callout label
Y_MAJOR = 50.0  # labelled attainment ticks (%)
Y_MINOR = 25.0
RATE_STEPS = [
    5.0,
    10.0,
    20.0,
]  # labelled rate-tick steps: the smallest giving at most RATE_MAX_TICKS
RATE_MAX_TICKS = 6
SCALE_TICKS = [
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    4.0,
]  # on a log axis: those within a panel's SLO scales
X_PAD = 0.03  # x range widened by this fraction of the span (log span on a log axis) at each end
# ============================================================================

NAME = "slo"
Y_TITLE = "SLO attain. (%)"
RATE_TITLE = "Rate (req/s)"
SCALE_TITLE = "SLO scale"
ROWS = [("rate", "(a) Rate sweep"), ("scale", "(b) SLO scale")]
PHASE = "sharegpt-rate"
OURS = "forespark"
FIXED_ARM = (
    "dspark"  # the scale panel's rate: the first measured one past this arm's capacity
)
# A model's fixed rate where the rule's leaves the arms too close: at GLM-5.3's 15 req/s DSpark, barely past its
# capacity, met the SLO again by 1.2x
FIXED_RATE = {"glm53": 16.0}
SCALES = [
    0.1,
    0.12,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
    2.0,
]  # even on a log axis
EXTRA_SCALES = {
    "glm53": [3.0, 4.0]
}  # GLM-5.3's baselines stay low to 2x at its fixed rate
DONE = 0  # a request record's status: [arrival s, ttft ms, tpot ms, e2e ms, tokens, status]
_GRAYS = st.grays(2)
# Series, ForeSpark last. (bench arm, legend label, line color, marker, marker fill)
SERIES = [
    ("nospec", "AR", st.BASELINE_LINE, st.MARKERS[0], _GRAYS[0]),
    ("dspark", "DSpark", st.BASELINE_LINE, st.MARKERS[1], _GRAYS[1]),
    (OURS, "ForeSpark", st.FORESPARK_COLOR, st.MARKERS[0], st.FORESPARK_COLOR),
]


def attainment(cell: dict, scale: float) -> float:
    """% of a rate cell's measured requests that finished within scale x both SLOs."""
    ttft, tpot = scale * cell["slo_ttft_ms"], scale * cell["slo_tpot_ms"]
    records = cell["requests"]
    return (
        100.0
        * sum(r[5] == DONE and r[1] <= ttft and r[2] <= tpot for r in records)
        / len(records)
    )


def curves(results, model: str) -> tuple[dict, float, tuple[float, float]]:
    """({(panel, arm): [(x, attainment %), ...]}, the scale panel's rate, (TTFT SLO ms, TPOT SLO ms))."""
    cells = {
        arm: {float(r["rate"]): r for r in st.rows(results, model, PHASE, arm)}
        for arm, *_ in SERIES
    }
    slos = {
        (c["slo_ttft_ms"], c["slo_tpot_ms"])
        for by_rate in cells.values()
        for c in by_rate.values()
    }
    if len(slos) != 1:
        raise SystemExit(
            f"{model}: rate cells scored against different SLOs {sorted(slos)}"
        )
    past = [
        rate
        for rate, cell in cells[FIXED_ARM].items()
        if 100 * cell["slo_attainment"] < SLO_FLOOR
    ]
    fixed = FIXED_RATE.get(model) or min(past, default=max(cells[FIXED_ARM]))
    out = {}
    for arm, by_rate in cells.items():
        out[("rate", arm)] = [
            (rate, 100 * by_rate[rate]["slo_attainment"]) for rate in sorted(by_rate)
        ]
        out[("scale", arm)] = [
            (x, attainment(by_rate[fixed], x))
            for x in SCALES + EXTRA_SCALES.get(model, [])
        ]
    return out, fixed, slos.pop()


def capacity(pts: list[tuple[float, float]]) -> float | None:
    """The highest rate still meeting the SLO for SLO_FLOOR % of requests, interpolated where
    attainment first falls below it; None if it never holds."""
    previous = None
    for rate, attained in pts:
        if attained < SLO_FLOOR:
            if previous is None:
                return None
            low, held = previous
            return low + (rate - low) * (held - SLO_FLOOR) / (held - attained)
        previous = (rate, attained)
    return previous[0] if previous else None


def rate_window(series: dict) -> float:
    """The first rate a rate panel shows: one measured rate before any arm drops below FULL_PCT."""
    rates = sorted({x for pts in series.values() for x, _ in pts})
    drops = [x for pts in series.values() for x, y in pts if y < FULL_PCT]
    first_drop = min(drops) if drops else rates[-1]
    before = [x for x in rates if x < first_drop]
    return before[-1] if before else rates[0]


def rate_ticks(series: dict, x_from: float) -> list[float]:
    x_to = max(x for pts in series.values() for x, _ in pts)
    for step in RATE_STEPS:
        ticks = [
            step * i for i in range(int(-(-x_from // step)), int(x_to // step) + 1)
        ]
        if len(ticks) <= RATE_MAX_TICKS:
            return ticks
    raise ValueError(f"rates {x_from}-{x_to} need a tick step above {RATE_STEPS[-1]}")


def draw_panel(ax, series: dict, *, ticks: list[float], x_from: float, x_title: str, show_ytitle: bool,
               log: bool = False) -> None:  # fmt: skip
    from matplotlib.ticker import (
        FixedLocator,
        FuncFormatter,
        MultipleLocator,
        NullFormatter,
        NullLocator,
    )

    for arm, _, line, marker, fill in SERIES:
        pts = series[arm]
        ax.plot([x for x, _ in pts], [y for _, y in pts], marker=marker, markersize=MARKER_SIZE,
                markeredgewidth=MARKER_EDGE_WIDTH, zorder=4 if arm == OURS else 3,
                linewidth=OURS_LINE_WIDTH if arm == OURS else LINE_WIDTH, **st.line_kw(line, fill=fill))  # fmt: skip
    ax.axhline(
        SLO_FLOOR, color="black", linewidth=REF_LINE_WIDTH, linestyle=REF_DASH, zorder=2
    )
    x_to = max(x for pts in series.values() for x, _ in pts)
    if log:
        ax.set_xscale("log")
        pad = (x_to / x_from) ** X_PAD
        ax.set_xlim(x_from / pad, x_to * pad)
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        span = x_to - x_from
        ax.set_xlim(x_from - X_PAD * span, x_to + X_PAD * span)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(0, Y_TOP)
    ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR))
    ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", zorder=0, **st.GRID)
    ax.set_axisbelow(True)
    for axis, which, length in (
        ("both", "major", TICK_LEN_CM),
        ("y", "minor", TICK_MINOR_LEN_CM),
    ):
        ax.tick_params(axis=axis, which=which, pad=TICK_PAD_CM * st.PT_PER_CM, length=length * st.PT_PER_CM,
                       width=TICK_WIDTH_CM * st.PT_PER_CM)  # fmt: skip
    ax.tick_params(axis="y", labelleft=show_ytitle)
    if show_ytitle:
        ax.set_ylabel(
            Y_TITLE, fontsize=st.FONT_AXIS, labelpad=YLABEL_PAD_CM * st.PT_PER_CM
        )
    ax.set_xlabel(x_title, fontsize=st.FONT_AXIS, labelpad=XLABEL_PAD_CM * st.PT_PER_CM)
    st.frame_above(ax)


def callout(ax, series: dict) -> None:
    """Arrow on the 90% line from the best baseline's capacity to ForeSpark's, labelled with the gain."""
    ours = capacity(series[OURS])
    best = max(
        (
            c
            for arm, *_ in SERIES
            if arm != OURS
            if (c := capacity(series[arm])) is not None
        ),
        default=None,
    )
    if ours is None or best is None or ours <= best:
        return
    ax.annotate("", xy=(ours, SLO_FLOOR), xytext=(best, SLO_FLOOR), zorder=5,
                arrowprops={"arrowstyle": f"-|>,head_length={CALLOUT_HEAD / 10},head_width={CALLOUT_HEAD / 20}",
                            "color": st.FORESPARK_COLOR, "linewidth": CALLOUT_WIDTH, "shrinkA": 0, "shrinkB": 0})  # fmt: skip
    ax.text((best + ours) / 2, CALLOUT_LABEL_Y, f"+{100 * (ours / best - 1):.0f}%", ha="center", va="bottom",
            fontsize=st.FONT_NUM, color=st.FORESPARK_COLOR, zorder=5)  # fmt: skip


def slo_text(slos: list[tuple[float, float]]) -> str:
    """The columns' (TTFT ms, TPOT ms) SLOs in column order, a value shared by every column once."""

    def values(column: list[float], scale: float) -> str:
        shown = column[:1] if len(set(column)) == 1 else column
        return " / ".join(f"{v / scale:g}" for v in shown)

    return f"SLO: TTFT {values([t for t, _ in slos], 1000)} s, TPOT {values([p for _, p in slos], 1)} ms"


def main() -> None:
    args = st.parser(NAME, __doc__).parse_args()
    st.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = st.available(args.results, PHASE, [arm for arm, *_ in SERIES])
    data = {model: curves(args.results, model) for model, _ in columns}
    row_h = (
        TITLE_BAND_CM + PANEL_H_CM + XLABEL_BAND_CM + CAPTION_GAP_CM + CAPTION_BAND_CM
    )
    height = (
        LEGEND_BAND_CM
        + len(ROWS) * row_h
        + (len(ROWS) - 1) * ROW_GAP_CM
        + BOTTOM_MARGIN_CM
    )
    fig = plt.figure(figsize=st.figsize(height))
    width = st.W_SINGLE_CM
    panel_w = (
        width - LEFT_BAND_CM - RIGHT_MARGIN_CM - (len(columns) - 1) * COL_GAP_CM
    ) / len(columns)
    details = {
        "rate": slo_text([data[m][2] for m, _ in columns]),
        "scale": f"at {' / '.join(f'{data[m][1]:g}' for m, _ in columns)} req/s",
    }

    top = height - LEGEND_BAND_CM
    for panel, caption in ROWS:
        bottom = top - TITLE_BAND_CM - PANEL_H_CM
        for i, (model, title) in enumerate(columns):
            series = {arm: data[model][0][(panel, arm)] for arm, *_ in SERIES}
            if panel == "rate":
                start = rate_window(series)
                ticks, x_title, log = rate_ticks(series, start), RATE_TITLE, False
            else:
                start = min(x for pts in series.values() for x, _ in pts)
                end = max(x for pts in series.values() for x, _ in pts)
                ticks, x_title, log = (
                    [t for t in SCALE_TICKS if start <= t <= end],
                    SCALE_TITLE,
                    True,
                )
            left = LEFT_BAND_CM + i * (panel_w + COL_GAP_CM)
            ax = fig.add_axes(
                (left / width, bottom / height, panel_w / width, PANEL_H_CM / height)
            )
            draw_panel(
                ax,
                series,
                ticks=ticks,
                x_from=start,
                x_title=x_title,
                show_ytitle=i == 0,
                log=log,
            )
            if panel == "rate":
                callout(ax, series)
            ax.set_title(
                title, fontsize=st.FONT_PANEL_ID, pad=PANEL_ID_PAD_CM * st.PT_PER_CM
            )
        caption_y = bottom - XLABEL_BAND_CM - CAPTION_GAP_CM
        center = (LEFT_BAND_CM + width - RIGHT_MARGIN_CM) / 2
        fig.text(center / width, caption_y / height, f"{caption} ({details[panel]})", ha="center", va="top",
                 fontproperties=st.caption_font(st.FONT_CAPTION))  # fmt: skip
        top = caption_y - CAPTION_BAND_CM - ROW_GAP_CM

    handles = [Line2D([], [], marker=marker, markersize=MARKER_SIZE, markeredgewidth=MARKER_EDGE_WIDTH,
                      linewidth=OURS_LINE_WIDTH if arm == OURS else LINE_WIDTH, label=label,
                      **st.line_kw(line, fill=fill))
               for arm, label, line, marker, fill in SERIES]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - LEGEND_TOP_GAP_CM / height),
               ncol=len(SERIES), frameon=False, fontsize=st.FONT_LEGEND, handlelength=LEGEND_HANDLELENGTH,
               handletextpad=0.4, columnspacing=LEGEND_COLSPACING, borderpad=0.0)  # fmt: skip
    print(f"wrote {st.save(fig, args.out)}")


if __name__ == "__main__":
    main()
