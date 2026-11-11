# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Throughput against per-user speed on ShareGPT, two workloads x the benchmarked models.

Single column, two rows (sub-plots): (a) ShareGPT-Decode (decode-only spans, scripts/bench/decode.sh) and (b)
ShareGPT-E2E (prefill and decode sharing batches, scripts/bench/e2e.sh). Each row holds one Pareto panel per model,
each on its own y-axis. One line per arm through its concurrency points (C = 1 at the right, the highest C at the
top): x = per-user tokens/s (1000 / median TPOT, log), y = total output tokens/s. Up and to the right is better. An
arrow marks ForeSpark's largest gain over the best other line at the same per-user speed.

    .venv/bin/python plot/pareto.py   ->   plot/plots/pareto.pdf
"""

from __future__ import annotations

import math

import style as st

# ============================ Layout knobs (all centimetres) =================
PANEL_H_CM = 1.45  # height of each panel's box
LEFT_BAND_CM = 0.80  # left edge -> panels: the y-axis title and its tick numbers
RIGHT_MARGIN_CM = (
    0.18  # right panel edge -> figure edge (room for a last x tick number at the edge)
)
COL_GAP_CM = 0.56  # between a row's panels: the next panel's y tick numbers
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
PANEL_ID_PAD_CM = 0.04  # gap between a panel and its identifier above it
YLABEL_PAD_CM = 0.05  # gap between the y-axis title and its tick numbers
XLABEL_PAD_CM = 0.02  # gap between the x tick numbers and the x-axis title
TICK_PAD_CM = 0.04  # gap between a tick number and the axis
TICK_LEN_CM = 0.07  # major tick length
TICK_MINOR_LEN_CM = 0.04  # minor tick length
TICK_WIDTH_CM = 0.02  # tick thickness
X_TICKS = [
    25,
    50,
    100,
    200,
    400,
    800,
]  # per-user tokens/s: every other one where they fall inside the axis
X_MARGIN = 1.15  # x range = data range widened by this factor at each end (log)
Y_HEADROOM = 1.08  # y ceiling = highest point x this, rounded up to half its step
GAIN_ROOM_TRIES = 3  # a panel whose gain label finds no clear spot is redrawn with half a step more, up to this often
Y_STEPS = [
    2500,
    5000,
    10000,
    20000,
    25000,
    50000,
]  # labelled y-tick steps (tokens/s, printed "5k"): a panel
Y_MAX_INTERVALS = 5  # takes the smallest step giving it at most this many intervals
# ============================================================================

NAME = "pareto"
Y_TITLE = "Total tok/s"
X_TITLE = "User interactivity"
ROWS = [
    ("sharegpt-decode", "(a) ShareGPT-Decode"),
    ("sharegpt", "(b) ShareGPT-E2E (Prefill+Decode)"),
]
OURS = "forespark"
_GRAYS = st.grays(2)
# Series, ForeSpark last. (bench arm, legend label, line color, marker, marker fill)
SERIES = [
    ("nospec", "AR", st.BASELINE_LINE, st.MARKERS[0], _GRAYS[0]),
    ("dspark", "DSpark", st.BASELINE_LINE, st.MARKERS[1], _GRAYS[1]),
    (OURS, "ForeSpark", st.FORESPARK_COLOR, st.MARKERS[0], st.FORESPARK_COLOR),
]


def points(results, model: str, phase: str) -> dict[str, list[tuple[float, float]]]:
    """{arm: [(per-user tok/s, tok/s), ...] by concurrency}."""
    return {
        arm: [(1000.0 / r["median_tpot_ms"], r["output_tokens_per_second"])
              for r in sorted(st.rows(results, model, phase, arm), key=lambda r: r["concurrency"])]
        for arm, *_ in SERIES
    }  # fmt: skip


def y_axis(top: float, extra: int = 0) -> tuple[float, float]:
    """(ceiling, step) for a panel whose highest point is `top`, with `extra` half steps more."""
    for step in Y_STEPS:
        if math.ceil(top * Y_HEADROOM / step) <= Y_MAX_INTERVALS:
            return (math.ceil(top * Y_HEADROOM / (step / 2)) + extra) * step / 2, step
    raise ValueError(f"{top:.0f} tok/s needs a y step above {Y_STEPS[-1]}")


def draw_panel(ax, series: dict, *, show_ytitle: bool, extra: int = 0) -> None:
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
    xs = [x for pts in series.values() for x, _ in pts]
    lo, hi = min(xs) / X_MARGIN, max(xs) * X_MARGIN
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    inside = [t for t in X_TICKS if lo <= t <= hi]
    ax.xaxis.set_major_locator(
        FixedLocator(inside[len(inside) % 2 == 0 :: 2])
    )  # narrow panels: every other
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    y_top, y_step = y_axis(max(y for pts in series.values() for _, y in pts), extra)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k")
    )
    ax.yaxis.set_minor_locator(MultipleLocator(y_step / 2))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", zorder=0, **st.GRID)
    ax.set_axisbelow(True)
    for axis, which, length in (
        ("both", "major", TICK_LEN_CM),
        ("y", "minor", TICK_MINOR_LEN_CM),
    ):
        ax.tick_params(axis=axis, which=which, pad=TICK_PAD_CM * st.PT_PER_CM, length=length * st.PT_PER_CM,
                       width=TICK_WIDTH_CM * st.PT_PER_CM)  # fmt: skip
    if show_ytitle:
        ax.set_ylabel(
            Y_TITLE, fontsize=st.FONT_AXIS, labelpad=YLABEL_PAD_CM * st.PT_PER_CM
        )
    ax.set_xlabel(X_TITLE, fontsize=st.FONT_AXIS, labelpad=XLABEL_PAD_CM * st.PT_PER_CM)
    st.frame_above(ax)


def main() -> None:
    args = st.parser(NAME, __doc__).parse_args()
    st.apply()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    arms = [arm for arm, *_ in SERIES]
    columns = [m for m in st.MODELS if m in st.available(args.results, ROWS[0][0], arms)
               and m in st.available(args.results, ROWS[1][0], arms)]  # fmt: skip
    if not columns:
        raise SystemExit("no model has both workloads for every arm")
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

    top = height - LEGEND_BAND_CM
    for phase, caption in ROWS:
        bottom = top - TITLE_BAND_CM - PANEL_H_CM
        for i, (model, title) in enumerate(columns):
            series = points(args.results, model, phase)
            left = LEFT_BAND_CM + i * (panel_w + COL_GAP_CM)
            for extra in range(GAIN_ROOM_TRIES):
                ax = fig.add_axes(
                    (
                        left / width,
                        bottom / height,
                        panel_w / width,
                        PANEL_H_CM / height,
                    )
                )
                draw_panel(ax, series, show_ytitle=i == 0, extra=extra)
                if st.annotate_gain(ax, series, OURS) or extra == GAIN_ROOM_TRIES - 1:
                    break
                ax.remove()
            ax.set_title(
                title, fontsize=st.FONT_PANEL_ID, pad=PANEL_ID_PAD_CM * st.PT_PER_CM
            )
        caption_y = bottom - XLABEL_BAND_CM - CAPTION_GAP_CM
        center = (LEFT_BAND_CM + width - RIGHT_MARGIN_CM) / 2
        fig.text(center / width, caption_y / height, caption, ha="center", va="top",
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
