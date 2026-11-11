# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Time per output token split into verify / draft / other, the benchmarked models on ShareGPT-Decode.

Single column, one row with a sub-plot per model, each grouping a few batch sizes (CONCURRENCY plus a high one:
TOP, else the highest every arm ran) with one bar per arm: AR, DSpark, ForeSpark. A bar is the arm's time per
output token divided by AR's at the same batch size (AR = 1, the dashed line), i.e. AR's throughput over the
arm's, stacked bottom to top in proportion to vLLM's step-timer segments (the cells' --step-timing): verify
(target forward + rejection sampling), draft (the drafter), other (input prep, post-processing, GPU idle). An arm
keeps its color and shows draft in a lighter shade; other is a thin white cap.

    .venv/bin/python plot/breakdown.py   ->   plot/plots/breakdown.pdf
"""

from __future__ import annotations

import math

import style as st

# ============================ Layout knobs (all centimetres) =================
PANEL_H_CM = 1.80  # height of each panel's box
LEFT_BAND_CM = 0.74  # left edge -> first panel: the y-axis title and its tick numbers
RIGHT_MARGIN_CM = 0.05  # last panel's right edge -> figure edge
COL_GAP_CM = 0.15  # between panels (they share the y-axis)
XLABEL_BAND_CM = 0.48  # below each panel: x tick numbers + the x-axis title
CAPTION_GAP_CM = 0.08  # x-axis title -> caption
CAPTION_BAND_CM = 0.26  # the (a)/(b)/(c) caption, hung from the band's top
BOTTOM_MARGIN_CM = 0.02  # caption -> figure bottom edge
LEGEND_TOP_GAP_CM = 0.02  # legend top <-> figure top edge
LEGEND_BAND_CM = (
    0.70  # figure top -> panel tops (legend included; clears the top y tick number)
)
LEGEND_NCOL = 3  # two rows: the verify shades, then other and the draft shades
LEGEND_COLSPACING = 1.0  # gap between legend columns (font-units)
LEGEND_LABELSPACING = 0.3  # gap between the legend rows (font-units)
LEGEND_HANDLELENGTH = 1.0  # swatch width (font-units)
BAR_WIDTH = 0.25  # one arm's bar, in groups (1 = the pitch between batch sizes)
REF_LINE_WIDTH = 0.6  # the dashed AR = 1 line
REF_DASH = (0, (3.0, 1.6))
YLABEL_PAD_CM = 0.05  # gap between the y-axis title and its tick numbers
XLABEL_PAD_CM = 0.02  # gap between the x tick numbers and the x-axis title
TICK_PAD_CM = 0.04  # gap between a tick number and the axis
TICK_LEN_CM = 0.07  # major tick length
TICK_MINOR_LEN_CM = 0.04  # minor tick length
TICK_WIDTH_CM = 0.02  # tick thickness
Y_MAJOR = 0.5  # labelled ticks (time per token, AR = 1)
Y_MINOR = 0.25
Y_HEADROOM = 1.05  # shared axis top = tallest bar x this, rounded up to Y_ROUND
Y_ROUND = 0.1
# ============================================================================

NAME = "breakdown"
Y_TITLE = "Norm. time/token"
X_TITLE = "Batch size"
PHASE = "sharegpt-decode"
CONCURRENCY = [1, 32, 64, 128]
TOP = {"qwen14": 256}  # a model's last group, where not the highest every arm ran
BASELINE = "nospec"
# Mean step-timer segments (ms) each stacked segment sums, bottom to top
SEGMENTS = {
    "verify": ("step_forward_ms", "step_sample_ms"),
    "draft": ("step_draft_ms",),
    "other": ("step_prep_ms", "step_post_ms", "step_idle_ms"),
}
_LIGHT, _MID, _DARK = st.grays(3)
# Bars in a group, ForeSpark last. (bench arm, legend label, verify color, draft color or None)
SERIES = [
    ("nospec", "AR", _LIGHT, None),
    ("dspark", "DSpark", _DARK, _MID),
    ("forespark", "ForeSpark", st.FORESPARK_COLOR, st.FORESPARK_LIGHT),
]


def bars(results, model: str) -> dict[tuple[str, int], dict[str, float]]:
    """{(arm, concurrency): {segment: time per token / AR's}}."""
    cells = {
        arm: {r["concurrency"]: r for r in st.rows(results, model, PHASE, arm)}
        for arm, *_ in SERIES
    }
    top = TOP.get(model) or max(
        set.intersection(*(set(by_c) for by_c in cells.values()))
    )
    out = {}
    for arm, by_c in cells.items():
        for c in sorted({*CONCURRENCY, top}):
            cell = by_c.get(c)
            if cell is None or not cell.get("step_timed"):
                raise SystemExit(
                    f"{model} {arm}: no timed ShareGPT-Decode cell at C = {c} (--step-timing)"
                )
            relative = (
                cells[BASELINE][c]["output_tokens_per_second"]
                / cell["output_tokens_per_second"]
            )
            parts = {
                seg: sum(cell[f] for f in fields) for seg, fields in SEGMENTS.items()
            }
            step = sum(parts.values())
            out[(arm, c)] = {seg: relative * ms / step for seg, ms in parts.items()}
    return out


def draw_panel(ax, groups: dict, y_top: float, *, show_ytitle: bool) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter

    concurrencies = sorted({c for _, c in groups})
    for g, c in enumerate(concurrencies):
        for i, (arm, _, verify, draft) in enumerate(SERIES):
            x = g + (i - (len(SERIES) - 1) / 2) * BAR_WIDTH
            bottom = 0.0
            for seg, color in zip(
                SEGMENTS, (verify, draft, st.REMAINDER_COLOR), strict=True
            ):
                if (height := groups[(arm, c)][seg]) > 0:
                    ax.bar(
                        x,
                        height,
                        width=BAR_WIDTH,
                        bottom=bottom,
                        zorder=3,
                        **st.bar_kw(color),
                    )
                    bottom += height
    ax.axhline(
        1.0, color="black", linewidth=REF_LINE_WIDTH, linestyle=REF_DASH, zorder=2
    )
    ax.set_xlim(-0.5, len(concurrencies) - 0.5)
    ax.set_xticks(range(len(concurrencies)), [f"{c}" for c in concurrencies])
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", zorder=0, **st.GRID)
    ax.set_axisbelow(True)
    pad, width = TICK_PAD_CM * st.PT_PER_CM, TICK_WIDTH_CM * st.PT_PER_CM
    ax.tick_params(axis="x", length=0, pad=pad)
    ax.tick_params(axis="y", which="major", length=TICK_LEN_CM * st.PT_PER_CM, pad=pad, width=width,
                   labelleft=show_ytitle)  # fmt: skip
    ax.tick_params(
        axis="y", which="minor", length=TICK_MINOR_LEN_CM * st.PT_PER_CM, width=width
    )
    if show_ytitle:
        ax.set_ylabel(
            Y_TITLE, fontsize=st.FONT_AXIS, labelpad=YLABEL_PAD_CM * st.PT_PER_CM
        )
    ax.set_xlabel(X_TITLE, fontsize=st.FONT_AXIS, labelpad=XLABEL_PAD_CM * st.PT_PER_CM)
    st.frame_above(ax)


def legend_entries() -> list[tuple[str, str]]:
    """(label, color) in reading order over LEGEND_NCOL columns: every arm's verify shade, then other and
    the draft shades (each under its arm's verify swatch)."""
    verify = [
        (label if draft is None else f"{label}: verify", color)
        for _, label, color, draft in SERIES
    ]
    drafts = [
        (f"{label}: draft", draft) for _, label, _, draft in SERIES if draft is not None
    ]
    return verify + [("Other", st.REMAINDER_COLOR)] + drafts


def main() -> None:
    args = st.parser(NAME, __doc__).parse_args()
    st.apply()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    panels = st.available(args.results, PHASE, [arm for arm, *_ in SERIES])
    data = {model: bars(args.results, model) for model, _ in panels}
    tallest = max(
        sum(parts.values()) for groups in data.values() for parts in groups.values()
    )
    y_top = round(math.ceil(tallest * Y_HEADROOM / Y_ROUND) * Y_ROUND, 6)
    height = (
        LEGEND_BAND_CM
        + PANEL_H_CM
        + XLABEL_BAND_CM
        + CAPTION_GAP_CM
        + CAPTION_BAND_CM
        + BOTTOM_MARGIN_CM
    )
    fig = plt.figure(figsize=st.figsize(height))
    width = st.W_SINGLE_CM
    panel_w = (
        width - LEFT_BAND_CM - RIGHT_MARGIN_CM - (len(panels) - 1) * COL_GAP_CM
    ) / len(panels)
    bottom = height - LEGEND_BAND_CM - PANEL_H_CM
    caption_y = bottom - XLABEL_BAND_CM - CAPTION_GAP_CM
    for i, (model, label) in enumerate(panels):
        left = LEFT_BAND_CM + i * (panel_w + COL_GAP_CM)
        ax = fig.add_axes(
            (left / width, bottom / height, panel_w / width, PANEL_H_CM / height)
        )
        draw_panel(ax, data[model], y_top, show_ytitle=i == 0)
        fig.text((left + panel_w / 2) / width, caption_y / height, f"({'abc'[i]}) {label}", ha="center",
                 va="top", fontproperties=st.caption_font(st.FONT_CAPTION))  # fmt: skip

    handles = [
        Patch(facecolor=color, label=label, **st.BAR)
        for label, color in legend_entries()
    ]
    fig.legend(handles=st.row_major(handles, LEGEND_NCOL), loc="upper center",
               bbox_to_anchor=(0.5, 1 - LEGEND_TOP_GAP_CM / height), ncol=LEGEND_NCOL, frameon=False,
               fontsize=st.FONT_LEGEND, handlelength=LEGEND_HANDLELENGTH, handleheight=0.7, handletextpad=0.4,
               columnspacing=LEGEND_COLSPACING, labelspacing=LEGEND_LABELSPACING, borderpad=0.0)  # fmt: skip
    print(f"wrote {st.save(fig, args.out)}")


if __name__ == "__main__":
    main()
