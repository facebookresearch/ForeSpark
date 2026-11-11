# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Draft tokens per request-step split into accepted / rejected / not verified, the benchmarked models on
ShareGPT-Decode: the token counts behind plot/breakdown.py.

Single column, one row with a sub-plot per model, grouping breakdown.py's batch sizes with one bar per drafter:
DSpark, ForeSpark. A bar is the drafter's draft tokens per request-step (the anchor excluded), stacked bottom to
top: accepted (the acceptance length less the bonus token), verified but rejected, drafted but never verified.
DSpark drafts its full block of K and verifies an adaptive budget; Fore-D plans both of ForeSpark's. A drafter
keeps its color in three shades, darkest for accepted. Every count is a window delta of vLLM's spec-decode
counters, as the cells record them.

    .venv/bin/python plot/acceptance.py   ->   plot/plots/acceptance.pdf
"""

from __future__ import annotations

import style as st
from breakdown import CONCURRENCY, PHASE, TOP

# ============================ Layout knobs (all centimetres) =================
PANEL_H_CM = 1.80  # height of each panel's box
LEFT_BAND_CM = 0.74  # left edge -> first panel: the y-axis title and its tick numbers
RIGHT_MARGIN_CM = 0.05  # last panel's right edge -> figure edge
COL_GAP_CM = 0.15  # between panels (they share the y-axis)
XLABEL_BAND_CM = 0.48  # below each panel: x tick numbers + the x-axis title
CAPTION_GAP_CM = 0.08  # x-axis title -> caption
CAPTION_BAND_CM = 0.26  # the (a)/(b)/(c) caption, hung from the band's top
BOTTOM_MARGIN_CM = 0.02  # caption -> figure bottom edge
LEGEND_TOP_GAP_CM = 0.02  # first legend's top <-> figure top edge
LEGEND_ROW_CM = 0.30  # second legend's top below the first's
LEGEND_BAND_CM = (
    0.70  # figure top -> panel tops (legends included; clears the top y tick number)
)
LEGEND_COLSPACING = 1.0  # gap between legend entries (font-units)
LEGEND_HANDLELENGTH = 1.0  # one swatch (font-units); a segment entry pairs two
BAR_WIDTH = 0.32  # one drafter's bar, in groups (1 = the pitch between batch sizes)
YLABEL_PAD_CM = 0.05  # gap between the y-axis title and its tick numbers
XLABEL_PAD_CM = 0.02  # gap between the x tick numbers and the x-axis title
TICK_PAD_CM = 0.04  # gap between a tick number and the axis
TICK_LEN_CM = 0.07  # major tick length
TICK_MINOR_LEN_CM = 0.04  # minor tick length
TICK_WIDTH_CM = 0.02  # tick thickness
Y_TOP = 15  # the full draft block, K
Y_MAJOR = 5  # labelled ticks (tokens per request-step)
Y_MINOR = 2.5
# ============================================================================

NAME = "acceptance"
Y_TITLE = "Tokens/request"
X_TITLE = "Batch size"
SEGMENTS = ["Accepted", "Rejected", "Not verified"]  # stacked bottom to top
_LIGHT, _MID, _DARK = st.grays(3)
_GRAY_SHADES = (_DARK, _MID, _LIGHT)
_RED_SHADES = (st.FORESPARK_COLOR, st.FORESPARK_LIGHT, st.FORESPARK_PALE)
# Bars in a group, ForeSpark last. (bench arm, legend label, segment colors bottom to top)
SERIES = [
    ("dspark", "DSpark", _GRAY_SHADES),
    ("forespark", "ForeSpark", _RED_SHADES),
]


def bars(results, model: str) -> dict[tuple[str, int], list[float]]:
    """{(arm, concurrency): [accepted, rejected, not verified] draft tokens per request-step}."""
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
            if cell is None or not cell.get("draft_steps"):
                raise SystemExit(
                    f"{model} {arm}: no ShareGPT-Decode cell with spec-decode counters at C = {c}"
                )
            drafted, verified = cell["draft_len_observed"], cell["verify_len_observed"]
            accepted = cell["acceptance_length"] - 1.0
            out[(arm, c)] = [accepted, verified - accepted, drafted - verified]
    return out


def draw_panel(ax, groups: dict, *, show_ytitle: bool) -> None:
    from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter

    concurrencies = sorted({c for _, c in groups})
    for g, c in enumerate(concurrencies):
        for i, (arm, _, colors) in enumerate(SERIES):
            x = g + (i - (len(SERIES) - 1) / 2) * BAR_WIDTH
            bottom = 0.0
            for tokens, color in zip(groups[(arm, c)], colors, strict=True):
                if tokens > 0:
                    ax.bar(
                        x,
                        tokens,
                        width=BAR_WIDTH,
                        bottom=bottom,
                        zorder=3,
                        **st.bar_kw(color),
                    )
                    bottom += tokens
    ax.set_xlim(-0.5, len(concurrencies) - 0.5)
    ax.set_xticks(range(len(concurrencies)), [f"{c}" for c in concurrencies])
    ax.set_ylim(0, Y_TOP)
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


def main() -> None:
    args = st.parser(NAME, __doc__).parse_args()
    st.apply()
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch

    panels = st.available(args.results, PHASE, [arm for arm, *_ in SERIES])
    data = {model: bars(args.results, model) for model, _ in panels}
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
        draw_panel(ax, data[model], show_ytitle=i == 0)
        fig.text((left + panel_w / 2) / width, caption_y / height, f"({'abc'[i]}) {label}", ha="center",
                 va="top", fontproperties=st.caption_font(st.FONT_CAPTION))  # fmt: skip

    # Drafters by their darkest shade; each segment as its gray | red pair.
    drafters = [Patch(facecolor=colors[0], **st.BAR) for _, _, colors in SERIES]
    shades = [(Patch(facecolor=gray, **st.BAR), Patch(facecolor=red, **st.BAR))
              for gray, red in zip(_GRAY_SHADES, _RED_SHADES, strict=True)]  # fmt: skip
    common = {"loc": "upper center", "frameon": False, "fontsize": st.FONT_LEGEND, "handletextpad": 0.4,
              "columnspacing": LEGEND_COLSPACING, "borderpad": 0.0, "handleheight": 0.7,
              "handler_map": {tuple: HandlerTuple(ndivide=None, pad=0.0)}}  # fmt: skip
    fig.legend(drafters, [label for _, label, _ in SERIES], ncol=len(drafters),
               handlelength=LEGEND_HANDLELENGTH, bbox_to_anchor=(0.5, 1 - LEGEND_TOP_GAP_CM / height),
               **common)  # fmt: skip
    fig.legend(shades, SEGMENTS, ncol=len(shades), handlelength=2 * LEGEND_HANDLELENGTH,
               bbox_to_anchor=(0.5, 1 - (LEGEND_TOP_GAP_CM + LEGEND_ROW_CM) / height), **common)  # fmt: skip
    print(f"wrote {st.save(fig, args.out)}")


if __name__ == "__main__":
    main()
