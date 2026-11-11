# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The paper's figure style, and the benchmark results the figures read.

* ForeSpark is locked: dark red, bars always solid, always last; its draft
  segments take a lighter tint of the same red.
* Baselines are drawn from a 3-shade grayscale ramp, assigned in the order a
  figure gives; a baseline's line is black, its markers carry the gray.
* Figures are drawn at the paper's column width divided by LATEX_SCALE and
  scaled up to \\columnwidth in LaTeX, so every font reads 1.1x its size here.
* Tahoma throughout, Times New Roman for the lettered captions, read from
  plot/fonts/tahoma.ttf and plot/fonts/times.ttf if present (commercial fonts,
  not shipped); DejaVu otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_CACHE = ROOT / ".cache" / "matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FONTS = ROOT / "fonts"

# The benchmark targets, in figure order: (results tree, label)
MODELS = [
    ("qwen14", "Qwen3-14B"),
    ("glm53flash", "GLM-5.3-Flash"),
    ("glm53", "GLM-5.3"),
]

# --- Fonts -----------------------------------------------------------------
FONT_FAMILY = "Tahoma"
FONT_CAPTION_FAMILY = "Times New Roman"
FALLBACK = {"Tahoma": "DejaVu Sans", "Times New Roman": "DejaVu Serif"}

FONT_CAPTION = 7  # sub-plot caption (Times New Roman)
FONT_PANEL_ID = 6  # panel identifier above a panel
FONT_AXIS = 6  # axis titles
FONT_LEGEND = 6  # legend labels
FONT_NUM = 5.5  # tick labels

# --- Figure widths (exact) ---------------------------------------------------
CM = 1 / 2.54  # inches per centimetre
PT_PER_CM = 72 / 2.54  # points per centimetre (for pads, which want points)
COLUMN_CM = 8.25  # the paper's column width
LATEX_SCALE = 1.1
W_SINGLE_CM = COLUMN_CM / LATEX_SCALE  # 7.5 cm
DPI = 400  # raster outputs (--out NAME.png): 1200 px across

# --- Colors ------------------------------------------------------------------
FORESPARK_COLOR = "#af2a2c"  # dark red; ForeSpark's bars are always solid
FORESPARK_LIGHT = "#f09799"  # the same red, lighter: ForeSpark's draft segments
FORESPARK_PALE = "#f8cbcc"  # lighter still: a third segment
GRAYS = ["#f2f2f2", "#b3b3b3", "#7f7f7f"]  # baselines, light -> dark
BASELINE_LINE = "black"
MARKERS = ["o", "s", "^", "D", "v", "P"]
REMAINDER_COLOR = "white"  # the small catch-all segment atop a stacked bar

SPINE_WIDTH = 0.5  # panel outline
BAR = {"edgecolor": "black", "linewidth": 0.4}  # every bar
GRID = {"color": "#d9d9d9", "linewidth": 0.3, "linestyle": "-"}


def grays(n: int) -> list[str]:
    """`n` baseline grays spread evenly across the ramp (endpoints included)."""
    if n <= 1:
        return GRAYS[:n]
    last = len(GRAYS) - 1
    return [GRAYS[round(i * last / (n - 1))] for i in range(n)]


def line_kw(color: str, fill: str | None = None) -> dict:
    """ax.plot(...) keywords: a solid line with black-edged markers filled with `fill`."""
    return {"color": color, "linestyle": "-", "markerfacecolor": fill or color,
            "markeredgecolor": BAR["edgecolor"]}  # fmt: skip


def bar_kw(color: str, hatch: str = "") -> dict:
    return {"color": color, "hatch": hatch, **BAR}


def _family(name: str) -> str:
    """`name` if matplotlib has it (system fonts and plot/fonts), else its DejaVu fallback."""
    from matplotlib import font_manager

    return (
        name
        if name in {f.name for f in font_manager.fontManager.ttflist}
        else FALLBACK[name]
    )


def apply() -> None:
    """Install the paper's rcParams. Call once per plot script."""
    from matplotlib import font_manager

    for ttf in FONTS.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    family = _family(FONT_FAMILY)
    plt.rcParams.update({
        "font.family": family, "font.size": FONT_AXIS, "axes.titlesize": FONT_CAPTION,
        "axes.labelsize": FONT_AXIS, "legend.fontsize": FONT_LEGEND, "xtick.labelsize": FONT_NUM,
        "ytick.labelsize": FONT_NUM, "axes.linewidth": SPINE_WIDTH, "mathtext.fontset": "custom",
        "mathtext.rm": family, "mathtext.it": f"{family}:italic", "mathtext.bf": family,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })  # fmt: skip


def caption_font(size: float = FONT_CAPTION):
    from matplotlib import font_manager

    return font_manager.FontProperties(family=_family(FONT_CAPTION_FAMILY), size=size)


def row_major(handles, ncol):
    """Legend handles reordered so they display left to right, row by row
    (matplotlib fills legend columns top to bottom)."""
    from matplotlib.patches import Patch

    nrow = math.ceil(len(handles) / ncol)
    blank = Patch(facecolor="none", edgecolor="none", label="")
    padded = list(handles) + [blank] * (nrow * ncol - len(handles))
    ordered = [None] * (nrow * ncol)
    for r in range(nrow):
        for c in range(ncol):
            ordered[c * nrow + r] = padded[r * ncol + c]
    return ordered


def frame_above(ax) -> None:
    """Draw the panel outline on top of the data (bars default above the spines)."""
    for spine in ax.spines.values():
        spine.set_zorder(1000)


def figsize(height_cm: float) -> tuple[float, float]:
    return (W_SINGLE_CM * CM, height_cm * CM)


def save(fig, out: Path) -> Path:
    """Write the figure at its exact width (no bbox_inches='tight', which would change it)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    return out


# --- Pareto gain at a fixed per-user speed ------------------------------------
GAIN_ARROW_WIDTH = 0.7  # arrow line width (points)
GAIN_ARROW_HEAD = 2.5  # arrow head size (points)
GAIN_LABEL_PAD = 1.5  # arrow -> its label (points)
GAIN_SAMPLES = 400  # per-user speeds searched, log-spaced over ForeSpark's line
# Compared only where ForeSpark and the best baseline each deliver at least this share of their peak tok/s: a
# lightly loaded line turns near-vertical (AR's ends at its single-request speed), where a ratio at one per-user
# speed hinges on interpolation and grows without bound.
GAIN_FLOOR = 0.5
GAIN_LABEL_CLEARANCE = (
    0.75  # least gap between the label and any line, marker, text or the frame (points)
)


def pareto_at(points: list[tuple[float, float]], x: float) -> float | None:
    """The most total tok/s a Pareto line (points in concurrency order) delivers at per-user tok/s
    >= x, linear in log x between its points (as drawn on a log axis); None if it never reaches x."""
    best = [y for px, y in points if px >= x]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if min(x0, x1) <= x <= max(x0, x1) and x0 != x1:
            best.append(y0 + (y1 - y0) * math.log(x / x0) / math.log(x1 / x0))
    return max(best, default=None)


def max_gain(
    ours: list[tuple[float, float]], baselines: dict[str, list[tuple[float, float]]]
):
    """(per-user tok/s, best baseline there, its tok/s, ours) where ours leads the best baseline by the
    largest factor at the same per-user speed (see GAIN_FLOOR); None if no baseline qualifies anywhere."""
    lo, hi = min(x for x, _ in ours), max(x for x, _ in ours)
    peak = max(y for _, y in ours)
    peaks = {name: max(y for _, y in pts) for name, pts in baselines.items()}
    best = None
    for i in range(GAIN_SAMPLES + 1):
        x = lo * (hi / lo) ** (i / GAIN_SAMPLES)
        rivals = {
            name: y
            for name, pts in baselines.items()
            if (y := pareto_at(pts, x)) is not None
        }
        mine = pareto_at(ours, x)
        if not rivals or mine is None or mine < GAIN_FLOOR * peak:
            continue
        name = max(rivals, key=rivals.get)
        if rivals[name] < GAIN_FLOOR * peaks[name]:
            continue
        if best is None or mine / rivals[name] > best[3] / best[2]:
            best = (x, name, rivals[name], mine)
    return best


# Label spots, nearest first: above the head, beside it, beside the shaft's middle; toward higher per-user speed
# before lower, as ForeSpark's line falls away that way.
_GAIN_SPOTS = [
    (anchor, side * dist, dy, "left" if side > 0 else "right", va)
    for dist in (1, 2, 4, 6, 8)
    for anchor, dy, va in (
        ("head", 1, "bottom"),
        ("head", 0, "center"),
        ("mid", 0, "center"),
    )
    for side in (1, -1)
]


def _segment_hits(box, p, q) -> bool:
    """Whether the segment p-q (display coordinates) touches the box."""
    (x0, y0), (x1, y1) = p, q
    t0, t1 = 0.0, 1.0
    for d, lo, hi, start in (
        (x1 - x0, box.x0, box.x1, x0),
        (y1 - y0, box.y0, box.y1, y0),
    ):
        if d == 0:
            if not lo <= start <= hi:
                return False
            continue
        a, b = sorted(((lo - start) / d, (hi - start) / d))
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


def _clear(ax, text, renderer) -> bool:
    """Whether the text sits inside the panel, off every plotted line, marker and other text."""
    box = text.get_window_extent(renderer).padded(
        GAIN_LABEL_CLEARANCE * ax.figure.dpi / 72
    )
    if not (
        ax.bbox.x0 <= box.x0
        and box.x1 <= ax.bbox.x1
        and ax.bbox.y0 <= box.y0
        and box.y1 <= ax.bbox.y1
    ):
        return False
    for line in ax.get_lines():
        xy = line.get_transform().transform(line.get_xydata())
        if line.get_linestyle() not in ("None", "", " ") and any(
            _segment_hits(box, p, q) for p, q in zip(xy, xy[1:])
        ):
            return False
        r = (
            line.get_markersize() * ax.figure.dpi / 72 / 2
            if line.get_marker() not in (None, "", "None")
            else 0
        )
        if r and any(
            box.x0 - r <= mx <= box.x1 + r and box.y0 - r <= my <= box.y1 + r
            for mx, my in xy
        ):
            return False
    return not any(other is not text and other.get_text() and box.overlaps(other.get_window_extent(renderer))
                   for other in ax.texts)  # fmt: skip


def gain_arrow(ax, x: float, base: float, ours: float) -> bool:
    """A vertical arrow at per-user speed x from the best baseline up to ForeSpark, labelled with the factor
    at the first of _GAIN_SPOTS clear of the panel's lines, markers and texts. Returns whether a spot was
    clear (False: the label took the first spot regardless)."""
    if ours <= base:
        return True
    head = f"-|>,head_length={GAIN_ARROW_HEAD / 10},head_width={GAIN_ARROW_HEAD / 20}"
    ax.annotate("", xy=(x, ours), xytext=(x, base), zorder=5,
                arrowprops={"arrowstyle": head, "color": FORESPARK_COLOR, "linewidth": GAIN_ARROW_WIDTH,
                            "shrinkA": 0, "shrinkB": 0})  # fmt: skip
    renderer = ax.figure.canvas.get_renderer()
    for anchor, dx, dy, ha, va in _GAIN_SPOTS:
        label = ax.annotate(f"{ours / base:.2f}×", xy=(x, ours if anchor == "head" else (base + ours) / 2),
                            xytext=(dx * GAIN_LABEL_PAD, dy * GAIN_LABEL_PAD), textcoords="offset points", ha=ha,
                            va=va, fontsize=FONT_NUM, color=FORESPARK_COLOR, zorder=5)  # fmt: skip
        if _clear(ax, label, renderer):
            return True
        label.remove()
    anchor, dx, dy, ha, va = _GAIN_SPOTS[0]
    ax.annotate(f"{ours / base:.2f}×", xy=(x, ours), xytext=(dx * GAIN_LABEL_PAD, dy * GAIN_LABEL_PAD),
                textcoords="offset points", ha=ha, va=va, fontsize=FONT_NUM, color=FORESPARK_COLOR, zorder=5)  # fmt: skip
    return False


def annotate_gain(
    ax, series: dict[str, list[tuple[float, float]]], ours: str = "forespark"
) -> bool:
    """gain_arrow at max_gain in a log-x Pareto panel of {arm: [(per-user tok/s, tok/s), ...]}."""
    lines = dict(series)
    gain = max_gain(lines.pop(ours), lines)
    if gain is None:
        return True
    x, _, base, mine = gain
    return gain_arrow(ax, x, base, mine)


# --- Results -----------------------------------------------------------------


def parser(name: str, doc: str) -> argparse.ArgumentParser:
    """A figure script's arguments: where the results trees are, and where the PDF goes."""
    from forespark.paths import RESULTS

    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="the results root (default: configs/paths.yaml's)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "plots" / f"{name}.pdf",
        help="the figure file; its extension picks the format (.pdf, .png)",
    )
    return p


def rows(results: Path, model: str, phase: str, arm: str) -> list[dict]:
    """One bench cell's result rows (scripts/bench), or [] if it has not run."""
    path = results / model / "bench" / f"{phase}-{arm}.json"
    return json.loads(path.read_text())["results"] if path.is_file() else []


def available(results: Path, phase: str, arms: list[str]) -> list[tuple[str, str]]:
    """The MODELS whose results tree holds `phase` for every arm, reporting the others."""
    found = []
    for model, label in MODELS:
        missing = [arm for arm in arms if not rows(results, model, phase, arm)]
        if missing:
            print(
                f"{label}: no {phase} results for {', '.join(missing)} under {results / model / 'bench'}"
            )
        else:
            found.append((model, label))
    if not found:
        raise SystemExit(
            f"no model has {phase} results for every arm: run scripts/bench first"
        )
    return found
