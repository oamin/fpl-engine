"""One figure of the decisions the published climb makes in a gameweek.

The boxes follow ``run_ft_season`` with an empty chip map. A chip is named
in the footnote because that map is what plays one, and the historical climb
does not pass one.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "data" / "plots"

NAVY = "#1b3a4b"
INK = "#1c2833"
BLUE = "#e8f1f8"
BLUE_EDGE = "#2c5f8a"
SAND = "#fff6df"
SAND_EDGE = "#8a6a2f"
GREEN = "#e5f4ec"
GREEN_EDGE = "#2e7d4f"
GREY = "#f4f5f6"
GREY_EDGE = "#6b7280"
SKIP = "#f8ecec"
SKIP_EDGE = "#8f3d3d"


def _box(ax, x, y, w, h, text, face, edge, *, size=8.0, weight="regular"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=2,
        )
    )
    ax.text(
        x, y, text, ha="center", va="center", fontsize=size, color=INK,
        fontweight=weight, linespacing=1.25, zorder=3,
    )


def _diamond(ax, x, y, w, h, text):
    ax.add_patch(
        Polygon(
            [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
            closed=True, facecolor=SAND, edgecolor=SAND_EDGE, linewidth=1.0, zorder=2,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=7.7, color=INK, linespacing=1.15, zorder=3)


def _arrow(ax, p1, p2, text="", *, dx=0.0, dy=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
            color=NAVY, shrinkA=0, shrinkB=0, zorder=1,
        )
    )
    if text:
        ax.text(
            (p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, text,
            ha="center", va="center", fontsize=7.0, color=NAVY, zorder=3,
        )


def draw(path: Path | None = None) -> Path:
    """Write the gameweek decision figure and return its path."""
    out = path or (PLOTS / "gw_decision_flow.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.4, 16.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.62, 0.978, "What the model decides in one gameweek",
        ha="center", va="top", fontsize=15, color=NAVY, fontweight="bold",
    )
    ax.text(
        0.62, 0.956,
        "Published climb   ·   empty chip map   ·   score_xp   ·   same-position transfers",
        ha="center", va="top", fontsize=8.0, color="#4b5563",
    )

    x, w = 0.64, 0.50
    left, lw = 0.16, 0.26

    steps = {
        "carry": 0.915,
        "sheet": 0.855,
        "pool": 0.785,
        "stub": 0.725,
        "fix": 0.668,
        "first": 0.600,
        "value": 0.520,
        "search": 0.445,
        "legal": 0.378,
        "margin": 0.305,
        "bar": 0.242,
        "xi": 0.185,
        "score": 0.112,
        "bank": 0.048,
    }

    _box(ax, x, steps["carry"], w, 0.036,
         "Carry in the 15, the money in the bank, and the free-transfer count",
         BLUE, BLUE_EDGE, size=8.2, weight="bold")
    _diamond(ax, x, steps["sheet"], 0.40, 0.052, "Any club on\nthis week's sheet?")
    _arrow(ax, (x, steps["carry"] - 0.018), (x, steps["sheet"] + 0.026))
    _box(ax, left, steps["sheet"], lw, 0.048,
         "Skip the week.\nNo sale, no lineup.\nFree transfers stay put.",
         SKIP, SKIP_EDGE, size=7.2)
    _arrow(ax, (x - 0.20, steps["sheet"]), (left + lw / 2, steps["sheet"]), "No", dy=0.012)

    _box(ax, x, steps["pool"], w, 0.050,
         "Build the pool.\nBuyable if the last three appearances average at least 45 minutes.\nAnyone already owned stays in, even below that bar.",
         BLUE, BLUE_EDGE, size=7.5)
    _arrow(ax, (x, steps["sheet"] - 0.026), (x, steps["pool"] + 0.025), "Yes", dx=0.03)
    _box(ax, x, steps["stub"], w, 0.040,
         "An owned player missing from the sheet takes his latest score\nfrom before this deadline. His points and minutes this week are 0.",
         GREY, GREY_EDGE, size=7.3)
    _arrow(ax, (x, steps["pool"] - 0.025), (x, steps["stub"] + 0.020))
    _box(ax, x, steps["fix"], w, 0.040,
         "A club with a row keeps its score.\nA club with no row is ranked at 0, this week and on the horizon.",
         BLUE, BLUE_EDGE, size=7.5)
    _arrow(ax, (x, steps["stub"] - 0.020), (x, steps["fix"] + 0.020))

    _diamond(ax, x, steps["first"], 0.36, 0.050, "First week\nof the climb?")
    _arrow(ax, (x, steps["fix"] - 0.020), (x, steps["first"] + 0.025))
    _box(ax, left, steps["first"], lw, 0.058,
         "Build a free 15.\nMaximise the sum of all\nfifteen scores. £100m,\n2/5/5/3, three per club.",
         GREEN, GREEN_EDGE, size=7.1)
    _arrow(ax, (x - 0.18, steps["first"]), (left + lw / 2, steps["first"]), "Yes", dy=0.012)

    _box(ax, x, steps["value"], w, 0.062,
         "Price the hold and the candidate sales over three playable weeks.\n"
         "V = this XI + 0.9 × next XI + 0.81 × the one after\n"
         "− 4 per hit − 1 per transfer.\n"
         "Later weeks reuse this week's score. A player with no roster row is 0.",
         BLUE, BLUE_EDGE, size=7.3)
    _arrow(ax, (x, steps["first"] - 0.025), (x, steps["value"] + 0.031), "No", dx=0.03)
    _box(ax, x, steps["search"], w, 0.055,
         "Same-position swaps only.\n"
         "Up to 35 singles. The best 6 may add a second swap.\n"
         "A third is tried only when two swaps is the current leader.\n"
         "At most three transfers, and at most two of them are hits.",
         BLUE, BLUE_EDGE, size=7.3)
    _arrow(ax, (x, steps["value"] - 0.031), (x, steps["search"] + 0.028))

    _diamond(ax, x, steps["legal"], 0.40, 0.048, "Is the held 15\nstill inside the rules?")
    _arrow(ax, (x, steps["search"] - 0.028), (x, steps["legal"] + 0.024))
    _box(ax, left, steps["legal"], lw, 0.050,
         "Take the best legal 15\nthe search found.\nIf it found none, the old 15 stays.",
         SKIP, SKIP_EDGE, size=7.0)
    _arrow(ax, (x - 0.20, steps["legal"]), (left + lw / 2, steps["legal"]), "No", dy=0.011)

    _diamond(ax, x, steps["margin"], 0.42, 0.050, "Does the best move beat\nthe hold by at least 1.25?")
    _arrow(ax, (x, steps["legal"] - 0.024), (x, steps["margin"] + 0.025), "Yes", dx=0.03)
    _box(ax, left, steps["margin"], lw, 0.036, "Keep the 15.", GREY, GREY_EDGE, size=8.0, weight="bold")
    _arrow(ax, (x - 0.21, steps["margin"]), (left + lw / 2, steps["margin"]), "No", dy=0.011)

    _box(ax, x, steps["bar"], w, 0.032,
         "The 15 is fixed.",
         GREEN, GREEN_EDGE, size=8.2, weight="bold")
    _arrow(ax, (x, steps["margin"] - 0.025), (x, steps["bar"] + 0.017), "Yes", dx=0.03)

    # Left-hand outcomes join the bar from beneath each box. The skipped week does not.
    rail = 0.345
    join_y = steps["bar"] + 0.016
    heights = {"first": 0.058, "legal": 0.050, "margin": 0.036}
    for key, h in heights.items():
        y = steps[key]
        right = left + lw / 2
        bottom = y - h / 2
        ax.plot([right, rail], [bottom, bottom], color=NAVY, lw=0.8, zorder=1)
        ax.plot([rail, rail], [bottom, join_y], color=NAVY, lw=0.8, zorder=1)
    _arrow(ax, (rail, join_y), (x - w / 2, steps["bar"]))
    ax.text(0.318, 0.47, "15 ready", rotation=90, ha="center", va="center", fontsize=7.0, color=NAVY)

    ax.text(
        x, 0.012,
        "A chip runs only when the week map names one. The published climb names none.",
        ha="center", va="center", fontsize=7.2, color="#4b5563",
    )

    _box(ax, x, steps["xi"], w, 0.058,
         "Pick the XI from that 15.\n"
         "Each shape takes the top score in each position and the highest sum wins.\n"
         "5-2-3 is not offered. On a tie at 0, a club that plays outranks one that does not.\n"
         "Captain and vice are the top two scores. Bench: goalkeeper, then score order.",
         BLUE, BLUE_EDGE, size=7.1)
    _arrow(ax, (x, steps["bar"] - 0.017), (x, steps["xi"] + 0.026))

    _box(ax, x, steps["score"], w, 0.058,
         "After the matches, score the week. This is not a new decision.\n"
         "A starter on 0 minutes is replaced if the next bench player played\n"
         "and the shape stays legal. An illegal substitute is skipped, not used up.\n"
         "Double the captain, or the vice if the captain played 0. Then subtract 4 per hit.",
         GREEN, GREEN_EDGE, size=7.2)
    _arrow(ax, (x, steps["xi"] - 0.026), (x, steps["score"] + 0.029))

    _box(ax, x, steps["bank"], w, 0.046,
         "Roll spare free transfers, add one, and cap the count at 5.\n"
         "A hit clears the bank before that extra one. The first week sets the count to 1.\n"
         "Carry the 15, the purchase prices, the bank, and the count into the next week.",
         BLUE, BLUE_EDGE, size=7.2)
    _arrow(ax, (x, steps["score"] - 0.029), (x, steps["bank"] + 0.023))

    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(draw())
