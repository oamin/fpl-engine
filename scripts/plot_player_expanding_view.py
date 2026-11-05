"""Plot one player's GW actuals — all FPL points channels.

Bar panels: total points, minutes, then every rule channel that can contribute
to ``total_points`` (appearance, G, A, CS, GC, saves, DefCon, bonus, cards,
own goals, penalties).

  PYTHONPATH=. python3 scripts/plot_player_expanding_view.py \\
    --player "Bukayo Saka" --season 2025-26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rules.fpl_2026 import (
    appearance_points,
    assist_points,
    card_points,
    clean_sheet_points,
    defcon_points,
    goal_points,
    goals_conceded_points,
    own_goal_points,
    penalty_points,
    save_points,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
PLOTS = ROOT / "data" / "plots"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ARTIFACTS = Path("/opt/cursor/artifacts")


def merged_path(season: str) -> Path:
    return CACHE / f"merged_gw_{season.replace('-', '_')}.csv"


def _fixture_point_channels(row: pd.Series) -> dict[str, float]:
    pos = str(row["position"])
    m = int(row["minutes"] or 0)
    g = int(row["goals_scored"] or 0)
    a = int(row["assists"] or 0)
    cs = bool(int(row["clean_sheets"] or 0))
    gc = int(row["goals_conceded"] or 0)
    saves = int(row["saves"] or 0) if "saves" in row.index else 0
    defcon = (
        int(row["defensive_contribution"] or 0)
        if "defensive_contribution" in row.index
        else 0
    )
    bonus = int(row["bonus"] or 0)
    yel = int(row["yellow_cards"] or 0)
    red = int(row["red_cards"] or 0)
    og = int(row["own_goals"] or 0)
    pen_save = int(row["penalties_saved"] or 0) if "penalties_saved" in row.index else 0
    pen_miss = int(row["penalties_missed"] or 0) if "penalties_missed" in row.index else 0

    return {
        "pts_appear": float(appearance_points(m)),
        "pts_goals": float(goal_points(pos, g)),
        "pts_assists": float(assist_points(a)),
        "pts_cs": float(clean_sheet_points(pos, m, cs)),
        "pts_gc": float(goals_conceded_points(pos, m, gc)),
        "pts_saves": float(save_points(saves)),
        "pts_defcon": float(defcon_points(pos, defcon)),
        "pts_bonus": float(bonus),
        "pts_cards": float(card_points(yel, red)),
        "pts_own_goal": float(own_goal_points(og)),
        "pts_pens": float(penalty_points(pen_save, pen_miss)),
    }


CHANNEL_PANELS: list[tuple[str, str, str]] = [
    ("total_points", "Points", "#5c6b7a"),
    ("minutes", "Minutes", "#3d6b4f"),
    ("pts_appear", "Appear", "#6b7c5c"),
    ("pts_goals", r"$G$ pts", "#a33b2b"),
    ("pts_assists", r"$A$ pts", "#8a5a00"),
    ("pts_cs", "CS pts", "#2f5f8a"),
    ("pts_gc", "GC pts", "#4a4a6a"),
    ("pts_saves", "Saves pts", "#5a6a8a"),
    ("pts_defcon", "DefCon pts", "#3a6a6a"),
    ("pts_bonus", "Bonus", "#7a5a2a"),
    ("pts_cards", "Cards", "#6a3a3a"),
    ("pts_own_goal", "OG pts", "#4a3a5a"),
    ("pts_pens", "Pen pts", "#5a4a3a"),
]

POINT_CHANNEL_COLS = [
    "pts_appear",
    "pts_goals",
    "pts_assists",
    "pts_cs",
    "pts_gc",
    "pts_saves",
    "pts_defcon",
    "pts_bonus",
    "pts_cards",
    "pts_own_goal",
    "pts_pens",
]


def build_frame(df: pd.DataFrame, player: str) -> pd.DataFrame:
    pl = df.loc[df["name"] == player].copy()
    if pl.empty:
        hits = df.loc[
            df["name"].str.contains(player, case=False, na=False), "name"
        ].unique()
        if len(hits) == 1:
            pl = df.loc[df["name"] == hits[0]].copy()
            player = str(hits[0])
        else:
            raise SystemExit(f"player not found: {player!r}; hits={list(hits)[:8]}")

    channels = pl.apply(_fixture_point_channels, axis=1, result_type="expand")
    pl = pd.concat([pl.reset_index(drop=True), channels.reset_index(drop=True)], axis=1)
    pl["pts_sum_channels"] = pl[POINT_CHANNEL_COLS].sum(axis=1)

    sum_cols = [
        "minutes",
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "bps",
        "saves",
        "defensive_contribution",
        "yellow_cards",
        "red_cards",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        *POINT_CHANNEL_COLS,
        "pts_sum_channels",
    ]
    present = [c for c in sum_cols if c in pl.columns]
    agg = (
        pl.groupby("GW", as_index=False)[present]
        .sum()
        .sort_values("GW")
        .rename(
            columns={
                "goals_scored": "goals",
                "defensive_contribution": "defcon",
            }
        )
    )
    agg["pts_residual"] = agg["total_points"] - agg["pts_sum_channels"]
    agg.attrs["player"] = player
    return agg


def plot_frame(agg: pd.DataFrame, player: str, season: str, path: Path) -> None:
    gw = agg["GW"].to_numpy()
    panels = [(c, lab, col) for c, lab, col in CHANNEL_PANELS if c in agg.columns]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(11, 1.05 * len(panels) + 1.5),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, ylabel, color) in zip(axes, panels):
        ax.bar(gw, agg[col], color=color, alpha=0.78, width=0.7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        if col == "minutes":
            ax.axhline(60, color="#888", ls=":", lw=1, alpha=0.7)
        # zero line helps read deductions
        if col.startswith("pts_") or col == "total_points":
            ax.axhline(0, color="#bbb", lw=0.6)
    axes[0].set_title(
        f"{player} — {season}: GW actuals (all points channels)", fontsize=11
    )
    axes[-1].set_xlabel("Gameweek")
    axes[-1].set_xticks(list(range(1, int(gw.max()) + 1, 2)))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, r in df.iterrows():
        cells = []
        for h in headers:
            v = r[h]
            cells.append(str(int(v)) if h == "GW" else f"{float(v):.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    agg: pd.DataFrame, player: str, season: str, png_name: str, csv: Path
) -> Path:
    tag = season.replace("-", "_")
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in player).strip("_")
    path = REPORTS / f"player_expanding_view_{slug}_{tag}.md"
    sample_gws = [1, 5, 10, 11, 15, 20, 25, 30, 35, 38]
    cols = [
        c
        for c in (
            "GW",
            "total_points",
            "minutes",
            *POINT_CHANNEL_COLS,
            "pts_sum_channels",
            "pts_residual",
        )
        if c in agg.columns
    ]
    sample = agg.loc[agg["GW"].isin(sample_gws), cols]
    max_abs_resid = float(np.nanmax(np.abs(agg["pts_residual"]))) if len(agg) else 0.0
    lines = [
        f"# Player GW actuals — {player} ({season})",
        "",
        "Scorer-simplify dig: bar plots of GW **total points**, **minutes**, and "
        "every FPL points channel (appearance, goals, assists, CS, GC, saves, "
        "DefCon, bonus, cards, OG, pens). Channels computed from "
        "`src.rules.fpl_2026` per fixture, then summed within the GW.",
        "",
        f"![plot](../data/plots/{png_name})",
        "",
        f"Channel sum vs `total_points`: max |residual| = **{max_abs_resid:.2f}**.",
        "",
        "## Snapshot rows",
        "",
        _md_table(sample),
        "",
        f"Full CSV: `{csv.as_posix()}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player", default="Bukayo Saka")
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--merged", type=Path, default=None)
    args = ap.parse_args(argv)

    season = str(args.season)
    tag = season.replace("-", "_")
    df = pd.read_csv(args.merged or merged_path(season))
    agg = build_frame(df, str(args.player))
    player = str(agg.attrs.get("player") or args.player)
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in player).strip("_")

    PLOTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    png_name = f"player_expanding_view_{slug}_{tag}.png"
    png = PLOTS / png_name
    csv = PROCESSED / f"player_expanding_view_{slug}_{tag}.csv"
    agg.to_csv(csv, index=False)
    plot_frame(agg, player, season, png)
    if ARTIFACTS.exists():
        plot_frame(agg, player, season, ARTIFACTS / png_name)
        plot_frame(agg, player, season, ARTIFACTS / f"player_gw_actuals_{slug}_{tag}.png")
    md = write_report(agg, player, season, png_name, csv)
    print(f"wrote {png}")
    print(f"wrote {csv}")
    print(f"wrote {md}")
    print(
        "max |pts residual|",
        float(np.nanmax(np.abs(agg["pts_residual"]))),
        "GW11 total",
        int(agg.loc[agg["GW"] == 11, "total_points"].sum()),
        "channels",
        {
            c: float(agg.loc[agg["GW"] == 11, c].sum())
            for c in POINT_CHANNEL_COLS
            if c in agg.columns
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
