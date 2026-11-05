"""Plot one player's expanding actuals + GW team scoreline / xscoreline.

Scorer-simplify dig: no shares, pots, or renorm — just expanding means of the
player's GW totals, minutes, and the team's GF/GA vs Opta Σ xG / opp Σ xG.

  PYTHONPATH=. python3 scripts/plot_player_expanding_view.py \\
    --player "Bukayo Saka" --season 2025-26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
PLOTS = ROOT / "data" / "plots"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ARTIFACTS = Path("/opt/cursor/artifacts")


def merged_path(season: str) -> Path:
    return CACHE / f"merged_gw_{season.replace('-', '_')}.csv"


def build_frame(df: pd.DataFrame, player: str) -> pd.DataFrame:
    team_xg = (
        df.groupby(["fixture", "team"], as_index=False)["expected_goals"]
        .sum()
        .rename(columns={"expected_goals": "team_xg"})
    )
    fx_pair = team_xg.merge(team_xg, on="fixture", suffixes=("", "_opp"))
    fx_pair = fx_pair.loc[fx_pair["team"] != fx_pair["team_opp"]].copy()
    fx_pair = fx_pair.rename(
        columns={"team_xg_opp": "team_xga", "team_opp": "opp_team"}
    )[["fixture", "team", "team_xg", "team_xga", "opp_team"]]

    pl = df.loc[df["name"] == player].copy()
    if pl.empty:
        # fuzzy: unique substring match
        hits = df.loc[df["name"].str.contains(player, case=False, na=False), "name"].unique()
        if len(hits) == 1:
            pl = df.loc[df["name"] == hits[0]].copy()
            player = str(hits[0])
        else:
            raise SystemExit(f"player not found: {player!r}; hits={list(hits)[:8]}")
    pl = pl.merge(fx_pair, on=["fixture", "team"], how="left")
    pl["gf"] = np.where(pl["was_home"], pl["team_h_score"], pl["team_a_score"]).astype(
        float
    )
    pl["ga"] = np.where(pl["was_home"], pl["team_a_score"], pl["team_h_score"]).astype(
        float
    )

    agg_kwargs: dict[str, tuple[str, str]] = {
        "minutes": ("minutes", "sum"),
        "total_points": ("total_points", "sum"),
        "goals": ("goals_scored", "sum"),
        "assists": ("assists", "sum"),
        "clean_sheets": ("clean_sheets", "sum"),
        "goals_conceded": ("goals_conceded", "sum"),
        "bonus": ("bonus", "sum"),
        "bps": ("bps", "sum"),
        "player_xg": ("expected_goals", "sum"),
        "player_xa": ("expected_assists", "sum"),
        "gf": ("gf", "mean"),
        "ga": ("ga", "mean"),
        "team_xg": ("team_xg", "mean"),
        "team_xga": ("team_xga", "mean"),
        "n_fixtures": ("fixture", "nunique"),
    }
    if "saves" in pl.columns:
        agg_kwargs["saves"] = ("saves", "sum")
    if "defensive_contribution" in pl.columns:
        agg_kwargs["defcon"] = ("defensive_contribution", "sum")

    agg = pl.groupby("GW", as_index=False).agg(**agg_kwargs).sort_values("GW")
    exp_cols = [
        c
        for c in (
            "minutes",
            "total_points",
            "goals",
            "assists",
            "clean_sheets",
            "goals_conceded",
            "bonus",
            "bps",
            "player_xg",
            "player_xa",
            "saves",
            "defcon",
        )
        if c in agg.columns
    ]
    for c in exp_cols:
        agg[f"exp_{c}"] = agg[c].expanding(min_periods=1).mean()
    agg["cum_points"] = agg["total_points"].cumsum()
    agg.attrs["player"] = player
    return agg


def plot_frame(agg: pd.DataFrame, player: str, season: str, path: Path) -> None:
    gw = agg["GW"].to_numpy()
    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"hspace": 0.08}
    )

    ax = axes[0]
    ax.bar(gw, agg["total_points"], color="#b0b7c3", alpha=0.45, label="GW points", width=0.7)
    ax.plot(
        gw,
        agg["exp_total_points"],
        color="#1f4e79",
        lw=2.2,
        label="Expanding mean total_points",
    )
    ax.set_ylabel("Points")
    ax.set_title(f"{player} — {season}: expanding actuals, minutes, team scoreline")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1]
    ax.bar(gw, agg["minutes"], color="#c5d5c5", alpha=0.55, label="GW minutes", width=0.7)
    ax.plot(
        gw, agg["exp_minutes"], color="#2f6f4e", lw=2.2, label="Expanding mean minutes"
    )
    ax.axhline(60, color="#666", ls=":", lw=1, alpha=0.7)
    ax.set_ylabel("Minutes")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[2]
    ax.plot(gw, agg["gf"], color="#c0392b", lw=1.8, marker="o", ms=3.5, label="GF (actual)")
    ax.plot(
        gw,
        agg["team_xg"],
        color="#c0392b",
        lw=1.4,
        ls="--",
        marker="o",
        ms=3,
        alpha=0.75,
        label="xGF (team Opta Σ)",
    )
    ax.plot(gw, agg["ga"], color="#2980b9", lw=1.8, marker="s", ms=3.5, label="GA (actual)")
    ax.plot(
        gw,
        agg["team_xga"],
        color="#2980b9",
        lw=1.4,
        ls="--",
        marker="s",
        ms=3,
        alpha=0.75,
        label="xGA (opp Opta Σ)",
    )
    ax.set_ylabel("Goals")
    ax.set_xlabel("Gameweek")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks(list(range(1, int(gw.max()) + 1, 2)))

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


def write_report(agg: pd.DataFrame, player: str, season: str, png_name: str, csv: Path) -> Path:
    tag = season.replace("-", "_")
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in player).strip("_")
    path = REPORTS / f"player_expanding_view_{slug}_{tag}.md"
    sample_gws = [1, 5, 10, 15, 20, 25, 30, 35, 38]
    cols = [
        c
        for c in (
            "GW",
            "minutes",
            "exp_minutes",
            "total_points",
            "exp_total_points",
            "goals",
            "exp_goals",
            "assists",
            "exp_assists",
            "clean_sheets",
            "exp_clean_sheets",
            "goals_conceded",
            "exp_goals_conceded",
            "gf",
            "ga",
            "team_xg",
            "team_xga",
        )
        if c in agg.columns
    ]
    sample = agg.loc[agg["GW"].isin(sample_gws), cols]
    lines = [
        f"# Player expanding view — {player} ({season})",
        "",
        "Scorer-simplify dig: raw expanding actuals + GW team scoreline / xscoreline.",
        "No shares, no pots, no renorm — just what the player and the match produced.",
        "",
        f"![plot](../data/plots/{png_name})",
        "",
        "## Definitions",
        "",
        "- **Expanding mean** of player GW totals through GW t (DGW rows summed into the GW).",
        "- **GF / GA**: team scoreline for that GW (mean across fixtures if DGW).",
        "- **xGF**: sum of team-mate `expected_goals` that fixture; **xGA**: opponent team Σ xG.",
        "",
        "## Snapshot rows",
        "",
        _md_table(sample),
        "",
        f"Full CSV: `{csv.as_posix()}`",
        "",
        "## End-of-season expanding means",
        "",
        f"- minutes: **{agg['exp_minutes'].iloc[-1]:.1f}**",
        f"- total_points: **{agg['exp_total_points'].iloc[-1]:.2f}**",
        f"- goals: **{agg['exp_goals'].iloc[-1]:.2f}**",
        f"- assists: **{agg['exp_assists'].iloc[-1]:.2f}**",
        f"- clean_sheets: **{agg['exp_clean_sheets'].iloc[-1]:.2f}**",
        f"- goals_conceded: **{agg['exp_goals_conceded'].iloc[-1]:.2f}**",
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
    md = write_report(agg, player, season, png_name, csv)
    print(f"wrote {png}")
    print(f"wrote {csv}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
