"""Stage 2 — team×pos pots and player channel shares."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CHANNEL_COLS = ("minutes", "goals", "assists", "xG", "xA", "total_points")


def build_shares(player_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = player_matches.copy()
    # Team×pos pot within each fixture side.
    group_keys = ["fixture_id", "team_norm", "position", "is_home"]
    for col in CHANNEL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    pots = (
        df.groupby(group_keys, as_index=False)
        .agg(
            team=("team", "first"),
            date=("date", "first"),
            n_players=("player_id", "count"),
            pot_minutes=("minutes", "sum"),
            pot_goals=("goals", "sum"),
            pot_assists=("assists", "sum"),
            pot_xg=("xG", "sum"),
            pot_xa=("xA", "sum"),
            pot_points=("total_points", "sum"),
            pot_cs=("clean_sheets", "max"),
        )
    )

    merged = df.merge(
        pots[
            group_keys
            + [
                "pot_minutes",
                "pot_goals",
                "pot_assists",
                "pot_xg",
                "pot_xa",
                "pot_points",
            ]
        ],
        on=group_keys,
        how="left",
    )
    eps = 1e-9
    merged["share_minutes"] = merged["minutes"] / (merged["pot_minutes"] + eps)
    merged["share_goals"] = np.where(
        merged["pot_goals"] > 0, merged["goals"] / merged["pot_goals"], 0.0
    )
    merged["share_assists"] = np.where(
        merged["pot_assists"] > 0, merged["assists"] / merged["pot_assists"], 0.0
    )
    merged["share_xg"] = np.where(
        merged["pot_xg"] > 0, merged["xG"] / merged["pot_xg"], 0.0
    )
    merged["share_xa"] = np.where(
        merged["pot_xa"] > 0, merged["xA"] / merged["pot_xa"], 0.0
    )
    merged["share_points"] = np.where(
        merged["pot_points"] > 0, merged["total_points"] / merged["pot_points"], 0.0
    )

    # Stability: expanding prior share of minutes / xG among ≥60' apps.
    merged = merged.sort_values(["player_id", "date", "fixture_id"], kind="mergesort")
    for share_col in ("share_minutes", "share_xg", "share_points"):
        merged[f"prior_{share_col}"] = merged.groupby("player_id")[share_col].transform(
            lambda s: s.shift(1).expanding().mean()
        )

    # Diagnostics: concentration of points share (HHI-like mean max share).
    max_share = (
        merged.groupby(group_keys)["share_points"].max().mean()
        if len(merged)
        else float("nan")
    )
    stats = {
        "n_player_rows": int(len(merged)),
        "n_team_pos_pots": int(len(pots)),
        "mean_max_points_share": float(max_share),
        "mean_share_minutes_60": float(
            merged.loc[merged["minutes"] >= 60, "share_minutes"].mean()
        )
        if (merged["minutes"] >= 60).any()
        else float("nan"),
        "pot_points_by_pos": pots.groupby("position")["pot_points"]
        .mean()
        .round(2)
        .to_dict(),
    }
    return merged, pots, stats


def write_stage2_report(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# Stage 2 — Team pots + channel shares",
        "",
        f"- Player rows with shares: **{stats['n_player_rows']}**",
        f"- Team×pos pot rows: **{stats['n_team_pos_pots']}**",
        f"- Mean max points-share within pot: **{stats['mean_max_points_share']:.3f}**",
        f"- Mean minutes-share (≥60′): **{stats['mean_share_minutes_60']:.3f}**",
        "",
        "## Mean pot points by position",
        "",
    ]
    for pos, val in (stats.get("pot_points_by_pos") or {}).items():
        lines.append(f"- {pos}: {val}")
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/player_shares.csv`",
        "- `data/processed/team_pos_pots.csv`",
        "",
        "Channels: minutes, goals, assists, xG, xA, total_points.",
        "Prior shares = expanding mean of previous appearances (no leakage).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_shares(player_matches_path: Path | None = None) -> dict[str, Any]:
    path = player_matches_path or (PROCESSED / "player_matches.csv")
    df = pd.read_csv(path)
    merged, pots, stats = build_shares(df)
    merged.to_csv(PROCESSED / "player_shares.csv", index=False)
    pots.to_csv(PROCESSED / "team_pos_pots.csv", index=False)
    write_stage2_report(REPORTS / "stage_2_shares.md", stats)
    return stats
