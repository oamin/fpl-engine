"""Per-week fixture context for a hold or a transfer.

A club has a fixture when the season sheet has a row for that club. A player
who is benched while his club plays is still ``fixture``. His score is
unchanged. A player whose club has no row is ``no_fixture``: this week's
ranking score is 0, and a score from an earlier week is not used. A week
with no clubs is not played.

The points model and the transfer penalty are unchanged. The record says
why the hold or the move won. It does not force a sale.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.teams import norm_team

# Ranking columns only. ``value`` is the price and must stay.
SCORE_COLS = (
    "score_xp",
    "xp",
    "exp_points",
    "score_exp_points",
    "score_team_prior",
    "score_roll3_points",
    "score_xmi",
    "score_price",
    "xfer_score",
    "xfer_exp",
)


def clubs_by_gw(roster: pd.DataFrame) -> dict[int, set[str]]:
    """Clubs that have at least one sheet row in each gameweek."""
    if roster.empty or "gw" not in roster.columns:
        return {}
    team_col = "team" if "team" in roster.columns else "team_norm"
    out: dict[int, set[str]] = {}
    frame = roster.dropna(subset=["gw"])
    for gw, block in frame.groupby(frame["gw"].astype(int), sort=False):
        clubs = {norm_team(t) for t in block[team_col].astype(str)}
        clubs.discard("")
        if clubs:
            out[int(gw)] = clubs
    return out


def playable_gws(gws: list[int], clubs: dict[int, set[str]]) -> list[int]:
    """Drop a week that has no clubs. 2022/23 Gameweek 7 is the case."""
    return [int(g) for g in gws if clubs.get(int(g))]


def fixture_tag(team: object, gw: int, clubs: dict[int, set[str]]) -> str:
    playing = clubs.get(int(gw), set())
    return "fixture" if norm_team(str(team)) in playing else "no_fixture"


def apply_fixture_tags(
    pool: pd.DataFrame,
    gw: int,
    clubs: dict[int, set[str]],
) -> pd.DataFrame:
    """Mark each row and zero ranking scores for clubs with no fixture.

    A score of 0 is written on the ranking columns, not left missing, so a
    fallback cannot refill it from an earlier week or from the price.
    """
    out = pool.copy()
    if out.empty:
        out["fixture_tag"] = []
        out["xi_priority"] = []
        return out
    team_col = "team" if "team" in out.columns else "team_norm"
    tags = [
        fixture_tag(team, gw, clubs) for team in out[team_col].astype(str)
    ]
    out["fixture_tag"] = tags
    out["xi_priority"] = [0.0 if tag == "no_fixture" else 1.0 for tag in tags]
    blank = out["fixture_tag"].eq("no_fixture")
    if blank.any():
        for col in SCORE_COLS:
            if col in out.columns:
                out.loc[blank, col] = 0.0
    return out


def horizon_sheet(
    squad_ids: set[str],
    score_now: dict[str, float],
    meta: dict[str, dict[str, Any]],
    gw: int,
    future_gws: list[int],
    clubs: dict[int, set[str]],
    *,
    score_col: str = "score",
    horizon: int = 3,
) -> list[dict[str, Any]]:
    """XI sum and no-fixture count for each week of the hold."""
    from src.models.season_climb import pick_xi

    later = playable_gws([g for g in future_gws if int(g) > int(gw)], clubs)
    weeks = [int(gw)] + later[: max(int(horizon) - 1, 0)]
    rows_out: list[dict[str, Any]] = []
    for g in weeks:
        built = []
        for pid in squad_ids:
            info = meta.get(pid)
            if info is None:
                continue
            tag = fixture_tag(info.get("team_norm", ""), g, clubs)
            score = 0.0 if tag == "no_fixture" else float(score_now.get(pid, 0.0))
            built.append(
                {
                    "player_id": pid,
                    "position": info.get("position", "MID"),
                    "team_norm": info.get("team_norm", ""),
                    score_col: score,
                    "xi_priority": 0.0 if tag == "no_fixture" else 1.0,
                    "fixture_tag": tag,
                }
            )
        frame = pd.DataFrame(built)
        if len(frame) < 11:
            rows_out.append(
                {"gw": g, "xi_sum": float("nan"), "n_no_fixture": 0, "played": True}
            )
            continue
        xi, _form = pick_xi(frame, score_col, priority_col="xi_priority")
        rows_out.append(
            {
                "gw": int(g),
                "xi_sum": float(pd.to_numeric(xi[score_col], errors="coerce").fillna(0).sum()),
                "n_no_fixture": int((xi["fixture_tag"] == "no_fixture").sum()),
                "played": True,
            }
        )
    return rows_out
