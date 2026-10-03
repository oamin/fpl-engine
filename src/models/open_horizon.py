"""Horizon step scores from opening 1X2, with shares frozen at the deadline.

The current week keeps the published closing-price score. A later week uses
that fixture's opening 1X2 and over/under. The player's share and minutes
stay on the deadline row. A missing total market is treated as an even
over/under. A fixture the book has not priced uses the club's earlier
scoring rate. A blank week is zero. Two fixtures are added.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.markets.shin import shin_1x2, shin_pair
from src.models.xp_engine import CS_PTS, GOAL_PTS, SAVES_PER_GC
from src.teams import norm_team

NEUTRAL_TOTAL = 2.45


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 1.0:
        return None
    return number


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def side_pot(
    home: float,
    draw: float,
    away: float,
    over: float | None,
    under: float | None,
    *,
    is_home: bool,
) -> dict[str, float]:
    """Team λ and clean-sheet proxy for one side of a match."""
    p_home, p_draw, p_away = shin_1x2(home, draw, away)
    if over is None or under is None:
        p_over, p_under = 0.5, 0.5
    else:
        p_over, p_under = shin_pair(over, under)
    e_total = NEUTRAL_TOTAL + 1.10 * (p_over - p_under)
    home_att = p_home + 0.5 * p_draw
    away_att = p_away + 0.5 * p_draw
    denom = home_att + away_att
    if is_home:
        att, other, p_win = home_att, away_att, p_home
    else:
        att, other, p_win = away_att, home_att, p_away
    lam = e_total * 0.5 if denom <= 0 else e_total * att / denom
    lam = _clip(lam, 0.2, 3.5)
    lam_assist = _clip(lam * 0.75, 0.1, 3.0)
    p_not_lose = p_win + 0.5 * p_draw
    p_cs = _clip(0.08 + 0.35 * p_not_lose + 0.15 * p_under, 0.05, 0.55)
    return {
        "lam_scored": lam,
        "lam_assist": lam_assist,
        "e_total": float(e_total),
        "p_cs_mkt": p_cs,
        "attack": float(att),
        "threat": float(other),
    }


def neutral_pot() -> dict[str, float]:
    """Even match. Used only when a club has no earlier rate and no price."""
    return side_pot(2.5, 3.2, 2.5, 1.9, 1.9, is_home=True)


def xp_on_pot(
    *,
    position: str,
    xmi: float,
    share_xg: float,
    share_xa: float,
    exp_defcon_hit: float,
    fwd_goal_scale: float,
    pot: dict[str, float],
) -> float:
    """One fixture, same components as ``compute_xp``, with the deadline scale."""
    pos = str(position)
    minutes = float(xmi) if math.isfinite(xmi) else 0.0
    p_play = _clip(minutes / 15.0, 0.0, 1.0)
    p60 = _clip((minutes - 30.0) / 30.0, 0.0, 1.0) if minutes >= 30.0 else 0.0
    appear = p_play + p60
    goal_pts = float(GOAL_PTS.get(pos, 4.0))
    cs_pts = float(CS_PTS.get(pos, 0.0))
    scale = float(fwd_goal_scale) if math.isfinite(fwd_goal_scale) else 1.0
    share_g = float(share_xg) if math.isfinite(share_xg) else 0.0
    share_a = float(share_xa) if math.isfinite(share_xa) else 0.0
    defcon = float(exp_defcon_hit) if math.isfinite(exp_defcon_hit) else 0.0
    xp_goals = share_g * pot["lam_scored"] * goal_pts * scale
    xp_assists = share_a * pot["lam_assist"] * 3.0
    xp_cs = p60 * pot["p_cs_mkt"] * cs_pts
    xp_defcon = p60 * defcon * 2.0 if pos in {"DEF", "MID", "FWD"} else 0.0
    lam_against = _clip(pot["e_total"] - pot["lam_scored"], 0.0, 5.0)
    xp_saves = p60 * (lam_against * SAVES_PER_GC / 3.0) if pos == "GKP" else 0.0
    xp_bps = 0.18 * xp_goals + 0.12 * xp_assists + 0.08 * xp_cs
    xp_gc = p60 * (lam_against / 2.0) if pos in {"GKP", "DEF"} else 0.0
    xp_card = (minutes / 90.0) * 0.15
    return (
        appear
        + xp_goals
        + xp_assists
        + xp_cs
        + xp_defcon
        + xp_saves
        + xp_bps
        - xp_gc
        - xp_card
    )


def opening_pots_by_team_gw(
    odds: pd.DataFrame,
    fixtures: list[dict[str, Any]],
    team_names: dict[int, str],
) -> dict[tuple[int, str], list[dict[str, float]]]:
    """``(gameweek, club)`` to that club's opening-price pots.

    ``odds`` is a football-data match table with ``Date``, ``HomeTeam``,
    ``AwayTeam`` and the opening columns. ``fixtures`` are FPL fixtures.
    """
    priced: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in odds.to_dict("records"):
        home = norm_team(row.get("HomeTeam"))
        away = norm_team(row.get("AwayTeam"))
        date = str(row.get("Date") or "")
        # football-data dates are d/m/Y. FPL kickoffs are ISO.
        parts = date.split("/")
        if len(parts) != 3:
            continue
        iso = f"{parts[2]}-{int(parts[1]):02d}-{int(parts[0]):02d}"
        h = _finite(row.get("AvgH"))
        d = _finite(row.get("AvgD"))
        a = _finite(row.get("AvgA"))
        if h is None or d is None or a is None:
            continue
        over = _finite(row.get("Avg>2.5"))
        under = _finite(row.get("Avg<2.5"))
        priced[(iso, home, away)] = {
            "home": h,
            "draw": d,
            "away": a,
            "over": over if over is not None else math.nan,
            "under": under if under is not None else math.nan,
        }

    out: dict[tuple[int, str], list[dict[str, float]]] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        kickoff = fixture.get("kickoff_time") or ""
        if event is None or not kickoff:
            continue
        gw = int(event)
        day = str(kickoff)[:10]
        home_name = team_names.get(int(fixture["team_h"]))
        away_name = team_names.get(int(fixture["team_a"]))
        if not home_name or not away_name:
            continue
        home = norm_team(home_name)
        away = norm_team(away_name)
        quote = priced.get((day, home, away))
        if quote is None:
            continue
        over = quote["over"]
        under = quote["under"]
        over_v = None if not math.isfinite(over) else over
        under_v = None if not math.isfinite(under) else under
        out.setdefault((gw, home), []).append(
            side_pot(quote["home"], quote["draw"], quote["away"], over_v, under_v, is_home=True)
        )
        out.setdefault((gw, away), []).append(
            side_pot(quote["home"], quote["draw"], quote["away"], over_v, under_v, is_home=False)
        )
    return out


def prior_pots(history: pd.DataFrame, before_gw: int) -> dict[str, dict[str, float]]:
    """Mean team rate from rows strictly before the deadline."""
    if history.empty or "gw" not in history.columns:
        return {}
    past = history.loc[pd.to_numeric(history["gw"], errors="coerce") < before_gw]
    if past.empty:
        return {}
    pots: dict[str, dict[str, float]] = {}
    for club, block in past.groupby(past["team_norm"].map(norm_team)):
        lam = pd.to_numeric(block.get("lam_scored"), errors="coerce").dropna()
        assist = pd.to_numeric(block.get("lam_assist"), errors="coerce").dropna()
        total = pd.to_numeric(block.get("e_total"), errors="coerce").dropna()
        cs = pd.to_numeric(block.get("p_cs_mkt"), errors="coerce").dropna()
        if lam.empty:
            continue
        lam_v = float(lam.mean())
        pots[str(club)] = {
            "lam_scored": lam_v,
            "lam_assist": float(assist.mean()) if not assist.empty else _clip(lam_v * 0.75, 0.1, 3.0),
            "e_total": float(total.mean()) if not total.empty else lam_v * 2.0,
            "p_cs_mkt": float(cs.mean()) if not cs.empty else 0.25,
        }
    return pots


def _row_float(row: dict[str, Any], key: str, default: float) -> float:
    value = row.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _share_missing(share: object) -> bool:
    if share is None:
        return True
    try:
        number = float(share)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return not math.isfinite(number)


def project_player(
    row: dict[str, Any],
    gw: int,
    pots: dict[tuple[int, str], list[dict[str, float]]],
    priors: dict[str, dict[str, float]],
    *,
    calendar: dict[tuple[int, str], int],
) -> float:
    """Step score for one player. Deadline share. No future player row.

    A blank this week writes 0 onto ``score_xp``. That 0 is not the score
    for a later week in which the club plays. A missing share on a normal
    week still keeps the deadline score, because there is no shot share
    to move onto the next opponent.
    """
    club = norm_team(str(row.get("team_norm") or ""))
    n_fix = int(calendar.get((gw, club), 0))
    if n_fix <= 0:
        return 0.0
    blanked = str(row.get("fixture_tag") or "") == "no_fixture"
    if _share_missing(row.get("share_xG")) and not blanked:
        return _row_float(row, "score_xp", 0.0)
    quotes = pots.get((gw, club), [])
    fallback = priors.get(club, neutral_pot())
    if not quotes:
        quotes = [fallback]
    elif len(quotes) < n_fix:
        quotes = list(quotes) + [fallback] * (n_fix - len(quotes))
    total = 0.0
    for pot in quotes[:n_fix]:
        total += xp_on_pot(
            position=str(row.get("position") or "MID"),
            xmi=_row_float(row, "xmi", 0.0),
            share_xg=_row_float(row, "share_xG", 0.0),
            share_xa=_row_float(row, "share_xA", 0.0),
            exp_defcon_hit=_row_float(row, "exp_defcon_hit", 0.0),
            fwd_goal_scale=_row_float(row, "fwd_goal_scale", 1.0),
            pot=pot,
        )
    return total


def make_horizon_scores(
    pots: dict[tuple[int, str], list[dict[str, float]]],
    history: pd.DataFrame,
    calendar: dict[tuple[int, str], int],
    *,
    horizon: int = 3,
):
    """Return the callback ``run_ft_season`` uses for this arm."""

    def horizon_scores(
        gw: int, pool: pd.DataFrame, gws: list[int]
    ) -> dict[int, dict[str, float]]:
        steps = [int(gw)] + [int(g) for g in gws if int(g) > int(gw)][: horizon - 1]
        priors = prior_pots(history, int(gw))
        rows = pool.to_dict("records")
        out: dict[int, dict[str, float]] = {}
        for step in steps:
            scores: dict[str, float] = {}
            for row in rows:
                pid = str(row.get("player_id"))
                if step == int(gw):
                    scores[pid] = _row_float(row, "score_xp", _row_float(row, "xp", 0.0))
                else:
                    scores[pid] = project_player(row, step, pots, priors, calendar=calendar)
            out[step] = scores
        return out

    return horizon_scores


def single_fixture_calendar(roster: pd.DataFrame) -> dict[tuple[int, str], int]:
    """One if the club has a sheet row that week. A double is still one.

    Adding the two fixtures together is a separate change. This calendar
    keeps that out of the horizon screen.
    """
    from src.models.blank_context import clubs_by_gw

    clubs = clubs_by_gw(roster)
    return {(int(gw), club): 1 for gw, names in clubs.items() for club in names}


def opening_pots_for_sheet(
    odds: pd.DataFrame, sheet: pd.DataFrame
) -> dict[tuple[int, str], list[dict[str, float]]]:
    """Opening 1X2 pots, joined on the sheet kickoff. No future player row.

    ``sheet`` needs ``team``, ``kickoff_time``, ``was_home``, and ``gw``
    (or ``GW``). The odds table is the football-data file, opening columns.
    """
    frame = sheet.copy()
    if "gw" not in frame.columns and "GW" in frame.columns:
        frame = frame.rename(columns={"GW": "gw"})
    frame = frame.dropna(subset=["kickoff_time", "team", "gw"])
    frame["kickoff_time"] = frame["kickoff_time"].astype(str)
    home_flag = frame["was_home"].astype(str).str.lower().isin(["true", "1"])
    homes = (
        frame.loc[home_flag, ["kickoff_time", "team", "gw"]]
        .drop_duplicates("kickoff_time")
        .rename(columns={"team": "home"})
    )
    aways = (
        frame.loc[~home_flag, ["kickoff_time", "team"]]
        .drop_duplicates("kickoff_time")
        .rename(columns={"team": "away"})
    )
    matched = homes.merge(aways, on="kickoff_time", how="inner").sort_values("kickoff_time")
    names = sorted(set(matched["home"]).union(set(matched["away"])))
    id_of = {name: i + 1 for i, name in enumerate(names)}
    fixtures = [
        {
            "event": int(row.gw),
            "kickoff_time": str(row.kickoff_time),
            "team_h": id_of[row.home],
            "team_a": id_of[row.away],
        }
        for row in matched.itertuples(index=False)
    ]
    team_names = {i: name for name, i in id_of.items()}
    return opening_pots_by_team_gw(odds, fixtures, team_names)


def deadline_rate_history(feat: pd.DataFrame) -> pd.DataFrame:
    """One team rate per week, for the fallback when a later line is missing."""
    cols = ["gw", "team_norm", "lam_scored", "lam_assist", "e_total", "p_cs_mkt"]
    missing = [col for col in cols if col not in feat.columns]
    if missing:
        raise RuntimeError(f"feature frame has no {missing}")
    return feat.loc[:, cols].drop_duplicates(["gw", "team_norm"])


def fixture_calendar(
    fixtures: list[dict[str, Any]], team_names: dict[int, str]
) -> dict[tuple[int, str], int]:
    """How many matches each club has in each gameweek. A missing key is a blank."""
    counts: dict[tuple[int, str], int] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None:
            continue
        gw = int(event)
        for side in ("team_h", "team_a"):
            name = team_names.get(int(fixture[side]))
            if not name:
                continue
            key = (gw, norm_team(name))
            counts[key] = counts.get(key, 0) + 1
    return counts
