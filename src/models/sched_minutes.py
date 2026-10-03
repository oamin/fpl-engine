"""Minutes prior that keeps bench weeks.

``xmi`` averages the last three appearances, so a week with no minutes never
moves it. ``xmi_sched`` averages the last three scheduled club gameweeks and
writes 0 when the player did not play. Shot share is unchanged. Goals and
assists are scaled by the chance of playing, which is 1 when the prior is at
least 15 minutes, so a regular's score matches the published one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.xp_engine import CS_PTS, GOAL_PTS, SAVES_PER_GC

WINDOW = 3


def per_fixture_minutes(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per player-gameweek. Minutes are per fixture, so a double is not 180."""
    frame = raw.copy()
    frame["element"] = frame["element"].astype(str)
    frame["gw"] = pd.to_numeric(frame["GW"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["gw"])
    frame["gw"] = frame["gw"].astype(int)
    summed = frame.groupby(["element", "gw"], as_index=False).agg(
        minutes=("minutes", "sum"),
        team=("team", "last"),
    )
    counted = (
        frame.groupby(["team", "gw"])["fixture"].nunique().rename("n_fix").reset_index()
    )
    summed = summed.merge(counted, on=["team", "gw"], how="left")
    summed["n_fix"] = pd.to_numeric(summed["n_fix"], errors="coerce").fillna(1).clip(lower=1)
    summed["m"] = summed["minutes"] / summed["n_fix"]
    summed["player_id"] = season + ":" + summed["element"]
    return summed[["player_id", "gw", "team", "m"]]


def xmi_sched_for(feat: pd.DataFrame, minutes: pd.DataFrame) -> pd.Series:
    """Shift-1 mean of up to three earlier scheduled weeks. No later week is read."""
    minutes = minutes.copy()
    minutes["player_id"] = minutes["player_id"].astype(str)
    minutes["gw"] = minutes["gw"].astype(int)
    by_player = {
        pid: block.sort_values("gw")
        for pid, block in minutes.groupby("player_id", sort=False)
    }
    club_weeks: dict[str, list[int]] = {}
    for team, block in minutes.groupby("team", sort=False):
        club_weeks[str(team)] = sorted(set(block["gw"].astype(int)))

    out = []
    index = []
    for row in feat.itertuples(index=False):
        pid = str(row.player_id)
        gw = int(row.gw)
        team = str(row.team)
        index.append(row.Index if hasattr(row, "Index") else None)
        played = by_player.get(pid)
        if played is None or played.empty:
            out.append(np.nan)
            continue
        first = int(played["gw"].min())
        prev = [g for g in club_weeks.get(team, []) if first <= g < gw][-WINDOW:]
        if not prev:
            out.append(np.nan)
            continue
        lookup = dict(zip(played["gw"].astype(int), played["m"].astype(float), strict=False))
        out.append(float(np.mean([lookup.get(g, 0.0) for g in prev])))
    return pd.Series(out, index=feat.index, dtype=float)


def _clip(value: pd.Series, lo: float, hi: float) -> pd.Series:
    return value.clip(lo, hi)


def score_xp_sched(feat: pd.DataFrame, xmi: pd.Series) -> pd.Series:
    """Published components, with minutes-driven terms rebuilt from ``xmi``."""
    x = pd.to_numeric(xmi, errors="coerce")
    p_play = _clip(x / 15.0, 0.0, 1.0)
    p60 = pd.Series(np.where(x >= 30.0, ((x - 30.0) / 30.0).clip(0.0, 1.0), 0.0), index=feat.index)
    appear = p_play + p60
    pos = feat["position"].astype(str)
    goal_pts = pos.map(GOAL_PTS).fillna(4.0).astype(float)
    cs_pts = pos.map(CS_PTS).fillna(0.0).astype(float)
    scale = pd.to_numeric(feat.get("fwd_goal_scale", 1.0), errors="coerce").fillna(1.0)
    share_g = pd.to_numeric(feat["share_xG"], errors="coerce").fillna(0.0)
    share_a = pd.to_numeric(feat["share_xA"], errors="coerce").fillna(0.0)
    lam = pd.to_numeric(feat["lam_scored"], errors="coerce").fillna(0.0)
    lam_a = pd.to_numeric(feat["lam_assist"], errors="coerce").fillna(0.0)
    xp_goals = p_play * share_g * lam * goal_pts * scale
    xp_assists = p_play * share_a * lam_a * 3.0
    xp_cs = p60 * pd.to_numeric(feat["p_cs_mkt"], errors="coerce").fillna(0.0) * cs_pts
    defcon = pd.to_numeric(feat.get("exp_defcon_hit", 0.0), errors="coerce").fillna(0.0)
    xp_defcon = pd.Series(
        np.where(pos.isin(["DEF", "MID", "FWD"]), p60 * defcon * 2.0, 0.0),
        index=feat.index,
    )
    lam_against = pd.to_numeric(feat.get("lam_conceded"), errors="coerce")
    if lam_against.isna().all() and "e_total" in feat.columns:
        lam_against = (
            pd.to_numeric(feat["e_total"], errors="coerce") - lam
        ).clip(0.0, 5.0)
    lam_against = lam_against.fillna(0.0).clip(0.0, 5.0)
    xp_saves = pd.Series(
        np.where(pos.eq("GKP"), p60 * (lam_against * SAVES_PER_GC / 3.0), 0.0),
        index=feat.index,
    )
    xp_bps = 0.18 * xp_goals + 0.12 * xp_assists + 0.08 * xp_cs
    xp_gc = pd.Series(
        np.where(pos.isin(["GKP", "DEF"]), p60 * (lam_against / 2.0), 0.0),
        index=feat.index,
    )
    xp_card = (x.fillna(0.0) / 90.0) * 0.15
    score = appear + xp_goals + xp_assists + xp_cs + xp_defcon + xp_saves + xp_bps - xp_gc - xp_card
    return score.where(x.notna())


def attach_scheduled_score(feat: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Add ``xmi_sched`` and ``score_xp_sched``. ``score_xp`` is left as it is."""
    out = feat.copy()
    out["xmi_sched"] = xmi_sched_for(out, minutes)
    out["score_xp_sched"] = score_xp_sched(out, out["xmi_sched"])
    return out
