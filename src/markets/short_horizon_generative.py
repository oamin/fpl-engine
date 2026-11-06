"""Generative attack scoring for priced short-horizon t:t+1.

Per fixture path:

1. ``k ∼ M`` (``match_outcome_roll`` from priced / FD λ).
2. Lineup world: frozen as-of mean minutes by default (``lineup_cov=False``);
   matchday/pair rolls parked.
3. Scorer / assist dice (``π`` props-shape or shares; ``ρ`` assists; α empties).
4. Defence from the **rolled** scoreline (``defence_roll``: CS if GA=0;
   GC = ⌊GA/2⌋) — no separate xCS / xGC moments on scenario strands.
5. Soft priors from **expanding means** (not the roll): DefCon, bonus, cards,
   saves. Bonus-from-roll is parked as overkill for now.

Mean-layer boards stay on ``score_team_side``; cascade_short **scenario**
strands use this path when ``generative_attack=True`` (default).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.markets.attack_roll import build_side_dice, roll_side_attack
from src.markets.defence_roll import xp_defence_on_roll
from src.markets.lineup_roll import (
    START_GATE,
    LineupWorld,
    minutes_corr_pairs,
    roll_club_gw_lineup,
    roll_lineup_world,
    select_disjoint_pairs,
)
from src.markets.match_outcome import (
    match_outcome_probs_from_lambdas,
    match_outcome_roll,
)
from src.markets.numagent_xp import SIG_FLOOR, defcon_xp, saves_xp
from src.rules.fpl_2026 import (
    APPEARANCE_60,
    APPEARANCE_UNDER_60,
    ASSIST_POINTS,
    DEFCON_THRESHOLD,
    GOAL_POINTS,
    MINUTES_FOR_APPEARANCE_BONUS,
    normalize_position,
)


def _appearance_xp(m: float) -> float:
    if m <= 0:
        return 0.0
    if m < MINUTES_FOR_APPEARANCE_BONUS:
        return float(APPEARANCE_UNDER_60)
    return float(APPEARANCE_60)


def _name_col(frame: pd.DataFrame) -> str:
    if "name" in frame.columns:
        return "name"
    if "player_name" in frame.columns:
        return "player_name"
    raise ValueError("club frame needs name or player_name")


def _id_col(frame: pd.DataFrame) -> str:
    if "element" in frame.columns:
        return "element"
    if "player_id" in frame.columns:
        return "player_id"
    raise ValueError("club frame needs element or player_id")


def trivial_lineup(mean_m: np.ndarray | Sequence[float], *, start_gate: float = START_GATE) -> LineupWorld:
    """Mean minutes as a fixed lineup world (no competitor draws)."""
    minutes = np.asarray(mean_m, dtype=float).copy()
    return LineupWorld(
        minutes=minutes,
        eligible=minutes >= float(start_gate),
        pair_choices=(),
    )


def build_club_dice_from_side(
    club: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
    G_team: float,
    prop_p_by_element: Mapping[int, float] | None = None,
) -> tuple[Any, pd.DataFrame]:
    """Build ``SideAttackDice`` + ordered club frame (playing m>0)."""
    c = club.copy()
    if "m" not in c.columns and "xmi" in c.columns:
        c["m"] = pd.to_numeric(c["xmi"], errors="coerce").fillna(0.0)
    c["m"] = pd.to_numeric(c["m"], errors="coerce").fillna(0.0)
    c = c.loc[c["m"] > 0].reset_index(drop=True)
    if c.empty:
        return None, c
    for col, default in (("rG", 0.0), ("rA", 0.0), ("drG", 0.0), ("drA", 0.0)):
        if col not in c.columns:
            c[col] = default
    if "sG" not in c.columns or "sA" not in c.columns:
        from src.markets import numagent_xp as nx

        wG = np.array(
            [
                nx.weight_and_delta(
                    float(c.iloc[i].get("rG", 0.0) or 0.0),
                    float(c.iloc[i].get("drG", 0.0) or 0.0),
                    float(c.iloc[i]["m"]),
                    float(c.iloc[i].get("sigma_m", 0.0) or 0.0),
                )[0]
                for i in range(len(c))
            ]
        )
        wA = np.array(
            [
                nx.weight_and_delta(
                    float(c.iloc[i].get("rA", 0.0) or 0.0),
                    float(c.iloc[i].get("drA", 0.0) or 0.0),
                    float(c.iloc[i]["m"]),
                    float(c.iloc[i].get("sigma_m", 0.0) or 0.0),
                )[0]
                for i in range(len(c))
            ]
        )
        sG, _ = nx.shares_and_deltas(wG, np.zeros_like(wG))
        sA, _ = nx.shares_and_deltas(wA, np.zeros_like(wA))
        c["sG"] = sG
        c["sA"] = sA

    prop_p: dict[int, float] | None = None
    if prop_p_by_element:
        prop_p = {}
        idc = _id_col(c)
        for i, raw in enumerate(c[idc].tolist()):
            key = int(str(raw).split(":")[-1])
            if key in prop_p_by_element:
                prop_p[i] = float(prop_p_by_element[key])
        if not prop_p:
            prop_p = None

    names = c[_name_col(c)].astype(str).tolist()
    dice = build_side_dice(
        names,
        c["sG"].to_numpy(float),
        c["sA"].to_numpy(float),
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        G_team=G_team,
        prop_p=prop_p,
    )
    return dice, c


def competitor_pairs_for_club(
    hist: pd.DataFrame | None,
    club: pd.DataFrame,
) -> list:
    if hist is None or hist.empty or club.empty:
        return []
    from src.markets.lineup_roll import minutes_panel

    idc = _id_col(club)
    name_h = "player_name" if "player_name" in hist.columns else "name"
    id_h = "element" if "element" in hist.columns else ("player_id" if "player_id" in hist.columns else idc)
    h = hist.copy()
    if "player_name" not in h.columns and "name" in h.columns:
        h = h.rename(columns={"name": "player_name"})
        name_h = "player_name"
    gw_col = "GW" if "GW" in h.columns else ("gw" if "gw" in h.columns else None)
    if gw_col is None:
        return []
    if gw_col != "gw":
        h = h.rename(columns={gw_col: "gw"})
    table = minutes_corr_pairs(h, id_col=id_h, name_col=name_h)
    panel = minutes_panel(h, id_col=id_h, gw_col="gw")
    ids = club[idc].astype(int).tolist()
    return select_disjoint_pairs(ids, table, panel=panel)


def xp_soft_priors_expanding(
    club: pd.DataFrame,
    *,
    positions: Sequence[str],
    minutes: np.ndarray,
) -> dict[str, np.ndarray]:
    """DefCon / bonus / cards / saves from expanding means (not the roll).

    Same priors as ``score_team_side``: ``*_mu`` columns on the as-of side
    table, gated by the path's minutes world. Bonus-from-scoreline is parked.
    """
    n = len(minutes)
    if club.empty or n == 0:
        z = np.zeros(0, dtype=float)
        return {
            "xp_defcon": z,
            "xp_bonus": z,
            "xp_cards": z,
            "xp_saves": z,
        }

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name in club.columns:
            return pd.to_numeric(club[name], errors="coerce").fillna(default).to_numpy(float)
        return np.full(n, default)

    playing = minutes > 0
    xp_bonus = np.where(playing, col("bonus_mu"), 0.0)
    xp_cards = np.where(playing, col("card_mu"), 0.0)

    dc_mu = col("defcon_mu")
    dc_sig = col("defcon_sig", SIG_FLOOR)
    dc_n = col("defcon_n").astype(int)
    xp_defcon = np.zeros(n)
    for i, p in enumerate(positions):
        tau = float(DEFCON_THRESHOLD.get(p, 0) or 0)
        xp_defcon[i], _, _ = defcon_xp(
            dc_mu[i], dc_sig[i], int(dc_n[i]), tau, float(minutes[i])
        )

    saves_mu = col("saves_mu")
    saves_se = col("saves_se")
    sigma_m = col("sigma_m", 0.0)
    xp_saves = np.zeros(n)
    for i, p in enumerate(positions):
        xp_saves[i], _ = saves_xp(
            saves_mu[i],
            saves_se[i],
            float(minutes[i]),
            float(sigma_m[i]),
            position=str(p),
        )
    return {
        "xp_defcon": xp_defcon,
        "xp_bonus": xp_bonus,
        "xp_cards": xp_cards,
        "xp_saves": xp_saves,
    }


def xp_board_from_side_roll(
    club: pd.DataFrame,
    *,
    attack,
    lineup: LineupWorld,
    goals_against: int,
) -> pd.DataFrame:
    """Map one side's generative draws onto per-player xP components.

    Roll-dependent: goals, assists, CS, GC (and appearance from minutes world).
    Expanding-mean priors: DefCon, bonus, cards, saves.
    """
    if club.empty:
        return club.copy()
    out = club.copy()
    n = len(out)
    name_col = _name_col(out)
    names = out[name_col].astype(str).tolist()
    pos = [
        normalize_position("GKP" if str(p) == "GK" else str(p))
        for p in out["position"].tolist()
    ]
    minutes = np.asarray(lineup.minutes, dtype=float)
    goals = Counter(attack.scorers)
    assists = Counter(attack.assisters)

    xp_appear = np.array([_appearance_xp(float(m)) for m in minutes])
    xp_goals = np.zeros(n)
    xp_assists = np.zeros(n)
    for i, name in enumerate(names):
        g = float(goals.get(name, 0))
        a = float(assists.get(name, 0))
        xp_goals[i] = g * float(GOAL_POINTS[pos[i]])
        xp_assists[i] = a * float(ASSIST_POINTS)

    # CS / GC are the rolled scoreline (opp goals), not moment xCS / xGC.
    xp_cs, xp_gc, defence = xp_defence_on_roll(pos, minutes, goals_against)
    soft = xp_soft_priors_expanding(out, positions=pos, minutes=minutes)
    xp_defcon = soft["xp_defcon"]
    xp_bonus = soft["xp_bonus"]
    xp_cards = soft["xp_cards"]
    xp_saves = soft["xp_saves"]

    xp = (
        xp_appear
        + xp_goals
        + xp_assists
        + xp_cs
        + xp_gc
        + xp_defcon
        + xp_bonus
        + xp_cards
        + xp_saves
    )
    out["m"] = minutes
    out["xp_appear"] = xp_appear
    out["xp_goals"] = xp_goals
    out["xp_assists"] = xp_assists
    out["xp_cs"] = xp_cs
    out["xp_gc"] = xp_gc
    out["xp_defcon"] = xp_defcon
    out["xp_bonus"] = xp_bonus
    out["xp_cards"] = xp_cards
    out["xp_saves"] = xp_saves
    out["xp"] = xp
    out["score_xp_v2"] = xp
    out["xp_se"] = 0.0
    out["goals_against"] = int(defence.goals_against)
    out["clean_sheet"] = bool(defence.clean_sheet)
    out["gc_bins"] = int(defence.gc_bins)
    out["E_G"] = np.array([float(goals.get(nm, 0)) for nm in names])
    out["E_A"] = np.array([float(assists.get(nm, 0)) for nm in names])
    return out


def _roll_or_trivial_lineup(
    club: pd.DataFrame,
    *,
    hist: pd.DataFrame | None,
    rng: Any,
    lineup_cov: bool,
) -> LineupWorld:
    mean_m = club["m"].to_numpy(float)
    if not lineup_cov:
        return trivial_lineup(mean_m)
    # Club matchday resample — full GW minutes vector, no corr threshold.
    return roll_club_gw_lineup(club, hist, rng, id_col_club=_id_col(club))


def score_fixture_generative(
    home_club: pd.DataFrame,
    away_club: pd.DataFrame,
    *,
    lam_h: float,
    lam_a: float,
    rho: float,
    alpha_g: float,
    alpha_a: float,
    rng: Any,
    home_hist: pd.DataFrame | None = None,
    away_hist: pd.DataFrame | None = None,
    home_props: Mapping[int, float] | None = None,
    away_props: Mapping[int, float] | None = None,
    lineup_cov: bool = True,
    goals_home: int | None = None,
    goals_away: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """One generative path for both sides of a fixture.

    Scoreline: use ``goals_home`` / ``goals_away`` when already rolled
    (cascade_long Opta path); otherwise ``k ∼ M`` from λ. Props optional —
    cascade_long always passes ``None`` (unavailable beyond priced weeks).
    """
    probs = match_outcome_probs_from_lambdas(lam_h, lam_a, rho)
    if goals_home is None or goals_away is None:
        roll = match_outcome_roll(probs, rng)
        gh, ga = int(roll.goals_home), int(roll.goals_away)
        roll_p = float(roll.p)
    else:
        gh, ga = int(goals_home), int(goals_away)
        n = probs.matrix.shape[0]
        i, j = min(gh, n - 1), min(ga, n - 1)
        mass = float(probs.matrix.sum())
        roll_p = float(probs.matrix[i, j] / mass) if mass > 0 else 0.0
    g_home = float(alpha_g) * float(probs.moments["xg_home"])
    g_away = float(alpha_g) * float(probs.moments["xg_away"])

    home_dice, home_c = build_club_dice_from_side(
        home_club, alpha_g=alpha_g, alpha_a=alpha_a, G_team=g_home, prop_p_by_element=home_props
    )
    away_dice, away_c = build_club_dice_from_side(
        away_club, alpha_g=alpha_g, alpha_a=alpha_a, G_team=g_away, prop_p_by_element=away_props
    )

    empty = home_club.iloc[0:0].copy()
    if home_dice is None or home_c.empty:
        home_xp = empty
        home_att = None
    else:
        home_lineup = _roll_or_trivial_lineup(
            home_c, hist=home_hist, rng=rng, lineup_cov=lineup_cov
        )
        home_att = roll_side_attack(
            home_dice, gh, rng, team="home", lineup=home_lineup
        )
        home_xp = xp_board_from_side_roll(
            home_c, attack=home_att, lineup=home_lineup, goals_against=ga
        )

    if away_dice is None or away_c.empty:
        away_xp = empty
        away_att = None
    else:
        away_lineup = _roll_or_trivial_lineup(
            away_c, hist=away_hist, rng=rng, lineup_cov=lineup_cov
        )
        away_att = roll_side_attack(
            away_dice, ga, rng, team="away", lineup=away_lineup
        )
        away_xp = xp_board_from_side_roll(
            away_c, attack=away_att, lineup=away_lineup, goals_against=gh
        )

    meta = {
        "goals_home": gh,
        "goals_away": ga,
        "p": roll_p,
        "lam_h": float(lam_h),
        "lam_a": float(lam_a),
        "home_cs": ga == 0,
        "away_cs": gh == 0,
        "home_ga": ga,
        "away_ga": gh,
        "n_home_scorers": 0 if home_att is None else len(home_att.scorers),
        "n_away_scorers": 0 if away_att is None else len(away_att.scorers),
    }
    return home_xp, away_xp, meta


def hist_by_team_norm(hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split as-of history by ``team_n`` for lineup-corr lookups."""
    if hist is None or hist.empty:
        return {}
    h = hist.copy()
    if "team_n" not in h.columns and "team" in h.columns:
        from src.teams import norm_team

        h["team_n"] = h["team"].map(norm_team)
    if "team_n" not in h.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for team_n, block in h.groupby("team_n", sort=False):
        out[str(team_n)] = block
    return out


def score_week_generative(
    side_table: pd.DataFrame,
    fixture_fits: Sequence[Mapping[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    rng: Any,
    hist_by_team: Mapping[str, pd.DataFrame] | None = None,
    prop_by_element: Mapping[int, float] | None = None,
    lineup_cov: bool = True,
    lam_overrides: Mapping[int, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Score a full GW by generative rolls on each priced fixture."""
    if side_table.empty or not fixture_fits:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for fit in fixture_fits:
        fx_id = int(fit["fixture"])
        ov = (lam_overrides or {}).get(fx_id)
        if ov is not None:
            lam_h = float(ov.get("lam_h", fit["lam_h"]))
            lam_a = float(ov.get("lam_a", fit["lam_a"]))
        else:
            lam_h = float(fit["lam_h"])
            lam_a = float(fit["lam_a"])
        rho = float(fit.get("rho", 0.0) or 0.0)
        home_n = str(fit["home_n"])
        away_n = str(fit["away_n"])
        home = side_table.loc[side_table["team"].astype(str) == str(fit["home"])].copy()
        if home.empty and "team_n" in side_table.columns:
            home = side_table.loc[side_table["team_n"] == home_n].copy()
        away = side_table.loc[side_table["team"].astype(str) == str(fit["away"])].copy()
        if away.empty and "team_n" in side_table.columns:
            away = side_table.loc[side_table["team_n"] == away_n].copy()
        h_hist = (hist_by_team or {}).get(home_n)
        a_hist = (hist_by_team or {}).get(away_n)
        home_xp, away_xp, _meta = score_fixture_generative(
            home,
            away,
            lam_h=lam_h,
            lam_a=lam_a,
            rho=rho,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            rng=rng,
            home_hist=h_hist,
            away_hist=a_hist,
            home_props=prop_by_element,
            away_props=prop_by_element,
            lineup_cov=lineup_cov,
        )
        for scored, team in ((home_xp, str(fit["home"])), (away_xp, str(fit["away"]))):
            if scored.empty:
                continue
            scored = scored.copy()
            scored["team"] = team
            scored["club"] = team
            scored["fixture"] = fx_id
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    all_p = pd.concat(parts, ignore_index=True)
    if "element" not in all_p.columns:
        raise ValueError("generative week score needs element ids on club frames")
    name_first = "name" if "name" in all_p.columns else _name_col(all_p)
    board = (
        all_p.groupby("element", as_index=False)
        .agg(
            **{
                "name": (name_first, "first"),
                "position": ("position", "first"),
                "team": ("team", "first"),
                "club": ("club", "first"),
                "score_xp_v2": ("xp", "sum"),
                "xp_se": ("xp_se", "sum"),
            }
        )
    )
    return board
