"""Long-horizon cascade scoring on Numagent 2.0 (``score_xp_v2``).

Opta / path scenarios still sample match λ; player xP uses the locked
assembler ``numagent_xp.score_team_side`` with same-season as-of rates —
not ``xp_on_pot`` / ``add_player_priors`` shares.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.scoreline import build_matrix, team_moments
from src.teams import norm_team

SCORER = numagent_xp.SCORE_NAME
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def side_moments_from_lams(
    lam_for: float,
    lam_against: float,
    *,
    rho: float = 0.0,
) -> dict[str, float]:
    """Dixon–Coles side moments treating ``lam_for`` as the home index."""
    mat = build_matrix(float(max(lam_for, 1e-6)), float(max(lam_against, 1e-6)), float(rho))
    mom = team_moments(mat)
    return {
        "xg": float(mom["xg_home"]),
        "xcs": float(mom["xcs_home"]),
        "xgc": float(mom["xgc_home"]),
        "lam_for": float(lam_for),
        "lam_against": float(lam_against),
    }


def normalize_live_history(logs: pd.DataFrame) -> pd.DataFrame:
    """Map live ``player_gw`` / log columns onto numagent history names."""
    h = logs.copy()
    rename = {}
    if "element" not in h.columns and "player_id" in h.columns:
        h["element"] = pd.to_numeric(h["player_id"], errors="coerce")
    if "GW" not in h.columns and "gw" in h.columns:
        rename["gw"] = "GW"
    if "goals_scored" not in h.columns and "goals" in h.columns:
        rename["goals"] = "goals_scored"
    if "defensive_contribution" not in h.columns and "defcon" in h.columns:
        rename["defcon"] = "defensive_contribution"
    if "name" not in h.columns and "player_name" in h.columns:
        rename["player_name"] = "name"
    if rename:
        h = h.rename(columns=rename)
    h["element"] = pd.to_numeric(h["element"], errors="coerce")
    h["GW"] = pd.to_numeric(h["GW"], errors="coerce")
    h = h.dropna(subset=["element", "GW"]).copy()
    h["element"] = h["element"].astype(int)
    h["GW"] = h["GW"].astype(int)
    if "team" in h.columns:
        h["team_n"] = h["team"].map(lambda t: norm_team(str(t)))
    elif "team_norm" in h.columns:
        h["team_n"] = h["team_norm"].map(lambda t: norm_team(str(t)))
        h["team"] = h.get("team", h["team_norm"])
    for col, default in (
        ("minutes", 0.0),
        ("goals_scored", 0.0),
        ("assists", 0.0),
        ("bonus", 0.0),
        ("yellow_cards", 0.0),
        ("red_cards", 0.0),
        ("defensive_contribution", 0.0),
        ("saves", 0.0),
    ):
        if col not in h.columns:
            h[col] = default
        else:
            h[col] = pd.to_numeric(h[col], errors="coerce").fillna(default)
    if "position" in h.columns:
        h["position"] = h["position"].astype(str)
    if "season" not in h.columns:
        h["season"] = "2026-27"
    return h


def rolling_m_sigma(hist: pd.DataFrame, element: int, *, window: int = 3) -> tuple[float, float]:
    past = hist.loc[hist["element"] == int(element)]
    if past.empty:
        return 0.0, numagent_xp.SIG_FLOOR
    by_gw = (
        past.sort_values("GW")
        .groupby("GW", sort=True)["minutes"]
        .max()
        .tail(int(window))
    )
    vals = by_gw.to_numpy(float)
    if vals.size == 0:
        return 0.0, numagent_xp.SIG_FLOOR
    m = float(vals.mean())
    if vals.size >= 2:
        sig = float(max(np.std(vals, ddof=1), numagent_xp.SIG_FLOOR))
    else:
        sig = float(max(10.0, numagent_xp.SIG_FLOOR))
    return m, sig


def build_frozen_pool(
    bootstrap: Mapping[str, Any],
    selectable_ids: set[str],
    history: pd.DataFrame,
    *,
    before_gw: int,
    season: str = "2026-27",
) -> pd.DataFrame:
    """Selectable candidates with frozen ``score_xp_v2`` rate/prior columns."""
    hist = numagent_xp.asof_share_history(
        normalize_live_history(history), before_gw=int(before_gw), season=season
    )
    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    rows: list[dict[str, Any]] = []
    for el in bootstrap["elements"]:
        pid = str(el["id"])
        if pid not in selectable_ids:
            continue
        eid = int(el["id"])
        club = teams[int(el["team"])]
        pos = POS_MAP.get(int(el["element_type"]), "MID")
        m, sig = rolling_m_sigma(hist, eid)
        rows.append(
            {
                "player_id": pid,
                "element": eid,
                "name": el.get("web_name"),
                "position": pos,
                "team": club,
                "team_norm": norm_team(club),
                "club": club,
                "value": int(el.get("now_cost") or 50),
                "m": float(min(90.0, max(0.0, m))),
                "sigma_m": float(sig),
                "eligible": True,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if not rates.empty:
        frame = frame.merge(rates, on="element", how="left")
    if not priors.empty:
        frame = frame.merge(
            priors.drop(columns=["position"], errors="ignore"),
            on="element",
            how="left",
        )
    for c, default in (
        ("rG", 0.0),
        ("drG", 0.0),
        ("rA", 0.0),
        ("drA", 0.0),
        ("bonus_mu", 0.2),
        ("bonus_se", 0.1),
        ("card_mu", -0.1),
        ("card_se", 0.05),
        ("saves_mu", 0.0),
        ("saves_se", 0.0),
        ("defcon_mu", 0.0),
        ("defcon_sig", numagent_xp.SIG_FLOOR),
        ("defcon_n", 5),
    ):
        if c not in frame.columns:
            frame[c] = default
        else:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").fillna(default)
    frame["fwd_goal_scale"] = 1.0
    # Legacy columns some callers still read for overlays / reports.
    frame["xmi"] = frame["m"]
    frame["share_xG"] = 0.0
    frame["share_xA"] = 0.0
    frame["exp_defcon_hit"] = 0.0
    return frame.reset_index(drop=True)


def score_club_side(
    club_players: pd.DataFrame,
    *,
    lam_for: float,
    lam_against: float,
    alpha_g: float,
    alpha_a: float,
    rho: float = 0.0,
) -> pd.DataFrame:
    """Score one club's rows for one fixture side via ``score_team_side``."""
    if club_players.empty:
        return club_players.copy()
    mom = side_moments_from_lams(lam_for, lam_against, rho=rho)
    return numagent_xp.score_team_side(
        club_players,
        xg=mom["xg"],
        xcs=mom["xcs"],
        xgc=mom["xgc"],
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
    )


def score_pool_on_lam_pots(
    pool: pd.DataFrame,
    pots: Mapping[tuple[int, str], list[dict[str, float]]],
    *,
    gws: Sequence[int],
    gamma: float = 0.9,
    alpha_g: float | None = None,
    alpha_a: float | None = None,
) -> pd.DataFrame:
    """Discounted multi-GW ``score_xp_v2`` for MILP from scenario λ pots.

    Each pot must carry ``lam_scored`` and ``threat`` (opp λ), as produced by
    ``pot_from_sampled_lam`` / path pots.
    """
    if alpha_g is None or alpha_a is None:
        ag, aa, _, _ = numagent_xp.load_alphas()
        alpha_g = ag if alpha_g is None else alpha_g
        alpha_a = aa if alpha_a is None else alpha_a

    out = pool.copy()
    totals = np.zeros(len(out), dtype=float)
    pid_to_idx = {str(p): i for i, p in enumerate(out["player_id"].astype(str))}

    for h, gw in enumerate(gws):
        # Collect clubs that have a pot this GW and score each club once.
        clubs = []
        for (g, club), lst in pots.items():
            if int(g) != int(gw) or not lst:
                continue
            clubs.append((str(club), dict(lst[0])))
        disc = float(gamma) ** h
        for club, pot in clubs:
            lam = float(pot.get("lam_scored") or pot.get("attack") or 0.0)
            opp = float(pot.get("threat") or pot.get("lam_opp") or 0.0)
            if lam <= 0 and opp <= 0:
                continue
            mask = out["team_norm"].map(norm_team) == norm_team(club)
            club_df = out.loc[mask].copy()
            if club_df.empty:
                continue
            scored = score_club_side(
                club_df,
                lam_for=lam,
                lam_against=max(opp, 1e-6),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
            )
            for i in range(len(scored)):
                pid = str(club_df.iloc[i]["player_id"])
                idx = pid_to_idx.get(pid)
                if idx is None:
                    continue
                totals[idx] += disc * float(scored.iloc[i]["xp"])
    out["mc_score"] = totals
    out[SCORER] = totals
    return out
