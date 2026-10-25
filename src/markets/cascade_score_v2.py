"""Long-horizon cascade scoring on Numagent 2.0 (``score_xp_v2``).

Opta / path scenarios sample match λ and roll ``k ∼ M``. Player xP on
scenario strands matches cascade_short generative (lineup → attack dice →
CS/GC from roll → soft priors) **without** player props (unavailable on
t+2…). Mean-layer helpers remain for diagnostics / fallbacks.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.scoreline import build_matrix, team_moments
from src.markets.short_horizon_generative import score_fixture_generative
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
    # Season-start priors (player prior else positional). Live bootstrap has no
    # same-season hist here; hist paths update via ``attach_rates_priors`` /
    # ``build_frozen_side_table``.
    if "name" in frame.columns and season:
        from src.markets.rate_baseline import apply_historical_rate_baseline

        frame = apply_historical_rate_baseline(frame, season=str(season), hist=None)
    return frame.reset_index(drop=True)


def score_club_side(
    club_players: pd.DataFrame,
    *,
    lam_for: float,
    lam_against: float,
    alpha_g: float,
    alpha_a: float,
    rho: float = 0.0,
    share_players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score one club's rows for one fixture side via ``score_team_side``.

    Pass ``share_players`` (full club ``m>0``) when ``club_players`` is a thin
    selection board so xG shares are not re-normalised on ≥60′ only.
    """
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
        share_players=share_players,
    )


def score_pool_on_lam_pots(
    pool: pd.DataFrame,
    pots: Mapping[tuple[int, str], list[dict[str, float]]],
    *,
    gws: Sequence[int],
    gamma: float = 1.0,
    alpha_g: float | None = None,
    alpha_a: float | None = None,
    share_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Multi-GW ``score_xp_v2`` for MILP from scenario λ pots (γ=1 default).

    Each pot must carry ``lam_scored`` and ``threat`` (opp λ), as produced by
    ``pot_from_sampled_lam`` / path pots.

    ``share_pool`` (full club ``m>0``) is the default xG share frame when the
    selection ``pool`` is thin (≥60′).
    """
    if alpha_g is None or alpha_a is None:
        ag, aa, _, _ = numagent_xp.load_alphas()
        alpha_g = ag if alpha_g is None else alpha_g
        alpha_a = aa if alpha_a is None else alpha_a

    out = pool.copy()
    totals = np.zeros(len(out), dtype=float)
    pid_to_idx = {str(p): i for i, p in enumerate(out["player_id"].astype(str))}
    share = share_pool if share_pool is not None else out

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
            share_df = share.loc[
                share["team_norm"].map(norm_team) == norm_team(club)
            ].copy()
            scored = score_club_side(
                club_df,
                lam_for=lam,
                lam_against=max(opp, 1e-6),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
                share_players=share_df if len(share_df) > len(club_df) else None,
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


def attach_rates_priors(
    players: pd.DataFrame,
    hist: pd.DataFrame,
    *,
    id_col: str = "element",
    finish_g_col: str | None = None,
    finish_a_col: str | None = None,
    season: str | None = None,
    historical_rate_baseline: bool = True,
) -> pd.DataFrame:
    """Merge as-of rates/priors onto a player table; optional finish scales on rG/rA.

    When ``historical_rate_baseline`` is True, overwrite ``rG``/``rA`` with
    prior rates at season start (player else positional), then update with
    same-season appearance counts in ``hist`` (GW ``< t``).
    """
    rates = numagent_xp.player_rates_from_history(hist, id_col=id_col)
    priors = numagent_xp.history_priors_for_xp(hist, id_col=id_col)
    out = players.copy()
    if not rates.empty:
        out = out.merge(rates, on=id_col, how="left")
    if not priors.empty:
        out = out.merge(
            priors.drop(columns=["position"], errors="ignore"),
            on=id_col,
            how="left",
        )
    for c, default in (
        ("rG", 0.0),
        ("drG", 0.0),
        ("rA", 0.0),
        ("drA", 0.0),
        ("bonus_mu", 0.0),
        ("bonus_se", 0.0),
        ("card_mu", 0.0),
        ("card_se", 0.0),
        ("saves_mu", 0.0),
        ("saves_se", 0.0),
        ("defcon_mu", 0.0),
        ("defcon_sig", numagent_xp.SIG_FLOOR),
        ("defcon_n", 0),
    ):
        if c not in out.columns:
            out[c] = default
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)
    if finish_g_col and finish_g_col in out.columns:
        scale = pd.to_numeric(out[finish_g_col], errors="coerce").fillna(1.0)
        out["rG"] = out["rG"] * scale
        out["drG"] = out["drG"] * scale
    if finish_a_col and finish_a_col in out.columns:
        scale = pd.to_numeric(out[finish_a_col], errors="coerce").fillna(1.0)
        out["rA"] = out["rA"] * scale
        out["drA"] = out["drA"] * scale
    if historical_rate_baseline and "name" in out.columns:
        from src.markets.rate_baseline import apply_historical_rate_baseline

        season_key = season
        if season_key is None and "season" in hist.columns and len(hist):
            season_key = str(hist["season"].iloc[0]).replace("_", "-")
        if season_key:
            out = apply_historical_rate_baseline(
                out, season=str(season_key), hist=hist
            )
    return out


def score_fixture_sides_on_lams(
    home_players: pd.DataFrame,
    away_players: pd.DataFrame,
    *,
    lam_h: float,
    lam_a: float,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    """Score home and away player tables for one fixture λ pair."""
    parts = []
    if not home_players.empty:
        h = score_club_side(
            home_players,
            lam_for=float(lam_h),
            lam_against=float(lam_a),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
        )
        h["side"] = "home"
        parts.append(h)
    if not away_players.empty:
        a = score_club_side(
            away_players,
            lam_for=float(lam_a),
            lam_against=float(lam_h),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
        )
        a["side"] = "away"
        parts.append(a)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def score_pool_generative_on_scenarios(
    pool: pd.DataFrame,
    scenarios_by_gw: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    gws: Sequence[int],
    alpha_g: float,
    alpha_a: float,
    rng: Any,
    hist_by_team: Mapping[str, pd.DataFrame] | None = None,
    gamma: float = 1.0,
    lineup_cov: bool = True,
    rho: float = 0.0,
    share_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Multi-GW generative xP (cascade_short stack; γ=1 default).

    Each scenario mapping needs ``home``, ``away``, ``lam_h``, ``lam_a``, and
    preferably rolled ``hg`` / ``ag``. Optional ``home_props`` / ``away_props``
    (element id → P(anytime)) are used on priced cascade_short weeks; Opta
    long weeks leave them unset. Missing goals trigger a fresh ``k ∼ M``
    draw from the fixture λ (scratch valuation; does not update ClubPath).

    Attack dice run on ``share_pool`` (full club ``m>0``) when provided;
    scores are written back onto the thin selection ``pool``.
    """
    out = pool.copy()
    if "team_n" not in out.columns and "team_norm" in out.columns:
        out["team_n"] = out["team_norm"].map(lambda t: norm_team(str(t)))
    totals = np.zeros(len(out), dtype=float)
    pid_to_idx = {str(p): i for i, p in enumerate(out["player_id"].astype(str))}
    dice_pool = share_pool if share_pool is not None else out
    if "team_n" not in dice_pool.columns and "team_norm" in dice_pool.columns:
        dice_pool = dice_pool.copy()
        dice_pool["team_n"] = dice_pool["team_norm"].map(lambda t: norm_team(str(t)))

    for h, gw in enumerate(gws):
        disc = float(gamma) ** h
        for sc in scenarios_by_gw.get(int(gw), ()) or ():
            home = str(sc["home"])
            away = str(sc["away"])
            home_n = norm_team(home)
            away_n = norm_team(away)
            team_col = "team_norm" if "team_norm" in dice_pool.columns else "team_n"
            home_club = dice_pool.loc[
                dice_pool[team_col].map(norm_team) == home_n
            ].copy()
            away_club = dice_pool.loc[
                dice_pool[team_col].map(norm_team) == away_n
            ].copy()
            if home_club.empty and away_club.empty:
                continue
            gh = sc.get("hg", sc.get("goals_home"))
            ga = sc.get("ag", sc.get("goals_away"))
            home_props = sc.get("home_props")
            away_props = sc.get("away_props")
            home_xp, away_xp, _meta = score_fixture_generative(
                home_club,
                away_club,
                lam_h=float(sc["lam_h"]),
                lam_a=float(sc["lam_a"]),
                rho=float(sc.get("rho", rho) or 0.0),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
                rng=rng,
                home_hist=(hist_by_team or {}).get(home_n),
                away_hist=(hist_by_team or {}).get(away_n),
                home_props=home_props if isinstance(home_props, Mapping) else None,
                away_props=away_props if isinstance(away_props, Mapping) else None,
                lineup_cov=lineup_cov,
                goals_home=None if gh is None else int(gh),
                goals_away=None if ga is None else int(ga),
            )
            for scored in (home_xp, away_xp):
                if scored is None or scored.empty:
                    continue
                for i in range(len(scored)):
                    if "player_id" in scored.columns:
                        pid = str(scored.iloc[i]["player_id"])
                    else:
                        pid = str(int(scored.iloc[i]["element"]))
                    idx = pid_to_idx.get(pid)
                    if idx is None:
                        continue
                    totals[idx] += disc * float(scored.iloc[i]["xp"])
    out["mc_score"] = totals
    out[SCORER] = totals
    return out
