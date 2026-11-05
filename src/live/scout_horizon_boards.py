"""Scout minutes across L4 priced near + Opta far centreline boards.

``protocol.l3_centreline_forecast`` live path: steps 0–1 from Odds pots;
steps 2…H−1 from Opta blind λ. When a scout packet is present, each step
uses ``scout.step_map(step)`` for appear/CS minutes and freezes shares from
expanding ``baseline_m``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import SEASON_E0
from src.markets import numagent_xp
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    match_prediction,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import build_frozen_side_table, moments_from_lams
from src.strategy.layer4_mupost_priced import _align_to_week0
from src.teams import norm_team

DEFAULT_CENTRELINE_HORIZON = 12  # t … t+11
BOOKIE_NEAR = 2


def opta_e0(season: str) -> str:
    if season in SEASON_E0:
        return SEASON_E0[season]
    y0, y1 = season.split("-")
    return f"E0_{y0[-2:]}{y1[-2:]}.csv"


def overlay_minutes(
    side: pd.DataFrame, m_map: Mapping[int, float] | None
) -> pd.DataFrame:
    """Copy ``side`` and replace ``m`` from ``m_map`` (fallback to existing m)."""
    if side.empty or not m_map:
        return side.copy()
    out = side.copy()
    ms: list[float] = []
    for eid, cur in zip(
        pd.to_numeric(out["element"], errors="coerce").fillna(-1).astype(int),
        pd.to_numeric(out["m"], errors="coerce").fillna(0.0),
    ):
        ms.append(float(m_map[eid]) if int(eid) in m_map else float(cur))
    out["m"] = ms
    return out


def _club_slice(side: pd.DataFrame, team: str, team_n: str) -> pd.DataFrame:
    club = side.loc[side["team"].astype(str) == team].copy()
    if club.empty and "team_n" in side.columns:
        club = side.loc[side["team_n"] == norm_team(team_n)].copy()
    if club.empty:
        club = side.loc[
            side["team"].map(lambda t: norm_team(str(t)) == norm_team(team_n))
        ].copy()
    return club


def opta_fits_from_fixture_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    date_iso: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for home, away in pairs:
        h = norm_team(home)
        a = norm_team(away)
        if h not in ranks or a not in ranks:
            continue
        pred = match_prediction(
            h,
            a,
            ranks=ranks,
            profile=profile,
            league_gf=float(league_gf),
            sd_league_gf=0.0,
            venue_mode="blind",
        )
        mom = moments_from_lams(float(pred["lam_home"]), float(pred["lam_away"]), 0.0)
        out.append(
            {
                "fixture": -1,
                "date_iso": date_iso,
                "home": home,
                "away": away,
                "home_n": h,
                "away_n": a,
                "xg_home": float(mom["xg_home"]),
                "xcs_home": float(mom["xcs_home"]),
                "xgc_home": float(mom["xgc_home"]),
                "xg_away": float(mom["xg_away"]),
                "xcs_away": float(mom["xcs_away"]),
                "xgc_away": float(mom["xgc_away"]),
                "source": "opta_centreline_blind",
            }
        )
    return out


def score_opta_fits_on_side(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    share_side: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if side.empty or not fits:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for fit in fits:
        for team, team_n, prefix in (
            (str(fit["home"]), str(fit["home_n"]), "home"),
            (str(fit["away"]), str(fit["away_n"]), "away"),
        ):
            club = _club_slice(side, team, str(team_n))
            if club.empty:
                continue
            share_players = None
            if share_side is not None and not share_side.empty:
                share_club = _club_slice(share_side, team, str(team_n))
                if not share_club.empty:
                    share_players = share_club
            scored = numagent_xp.score_team_side(
                club,
                xg=float(fit[f"xg_{prefix}"]),
                xcs=float(fit[f"xcs_{prefix}"]),
                xgc=float(fit[f"xgc_{prefix}"]),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
                share_players=share_players,
            ).copy()
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["club"] = team
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    out = (
        pd.concat(parts, ignore_index=True)
        .groupby("element", as_index=False)
        .agg(
            {
                **{c: "sum" for c in ("xp", "xp_se") if c in parts[0].columns},
                "name": "first",
                "position": "first",
                "team": "first",
                "club": "first",
            }
        )
    )
    out["score_xp_v2"] = out["xp"]
    return out


def extend_pools_opta_far_with_scout(
    pools: list[pd.DataFrame],
    *,
    decision_gw: int,
    week0_pool: pd.DataFrame,
    hist: pd.DataFrame,
    teams: Mapping[int, str],
    fetch_fixtures,
    horizon: int = DEFAULT_CENTRELINE_HORIZON,
    season: str = "2026-27",
    alpha_g: float = 0.96,
    alpha_a: float = 0.90,
    scout_step_maps: Mapping[int, Mapping[int, float]] | None = None,
    share_baseline_m: Mapping[int, float] | None = None,
    near_depth: int = BOOKIE_NEAR,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Append Opta-blind boards for steps ``near_depth … horizon-1``.

    ``scout_step_maps[step]`` optional element→m for that horizon step.
    ``share_baseline_m`` freezes sG/sA (expanding) when provided.
    ``fetch_fixtures(gw) -> list[fixture dict]`` injects the FPL fixtures call.
    """
    H = max(len(pools), int(horizon))
    if len(pools) >= H:
        return pools, {
            "protocol": "l3_centreline_forecast",
            "horizon": len(pools),
            "opta_gws": [],
            "far": "none",
        }

    side, alpha_g, alpha_a = build_frozen_side_table(
        hist,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        season=season,
        historical_rate_baseline=True,
    )
    if side.empty:
        return pools, {
            "protocol": "l3_centreline_forecast",
            "horizon": len(pools),
            "opta_gws": [],
            "error": "empty_frozen_side",
            "fallback": "priced_only",
        }

    share_side = None
    if share_baseline_m is not None:
        share_side = overlay_minutes(side, dict(share_baseline_m))
        share_side = share_side.loc[
            pd.to_numeric(share_side["m"], errors="coerce").fillna(0.0) > 0.0
        ].copy()

    e0 = opta_e0(season)
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = float(league_avg_gf(profile))

    out_pools = list(pools)
    opta_gws: list[int] = []
    scout_steps_applied: list[int] = []
    for step in range(int(near_depth), H):
        gw_f = int(decision_gw) + int(step)
        fx_f = fetch_fixtures(gw_f) or []
        if not fx_f:
            break
        pairs_f = [
            (teams[int(f["team_h"])], teams[int(f["team_a"])])
            for f in fx_f
            if f.get("team_h") is not None and f.get("team_a") is not None
        ]
        fits = opta_fits_from_fixture_pairs(
            pairs_f, ranks=ranks, profile=profile, league_gf=league_gf
        )
        m_map = None
        if scout_step_maps is not None and int(step) in scout_step_maps:
            m_map = scout_step_maps[int(step)]
            scout_steps_applied.append(int(step))
        side_h = overlay_minutes(side, m_map)
        raw = score_opta_fits_on_side(
            side_h,
            fits,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            share_side=share_side,
        )
        if raw.empty:
            break
        raw = raw.copy()
        raw["player_id"] = raw["element"].astype(str)
        raw["mu_post"] = raw["score_xp_v2"]
        if "xp_se" not in raw.columns:
            raw["xp_se"] = 0.0
        out_pools.append(_align_to_week0(week0_pool, raw))
        opta_gws.append(gw_f)

    return out_pools, {
        "protocol": "l3_centreline_forecast",
        "horizon": len(out_pools),
        "opta_gws": opta_gws,
        "opta_e0": e0,
        "near": "priced_odds_t_t1",
        "far": "opta_blind",
        "scout_steps_applied": scout_steps_applied,
        "share_freeze": share_baseline_m is not None,
    }
