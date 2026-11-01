"""Locked L3 centreline horizon boards for the strategiser.

``protocol.l3_centreline_forecast`` (2026-10-30):

* steps ``0..near-1`` (default 2): priced / FD centreline → ``score_team_side``
* steps ``near..H-1``: Opta blind place λ → same ``score_team_side``

Feeds ``pools_from_mu_boards`` / ``solve_horizon_transfers``. Generative
MC-mean is not used.

  from src.strategy.l3_centreline_horizon import build_l3_centreline_horizon
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import SEASON_E0
from src.eval.l3_opta_lambda_horizon import _opta_fits_for_gw
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    match_prediction,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import build_frozen_side_table, moments_from_lams
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer4_bookie_path import BOOKIE_DEPTH, _fd_index, _load_merged_enriched
from src.strategy.layer4_mupost_priced import (
    _align_to_week0,
    _attach_rates_priors_baseline,
    _attach_store_ids,
    _mean_minutes_asof,
    build_mupost_priced_horizon,
)
from src.strategy.shrinkage import CohortMoments
from src.teams import norm_team

# Locked product defaults (protocol.l3_centreline_forecast).
DEFAULT_HORIZON = 12  # t … t+11 (priced near + Opta far)
NEAR_DEPTH = int(BOOKIE_DEPTH)  # t, t+1 priced / FD


@dataclass(frozen=True)
class L3CentrelineHorizon:
    season: str
    decision_gw: int
    pools: list[pd.DataFrame]
    priced_gws: tuple[int, ...]
    opta_gws: tuple[int, ...]
    n_projected_fixtures: int
    meta: dict[str, Any]


def _opta_e0(season: str) -> str:
    if season in SEASON_E0:
        return SEASON_E0[season]
    # Live / open season: prefer matching E0 file if present.
    y0, y1 = season.split("-")
    cand = f"E0_{y0[-2:]}{y1[-2:]}.csv"
    return cand


def _club_slice(side: pd.DataFrame, team: str, team_n: str) -> pd.DataFrame:
    club = side.loc[side["team"].astype(str) == team].copy()
    if club.empty and "team_n" in side.columns:
        club = side.loc[side["team_n"] == norm_team(team_n)].copy()
    if club.empty:
        club = side.loc[
            side["team"].map(lambda t: norm_team(str(t)) == norm_team(team_n))
        ].copy()
    return club


def _share_club_for(
    share_side: pd.DataFrame | None, club: pd.DataFrame, team: str, team_n: str
) -> pd.DataFrame | None:
    if share_side is None or share_side.empty or club.empty:
        return None
    share_club = _club_slice(share_side, team, team_n)
    if share_club.empty or len(share_club) <= len(club):
        return None
    return share_club


def project_week_opta_centreline(
    hist: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    decision_gw: int,
    target_gw: int,
    alpha_g: float,
    alpha_a: float,
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    side: pd.DataFrame | None = None,
    share_side: pd.DataFrame | None = None,
    season: str | None = None,
) -> tuple[pd.DataFrame, int]:
    """``score_team_side`` for ``target_gw`` with Opta blind λ (frozen side).

    Full-club ``m>0`` shares are projected onto the selection side when
    ``share_side`` is wider (``protocol.score_xp_v2_full_share_likelihood``).
    """
    if side is None:
        side, _, _ = build_frozen_side_table(
            hist,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            season=season,
            historical_rate_baseline=True,
        )
    if side.empty:
        return pd.DataFrame(), 0
    if share_side is None:
        share_roster = _mean_minutes_asof(hist, min_m=0.0)
        if not share_roster.empty and len(share_roster) > len(side):
            share_side = _attach_rates_priors_baseline(
                share_roster,
                hist,
                season=season,
                historical_rate_baseline=True,
            )
    fits = _opta_fits_for_gw(
        enriched,
        gw=int(target_gw),
        ranks=ranks,
        profile=profile,
        league_gf=float(league_gf),
    )
    if not fits:
        return pd.DataFrame(), 0
    parts: list[pd.DataFrame] = []
    n_ok = 0
    for fit in fits:
        for team, team_n, prefix in (
            (str(fit["home"]), str(fit["home_n"]), "home"),
            (str(fit["away"]), str(fit["away_n"]), "away"),
        ):
            club = _club_slice(side, team, str(team_n))
            if club.empty:
                continue
            scored = numagent_xp.score_team_side(
                club,
                xg=float(fit[f"xg_{prefix}"]),
                xcs=float(fit[f"xcs_{prefix}"]),
                xgc=float(fit[f"xgc_{prefix}"]),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
                share_players=_share_club_for(share_side, club, team, str(team_n)),
            ).copy()
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["club"] = team
            scored["gw"] = int(target_gw)
            scored["side"] = prefix
            scored["source"] = "opta_centreline_blind"
            parts.append(scored)
            n_ok += 1
    if not parts:
        return pd.DataFrame(), 0
    out = (
        pd.concat(parts, ignore_index=True)
        .groupby("element", as_index=False)
        .agg(
            {
                **{
                    c: "sum"
                    for c in (
                        "xp",
                        "xp_goals",
                        "xp_assists",
                        "xp_cs",
                        "xp_gc",
                        "xp_appear",
                        "xp_defcon",
                        "xp_bonus",
                        "xp_cards",
                        "xp_saves",
                        "xp_se",
                    )
                    if c in parts[0].columns
                },
                "name": "first",
                "position": "first",
                "team": "first",
                "club": "first",
                "gw": "first",
                "source": "first",
            }
        )
    )
    out["score_xp_v2"] = out["xp"]
    return out, n_ok


def build_l3_centreline_horizon(
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None = None,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    enriched: pd.DataFrame | None = None,
    horizon: int = DEFAULT_HORIZON,
    near_depth: int = NEAR_DEPTH,
    minutes_rho: float = 1.0,
    minutes_window: int = 3,
    minutes_by_element: Mapping[int, float] | None = None,
) -> L3CentrelineHorizon:
    """Pools length ≤ ``horizon``: FD/priced near + Opta blind far.

    ``minutes_by_element``: optional as-of map applied to near projections and
    Opta far side tables (e.g. sticky xmi portfolio).
    """
    H = max(1, int(horizon))
    near = max(1, min(int(near_depth), H))

    if merged is None:
        merged = _load_merged_enriched(season)
    if fd is None:
        fd = _fd_index(season)
    if enriched is None:
        enriched = merged

    priced = build_mupost_priced_horizon(
        season,
        decision_gw,
        week0_pool=week0_pool,
        moments=moments,
        merged=merged,
        fd=fd,
        prices=prices,
        minutes_rho=minutes_rho,
        minutes_window=minutes_window,
        minutes_by_element=minutes_by_element,
    )
    pools = [p.copy() for p in priced.pools[:near]]
    priced_gws = list(priced.priced_gws[:near])
    opta_gws: list[int] = []
    n_fx = int(priced.n_projected_fixtures)

    if H <= len(pools):
        return L3CentrelineHorizon(
            season=season,
            decision_gw=int(decision_gw),
            pools=pools[:H],
            priced_gws=tuple(priced_gws[:H]),
            opta_gws=(),
            n_projected_fixtures=n_fx,
            meta={
                **priced.meta,
                "horizon": len(pools[:H]),
                "near_depth": near,
                "opta_gws": [],
                "protocol": "l3_centreline_forecast",
                "minutes_map": minutes_by_element is not None,
            },
        )

    hist = numagent_xp.asof_share_history(
        merged, before_gw=int(decision_gw), season=season
    )
    alpha_g, alpha_a = _asof_alphas(hist)
    if minutes_by_element is not None:
        from src.strategy.layer4_mupost_priced import _roster_from_minutes_map

        sel = _roster_from_minutes_map(hist, minutes_by_element, min_m=0.0)
        side = _attach_rates_priors_baseline(
            sel,
            hist,
            season=season,
            historical_rate_baseline=True,
        )
        share_roster = _roster_from_minutes_map(
            hist, minutes_by_element, min_m=0.0
        )
        share_side: pd.DataFrame | None = None
        if not share_roster.empty and len(share_roster) > len(side):
            share_side = _attach_rates_priors_baseline(
                share_roster,
                hist,
                season=season,
                historical_rate_baseline=True,
            )
        elif not share_roster.empty:
            share_side = side
    else:
        side, alpha_g, alpha_a = build_frozen_side_table(
            hist,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            season=season,
            historical_rate_baseline=True,
        )
        share_side = None
        share_roster = _mean_minutes_asof(hist, min_m=0.0)
        if not share_roster.empty and len(share_roster) > len(side):
            share_side = _attach_rates_priors_baseline(
                share_roster,
                hist,
                season=season,
                historical_rate_baseline=True,
            )
    e0 = _opta_e0(season)
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = float(league_avg_gf(profile))
    avail = set(int(g) for g in merged["GW"].unique())
    w0 = pools[0]

    if prices is None:
        from src.strategy.layer4_mupost_priced import PROCESSED

        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)

    for step in range(len(pools), H):
        gw_f = int(decision_gw) + step
        if gw_f not in avail:
            break
        raw, n = project_week_opta_centreline(
            hist,
            enriched,
            decision_gw=int(decision_gw),
            target_gw=gw_f,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            side=side,
            share_side=share_side,
            season=season,
        )
        if raw.empty:
            break
        n_fx += int(n)
        board = _attach_store_ids(raw, prices, int(decision_gw), season)
        board, _ = attach_mu_post(
            board,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            out_col="mu_post",
        )
        pools.append(_align_to_week0(w0, board))
        opta_gws.append(gw_f)

    return L3CentrelineHorizon(
        season=season,
        decision_gw=int(decision_gw),
        pools=pools,
        priced_gws=tuple(priced_gws),
        opta_gws=tuple(opta_gws),
        n_projected_fixtures=n_fx,
        meta={
            **priced.meta,
            "horizon": len(pools),
            "near_depth": near,
            "opta_gws": list(opta_gws),
            "opta_e0": e0,
            "protocol": "l3_centreline_forecast",
            "lambda_far": "opta_blind_rankmean",
            "minutes_map": minutes_by_element is not None,
        },
    )


def opta_fits_from_fixture_pairs(
    pairs: list[tuple[str, str]],
    *,
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    date_iso: str = "",
) -> list[dict[str, Any]]:
    """Build Opta centreline fits from (home, away) display-name pairs (live)."""
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
        lam_h = float(pred["lam_home"])
        lam_a = float(pred["lam_away"])
        mom = moments_from_lams(lam_h, lam_a, 0.0)
        out.append(
            {
                "fixture": -1,
                "date_iso": date_iso,
                "home": home,
                "away": away,
                "home_n": h,
                "away_n": a,
                "lam_h": lam_h,
                "lam_a": lam_a,
                "rho": 0.0,
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
    """Aggregate ``score_team_side`` over Opta fits (DGW-safe)."""
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
            scored = numagent_xp.score_team_side(
                club,
                xg=float(fit[f"xg_{prefix}"]),
                xcs=float(fit[f"xcs_{prefix}"]),
                xgc=float(fit[f"xgc_{prefix}"]),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
                share_players=_share_club_for(share_side, club, team, str(team_n)),
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
                **{
                    c: "sum"
                    for c in ("xp", "xp_se")
                    if c in parts[0].columns
                },
                "name": "first",
                "position": "first",
                "team": "first",
                "club": "first",
            }
        )
    )
    out["score_xp_v2"] = out["xp"]
    return out


def extend_pools_with_opta_far(
    pools: list[pd.DataFrame],
    *,
    season: str,
    decision_gw: int,
    week0_pool: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    moments: dict[str, CohortMoments] | None = None,
    merged: pd.DataFrame | None = None,
    enriched: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """Append Opta-blind boards onto already-built near pools (e.g. live t/t+1)."""
    H = max(len(pools), int(horizon))
    if len(pools) >= H:
        return pools, {"horizon": len(pools), "opta_gws": []}
    if merged is None:
        merged = _load_merged_enriched(season)
    if enriched is None:
        enriched = merged
    hist = numagent_xp.asof_share_history(
        merged, before_gw=int(decision_gw), season=season
    )
    alpha_g, alpha_a = _asof_alphas(hist)
    side, alpha_g, alpha_a = build_frozen_side_table(
        hist,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        season=season,
        historical_rate_baseline=True,
    )
    share_side: pd.DataFrame | None = None
    share_roster = _mean_minutes_asof(hist, min_m=0.0)
    if not share_roster.empty and len(share_roster) > len(side):
        share_side = _attach_rates_priors_baseline(
            share_roster,
            hist,
            season=season,
            historical_rate_baseline=True,
        )
    e0 = _opta_e0(season)
    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = float(league_avg_gf(profile))
    avail = set(int(g) for g in merged["GW"].unique())
    if prices is None:
        from src.strategy.layer4_mupost_priced import PROCESSED

        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)

    out = [p.copy() for p in pools]
    w0 = week0_pool
    opta_gws: list[int] = []
    for step in range(len(out), H):
        gw_f = int(decision_gw) + step
        if gw_f not in avail:
            break
        raw, _n = project_week_opta_centreline(
            hist,
            enriched,
            decision_gw=int(decision_gw),
            target_gw=gw_f,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            side=side,
            share_side=share_side,
            season=season,
        )
        if raw.empty:
            break
        board = _attach_store_ids(raw, prices, int(decision_gw), season)
        board, _ = attach_mu_post(
            board,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            out_col="mu_post",
        )
        out.append(_align_to_week0(w0, board))
        opta_gws.append(gw_f)
    return out, {
        "horizon": len(out),
        "opta_gws": opta_gws,
        "opta_e0": e0,
        "protocol": "l3_centreline_forecast",
    }
