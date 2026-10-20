"""Short-horizon scenario advisor on priced GW t and t+1 — Numagent 2.0.

Architecture (does **not** replace published ``score_xp_v2`` / L4 mean path):

1. **Mean layer** — frozen as-of rates (same-season GW ``< t``) + football-data
   opening AH+OU Dixon–Coles moments → ``numagent_xp.score_team_side``
   (``SCORE_NAME = score_xp_v2``). Same assembler as L4 ``project_week_score_xp_v2``.
2. **Scenario layer** — sample ``λ_H``, ``λ_A`` inside ±15% of the FD fit,
   rebuild ``M`` (ρ held), then **generative** attack on each priced fixture:
   ``k ∼ M`` → lineup minutes cov → scorer/assist dice (α empties). Firm XI
   override deferred. Toggle ``generative_attack=False`` to fall back to
   closed-form ``score_team_side`` on sampled moments.
3. **Path FT (default)** — L1 free rebuild at ``t``, then L3 transfers+hits at
   ``t+1`` with FT bank 1 (wildcard-like open → banked FT).
4. **L4 inject (optional)** — ``scored_weeks_to_l4_pools`` maps scored boards
   into ``solve_horizon_transfers`` via ``layer4_mu_inject`` (no FD factory).
   Default advisor stays L1→L3 until a gated path-V screen clears — see
   ``reports/CASCADE_PLAN.md``.

Diagnostic only until a path-V screen clears. Historical FD openings only
(no Odds API in this screen). Long-horizon Opta cascade stays at t+2….
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas
from src.markets.scoreline import build_matrix, team_moments
from src.markets.short_horizon_generative import (
    hist_by_team_norm,
    score_week_generative,
)
from src.models.player_pool import filter_board_to_pool, load_player_pool_ids
from src.strategy.layer1_week import solve_week
from src.strategy.layer3_week import held_from_frame, solve_week_transfers
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched, _schedule
from src.strategy.layer4_mu_inject import pools_from_mu_boards
from src.strategy.layer4_mupost_priced import (
    _attach_store_ids,
    _mean_minutes_asof,
    project_week_score_xp_v2,
)
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

MARKET_BAND = 0.15
DEFAULT_N = 8
DEFAULT_SEED = 7
DEFAULT_GAMMA = 0.9
POOL_CAP_PER_POS = 30
MIN_POOL = 40
SCORER = numagent_xp.SCORE_NAME  # score_xp_v2
# Scenario strands on priced t:t+1 use generative attack by default.
DEFAULT_GENERATIVE_ATTACK = True
DEFAULT_LINEUP_COV = True


def load_prices(season: str) -> pd.DataFrame:
    path = PROCESSED / f"score_xp_{season.replace('-', '_')}.csv"
    prices = pd.read_csv(
        path,
        usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
    )
    prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
    prices["element"] = pd.to_numeric(prices["element"], errors="coerce").astype(int)
    return prices


def fixtures_for_gw(merged: pd.DataFrame, gw: int) -> list[dict[str, Any]]:
    """Unique fixtures in ``gw`` with home/away + date for FD join."""
    sched = _schedule(merged)
    week = sched.loc[sched["GW"] == int(gw)]
    rows: list[dict[str, Any]] = []
    for row in week.itertuples(index=False):
        rows.append(
            {
                "fixture": int(row.fixture),
                "date_iso": row.date_iso,
                "home": str(row.home_team),
                "away": str(row.away_team),
                "home_n": str(row.home_n),
                "away_n": str(row.away_n),
            }
        )
    return rows


def centreline_fixture_fits(
    fixtures: Sequence[Mapping[str, Any]],
    fd: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Attach FD lam/moments to each fixture (home-side row carries lam_h/lam_a)."""
    out: list[dict[str, Any]] = []
    for fx in fixtures:
        home = fd.loc[(fd["team_n"] == fx["home_n"]) & (fd["date_iso"] == fx["date_iso"])]
        away = fd.loc[(fd["team_n"] == fx["away_n"]) & (fd["date_iso"] == fx["date_iso"])]
        if home.empty or away.empty:
            continue
        h, a = home.iloc[0], away.iloc[0]
        out.append(
            {
                **dict(fx),
                "lam_h": float(h["lam_h"]),
                "lam_a": float(h["lam_a"]),
                "rho": float(h["rho"] if "rho" in h.index and pd.notna(h["rho"]) else 0.0),
                "xg_home": float(h["xg"]),
                "xcs_home": float(h["xcs"]),
                "xgc_home": float(h["xgc"]),
                "xg_away": float(a["xg"]),
                "xcs_away": float(a["xcs"]),
                "xgc_away": float(a["xgc"]),
            }
        )
    return out


def moments_from_lams(lam_h: float, lam_a: float, rho: float) -> dict[str, float]:
    mat = build_matrix(float(lam_h), float(lam_a), float(rho))
    return team_moments(mat)


def sample_fixture_moments(
    fit: Mapping[str, Any],
    rng: random.Random,
    *,
    band: float = MARKET_BAND,
) -> dict[str, float]:
    """Perturb FD λ ±band, rebuild Dixon–Coles M, return side moments."""
    lh = float(fit["lam_h"])
    la = float(fit["lam_a"])
    rho = float(fit.get("rho", 0.0) or 0.0)
    lh_s = lh * rng.uniform(1.0 - band, 1.0 + band)
    la_s = la * rng.uniform(1.0 - band, 1.0 + band)
    mom = moments_from_lams(max(lh_s, 1e-6), max(la_s, 1e-6), rho)
    return {
        "xg_home": mom["xg_home"],
        "xcs_home": mom["xcs_home"],
        "xgc_home": mom["xgc_home"],
        "xg_away": mom["xg_away"],
        "xcs_away": mom["xcs_away"],
        "xgc_away": mom["xgc_away"],
        "lam_h": lh_s,
        "lam_a": la_s,
    }


def build_frozen_side_table(
    hist: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
) -> tuple[pd.DataFrame, float, float]:
    """As-of roster with rates/priors/minutes frozen (same-season GW < t).

    ``hist`` must already be same-season rows with GW ``< t`` (caller uses
    ``asof_share_history``).
    """
    numagent_xp.assert_single_season_rate_history(hist)
    roster = _mean_minutes_asof(hist, rho=1.0, window=3)
    if roster.empty:
        return pd.DataFrame(), float(alpha_g), float(alpha_a)
    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    side = roster.merge(rates, on="element", how="left").merge(
        priors.drop(columns=["position"], errors="ignore"),
        on="element",
        how="left",
        suffixes=("", "_pr"),
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
        if c not in side.columns:
            side[c] = default
        else:
            side[c] = pd.to_numeric(side[c], errors="coerce").fillna(default)
    side["player_id"] = side["element"].astype(str)
    side["team_n"] = side["team"].map(norm_team)
    return side, float(alpha_g), float(alpha_a)


def score_week_on_fixture_moments(
    side_table: pd.DataFrame,
    fixture_fits: Sequence[Mapping[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    moment_overrides: Mapping[int, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Score all clubs in the week; optional per-fixture moment overrides."""
    if side_table.empty or not fixture_fits:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for fit in fixture_fits:
        fx_id = int(fit["fixture"])
        ov = (moment_overrides or {}).get(fx_id)
        for team, team_n, prefix in (
            (str(fit["home"]), str(fit["home_n"]), "home"),
            (str(fit["away"]), str(fit["away_n"]), "away"),
        ):
            club = side_table.loc[side_table["team"].astype(str) == team].copy()
            if club.empty:
                club = side_table.loc[side_table["team_n"] == team_n].copy()
            if club.empty:
                continue
            if ov is not None:
                xg = float(ov[f"xg_{prefix}"])
                xcs = float(ov[f"xcs_{prefix}"])
                xgc = float(ov[f"xgc_{prefix}"])
            else:
                xg = float(fit[f"xg_{prefix}"])
                xcs = float(fit[f"xcs_{prefix}"])
                xgc = float(fit[f"xgc_{prefix}"])
            scored = numagent_xp.score_team_side(
                club,
                xg=xg,
                xcs=xcs,
                xgc=xgc,
                alpha_g=alpha_g,
                alpha_a=alpha_a,
            )
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["club"] = team
            scored["fixture"] = fx_id
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    all_p = pd.concat(parts, ignore_index=True)
    board = (
        all_p.groupby("element", as_index=False)
        .agg(
            name=("name", "first") if "name" in all_p.columns else ("element", "first"),
            position=("position", "first"),
            team=("team", "first"),
            club=("club", "first"),
            score_xp_v2=("xp", "sum"),
            xp_se=("xp_se", lambda s: float(np.sqrt((s.astype(float) ** 2).sum()))),
        )
    )
    if "name" not in board.columns or board["name"].isna().all():
        board = board.drop(columns=["name"], errors="ignore")
        if "name" in side_table.columns:
            board = board.merge(side_table[["element", "name"]], on="element", how="left")
    return board


def attach_prices_and_cap(
    board: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    season: str,
    decision_gw: int,
    force_ids: Sequence[str] | None = None,
    selectable_ids: set[str] | None = None,
    apply_player_pool: bool = True,
) -> pd.DataFrame:
    """L4-style price attach + shared player_pool filter + per-position cap.

    ``force_ids`` (e.g. owned squad) are always kept even under the pool cut
    and the top-N cap. Buy universe defaults to the sticky FFScout ∩ minutes
    ``player_pool`` snapshot (same set as cascade_long).
    """
    if board.empty:
        return board
    out = _attach_store_ids(board, prices, decision_gw, season)
    out["mu_post"] = pd.to_numeric(out[SCORER], errors="coerce").fillna(0.0)
    force = {str(x) for x in (force_ids or ())}
    if apply_player_pool:
        ids = selectable_ids
        if ids is None:
            ids = load_player_pool_ids(missing_ok=True)
        if ids:
            out = filter_board_to_pool(out, ids, force_ids=force)
    # Cap MILP: highest μ within each position, then re-attach forced ids.
    if len(out) > 100:
        parts = []
        for _, block in out.groupby("position", sort=False):
            parts.append(block.nlargest(min(len(block), POOL_CAP_PER_POS), "mu_post"))
        capped = pd.concat(parts, ignore_index=True)
        if force:
            have = set(capped["player_id"].astype(str))
            missing = force - have
            if missing:
                extra = out.loc[out["player_id"].astype(str).isin(missing)]
                capped = pd.concat([capped, extra], ignore_index=True)
        out = capped.drop_duplicates("player_id", keep="first")
    return out.reset_index(drop=True)


def scored_weeks_to_l4_pools(
    week0_pool: pd.DataFrame,
    *scored_weeks: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Map priced/scenario scored boards onto L4 pools (μ inject).

    ``week0_pool`` supplies universe / prices / club meta. Each scored week
    needs ``element`` and ``score_xp_v2`` (or ``mu_post``). Does not call
    ``project_week_score_xp_v2`` or Odds.
    """
    if not scored_weeks:
        raise ValueError("need at least one scored week board")
    boards: list[pd.DataFrame] = []
    for board in scored_weeks:
        if board is None or board.empty:
            raise ValueError("scored week board is empty")
        boards.append(board)
    return pools_from_mu_boards(week0_pool, boards, out_col="mu_post")


def mean_player_xp_delta(
    side_table: pd.DataFrame,
    fixture_fits: Sequence[Mapping[str, Any]],
    sample_overrides: Sequence[Mapping[int, Mapping[str, float]]],
    *,
    alpha_g: float,
    alpha_a: float,
) -> dict[str, float]:
    base = score_week_on_fixture_moments(
        side_table, fixture_fits, alpha_g=alpha_g, alpha_a=alpha_a
    )
    if base.empty:
        return {"mean_abs_player_delta": float("nan"), "max_mean_abs_player_delta": float("nan")}
    base_map = dict(zip(base["element"].astype(int), base[SCORER].astype(float)))
    deltas: list[float] = []
    for ov in sample_overrides:
        scored = score_week_on_fixture_moments(
            side_table,
            fixture_fits,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            moment_overrides=ov,
        )
        if scored.empty:
            continue
        d = []
        for row in scored.itertuples(index=False):
            eid = int(row.element)
            d.append(abs(float(getattr(row, SCORER)) - float(base_map.get(eid, 0.0))))
        if d:
            deltas.append(float(np.mean(d)))
    return {
        "mean_abs_player_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "max_mean_abs_player_delta": float(np.max(deltas)) if deltas else float("nan"),
    }


def _lam_overrides_from_moments(
    moment_ov: Mapping[int, Mapping[str, float]],
) -> dict[int, dict[str, float]]:
    """Keep λ draws for generative rolls (moments also carry xg/xcs/xgc)."""
    out: dict[int, dict[str, float]] = {}
    for fx_id, mom in moment_ov.items():
        out[int(fx_id)] = {
            "lam_h": float(mom["lam_h"]),
            "lam_a": float(mom["lam_a"]),
        }
    return out


def score_week_scenario(
    side_table: pd.DataFrame,
    fixture_fits: Sequence[Mapping[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    moment_overrides: Mapping[int, Mapping[str, float]] | None = None,
    generative_attack: bool = DEFAULT_GENERATIVE_ATTACK,
    lineup_cov: bool = DEFAULT_LINEUP_COV,
    hist_by_team: Mapping[str, pd.DataFrame] | None = None,
    rng: Any | None = None,
) -> pd.DataFrame:
    """Score one scenario week: generative attack (default) or mean-layer moments."""
    if generative_attack:
        lam_ov = (
            _lam_overrides_from_moments(moment_overrides)
            if moment_overrides is not None
            else None
        )
        gen_rng = rng if rng is not None else np.random.default_rng(0)
        return score_week_generative(
            side_table,
            fixture_fits,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            rng=gen_rng,
            hist_by_team=hist_by_team,
            lineup_cov=lineup_cov,
            lam_overrides=lam_ov,
        )
    return score_week_on_fixture_moments(
        side_table,
        fixture_fits,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        moment_overrides=moment_overrides,
    )


def run_decision(
    season: str,
    decision_gw: int,
    *,
    n_strands: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    gamma: float = DEFAULT_GAMMA,
    band: float = MARKET_BAND,
    time_limit: float = 8.0,
    generative_attack: bool = DEFAULT_GENERATIVE_ATTACK,
    lineup_cov: bool = DEFAULT_LINEUP_COV,
) -> dict[str, Any]:
    """One deadline: independent E[V] vs scenario path V on t and t+1 (score_xp_v2)."""
    t = int(decision_gw)
    t1 = t + 1
    merged = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    fx_t = fixtures_for_gw(merged, t)
    fx_t1 = fixtures_for_gw(merged, t1)
    fits_t = centreline_fixture_fits(fx_t, fd)
    fits_t1 = centreline_fixture_fits(fx_t1, fd)
    if len(fits_t) < 5 or len(fits_t1) < 5:
        raise RuntimeError(
            f"thin FD fits GW{t}/{t1}: {len(fits_t)}/{len(fits_t1)} "
            f"(fixtures {len(fx_t)}/{len(fx_t1)})"
        )

    hist = merged.loc[merged["GW"] < t].copy()
    if hist.empty:
        raise RuntimeError(f"no history before GW{t}")
    alpha_g, alpha_a = _asof_alphas(hist)
    # Freeze rates on GW < t explicitly (asof helper).
    hist = numagent_xp.asof_share_history(hist, before_gw=t, season=season)
    side_table, alpha_g, alpha_a = build_frozen_side_table(
        hist, alpha_g=alpha_g, alpha_a=alpha_a
    )
    if side_table.empty:
        raise RuntimeError(f"empty as-of roster before GW{t}")

    team_hist = hist_by_team_norm(hist) if generative_attack else {}

    # Cross-check mean layer against L4 projector (same assembler).
    proj_t, n_fx_t = project_week_score_xp_v2(
        hist,
        merged,
        fd,
        decision_gw=t,
        target_gw=t,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
    )
    if proj_t.empty:
        raise RuntimeError(f"project_week_score_xp_v2 empty for GW{t}")

    # Mean / independent boards stay closed-form score_team_side.
    scored_t = attach_prices_and_cap(
        score_week_on_fixture_moments(
            side_table, fits_t, alpha_g=alpha_g, alpha_a=alpha_a
        ),
        prices,
        season=season,
        decision_gw=t,
    )
    scored_t1 = attach_prices_and_cap(
        score_week_on_fixture_moments(
            side_table, fits_t1, alpha_g=alpha_g, alpha_a=alpha_a
        ),
        prices,
        season=season,
        decision_gw=t,
    )
    if len(scored_t) < MIN_POOL or len(scored_t1) < MIN_POOL:
        raise RuntimeError(
            f"thin scored pools GW{t}/{t1}: {len(scored_t)}/{len(scored_t1)}"
        )

    indep_t = solve_week(scored_t, mu_col=SCORER, time_limit=time_limit)
    indep_t1 = solve_week(scored_t1, mu_col=SCORER, time_limit=time_limit)
    v_indep = float(indep_t.objective + gamma * indep_t1.objective)
    indep_ids_t = set(str(x) for x in indep_t.squad["player_id"])
    indep_ids_t1 = set(str(x) for x in indep_t1.squad["player_id"])
    indep_overlap = len(indep_ids_t & indep_ids_t1) / 15.0

    rng = random.Random(int(seed))
    strand_rows: list[dict[str, Any]] = []
    sample_ov_t: list[dict[int, dict[str, float]]] = []
    sample_ov_t1: list[dict[int, dict[str, float]]] = []

    for s_i in range(int(n_strands)):
        ov_t: dict[int, dict[str, float]] = {}
        ov_t1: dict[int, dict[str, float]] = {}
        for fit in fits_t:
            ov_t[int(fit["fixture"])] = sample_fixture_moments(fit, rng, band=band)
        for fit in fits_t1:
            ov_t1[int(fit["fixture"])] = sample_fixture_moments(fit, rng, band=band)
        sample_ov_t.append(ov_t)
        sample_ov_t1.append(ov_t1)

        gen_rng = np.random.default_rng(int(seed) * 1_000_003 + int(s_i) * 97 + 13)
        board_t = score_week_scenario(
            side_table,
            fits_t,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            moment_overrides=ov_t,
            generative_attack=generative_attack,
            lineup_cov=lineup_cov,
            hist_by_team=team_hist,
            rng=gen_rng,
        )
        board_t1 = score_week_scenario(
            side_table,
            fits_t1,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            moment_overrides=ov_t1,
            generative_attack=generative_attack,
            lineup_cov=lineup_cov,
            hist_by_team=team_hist,
            rng=gen_rng,
        )
        st = attach_prices_and_cap(board_t, prices, season=season, decision_gw=t)
        pick_t = solve_week(st, mu_col=SCORER, time_limit=time_limit)
        held = held_from_frame(pick_t.squad, bank=0, ft=1)
        # L3 requires every held id in the pool — force owned into the capped set.
        # Players blank at t+1 get μ=0 from a price-only stub so the MILP can keep them.
        st1 = attach_prices_and_cap(
            board_t1,
            prices,
            season=season,
            decision_gw=t,
            force_ids=held.player_ids,
        )
        have = set(st1["player_id"].astype(str))
        still_missing = [pid for pid in held.player_ids if pid not in have]
        if still_missing:
            stubs = scored_t.loc[scored_t["player_id"].astype(str).isin(still_missing)].copy()
            stubs[SCORER] = 0.0
            stubs["mu_post"] = 0.0
            stubs["xp_se"] = 0.0
            st1 = pd.concat([st1, stubs], ignore_index=True).drop_duplicates(
                "player_id", keep="first"
            )
        pick_t1 = solve_week_transfers(
            st1, held, mu_col=SCORER, time_limit=time_limit
        )
        owned_t = set(str(x) for x in pick_t.squad["player_id"])
        owned_t1 = set(str(x) for x in pick_t1.squad["player_id"])
        stick = len(owned_t & owned_t1) / 15.0
        v_path = float(pick_t.objective + gamma * pick_t1.objective)
        strand_rows.append(
            {
                "strand": s_i,
                "v_path": v_path,
                "obj_t": float(pick_t.objective),
                "obj_t1_net": float(pick_t1.objective),
                "transfers_t1": int(pick_t1.n_transfers),
                "hits_t1": int(pick_t1.hits),
                "stick_rate": stick,
                "captain_t": pick_t.captain_id,
                "captain_t1": pick_t1.captain_id,
            }
        )

    frame = pd.DataFrame(strand_rows)
    xp_delta = mean_player_xp_delta(
        side_table, fits_t, sample_ov_t, alpha_g=alpha_g, alpha_a=alpha_a
    )
    xp_delta_t1 = mean_player_xp_delta(
        side_table, fits_t1, sample_ov_t1, alpha_g=alpha_g, alpha_a=alpha_a
    )

    # Sanity: mean-layer board should correlate with L4 projector.
    proj_map = dict(
        zip(proj_t["element"].astype(int), proj_t[SCORER].astype(float))
    )
    common = scored_t.loc[scored_t["element"].astype(int).isin(proj_map)]
    if len(common) >= 20:
        a = common[SCORER].astype(float).to_numpy()
        b = common["element"].astype(int).map(proj_map).to_numpy(float)
        corr = float(pd.Series(a).corr(pd.Series(b)))
    else:
        corr = float("nan")

    scenario_note = (
        "generative k∼M → lineup cov → attack dice (α empties); firm XI later"
        if generative_attack
        else f"closed-form score_team_side ({SCORER}) on sampled moments"
    )
    return {
        "season": season,
        "decision_gw": t,
        "next_gw": t1,
        "scorer": SCORER,
        "assembler": "src/markets/numagent_xp.py",
        "scenario_scorer": (
            "short_horizon_generative" if generative_attack else "score_team_side"
        ),
        "generative_attack": bool(generative_attack),
        "lineup_cov": bool(lineup_cov),
        "n_strands": int(n_strands),
        "seed": int(seed),
        "gamma": float(gamma),
        "band": float(band),
        "alpha_g": float(alpha_g),
        "alpha_a": float(alpha_a),
        "n_pool_t": int(len(scored_t)),
        "n_pool_t1": int(len(scored_t1)),
        "n_fixtures_t": len(fits_t),
        "n_fixtures_t1": len(fits_t1),
        "n_proj_fixtures_t": int(n_fx_t),
        "mean_layer_vs_l4_corr": corr,
        "v_indep": v_indep,
        "indep_overlap_t_t1": indep_overlap,
        "mean_v_path": float(frame["v_path"].mean()),
        "std_v_path": float(frame["v_path"].std(ddof=0)),
        "delta_v_path_minus_indep": float(frame["v_path"].mean() - v_indep),
        "mean_transfers_t1": float(frame["transfers_t1"].mean()),
        "mean_hits_t1": float(frame["hits_t1"].mean()),
        "mean_stick_rate": float(frame["stick_rate"].mean()),
        "player_xp_delta_t": xp_delta,
        "player_xp_delta_t1": xp_delta_t1,
        "strands": strand_rows,
        "notes": (
            "Rates/minutes frozen at GW < t (same-season only). "
            "FD AH+OU → Dixon–Coles M; λ ± band sampled. "
            f"Mean layer: score_team_side ({SCORER}). "
            f"Scenario strands: {scenario_note}. "
            "Path: L1 then L3. Does not replace L4 mean path."
        ),
    }


def run_screen(
    *,
    season: str = "2024-25",
    decision_gws: Sequence[int] = (10, 20, 28),
    n_strands: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    generative_attack: bool = DEFAULT_GENERATIVE_ATTACK,
    lineup_cov: bool = DEFAULT_LINEUP_COV,
) -> dict[str, Any]:
    """Multi-deadline diagnostic on one closed season."""
    rows = []
    for gw in decision_gws:
        print(f"{season} GW{gw} ({SCORER})…", flush=True)
        try:
            rows.append(
                run_decision(
                    season,
                    int(gw),
                    n_strands=n_strands,
                    seed=seed,
                    generative_attack=generative_attack,
                    lineup_cov=lineup_cov,
                )
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic continues
            rows.append(
                {
                    "season": season,
                    "decision_gw": int(gw),
                    "scorer": SCORER,
                    "error": str(exc),
                }
            )

    ok = [r for r in rows if "error" not in r]
    summary = {
        "season": season,
        "scorer": SCORER,
        "decision_gws": list(decision_gws),
        "n_ok": len(ok),
        "mean_delta_v": float(np.mean([r["delta_v_path_minus_indep"] for r in ok]))
        if ok
        else float("nan"),
        "mean_stick": float(np.mean([r["mean_stick_rate"] for r in ok])) if ok else float("nan"),
        "mean_hits": float(np.mean([r["mean_hits_t1"] for r in ok])) if ok else float("nan"),
        "mean_abs_player_xp_delta_t": float(
            np.mean([r["player_xp_delta_t"]["mean_abs_player_delta"] for r in ok])
        )
        if ok
        else float("nan"),
        "mean_layer_vs_l4_corr": float(
            np.nanmean([r.get("mean_layer_vs_l4_corr", float("nan")) for r in ok])
        )
        if ok
        else float("nan"),
    }
    indep_overlap = (
        float(np.mean([r["indep_overlap_t_t1"] for r in ok])) if ok else float("nan")
    )
    path_signal = bool(ok and summary["mean_stick"] >= indep_overlap + 0.05)
    mean_layer_quiet = bool(ok and summary["mean_abs_player_xp_delta_t"] < 0.25)
    decision = "keep_scenario_advisor" if path_signal else "park_scenario_layer"
    return {
        "task": "short_horizon_scenario",
        "scorer": SCORER,
        "scenario_scorer": (
            "short_horizon_generative" if generative_attack else "score_team_side"
        ),
        "generative_attack": bool(generative_attack),
        "lineup_cov": bool(lineup_cov),
        "decision": decision,
        "path_signal": path_signal,
        "mean_layer_quiet": mean_layer_quiet,
        "summary": summary,
        "indep_overlap_mean": indep_overlap,
        "per_deadline": [
            {k: v for k, v in r.items() if k != "strands"} for r in rows
        ],
        "strands_by_deadline": {
            str(r.get("decision_gw")): r.get("strands") for r in ok
        },
    }


def write_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    out = path or (REPORTS / "short_horizon_scenario.md")
    s = payload["summary"]
    lines = [
        "# Short-horizon scenario layer (priced t : t+1) — Numagent 2.0",
        "",
        f"Advisor layer on market-priced weeks. Mean scorer: **`{SCORER}`** "
        "(`src/markets/numagent_xp.py`). Scenario strands: "
        f"**`{payload.get('scenario_scorer', 'short_horizon_generative')}`** "
        f"(generative_attack={payload.get('generative_attack')}, "
        f"lineup_cov={payload.get('lineup_cov')}). Does **not** replace the "
        "published L4 mean path. Rates frozen at the deadline; λ sampled "
        "inside ±15% of FD Dixon–Coles fits; path FT via L1 → L3.",
        "",
        "## Decision",
        "",
        f"**{payload['decision']}**",
        "",
        "- Keep advisor if path stick ≥ independent t∩t+1 overlap + 0.05. "
        f"Published `{SCORER}` stays the centreline mean layer either way.",
        f"- path_signal={payload['path_signal']} · "
        f"mean_layer_quiet={payload.get('mean_layer_quiet')} "
        f"(|ΔxP|<0.25)",
        "",
        "## Summary",
        "",
        f"| season | {s['season']} |",
        f"| mean scorer | {s.get('scorer', SCORER)} |",
        f"| scenario scorer | {payload.get('scenario_scorer')} |",
        f"| deadlines ok | {s['n_ok']} / {len(payload.get('per_deadline') or [])} |",
        f"| mean ΔV (path − indep) | {s['mean_delta_v']:+.3f} |",
        f"| mean stick (path) | {s['mean_stick']:.3f} |",
        f"| mean indep overlap t∩t+1 | {payload['indep_overlap_mean']:.3f} |",
        f"| mean hits t+1 | {s['mean_hits']:.2f} |",
        f"| mean |ΔxP| vs centreline (t) | {s['mean_abs_player_xp_delta_t']:.4f} |",
        f"| mean-layer vs L4 projector corr | {s.get('mean_layer_vs_l4_corr', float('nan')):.3f} |",
        "",
        "## Per deadline",
        "",
        "| GW | V_indep | mean V_path | ΔV | stick | hits | |ΔxP| | L4 corr |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["per_deadline"]:
        if "error" in r:
            lines.append(f"| {r['decision_gw']} | error: {r['error']} | | | | | | |")
            continue
        lines.append(
            f"| {r['decision_gw']} | {r['v_indep']:.2f} | {r['mean_v_path']:.2f} | "
            f"{r['delta_v_path_minus_indep']:+.2f} | {r['mean_stick_rate']:.2f} | "
            f"{r['mean_hits_t1']:.2f} | {r['player_xp_delta_t']['mean_abs_player_delta']:.4f} | "
            f"{r.get('mean_layer_vs_l4_corr', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## Mechanics (aligned to Numagent 2.0)",
        "",
        "1. Scoreline: FD opening AH+OU → Dixon–Coles `M` (`scoreline.build_matrix`).",
        "2. Mean layer: team pots `α_G/α_A × xG_M`; defence from `xCS` / `xGC`; "
        "`numagent_xp.score_team_side` → `score_xp_v2`.",
        "3. Player rates: same-season GW `< t` only (`asof_share_history` + RATE_M0).",
        "4. Scenario (priced t:t+1): sample λ ±15%; rebuild `M`; then "
        "`k ∼ M` → lineup minutes cov → scorer/assist dice "
        "(`short_horizon_generative`). Firm XI override deferred.",
        "5. Path: L1 free rebuild at t, L3 FT/hits at t+1.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.models.short_horizon_cascade",
        "```",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


def main() -> None:
    payload = run_screen()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in payload.items() if k != "strands_by_deadline"}
    (PROCESSED / "short_horizon_scenario.json").write_text(
        json.dumps(slim, indent=2, default=str) + "\n"
    )
    report = write_report(payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "scorer": payload["scorer"],
                "summary": payload["summary"],
            },
            indent=2,
        )
    )
    print(f"report={report}")


if __name__ == "__main__":
    main()
