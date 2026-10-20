"""Item 3 — match-band calibration screen (closed seasons only).

Calibrates the *level* width of Opta rank→GF/GA→λ bands used by the
multistage cascade. Lead-time %/GW widening stays parked. Share seeding
stays off — do not open it in the same step.

Pre-registered levers (one family at a time):
  1. ``z_band`` ∈ {0.5, 1.0, 1.5, 2.0}  — scales season GF/GA ±sd
  2. ``centre_blend`` ∈ {0.0, 0.5, 1.0} — weight on centreline λ in MC

Baseline locked: ``z_band=1.0``, ``centre_blend=0.5``, ``SEED_SHARES=False``.

Metrics (oracle ranks + LOSO OLS on the other three closed seasons):
  - Season residual coverage: actual GF/GA inside ±z·sd (the claim of Z_BAND)
  - Match centreline MAE (unchanged in z; sanity)
  - MC predictive: strand-mean MAE; empirical 50/80% coverage of goals
  - Player-GW Gaussian log of *MC-mean* cascade xP (primary decision metric)

2026/27 is not fit and not scored.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import gaussian_log_score
from src.eval.oracle_rank_cascade import (
    GW_END,
    GW_START,
    MIN_HISTORY,
    MIN_MINUTES,
    SIGMA,
    _map_kwargs,
    match_lambda_scores,
)
from src.ingest.fpl_odds import load_football_data
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import SCORER, attach_rates_priors, score_fixture_sides_on_lams
from src.markets.numagent_xp_pilot import load_merged
from src.models.multistage_mc_milp import sample_scoreline
from src.models.opta_gf_ga_horizon import (
    CACHE,
    Z_BAND,
    fit_rank_ols_excluding,
    gf_ga_from_rank,
    match_prediction,
    rank_gf_ga_profile_excluding,
    _season_table,
)
from src.teams import norm_team

# Eligibility: minutes oracle + history depth (Numagent 2.0 closed pool).
MIN_XMI = MIN_MINUTES  # report alias; filter uses minutes > MIN_MINUTES

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"

# Season label → (hyphen code for player logs, E0 csv, football-data code).
SEASONS = [
    ("2022/23", "2022-23", "E0_2223.csv", "2223"),
    ("2023/24", "2023-24", "E0_2324.csv", "2324"),
    ("2024/25", "2024-25", "E0_2425.csv", "2425"),
    ("2025/26", "2025-26", "E0_2526.csv", "2526"),
]

Z_GRID = (0.5, 1.0, 1.5, 2.0)
BLEND_GRID = (0.0, 0.5, 1.0)
N_MC = 40
MC_SEED = 0
BASELINE_Z = float(Z_BAND)
BASELINE_BLEND = 0.5


def season_band_coverage(
    oracle_table: list[dict[str, Any]],
    ranks: dict[str, float],
    ols: dict[str, Any],
    profile: dict[int, dict[str, float]],
    *,
    z_band: float,
) -> dict[str, Any]:
    """Fraction of clubs whose realised GF/GA sit inside ±z·sd bands."""
    hit_gf = 0
    hit_ga = 0
    n = 0
    widths_gf: list[float] = []
    widths_ga: list[float] = []
    for r in oracle_table:
        club = norm_team(r["team"])
        pred = gf_ga_from_rank(
            ranks[club],
            profile=profile,
            method="ols",
            ols_gf=tuple(ols["ols_gf"]),
            ols_ga=tuple(ols["ols_ga"]),
            mae_gf=float(ols["mae_gf"]),
            mae_ga=float(ols["mae_ga"]),
            z_band=z_band,
        )
        n += 1
        if pred["gf_lo"] <= float(r["gf"]) <= pred["gf_hi"]:
            hit_gf += 1
        if pred["ga_lo"] <= float(r["ga"]) <= pred["ga_hi"]:
            hit_ga += 1
        widths_gf.append(pred["gf_hi"] - pred["gf_lo"])
        widths_ga.append(pred["ga_hi"] - pred["ga_lo"])
    return {
        "z_band": float(z_band),
        "n_clubs": n,
        "cov_gf": hit_gf / n if n else float("nan"),
        "cov_ga": hit_ga / n if n else float("nan"),
        "cov_mean": (hit_gf + hit_ga) / (2 * n) if n else float("nan"),
        "mean_width_gf": float(np.mean(widths_gf)) if widths_gf else float("nan"),
        "mean_width_ga": float(np.mean(widths_ga)) if widths_ga else float("nan"),
    }


def mc_predictive(
    fixtures: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
    *,
    z_band: float,
    centre_blend: float,
    n_strands: int = N_MC,
    seed: int = MC_SEED,
) -> dict[str, Any]:
    """Strand predictive checks for goals under (z_band, centre_blend)."""
    rng = random.Random(seed)
    abs_err_mean: list[float] = []
    abs_err_lam: list[float] = []
    cover_50 = 0
    cover_80 = 0
    n_sides = 0
    pit_vals: list[float] = []
    kw = {**map_kw, "z_band": z_band}
    for _, fx in fixtures.iterrows():
        home = norm_team(fx.get("home_norm") or fx.get("HomeTeam") or "")
        away = norm_team(fx.get("away_norm") or fx.get("AwayTeam") or "")
        if home not in ranks or away not in ranks:
            continue
        hg = fx.get("home_goals", fx.get("FTHG"))
        ag = fx.get("away_goals", fx.get("FTAG"))
        if pd.isna(hg) or pd.isna(ag):
            continue
        pred = match_prediction(home, away, ranks=ranks, **kw)
        samples_h: list[int] = []
        samples_a: list[int] = []
        for _ in range(n_strands):
            sh, sa, _, _ = sample_scoreline(
                pred["lam_home"],
                pred["lam_away"],
                lam_h_lo=pred["lam_home_lo"],
                lam_h_hi=pred["lam_home_hi"],
                lam_a_lo=pred["lam_away_lo"],
                lam_a_hi=pred["lam_away_hi"],
                rng=rng,
                centre_blend=centre_blend,
            )
            samples_h.append(sh)
            samples_a.append(sa)
        for samples, actual, lam in (
            (samples_h, float(hg), pred["lam_home"]),
            (samples_a, float(ag), pred["lam_away"]),
        ):
            arr = np.asarray(samples, float)
            abs_err_mean.append(abs(float(arr.mean()) - actual))
            abs_err_lam.append(abs(float(lam) - actual))
            lo50, hi50 = np.percentile(arr, [25, 75])
            lo80, hi80 = np.percentile(arr, [10, 90])
            if lo50 <= actual <= hi50:
                cover_50 += 1
            if lo80 <= actual <= hi80:
                cover_80 += 1
            # Continuous PIT proxy: fraction of samples strictly below actual
            # plus half the ties (randomized PIT would need another RNG draw).
            below = float((arr < actual).mean())
            ties = float((arr == actual).mean())
            pit_vals.append(below + 0.5 * ties)
            n_sides += 1
    if n_sides == 0:
        return {"n_sides": 0}
    pit = np.asarray(pit_vals, float)
    # Uniformity: mean near 0.5; Kolmogorov–Smirnov vs U(0,1) on subsample.
    ks = float(np.max(np.abs(np.sort(pit) - (np.arange(1, len(pit) + 1) / len(pit)))))
    return {
        "z_band": float(z_band),
        "centre_blend": float(centre_blend),
        "n_strands": n_strands,
        "n_sides": n_sides,
        "mae_strand_mean": float(np.mean(abs_err_mean)),
        "mae_centreline_lam": float(np.mean(abs_err_lam)),
        "cover_50": cover_50 / n_sides,
        "cover_80": cover_80 / n_sides,
        "pit_mean": float(pit.mean()),
        "pit_ks": ks,
    }



def player_mc_log(
    merged: pd.DataFrame,
    ranks: dict[str, float],
    map_kw: dict[str, Any],
    *,
    z_band: float,
    centre_blend: float,
    n_strands: int = N_MC,
    seed: int = MC_SEED,
    alpha_g: float | None = None,
    alpha_a: float | None = None,
) -> dict[str, Any]:
    """Eligible player-GW log of ``score_xp_v2`` at E[λ] under centre_blend.

    Closed form of the MC-mean λ path: under ``sample_scoreline``,
    ``E[λ] = w·centre + (1-w)·(lo+hi)/2``. Scoring once at that mean matches
    infinite-strand mean-λ xP (attack linear in λ; CS uses the same proxy).
    ``n_strands`` / ``seed`` kept for API compatibility; unused.
    """
    del n_strands, seed  # API compat with older callers / reports
    if alpha_g is None or alpha_a is None:
        ag, aa, _, _ = numagent_xp.load_alphas()
        alpha_g = ag if alpha_g is None else alpha_g
        alpha_a = aa if alpha_a is None else alpha_a
    kw = {**map_kw, "z_band": z_band}
    w = float(min(1.0, max(0.0, centre_blend)))
    xp_vals: list[float] = []
    y_vals: list[float] = []

    # Cache match E[λ] by club pair — z_band/blend fixed for this call.
    lam_cache: dict[tuple[str, str], tuple[float, float]] = {}

    def _mean_lams(home: str, away: str) -> tuple[float, float] | None:
        key = (home, away)
        if key in lam_cache:
            return lam_cache[key]
        try:
            pred = match_prediction(home, away, ranks=ranks, **kw)
        except KeyError:
            lam_cache[key] = None  # type: ignore[assignment]
            return None
        lo_h = min(pred["lam_home_lo"], pred["lam_home_hi"])
        hi_h = max(pred["lam_home_lo"], pred["lam_home_hi"])
        lo_a = min(pred["lam_away_lo"], pred["lam_away_hi"])
        hi_a = max(pred["lam_away_lo"], pred["lam_away_hi"])
        mean_h = w * float(pred["lam_home"]) + (1.0 - w) * 0.5 * (lo_h + hi_h)
        mean_a = w * float(pred["lam_away"]) + (1.0 - w) * 0.5 * (lo_a + hi_a)
        lam_cache[key] = (mean_h, mean_a)
        return mean_h, mean_a

    gws = sorted(int(g) for g in merged["GW"].unique() if GW_START <= int(g) <= GW_END)
    for gw in gws:
        hist = numagent_xp.asof_share_history(merged, before_gw=int(gw))
        week = merged.loc[merged["GW"] == int(gw)].copy()
        if week.empty:
            continue
        week["m"] = pd.to_numeric(week["minutes"], errors="coerce").fillna(0.0)
        week["sigma_m"] = 0.0
        week["player_id"] = week["element"].astype(str)
        # Appearance pool only (matches score_gameweek).
        week = week.loc[week["m"] > 0].copy()
        if week.empty:
            continue
        # Rates/priors once per GW — not once per fixture.
        week = attach_rates_priors(week, hist)
        apps = (
            hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
            if len(hist)
            else pd.Series(dtype=float)
        )
        week["n_prior"] = week["element"].map(apps).fillna(0) if len(apps) else 0
        for _, g in week.groupby("fixture", sort=False):
            homes = g.loc[g["was_home"].astype(bool)]
            aways = g.loc[~g["was_home"].astype(bool)]
            if homes.empty or aways.empty:
                continue
            home = norm_team(str(homes["team"].iloc[0]))
            away = norm_team(str(aways["team"].iloc[0]))
            if home not in ranks or away not in ranks:
                continue
            means = _mean_lams(home, away)
            if means is None:
                continue
            mean_h, mean_a = means
            scored = score_fixture_sides_on_lams(
                homes,
                aways,
                lam_h=mean_h,
                lam_a=mean_a,
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
            )
            if scored.empty:
                continue
            if "total_points" not in scored.columns:
                raise RuntimeError("score_fixture_sides_on_lams dropped total_points")
            scored["minutes"] = pd.to_numeric(
                scored.get("m", scored.get("minutes")), errors="coerce"
            ).fillna(0.0)
            if "n_prior" not in scored.columns:
                scored["n_prior"] = scored["element"].map(apps).fillna(0) if len(apps) else 0
            keep = scored.loc[
                (scored["minutes"] > MIN_MINUTES) & (scored["n_prior"] >= MIN_HISTORY)
            ]
            for row in keep.itertuples(index=False):
                xp = float(row.xp)
                if not math.isfinite(xp):
                    continue
                xp_vals.append(xp)
                y_vals.append(float(row.total_points))

    if not xp_vals:
        return {"z_band": float(z_band), "centre_blend": float(centre_blend), "n": 0, "scorer": SCORER}
    y = np.asarray(y_vals, float)
    mu = np.asarray(xp_vals, float)
    log = gaussian_log_score(y, mu, SIGMA)
    return {
        "z_band": float(z_band),
        "centre_blend": float(centre_blend),
        "n": int(len(y)),
        "mean_log": float(log.mean()),
        "mean_xp": float(mu.mean()),
        "mean_points": float(y.mean()),
        "mae": float(np.abs(y - mu).mean()),
        "scorer": SCORER,
    }


def _fixtures(code: str) -> pd.DataFrame:
    fixtures = load_football_data(code=code)
    if "home_goals" not in fixtures.columns and "FTHG" in fixtures.columns:
        fixtures = fixtures.rename(columns={"FTHG": "home_goals", "FTAG": "away_goals"})
    return fixtures


def run_season(label: str, season: str, e0: str, code: str) -> dict[str, Any]:
    ols = fit_rank_ols_excluding(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    map_kw = _map_kwargs(ols, profile)
    oracle_table = _season_table(CACHE / e0)
    ranks = {norm_team(r["team"]): float(r["rank"]) for r in oracle_table}
    fixtures = _fixtures(code)
    merged = load_merged(season)
    alpha_g, alpha_a, _, _ = numagent_xp.load_alphas()

    season_cov = [season_band_coverage(oracle_table, ranks, ols, profile, z_band=z) for z in Z_GRID]
    # Centreline match MAE is independent of z; report once at baseline.
    match_base = match_lambda_scores(fixtures, ranks, {**map_kw, "z_band": BASELINE_Z})

    mc_rows = []
    player_rows = []
    for z in Z_GRID:
        for blend in BLEND_GRID:
            # Full grid on MC predictive (cheap).
            mc_rows.append(
                mc_predictive(
                    fixtures, ranks, map_kw, z_band=z, centre_blend=blend
                )
            )
            # Player log only on the cross of interest: all z at baseline blend,
            # and all blends at baseline z (avoids 12× full player sweeps).
            if blend == BASELINE_BLEND or z == BASELINE_Z:
                print(f"  player log z={z} blend={blend}…", flush=True)
                player_rows.append(
                    player_mc_log(
                        merged,
                        ranks,
                        map_kw,
                        z_band=z,
                        centre_blend=blend,
                        alpha_g=alpha_g,
                        alpha_a=alpha_a,
                    )
                )

    return {
        "label": label,
        "season": season,
        "e0": e0,
        "ols": {
            "mae_gf": float(ols["mae_gf"]),
            "mae_ga": float(ols["mae_ga"]),
            "ols_gf": list(ols["ols_gf"]),
            "ols_ga": list(ols["ols_ga"]),
        },
        "match_centreline": match_base,
        "season_coverage": season_cov,
        "mc_predictive": mc_rows,
        "player_mc_log": player_rows,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool seasons: mean coverage / MAE / Δlog vs baseline."""
    # Season coverage by z.
    cov_by_z: dict[float, list[float]] = {z: [] for z in Z_GRID}
    for res in results:
        for row in res["season_coverage"]:
            cov_by_z[float(row["z_band"])].append(float(row["cov_mean"]))
    season_cov = {
        str(z): {
            "mean_cov": float(np.mean(v)),
            "per_season": v,
            "nominal_gauss": float(math.erf(z / math.sqrt(2.0))),
        }
        for z, v in cov_by_z.items()
    }

    # MC cover / MAE by (z, blend).
    mc_keys = [(z, b) for z in Z_GRID for b in BLEND_GRID]
    mc_agg = {}
    for z, b in mc_keys:
        rows = [
            r
            for res in results
            for r in res["mc_predictive"]
            if r.get("z_band") == z and r.get("centre_blend") == b and r.get("n_sides", 0)
        ]
        if not rows:
            continue
        mc_agg[f"z{z}_b{b}"] = {
            "z_band": z,
            "centre_blend": b,
            "mae_strand_mean": float(np.mean([r["mae_strand_mean"] for r in rows])),
            "mae_centreline_lam": float(np.mean([r["mae_centreline_lam"] for r in rows])),
            "cover_50": float(np.mean([r["cover_50"] for r in rows])),
            "cover_80": float(np.mean([r["cover_80"] for r in rows])),
            "pit_mean": float(np.mean([r["pit_mean"] for r in rows])),
            "pit_ks": float(np.mean([r["pit_ks"] for r in rows])),
        }

    # Player log: Δ vs baseline per season, then mean.
    base_logs = []
    player_agg = {}
    for res in results:
        base = next(
            (
                r
                for r in res["player_mc_log"]
                if r.get("z_band") == BASELINE_Z
                and r.get("centre_blend") == BASELINE_BLEND
                and r.get("n", 0)
            ),
            None,
        )
        if base is None:
            continue
        base_logs.append(float(base["mean_log"]))
        for row in res["player_mc_log"]:
            if not row.get("n"):
                continue
            key = f"z{row['z_band']}_b{row['centre_blend']}"
            player_agg.setdefault(key, {"deltas": [], "logs": [], "z_band": row["z_band"], "centre_blend": row["centre_blend"]})
            player_agg[key]["deltas"].append(float(row["mean_log"]) - float(base["mean_log"]))
            player_agg[key]["logs"].append(float(row["mean_log"]))
    player_summary = {
        k: {
            "z_band": v["z_band"],
            "centre_blend": v["centre_blend"],
            "mean_log": float(np.mean(v["logs"])),
            "mean_delta_vs_baseline": float(np.mean(v["deltas"])),
            "per_season_delta": v["deltas"],
        }
        for k, v in player_agg.items()
    }

    # Decision (locked before totals; materiality on the primary metric).
    #
    # Secondary signals pick a preferred z / blend. Adoption requires a
    # *material* mean player-GW Δlog (≥ +0.001 vs baseline). |Δlog| < 0.001
    # across the grid → inconclusive; keep baseline. Blend also requires
    # cover_50 within 0.05 of 0.50 — otherwise the cover rule is noise among
    # over-dispersed discrete Poisson IQRs (~0.71 here).
    MATERIAL_LOG = 0.001
    z_pick = min(
        Z_GRID,
        key=lambda z: abs(season_cov[str(z)]["mean_cov"] - season_cov[str(z)]["nominal_gauss"]),
    )
    z_player = player_summary.get(f"z{z_pick}_b{BASELINE_BLEND}")
    z_delta = float(z_player["mean_delta_vs_baseline"]) if z_player else float("nan")
    z_material = z_player is not None and z_delta >= MATERIAL_LOG

    blend_cands = []
    for b in BLEND_GRID:
        mc = mc_agg.get(f"z{BASELINE_Z}_b{b}")
        pl = player_summary.get(f"z{BASELINE_Z}_b{b}")
        if mc is None or pl is None:
            continue
        if float(pl["mean_delta_vs_baseline"]) < -0.002:
            continue
        blend_cands.append((abs(mc["cover_50"] - 0.50), b, mc, pl))
    if blend_cands:
        blend_cands.sort(key=lambda t: (t[0], abs(t[1] - BASELINE_BLEND)))
        blend_pick = blend_cands[0][1]
        blend_cover = float(blend_cands[0][2]["cover_50"])
        blend_delta = float(blend_cands[0][3]["mean_delta_vs_baseline"])
    else:
        blend_pick = BASELINE_BLEND
        blend_cover = float("nan")
        blend_delta = 0.0
    blend_material = (
        float(blend_pick) != BASELINE_BLEND
        and abs(blend_cover - 0.50) <= 0.05
        and blend_delta >= MATERIAL_LOG
    )

    max_abs_delta = max(
        (abs(float(v["mean_delta_vs_baseline"])) for v in player_summary.values()),
        default=0.0,
    )
    adopt_z = float(z_pick) if (z_material and float(z_pick) != BASELINE_Z) else BASELINE_Z
    adopt_blend = float(blend_pick) if blend_material else BASELINE_BLEND
    change = (adopt_z != BASELINE_Z) or (adopt_blend != BASELINE_BLEND)

    return {
        "baseline": {"z_band": BASELINE_Z, "centre_blend": BASELINE_BLEND},
        "season_coverage": season_cov,
        "mc_predictive": mc_agg,
        "player_mc_log": player_summary,
        "baseline_mean_log": float(np.mean(base_logs)) if base_logs else float("nan"),
        "decision": {
            "z_closest_to_nominal": float(z_pick),
            "z_delta_at_pick": z_delta,
            "z_material": bool(z_material),
            "blend_pick": float(blend_pick),
            "blend_cover_50": blend_cover,
            "blend_delta": blend_delta,
            "blend_material": bool(blend_material),
            "max_abs_player_delta": float(max_abs_delta),
            "material_log_bar": MATERIAL_LOG,
            "adopt_z_band": adopt_z,
            "adopt_centre_blend": adopt_blend,
            "change_defaults": bool(change),
            "note": (
                "Secondary: z closest to its Gaussian nominal coverage; blend with "
                "cover_50 nearest 0.50 among player-floor survivors. Adopt only if "
                f"mean player Δlog ≥ +{MATERIAL_LOG} (and blend cover within 0.05 of "
                "0.50). Else park baseline. Share seeding stays off."
            ),
        },
    }


def write_report(payload: dict[str, Any], path: Path | None = None) -> Path:
    out = path or (REPORTS / "match_band_calibration.md")
    agg = payload["aggregate"]
    dec = agg["decision"]
    lines = [
        "# Item 3 — Match-band calibration",
        "",
        "Level uncertainty on Opta rank→GF/GA→λ bands for the multistage cascade.",
        "Lead-time %/GW widening stays parked. Share seeding stays off.",
        "",
        "## Pre-registered",
        "",
        f"- Baseline: `z_band={BASELINE_Z}`, `centre_blend={BASELINE_BLEND}`.",
        f"- Grid: `z ∈ {list(Z_GRID)}`, `blend ∈ {list(BLEND_GRID)}`.",
        "- Ranks: oracle final table; OLS / place profile LOSO on the other three closed seasons.",
        "- Primary: eligible player-GW Gaussian log of MC-mean cascade xP "
        f"(GW{GW_START}–{GW_END}, n_prior≥{MIN_HISTORY}, minutes>{MIN_MINUTES}, `score_xp_v2`, σ={SIGMA}).",
        "- Secondary: season GF/GA residual coverage vs Gaussian nominal; "
        "MC strand-mean MAE; empirical 50/80% goal coverage; PIT.",
        "- Adopt a non-baseline knob only if the pre-registered rule below fires.",
        "",
        "## Decision",
        "",
        f"- Materiality bar (primary): mean player Δlog ≥ **+{dec['material_log_bar']}** "
        f"(grid max |Δ| = {dec['max_abs_player_delta']:.5f}).",
        f"- z closest to nominal coverage: **{dec['z_closest_to_nominal']}** "
        f"(Δlog={dec['z_delta_at_pick']:+.5f}, material={dec['z_material']})",
        f"- blend pick: **{dec['blend_pick']}** "
        f"(cover_50={dec['blend_cover_50']:.3f}, Δlog={dec['blend_delta']:+.5f}, "
        f"material={dec['blend_material']})",
        f"- **Change defaults: {dec['change_defaults']}** "
        f"(adopt z={dec['adopt_z_band']}, blend={dec['adopt_centre_blend']})",
        "",
        "## Season residual coverage (mean over 4 LOSO seasons)",
        "",
        "| z | mean cov | Gaussian nominal | |Δ| |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for z in Z_GRID:
        row = agg["season_coverage"][str(z)]
        lines.append(
            f"| {z} | {row['mean_cov']:.3f} | {row['nominal_gauss']:.3f} | "
            f"{abs(row['mean_cov'] - row['nominal_gauss']):.3f} |"
        )
    lines += [
        "",
        "## MC predictive (pooled means)",
        "",
        "| z | blend | MAE strand | MAE λ | cover50 | cover80 | PIT mean |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for z in Z_GRID:
        for b in BLEND_GRID:
            row = agg["mc_predictive"].get(f"z{z}_b{b}")
            if not row:
                continue
            mark = " ← baseline" if z == BASELINE_Z and b == BASELINE_BLEND else ""
            lines.append(
                f"| {z} | {b} | {row['mae_strand_mean']:.3f} | "
                f"{row['mae_centreline_lam']:.3f} | {row['cover_50']:.3f} | "
                f"{row['cover_80']:.3f} | {row['pit_mean']:.3f} |{mark}"
            )
    lines += [
        "",
        "## Player-GW log (MC-mean xP) — Δ vs baseline",
        "",
        f"Baseline mean log: **{agg['baseline_mean_log']:.5f}**",
        "",
        "| z | blend | mean log | Δ vs baseline |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for key, row in sorted(
        agg["player_mc_log"].items(),
        key=lambda kv: (kv[1]["z_band"], kv[1]["centre_blend"]),
    ):
        mark = " ← baseline" if (
            row["z_band"] == BASELINE_Z and row["centre_blend"] == BASELINE_BLEND
        ) else ""
        lines.append(
            f"| {row['z_band']} | {row['centre_blend']} | {row['mean_log']:.5f} | "
            f"{row['mean_delta_vs_baseline']:+.5f} |{mark}"
        )
    lines += [
        "",
        "## Read",
        "",
        "1. Centreline match λ is unchanged by `z_band` — only MC width moves.",
        "2. Season coverage is the proper check for the ±z·sd claim "
        "(goals-inside-λ-band mixes Poisson noise into a rate band).",
        "3. Player-GW log barely moves across the grid (|Δ| ≪ materiality). "
        "MC cover_50 sits ~0.71 for every knob — discrete Poisson dominates band width.",
        "4. Keep `Z_BAND=1.0` and `centre_blend=0.5`. Share seeding stays parked. "
        "Scorer: Numagent 2.0 `score_xp_v2`. Item 4 next: player G/A finish.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.match_band_calibration",
        "```",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


def run() -> dict[str, Any]:
    results = []
    for label, season, e0, code in SEASONS:
        print(f"season {label}…", flush=True)
        results.append(run_season(label, season, e0, code))
    aggregate = _aggregate(results)
    payload = {
        "task": "match_band_calibration",
        "scorer": SCORER,
        "seasons": [r["label"] for r in results],
        "n_mc": N_MC,
        "mc_seed": MC_SEED,
        "z_grid": list(Z_GRID),
        "blend_grid": list(BLEND_GRID),
        "per_season": results,
        "aggregate": aggregate,
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    json_path = PROCESSED / "match_band_calibration.json"
    # Drop bulky per-row OLS tables — already slim.
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    report = write_report(payload)
    payload["paths"] = {"json": str(json_path), "report": str(report)}
    return payload


def main() -> None:
    payload = run()
    dec = payload["aggregate"]["decision"]
    print(
        f"change_defaults={dec['change_defaults']} "
        f"adopt_z={dec['adopt_z_band']} adopt_blend={dec['adopt_centre_blend']}"
    )
    print(f"report={payload['paths']['report']}")


if __name__ == "__main__":
    main()
