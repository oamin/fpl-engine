"""Overnight cascade short→long viability batch (A→D).

Runs integrity, player-GW likelihood arms (B1–B3), and FT+captain sanity
without interactive gates. Writes incremental JSON + a morning report.

  PYTHONPATH=. python3 -m src.eval.cascade_viability_batch
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    run_asof_audit,
    write_gated_report,
)
from src.eval.oracle_rank_cascade import _season_table
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import SCORER, side_moments_from_lams
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm, score_week_generative
from src.models.opta_gf_ga_horizon import match_prediction
from src.models.short_horizon_cascade import (
    DEFAULT_LINEUP_COV,
    MARKET_BAND,
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
    score_week_on_fixture_moments,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "data" / "processed" / "cascade_viability"
REPORT_MD = REPORTS / "cascade_short_long_viability_results.md"
STATE_JSON = OUT_DIR / "batch_state.json"

# Likelihood window — skip early GWs with thin rates.
GW_START = 8
GW_END = 28
MIN_MINUTES = 60.0
MIN_HISTORY = 3
N_MC = 6
SIGMA = 3.0
BOOT = 2000
SEED = 7

E0 = {
    "2022-23": "E0_2223.csv",
    "2023-24": "E0_2324.csv",
    "2024-25": "E0_2425.csv",
    "2025-26": "E0_2526.csv",
}


def _save(state: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _now() -> float:
    return time.time()


def run_A_integrity(state: dict[str, Any]) -> None:
    t0 = _now()
    results: dict[str, Any] = {"started": t0}
    # A1 as-of audit
    audit = run_asof_audit()
    results["A1_asof"] = audit
    # A2 no jitter
    from src.models.multistage_mc_milp import sample_priced_scenario

    jitter_ok = MARKET_BAND == 0.0
    raised = False
    try:
        sample_priced_scenario(
            {
                "home": "arsenal",
                "away": "everton",
                "lam_h": 2.0,
                "lam_a": 1.0,
                "rho": 0.0,
            },
            random.Random(0),
            band=0.15,
        )
    except ValueError:
        raised = True
    results["A2_no_jitter"] = {
        "MARKET_BAND": MARKET_BAND,
        "band_is_zero": jitter_ok,
        "rejects_nonzero_band": raised,
        "passed": bool(jitter_ok and raised),
    }
    # A3 k∼M support
    from src.markets.match_outcome import (
        match_outcome_probs_from_lambdas,
        match_outcome_roll,
    )

    probs = match_outcome_probs_from_lambdas(2.0, 0.8, -0.1)
    rng = random.Random(0)
    bad = 0
    gh_sum = ga_sum = 0.0
    n = 2000
    for _ in range(n):
        roll = match_outcome_roll(probs, rng)
        gh_sum += roll.goals_home
        ga_sum += roll.goals_away
        if float(probs.matrix[roll.i, roll.j]) <= 0.0:
            bad += 1
    results["A3_k_in_M"] = {
        "n": n,
        "bad_cells": bad,
        "mean_gh": gh_sum / n,
        "mean_ga": ga_sum / n,
        "xg_home": float(probs.moments["xg_home"]),
        "xg_away": float(probs.moments["xg_away"]),
        "passed": bad == 0,
    }
    # A4/A5 routing + props via unit-like smoke on GW8 odds if present
    odds = ROOT / "data/scratch/gw08_odds_full.json"
    routing: dict[str, Any] = {"odds_present": odds.is_file()}
    if odds.is_file():
        from src.models.multistage_mc_milp import (
            load_priced_week_fits,
            sample_week_scenario,
            init_paths_from_opta,
            _priced_index,
        )
        from src.models.opta_gf_ga_horizon import league_avg_gf, rank_gf_ga_profile

        boot_path = (
            ROOT
            / "data/predictions/2026-27/gw06/betfair_20261008/bootstrap.json"
        )
        boot = json.loads(boot_path.read_text())
        fits = load_priced_week_fits(odds, boot, gw=8)
        pidx = _priced_index({8: fits})
        paths = init_paths_from_opta()
        lg = league_avg_gf(rank_gf_ga_profile())
        rng = random.Random(1)
        short = sample_week_scenario(
            fits[0]["home"],
            fits[0]["away"],
            gw=8,
            priced_index=pidx,
            paths=paths,
            league_gf=lg,
            rng=rng,
        )
        long = sample_week_scenario(
            fits[0]["home"],
            fits[0]["away"],
            gw=12,
            priced_index=pidx,
            paths=paths,
            league_gf=lg,
            rng=rng,
        )
        routing.update(
            {
                "n_fits": len(fits),
                "n_props": int(sum(f.get("n_props_matched", 0) for f in fits)),
                "gw8_source": short.get("source"),
                "gw12_source": long.get("source", "cascade_long_opta"),
                "gw8_lam_matches_fit": abs(short["lam_h"] - fits[0]["lam_h"]) < 1e-12,
                "gw8_has_props": bool(short.get("home_props") or short.get("away_props")),
                "gw12_props_empty": not bool(
                    long.get("home_props") or long.get("away_props")
                ),
                "passed": (
                    short.get("source") == "cascade_short_priced"
                    and abs(short["lam_h"] - fits[0]["lam_h"]) < 1e-12
                    and long.get("source", "cascade_long_opta") != "cascade_short_priced"
                ),
            }
        )
    results["A4_A5_routing_props"] = routing
    # A6 FT/captain legality — light: run existing unit tests
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_multistage_mc_milp.XiCaptainMilpTest",
            "tests.test_short_horizon_v2",
            "-v",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    results["A6_milp_units"] = {
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stderr_tail": (proc.stderr or "")[-800:],
    }
    results["elapsed_s"] = _now() - t0
    results["passed"] = bool(
        audit.get("passed")
        and results["A2_no_jitter"]["passed"]
        and results["A3_k_in_M"]["passed"]
        and results.get("A4_A5_routing_props", {}).get("passed", False)
        and results["A6_milp_units"]["passed"]
    )
    state["A"] = results
    _save(state)
    print(f"[A] integrity passed={results['passed']} ({results['elapsed_s']:.1f}s)", flush=True)


def _oracle_ranks(season: str) -> dict[str, float]:
    table = _season_table(ROOT / "data" / "cache" / E0[season])
    return {norm_team(r["team"]): float(r["rank"]) for r in table}


def _gen_mean_board(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    hist_by_team: dict[str, pd.DataFrame],
    rng: random.Random,
    n_mc: int,
) -> pd.DataFrame:
    """MC-average generative xP over centreline M rolls."""
    if side.empty or not fits:
        return pd.DataFrame()
    acc: dict[int, float] = defaultdict(float)
    meta: dict[int, dict[str, Any]] = {}
    for i in range(int(n_mc)):
        board = score_week_generative(
            side,
            fits,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            rng=rng,
            hist_by_team=hist_by_team,
            lineup_cov=DEFAULT_LINEUP_COV,
            lam_overrides=None,
        )
        if board.empty:
            continue
        for row in board.itertuples(index=False):
            eid = int(row.element)
            acc[eid] += float(row.score_xp_v2)
            if eid not in meta:
                meta[eid] = {
                    "element": eid,
                    "name": getattr(row, "name", ""),
                    "position": getattr(row, "position", ""),
                    "team": getattr(row, "team", ""),
                }
    if not acc:
        return pd.DataFrame()
    rows = []
    for eid, total in acc.items():
        r = dict(meta[eid])
        r["xp"] = total / float(n_mc)
        r[SCORER] = r["xp"]
        rows.append(r)
    return pd.DataFrame(rows)


def _opta_fits_for_gw(
    week: pd.DataFrame,
    ranks: dict[str, float],
) -> list[dict[str, Any]]:
    fits: list[dict[str, Any]] = []
    for fid, g in week.groupby("fixture", sort=False):
        homes = g.loc[g["was_home"].astype(bool)]
        aways = g.loc[~g["was_home"].astype(bool)]
        if homes.empty or aways.empty:
            continue
        home = str(homes["team"].iloc[0])
        away = str(aways["team"].iloc[0])
        hn, an = norm_team(home), norm_team(away)
        if hn not in ranks or an not in ranks:
            continue
        try:
            pred = match_prediction(hn, an, ranks=ranks)
        except KeyError:
            continue
        fits.append(
            {
                "fixture": int(fid),
                "home": home,
                "away": away,
                "home_n": hn,
                "away_n": an,
                "lam_h": float(pred["lam_home"]),
                "lam_a": float(pred["lam_away"]),
                "rho": 0.0,
                "xg_home": float(pred["lam_home"]),
                "xg_away": float(pred["lam_away"]),
                "xcs_home": float(pred.get("p_cs_home") or 0.0),
                "xcs_away": float(pred.get("p_cs_away") or 0.0),
                "xgc_home": float(side_moments_from_lams(pred["lam_home"], pred["lam_away"])["xgc"]),
                "xgc_away": float(side_moments_from_lams(pred["lam_away"], pred["lam_home"])["xgc"]),
            }
        )
    return fits


def _pair_rows(
    mean_board: pd.DataFrame,
    gen_board: pd.DataFrame,
    week: pd.DataFrame,
    hist: pd.DataFrame,
    *,
    season: str,
    gw: int,
    prefix: str,
) -> list[dict[str, Any]]:
    if mean_board.empty or gen_board.empty:
        return []
    pts = week.groupby("element", as_index=False).agg(
        total_points=("total_points", "sum"),
        minutes=("minutes", "sum"),
    )
    apps = (
        hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
        if len(hist)
        else pd.Series(dtype=float)
    )
    m = mean_board[["element", "xp"]].rename(columns={"xp": f"{prefix}_mean"})
    g = gen_board[["element", "xp"]].rename(columns={"xp": f"{prefix}_gen"})
    j = pts.merge(m, on="element").merge(g, on="element")
    j["n_prior"] = j["element"].map(apps).fillna(0)
    j = j.loc[
        (pd.to_numeric(j["minutes"], errors="coerce") > MIN_MINUTES)
        & (j["n_prior"] >= MIN_HISTORY)
    ]
    out = []
    for row in j.itertuples(index=False):
        out.append(
            {
                "season": season,
                "gw": int(gw),
                "element": int(row.element),
                "total_points": float(row.total_points),
                "minutes": float(row.minutes),
                f"{prefix}_mean": float(getattr(row, f"{prefix}_mean")),
                f"{prefix}_gen": float(getattr(row, f"{prefix}_gen")),
            }
        )
    return out


def run_B_likelihood(state: dict[str, Any]) -> None:
    t0 = _now()
    rows_short: list[dict[str, Any]] = []
    rows_long: list[dict[str, Any]] = []
    per_season: dict[str, Any] = {}
    rng = random.Random(SEED)

    for season in CLOSED_SEASONS:
        print(f"[B] season {season}…", flush=True)
        st0 = _now()
        merged = load_merged(season)
        # Prefer enriched for fixture schedule when available.
        try:
            enriched = _load_merged_enriched(season)
            fd_idx = _fd_index(season)
            use_enriched = True
        except Exception as exc:  # noqa: BLE001
            enriched = merged
            fd_idx = None
            use_enriched = False
            print(f"  enriched unavailable ({exc}); FD fits limited", flush=True)
        ranks = _oracle_ranks(season)
        n_gw = 0
        for gw in range(GW_START, GW_END + 1):
            hist = numagent_xp.asof_share_history(merged, before_gw=int(gw), season=season)
            week = merged.loc[merged["GW"] == int(gw)].copy()
            if week.empty or hist.empty:
                continue
            alpha_g, alpha_a = _asof_alphas(hist)
            side, alpha_g, alpha_a = build_frozen_side_table(
                hist, alpha_g=alpha_g, alpha_a=alpha_a
            )
            if side.empty:
                continue
            team_hist = hist_by_team_norm(hist)

            # Short: FD centreline fits
            short_fits: list[dict[str, Any]] = []
            gen_s = pd.DataFrame()
            if use_enriched and fd_idx is not None:
                fx = fixtures_for_gw(enriched, int(gw))
                short_fits = centreline_fixture_fits(fx, fd_idx)
            if len(short_fits) >= 5:
                mean_s = score_week_on_fixture_moments(
                    side, short_fits, alpha_g=alpha_g, alpha_a=alpha_a
                )
                gen_s = _gen_mean_board(
                    side,
                    short_fits,
                    alpha_g=alpha_g,
                    alpha_a=alpha_a,
                    hist_by_team=team_hist,
                    rng=rng,
                    n_mc=N_MC,
                )
                if not mean_s.empty and "xp" not in mean_s.columns and SCORER in mean_s.columns:
                    mean_s = mean_s.rename(columns={SCORER: "xp"})
                rows_short.extend(
                    _pair_rows(
                        mean_s, gen_s, week, hist, season=season, gw=gw, prefix="short"
                    )
                )

            # Long: oracle-rank Opta map
            long_fits = _opta_fits_for_gw(week, ranks)
            if len(long_fits) >= 5:
                mean_l = score_week_on_fixture_moments(
                    side, long_fits, alpha_g=alpha_g, alpha_a=alpha_a
                )
                gen_l = _gen_mean_board(
                    side,
                    long_fits,
                    alpha_g=alpha_g,
                    alpha_a=alpha_a,
                    hist_by_team=team_hist,
                    rng=rng,
                    n_mc=N_MC,
                )
                if not mean_l.empty and "xp" not in mean_l.columns and SCORER in mean_l.columns:
                    mean_l = mean_l.rename(columns={SCORER: "xp"})
                rows_long.extend(
                    _pair_rows(
                        mean_l, gen_l, week, hist, season=season, gw=gw, prefix="long"
                    )
                )
            n_gw += 1
            if n_gw % 5 == 0:
                print(f"  GW{gw} done ({n_gw} weeks)", flush=True)
        per_season[season] = {
            "elapsed_s": _now() - st0,
            "n_short_rows": sum(1 for r in rows_short if r["season"] == season),
            "n_long_rows": sum(1 for r in rows_long if r["season"] == season),
        }
        _save({**state, "B_progress": per_season})

    def _gw_deltas(rows: list[dict[str, Any]], left: str, right: str) -> list[dict[str, Any]]:
        out = []
        by = defaultdict(list)
        for r in rows:
            by[(r["season"], r["gw"])].append(r)
        for (season, gw), block in sorted(by.items()):
            y = np.array([b["total_points"] for b in block], float)
            a = np.array([b[left] for b in block], float)
            b = np.array([b[right] for b in block], float)
            if len(y) < 10:
                continue
            la = gaussian_log_score(y, a, SIGMA)
            lb = gaussian_log_score(y, b, SIGMA)
            out.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "n_rows": int(len(y)),
                    "delta": float((la - lb).mean()),
                    "mean_log_a": float(la.mean()),
                    "mean_log_b": float(lb.mean()),
                    "comparison": comparison_key(left, right),
                }
            )
        return out

    b1_rows = _gw_deltas(rows_short, "short_gen", "short_mean")
    b2_rows = _gw_deltas(rows_long, "long_gen", "long_mean")

    # B3: join short_gen vs long_gen on (season,gw,element)
    short_df = pd.DataFrame(rows_short)
    long_df = pd.DataFrame(rows_long)
    b3_rows: list[dict[str, Any]] = []
    if not short_df.empty and not long_df.empty:
        j = short_df.merge(
            long_df[["season", "gw", "element", "long_gen", "long_mean"]],
            on=["season", "gw", "element"],
            how="inner",
        )
        b3_rows = _gw_deltas(j.to_dict("records"), "short_gen", "long_gen")

    def _interval(gw_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
        by_season = {
            s: np.asarray(
                [r["delta"] for r in gw_rows if r["season"] == s and r["comparison"] == key],
                dtype=float,
            )
            for s in CLOSED_SEASONS
        }
        return cluster_interval(by_season, seasons=CLOSED_SEASONS, n_boot=BOOT, seed=SEED)

    key_b1 = comparison_key("short_gen", "short_mean")
    key_b2 = comparison_key("long_gen", "long_mean")
    key_b3 = comparison_key("short_gen", "long_gen")
    intervals = {
        key_b1: _interval(b1_rows, key_b1) if b1_rows else {"error": "no_rows"},
        key_b2: _interval(b2_rows, key_b2) if b2_rows else {"error": "no_rows"},
        key_b3: _interval(b3_rows, key_b3) if b3_rows else {"error": "no_rows"},
    }

    def _verdict(iv: dict[str, Any]) -> str:
        if "error" in iv:
            return "no_data"
        lo, hi = float(iv["lo"]), float(iv["hi"])
        if hi < 0:
            return "clearly_loses"
        if lo > 0:
            return "clearly_wins"
        return "does_not_clearly_lose"

    state["B"] = {
        "elapsed_s": _now() - t0,
        "gw_start": GW_START,
        "gw_end": GW_END,
        "n_mc": N_MC,
        "sigma": SIGMA,
        "per_season": per_season,
        "n_gw_pairs_b1": len(b1_rows),
        "n_gw_pairs_b2": len(b2_rows),
        "n_gw_pairs_b3": len(b3_rows),
        "intervals": intervals,
        "verdicts": {k: _verdict(v) for k, v in intervals.items()},
        "note": (
            "B1: short generative MC-mean vs FD mean layer. "
            "B2: oracle-rank Opta generative vs Opta mean (diagnostic ranks). "
            "B3: short_gen vs long_gen on overlapping player-GWs (priced info value)."
        ),
    }
    # Persist row samples for morning inspection
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(b1_rows).to_csv(OUT_DIR / "b1_gw_deltas.csv", index=False)
    pd.DataFrame(b2_rows).to_csv(OUT_DIR / "b2_gw_deltas.csv", index=False)
    pd.DataFrame(b3_rows).to_csv(OUT_DIR / "b3_gw_deltas.csv", index=False)
    _save(state)
    print(f"[B] done verdicts={state['B']['verdicts']} ({state['B']['elapsed_s']:.1f}s)", flush=True)


def run_C_ft_captain(state: dict[str, Any]) -> None:
    t0 = _now()
    out = ROOT / "data/predictions/2026-27/cascade_viability_ft"
    cmd = [
        sys.executable,
        "-m",
        "src.models.multistage_mc_milp",
        "--decision-gw",
        "6",
        "--horizon-end",
        "19",
        "--n",
        "12",
        "--seed",
        "7",
        "--snapshot-every",
        "1",
        "--objective",
        "xi_plus_captain_ft",
        "--no-minutes-overlay",
        "--priced-gw",
        "8",
        "--out",
        str(out),
    ]
    print("[C] running FT+captain N=12…", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    trial_path = out / "multistage_xi_captain_ft_trial.json"
    trial: dict[str, Any] = {}
    if trial_path.is_file():
        trial = json.loads(trial_path.read_text())
    # Captain / squad_sum contrast: second run with xi_plus_captain (no FT path)
    out2 = ROOT / "data/predictions/2026-27/cascade_viability_xi_cap"
    cmd2 = cmd.copy()
    cmd2[cmd2.index("xi_plus_captain_ft")] = "xi_plus_captain"
    cmd2[cmd2.index(str(out))] = str(out2)
    print("[C] running xi_plus_captain N=12…", flush=True)
    proc2 = subprocess.run(
        cmd2,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    trial2: dict[str, Any] = {}
    tp2 = out2 / "multistage_xi_captain_trial.json"
    # write_trial_outputs stem for xi_plus_captain is multistage_xi_captain
    if not tp2.is_file():
        cands = list(out2.glob("*trial.json"))
        if cands:
            tp2 = cands[0]
    if tp2.is_file():
        trial2 = json.loads(tp2.read_text())

    def _top(trial_obj: dict[str, Any], n: int = 10) -> list[dict[str, Any]]:
        freqs = trial_obj.get("frequencies") or []
        return [
            {
                "name": r.get("name"),
                "freq": r.get("freq"),
                "freq_xi": r.get("freq_xi"),
                "freq_captain": r.get("freq_captain"),
                "team": r.get("team_norm"),
            }
            for r in freqs[:n]
        ]

    sources = {
        s.get("source")
        for s in (trial.get("sample_strand_scorelines") or [])
        if isinstance(s, dict)
    }
    state["C"] = {
        "elapsed_s": _now() - t0,
        "ft": {
            "returncode": proc.returncode,
            "n_squads": trial.get("n_squads_solved"),
            "mean_transfers": trial.get("mean_transfers"),
            "mean_hits": trial.get("mean_hits"),
            "props": trial.get("props"),
            "priced": trial.get("priced"),
            "scoring": trial.get("scoring"),
            "sources_in_sample": sorted(sources),
            "top10": _top(trial),
            "stdout_tail": (proc.stdout or "")[-600:],
            "stderr_tail": (proc.stderr or "")[-600:],
        },
        "xi_captain": {
            "returncode": proc2.returncode,
            "n_squads": trial2.get("n_squads_solved"),
            "top10": _top(trial2),
        },
        "passed_smoke": proc.returncode == 0 and int(trial.get("n_squads_solved") or 0) > 0,
        "note": (
            "Diagnostic frequency only — not a gated path-V promote. "
            "Confirm GW8 source is cascade_short_priced and scoring mentions no λ jitter."
        ),
    }
    _save(state)
    print(
        f"[C] FT squads={state['C']['ft'].get('n_squads')} "
        f"hits={state['C']['ft'].get('mean_hits')} ({state['C']['elapsed_s']:.1f}s)",
        flush=True,
    )


def run_D_smoke(state: dict[str, Any]) -> None:
    t0 = _now()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_multistage_mc_milp",
            "tests.test_opta_gf_ga_horizon",
            "tests.test_short_horizon_v2",
            "-v",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    banner = REPORTS / "multistage_xi_captain_ft_short_long_trial.md"
    contaminated = False
    if banner.is_file():
        contaminated = "CONTAMINATED" in banner.read_text(encoding="utf-8")
    state["D"] = {
        "elapsed_s": _now() - t0,
        "unittest_returncode": proc.returncode,
        "unittest_passed": proc.returncode == 0,
        "contaminated_banner": contaminated,
        "MARKET_BAND": MARKET_BAND,
        "passed": proc.returncode == 0 and MARKET_BAND == 0.0 and contaminated,
    }
    _save(state)
    print(f"[D] smoke passed={state['D']['passed']}", flush=True)


def write_morning_report(state: dict[str, Any]) -> None:
    audit = run_asof_audit()
    b = state.get("B") or {}
    intervals = b.get("intervals") or {}
    # Only write gated report when we have all four-season intervals.
    gated_ok = False
    try:
        clean = {k: v for k, v in intervals.items() if "error" not in v and "lo" in v}
        if len(clean) >= 1 and audit.get("passed"):
            # write_gated_report requires ALL closed seasons present in every interval
            write_gated_report(
                REPORTS / "cascade_short_long_viability_gated.md",
                audit,
                clean,
                [
                    "# Cascade short→long viability — gated likelihood",
                    "",
                    f"N_MC={N_MC} · GW{GW_START}–{GW_END} · sigma={SIGMA}",
                    "",
                    "## Intervals (treatment − baseline)",
                    "",
                ]
                + [
                    f"- `{k}`: lo={v.get('lo'):+.4f} hi={v.get('hi'):+.4f} "
                    f"mean={v.get('mean'):+.4f} · verdict={b.get('verdicts', {}).get(k)}"
                    for k, v in clean.items()
                ]
                + ["", "See `cascade_short_long_viability_results.md` for full batch."],
            )
            gated_ok = True
    except Exception as exc:  # noqa: BLE001
        state["gated_report_error"] = str(exc)

    lines = [
        "# Cascade short→long viability — overnight results",
        "",
        f"Batch finished. Locked stack: centreline `M` → `k ∼ M` (MARKET_BAND={MARKET_BAND}).",
        "",
        "## A — Integrity",
        "",
        f"- Passed: **{(state.get('A') or {}).get('passed')}**",
        f"- As-of audit: **{(state.get('A') or {}).get('A1_asof', {}).get('passed')}**",
        f"- No λ jitter: **{(state.get('A') or {}).get('A2_no_jitter', {}).get('passed')}**",
        f"- k∈supp(M): **{(state.get('A') or {}).get('A3_k_in_M', {}).get('passed')}**",
        f"- Routing/props: **{(state.get('A') or {}).get('A4_A5_routing_props', {}).get('passed')}**",
        f"- MILP units: **{(state.get('A') or {}).get('A6_milp_units', {}).get('passed')}**",
        "",
        "## B — Likelihood (player-GW log)",
        "",
        f"- Window GW{GW_START}–{GW_END} · N_MC={N_MC} · σ={SIGMA}",
        f"- Verdicts: `{b.get('verdicts')}`",
        f"- GW pairs: B1={b.get('n_gw_pairs_b1')} B2={b.get('n_gw_pairs_b2')} B3={b.get('n_gw_pairs_b3')}",
        "",
    ]
    for k, v in (intervals or {}).items():
        if "error" in v:
            lines.append(f"- `{k}`: {v['error']}")
        else:
            lines.append(
                f"- `{k}`: Δ mean={v.get('mean'):+.4f} "
                f"[{v.get('lo'):+.4f}, {v.get('hi'):+.4f}] · "
                f"**{b.get('verdicts', {}).get(k)}**"
            )
    lines += [
        "",
        f"- Gated markdown written: **{gated_ok}**",
        "",
        "## C — FT + captain (diagnostic)",
        "",
    ]
    c = state.get("C") or {}
    ft = c.get("ft") or {}
    lines += [
        f"- Smoke passed: **{c.get('passed_smoke')}**",
        f"- Squads: {ft.get('n_squads')} · mean transfers {ft.get('mean_transfers')} · "
        f"mean hits {ft.get('mean_hits')}",
        f"- Sources in sample: {ft.get('sources_in_sample')}",
        f"- Scoring: {ft.get('scoring')}",
        "",
        "Top FT freq:",
        "",
    ]
    for r in ft.get("top10") or []:
        lines.append(
            f"- {r.get('name')} ({r.get('team')}): freq={r.get('freq'):.3f} "
            f"xi={r.get('freq_xi')} cap={r.get('freq_captain')}"
        )
    lines += [
        "",
        "## D — Smoke",
        "",
        f"- Passed: **{(state.get('D') or {}).get('passed')}**",
        "",
        "## Morning decision aid",
        "",
        "| Arm | Result |",
        "| --- | --- |",
        f"| A integrity | {(state.get('A') or {}).get('passed')} |",
        f"| B1 short gen−mean | {(b.get('verdicts') or {}).get(comparison_key('short_gen','short_mean'))} |",
        f"| B2 long gen−mean | {(b.get('verdicts') or {}).get(comparison_key('long_gen','long_mean'))} |",
        f"| B3 short_gen−long_gen | {(b.get('verdicts') or {}).get(comparison_key('short_gen','long_gen'))} |",
        f"| C FT+cap smoke | {c.get('passed_smoke')} |",
        f"| D regression | {(state.get('D') or {}).get('passed')} |",
        "",
        "Artifacts: `data/processed/cascade_viability/`",
        "",
        "Reproduce:",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_viability_batch",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    state["report"] = str(REPORT_MD)
    _save(state)


def main() -> None:
    state: dict[str, Any] = {
        "task": "cascade_short_long_viability_batch",
        "started_unix": _now(),
        "MARKET_BAND": MARKET_BAND,
        "params": {
            "gw_start": GW_START,
            "gw_end": GW_END,
            "n_mc": N_MC,
            "sigma": SIGMA,
            "seasons": list(CLOSED_SEASONS),
        },
    }
    _save(state)
    errors: list[str] = []
    for name, fn in (
        ("D", run_D_smoke),
        ("A", run_A_integrity),
        ("B", run_B_likelihood),
        ("C", run_C_ft_captain),
    ):
        try:
            fn(state)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            errors.append(f"{name}: {exc}")
            state[f"{name}_error"] = tb
            _save(state)
            print(f"[{name}] FAILED: {exc}", flush=True)
            print(tb, flush=True)
    try:
        write_morning_report(state)
    except Exception as exc:  # noqa: BLE001
        state["report_error"] = traceback.format_exc()
        _save(state)
        print(f"[report] FAILED: {exc}", flush=True)
    state["finished_unix"] = _now()
    state["elapsed_s"] = state["finished_unix"] - state["started_unix"]
    state["errors"] = errors
    _save(state)
    (ROOT / "REPORT_STATE.json").write_text(
        json.dumps(
            {
                "task": "cascade_short_long_viability_batch",
                "status": "done" if not errors else "done_with_errors",
                "elapsed_s": state["elapsed_s"],
                "errors": errors,
                "report": str(REPORT_MD),
                "state": str(STATE_JSON),
                "verdicts": (state.get("B") or {}).get("verdicts"),
                "A_passed": (state.get("A") or {}).get("passed"),
                "C_passed": (state.get("C") or {}).get("passed_smoke"),
                "D_passed": (state.get("D") or {}).get("passed"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[batch] finished in {state['elapsed_s']:.1f}s errors={errors}", flush=True)


if __name__ == "__main__":
    main()
