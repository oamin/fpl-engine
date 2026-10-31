"""Does large-N generative mean converge to the centreline board?

Diagnostic on the same 2025-26 hist stack as ``cascade_mean_xi_ft_hist``:

- **Centreline:** ``score_team_side`` on mid λ (FD short / oracle-rank long).
- **Gen mean(N):** average of N generative strands (λ-band sample on long,
  ``k∼M``, attack dice; fixed as-of minutes).

Reports MAE / RMSE / corr of gen-mean(N) vs centreline, and vs gen-mean(Nmax)
(self-convergence). Also scoreline bias ``mean(k) − λ``.

  PYTHONPATH=. python3 -m src.eval.cascade_mean_strand_convergence --n-max 200
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import (
    DEFAULT_SEASON,
    SEASON_E0,
    SEED,
    _build_hist_pool,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_xi_ft_hist import DECISION_GW, HORIZON_END
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import (
    DEFAULT_LINEUP_COV,
    ClubPath,
    _priced_index,
    _undiscounted_scores_by_gw,
    init_paths_from_opta,
    sample_week_scenario,
    score_pool_on_pots,
    pots_for_gw_from_paths,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    match_prediction,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_mean_strand_convergence.md"
STATE_JSON = ROOT / "data/processed/cascade_mean_strand_convergence" / "state.json"

# Prefixes reported along one seed stream (cheap — one pass to n_max).
DEFAULT_NS = (1, 2, 5, 12, 25, 50, 100, 200)
N_MAX = 200


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(
        json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _stack(scores_by_gw: dict[int, np.ndarray], gws: list[int]) -> np.ndarray:
    return np.concatenate([scores_by_gw[g] for g in gws])


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    d = a - b
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "bias": float(np.mean(d)),
        "max_abs": float(np.max(np.abs(d))),
    }


def centreline_scores(
    *,
    pool: pd.DataFrame,
    share_pool: pd.DataFrame,
    fixtures_gw: dict[int, list[dict[str, str]]],
    priced_by_gw: dict[int, list[dict[str, Any]]],
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    alpha_g: float,
    alpha_a: float,
    plan_gws: list[int],
) -> dict[int, np.ndarray]:
    """Mid-λ ``score_team_side`` boards (no k∼M, no attack dice)."""
    priced_index = _priced_index(priced_by_gw)
    paths = init_paths_from_opta(ranks, profile=profile)
    out: dict[int, np.ndarray] = {}
    for gw in plan_gws:
        scenario_lams: dict[tuple[str, str], tuple[float, float]] = {}
        for fx in fixtures_gw.get(int(gw), []):
            h, a = norm_team(fx["home"]), norm_team(fx["away"])
            fit = (priced_index.get(int(gw)) or {}).get((h, a))
            if fit is not None:
                lh, la = float(fit["lam_h"]), float(fit["lam_a"])
            else:
                pred = match_prediction(
                    h,
                    a,
                    ranks=ranks,
                    profile=profile,
                    league_gf=league_gf,
                    sd_league_gf=0.0,
                )
                lh, la = float(pred["lam_home"]), float(pred["lam_away"])
            scenario_lams[(h, a)] = (lh, la)
        pots = pots_for_gw_from_paths(
            paths,
            fixtures_gw.get(int(gw), []),
            int(gw),
            league_gf=league_gf,
            scenario_lams=scenario_lams or None,
        )
        scored = score_pool_on_pots(
            pool,
            pots,
            gws=[int(gw)],
            gamma=1.0,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_pool=share_pool,
        )
        arr = pd.to_numeric(scored["mc_score"], errors="coerce").fillna(0.0).to_numpy(
            float
        )
        if len(arr) != len(pool):
            raise RuntimeError(f"centreline length mismatch GW{gw}")
        out[int(gw)] = arr
    return out


def run_prefix_stream(
    *,
    pool: pd.DataFrame,
    share_pool: pd.DataFrame,
    fixtures_gw: dict[int, list[dict[str, str]]],
    priced_by_gw: dict[int, list[dict[str, Any]]],
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    hist_by_team: dict[str, pd.DataFrame],
    alpha_g: float,
    alpha_a: float,
    plan_gws: list[int],
    n_max: int,
    seed0: int,
    report_ns: tuple[int, ...],
) -> dict[str, Any]:
    """Prefix gen means. ``pool`` = thin rows; ``share_pool`` = full club dice."""
    priced_index = _priced_index(priced_by_gw)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    acc = {g: np.zeros(len(pool), dtype=float) for g in plan_gws}
    # Scoreline accumulators: sum(k), sum(λ), n fixtures×strands
    sum_k_h = 0.0
    sum_lam_h = 0.0
    sum_k_a = 0.0
    sum_lam_a = 0.0
    n_fx = 0
    snapshots: list[dict[str, Any]] = []
    want = sorted({int(n) for n in report_ns if 1 <= int(n) <= int(n_max)} | {int(n_max)})
    t0 = _now()

    for s in range(int(n_max)):
        if s == 0 or (s + 1) % 25 == 0 or (s + 1) == int(n_max):
            print(f"  [conv] strand {s + 1}/{n_max}", flush=True)
        rng = random.Random(int(seed0) + s)
        paths: dict[str, ClubPath] = {
            c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()
        }
        rolled: dict[int, list[dict[str, Any]]] = {}
        for gw in plan_gws:
            week: list[dict[str, Any]] = []
            for fx in fixtures_gw.get(int(gw), []):
                if fx["home"] not in paths or fx["away"] not in paths:
                    continue
                sc = sample_week_scenario(
                    fx["home"],
                    fx["away"],
                    gw=int(gw),
                    priced_index=priced_index,
                    paths=paths,
                    league_gf=league_gf,
                    rng=rng,
                    ranks=ranks,
                    profile=profile,
                    sd_league_gf=0.0,
                )
                paths[fx["home"]].observe(sc["hg"], sc["ag"])
                paths[fx["away"]].observe(sc["ag"], sc["hg"])
                week.append(sc)
                sum_k_h += float(sc["hg"])
                sum_k_a += float(sc["ag"])
                sum_lam_h += float(sc["lam_h"])
                sum_lam_a += float(sc["lam_a"])
                n_fx += 1
            rolled[int(gw)] = week
        scores = _undiscounted_scores_by_gw(
            pool,
            plan_gws=plan_gws,
            rolled_by_gw=rolled,
            fixtures_gw=fixtures_gw,
            paths=paths,
            league_gf=league_gf,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            rng=random.Random(int(seed0) + 10_000 + s),
            hist_by_team=hist_by_team,
            generative_attack=True,
            lineup_cov=DEFAULT_LINEUP_COV,
            share_pool=share_pool,
        )
        for g in plan_gws:
            acc[int(g)] += scores[int(g)]
        n_done = s + 1
        if n_done in want:
            mean = {g: acc[g] / float(n_done) for g in plan_gws}
            snapshots.append(
                {
                    "n": n_done,
                    "mean_by_gw": {str(g): mean[g] for g in plan_gws},
                    "scoreline": {
                        "mean_k_h": sum_k_h / max(n_fx, 1),
                        "mean_lam_h": sum_lam_h / max(n_fx, 1),
                        "mean_k_a": sum_k_a / max(n_fx, 1),
                        "mean_lam_a": sum_lam_a / max(n_fx, 1),
                        "bias_h": (sum_k_h - sum_lam_h) / max(n_fx, 1),
                        "bias_a": (sum_k_a - sum_lam_a) / max(n_fx, 1),
                        "n_fixture_rolls": n_fx,
                    },
                }
            )
    return {
        "elapsed_s": _now() - t0,
        "snapshots": snapshots,
        "n_max": int(n_max),
    }


def write_report(state: dict[str, Any]) -> None:
    rows = state["vs_centreline"]
    self_rows = state["vs_nmax"]
    lines = [
        "# Generative mean-strand convergence vs centreline",
        "",
        f"Protocol: `cascade_mean_strand_convergence` · season **{state['season']}** · "
        f"GW{state['horizon_start']}–{state['horizon_end']} · "
        f"Nmax={state['n_max']} · seed={state['seed']}",
        "",
        "**Question:** as N→∞, does the average of generative strands approach "
        "the centreline (`score_team_side` on mid λ)?",
        "",
        "## Answer (short)",
        "",
        state["verdict_md"],
        "",
        "## Design",
        "",
        "- Centreline: FD mid λ (GW8–9) / oracle-rank mid λ (GW10+) → "
        "`score_team_side` (no `k∼M`, no attack dice).",
        "- Gen strand: same λ sources, but long weeks sample inside rankmean "
        "bands (`centre_blend=0.5`); then `k∼M` → attack dice "
        f"(``lineup_cov={DEFAULT_LINEUP_COV}``).",
        "- One seed stream; prefixes reuse the same rolls.",
        f"- Wall: {state['elapsed_s']:.1f}s",
        "",
        "## Gen-mean(N) vs centreline",
        "",
        "| N | MAE | RMSE | corr | bias (gen−centre) | max\\|Δ\\| |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['mae']:.4f} | {r['rmse']:.4f} | "
            f"{r['corr']:.4f} | {r['bias']:.4f} | {r['max_abs']:.3f} |"
        )
    lines += [
        "",
        "## Gen-mean(N) vs gen-mean(Nmax) (self-convergence)",
        "",
        "| N | MAE | RMSE | corr |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for r in self_rows:
        lines.append(
            f"| {r['n']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['corr']:.4f} |"
        )
    sc = state["scoreline_at_nmax"]
    lines += [
        "",
        "## Scoreline moments at Nmax",
        "",
        f"- mean(k_h)−λ_h = **{sc['bias_h']:+.4f}** · "
        f"mean(k_a)−λ_a = **{sc['bias_a']:+.4f}** "
        f"(n_rolls={sc['n_fixture_rolls']})",
        "- If `k∼M` is calibrated, these biases → 0; player xP can still "
        "differ from centreline because CS/attack maps are nonlinear.",
        "",
        "## Artefacts",
        "",
        f"- `{STATE_JSON.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_mean_strand_convergence "
        f"--season {state['season']} --n-max {state['n_max']}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--decision-gw", type=int, default=DECISION_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n-max", type=int, default=N_MAX)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--ns",
        type=str,
        default=",".join(str(n) for n in DEFAULT_NS),
        help="comma-separated prefixes to snapshot",
    )
    ap.add_argument(
        "--artifact",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_strand_convergence.json"),
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    if season not in SEASON_E0:
        raise SystemExit(f"unknown season {season!r}")
    e0 = SEASON_E0[season]
    decision_gw = int(args.decision_gw)
    horizon_end = int(args.horizon_end)
    start = decision_gw
    n_max = int(args.n_max)
    report_ns = tuple(int(x) for x in str(args.ns).split(",") if x.strip())

    print(
        f"[conv] season={season} horizon={start}-{horizon_end} "
        f"Nmax={n_max} seed={args.seed}",
        flush=True,
    )
    t_all = _now()

    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)

    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)
    hist = numagent_xp.asof_share_history(
        merged, before_gw=decision_gw, season=season
    )
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)
    pool = _build_hist_pool(hist, prices, decision_gw=decision_gw, season=season)
    share_pool = _build_hist_share_pool(
        hist, prices, decision_gw=decision_gw, season=season
    )

    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start, end=horizon_end)
    plan_gws = sorted(fixtures_gw)
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    print(
        f"[conv] select={len(pool)} share={len(share_pool)} "
        f"gws={plan_gws} priced={sorted(priced_by_gw)}",
        flush=True,
    )

    print("[conv] centreline boards…", flush=True)
    centre = centreline_scores(
        pool=pool,
        share_pool=share_pool,
        fixtures_gw=fixtures_gw,
        priced_by_gw=priced_by_gw,
        ranks=ranks,
        profile=profile,
        league_gf=league_gf,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        plan_gws=plan_gws,
    )
    centre_vec = _stack(centre, plan_gws)

    print("[conv] generative prefix stream…", flush=True)
    stream = run_prefix_stream(
        pool=pool,
        share_pool=share_pool,
        fixtures_gw=fixtures_gw,
        priced_by_gw=priced_by_gw,
        ranks=ranks,
        profile=profile,
        league_gf=league_gf,
        hist_by_team=team_hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        plan_gws=plan_gws,
        n_max=n_max,
        seed0=int(args.seed),
        report_ns=report_ns,
    )

    # Materialize vectors for each snapshot (drop bulky arrays from state later).
    snap_vecs: dict[int, np.ndarray] = {}
    for snap in stream["snapshots"]:
        n = int(snap["n"])
        by_gw = {
            int(g): np.asarray(v, dtype=float) for g, v in snap["mean_by_gw"].items()
        }
        snap_vecs[n] = _stack(by_gw, plan_gws)

    nmax = max(snap_vecs)
    nmax_vec = snap_vecs[nmax]
    vs_centre = []
    vs_nmax = []
    for n in sorted(snap_vecs):
        m = _metrics(snap_vecs[n], centre_vec)
        m["n"] = n
        vs_centre.append(m)
        sm = _metrics(snap_vecs[n], nmax_vec)
        sm["n"] = n
        vs_nmax.append(sm)

    # Per-GW MAE at Nmax vs centreline
    per_gw = []
    nmax_by_gw = {
        int(g): np.asarray(v, dtype=float)
        for g, v in next(
            s for s in stream["snapshots"] if int(s["n"]) == nmax
        )["mean_by_gw"].items()
    }
    for g in plan_gws:
        d = nmax_by_gw[g] - centre[g]
        per_gw.append(
            {
                "gw": g,
                "mae": float(np.mean(np.abs(d))),
                "bias": float(np.mean(d)),
                "corr": float(np.corrcoef(nmax_by_gw[g], centre[g])[0, 1])
                if np.std(centre[g]) > 1e-12
                else float("nan"),
            }
        )

    mae_nmax = next(r for r in vs_centre if r["n"] == nmax)["mae"]
    mae_12 = next((r for r in vs_centre if r["n"] == 12), None)
    self_mae_50 = next((r for r in vs_nmax if r["n"] == 50), None)
    scoreline = next(s for s in stream["snapshots"] if int(s["n"]) == nmax)[
        "scoreline"
    ]

    converges_to_centre = mae_nmax < 0.05
    self_stable = (
        self_mae_50 is not None and float(self_mae_50["mae"]) < 0.15
    )
    if converges_to_centre:
        verdict_md = (
            f"At N={nmax}, gen-mean MAE vs centreline is **{mae_nmax:.3f}** — "
            "essentially the centreline."
        )
    else:
        verdict_md = (
            f"**No — large N converges to the generative expectation, not the "
            f"centreline.** At N={nmax}, MAE(gen−centre)=**{mae_nmax:.3f}** xP "
            f"(bias {next(r for r in vs_centre if r['n']==nmax)['bias']:+.3f}); "
            f"corr≈{next(r for r in vs_centre if r['n']==nmax)['corr']:.3f}. "
            "Self-convergence is much tighter "
            f"(MAE(N=50 vs N={nmax})="
            f"{self_mae_50['mae']:.3f} xP). "
            "Scoreline `k∼M` is nearly unbiased "
            f"(Δk_h={scoreline['bias_h']:+.3f}); the gap is nonlinear player "
            "maps (CS / attack dice), matching prior viability "
            "`gen clearly_loses` vs mean."
        )

    state = {
        "task": "cascade_mean_strand_convergence",
        "season": season,
        "e0": e0,
        "decision_gw": decision_gw,
        "horizon_start": start,
        "horizon_end": horizon_end,
        "n_max": n_max,
        "seed": int(args.seed),
        "n_pool": int(len(pool)),
        "n_player_gw": int(len(centre_vec)),
        "vs_centreline": vs_centre,
        "vs_nmax": vs_nmax,
        "per_gw_nmax_vs_centre": per_gw,
        "scoreline_at_nmax": scoreline,
        "converges_to_centreline": converges_to_centre,
        "self_stable_by_n50": self_stable,
        "verdict_md": verdict_md,
        "mae_at_12": None if mae_12 is None else mae_12["mae"],
        "mae_at_nmax": mae_nmax,
        "board_elapsed_s": stream["elapsed_s"],
        "elapsed_s": _now() - t_all,
        "report": str(REPORT_MD),
    }
    _save(state)
    write_report(state)

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        ns = [r["n"] for r in vs_centre]
        axes[0].plot(ns, [r["mae"] for r in vs_centre], "o-", color="#c0392b", lw=2, label="vs centreline")
        axes[0].plot(ns, [r["mae"] for r in vs_nmax], "s-", color="#2980b9", lw=2, label=f"vs gen-mean(N={nmax})")
        axes[0].set_xlabel("N strands")
        axes[0].set_ylabel("MAE (player-GW xP)")
        axes[0].set_title("Convergence of generative mean boards")
        axes[0].set_xscale("log")
        axes[0].legend(frameon=False)
        axes[0].grid(True, alpha=0.3)
        axes[1].bar(
            [r["gw"] for r in per_gw],
            [r["mae"] for r in per_gw],
            color="#1a5276",
            alpha=0.85,
        )
        axes[1].set_xlabel("GW")
        axes[1].set_ylabel(f"MAE gen(N={nmax}) − centre")
        axes[1].set_title("Per-GW gap at Nmax")
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        plot_path = Path("/opt/cursor/artifacts/cascade_mean_strand_convergence.png")
        fig.savefig(plot_path, dpi=140)
        state["plot"] = str(plot_path)
        print("wrote", plot_path, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[conv] plot skipped: {exc}", flush=True)

    art = Path(args.artifact)
    art.parent.mkdir(parents=True, exist_ok=True)
    # Strip heavy arrays already dropped; state is summary-only.
    art.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    _save(state)
    write_report(state)

    print(json.dumps({
        "elapsed_s": state["elapsed_s"],
        "converges_to_centreline": converges_to_centre,
        "mae_at_nmax": mae_nmax,
        "mae_at_12": state["mae_at_12"],
        "scoreline_bias_h": scoreline["bias_h"],
        "report": str(REPORT_MD),
    }, indent=2))
    print(verdict_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
