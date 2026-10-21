"""Trial: freeze minutes (lineup_cov=False) on short generative vs mean.

Protocol: ``cascade_lineup_off_tt1``.

Same product window as ``cascade_short_mean_tt1_vs_gen`` (decision GW t:t+1),
but generative paths use trivial/mean minutes — scoreline→attack→CS/GC only.

Primary: gen_off − mean. Diagnostic: gen_on − gen_off (does lineup roll hurt?).

  PYTHONPATH=. python3 -m src.eval.cascade_lineup_off_tt1 --n-mc 20
"""

from __future__ import annotations

import json
import random
import time
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
)
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import SCORER
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm, score_week_generative
from src.models.short_horizon_cascade import (
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
    score_week_on_fixture_moments,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "cascade_lineup_off"
REPORT_MD = ROOT / "reports" / "cascade_lineup_off_tt1.md"
STATE_JSON = OUT_DIR / "state.json"

GW_START = 8
GW_END = 27
MIN_MINUTES = 60.0
MIN_HISTORY = 3
N_MC = 20
SIGMA = 3.0
BOOT = 2000
SEED = 7
MIN_FITS = 5


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _gen_mean_board(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    hist_by_team: dict[str, pd.DataFrame],
    rng: random.Random,
    n_mc: int,
    lineup_cov: bool,
) -> pd.DataFrame:
    if side.empty or not fits:
        return pd.DataFrame()
    acc: dict[int, float] = defaultdict(float)
    meta: dict[int, dict[str, Any]] = {}
    for _ in range(int(n_mc)):
        board = score_week_generative(
            side,
            fits,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            rng=rng,
            hist_by_team=hist_by_team if lineup_cov else {},
            lineup_cov=lineup_cov,
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


def _ensure_xp(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board
    if "xp" not in board.columns and SCORER in board.columns:
        return board.rename(columns={SCORER: "xp"})
    return board


def _pair_week(
    mean_board: pd.DataFrame,
    gen_off: pd.DataFrame,
    gen_on: pd.DataFrame,
    week: pd.DataFrame,
    hist: pd.DataFrame,
    *,
    season: str,
    decision_gw: int,
    score_gw: int,
) -> list[dict[str, Any]]:
    if mean_board.empty or gen_off.empty or gen_on.empty or week.empty:
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
    m = mean_board[["element", "xp"]].rename(columns={"xp": "mean_xp"})
    go = gen_off[["element", "xp"]].rename(columns={"xp": "gen_off"})
    gn = gen_on[["element", "xp"]].rename(columns={"xp": "gen_on"})
    j = pts.merge(m, on="element").merge(go, on="element").merge(gn, on="element")
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
                "decision_gw": int(decision_gw),
                "score_gw": int(score_gw),
                "element": int(row.element),
                "total_points": float(row.total_points),
                "minutes": float(row.minutes),
                "mean_xp": float(row.mean_xp),
                "gen_off": float(row.gen_off),
                "gen_on": float(row.gen_on),
            }
        )
    return out


def _verdict(interval: dict[str, Any]) -> str:
    lo, hi = float(interval["lo"]), float(interval["hi"])
    if hi < 0:
        return "clearly_loses"
    if lo > 0:
        return "clearly_wins"
    return "does_not_clearly_lose"


def _decision_deltas(
    all_rows: list[dict[str, Any]], left: str, right: str
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    by: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        by[(r["season"], r["decision_gw"])].append(r)
    gw_deltas: list[dict[str, Any]] = []
    by_season: dict[str, list[float]] = {s: [] for s in CLOSED_SEASONS}
    key = comparison_key(left, right)
    for (season, dgw), block in sorted(by.items()):
        y = np.array([b["total_points"] for b in block], float)
        a = np.array([b[left] for b in block], float)
        b = np.array([b[right] for b in block], float)
        if len(y) < 10:
            continue
        la = gaussian_log_score(y, a, SIGMA)
        lb = gaussian_log_score(y, b, SIGMA)
        delta = float((la - lb).mean())
        gw_deltas.append(
            {
                "season": season,
                "decision_gw": int(dgw),
                "n_rows": int(len(y)),
                "delta": delta,
                "comparison": key,
            }
        )
        by_season[season].append(delta)
    arr = {s: np.asarray(by_season[s], float) for s in CLOSED_SEASONS}
    return gw_deltas, arr


def run(*, n_mc: int = N_MC, seed: int = SEED) -> dict[str, Any]:
    t0 = _now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "task": "cascade_lineup_off_tt1",
        "started_unix": t0,
        "n_mc": int(n_mc),
        "sigma": SIGMA,
        "seed": int(seed),
        "gw_start": GW_START,
        "gw_end": GW_END,
        "lineup_cov_treatment": False,
    }
    audit = run_asof_audit()
    state["asof_audit"] = {"passed": bool(audit.get("passed")), "failures": audit.get("failures")}
    _save(state)
    if not audit.get("passed"):
        raise RuntimeError(f"as-of audit failed: {audit.get('failures')}")

    # Separate RNGs so on/off draws are comparable but not coupled mid-path.
    rng_off = random.Random(seed)
    rng_on = random.Random(seed + 10_007)
    all_rows: list[dict[str, Any]] = []
    per_season: dict[str, Any] = {}

    for season in CLOSED_SEASONS:
        st0 = _now()
        print(f"[lineup_off] season {season}…", flush=True)
        merged = load_merged(season)
        enriched = _load_merged_enriched(season)
        fd_idx = _fd_index(season)
        n_dec = 0
        for decision_gw in range(GW_START, GW_END + 1):
            hist = numagent_xp.asof_share_history(
                merged, before_gw=int(decision_gw), season=season
            )
            if hist.empty:
                continue
            alpha_g, alpha_a = _asof_alphas(hist)
            side, alpha_g, alpha_a = build_frozen_side_table(
                hist, alpha_g=alpha_g, alpha_a=alpha_a
            )
            if side.empty:
                continue
            team_hist = hist_by_team_norm(hist)
            fits_t = centreline_fixture_fits(
                fixtures_for_gw(enriched, int(decision_gw)), fd_idx
            )
            fits_tp1 = centreline_fixture_fits(
                fixtures_for_gw(enriched, int(decision_gw) + 1), fd_idx
            )
            if len(fits_t) < MIN_FITS or len(fits_tp1) < MIN_FITS:
                continue

            for score_gw, fits in (
                (int(decision_gw), fits_t),
                (int(decision_gw) + 1, fits_tp1),
            ):
                week = merged.loc[merged["GW"] == int(score_gw)].copy()
                if week.empty:
                    continue
                mean_b = _ensure_xp(
                    score_week_on_fixture_moments(
                        side, fits, alpha_g=alpha_g, alpha_a=alpha_a
                    )
                )
                gen_off = _gen_mean_board(
                    side,
                    fits,
                    alpha_g=alpha_g,
                    alpha_a=alpha_a,
                    hist_by_team=team_hist,
                    rng=rng_off,
                    n_mc=n_mc,
                    lineup_cov=False,
                )
                gen_on = _gen_mean_board(
                    side,
                    fits,
                    alpha_g=alpha_g,
                    alpha_a=alpha_a,
                    hist_by_team=team_hist,
                    rng=rng_on,
                    n_mc=n_mc,
                    lineup_cov=True,
                )
                all_rows.extend(
                    _pair_week(
                        mean_b,
                        gen_off,
                        gen_on,
                        week,
                        hist,
                        season=season,
                        decision_gw=int(decision_gw),
                        score_gw=int(score_gw),
                    )
                )
            n_dec += 1
            if n_dec % 5 == 0:
                print(f"  decision GW{decision_gw} ({n_dec} done)", flush=True)

        per_season[season] = {
            "elapsed_s": _now() - st0,
            "n_decision_gws": n_dec,
            "n_rows": sum(1 for r in all_rows if r["season"] == season),
        }
        state["progress"] = per_season
        _save(state)
        print(
            f"  {season} done in {per_season[season]['elapsed_s']:.1f}s "
            f"decisions={n_dec}",
            flush=True,
        )

    pd.DataFrame(all_rows).to_csv(OUT_DIR / "player_gw_rows.csv", index=False)

    d_off_mean, arr_off = _decision_deltas(all_rows, "gen_off", "mean_xp")
    d_on_off, arr_on = _decision_deltas(all_rows, "gen_on", "gen_off")
    d_on_mean, arr_on_mean = _decision_deltas(all_rows, "gen_on", "mean_xp")
    pd.DataFrame(d_off_mean).to_csv(OUT_DIR / "deltas_gen_off_minus_mean.csv", index=False)
    pd.DataFrame(d_on_off).to_csv(OUT_DIR / "deltas_gen_on_minus_gen_off.csv", index=False)

    iv_off = cluster_interval(arr_off, seasons=CLOSED_SEASONS, n_boot=BOOT, seed=seed)
    iv_on_off = cluster_interval(arr_on, seasons=CLOSED_SEASONS, n_boot=BOOT, seed=seed)
    iv_on_mean = cluster_interval(arr_on_mean, seasons=CLOSED_SEASONS, n_boot=BOOT, seed=seed + 1)

    v_off = _verdict(iv_off)
    v_on_off = _verdict(iv_on_off)
    v_on_mean = _verdict(iv_on_mean)

    state["elapsed_s"] = _now() - t0
    state["n_decision_pairs"] = len(d_off_mean)
    state["primary"] = {
        "comparison": comparison_key("gen_off", "mean_xp"),
        "interval": iv_off,
        "verdict": v_off,
    }
    state["diagnostic_on_minus_off"] = {
        "comparison": comparison_key("gen_on", "gen_off"),
        "interval": iv_on_off,
        "verdict": v_on_off,
    }
    state["diagnostic_on_minus_mean"] = {
        "comparison": comparison_key("gen_on", "mean_xp"),
        "interval": iv_on_mean,
        "verdict": v_on_mean,
        "note": "Replay of tt1-shaped gen-with-lineup vs mean (this stack)",
    }
    state["finished_unix"] = _now()
    _save(state)

    lines = [
        "# Lineup-off trial — freeze minutes on short generative",
        "",
        f"Protocol: `cascade_lineup_off_tt1` · N_MC={n_mc} · σ={SIGMA} · "
        f"decision GW{GW_START}–{GW_END} · seed={seed}",
        "",
        "Treatment: generative MC-mean with **`lineup_cov=False`** (as-of mean "
        "minutes; scoreline → attack → CS/GC only).  ",
        "Baseline: `score_team_side` on the same FD centreline fits.  ",
        "Diagnostic: same gen with `lineup_cov=True` (joint historical pairs).",
        "",
        f"- As-of audit: **{bool(audit.get('passed'))}**",
        f"- Decision-GW pairs: {len(d_off_mean)}",
        f"- Wall: {state['elapsed_s']:.1f}s",
        "",
        "## Primary — gen_off − mean",
        "",
        f"- Δ: **{iv_off['mean']:+.4f}** [{iv_off['lo']:+.4f}, {iv_off['hi']:+.4f}]",
        f"- Verdict: **{v_off}**",
        "",
        "## Diagnostic — gen_on − gen_off (does lineup roll hurt?)",
        "",
        f"- Δ: **{iv_on_off['mean']:+.4f}** [{iv_on_off['lo']:+.4f}, {iv_on_off['hi']:+.4f}]",
        f"- Verdict: **{v_on_off}**",
        "",
        "## Diagnostic — gen_on − mean (current lineup stack vs mean)",
        "",
        f"- Δ: **{iv_on_mean['mean']:+.4f}** [{iv_on_mean['lo']:+.4f}, {iv_on_mean['hi']:+.4f}]",
        f"- Verdict: **{v_on_mean}**",
        "",
        "Artifacts: `data/processed/cascade_lineup_off/`",
        "",
        "Reproduce:",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_lineup_off_tt1 --n-mc {n_mc}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    state["report"] = str(REPORT_MD)
    _save(state)
    print(
        f"[lineup_off] primary={v_off} Δ={iv_off['mean']:+.4f} | "
        f"on−off={v_on_off} Δ={iv_on_off['mean']:+.4f} ({state['elapsed_s']:.1f}s)",
        flush=True,
    )
    return state


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-mc", type=int, default=N_MC)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    run(n_mc=int(args.n_mc), seed=int(args.seed))


if __name__ == "__main__":
    main()
