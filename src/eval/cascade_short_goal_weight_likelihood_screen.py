"""Gated screen: cascade_short generative π∝r vs rate_minutes (+ L3 reference).

Pre-registered: ``protocol.cascade_short_goal_weight_likelihood_screen``.

Baseline: cascade_short generative MC-mean with ``goal_weight=rate_minutes``.
Treatment: same stack with ``goal_weight=rate`` (π∝r; minutes gate only).
Reference: FD centreline ``score_xp_v2`` (L3/L4 mean layer) on the same fits.

Pool: minutes>60, ≥3 prior apps, FD AH+OU priced weeks, GW8–38, σ=3.
Metric: player-GW Gaussian log score; paired treatment−baseline by GW;
cluster bootstrap across the four closed seasons.

Does **not** flip the default. Centreline ``score_xp_v2_goal_share_rate``
already parked π∝r — this asks whether cascade_short generative changes that.

  PYTHONPATH=. python3 -m src.eval.cascade_short_goal_weight_likelihood_screen
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

from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets import numagent_xp
from src.markets.cascade_score_v2 import SCORER
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison
from src.markets.short_horizon_generative import hist_by_team_norm, score_week_generative
from src.models.short_horizon_cascade import (
    DEFAULT_LINEUP_COV,
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
    score_week_on_fixture_moments,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "cascade_short_goal_weight_likelihood_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/cascade_short_goal_weight_likelihood_screen.json")
PROCESSED = ROOT / "data" / "processed"

TREATMENT = "cascade_short_gen_goal_share_rate"
BASELINE = "cascade_short_gen_goal_share_rate_minutes"
L3_REF = "l3_score_xp_v2"

GW_START = 8
GW_END = 38
MIN_MINUTES = 60.0
MIN_HISTORY = 3
N_MC = 6
SIGMA = 3.0
SEED = 7


def _now() -> float:
    return time.time()


def _gen_mean_board(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
    hist_by_team: dict[str, pd.DataFrame],
    rng: random.Random,
    n_mc: int,
    goal_weight: str,
) -> pd.DataFrame:
    if side.empty or not fits:
        return pd.DataFrame()
    acc: dict[int, float] = {}
    meta: dict[int, dict[str, Any]] = {}
    for _ in range(int(n_mc)):
        board = score_week_generative(
            side,
            fits,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            rng=rng,
            hist_by_team=hist_by_team,
            lineup_cov=DEFAULT_LINEUP_COV,
            lam_overrides=None,
            goal_weight=goal_weight,
        )
        if board.empty:
            continue
        for row in board.itertuples(index=False):
            eid = int(row.element)
            acc[eid] = acc.get(eid, 0.0) + float(row.score_xp_v2)
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


def _eligible_join(
    week: pd.DataFrame,
    hist: pd.DataFrame,
    boards: dict[str, pd.DataFrame],
    *,
    season: str,
    gw: int,
) -> pd.DataFrame:
    pts = week.groupby("element", as_index=False).agg(
        total_points=("total_points", "sum"),
        minutes=("minutes", "sum"),
    )
    apps = (
        hist.loc[hist["minutes"] > 0].groupby("element")["GW"].nunique()
        if len(hist)
        else pd.Series(dtype=float)
    )
    j = pts.copy()
    for name, board in boards.items():
        if board.empty:
            return pd.DataFrame()
        col = "xp" if "xp" in board.columns else SCORER
        piece = board[["element", col]].rename(columns={col: name})
        j = j.merge(piece, on="element", how="inner")
    j["n_prior"] = j["element"].map(apps).fillna(0)
    j = j.loc[
        (pd.to_numeric(j["minutes"], errors="coerce") > MIN_MINUTES)
        & (j["n_prior"] >= MIN_HISTORY)
    ].copy()
    if j.empty:
        return j
    j["season"] = season
    j["gw"] = int(gw)
    return j


def run_season(
    season: str, *, n_mc: int, seed0: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd_idx = _fd_index(season)
    rng = random.Random(int(seed0))
    parts: list[pd.DataFrame] = []
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
        fx = fixtures_for_gw(enriched, int(gw))
        fits = centreline_fixture_fits(fx, fd_idx)
        if len(fits) < 5:
            continue
        team_hist = hist_by_team_norm(hist)
        mean_l3 = score_week_on_fixture_moments(
            side, fits, alpha_g=alpha_g, alpha_a=alpha_a
        )
        if mean_l3.empty:
            continue
        if "xp" not in mean_l3.columns and SCORER in mean_l3.columns:
            mean_l3 = mean_l3.rename(columns={SCORER: "xp"})
        gen_rm = _gen_mean_board(
            side,
            fits,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            hist_by_team=team_hist,
            rng=rng,
            n_mc=n_mc,
            goal_weight="rate_minutes",
        )
        gen_rate = _gen_mean_board(
            side,
            fits,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            hist_by_team=team_hist,
            rng=rng,
            n_mc=n_mc,
            goal_weight="rate",
        )
        joined = _eligible_join(
            week,
            hist,
            {
                "xp_rm": gen_rm,
                "xp_rate": gen_rate,
                "xp_l3": mean_l3,
            },
            season=season,
            gw=gw,
        )
        if not joined.empty:
            parts.append(joined)
        if gw % 5 == 0:
            print(f"  [{season}] GW{gw} rows={len(joined)}", flush=True)
    if not parts:
        raise RuntimeError(f"no eligible rows for {season}")
    rows = pd.concat(parts, ignore_index=True)
    y = rows["total_points"].to_numpy(float)
    rows["log_rm"] = gaussian_log_score(y, rows["xp_rm"].to_numpy(float), SIGMA)
    rows["log_rate"] = gaussian_log_score(y, rows["xp_rate"].to_numpy(float), SIGMA)
    rows["log_l3"] = gaussian_log_score(y, rows["xp_l3"].to_numpy(float), SIGMA)
    rows["d_log"] = rows["log_rate"] - rows["log_rm"]
    rows["d_log_rm_minus_l3"] = rows["log_rm"] - rows["log_l3"]
    rows["d_log_rate_minus_l3"] = rows["log_rate"] - rows["log_l3"]
    rows["abs_err_rm"] = (y - rows["xp_rm"].to_numpy(float)).abs()
    rows["abs_err_rate"] = (y - rows["xp_rate"].to_numpy(float)).abs()
    rows["abs_err_l3"] = (y - rows["xp_l3"].to_numpy(float)).abs()
    gw = (
        rows.groupby("gw", sort=True)
        .agg(
            n=("d_log", "size"),
            d_log=("d_log", "mean"),
            d_log_rm_minus_l3=("d_log_rm_minus_l3", "mean"),
            d_log_rate_minus_l3=("d_log_rate_minus_l3", "mean"),
            log_rm=("log_rm", "mean"),
            log_rate=("log_rate", "mean"),
            log_l3=("log_l3", "mean"),
            mae_rm=("abs_err_rm", "mean"),
            mae_rate=("abs_err_rate", "mean"),
            mae_l3=("abs_err_l3", "mean"),
        )
        .reset_index()
    )
    gw["season"] = season
    return rows, gw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-mc", type=int, default=N_MC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    n_mc = int(args.n_mc)
    seed0 = int(args.seed)
    t0 = _now()

    protocol = load_protocol()
    screen = protocol.get("cascade_short_goal_weight_likelihood_screen") or {}
    if screen.get("treatment") != TREATMENT:
        raise RuntimeError(
            f"protocol treatment must be {TREATMENT!r}, got {screen.get('treatment')!r}"
        )
    if screen.get("baseline") != BASELINE:
        raise RuntimeError(
            f"protocol baseline must be {BASELINE!r}, got {screen.get('baseline')!r}"
        )

    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError("score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"]))

    print(
        f"[casc-short-gw] seasons={CLOSED_SEASONS} GW{GW_START}–{GW_END} "
        f"N_MC={n_mc} pool=m>{MIN_MINUTES}, hist≥{MIN_HISTORY}",
        flush=True,
    )

    row_parts: list[pd.DataFrame] = []
    gw_parts: list[pd.DataFrame] = []
    season_summaries: list[dict[str, Any]] = []
    for i, season in enumerate(CLOSED_SEASONS):
        print(f"[casc-short-gw] {season} …", flush=True)
        rows, gw = run_season(season, n_mc=n_mc, seed0=seed0 + 1000 * i)
        row_parts.append(rows)
        gw_parts.append(gw)
        season_summaries.append(
            {
                "season": season,
                "n_rows": int(len(rows)),
                "n_gws": int(len(gw)),
                "mean_d_log": float(gw["d_log"].mean()),
                "mean_d_log_rm_minus_l3": float(gw["d_log_rm_minus_l3"].mean()),
                "mean_d_log_rate_minus_l3": float(gw["d_log_rate_minus_l3"].mean()),
                "mae_rm": float(rows["abs_err_rm"].mean()),
                "mae_rate": float(rows["abs_err_rate"].mean()),
                "mae_l3": float(rows["abs_err_l3"].mean()),
            }
        )
        print(
            f"  n={len(rows)} d_log={season_summaries[-1]['mean_d_log']:+.4f} "
            f"rm−l3={season_summaries[-1]['mean_d_log_rm_minus_l3']:+.4f}",
            flush=True,
        )

    rows_all = pd.concat(row_parts, ignore_index=True)
    gw_all = pd.concat(gw_parts, ignore_index=True)
    deltas = {
        season: gw_all.loc[gw_all["season"] == season, "d_log"].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    interval = cluster_interval(deltas, seasons=CLOSED_SEASONS)
    for season, arr in deltas.items():
        if arr.size < 20:
            raise RuntimeError(f"{season} has fewer than 20 gameweeks")

    # Reference: gen vs L3 (not the promote pair).
    deltas_rm_l3 = {
        season: gw_all.loc[
            gw_all["season"] == season, "d_log_rm_minus_l3"
        ].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    deltas_rate_l3 = {
        season: gw_all.loc[
            gw_all["season"] == season, "d_log_rate_minus_l3"
        ].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    iv_rm_l3 = cluster_interval(deltas_rm_l3, seasons=CLOSED_SEASONS)
    iv_rate_l3 = cluster_interval(deltas_rate_l3, seasons=CLOSED_SEASONS)

    key = comparison_key(TREATMENT, BASELINE)
    comparisons = {
        **_certified_protocol_comparison(),
        key: {
            **interval,
            "incomplete_seasons": {},
            "n_rows": int(len(rows_all)),
        },
        comparison_key(BASELINE, L3_REF): {
            **iv_rm_l3,
            "incomplete_seasons": {},
            "n_rows": int(len(rows_all)),
        },
        comparison_key(TREATMENT, L3_REF): {
            **iv_rate_l3,
            "incomplete_seasons": {},
            "n_rows": int(len(rows_all)),
        },
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "cascade_short_goal_weight_likelihood_screen",
    }

    lo = float(interval["lo"])
    hi = float(interval["hi"])
    if lo > 0:
        verdict = "promote"
    elif hi < 0:
        verdict = "park_clear_lose"
    else:
        verdict = "inconclusive"

    summary = {
        "task": "cascade_short_goal_weight_likelihood_screen",
        "protocol": "cascade_short_goal_weight_likelihood_screen",
        "pre_registered": "2026-10-28",
        "treatment": TREATMENT,
        "baseline": BASELINE,
        "l3_reference": L3_REF,
        "n_mc": n_mc,
        "gw_range": [GW_START, GW_END],
        "pool": {"min_minutes": MIN_MINUTES, "min_history": MIN_HISTORY},
        "seasons": season_summaries,
        "cluster_interval": interval,
        "cluster_interval_rm_minus_l3": iv_rm_l3,
        "cluster_interval_rate_minus_l3": iv_rate_l3,
        "promote_if": screen.get("promote_if"),
        "covers_zero": bool(lo <= 0 <= hi),
        "verdict": verdict,
        "default_unchanged": True,
        "elapsed_s": _now() - t0,
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows_all.to_csv(
        PROCESSED / "cascade_short_goal_weight_likelihood_rows.csv", index=False
    )
    gw_all.to_csv(
        PROCESSED / "cascade_short_goal_weight_likelihood_gw.csv", index=False
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (PROCESSED / "cascade_short_goal_weight_likelihood_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n"
    )

    lines = [
        "# Cascade_short generative goal shares π∝r vs π∝r·m (+ L3)",
        "",
        "Pre-registered: `protocol.cascade_short_goal_weight_likelihood_screen`.",
        "",
        "Treatment: cascade_short generative MC-mean with ``goal_weight=rate`` "
        "(π∝r). Baseline: ``goal_weight=rate_minutes``. Reference: FD "
        "centreline ``score_xp_v2`` (L3/L4 mean layer) on the same fits.",
        "",
        f"N_MC={n_mc} · GW{GW_START}–{GW_END} · wall {summary['elapsed_s']:.1f}s",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Result (treatment − baseline = π∝r − rate_minutes)",
        "",
        f"- Mean Δ log-score / GW: **{interval['mean']:+.4f}** "
        f"[{interval['lo']:+.4f}, {interval['hi']:+.4f}]",
        f"- Covers 0: **{summary['covers_zero']}**",
        f"- Verdict: **{summary['verdict']}**",
        f"- Pooled player-GW rows: {len(rows_all)}",
        "- Default `goal_weight=rate_minutes` **not** flipped here.",
        "",
        "## vs L3 reference (diagnostic)",
        "",
        f"- rate_minutes − L3: **{iv_rm_l3['mean']:+.4f}** "
        f"[{iv_rm_l3['lo']:+.4f}, {iv_rm_l3['hi']:+.4f}]",
        f"- π∝r − L3: **{iv_rate_l3['mean']:+.4f}** "
        f"[{iv_rate_l3['lo']:+.4f}, {iv_rate_l3['hi']:+.4f}]",
        "",
        "| Season | n | mean Δ log | MAE rm | MAE rate | MAE L3 | rm−L3 | rate−L3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in season_summaries:
        lines.append(
            f"| {s['season']} | {s['n_rows']} | {s['mean_d_log']:+.4f} | "
            f"{s['mae_rm']:.3f} | {s['mae_rate']:.3f} | {s['mae_l3']:.3f} | "
            f"{s['mean_d_log_rm_minus_l3']:+.4f} | "
            f"{s['mean_d_log_rate_minus_l3']:+.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Positive Δ = π∝r beats rate_minutes on Gaussian log score.",
        "- Same weight formula as parked `score_xp_v2_goal_share_rate`, but on "
        "**cascade_short generative** dice (not centreline `score_team_side`).",
        "- Eligible pool (not path XI): minutes>60, ≥3 prior apps, FD-priced weeks.",
        "- Dig B XI resid (+100 vs +14) is a different metric; this is the gate.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_short_goal_weight_likelihood_screen",
        "```",
        "",
    ]
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}
    write_gated_report(REPORT, audit_out, intervals, lines)

    STATE.write_text(
        json.dumps(
            {
                "task": "cascade_short_goal_weight_likelihood_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated cascade_short gen π∝r vs rate_minutes (+ L3 ref)",
                "verdict": {
                    "verdict": verdict,
                    "mean_d_log": interval["mean"],
                    "lo": lo,
                    "hi": hi,
                    "covers_zero": summary["covers_zero"],
                    "default_unchanged": True,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {args.out}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
