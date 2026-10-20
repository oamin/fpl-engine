"""Trial: early-season rate M0 vs raw rates on ``score_xp_v2`` (closed seasons).

Pre-registered: ``protocol.score_xp_v2_rate_m0``.
Treatment: ``r=G/(M+M0)``, ``Δr=max(√G,c)/(M+M0)`` with M0=270, c=0.25.
Baseline: M0=0, c=0. Both arms keep share-SE shrink on (current default).

  python -m src.markets.rate_m0_trial
"""

from __future__ import annotations

import json
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
from src.markets.numagent_xp import SCORE_NAME
from src.markets.numagent_xp_pilot import (
    GW_EVAL_START,
    SIGMA,
    _asof_alphas,
    load_fd_moments,
    load_merged,
    load_score_xp,
    score_gameweek,
)
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TREATMENT = f"{SCORE_NAME}_rate_m0"
BASELINE = SCORE_NAME
M0 = 270.0
RATE_SE_FLOOR = 0.25


def score_gameweek_pair(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    """Same week scored with raw rates and M0 rates (share-SE on both)."""
    raw = score_gameweek(
        hist, week, fd, alpha_g=alpha_g, alpha_a=alpha_a, m0=0.0, rate_se_floor=0.0
    )
    if raw.empty:
        return raw
    m0 = score_gameweek(
        hist,
        week,
        fd,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        m0=M0,
        rate_se_floor=RATE_SE_FLOOR,
    )
    if m0.empty:
        return pd.DataFrame()

    keep = [
        "element",
        "gw",
        "fixture",
        "team_n",
        "date_iso",
        "home",
        "away",
        "xp",
        "xp_se",
        "sG",
        "sA",
        "rG",
        "rA",
        "total_points",
        "minutes",
        "position",
    ]
    raw_k = raw[[c for c in keep if c in raw.columns]].copy()
    m0_k = m0[[c for c in keep if c in m0.columns]].copy()
    raw_k = raw_k.rename(
        columns={"xp": "xp_raw", "xp_se": "xp_se_raw", "sG": "sG_raw", "sA": "sA_raw", "rG": "rG_raw", "rA": "rA_raw"}
    )
    m0_k = m0_k.rename(
        columns={"xp": "xp_m0", "xp_se": "xp_se_m0", "sG": "sG_m0", "sA": "sA_m0", "rG": "rG_m0", "rA": "rA_m0"}
    )
    on = [c for c in ("element", "gw", "fixture", "team_n") if c in raw_k.columns and c in m0_k.columns]
    return raw_k.merge(m0_k, on=on, how="inner", suffixes=("", "_y"))


def run_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = season.replace("-", "_")
    merged = load_merged(stem)
    fd = load_fd_moments(stem)
    _ = load_score_xp(stem)
    parts: list[pd.DataFrame] = []
    gws = sorted(int(g) for g in merged["GW"].unique() if g >= 2)
    for gw in gws:
        if gw < GW_EVAL_START:
            continue
        hist = merged[merged["GW"] < gw]
        week = merged[merged["GW"] == gw]
        ag, aa = _asof_alphas(hist)
        pair = score_gameweek_pair(hist, week, fd, alpha_g=ag, alpha_a=aa)
        if not pair.empty:
            parts.append(pair)
    if not parts:
        raise RuntimeError(f"no rows for {season}")
    rows = pd.concat(parts, ignore_index=True)
    rows["element"] = rows["element"].astype(int)
    rows["season"] = season

    y = rows["total_points"].to_numpy(float)
    mu0 = rows["xp_raw"].to_numpy(float)
    mu1 = rows["xp_m0"].to_numpy(float)
    rows["log_raw"] = gaussian_log_score(y, mu0, SIGMA)
    rows["log_m0"] = gaussian_log_score(y, mu1, SIGMA)
    rows["d_log"] = rows["log_m0"] - rows["log_raw"]
    rows["abs_err_raw"] = np.abs(y - mu0)
    rows["abs_err_m0"] = np.abs(y - mu1)

    gw = (
        rows.groupby("gw", sort=True)
        .agg(
            n=("d_log", "size"),
            d_log=("d_log", "mean"),
            log_raw=("log_raw", "mean"),
            log_m0=("log_m0", "mean"),
            mae_raw=("abs_err_raw", "mean"),
            mae_m0=("abs_err_m0", "mean"),
        )
        .reset_index()
    )
    gw["season"] = season
    return rows, gw


def run() -> dict[str, Any]:
    protocol = load_protocol()
    screen = protocol.get("score_xp_v2_rate_m0") or {}
    if not str(screen.get("baseline") or "").startswith(BASELINE):
        raise RuntimeError("protocol baseline must start with score_xp_v2")

    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError("score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"]))

    row_parts: list[pd.DataFrame] = []
    gw_parts: list[pd.DataFrame] = []
    season_summaries: list[dict[str, Any]] = []
    for season in CLOSED_SEASONS:
        print(f"rate-M0 trial {season} …", flush=True)
        rows, gw = run_season(season)
        row_parts.append(rows)
        gw_parts.append(gw)
        season_summaries.append(
            {
                "season": season,
                "n_rows": int(len(rows)),
                "n_gws": int(len(gw)),
                "mean_d_log": float(gw["d_log"].mean()),
                "mae_raw": float(rows["abs_err_raw"].mean()),
                "mae_m0": float(rows["abs_err_m0"].mean()),
            }
        )
        print(
            f"  n={len(rows)} d_log={season_summaries[-1]['mean_d_log']:+.4f}",
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

    key = comparison_key(TREATMENT, BASELINE)
    comparisons = {
        **_certified_protocol_comparison(),
        key: {
            **interval,
            "incomplete_seasons": {},
            "n_rows": int(len(rows_all)),
        },
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "score_xp_v2_rate_m0",
    }

    lo = float(interval.get("lo", -1))
    hi = float(interval.get("hi", 1))
    covers_zero = bool(lo <= 0 <= hi)
    if lo > 0:
        verdict = "promote"
    elif hi < 0:
        verdict = "park_clear_lose"
    else:
        verdict = "inconclusive"

    summary = {
        "treatment": TREATMENT,
        "baseline": BASELINE,
        "m0": M0,
        "rate_se_floor": RATE_SE_FLOOR,
        "formula": screen.get("formula"),
        "seasons": season_summaries,
        "cluster_interval": interval,
        "promote_if": screen.get("promote_if"),
        "covers_zero": covers_zero,
        "verdict": verdict,
    }

    rows_all.to_csv(PROCESSED / "score_xp_v2_rate_m0.csv", index=False)
    gw_all.to_csv(PROCESSED / "score_xp_v2_rate_m0_gw.csv", index=False)
    (PROCESSED / "score_xp_v2_rate_m0_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (PROCESSED / "score_xp_v2_rate_m0_intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )

    lines = [
        "# `score_xp_v2` rate M0 vs raw rates",
        "",
        "Pre-registered: `protocol.score_xp_v2_rate_m0`.",
        "",
        "Treatment applies early-season minutes pseudo-count and zero-event SE floor "
        "on `rG`/`rA` before weights/shares. Share-SE shrink stays on for both arms.",
        "",
        f"Formula: `{screen.get('formula')}`",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed** (rates/priors GW `< t`; minutes oracle).",
        "",
        "## Result (treatment − baseline)",
        "",
        f"- Mean Δ log-score / GW: **{interval['mean']:.4f}** "
        f"[{interval['lo']:.4f}, {interval['hi']:.4f}]",
        f"- Covers 0: **{covers_zero}**",
        f"- Verdict: **{verdict}**",
        f"- Pooled player-fixture rows: {len(rows_all)}",
        "",
        "| Season | n | mean Δ log | MAE raw | MAE M0 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in season_summaries:
        lines.append(
            f"| {s['season']} | {s['n_rows']} | {s['mean_d_log']:+.4f} | "
            f"{s['mae_raw']:.3f} | {s['mae_m0']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Pool matches `score_xp_v2_screen` (minutes>60, FD AH+OU, GW5–38, σ=3).",
        "- Positive Δ = M0 rates beat current `score_xp_v2` on Gaussian log score.",
        "- Promoted 2026-10-16: default `RATE_M0=270`, `RATE_SE_FLOOR=0.25`.",
        "",
    ]
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}
    write_gated_report(
        REPORTS / "score_xp_v2_rate_m0.md",
        audit_out,
        intervals,
        lines,
    )
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2))
    print(f"wrote {REPORTS / 'score_xp_v2_rate_m0.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
