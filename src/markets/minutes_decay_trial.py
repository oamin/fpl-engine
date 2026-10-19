"""Trial: geometric minutes decay vs flat last-3 mean on ``score_xp_v2``.

Pre-registered: ``protocol.score_xp_v2_minutes_decay``.
Both arms use **predicted** minutes from GW ``< t`` (not oracle). Rates /
priors / share-SE / M0 rates stay at current defaults. FD pots for the target
week. Pool: actual minutes > 60 with FD AH+OU, GW5–38.

  python -m src.markets.minutes_decay_trial
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
from src.live.news_packets import rolling_game_minutes, rolling_game_minutes_sigma
from src.markets import numagent_xp
from src.markets.numagent_xp import SCORE_NAME
from src.markets.numagent_xp_pilot import (
    GW_EVAL_START,
    MIN_MINUTES,
    SIGMA,
    _asof_alphas,
    load_fd_moments,
    load_merged,
)
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TREATMENT = f"{SCORE_NAME}_min_decay"
BASELINE = f"{SCORE_NAME}_min_flat"
RHO = 0.7
WINDOW = 3


def _minutes_series_by_element(hist: pd.DataFrame) -> dict[int, list[float]]:
    """Chronological minutes per element (zeros included) from as-of history."""
    if hist.empty:
        return {}
    h = hist.copy()
    h["element"] = pd.to_numeric(h["element"], errors="coerce")
    h["GW"] = pd.to_numeric(h["GW"], errors="coerce")
    h["minutes"] = pd.to_numeric(h["minutes"], errors="coerce").fillna(0.0)
    h = h.dropna(subset=["element", "GW"])
    out: dict[int, list[float]] = {}
    for eid, g in h.groupby("element", sort=False):
        g2 = g.sort_values("GW")
        # One row per GW (sum if DGW duplicate elements rare — take max minutes)
        by_gw = g2.groupby("GW", sort=True)["minutes"].max()
        out[int(eid)] = [float(v) for v in by_gw.tolist()]
    return out


def score_gameweek_predicted(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
    rho: float,
) -> pd.DataFrame:
    """Score one GW using predicted minutes (flat or decay) as ``m``."""
    if week.empty:
        return pd.DataFrame()
    numagent_xp.assert_single_season_rate_history(hist)
    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    series_map = _minutes_series_by_element(hist)

    w = week.merge(fd, on=["team_n", "date_iso"], how="left")
    w = w[w["xg"].notna()].copy()
    if w.empty:
        return pd.DataFrame()
    apps = w[w["minutes"] > 0].copy()
    if apps.empty:
        return pd.DataFrame()

    if rates.empty:
        for c in ("rG", "drG", "rA", "drA"):
            apps[c] = 0.0
    else:
        apps = apps.merge(rates, on="element", how="left")
        for c in ("rG", "drG", "rA", "drA"):
            apps[c] = apps[c].fillna(0.0)

    if priors.empty:
        for c in (
            "bonus_mu",
            "bonus_se",
            "card_mu",
            "card_se",
            "saves_mu",
            "saves_se",
            "defcon_mu",
            "defcon_sig",
            "defcon_n",
        ):
            apps[c] = 0.0
        apps["defcon_sig"] = numagent_xp.SIG_FLOOR
    else:
        apps = apps.merge(
            priors.drop(columns=["position"], errors="ignore"), on="element", how="left"
        )
        for c, d in (
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
            apps[c] = apps[c].fillna(d)
        apps["defcon_n"] = apps["defcon_n"].astype(int)

    pred_m: list[float] = []
    pred_s: list[float] = []
    for eid in apps["element"].astype(int):
        series = series_map.get(int(eid), [])
        m_hat = rolling_game_minutes(series, window=WINDOW, rho=rho)
        if m_hat is None:
            m_hat = 0.0
        pred_m.append(float(m_hat))
        pred_s.append(float(rolling_game_minutes_sigma(series, window=WINDOW)))
    apps["m"] = pred_m
    apps["sigma_m"] = pred_s
    apps["player_id"] = apps["element"]

    parts: list[pd.DataFrame] = []
    for (_, _), g in apps.groupby(["fixture", "team_n"], sort=False):
        side = numagent_xp.score_team_side(
            g,
            xg=float(g["xg"].iloc[0]),
            xcs=float(g["xcs"].iloc[0]),
            xgc=float(g["xgc"].iloc[0]),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
        )
        parts.append(side)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Eval pool: actual minutes (column still from week)
    out = out[out["minutes"] > MIN_MINUTES].copy()
    out["gw"] = int(week["GW"].iloc[0])
    out["minutes_rho"] = float(rho)
    return out


def score_gameweek_pair(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    flat = score_gameweek_predicted(
        hist, week, fd, alpha_g=alpha_g, alpha_a=alpha_a, rho=1.0
    )
    if flat.empty:
        return flat
    dec = score_gameweek_predicted(
        hist, week, fd, alpha_g=alpha_g, alpha_a=alpha_a, rho=RHO
    )
    if dec.empty:
        return pd.DataFrame()
    keep = [
        "element",
        "gw",
        "fixture",
        "team_n",
        "xp",
        "xp_se",
        "m",
        "total_points",
        "minutes",
        "position",
    ]
    f = flat[[c for c in keep if c in flat.columns]].rename(
        columns={"xp": "xp_flat", "xp_se": "xp_se_flat", "m": "m_flat"}
    )
    d = dec[[c for c in keep if c in dec.columns]].rename(
        columns={"xp": "xp_dec", "xp_se": "xp_se_dec", "m": "m_dec"}
    )
    on = [c for c in ("element", "gw", "fixture", "team_n") if c in f.columns and c in d.columns]
    return f.merge(d, on=on, how="inner", suffixes=("", "_y"))


def run_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = season.replace("-", "_")
    merged = load_merged(stem)
    fd = load_fd_moments(stem)
    parts: list[pd.DataFrame] = []
    for gw in sorted(int(g) for g in merged["GW"].unique() if g >= 2):
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
    mu0 = rows["xp_flat"].to_numpy(float)
    mu1 = rows["xp_dec"].to_numpy(float)
    rows["log_flat"] = gaussian_log_score(y, mu0, SIGMA)
    rows["log_dec"] = gaussian_log_score(y, mu1, SIGMA)
    rows["d_log"] = rows["log_dec"] - rows["log_flat"]
    rows["abs_err_flat"] = np.abs(y - mu0)
    rows["abs_err_dec"] = np.abs(y - mu1)
    gw = (
        rows.groupby("gw", sort=True)
        .agg(
            n=("d_log", "size"),
            d_log=("d_log", "mean"),
            mae_flat=("abs_err_flat", "mean"),
            mae_dec=("abs_err_dec", "mean"),
        )
        .reset_index()
    )
    gw["season"] = season
    return rows, gw


def run() -> dict[str, Any]:
    protocol = load_protocol()
    screen = protocol.get("score_xp_v2_minutes_decay") or {}
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
        print(f"minutes-decay trial {season} …", flush=True)
        rows, gw = run_season(season)
        row_parts.append(rows)
        gw_parts.append(gw)
        season_summaries.append(
            {
                "season": season,
                "n_rows": int(len(rows)),
                "n_gws": int(len(gw)),
                "mean_d_log": float(gw["d_log"].mean()),
                "mae_flat": float(rows["abs_err_flat"].mean()),
                "mae_dec": float(rows["abs_err_dec"].mean()),
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

    lo = float(interval.get("lo", -1))
    hi = float(interval.get("hi", 1))
    covers_zero = bool(lo <= 0 <= hi)
    if lo > 0:
        verdict = "promote"
    elif hi < 0:
        verdict = "park_clear_lose"
    else:
        verdict = "inconclusive"

    key = comparison_key(TREATMENT, BASELINE)
    comparisons = {
        **_certified_protocol_comparison(),
        key: {**interval, "incomplete_seasons": {}, "n_rows": int(len(rows_all))},
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "score_xp_v2_minutes_decay",
    }
    summary = {
        "treatment": TREATMENT,
        "baseline": BASELINE,
        "rho": RHO,
        "window": WINDOW,
        "formula": screen.get("formula"),
        "seasons": season_summaries,
        "cluster_interval": interval,
        "promote_if": screen.get("promote_if"),
        "covers_zero": covers_zero,
        "verdict": verdict,
    }
    rows_all.to_csv(PROCESSED / "score_xp_v2_minutes_decay.csv", index=False)
    gw_all.to_csv(PROCESSED / "score_xp_v2_minutes_decay_gw.csv", index=False)
    (PROCESSED / "score_xp_v2_minutes_decay_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (PROCESSED / "score_xp_v2_minutes_decay_intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )

    lines = [
        "# `score_xp_v2` minutes decay vs flat last-3",
        "",
        "Pre-registered: `protocol.score_xp_v2_minutes_decay`.",
        "",
        "Both arms use **predicted** minutes from GW `< t` (not oracle). "
        f"Treatment: geometric decay ρ={RHO} on last {WINDOW} GW rows (zeros included). "
        "Baseline: flat mean of the same window.",
        "",
        f"Formula: `{screen.get('formula')}`",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Result (treatment − baseline)",
        "",
        f"- Mean Δ log-score / GW: **{interval['mean']:.4f}** "
        f"[{interval['lo']:.4f}, {interval['hi']:.4f}]",
        f"- Covers 0: **{covers_zero}**",
        f"- Verdict: **{verdict}**",
        f"- Pooled player-fixture rows: {len(rows_all)}",
        "",
        "| Season | n | mean Δ log | MAE flat | MAE decay |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in season_summaries:
        lines.append(
            f"| {s['season']} | {s['n_rows']} | {s['mean_d_log']:+.4f} | "
            f"{s['mae_flat']:.3f} | {s['mae_dec']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Pool: actual minutes>60, FD AH+OU, GW5–38, σ=3 (same as score_xp_v2_screen).",
        "- Positive Δ = decay beats flat predicted minutes on Gaussian log score.",
        "- Live / L4 projection default stays flat (ρ=1) until promote_if clears.",
        "",
    ]
    write_gated_report(
        REPORTS / "score_xp_v2_minutes_decay.md",
        {"passed": True, "failures": [], "score_xp_v2": v2_asof},
        intervals,
        lines,
    )
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2))
    print(f"wrote {REPORTS / 'score_xp_v2_minutes_decay.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
