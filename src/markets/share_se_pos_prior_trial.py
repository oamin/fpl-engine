"""Trial: share-SE shrink with position×minutes prior vs raw ``score_xp_v2``.

Pre-registered: ``protocol.score_xp_v2_share_se_pos_prior``.
``s0 ∝ m · w(pos)`` with locked closed-season per-90 weights (MID=1).

  python -m src.markets.share_se_pos_prior_trial
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
)
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TREATMENT = f"{SCORE_NAME}_share_se_pos"
BASELINE = SCORE_NAME


def _score_sides(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
    share_se_shrink: bool,
    share_se_prior: str = "position_minutes",
) -> pd.DataFrame:
    from src.markets import numagent_xp

    numagent_xp.assert_single_season_rate_history(hist)
    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    w = week.merge(fd, on=["team_n", "date_iso"], how="left")
    w = w[w["xg"].notna()].copy()
    apps = w[w["minutes"] > 0].copy()
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
    apps["m"] = apps["minutes"].astype(float)
    apps["sigma_m"] = 0.0
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
            share_se_shrink=share_se_shrink,
            share_se_prior=share_se_prior,
        )
        parts.append(side)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out[out["minutes"] > 60.0].copy()
    out["gw"] = int(week["GW"].iloc[0])
    return out


def score_gameweek_pair(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    raw = _score_sides(
        hist, week, fd, alpha_g=alpha_g, alpha_a=alpha_a, share_se_shrink=False
    )
    shr = _score_sides(
        hist,
        week,
        fd,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        share_se_shrink=True,
        share_se_prior="position_minutes",
    )
    if raw.empty or shr.empty:
        return pd.DataFrame()
    keep = [
        "element",
        "gw",
        "fixture",
        "team_n",
        "xp",
        "xp_se",
        "sG",
        "sA",
        "total_points",
        "minutes",
        "position",
    ]
    raw_k = raw[[c for c in keep if c in raw.columns]].copy()
    shr_k = shr[[c for c in keep if c in shr.columns]].copy()
    raw_k = raw_k.rename(
        columns={"xp": "xp_raw", "xp_se": "xp_se_raw", "sG": "sG_raw", "sA": "sA_raw"}
    )
    shr_k = shr_k.rename(
        columns={"xp": "xp_shr", "xp_se": "xp_se_shr", "sG": "sG_shr", "sA": "sA_shr"}
    )
    # Avoid total_points_x/_y collision on merge.
    shr_k = shr_k.drop(columns=["total_points"], errors="ignore")
    on = [c for c in ("element", "gw", "fixture", "team_n") if c in raw_k.columns]
    return raw_k.merge(shr_k, on=on, how="inner")


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
    mu1 = rows["xp_shr"].to_numpy(float)
    rows["log_raw"] = gaussian_log_score(y, mu0, SIGMA)
    rows["log_shr"] = gaussian_log_score(y, mu1, SIGMA)
    rows["d_log"] = rows["log_shr"] - rows["log_raw"]
    rows["abs_err_raw"] = np.abs(y - mu0)
    rows["abs_err_shr"] = np.abs(y - mu1)

    gw = (
        rows.groupby("gw", sort=True)
        .agg(
            n=("d_log", "size"),
            d_log=("d_log", "mean"),
            log_raw=("log_raw", "mean"),
            log_shr=("log_shr", "mean"),
            mae_raw=("abs_err_raw", "mean"),
            mae_shr=("abs_err_shr", "mean"),
        )
        .reset_index()
    )
    gw["season"] = season
    return rows, gw


def run() -> dict[str, Any]:
    protocol = load_protocol()
    screen = protocol.get("score_xp_v2_share_se_pos_prior") or {}
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
        print(f"share-SE position prior trial {season} …", flush=True)
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
                "mae_shr": float(rows["abs_err_shr"].mean()),
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
        "screen": "score_xp_v2_share_se_pos_prior",
    }

    lo = float(interval.get("lo", -1))
    hi = float(interval.get("hi", 1))
    if lo > 0:
        verdict = "promote"
    elif hi < 0:
        verdict = "park_clear_lose"
    else:
        verdict = "inconclusive"

    summary = {
        "treatment": TREATMENT,
        "baseline": BASELINE,
        "formula": screen.get("formula"),
        "seasons": season_summaries,
        "cluster_interval": interval,
        "promote_if": screen.get("promote_if"),
        "covers_zero": bool(lo <= 0 <= hi),
        "verdict": verdict,
    }

    rows_all.to_csv(PROCESSED / "score_xp_v2_share_se_pos_prior.csv", index=False)
    gw_all.to_csv(PROCESSED / "score_xp_v2_share_se_pos_prior_gw.csv", index=False)
    (PROCESSED / "score_xp_v2_share_se_pos_prior_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (PROCESSED / "score_xp_v2_share_se_pos_prior_intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )

    lines = [
        "# `score_xp_v2` share-SE shrink (position×minutes prior) vs raw",
        "",
        "Pre-registered: `protocol.score_xp_v2_share_se_pos_prior`.",
        "",
        "Treatment shrinks **sG/sA** toward ``s0 ∝ m · w(pos)`` with locked "
        "closed-season per-90 position weights (MID=1; GKP goals 0). Addresses "
        "sparse early-season CB goal monopoly without all-minutes invent.",
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
        f"- Covers 0: **{summary['covers_zero']}**",
        f"- Verdict: **{summary['verdict']}**",
        f"- Pooled player-fixture rows: {len(rows_all)}",
        "",
        "| Season | n | mean Δ log | MAE raw | MAE shr |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for s in season_summaries:
        lines.append(
            f"| {s['season']} | {s['n_rows']} | {s['mean_d_log']:+.4f} | "
            f"{s['mae_raw']:.3f} | {s['mae_shr']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Pool matches `score_xp_v2_screen` (minutes>60, FD AH+OU, GW5–38, σ=3).",
        "- Positive Δ = position×minutes share-SE beats raw on Gaussian log score.",
        "- Weights locked from pooled 2022-23…2025-26 per-90 rates (not fit on holdout).",
        "",
    ]
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}
    write_gated_report(
        REPORTS / "score_xp_v2_share_se_pos_prior.md",
        audit_out,
        intervals,
        lines,
    )
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2))
    print(f"wrote {REPORTS / 'score_xp_v2_share_se_pos_prior.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
