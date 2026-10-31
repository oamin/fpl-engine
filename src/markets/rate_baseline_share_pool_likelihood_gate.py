"""Gated likelihood: rate baseline + full-club shares vs published ``score_xp_v2``.

Pre-registered: ``protocol.score_xp_v2_rate_baseline_likelihood`` and
``protocol.score_xp_v2_full_share_likelihood``.

Two closed-season player-GW log-score screens on FD AH+OU moments (σ=3):

1. **Rate baseline** — ``rG``/``rA`` from prior+season
   (``apply_historical_rate_baseline``) vs same-season-only published rates.
   Shares stay full-club ``minutes>0`` (current pilot).
2. **Full vs thin shares** — both use same-season rates; treatment builds
   ``sG``/``sA`` on the full club (``minutes>0``) and projects onto the
   minutes>60 eval pool; baseline builds shares only among minutes>60.

Eval pool matches the v2 pilot: minutes>60, GW≥5, four closed seasons.

  PYTHONPATH=. python3 -m src.markets.rate_baseline_share_pool_likelihood_gate
  PYTHONPATH=. python3 -m src.markets.rate_baseline_share_pool_likelihood_gate --smoke
"""

from __future__ import annotations

import argparse
import json
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
from src.markets.numagent_xp import SCORE_NAME
from src.markets.numagent_xp_pilot import (
    GW_EVAL_START,
    MIN_MINUTES,
    SIGMA,
    _asof_alphas,
    load_fd_moments,
    load_merged,
)
from src.markets.rate_baseline import apply_historical_rate_baseline
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
STATE = ROOT / "REPORT_STATE.json"

BASE_V2 = SCORE_NAME
TREAT_RATE = f"{SCORE_NAME}_rate_baseline"
TREAT_FULL = f"{SCORE_NAME}_full_share"
BASE_THIN = f"{SCORE_NAME}_thin_share"

N_BOOT = 1000
SEED = 0


def _attach_rates_priors(
    apps: pd.DataFrame,
    hist: pd.DataFrame,
    *,
    season: str,
    rate_baseline: bool,
) -> pd.DataFrame:
    out = apps.copy()
    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    if rates.empty:
        for c in ("rG", "drG", "rA", "drA"):
            out[c] = 0.0
    else:
        out = out.merge(rates, on="element", how="left")
        for c in ("rG", "drG", "rA", "drA"):
            out[c] = out[c].fillna(0.0)
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
            out[c] = 0.0
        out["defcon_sig"] = numagent_xp.SIG_FLOOR
    else:
        out = out.merge(
            priors.drop(columns=["position"], errors="ignore"),
            on="element",
            how="left",
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
            if c not in out.columns:
                out[c] = d
            else:
                out[c] = out[c].fillna(d)
        out["defcon_n"] = out["defcon_n"].fillna(0).astype(int)

    out["m"] = out["minutes"].astype(float)
    out["sigma_m"] = 0.0
    out["player_id"] = out["element"]
    if rate_baseline:
        out = apply_historical_rate_baseline(out, season=season, hist=hist)
    return out


def _score_club(
    club: pd.DataFrame,
    *,
    xg: float,
    xcs: float,
    xgc: float,
    alpha_g: float,
    alpha_a: float,
    share_players: pd.DataFrame | None,
) -> pd.DataFrame:
    return numagent_xp.score_team_side(
        club,
        xg=float(xg),
        xcs=float(xcs),
        xgc=float(xgc),
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        share_players=share_players,
    )


def score_gameweek_arms(
    hist: pd.DataFrame,
    week: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    season: str,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    """Score one GW under baseline / rate-baseline / thin-share / full-share."""
    if week.empty:
        return pd.DataFrame()
    numagent_xp.assert_single_season_rate_history(hist)
    w = week.merge(fd, on=["team_n", "date_iso"], how="left")
    w = w[w["xg"].notna()].copy()
    if w.empty:
        return pd.DataFrame()
    apps = w[w["minutes"] > 0].copy()
    if apps.empty:
        return pd.DataFrame()

    base_frame = _attach_rates_priors(
        apps, hist, season=season, rate_baseline=False
    )
    rate_frame = _attach_rates_priors(
        apps, hist, season=season, rate_baseline=True
    )

    parts: list[pd.DataFrame] = []
    # One row per element for rate overlay (DGW can duplicate element across fixtures).
    rate_by_el = (
        rate_frame.drop_duplicates(subset=["element"], keep="first")
        .set_index("element")[["rG", "drG", "rA", "drA"]]
    )
    for (fx, team_n), g_base in base_frame.groupby(["fixture", "team_n"], sort=False):
        g_base = g_base.drop_duplicates(subset=["element"], keep="first").copy()
        ids = g_base["element"].to_numpy()
        g_rate = g_base.copy()
        overlay = rate_by_el.reindex(ids)
        for c in ("rG", "drG", "rA", "drA"):
            g_rate[c] = overlay[c].to_numpy(dtype=float)
            g_rate[c] = g_rate[c].fillna(0.0)
        xg = float(g_base["xg"].iloc[0])
        xcs = float(g_base["xcs"].iloc[0])
        xgc = float(g_base["xgc"].iloc[0])

        # Published / full-share: shares among minutes>0, no share_players arg.
        scored_base = _score_club(
            g_base,
            xg=xg,
            xcs=xcs,
            xgc=xgc,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_players=None,
        )
        scored_rate = _score_club(
            g_rate,
            xg=xg,
            xcs=xcs,
            xgc=xgc,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_players=None,
        )

        thin = g_base.loc[
            pd.to_numeric(g_base["minutes"], errors="coerce").fillna(0.0)
            > MIN_MINUTES
        ].copy()
        if thin.empty:
            continue
        # Thin shares among >60 only; full projects m>0 shares onto thin pool.
        scored_thin = _score_club(
            thin,
            xg=xg,
            xcs=xcs,
            xgc=xgc,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_players=None,
        )
        scored_full = _score_club(
            thin,
            xg=xg,
            xcs=xcs,
            xgc=xgc,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            share_players=g_base,
        )

        keep_ids = set(scored_thin["element"].astype(int))
        sb = scored_base.loc[scored_base["element"].astype(int).isin(keep_ids)].copy()
        sr = scored_rate.loc[scored_rate["element"].astype(int).isin(keep_ids)].copy()
        if sb.empty or sr.empty or scored_thin.empty or scored_full.empty:
            continue

        m = sb[["element", "total_points", "minutes", "position", "xp", "xp_se"]].rename(
            columns={"xp": "xp_base", "xp_se": "xp_se_base"}
        )
        m = m.merge(
            sr[["element", "xp", "xp_se"]].rename(
                columns={"xp": "xp_rate", "xp_se": "xp_se_rate"}
            ),
            on="element",
            how="inner",
        )
        m = m.merge(
            scored_thin[["element", "xp", "xp_se"]].rename(
                columns={"xp": "xp_thin", "xp_se": "xp_se_thin"}
            ),
            on="element",
            how="inner",
        )
        m = m.merge(
            scored_full[["element", "xp", "xp_se"]].rename(
                columns={"xp": "xp_full", "xp_se": "xp_se_full"}
            ),
            on="element",
            how="inner",
        )
        m["fixture"] = int(fx)
        m["team_n"] = str(team_n)
        m["gw"] = int(week["GW"].iloc[0])
        parts.append(m)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    # Pilot eval pool
    out = out.loc[
        pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0) > MIN_MINUTES
    ].copy()
    return out


def run_season(season: str, *, gw_end: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    stem = season.replace("-", "_")
    merged = load_merged(stem)
    fd = load_fd_moments(stem)
    parts: list[pd.DataFrame] = []
    gws = sorted(int(g) for g in merged["GW"].unique() if g >= 2)
    for gw in gws:
        if gw < GW_EVAL_START:
            continue
        if gw_end is not None and gw > int(gw_end):
            continue
        hist = merged[merged["GW"] < gw]
        week = merged[merged["GW"] == gw]
        ag, aa = _asof_alphas(hist)
        pair = score_gameweek_arms(
            hist, week, fd, season=season, alpha_g=ag, alpha_a=aa
        )
        if not pair.empty:
            parts.append(pair)
    if not parts:
        raise RuntimeError(f"no rows for {season}")
    rows = pd.concat(parts, ignore_index=True)
    rows["element"] = rows["element"].astype(int)
    rows["season"] = season

    y = rows["total_points"].to_numpy(float)
    rows["log_base"] = gaussian_log_score(y, rows["xp_base"].to_numpy(float), SIGMA)
    rows["log_rate"] = gaussian_log_score(y, rows["xp_rate"].to_numpy(float), SIGMA)
    rows["log_thin"] = gaussian_log_score(y, rows["xp_thin"].to_numpy(float), SIGMA)
    rows["log_full"] = gaussian_log_score(y, rows["xp_full"].to_numpy(float), SIGMA)
    rows["d_log_rate"] = rows["log_rate"] - rows["log_base"]
    rows["d_log_share"] = rows["log_full"] - rows["log_thin"]
    rows["abs_err_base"] = np.abs(y - rows["xp_base"].to_numpy(float))
    rows["abs_err_rate"] = np.abs(y - rows["xp_rate"].to_numpy(float))
    rows["abs_err_thin"] = np.abs(y - rows["xp_thin"].to_numpy(float))
    rows["abs_err_full"] = np.abs(y - rows["xp_full"].to_numpy(float))

    gw = (
        rows.groupby("gw", sort=True)
        .agg(
            n=("d_log_rate", "size"),
            d_log_rate=("d_log_rate", "mean"),
            d_log_share=("d_log_share", "mean"),
            log_base=("log_base", "mean"),
            log_rate=("log_rate", "mean"),
            log_thin=("log_thin", "mean"),
            log_full=("log_full", "mean"),
            mae_base=("abs_err_base", "mean"),
            mae_rate=("abs_err_rate", "mean"),
            mae_thin=("abs_err_thin", "mean"),
            mae_full=("abs_err_full", "mean"),
        )
        .reset_index()
    )
    gw["season"] = season
    return rows, gw


def _verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "promote"
    if hi < 0:
        return "park_clear_lose"
    return "inconclusive"


def run(*, smoke: bool = False) -> dict[str, Any]:
    protocol = load_protocol()
    rate_screen = protocol.get("score_xp_v2_rate_baseline_likelihood") or {}
    share_screen = protocol.get("score_xp_v2_full_share_likelihood") or {}
    if not rate_screen or not share_screen:
        raise RuntimeError(
            "protocol.score_xp_v2_rate_baseline_likelihood / "
            "score_xp_v2_full_share_likelihood missing — pre-register first"
        )

    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )

    seasons = ["2025-26"] if smoke else list(CLOSED_SEASONS)
    gw_end = 12 if smoke else None
    t0 = time.time()
    row_parts: list[pd.DataFrame] = []
    gw_parts: list[pd.DataFrame] = []
    season_summaries: list[dict[str, Any]] = []
    for season in seasons:
        print(f"rate/share likelihood {season} …", flush=True)
        rows, gw = run_season(season, gw_end=gw_end)
        row_parts.append(rows)
        gw_parts.append(gw)
        season_summaries.append(
            {
                "season": season,
                "n_rows": int(len(rows)),
                "n_gws": int(len(gw)),
                "mean_d_log_rate": float(gw["d_log_rate"].mean()),
                "mean_d_log_share": float(gw["d_log_share"].mean()),
                "mae_base": float(rows["abs_err_base"].mean()),
                "mae_rate": float(rows["abs_err_rate"].mean()),
                "mae_thin": float(rows["abs_err_thin"].mean()),
                "mae_full": float(rows["abs_err_full"].mean()),
            }
        )
        print(
            f"  n={len(rows)} d_rate={season_summaries[-1]['mean_d_log_rate']:+.4f} "
            f"d_share={season_summaries[-1]['mean_d_log_share']:+.4f}",
            flush=True,
        )

    rows_all = pd.concat(row_parts, ignore_index=True)
    gw_all = pd.concat(gw_parts, ignore_index=True)
    elapsed = time.time() - t0

    if smoke:
        out = {
            "smoke": True,
            "seasons": seasons,
            "season_summaries": season_summaries,
            "elapsed_s": elapsed,
            "n_rows": int(len(rows_all)),
        }
        PROCESSED.mkdir(parents=True, exist_ok=True)
        (PROCESSED / "rate_baseline_share_pool_likelihood_smoke.json").write_text(
            json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(out, indent=2))
        return out

    rate_deltas = {
        season: gw_all.loc[gw_all["season"] == season, "d_log_rate"].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    share_deltas = {
        season: gw_all.loc[gw_all["season"] == season, "d_log_share"].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    for name, deltas in (("rate", rate_deltas), ("share", share_deltas)):
        if set(deltas) != set(CLOSED_SEASONS):
            raise RuntimeError(f"{name}: missing seasons {sorted(deltas)}")
        for season, arr in deltas.items():
            if arr.size < 20:
                raise RuntimeError(f"{name} {season}: <20 GWs ({arr.size})")

    rate_iv = cluster_interval(
        rate_deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED
    )
    share_iv = cluster_interval(
        share_deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED
    )
    rate_key = comparison_key(TREAT_RATE, BASE_V2)
    share_key = comparison_key(TREAT_FULL, BASE_THIN)
    rate_verdict = _verdict(float(rate_iv["lo"]), float(rate_iv["hi"]))
    share_verdict = _verdict(float(share_iv["lo"]), float(share_iv["hi"]))

    comparisons = {
        **_certified_protocol_comparison(),
        rate_key: {
            **{k: rate_iv[k] for k in ("mean", "lo", "hi", "n_gws")},
            "incomplete_seasons": {},
            "covers_zero": bool(rate_iv["lo"] <= 0 <= rate_iv["hi"]),
            "verdict": rate_verdict,
            "treatment": TREAT_RATE,
            "baseline": BASE_V2,
            "n_rows": int(len(rows_all)),
            "screen": "score_xp_v2_rate_baseline_likelihood",
        },
        share_key: {
            **{k: share_iv[k] for k in ("mean", "lo", "hi", "n_gws")},
            "incomplete_seasons": {},
            "covers_zero": bool(share_iv["lo"] <= 0 <= share_iv["hi"]),
            "verdict": share_verdict,
            "treatment": TREAT_FULL,
            "baseline": BASE_THIN,
            "n_rows": int(len(rows_all)),
            "screen": "score_xp_v2_full_share_likelihood",
        },
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "rate_baseline_share_pool_likelihood",
    }

    summary = {
        "rate": {
            "treatment": TREAT_RATE,
            "baseline": BASE_V2,
            "cluster_interval": {
                k: rate_iv[k] for k in ("mean", "lo", "hi", "n_gws")
            },
            "covers_zero": comparisons[rate_key]["covers_zero"],
            "verdict": rate_verdict,
        },
        "share": {
            "treatment": TREAT_FULL,
            "baseline": BASE_THIN,
            "cluster_interval": {
                k: share_iv[k] for k in ("mean", "lo", "hi", "n_gws")
            },
            "covers_zero": comparisons[share_key]["covers_zero"],
            "verdict": share_verdict,
        },
        "seasons": season_summaries,
        "elapsed_s": elapsed,
        "n_rows": int(len(rows_all)),
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows_all.to_csv(
        PROCESSED / "rate_baseline_share_pool_likelihood.csv", index=False
    )
    gw_all.to_csv(
        PROCESSED / "rate_baseline_share_pool_likelihood_gw.csv", index=False
    )
    (PROCESSED / "rate_baseline_share_pool_likelihood_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (PROCESSED / "rate_baseline_share_pool_likelihood_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# Rate baseline + full-club share likelihood gates",
        "",
        "Pre-registered:",
        "",
        "- `protocol.score_xp_v2_rate_baseline_likelihood`",
        "- `protocol.score_xp_v2_full_share_likelihood`",
        "",
        "Published FD AH+OU `score_xp_v2` stack (minutes oracle, GW5–38, σ=3). "
        "Not a climb.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## 1. Rate baseline − published same-season rates",
        "",
        f"- Mean Δ log / GW: **{rate_iv['mean']:+.4f}** "
        f"[{rate_iv['lo']:+.4f}, {rate_iv['hi']:+.4f}]",
        f"- Covers 0: **{comparisons[rate_key]['covers_zero']}**",
        f"- Verdict: **{rate_verdict}**",
        f"- Rows: {len(rows_all)}",
        "",
        "## 2. Full-club shares − thin (>60′) shares",
        "",
        f"- Mean Δ log / GW: **{share_iv['mean']:+.4f}** "
        f"[{share_iv['lo']:+.4f}, {share_iv['hi']:+.4f}]",
        f"- Covers 0: **{comparisons[share_key]['covers_zero']}**",
        f"- Verdict: **{share_verdict}**",
        "",
        "| Season | n | Δ rate | Δ share | MAE base | MAE rate | MAE thin | MAE full |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in season_summaries:
        lines.append(
            f"| {s['season']} | {s['n_rows']} | {s['mean_d_log_rate']:+.4f} | "
            f"{s['mean_d_log_share']:+.4f} | {s['mae_base']:.3f} | "
            f"{s['mae_rate']:.3f} | {s['mae_thin']:.3f} | {s['mae_full']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Positive rate Δ = prior+season baseline beats published same-season "
        "`rG`/`rA` on Gaussian log score.",
        "- Positive share Δ = building shares on full club `m>0` and projecting "
        "onto the >60′ pool beats shares built only among >60′.",
        "- Promote either arm into published / centreline callers only after "
        "this gate; then run a joint-path climb sanity check.",
        f"- Wall {elapsed:.0f}s.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.markets.rate_baseline_share_pool_likelihood_gate",
        "```",
        "",
    ]
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}
    write_gated_report(
        REPORTS / "rate_baseline_share_pool_likelihood_gate.md",
        audit_out,
        intervals,
        lines,
    )

    STATE.write_text(
        json.dumps(
            {
                "task": "rate_baseline_share_pool_likelihood",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated rate-baseline + full-share likelihood screens",
                "verdicts": {
                    "rate": rate_verdict,
                    "share": share_verdict,
                },
                "means": {
                    "rate": float(rate_iv["mean"]),
                    "share": float(share_iv["mean"]),
                },
                "report": "reports/rate_baseline_share_pool_likelihood_gate.md",
                "elapsed_s": elapsed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    run(smoke=bool(args.smoke))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
