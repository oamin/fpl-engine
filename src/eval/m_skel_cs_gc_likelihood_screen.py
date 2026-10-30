"""Gated screen: same-skeleton discrete CS/GC vs Gaussian CS/GC (GKP/DEF).

Pre-registered: ``protocol.m_skel_cs_gc_likelihood_screen``.

Both arms use the same five-term log score from the locked FD centreline
board (Gaussian G/A/rest). Only CS/GC laws differ:

* treatment — Bernoulli P(CS) from M; categorical ⌊GA/2⌋ from M
* baseline — Gaussian(P_CS | xp_cs, σ); Gaussian(P_GC | xp_gc, σ)

Pool is GKP/DEF only (MID/FWD excluded so null CS/GC terms cannot inflate Δ).
Mean layer untouched.

  PYTHONPATH=. python3 -m src.eval.m_skel_cs_gc_likelihood_screen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval import m_discrete_event_likelihood_screen as base
from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "m_skel_cs_gc_likelihood_screen.md"
STATE = ROOT / "REPORT_STATE.json"
OUT = Path("/opt/cursor/artifacts/m_skel_cs_gc_likelihood_screen.json")
PROCESSED = ROOT / "data" / "processed"

TREATMENT = "l3_skel_disc_cs_gc"
BASELINE = "l3_skel_gauss_cs_gc"
POSITIONS = ("GKP", "DEF")
SIGMA = base.SIGMA


def _now() -> float:
    return time.time()


def _skel_rows(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse centreline board rows; keep GKP/DEF; score both skeleton arms."""
    rows, _ = base.run_season(season)
    rows = rows.loc[rows["position"].isin(POSITIONS)].copy()
    if rows.empty:
        raise RuntimeError(f"no GKP/DEF eligible rows for {season}")

    log_cs_g = gaussian_log_score(
        rows["P_CS"].to_numpy(float), rows["xp_cs"].to_numpy(float), SIGMA
    )
    log_gc_g = gaussian_log_score(
        rows["P_GC"].to_numpy(float), rows["xp_gc"].to_numpy(float), SIGMA
    )
    rows["log_cs_gauss"] = log_cs_g
    rows["log_gc_gauss"] = log_gc_g
    rows["log_disc"] = (
        rows["log_g"]
        + rows["log_a"]
        + rows["log_cs"]
        + rows["log_gc"]
        + rows["log_rest"]
    )
    rows["log_gauss_skel"] = (
        rows["log_g"]
        + rows["log_a"]
        + rows["log_cs_gauss"]
        + rows["log_gc_gauss"]
        + rows["log_rest"]
    )
    rows["d_log"] = rows["log_disc"] - rows["log_gauss_skel"]
    # Drop the prior screen's incommensurate total-Gaussian baseline column use.
    gw = (
        rows.groupby(["season", "gw"], as_index=False)
        .agg(
            d_log=("d_log", "mean"),
            log_disc=("log_disc", "mean"),
            log_gauss_skel=("log_gauss_skel", "mean"),
            n=("element", "count"),
        )
    )
    return rows, gw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("m_skel_cs_gc_likelihood_screen") or {}
    print("[skel-cs-gc] as-of audit…", flush=True)
    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )

    t0 = _now()
    row_parts: list[pd.DataFrame] = []
    gw_parts: list[pd.DataFrame] = []
    season_summaries: list[dict[str, Any]] = []

    for season in CLOSED_SEASONS:
        print(f"[skel-cs-gc] {season} …", flush=True)
        rows, gw = _skel_rows(season)
        row_parts.append(rows)
        gw_parts.append(gw)
        season_summaries.append(
            {
                "season": season,
                "n_rows": int(len(rows)),
                "n_gws": int(len(gw)),
                "mean_d_log": float(gw["d_log"].mean()),
                "mean_log_disc": float(rows["log_disc"].mean()),
                "mean_log_gauss_skel": float(rows["log_gauss_skel"].mean()),
                "mean_d_cs": float((rows["log_cs"] - rows["log_cs_gauss"]).mean()),
                "mean_d_gc": float((rows["log_gc"] - rows["log_gc_gauss"]).mean()),
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
        "screen": "m_skel_cs_gc_likelihood_screen",
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
        "task": "m_skel_cs_gc_likelihood_screen",
        "protocol": "m_skel_cs_gc_likelihood_screen",
        "pre_registered": "2026-10-30",
        "treatment": TREATMENT,
        "baseline": BASELINE,
        "positions": list(POSITIONS),
        "gw_range": [base.GW_START, base.GW_END],
        "pool": {
            "min_minutes": base.MIN_MINUTES,
            "min_history": base.MIN_HISTORY,
            "positions": list(POSITIONS),
        },
        "seasons": season_summaries,
        "cluster_interval": interval,
        "promote_if": screen.get("promote_if"),
        "covers_zero": bool(lo <= 0 <= hi),
        "verdict": verdict,
        "default_unchanged": True,
        "mean_layer_unchanged": True,
        "elapsed_s": _now() - t0,
        "component_mean_delta": {
            "d_cs": float((rows_all["log_cs"] - rows_all["log_cs_gauss"]).mean()),
            "d_gc": float((rows_all["log_gc"] - rows_all["log_gc_gauss"]).mean()),
        },
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows_all.to_csv(PROCESSED / "m_skel_cs_gc_likelihood_rows.csv", index=False)
    gw_all.to_csv(PROCESSED / "m_skel_cs_gc_likelihood_gw.csv", index=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (PROCESSED / "m_skel_cs_gc_likelihood_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n"
    )

    lines = [
        "# Same-skeleton discrete CS/GC vs Gaussian CS/GC (GKP/DEF)",
        "",
        "Pre-registered: `protocol.m_skel_cs_gc_likelihood_screen`.",
        "",
        "Both arms: five-term log score from the **same** locked FD centreline μ —",
        "Gaussian G/A/rest identical. Only CS/GC laws change.",
        "",
        f"- Treatment `{TREATMENT}`: Bernoulli P(CS) from M + categorical floor(GA/2) from M.",
        f"- Baseline `{BASELINE}`: Gaussian(P_CS | xp_cs, σ={SIGMA}) + "
        f"Gaussian(P_GC | xp_gc, σ={SIGMA}).",
        "",
        f"Pool: GKP/DEF · GW{base.GW_START}–{base.GW_END} · wall "
        f"{summary['elapsed_s']:.1f}s · n={len(rows_all)}",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Result (treatment − baseline)",
        "",
        f"- Mean Δ log-score / GW: **{interval['mean']:+.4f}** "
        f"[{interval['lo']:+.4f}, {interval['hi']:+.4f}]",
        f"- Covers 0: **{summary['covers_zero']}**",
        f"- Verdict: **{summary['verdict']}**",
        "- Mean layer / defaults **unchanged**.",
        "",
        "| Season | n | mean Δ log | mean log_disc | mean log_gauss_skel | "
        "ΔCS | ΔGC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in season_summaries:
        lines.append(
            f"| {s['season']} | {s['n_rows']} | {s['mean_d_log']:+.4f} | "
            f"{s['mean_log_disc']:+.4f} | {s['mean_log_gauss_skel']:+.4f} | "
            f"{s['mean_d_cs']:+.4f} | {s['mean_d_gc']:+.4f} |"
        )
    cd = summary["component_mean_delta"]
    lines += [
        "",
        "## Component Δ (pooled, treatment − baseline)",
        "",
        f"- Δ log_CS: {cd['d_cs']:+.4f}",
        f"- Δ log_GC: {cd['d_gc']:+.4f}",
        "",
        "## Notes",
        "",
        "- Positive Δ = discrete CS/GC beats Gaussian CS/GC under the same skeleton.",
        "- MID/FWD excluded by design (null-term inflation in the prior diagnostic).",
        "- Follows parked `m_discrete_event_likelihood_screen` (incommensurate total Gaussian).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.m_skel_cs_gc_likelihood_screen",
        "```",
        "",
    ]
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}
    write_gated_report(REPORT, audit_out, intervals, lines)

    STATE.write_text(
        json.dumps(
            {
                "task": "m_skel_cs_gc_likelihood_screen",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated GKP/DEF same-skeleton discrete CS/GC vs Gaussian CS/GC",
                "verdict": {
                    "verdict": verdict,
                    "mean_d_log": interval["mean"],
                    "lo": lo,
                    "hi": hi,
                    "covers_zero": summary["covers_zero"],
                    "default_unchanged": True,
                    "mean_layer_unchanged": True,
                },
                "artifact": str(args.out),
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )

    proto_path = ROOT / "experiments" / "protocol.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8"))
    entry = proto.get("m_skel_cs_gc_likelihood_screen") or {}
    entry["status"] = (
        "done_promote"
        if verdict == "promote"
        else (
            "done_park_clear_lose"
            if verdict == "park_clear_lose"
            else "done_inconclusive"
        )
    )
    entry["result"] = (
        f"Δ log {interval['mean']:+.4f} [{lo:+.4f},{hi:+.4f}] → {verdict}; "
        "mean layer unchanged"
    )
    proto["m_skel_cs_gc_likelihood_screen"] = entry
    proto_path.write_text(
        json.dumps(proto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"[skel-cs-gc] done verdict={verdict} Δ={interval['mean']:+.4f} "
        f"[{lo:+.4f},{hi:+.4f}] → {REPORT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
