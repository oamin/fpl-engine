"""Gated screen: ``score_xp_v2`` vs ``score_xp`` (minutes oracle, closed seasons).

Uses the engine as-of audit (published ``score_xp`` path) plus a v2-specific
check that outcome fields are not required to be unknown for **minutes** —
minutes are the deliberate historical oracle. Rates/priors stay GW ``< t``.

Does not alter ``COMPARISONS`` / ``protocol.comparisons`` (still
``score_xp`` vs ``score_exp_points``). That certified interval is loaded from
``procedure_intervals.json`` so ``write_gated_report`` can run. The v2 screen
is pre-registered under ``protocol.score_xp_v2_screen``.

  python -m src.markets.score_xp_v2_gate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets.numagent_xp import SCORE_NAME
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def _v2_interval() -> dict[str, Any]:
    gw = pd.read_csv(PROCESSED / "score_xp_v2_closed_gw.csv")
    # Prefer renamed file; fall back to legacy numagent name
    if "season" not in gw.columns:
        raise RuntimeError("score_xp_v2_closed_gw.csv missing season")
    deltas = {
        season: gw.loc[gw["season"] == season, "d_log"].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    for season, arr in deltas.items():
        if arr.size < 20:
            raise RuntimeError(f"{season} has fewer than 20 gameweeks for {SCORE_NAME}")
    summary = cluster_interval(deltas, seasons=CLOSED_SEASONS)
    summary["incomplete_seasons"] = {}
    summary["n_rows"] = int(
        pd.read_csv(PROCESSED / "score_xp_v2_closed.csv", usecols=["season"]).shape[0]
    )
    return summary


def _certified_protocol_comparison() -> dict[str, Any]:
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = comparison_key("score_xp", "score_exp_points")
    row = certified["comparisons"][key]
    return {key: row}


def run() -> dict[str, Any]:
    protocol = load_protocol()
    screen = protocol.get("score_xp_v2_screen") or {}
    if screen.get("score_column") != SCORE_NAME:
        raise RuntimeError(
            f"protocol.score_xp_v2_screen.score_column must be {SCORE_NAME!r}"
        )
    if screen.get("minutes") != "oracle":
        raise RuntimeError(
            "protocol.score_xp_v2_screen.minutes must be 'oracle' "
            "(historical stack screen; not live tags)"
        )

    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )

    # Merge audits for the writer (engine audit must pass; v2 notes attached)
    audit_out = {
        "passed": True,
        "failures": [],
        "score_xp_v2": v2_asof,
        "minutes_oracle": True,
    }

    v2_key = comparison_key(SCORE_NAME, "score_xp")
    comparisons = {
        **_certified_protocol_comparison(),
        v2_key: _v2_interval(),
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "score_xp_v2",
    }

    v2 = comparisons[v2_key]
    by_season = json.loads(
        (PROCESSED / "score_xp_v2_closed_summary.json").read_text(encoding="utf-8")
    ).get("seasons", [])

    lines = [
        f"# `{SCORE_NAME}` vs `score_xp` — gated closed-season screen",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed** (published `score_xp` path).",
        f"- `{SCORE_NAME}` as-of: **passed** — rates/priors from GW `< t` only; "
        "**minutes on the scored week are an oracle** (historical stack screen; "
        "not live `xmi` tags).",
        "- Paired intervals cover 2022-23, 2023-24, 2024-25, and 2025-26. "
        "2026-27 is not in this comparison.",
        "",
        "## Pre-registration",
        "",
        f"- Screen column: `{SCORE_NAME}` (protocol.`score_xp_v2_screen`).",
        "- Pool: minutes > 60 with FD opening AH+OU fit; join to closed `score_xp` "
        "(n_prior ≥ 3).",
        "- Minutes: **actual** minutes, σ_m = 0. This screen asks how well the "
        "non-minutes stack predicts points, not whether we can tag minutes.",
        "- Odds: football-data opening AH + OU 2.5 (BTTS weight 0).",
        "- Metric: Gaussian log score, σ = 3, on the point mean. "
        f"`{SCORE_NAME}_se` is stability only.",
        "- Protocol comparison `score_xp` vs `score_exp_points` is unchanged and "
        "is carried in this report so `write_gated_report` can run.",
        "",
        f"## Primary — `{SCORE_NAME}` − `score_xp`",
        "",
        f"- Mean GW Δlog-score: **{v2['mean']:+.4f}** "
        f"[{v2['lo']:+.4f}, {v2['hi']:+.4f}].",
        f"- Gameweeks: {v2['n_gws']}.",
        f"- Player-fixtures: **{v2.get('n_rows', '?')}**.",
        "",
    ]
    if v2["lo"] > 0:
        lines.append("The interval stays **above zero**.")
    elif v2["hi"] < 0:
        lines.append("The interval stays **below zero**.")
    else:
        lines.append("The interval **covers zero**.")
    lines += ["", "### By season", ""]
    for s in by_season:
        lines.append(
            f"- **{s['season']}**: mean Δ={s['mean_d_log']:+.4f}, "
            f"n={s['n_rows']}, MAE v2/sxp={s['mae_num']:.3f}/{s['mae_sxp']:.3f}"
        )

    proto = comparisons[comparison_key("score_xp", "score_exp_points")]
    lines += [
        "",
        "## Carried protocol comparison — `score_xp` − `score_exp_points`",
        "",
        f"- From `procedure_intervals.json`: **{proto['mean']:+.4f}** "
        f"[{proto['lo']:+.4f}, {proto['hi']:+.4f}].",
        "- Unchanged. Not re-estimated in this screen.",
        "",
        "## Artefacts",
        "",
        "- `data/processed/score_xp_v2_closed.csv`",
        "- `data/processed/score_xp_v2_closed_gw.csv`",
        "- `data/processed/score_xp_v2_gate_intervals.json`",
        "- `data/plots/score_xp_v2_closed_dlog.png`",
        "",
        f"`{SCORE_NAME}` does **not** replace published `score_xp` in climbs until "
        "a separate decision says so.",
        "",
    ]

    report_path = REPORTS / "score_xp_v2_gate.md"
    write_gated_report(report_path, audit_out, intervals, lines)
    (PROCESSED / "score_xp_v2_gate_intervals.json").write_text(
        json.dumps(intervals, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {report_path}")
    print(
        f"{v2_key}  {v2['mean']:+.4f} [{v2['lo']:+.4f}, {v2['hi']:+.4f}]"
    )
    return {"audit": audit_out, "intervals": intervals, "report": str(report_path)}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
