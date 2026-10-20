"""Four-season ``score_xp_v2`` walk-forward vs ``score_xp`` (prep for gated screen).

Pool matches the 2025/26 pilot: minutes > 60, actual m, σ_m=0, FD AH+OU.
Eval window GWs 5–38. Uses ``cluster_interval`` across closed seasons.

Does **not** call ``write_gated_report`` until the comparison is pre-registered
alongside the protocol ``score_xp − score_exp_points`` requirement and the
as-of audit is run on the intended feature path.

  python -m src.markets.numagent_xp_closed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.markets.numagent_xp import SCORE_NAME
from src.markets.numagent_xp_pilot import (
    GW_EVAL_START,
    SIGMA,
    run_pilot,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "data" / "plots"


def run_closed() -> dict[str, Any]:
    row_parts: list[pd.DataFrame] = []
    gw_parts: list[pd.DataFrame] = []
    season_summaries: list[dict[str, Any]] = []

    for season in CLOSED_SEASONS:
        print(f"scoring {season} …")
        result = run_pilot(season.replace("-", "_"))
        rows = result["rows"]
        gw = result["gw"].copy()
        gw["season"] = season
        # Protocol window
        gw = gw[gw["gw"] >= GW_EVAL_START].copy()
        rows = rows[rows["gw"] >= GW_EVAL_START].copy()
        row_parts.append(rows)
        gw_parts.append(gw)
        season_summaries.append(
            {
                "season": season,
                "n_rows": int(len(rows)),
                "n_gws": int(len(gw)),
                "mean_d_log": float(gw["d_log"].mean()) if len(gw) else float("nan"),
                "mae_num": float(rows["abs_err_num"].mean()) if len(rows) else float("nan"),
                "mae_sxp": float(rows["abs_err_sxp"].mean()) if len(rows) else float("nan"),
            }
        )
        print(
            f"  gw{GW_EVAL_START}-38  n={len(rows)}  "
            f"d_log={season_summaries[-1]['mean_d_log']:+.4f}"
        )

    rows_all = pd.concat(row_parts, ignore_index=True)
    gw_all = pd.concat(gw_parts, ignore_index=True)

    deltas_by_season = {
        season: gw_all.loc[gw_all["season"] == season, "d_log"].to_numpy(float)
        for season in CLOSED_SEASONS
    }
    interval = cluster_interval(deltas_by_season, seasons=CLOSED_SEASONS)

    summary: dict[str, Any] = {
        "gated": False,
        "score_name": SCORE_NAME,
        "pre_register": f"{SCORE_NAME}_minus_score_xp (not yet in protocol.comparisons)",
        "pool": "minutes > 60; join to score_xp (n_prior>=3 from closed rebuild)",
        "minutes_policy": "actual minutes, sigma_m=0",
        "odds": "football-data opening AH+OU (BTTS weight 0)",
        "gw_window": [GW_EVAL_START, 38],
        "sigma_log_score": SIGMA,
        "n_player_fixture": int(len(rows_all)),
        "seasons": season_summaries,
        "cluster_interval": interval,
        "positive_delta": f"{SCORE_NAME} higher log-score than score_xp",
        "next_for_gate": [
            "run_asof_audit on the scoring feature path",
            "pre-register comparison without breaking COMPARISONS requirement",
            "write_gated_report with protocol intervals + this cluster interval",
        ],
    }
    return {"rows": rows_all, "gw": gw_all, "summary": summary}


def write_outputs(result: dict[str, Any]) -> dict[str, str]:
    rows = result["rows"]
    gw = result["gw"]
    summary = result["summary"]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    row_path = PROCESSED / "score_xp_v2_closed.csv"
    gw_path = PROCESSED / "score_xp_v2_closed_gw.csv"
    json_path = PROCESSED / "score_xp_v2_closed_summary.json"
    plot_path = PLOTS / "score_xp_v2_closed_dlog.png"
    report_path = REPORTS / "score_xp_v2_closed.md"

    if SCORE_NAME not in rows.columns and "xp" in rows.columns:
        rows = rows.copy()
        rows[SCORE_NAME] = rows["xp"]
    keep = [
        c
        for c in [
            "season",
            "gw",
            "element",
            "name",
            "team",
            "position",
            "fixture_id",
            "minutes",
            "total_points",
            SCORE_NAME,
            "xp_se",
            "score_xp",
            "d_log",
            "log_numagent",
            "log_score_xp",
        ]
        if c in rows.columns
    ]
    rows[keep].to_csv(row_path, index=False)
    gw.to_csv(gw_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axhline(0.0, color="#888", lw=0.8)
    for season, g in gw.groupby("season", sort=True):
        ax.plot(g["gw"], g["d_log"], marker="o", ms=3, lw=1.0, label=season)
    iv = summary["cluster_interval"]
    ax.set_title(
        f"{SCORE_NAME} − score_xp · closed seasons · GW{GW_EVAL_START}–38 · "
        f"mean {iv['mean']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]"
    )
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Δ log-score")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    iv = summary["cluster_interval"]
    lines = [
        f"# `{SCORE_NAME}` vs `score_xp` — closed seasons (prep)",
        "",
        f"Candidate score column: **`{SCORE_NAME}`** (method spec was Numagent 2.0). "
        "Not gated. Same pool as the 2025/26 pilot (minutes > 60, σ_m = 0, FD openings). "
        "Cluster interval uses the protocol bootstrap (resample GWs within each season).",
        "",
        "## Cluster interval (GW 5–38)",
        "",
        f"- Mean Δlog-score ({SCORE_NAME} − score_xp): **{iv['mean']:+.4f}** "
        f"[{iv['lo']:+.4f}, {iv['hi']:+.4f}]",
        f"- Gameweeks per season: {iv['n_gws']}",
        "",
        "## By season",
        "",
    ]
    for s in summary["seasons"]:
        lines.append(
            f"- **{s['season']}**: n={s['n_rows']}, "
            f"mean Δ={s['mean_d_log']:+.4f}, "
            f"MAE v2/sxp={s['mae_num']:.3f}/{s['mae_sxp']:.3f}"
        )
    lines += [
        "",
        "## Gate checklist (remaining)",
        "",
        "- [ ] As-of audit on the feature path used here",
        f"- [ ] Pre-register `{SCORE_NAME}` vs `score_xp` without dropping the "
        "required `score_xp` vs `score_exp_points` interval",
        "- [ ] `write_gated_report`",
        "",
        "## Artefacts",
        "",
        f"- `{row_path.relative_to(ROOT)}`",
        f"- `{gw_path.relative_to(ROOT)}`",
        f"- `{json_path.relative_to(ROOT)}`",
        f"- `{plot_path.relative_to(ROOT)}`",
        "",
        "Older `numagent_xp_closed*` filenames are superseded by these.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "rows": str(row_path),
        "gw": str(gw_path),
        "summary": str(json_path),
        "plot": str(plot_path),
        "report": str(report_path),
    }


def main() -> None:
    result = run_closed()
    paths = write_outputs(result)
    iv = result["summary"]["cluster_interval"]
    print(
        f"closed  d_log={iv['mean']:+.4f} [{iv['lo']:+.4f}, {iv['hi']:+.4f}]  "
        f"n={result['summary']['n_player_fixture']}"
    )
    print(paths["report"])


if __name__ == "__main__":
    main()
