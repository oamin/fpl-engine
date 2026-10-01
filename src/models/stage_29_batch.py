"""Stage 29 — pre-registered batch, fast screen first.

Arm 1: score = score_xp - 0.25 * sigma. Fast XI screen. Skip the
free-transfer climb if that total is 100 or more below paired xp.
Arm 2: XI and captain stay on score_xp. The same risk score is used
only inside transfer value. Free-transfer climb only.
Arm 3: score_xp, SWITCH_PENALTY 2.5 instead of 1.0. Free-transfer climb only.

Pass bar on the free-transfer climb remains paired xp_ft + 34.
2025/26, GW5–38, H=3. No chips.

Writes:
  data/processed/stage_29_batch.csv
  reports/stage_29_batch.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.ridge_multiseason import EVAL_SEASON, SEASONS, build_one_season
from src.models.season_climb import pick_xi, summarize
from src.models.season_climb_ft import HORIZON, SWITCH_PENALTY, load_vaastav_roster, run_ft_season
from src.models.sharpe_u import LAMBDA_RISK, add_causal_sharpe_u

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PASS_MARGIN = 34.0
KILL_GAP = 100.0
SWAP_PENALTY = 2.5
GW_START = 5
GW_END = 38


def _fd_code(season: str) -> str:
    for s, code in SEASONS:
        if s == season:
            return code
    raise KeyError(season)


def fast_xi(feat: pd.DataFrame, gws: list[int], cols: dict[str, str]) -> pd.DataFrame:
    """Best eligible XI each week. No squad carried forward, no transfer search."""
    rows: list[dict[str, Any]] = []
    for gw in gws:
        gw_df = feat.loc[(feat["gw"] == gw) & (feat["eligible"])].copy()
        if gw_df["position"].nunique() < 4:
            continue
        for name, col in cols.items():
            sel, form = pick_xi(gw_df, col)
            cap_idx = sel[col].idxmax()
            pts = float(sel["total_points"].sum())
            cap = float(sel.loc[cap_idx, "total_points"])
            rows.append(
                {
                    "gw": int(gw),
                    "method": name,
                    "mode": "fast_xi",
                    "xi_points": pts,
                    "xi_points_cap": pts + cap,
                    "hit_cost": 0.0,
                    "n_transfers": 0,
                    "hits": 0,
                    "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                }
            )
    return pd.DataFrame(rows)


def _tot(summary: pd.DataFrame, method: str) -> float:
    hit = summary.loc[summary["method"] == method, "total_points"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def _tx(weekly: pd.DataFrame, method: str) -> tuple[float, float, float, int]:
    frame = weekly.loc[weekly["method"] == method]
    if frame.empty:
        return (float("nan"), float("nan"), float("nan"), 0)
    return (
        float(frame["n_transfers"].mean()),
        float(frame["hits"].sum()),
        float(frame["hit_cost"].sum()),
        int((frame["n_transfers"] == 0).sum()),
    )


def write_report(
    path: Path,
    fast_summary: pd.DataFrame,
    ft_summary: pd.DataFrame,
    weekly: pd.DataFrame,
    gws: list[int],
    arm1_killed: bool,
) -> None:
    xp_fast = _tot(fast_summary, "xp_fast")
    risk_fast = _tot(fast_summary, "risk_fast")
    xp_ft = _tot(ft_summary, "xp_ft")
    lines = [
        "# Stage 29 — method batch",
        "",
        "Locked with the Co-PI before any total was read. 2025/26, "
        f"GW{gws[0]}–{gws[-1]} (n={len(gws)}).",
        f"Fast XI screen has no squad continuity. Kill arm 1 if its fast total "
        f"is {KILL_GAP:.0f} or more below paired xp.",
        f"Free-transfer pass bar: paired xp_ft + {PASS_MARGIN:.0f}. "
        f"Default swap penalty is {SWITCH_PENALTY}.",
        "",
        "## Fast XI screen",
        "",
        "| method | total | vs xp_fast |",
        "|---|---:|---:|",
        f"| xp_fast | {xp_fast:.0f} | +0 |",
        f"| risk_fast (xp - {LAMBDA_RISK} sigma) | {risk_fast:.0f} | {risk_fast - xp_fast:+.0f} |",
        "",
        f"Arm 1 free-transfer climb: **{'skipped' if arm1_killed else 'run'}**.",
        "",
        "## Free-transfer climb (captain ×2 − hits)",
        "",
        "| method | total | vs xp_ft | mean tx/GW | hits | hold weeks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["xp_ft", "risk_ft", "risk_value_ft", "swap_2_5_ft"]
    labels = {
        "xp_ft": "xp_ft",
        "risk_ft": "arm1 xp-0.25sigma everywhere",
        "risk_value_ft": "arm2 sigma inside V only",
        "swap_2_5_ft": "arm3 swap penalty 2.5",
    }
    for method in order:
        if method not in set(ft_summary["method"]):
            continue
        total = _tot(ft_summary, method)
        mean_tx, hits, _cost, holds = _tx(weekly, method)
        n_gw = int((weekly["method"] == method).sum())
        lines.append(
            f"| {labels[method]} | {total:.0f} | {total - xp_ft:+.0f} | "
            f"{mean_tx:.2f} | {hits:.0f} | {holds}/{n_gw} |"
        )
    lines += [
        "",
        "Arm 2 picks the XI and the captain on expected points. "
        "Sigma changes only which transfers look good.",
        "",
        "## Read",
        "",
    ]
    survivors = []
    for method in ("risk_ft", "risk_value_ft", "swap_2_5_ft"):
        if method not in set(ft_summary["method"]):
            continue
        delta = _tot(ft_summary, method) - xp_ft
        if delta >= PASS_MARGIN:
            survivors.append(f"{labels[method]} ({delta:+.0f})")
    if survivors:
        lines.append(
            "Cleared the pre-registered bar: " + "; ".join(survivors) + ". "
            "That is the best of this batch, not a confirmed edge."
        )
    else:
        lines.append(
            "None cleared xp_ft + 34. Report the misses; do not retune lambda, "
            "the kill gap, or the swap penalty against these totals."
        )
    lines += ["", "- `data/processed/stage_29_batch.csv`", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print(f"Building {EVAL_SEASON}…", flush=True)
    feat = build_one_season(EVAL_SEASON, _fd_code(EVAL_SEASON))
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    feat = add_causal_sharpe_u(feat)
    gws = [
        g
        for g in sorted(int(x) for x in feat["gw"].unique())
        if GW_START <= g <= GW_END
    ]
    print("Fast XI screen…", flush=True)
    fast = fast_xi(feat, gws, {"xp_fast": "score_xp", "risk_fast": "score_risk"})
    fast_summary = summarize(fast)
    print(fast_summary.to_string(index=False), flush=True)
    xp_fast = _tot(fast_summary, "xp_fast")
    risk_fast = _tot(fast_summary, "risk_fast")
    arm1_killed = risk_fast <= xp_fast - KILL_GAP
    print(f"arm1 killed={arm1_killed}", flush=True)

    roster = load_vaastav_roster(EVAL_SEASON)
    frames = [fast]
    print("xp_ft…", flush=True)
    frames.append(
        run_ft_season(feat, {"xp": "score_xp"}, gws, roster=roster, horizon=HORIZON)
    )
    if not arm1_killed:
        print("arm1 risk_ft…", flush=True)
        frames.append(
            run_ft_season(
                feat, {"risk": "score_risk"}, gws, roster=roster, horizon=HORIZON
            )
        )
    print("arm2 risk inside V…", flush=True)
    frames.append(
        run_ft_season(
            feat,
            {"risk_value": "score_xp"},
            gws,
            roster=roster,
            horizon=HORIZON,
            value_col="score_risk",
        )
    )
    print("arm3 swap penalty…", flush=True)
    frames.append(
        run_ft_season(
            feat,
            {"swap_2_5": "score_xp"},
            gws,
            roster=roster,
            horizon=HORIZON,
            switch_penalty=SWAP_PENALTY,
        )
    )
    weekly = pd.concat(frames, ignore_index=True)
    ft_summary = summarize(weekly.loc[weekly["mode"] == "ft"])
    print(ft_summary.to_string(index=False), flush=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "stage_29_batch.csv", index=False)
    write_report(
        REPORTS / "stage_29_batch.md",
        fast_summary,
        ft_summary,
        weekly,
        gws,
        arm1_killed,
    )
    print(f"Wrote {REPORTS / 'stage_29_batch.md'}", flush=True)
    return {"fast": fast_summary, "ft": ft_summary, "killed": arm1_killed}


if __name__ == "__main__":
    run()
