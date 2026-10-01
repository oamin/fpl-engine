"""Stage 21 — Smoothed Ridge (H=3 target) + xP squad / Ridge XI under FTs.

Experiments (same FT / V as stage 19b + switch penalty):
  1. ridge_h3_ft — Ridge trained on mean(points t..t+2)
  2. ridge_1gw_ft — stage-17 1-GW Ridge (control)
  3. xp_ft — xP control
  4. dual_xp_ridge_ft — xP init + transfers; Ridge XI + captain

Writes:
  data/processed/season_climb_stage21.csv
  data/plots/season_climb_stage21.png
  reports/stage_21_season_climb_smoothed.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.models.ridge_starters import (
    EVAL_SEASON,
    build_fresh_seasons,
    walk_forward_global_ridge_h3,
    walk_forward_global_ridge_starters,
)
from src.models.season_climb import summarize
from src.models.season_climb_budget import run_budgeted_season
from src.models.season_climb_ft import (
    HOLD_EPS,
    HORIZON,
    SWITCH_PENALTY,
    load_vaastav_roster,
)
from src.models.season_climb_hybrid import run_ft_season_layered

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

RIDGE_1GW = "score_ridge_global_starters"
RIDGE_H3 = "score_ridge_h3"
XP = "score_xp"

# Stage 19 baselines for comparison in report
STAGE19 = {
    "xp_ft": 1848.0,
    "ridge_ft": 1746.0,
}


def write_report(
    path: Path, summary: pd.DataFrame, weekly: pd.DataFrame, scored: list[int]
) -> None:
    def tot(name: str) -> float:
        hit = summary.loc[summary["method"] == name, "total_points"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    h3 = tot("ridge_h3_ft")
    r1 = tot("ridge_1gw_ft")
    xp = tot("xp_ft")
    dual = tot("dual_xp_ridge_ft")
    xp_b = tot("xp_budget")

    def tx(method: str) -> str:
        g = weekly.loc[weekly["method"] == method]
        if g.empty:
            return "n/a"
        return (
            f"tx={g['n_transfers'].mean():.2f}, "
            f"holds={(g['n_transfers'] == 0).sum()}/{len(g)}, "
            f"hits={g['hits'].sum():.0f}"
        )

    lines = [
        "# Stage 21 — Smoothed Ridge + dual portfolio/XI",
        "",
        "**1.** Ridge retrained on **forward mean points** "
        "`y = mean(pts_t, pts_{t+1}, pts_{t+2})` (within season/player).",
        "",
        "**2.** Transfer **hysteresis:** "
        f"`V − {SWITCH_PENALTY}×n_transfers` plus hold unless ΔV ≥ {HOLD_EPS}.",
        "",
        "**3.** **dual_xp_ridge:** xP for GW1 build + swaps; Ridge for XI/captain.",
        "",
        f"- Horizon V for swaps: H={HORIZON} (unchanged from stage 19b)",
        f"- GWs: **{scored[0]}–{scored[-1]}** (n={len(scored)})",
        "",
        "## Final standings (captain ×2 − hits)",
        "",
        "| method | total | mean/GW | vs ridge_h3 | vs stage19 xp_ft |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        m = str(r.method)
        if not (m.endswith("_ft") or m == "xp_budget"):
            continue
        if m == "xp_ft":
            s19 = f"{r.total_points - STAGE19['xp_ft']:+.0f}"
        elif m == "ridge_1gw_ft":
            s19 = f"{r.total_points - STAGE19['ridge_ft']:+.0f}"
        else:
            s19 = "—"
        lines.append(
            f"| {m} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - h3:+.0f} | {s19} |"
        )

    lines += [
        "",
        "## Transfer behaviour",
        "",
        f"- ridge_h3_ft: {tx('ridge_h3_ft')}",
        f"- ridge_1gw_ft: {tx('ridge_1gw_ft')}",
        f"- xp_ft: {tx('xp_ft')}",
        f"- dual_xp_ridge_ft: {tx('dual_xp_ridge_ft')}",
        "",
        "## Key deltas",
        "",
        f"- ridge_h3 vs ridge_1gw: **{h3 - r1:+.0f}**",
        f"- ridge_h3 vs xp_ft: **{h3 - xp:+.0f}**",
        f"- dual vs xp_ft: **{dual - xp:+.0f}**",
        f"- dual vs ridge_1gw: **{dual - r1:+.0f}**",
        f"- Best FT vs xp_budget ({xp_b:.0f}): **{max(h3, xp, dual) - xp_b:+.0f}**",
        "",
    ]

    best_ft = max(h3, xp, dual, r1)
    if h3 >= xp and h3 > r1 + 5:
        verdict = (
            f"PASS — smoothed Ridge leads FT methods "
            f"(h3 vs xp {h3 - xp:+.0f}, vs 1gw {h3 - r1:+.0f})"
        )
    elif dual >= xp and dual > r1 + 5:
        verdict = (
            f"PASS — dual xP/Ridge leads "
            f"(dual vs xp {dual - xp:+.0f}, vs 1gw {dual - r1:+.0f})"
        )
    elif best_ft >= xp - 20:
        verdict = (
            f"PARTIAL — close to xp_ft (best FT {best_ft:.0f} vs xp {xp:.0f}) "
            f"but no clear Ridge win"
        )
    else:
        verdict = (
            f"FAIL — xp_ft still leads; h3 {h3 - xp:+.0f}, dual {dual - xp:+.0f}"
        )

    lines += [
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_stage21.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_stage21.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    style = {
        "ridge_h3_ft": ("ridge_h3", "darkgreen", 2.8),
        "dual_xp_ridge_ft": ("dual xP/Ridge", "darkorange", 2.6),
        "xp_ft": ("xp_ft", "steelblue", 2.2),
        "ridge_1gw_ft": ("ridge_1gw", "crimson", 2.0),
        "xp_budget": ("xp_budget", "lightblue", 1.3),
    }
    for method, (label, color, lw) in style.items():
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        ax.plot(
            g["gw"],
            g["xi_points_cap"].cumsum(),
            "-o",
            ms=3,
            lw=lw,
            color=color,
            label=label,
        )
    ax.set_xlabel("Gameweek (2025/26)")
    ax.set_ylabel("Cumulative XI points (cap×2 − hits)")
    ax.set_title("Stage 21 — H3 Ridge + dual portfolio/XI")
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def run() -> dict[str, Any]:
    print("Building features…")
    all_feat = build_fresh_seasons()
    print("Walk-forward ridge 1-GW…")
    pred1, scored1 = walk_forward_global_ridge_starters(all_feat)
    print("Walk-forward ridge H=3 target…")
    pred_h3, scored_h3 = walk_forward_global_ridge_h3(all_feat)
    scored = sorted(set(scored1) & set(scored_h3))
    if not scored:
        raise RuntimeError("No scored GWs")

    eval_feat = all_feat.loc[all_feat["season"] == EVAL_SEASON].copy()
    eval_feat[RIDGE_1GW] = pred1.loc[eval_feat.index]
    eval_feat[RIDGE_H3] = pred_h3.loc[eval_feat.index]

    configs = {
        "ridge_1gw": {"init": RIDGE_1GW, "xfer": RIDGE_1GW, "xi": RIDGE_1GW},
        "ridge_h3": {"init": RIDGE_H3, "xfer": RIDGE_H3, "xi": RIDGE_H3},
        "xp": {"init": XP, "xfer": XP, "xi": XP},
        "dual_xp_ridge": {"init": XP, "xfer": XP, "xi": RIDGE_1GW},
    }

    roster = load_vaastav_roster(EVAL_SEASON)
    print(
        f"Stage 21 FT climb GWs {scored[0]}–{scored[-1]} "
        f"(switch_penalty={SWITCH_PENALTY})…"
    )
    weekly_ft = run_ft_season_layered(eval_feat, configs, scored, roster)
    weekly_b = run_budgeted_season(eval_feat, {"xp": XP}, scored)
    weekly = pd.concat([weekly_ft, weekly_b], ignore_index=True)

    counts = weekly.groupby("method")["gw"].nunique()
    weekly = weekly.loc[
        weekly["method"].isin(counts[counts >= len(scored)].index)
    ].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_stage21.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_stage21.png")
    write_report(REPORTS / "stage_21_season_climb_smoothed.md", summary, weekly, scored)
    return {"summary": summary, "weekly": weekly, "scored_gws": scored}


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_21_season_climb_smoothed.md")
