"""Stage 18 — Budgeted season climb (£100.0m squad → best XI).

Each GW (eligible pool from stage 17):
  1. Select a 15-man squad under FPL constraints using a score
  2. Pick best legal XI from that squad
  3. Captain = highest score in XI; bank actual points

Squad constraints:
  2 GKP, 5 DEF, 5 MID, 3 FWD
  Σ value ≤ 1000 (£100.0m; Vaastav value is tenths)
  ≤ 3 players per club

Also runs unconstrained eligible climb for the same scorer (comparison).

Primary scorer: ``ridge_global_starters`` (stage 17). Baselines: xp, exp_points, price.

Writes:
  data/processed/season_climb_budget.csv
  data/plots/season_climb_budget.png
  reports/stage_18_season_climb_budget.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.models.ridge_starters import (
    EVAL_SEASON,
    build_fresh_seasons,
    walk_forward_global_ridge_starters,
)
from src.models.season_climb import bank_squad_gw, pick_xi, summarize
from src.rules.fpl_2026 import BUDGET_TENTHS as BUDGET
from src.rules.fpl_2026 import MAX_PER_CLUB, SQUAD_QUOTA

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"


def pick_squad(
    gw_df: pd.DataFrame,
    score_col: str,
    budget: int = BUDGET,
    max_per_club: int = MAX_PER_CLUB,
) -> pd.DataFrame:
    """MILP: maximise Σ score subject to FPL squad rules (scipy.optimize.milp)."""
    df = gw_df.loc[np.isfinite(gw_df[score_col])].copy()
    df = df.drop_duplicates("player_id", keep="first")
    if len(df) < 15:
        raise RuntimeError(f"Need ≥15 candidates, got {len(df)}")

    df = df.reset_index(drop=True)
    n = len(df)
    scores = df[score_col].to_numpy(float)
    values = pd.to_numeric(df["value"], errors="coerce").fillna(50.0).to_numpy(float)
    positions = df["position"].astype(str).to_numpy()
    clubs = df["team_norm"].astype(str).to_numpy()

    # milp minimises → use negative scores
    c = -scores
    constraints: list[LinearConstraint] = [
        LinearConstraint(values, -np.inf, float(budget)),
        LinearConstraint(np.ones(n), 15.0, 15.0),
    ]
    for pos, need in SQUAD_QUOTA.items():
        row = (positions == pos).astype(float)
        constraints.append(LinearConstraint(row, float(need), float(need)))
    for club in sorted(set(clubs.tolist())):
        row = (clubs == club).astype(float)
        constraints.append(LinearConstraint(row, -np.inf, float(max_per_club)))

    res = milp(
        c=c,
        constraints=constraints,
        bounds=Bounds(0, 1),
        integrality=np.ones(n, dtype=int),
        options={"time_limit": 20},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"Squad MILP failed: {res.message}")

    chosen = np.where(res.x > 0.5)[0]
    if len(chosen) != 15:
        raise RuntimeError(f"Expected 15 players, got {len(chosen)}")
    return df.iloc[chosen].copy()


def pick_xi_from_squad(
    squad: pd.DataFrame, score_col: str
) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    return pick_xi(squad, score_col)


def run_budgeted_season(
    feat: pd.DataFrame,
    score_cols: dict[str, str],
    gws: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gw in gws:
        pool = feat.loc[(feat["gw"] == gw) & (feat["eligible"])].copy()
        if pool["position"].nunique() < 4:
            continue
        for method, col in score_cols.items():
            sub = pool.loc[np.isfinite(pool[col])]
            if len(sub) < 15 or sub["position"].nunique() < 4:
                continue
            try:
                squad = pick_squad(sub, col)
                banked = bank_squad_gw(squad, col, use_autosubs=True)
            except RuntimeError:
                continue
            form = banked["form"]
            pts_sum = float(banked["xi_points"])
            cap_pts = float(banked["cap_extra"])
            squad_val = float(pd.to_numeric(squad["value"], errors="coerce").sum())
            rows.append(
                {
                    "gw": int(gw),
                    "method": f"{method}_budget",
                    "mode": "budget",
                    "xi_points": pts_sum,
                    "xi_points_cap": pts_sum + cap_pts,
                    "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                    "squad_value": squad_val,
                    "n_eligible": int(len(pool)),
                    "n_autosubs": int(banked["n_autosubs"]),
                    "n_blank_intended": int(banked["n_blank_intended"]),
                    "n_blank_final": int(banked["n_blank_final"]),
                }
            )
    return pd.DataFrame(rows)


def run_free_season(
    feat: pd.DataFrame,
    score_cols: dict[str, str],
    gws: list[int],
) -> pd.DataFrame:
    """Unconstrained XI from eligible pool for delta-to-budget."""
    rows: list[dict[str, Any]] = []
    for gw in gws:
        pool = feat.loc[(feat["gw"] == gw) & (feat["eligible"])].copy()
        if pool["position"].nunique() < 4:
            continue
        for method, col in score_cols.items():
            sub = pool.loc[np.isfinite(pool[col])]
            if sub["position"].nunique() < 4:
                continue
            try:
                xi, form = pick_xi(sub, col)
            except RuntimeError:
                continue
            cap_idx = xi[col].idxmax()
            pts_sum = float(xi["total_points"].sum())
            cap_pts = float(xi.loc[cap_idx, "total_points"])
            rows.append(
                {
                    "gw": int(gw),
                    "method": f"{method}_free",
                    "mode": "free",
                    "xi_points": pts_sum,
                    "xi_points_cap": pts_sum + cap_pts,
                    "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                    "squad_value": float("nan"),
                    "n_eligible": int(len(pool)),
                }
            )
    return pd.DataFrame(rows)


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    rename_plot = {
        "ridge_global_starters_budget": ("ridge_budget", "crimson"),
        "ridge_global_starters_free": ("ridge_free", "tomato"),
        "xp_budget": ("xp_budget", "steelblue"),
        "xp_free": ("xp_free", "lightblue"),
        "exp_points_budget": ("exp_points_budget", "gray"),
        "price_budget": ("price_budget", "seagreen"),
    }
    for method, (label, color) in rename_plot.items():
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        ax.plot(
            g["gw"],
            g["xi_points_cap"].cumsum(),
            "-o",
            ms=3,
            lw=2.4 if "ridge" in method else 1.4,
            color=color,
            label=label,
        )
    ax.set_xlabel("Gameweek (2025/26)")
    ax.set_ylabel("Cumulative XI points (captain ×2)")
    ax.set_title("Budgeted climb (£100.0m / 15) vs free XI — eligible pool")
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path, summary: pd.DataFrame, weekly: pd.DataFrame, scored: list[int]
) -> None:
    def tot(name: str) -> float:
        hit = summary.loc[summary["method"] == name, "total_points"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    ridge_b = tot("ridge_global_starters_budget")
    ridge_f = tot("ridge_global_starters_free")
    xp_b = tot("xp_budget")
    exp_b = tot("exp_points_budget")

    lines = [
        "# Stage 18 — Budgeted season climb",
        "",
        "Each GW on the **eligible** (prior-3 max minutes ≥ 60) pool:",
        "",
        "1. Build a **15-man squad** maximising Σ score under FPL rules "
        f"(2/5/5/3, ≤{MAX_PER_CLUB}/club, Σ value ≤ **{BUDGET}** = £100.0m)",
        "2. Pick best legal **XI** from that squad by the same score",
        "3. Captain = top score in XI; bank actual points",
        "",
        "Fresh rebuild every GW (no transfer continuity). "
        "Primary scorer = stage-17 `ridge_global_starters`.",
        "",
        f"- GWs: **{scored[0]}–{scored[-1]}** (n={len(scored)})",
        "",
        "## Final standings (captain ×2)",
        "",
        "| method | mode | total | mean/GW | vs ridge_budget |",
        "|---|---|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        mode = "budget" if str(r.method).endswith("_budget") else "free"
        lines.append(
            f"| {r.method} | {mode} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - ridge_b:+.0f} |"
        )

    lines += [
        "",
        "## Budget tax (same scorer)",
        "",
        f"- Ridge free − budget: **{ridge_f - ridge_b:+.0f}** "
        f"({(ridge_f - ridge_b) / max(len(scored), 1):+.2f} / GW)",
        f"- Ridge budget vs exp budget: **{ridge_b - exp_b:+.0f}**",
        f"- Ridge budget vs xp budget: **{ridge_b - xp_b:+.0f}**",
        "",
    ]

    mean_val = weekly.loc[
        weekly["method"] == "ridge_global_starters_budget", "squad_value"
    ].mean()
    if np.isfinite(mean_val):
        lines.append(f"- Mean ridge budget squad value: **{mean_val:.0f}** / {BUDGET}")
        lines.append("")

    if ridge_b > exp_b + 10 and ridge_b > xp_b:
        verdict = (
            f"PASS — ridge_budget beats exp_budget ({ridge_b - exp_b:+.0f}) "
            f"and xp_budget ({ridge_b - xp_b:+.0f}); "
            f"budget tax vs free ridge {ridge_b - ridge_f:+.0f}"
        )
    elif ridge_b > exp_b + 10:
        verdict = (
            f"PARTIAL — beats exp_budget but not xp_budget "
            f"(vs xp {ridge_b - xp_b:+.0f})"
        )
    else:
        verdict = "FAIL — ridge_budget trails baselines under budget"

    lines += [
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_budget.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_budget.csv`",
        "",
        "## Read",
        "",
        "- Large free−budget gap ⇒ unconstrained climb was premium-inflated.",
        "- Ridge still useful under budget if it beats exp/xp on the budgeted gate.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print("Building features + walk-forward ridge_global_starters…")
    all_feat = build_fresh_seasons()
    pred, scored = walk_forward_global_ridge_starters(all_feat)
    if not scored:
        raise RuntimeError("No scored GWs")

    eval_feat = all_feat.loc[all_feat["season"] == EVAL_SEASON].copy()
    eval_feat["score_ridge_global_starters"] = pred.loc[eval_feat.index]

    score_cols = {
        "ridge_global_starters": "score_ridge_global_starters",
        "xp": "score_xp",
        "exp_points": "score_exp_points",
        "price": "score_price",
    }

    print(f"Budgeted climb on GWs {scored[0]}–{scored[-1]} (n={len(scored)})…")
    weekly_b = run_budgeted_season(eval_feat, score_cols, scored)
    print("Free (unconstrained) climb for comparison…")
    weekly_f = run_free_season(eval_feat, score_cols, scored)
    weekly = pd.concat([weekly_b, weekly_f], ignore_index=True)

    counts = weekly.groupby("method")["gw"].nunique()
    weekly = weekly.loc[weekly["method"].isin(counts[counts >= len(scored)].index)].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_budget.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_budget.png")
    write_report(REPORTS / "stage_18_season_climb_budget.md", summary, weekly, scored)
    return {"summary": summary, "weekly": weekly, "scored_gws": scored}


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_18_season_climb_budget.md")
