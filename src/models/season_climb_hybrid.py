"""Stage 20 — Hybrid μ under FT climb (Gemini dual-layer).

Hypothesis: Ridge wins unconstrained XI ranking but is twitchy on
marginal swaps; xP is smoother. Dual-layer:

  - GW1 / wildcard build + weekly XI / captain: ``ridge_global_starters``
  - Transfer ΔV (horizon V): ``μ_hybrid = α·z(xP) + (1−α)·z(Ridge)``
    with α = 0.6 (within-GW z-scores so scales match)

Comparators (same FT rules / same V machinery as stage 19b):
  - ridge_ft: ridge for init + transfers + XI
  - xp_ft: xp for all
  - hybrid_ft: dual-layer as above
  - hybrid_all_ft: hybrid for init + transfers + XI (ablation)

Writes:
  data/processed/season_climb_hybrid.csv
  data/plots/season_climb_hybrid.png
  reports/stage_20_season_climb_hybrid.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.ridge_starters import (
    EVAL_SEASON,
    build_fresh_seasons,
    walk_forward_global_ridge_starters,
)
from src.models.season_climb import bank_squad_gw, pick_xi, summarize
from src.models.season_climb_budget import run_budgeted_season
from src.models.season_climb_ft import (
    HIT_COST,
    HOLD_EPS,
    HORIZON,
    GAMMA,
    MAX_FT,
    MAX_HITS,
    SquadState,
    _as_int_value,
    _fill_score,
    _gw_pool,
    advance_ft,
    choose_transfers,
    initial_squad,
    load_vaastav_roster,
    sell_price,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

HYBRID_ALPHA = 0.6  # weight on xP in transfer μ
RIDGE_COL = "score_ridge_global_starters"
XP_COL = "score_xp"
HYBRID_COL = "score_hybrid"


def add_hybrid_scores(feat: pd.DataFrame, alpha: float = HYBRID_ALPHA) -> pd.DataFrame:
    """Within-GW z-scored blend: α·z(xP) + (1−α)·z(Ridge)."""
    out = feat.copy()
    z_xp = pd.Series(np.nan, index=out.index, dtype=float)
    z_ridge = pd.Series(np.nan, index=out.index, dtype=float)
    for _gw, idx in out.groupby("gw").groups.items():
        ix = list(idx)
        for src, dest in ((XP_COL, z_xp), (RIDGE_COL, z_ridge)):
            s = pd.to_numeric(out.loc[ix, src], errors="coerce")
            mu, sd = s.mean(), s.std(ddof=0)
            if not np.isfinite(sd) or sd < 1e-9:
                dest.loc[ix] = 0.0
            else:
                dest.loc[ix] = (s - mu) / sd
    out[HYBRID_COL] = alpha * z_xp + (1.0 - alpha) * z_ridge
    return out


def run_ft_season_layered(
    feat: pd.DataFrame,
    configs: dict[str, dict[str, str]],
    gws: list[int],
    roster: pd.DataFrame,
) -> pd.DataFrame:
    """configs[method] = {init, xfer, xi} score column names."""
    rows: list[dict[str, Any]] = []
    roster_by_gw: dict[int, set[str]] = {
        int(g): set(gdf["player_id"].astype(str))
        for g, gdf in roster.groupby("gw")
    }

    for method, cols in configs.items():
        init_col = cols["init"]
        xfer_col = cols["xfer"]
        xi_col = cols["xi"]
        state: SquadState | None = None

        for i, gw in enumerate(gws):
            owned = state.ids() if state else set()
            pool = _gw_pool(feat, roster, gw, owned)
            if pool["position"].nunique() < 4:
                continue

            if state is None:
                try:
                    state = initial_squad(pool, init_col)
                except RuntimeError:
                    break
                n_tx, hits = 0, 0
                ft_before = 0
            else:
                ft_before = state.ft
                new_state, n_tx, hits = choose_transfers(
                    state,
                    pool,
                    xfer_col,
                    gw=int(gw),
                    future_gws=list(gws),
                    roster_by_gw=roster_by_gw,
                )
                state = new_state

            squad_df = pool.loc[pool["player_id"].isin(state.ids())].copy()
            squad_df = squad_df.drop_duplicates("player_id", keep="first")
            if len(squad_df) < 11 or len(state.purchase) != 15:
                break
            squad_df[xi_col] = _fill_score(squad_df, xi_col)
            try:
                banked = bank_squad_gw(squad_df, xi_col, use_autosubs=True)
            except RuntimeError:
                break
            form = banked["form"]
            pts_sum = float(banked["xi_points"])
            cap_pts = float(banked["cap_extra"])
            hit_pts = HIT_COST * hits
            squad_val = float(
                sum(
                    sell_price(
                        state.purchase[str(r.player_id)],
                        _as_int_value(r.value),
                    )
                    for r in squad_df.itertuples()
                )
            )

            rows.append(
                {
                    "gw": int(gw),
                    "method": f"{method}_ft",
                    "mode": "ft",
                    "xi_points": pts_sum,
                    "xi_points_cap": pts_sum + cap_pts - hit_pts,
                    "hit_cost": hit_pts,
                    "n_transfers": n_tx,
                    "hits": hits,
                    "ft_before": ft_before,
                    "ft_after": 1 if i == 0 else advance_ft(ft_before, n_tx),
                    "bank": state.bank,
                    "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                    "squad_sell_value": squad_val,
                    "n_eligible": int(pool["eligible"].sum()),
                    "init_col": init_col,
                    "xfer_col": xfer_col,
                    "xi_col": xi_col,
                    "n_autosubs": int(banked["n_autosubs"]),
                    "n_blank_intended": int(banked["n_blank_intended"]),
                    "n_blank_final": int(banked["n_blank_final"]),
                }
            )
            if i == 0:
                state.ft = 1
            else:
                state.ft = advance_ft(ft_before, n_tx)

    return pd.DataFrame(rows)


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    style = {
        "hybrid_ft": ("hybrid_ft (dual)", "darkorange", 2.8),
        "hybrid_all_ft": ("hybrid_all_ft", "gold", 1.8),
        "xp_ft": ("xp_ft", "steelblue", 2.2),
        "ridge_global_starters_ft": ("ridge_ft", "crimson", 2.0),
        "ridge_global_starters_budget": ("ridge_budget", "tomato", 1.4),
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
    ax.set_title(
        f"Stage 20 — hybrid dual-layer FT climb (α={HYBRID_ALPHA}, H={HORIZON})"
    )
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

    hybrid = tot("hybrid_ft")
    hybrid_all = tot("hybrid_all_ft")
    ridge_ft = tot("ridge_global_starters_ft")
    xp_ft = tot("xp_ft")
    ridge_b = tot("ridge_global_starters_budget")

    def tx_stats(method: str) -> str:
        g = weekly.loc[weekly["method"] == method]
        if g.empty:
            return "n/a"
        return (
            f"mean_tx={g['n_transfers'].mean():.2f}, "
            f"holds={(g['n_transfers'] == 0).sum()}/{len(g)}, "
            f"max_ft={g['ft_before'].max():.0f}, "
            f"hits={g['hits'].sum():.0f}"
        )

    lines = [
        "# Stage 20 — Hybrid μ FT climb (dual-layer)",
        "",
        "Gemini recommendation: keep Ridge for free rebuild / XI; dampen swaps with xP.",
        "",
        f"- **hybrid_ft (dual):** init+XI = Ridge; transfers = "
        f"`{HYBRID_ALPHA}·z(xP) + {1 - HYBRID_ALPHA}·z(Ridge)`",
        f"- **hybrid_all_ft:** hybrid for init + transfers + XI",
        "- **ridge_ft / xp_ft:** single-μ controls (same V as stage 19b)",
        "",
        f"- V: Σ γ^h XI_score (H={HORIZON}, γ={GAMMA}) − 4·hits; "
        f"hold unless ΔV ≥ {HOLD_EPS}",
        f"- FTs: 1/GW stack to {MAX_FT}; max hits explored {MAX_HITS}",
        f"- GWs: **{scored[0]}–{scored[-1]}** (n={len(scored)})",
        "",
        "## Final standings (captain ×2 − hits)",
        "",
        "| method | mode | total | mean/GW | vs hybrid_ft |",
        "|---|---|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        mode = "ft" if str(r.method).endswith("_ft") else "budget"
        lines.append(
            f"| {r.method} | {mode} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - hybrid:+.0f} |"
        )

    lines += [
        "",
        "## Transfer behaviour",
        "",
        f"- hybrid_ft: {tx_stats('hybrid_ft')}",
        f"- hybrid_all_ft: {tx_stats('hybrid_all_ft')}",
        f"- xp_ft: {tx_stats('xp_ft')}",
        f"- ridge_ft: {tx_stats('ridge_global_starters_ft')}",
        "",
        "## Deltas",
        "",
        f"- hybrid_ft vs xp_ft: **{hybrid - xp_ft:+.0f}**",
        f"- hybrid_ft vs ridge_ft: **{hybrid - ridge_ft:+.0f}**",
        f"- hybrid_ft vs hybrid_all: **{hybrid - hybrid_all:+.0f}**",
        f"- hybrid_ft vs ridge_budget: **{hybrid - ridge_b:+.0f}**",
        "",
    ]

    # Target: close gap toward xp_ft 1848 and budget 2114
    if hybrid >= xp_ft and hybrid > ridge_ft + 10:
        verdict = (
            f"PASS — hybrid_ft leads FT methods "
            f"(vs xp {hybrid - xp_ft:+.0f}, vs ridge {hybrid - ridge_ft:+.0f})"
        )
    elif hybrid > ridge_ft + 10:
        verdict = (
            f"PARTIAL — beats ridge_ft ({hybrid - ridge_ft:+.0f}) "
            f"but not xp_ft ({hybrid - xp_ft:+.0f})"
        )
    elif hybrid > ridge_ft:
        verdict = (
            f"WEAK — hybrid edges ridge ({hybrid - ridge_ft:+.0f}) "
            f"but trails xp ({hybrid - xp_ft:+.0f})"
        )
    else:
        verdict = (
            f"FAIL — hybrid_ft does not beat ridge_ft "
            f"({hybrid - ridge_ft:+.0f}); vs xp {hybrid - xp_ft:+.0f}"
        )

    lines += [
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_hybrid.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_hybrid.csv`",
        "",
        "## Read",
        "",
        "- Dual-layer tests whether Ridge μ + smooth swap signal closes the FT gap.",
        "- hybrid_all isolates whether dual-layer (vs blend everywhere) matters.",
        "- Budget ceiling remains an upper bound, not the fair peer.",
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
    eval_feat[RIDGE_COL] = pred.loc[eval_feat.index]
    eval_feat = add_hybrid_scores(eval_feat, HYBRID_ALPHA)

    configs = {
        "ridge_global_starters": {
            "init": RIDGE_COL,
            "xfer": RIDGE_COL,
            "xi": RIDGE_COL,
        },
        "xp": {"init": XP_COL, "xfer": XP_COL, "xi": XP_COL},
        "hybrid": {
            "init": RIDGE_COL,
            "xfer": HYBRID_COL,
            "xi": RIDGE_COL,
        },
        "hybrid_all": {
            "init": HYBRID_COL,
            "xfer": HYBRID_COL,
            "xi": HYBRID_COL,
        },
    }

    roster = load_vaastav_roster(EVAL_SEASON)
    print(
        f"Stage 20 FT climb on GWs {scored[0]}–{scored[-1]} "
        f"(n={len(scored)}, α={HYBRID_ALPHA})…"
    )
    weekly_ft = run_ft_season_layered(eval_feat, configs, scored, roster)

    print("Budget rebuild (ridge/xp) for ceiling…")
    weekly_b = run_budgeted_season(
        eval_feat,
        {
            "ridge_global_starters": RIDGE_COL,
            "xp": XP_COL,
        },
        scored,
    )
    weekly = pd.concat([weekly_ft, weekly_b], ignore_index=True)

    counts = weekly.groupby("method")["gw"].nunique()
    weekly = weekly.loc[
        weekly["method"].isin(counts[counts >= len(scored)].index)
    ].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_hybrid.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_hybrid.png")
    write_report(REPORTS / "stage_20_season_climb_hybrid.md", summary, weekly, scored)
    return {"summary": summary, "weekly": weekly, "scored_gws": scored}


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_20_season_climb_hybrid.md")
