"""Stage 22 — Fixture-aware V + structural 2-transfers (ridge_h3 default).

Upgrades on stage 21:
  1. Horizon V scales by Vaastav fixture count (0=blank, 1=SGW, 2=DGW)
  2. H=4 fixture-aware rollout
  3. Simultaneous 2-transfers (structural cross-pos pairs, e.g. fund premium FWD)

Arms:
  - ridge_h3_ft: stage-21 μ + fixture V + structural 2tx (primary)
  - ridge_h3_nofixture_ft: same μ, H=4 blanks-only (no DGW×2) + structural
  - ridge_h3_1pos_ft: fixture V but same-pos beam only (no structural 2tx)
  - xp_ft: xP control with full stage-22 policy

Writes:
  data/processed/season_climb_stage22.csv
  data/plots/season_climb_stage22.png
  reports/stage_22_fixture_structural.md
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
)
from src.models.season_climb import bank_squad_gw, pick_xi, summarize
from src.models.season_climb_budget import run_budgeted_season
from src.models.season_climb_ft import (
    HIT_COST,
    HOLD_EPS,
    SWITCH_PENALTY,
    SquadState,
    _as_int_value,
    _fill_score,
    _gw_pool,
    advance_ft,
    choose_transfers,
    initial_squad,
    load_fixture_counts,
    load_vaastav_roster,
    sell_price,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

RIDGE_H3 = "score_ridge_h3"
XP = "score_xp"
HORIZON_S22 = 4
STAGE21_H3 = 1814.0


def run_ft_season_s22(
    feat: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    gws: list[int],
    roster: pd.DataFrame,
    fixture_counts: dict[int, dict[str, int]],
) -> pd.DataFrame:
    """configs[method] = init/xfer/xi cols + use_fixtures + structural flags."""
    rows: list[dict[str, Any]] = []
    roster_by_gw: dict[int, set[str]] = {
        int(g): set(gdf["player_id"].astype(str))
        for g, gdf in roster.groupby("gw")
    }

    for method, cfg in configs.items():
        init_col = cfg["init"]
        xfer_col = cfg["xfer"]
        xi_col = cfg["xi"]
        use_fix = bool(cfg.get("use_fixtures", True))
        structural = bool(cfg.get("structural_2tx", True))
        horizon = int(cfg.get("horizon", HORIZON_S22))
        fc = fixture_counts if use_fix else None

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
                    horizon=horizon,
                    fixture_counts=fc,
                    structural_2tx=structural,
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
                    "use_fixtures": use_fix,
                    "structural_2tx": structural,
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
        "ridge_h3_ft": ("ridge_h3 + fix+2tx", "darkgreen", 2.8),
        "ridge_h3_1pos_ft": ("ridge_h3 fixture only", "seagreen", 2.0),
        "ridge_h3_nofixture_ft": ("ridge_h3 struct no-DGW", "olive", 1.8),
        "xp_ft": ("xp + fix+2tx", "steelblue", 2.2),
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
    ax.set_title(f"Stage 22 — fixture-aware H={HORIZON_S22} + structural 2tx")
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

    primary = tot("ridge_h3_ft")
    xp = tot("xp_ft")
    nofix = tot("ridge_h3_nofixture_ft")
    onepos = tot("ridge_h3_1pos_ft")
    xp_b = tot("xp_budget")

    def tx(method: str) -> str:
        g = weekly.loc[weekly["method"] == method]
        if g.empty:
            return "n/a"
        n2 = int((g["n_transfers"] >= 2).sum())
        return (
            f"tx={g['n_transfers'].mean():.2f}, "
            f"holds={(g['n_transfers'] == 0).sum()}/{len(g)}, "
            f"≥2tx={n2}, hits={g['hits'].sum():.0f}"
        )

    lines = [
        "# Stage 22 — Fixture-aware horizon + structural 2-transfers",
        "",
        "On top of stage-21 **ridge_h3** + switch penalty:",
        "",
        f"1. **Fixture-aware V:** scale frozen score by Vaastav fixture count "
        f"(0 blank / 1 SGW / 2 DGW); horizon **H={HORIZON_S22}**",
        "2. **Structural 2-transfers:** simultaneous sells/buys preserving "
        "2/5/5/3 (fund premium via cross-pos pairs)",
        f"3. Hold unless ΔV ≥ {HOLD_EPS}; switch penalty {SWITCH_PENALTY}/tx",
        "",
        f"- GWs: **{scored[0]}–{scored[-1]}** (n={len(scored)})",
        f"- Stage-21 ridge_h3 baseline: **{STAGE21_H3:.0f}**",
        "",
        "## Final standings (captain ×2 − hits)",
        "",
        "| method | total | mean/GW | vs ridge_h3_ft | vs stage21 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        m = str(r.method)
        if not (m.endswith("_ft") or m == "xp_budget"):
            continue
        vs21 = f"{r.total_points - STAGE21_H3:+.0f}" if "ridge_h3" in m else "—"
        lines.append(
            f"| {m} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - primary:+.0f} | {vs21} |"
        )

    lines += [
        "",
        "## Transfer behaviour",
        "",
        f"- ridge_h3_ft: {tx('ridge_h3_ft')}",
        f"- ridge_h3_1pos_ft: {tx('ridge_h3_1pos_ft')}",
        f"- ridge_h3_nofixture_ft: {tx('ridge_h3_nofixture_ft')}",
        f"- xp_ft: {tx('xp_ft')}",
        "",
        "## Ablations",
        "",
        f"- Fixture+DGW vs blanks-only: **{primary - nofix:+.0f}**",
        f"- Structural 2tx vs same-pos beam: **{primary - onepos:+.0f}**",
        f"- ridge_h3_ft vs xp_ft: **{primary - xp:+.0f}**",
        f"- Best FT vs xp_budget ({xp_b:.0f}): "
        f"**{max(primary, xp, nofix, onepos) - xp_b:+.0f}**",
        "",
    ]

    if primary >= xp and primary > STAGE21_H3 + 10:
        verdict = (
            f"PASS — ridge_h3_ft leads and beats stage21 "
            f"(vs xp {primary - xp:+.0f}, vs s21 {primary - STAGE21_H3:+.0f})"
        )
    elif primary >= xp:
        verdict = (
            f"PARTIAL — leads xp ({primary - xp:+.0f}) "
            f"but vs stage21 {primary - STAGE21_H3:+.0f}"
        )
    elif primary > STAGE21_H3:
        verdict = (
            f"PARTIAL — beats stage21 ({primary - STAGE21_H3:+.0f}) "
            f"but trails xp ({primary - xp:+.0f})"
        )
    else:
        verdict = (
            f"FAIL — ridge_h3_ft {primary:.0f} vs xp {xp:.0f} "
            f"(vs s21 {primary - STAGE21_H3:+.0f})"
        )

    lines += [
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_stage22.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_stage22.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print("Building features + ridge_h3…")
    all_feat = build_fresh_seasons()
    pred_h3, scored = walk_forward_global_ridge_h3(all_feat)
    if not scored:
        raise RuntimeError("No scored GWs")

    eval_feat = all_feat.loc[all_feat["season"] == EVAL_SEASON].copy()
    eval_feat[RIDGE_H3] = pred_h3.loc[eval_feat.index]

    roster = load_vaastav_roster(EVAL_SEASON)
    fixture_counts = load_fixture_counts(EVAL_SEASON)
    print(
        f"Fixture map: {len(fixture_counts)} GWs, "
        f"DGW rows="
        f"{sum(1 for g in fixture_counts.values() for n in g.values() if n >= 2)}"
    )

    configs = {
        "ridge_h3": {
            "init": RIDGE_H3,
            "xfer": RIDGE_H3,
            "xi": RIDGE_H3,
            "use_fixtures": True,
            "structural_2tx": True,
            "horizon": HORIZON_S22,
        },
        "ridge_h3_1pos": {
            "init": RIDGE_H3,
            "xfer": RIDGE_H3,
            "xi": RIDGE_H3,
            "use_fixtures": True,
            "structural_2tx": False,
            "horizon": HORIZON_S22,
        },
        "ridge_h3_nofixture": {
            "init": RIDGE_H3,
            "xfer": RIDGE_H3,
            "xi": RIDGE_H3,
            "use_fixtures": False,
            "structural_2tx": True,
            "horizon": HORIZON_S22,
        },
        "xp": {
            "init": XP,
            "xfer": XP,
            "xi": XP,
            "use_fixtures": True,
            "structural_2tx": True,
            "horizon": HORIZON_S22,
        },
    }

    print(f"Stage 22 FT climb GWs {scored[0]}–{scored[-1]} (H={HORIZON_S22})…")
    weekly_ft = run_ft_season_s22(eval_feat, configs, scored, roster, fixture_counts)
    weekly_b = run_budgeted_season(eval_feat, {"xp": XP}, scored)
    weekly = pd.concat([weekly_ft, weekly_b], ignore_index=True)

    counts = weekly.groupby("method")["gw"].nunique()
    weekly = weekly.loc[
        weekly["method"].isin(counts[counts >= len(scored)].index)
    ].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_stage22.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_stage22.png")
    write_report(REPORTS / "stage_22_fixture_structural.md", summary, weekly, scored)
    return {"summary": summary, "weekly": weekly, "scored_gws": scored}


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_22_fixture_structural.md")
