"""Stage 23 — Horizon uncertainty + FDR + transfer-flow prior.

Extends stage-22 fixture-aware V:

  sc = μ · n_fixtures · w_σ(h) · fdr(gw_h) · flow(t)

where:
  - σ0 = expanding std(total_points − score_xp), prior-only
  - σ_h = σ0 · √(1+β·h); w_σ = τ²/(τ²+σ_h²)
  - fdr from attack_strength / defend_threat at future GW
  - flow from decision-GW transfers_balance (crowd prior, ±15%)

Arms:
  - ridge_h3_ft: full stack (primary)
  - ridge_h3_nounc_ft: stage-22 fixture only
  - ridge_h3_fdr_ft: fixture + FDR
  - ridge_h3_flow_ft: fixture + flow
  - xp_ft: full stack on xP

Writes:
  data/processed/season_climb_stage23.csv
  data/plots/season_climb_stage23.png
  reports/stage_23_uncertainty_fdr.md
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
    walk_forward_global_ridge_h3,
)
from src.models.season_climb import bank_squad_gw, pick_xi, summarize
from src.models.season_climb_budget import run_budgeted_season
from src.models.season_climb_ft import (
    HIT_COST,
    HOLD_EPS,
    SWITCH_PENALTY,
    UNC_BETA,
    UNC_TAU,
    SquadState,
    _as_int_value,
    _fill_score,
    _gw_pool,
    advance_ft,
    choose_transfers,
    initial_squad,
    load_fixture_counts,
    load_vaastav_roster,
    precision_weight,
    sell_price,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

RIDGE_H3 = "score_ridge_h3"
XP = "score_xp"
HORIZON_S23 = 4
FLOW_KAPPA = 0.06
STAGE22_H3 = 1872.0


def add_xp_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-free expanding residual σ of (points − score_xp)."""
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    xp = pd.to_numeric(out.get("score_xp", out.get("xp")), errors="coerce")
    pts = pd.to_numeric(out["total_points"], errors="coerce")
    resid = pts - xp
    prior = resid.groupby(out["player_id"], sort=False).shift(1)
    sigma = prior.groupby(out["player_id"], sort=False).transform(
        lambda s: s.expanding(min_periods=3).std()
    )
    out["sigma_xp"] = pd.to_numeric(sigma, errors="coerce")
    pos_med = out.groupby("position")["sigma_xp"].transform("median")
    out["sigma_xp"] = out["sigma_xp"].fillna(pos_med).fillna(UNC_TAU)
    out["sigma_xp"] = out["sigma_xp"].clip(0.5, 8.0)
    return out


def add_fdr_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Position-aware fixture difficulty scale from market strengths."""
    out = df.copy()
    att = pd.to_numeric(out.get("attack_strength"), errors="coerce").fillna(0.5)
    threat = pd.to_numeric(out.get("defend_threat"), errors="coerce").fillna(0.5)
    pos = out["position"].astype(str)
    fdr = np.where(
        pos.isin(["GKP", "DEF"]),
        0.65 + 0.7 * (1.0 - threat),
        0.65 + 0.7 * att,
    )
    out["fdr_scale"] = np.clip(fdr, 0.55, 1.35)
    return out


def add_flow_scale(df: pd.DataFrame, kappa: float = FLOW_KAPPA) -> pd.DataFrame:
    """Within-GW z-scored transfer_balance → soft ±15% band (decision-time)."""
    out = df.copy()
    bal = pd.to_numeric(out.get("transfers_balance"), errors="coerce").fillna(0.0)
    flow_raw = np.sign(bal) * np.log1p(np.abs(bal))
    out["_flow_raw"] = flow_raw
    z = pd.Series(0.0, index=out.index, dtype=float)
    for _gw, idx in out.groupby("gw").groups.items():
        ix = list(idx)
        s = out.loc[ix, "_flow_raw"]
        mu, sd = float(s.mean()), float(s.std(ddof=0))
        if sd < 1e-9 or not np.isfinite(sd):
            z.loc[ix] = 0.0
        else:
            z.loc[ix] = (s - mu) / sd
    out["flow_scale"] = np.clip(1.0 + kappa * z, 0.85, 1.15)
    out = out.drop(columns=["_flow_raw"])
    return out


def build_fdr_map(feat: pd.DataFrame) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    sub = feat[["gw", "player_id", "fdr_scale"]].copy()
    sub["player_id"] = sub["player_id"].astype(str)
    for g, gdf in sub.groupby("gw"):
        out[int(g)] = {
            str(r.player_id): float(r.fdr_scale) for r in gdf.itertuples()
        }
    return out


def sigma_map_for_gw(feat: pd.DataFrame, gw: int) -> dict[str, float]:
    gdf = feat.loc[feat["gw"] == gw, ["player_id", "sigma_xp"]].drop_duplicates(
        "player_id"
    )
    return {
        str(r.player_id): float(r.sigma_xp) for r in gdf.itertuples()
    }


def flow_map_for_gw(feat: pd.DataFrame, gw: int) -> dict[str, float]:
    gdf = feat.loc[feat["gw"] == gw, ["player_id", "flow_scale"]].drop_duplicates(
        "player_id"
    )
    return {
        str(r.player_id): float(r.flow_scale) for r in gdf.itertuples()
    }


def run_ft_season_s23(
    feat: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    gws: list[int],
    roster: pd.DataFrame,
    fixture_counts: dict[int, dict[str, int]],
    fdr_by_gw: dict[int, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roster_by_gw: dict[int, set[str]] = {
        int(g): set(gdf["player_id"].astype(str))
        for g, gdf in roster.groupby("gw")
    }
    weight_rows: list[dict[str, Any]] = []

    for method, cfg in configs.items():
        init_col = cfg["init"]
        xfer_col = cfg["xfer"]
        xi_col = cfg["xi"]
        use_sigma = bool(cfg.get("use_sigma", False))
        use_fdr = bool(cfg.get("use_fdr", False))
        use_flow = bool(cfg.get("use_flow", False))
        horizon = int(cfg.get("horizon", HORIZON_S23))
        structural = bool(cfg.get("structural_2tx", True))

        state: SquadState | None = None
        for i, gw in enumerate(gws):
            owned = state.ids() if state else set()
            pool = _gw_pool(feat, roster, gw, owned)
            if pool["position"].nunique() < 4:
                continue

            sigma0 = sigma_map_for_gw(feat, int(gw)) if use_sigma else None
            fdr = fdr_by_gw if use_fdr else None
            flow = flow_map_for_gw(feat, int(gw)) if use_flow else None

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
                    fixture_counts=fixture_counts,
                    structural_2tx=structural,
                    sigma0=sigma0,
                    fdr_by_gw=fdr,
                    flow_by_pid=flow,
                )
                state = new_state

            # Diagnostics: mean weights for owned squad at this GW
            if use_sigma or use_fdr or use_flow:
                for h in range(horizon):
                    future = [g for g in gws if g >= gw]
                    if h >= len(future):
                        break
                    gh = int(future[h])
                    ws, fs, fls = [], [], []
                    for pid in state.ids():
                        if use_sigma and sigma0 is not None:
                            ws.append(
                                precision_weight(sigma0.get(pid, UNC_TAU), h)
                            )
                        if use_fdr:
                            fs.append(float(fdr_by_gw.get(gh, {}).get(pid, 1.0)))
                        if use_flow and flow is not None:
                            fls.append(float(flow.get(pid, 1.0)))
                    weight_rows.append(
                        {
                            "method": f"{method}_ft",
                            "gw": int(gw),
                            "h": h,
                            "mean_w_sigma": float(np.mean(ws)) if ws else float("nan"),
                            "mean_fdr": float(np.mean(fs)) if fs else float("nan"),
                            "mean_flow": float(np.mean(fls)) if fls else float("nan"),
                        }
                    )

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
                    "use_sigma": use_sigma,
                    "use_fdr": use_fdr,
                    "use_flow": use_flow,
                    "n_autosubs": int(banked["n_autosubs"]),
                    "n_blank_intended": int(banked["n_blank_intended"]),
                    "n_blank_final": int(banked["n_blank_final"]),
                }
            )
            if i == 0:
                state.ft = 1
            else:
                state.ft = advance_ft(ft_before, n_tx)

    weekly = pd.DataFrame(rows)
    weekly.attrs["weights"] = pd.DataFrame(weight_rows)
    return weekly


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    style = {
        "ridge_h3_ft": ("ridge_h3 full", "darkgreen", 2.8),
        "ridge_h3_fdr_ft": ("ridge_h3 FDR", "seagreen", 2.0),
        "ridge_h3_flow_ft": ("ridge_h3 flow", "olive", 1.8),
        "ridge_h3_nounc_ft": ("ridge_h3 stage22", "gray", 1.6),
        "xp_ft": ("xp full", "steelblue", 2.2),
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
    ax.set_title("Stage 23 — uncertainty + FDR + transfer-flow in V")
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    weights: pd.DataFrame,
    scored: list[int],
) -> None:
    def tot(name: str) -> float:
        hit = summary.loc[summary["method"] == name, "total_points"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    primary = tot("ridge_h3_ft")
    nounc = tot("ridge_h3_nounc_ft")
    fdr_only = tot("ridge_h3_fdr_ft")
    flow_only = tot("ridge_h3_flow_ft")
    xp = tot("xp_ft")
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
        "# Stage 23 — Horizon uncertainty + FDR + transfer-flow",
        "",
        "Transfer score at horizon step `h`:",
        "",
        "```text",
        "sc = μ · n_fixtures · w_σ(h) · fdr(gw_h) · flow(t)",
        "σ_h = σ0 · √(1+β·h);  w_σ = τ²/(τ²+σ_h²)",
        f"β={UNC_BETA}, τ={UNC_TAU}, flow κ={FLOW_KAPPA}, H={HORIZON_S23}",
        "```",
        "",
        "- **σ0:** expanding std of `(points − score_xp)`, prior-only",
        "- **fdr:** MID/FWD via `attack_strength`; GKP/DEF via `1−defend_threat`",
        "- **flow:** decision-GW `transfers_balance` within-GW z → clip ±15%",
        "",
        f"- GWs: **{scored[0]}–{scored[-1]}** (n={len(scored)})",
        f"- Stage-22 ridge_h3 baseline: **{STAGE22_H3:.0f}**",
        "",
        "## Final standings (captain ×2 − hits)",
        "",
        "| method | total | mean/GW | vs full | vs stage22 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        m = str(r.method)
        if not (m.endswith("_ft") or m == "xp_budget"):
            continue
        vs22 = (
            f"{r.total_points - STAGE22_H3:+.0f}"
            if "ridge_h3" in m
            else "—"
        )
        lines.append(
            f"| {m} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - primary:+.0f} | {vs22} |"
        )

    lines += [
        "",
        "## Transfer behaviour",
        "",
        f"- ridge_h3_ft (full): {tx('ridge_h3_ft')}",
        f"- ridge_h3_nounc_ft: {tx('ridge_h3_nounc_ft')}",
        f"- ridge_h3_fdr_ft: {tx('ridge_h3_fdr_ft')}",
        f"- ridge_h3_flow_ft: {tx('ridge_h3_flow_ft')}",
        f"- xp_ft: {tx('xp_ft')}",
        "",
        "## Ablations",
        "",
        f"- Full vs stage22 nounc: **{primary - nounc:+.0f}**",
        f"- FDR-only vs nounc: **{fdr_only - nounc:+.0f}**",
        f"- Flow-only vs nounc: **{flow_only - nounc:+.0f}**",
        f"- Full vs xp_ft: **{primary - xp:+.0f}**",
        f"- Best FT vs xp_budget ({xp_b:.0f}): "
        f"**{max(primary, nounc, fdr_only, flow_only, xp) - xp_b:+.0f}**",
        "",
    ]

    if not weights.empty:
        lines += [
            "## Mean weights by horizon (ridge_h3 full squad)",
            "",
            "| h | mean w_σ | mean fdr | mean flow |",
            "|---:|---:|---:|---:|",
        ]
        wfull = weights.loc[weights["method"] == "ridge_h3_ft"]
        for h, hdf in wfull.groupby("h"):
            lines.append(
                f"| {int(h)} | {hdf['mean_w_sigma'].mean():.3f} | "
                f"{hdf['mean_fdr'].mean():.3f} | {hdf['mean_flow'].mean():.3f} |"
            )
        lines.append("")

    if primary >= nounc and primary >= xp:
        verdict = (
            f"PASS — full stack leads "
            f"(vs nounc {primary - nounc:+.0f}, vs xp {primary - xp:+.0f})"
        )
    elif primary > nounc + 5:
        verdict = (
            f"PARTIAL — beats stage22 ({primary - nounc:+.0f}) "
            f"but vs xp {primary - xp:+.0f}"
        )
    elif max(fdr_only, flow_only) > nounc + 5:
        best_abl = "fdr" if fdr_only >= flow_only else "flow"
        verdict = (
            f"PARTIAL — component helps ({best_abl}) but full stack "
            f"{primary - nounc:+.0f} vs nounc"
        )
    else:
        verdict = (
            f"FAIL — full {primary:.0f} vs nounc {nounc:.0f} "
            f"(vs xp {primary - xp:+.0f})"
        )

    lines += [
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_stage23.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_stage23.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print("Building features + ridge_h3…")
    all_feat = build_fresh_seasons()
    # Ensure score_xp exists (build_one_season sets it)
    if "score_xp" not in all_feat.columns and "xp" in all_feat.columns:
        all_feat["score_xp"] = all_feat["xp"]
    all_feat = add_xp_uncertainty(all_feat)
    all_feat = add_fdr_scale(all_feat)
    all_feat = add_flow_scale(all_feat)

    pred_h3, scored = walk_forward_global_ridge_h3(all_feat)
    if not scored:
        raise RuntimeError("No scored GWs")

    eval_feat = all_feat.loc[all_feat["season"] == EVAL_SEASON].copy()
    eval_feat[RIDGE_H3] = pred_h3.loc[eval_feat.index]
    # Recompute flow z on eval season only for cleaner within-season scaling
    eval_feat = add_flow_scale(eval_feat)

    roster = load_vaastav_roster(EVAL_SEASON)
    fixture_counts = load_fixture_counts(EVAL_SEASON)
    fdr_by_gw = build_fdr_map(eval_feat)
    print(
        f"sigma_xp median={eval_feat['sigma_xp'].median():.2f}; "
        f"fdr mean={eval_feat['fdr_scale'].mean():.3f}; "
        f"flow mean={eval_feat['flow_scale'].mean():.3f}"
    )

    configs = {
        "ridge_h3": {
            "init": RIDGE_H3,
            "xfer": RIDGE_H3,
            "xi": RIDGE_H3,
            "use_sigma": True,
            "use_fdr": True,
            "use_flow": True,
            "horizon": HORIZON_S23,
            "structural_2tx": True,
        },
        "ridge_h3_nounc": {
            "init": RIDGE_H3,
            "xfer": RIDGE_H3,
            "xi": RIDGE_H3,
            "use_sigma": False,
            "use_fdr": False,
            "use_flow": False,
            "horizon": HORIZON_S23,
            "structural_2tx": True,
        },
        "ridge_h3_fdr": {
            "init": RIDGE_H3,
            "xfer": RIDGE_H3,
            "xi": RIDGE_H3,
            "use_sigma": False,
            "use_fdr": True,
            "use_flow": False,
            "horizon": HORIZON_S23,
            "structural_2tx": True,
        },
        "ridge_h3_flow": {
            "init": RIDGE_H3,
            "xfer": RIDGE_H3,
            "xi": RIDGE_H3,
            "use_sigma": False,
            "use_fdr": False,
            "use_flow": True,
            "horizon": HORIZON_S23,
            "structural_2tx": True,
        },
        "xp": {
            "init": XP,
            "xfer": XP,
            "xi": XP,
            "use_sigma": True,
            "use_fdr": True,
            "use_flow": True,
            "horizon": HORIZON_S23,
            "structural_2tx": True,
        },
    }

    print(f"Stage 23 FT climb GWs {scored[0]}–{scored[-1]}…")
    weekly_ft = run_ft_season_s23(
        eval_feat, configs, scored, roster, fixture_counts, fdr_by_gw
    )
    weights = weekly_ft.attrs.get("weights", pd.DataFrame())
    weekly_b = run_budgeted_season(eval_feat, {"xp": XP}, scored)
    weekly = pd.concat([weekly_ft, weekly_b], ignore_index=True)

    counts = weekly.groupby("method")["gw"].nunique()
    weekly = weekly.loc[
        weekly["method"].isin(counts[counts >= len(scored)].index)
    ].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_stage23.csv", index=False)
    if isinstance(weights, pd.DataFrame) and not weights.empty:
        weights.to_csv(PROCESSED / "season_climb_stage23_weights.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_stage23.png")
    write_report(
        REPORTS / "stage_23_uncertainty_fdr.md",
        summary,
        weekly,
        weights if isinstance(weights, pd.DataFrame) else pd.DataFrame(),
        scored,
    )
    return {
        "summary": summary,
        "weekly": weekly,
        "weights": weights,
        "scored_gws": scored,
    }


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_23_uncertainty_fdr.md")
