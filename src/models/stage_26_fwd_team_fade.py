"""Stage 26 — FWD goal calibration + light team-prior horizon fade.

1. Leakage-free FWD goal scale inside ``compute_xp`` (position expanding
   mean(goals)/mean(e_goals), clipped).
2. Optional multi-GW V blend toward ``xp_team_prior`` (expanding team λ/CS),
   with ``w(h)=1`` for h≤3 then linear fade to 0.60 at h=5.
3. Gate vs Stage-25 pure xP FT baseline (1987): FWD bias, climb pts, transfers.

Writes:
  data/processed/stage_26_quality.csv
  data/processed/stage_26_ft_climb.csv
  data/plots/stage_26_fwd_team_fade.png
  reports/stage_26_fwd_team_fade.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.ridge_multiseason import EVAL_SEASON, SEASONS, build_one_season
from src.models.season_climb import run_season, summarize
from src.models.season_climb_ft import load_vaastav_roster, run_ft_season
from src.models.xp_engine import TEAM_FADE_END_W, TEAM_FADE_HOLD_H, team_prior_fade_weight

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

FT_HORIZON = 5
BASELINE_XP_FT = 1987.0  # stage-25 pure xP before FWD cal
MAX_TRANSFER_INFLATION = 0.05


def _fd_code(season: str) -> str:
    for s, code in SEASONS:
        if s == season:
            return code
    raise KeyError(season)


def point_quality(feat: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for univ, chunk in (
        ("ALL", feat),
        ("GKP", feat.loc[feat["position"] == "GKP"]),
        ("DEF", feat.loc[feat["position"] == "DEF"]),
        ("MID", feat.loc[feat["position"] == "MID"]),
        ("FWD", feat.loc[feat["position"] == "FWD"]),
    ):
        if len(chunk) < 40:
            continue
        y = chunk["total_points"].to_numpy(float)
        for col, label in (("xp", "xP v2+FWD cal"), ("exp_points", "exp points")):
            p = chunk[col].to_numpy(float)
            mask = np.isfinite(y) & np.isfinite(p)
            if mask.sum() < 40:
                continue
            err = p[mask] - y[mask]
            rows.append(
                {
                    "universe": univ,
                    "predictor": col,
                    "label": label,
                    "n": int(mask.sum()),
                    "spearman": float(
                        pd.Series(p[mask]).corr(pd.Series(y[mask]), method="spearman")
                    ),
                    "mae": float(np.mean(np.abs(err))),
                    "bias": float(np.mean(err)),
                    "mean_fwd_scale": float(
                        chunk.loc[mask, "fwd_goal_scale"].mean()
                    )
                    if "fwd_goal_scale" in chunk.columns
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_gate(
    weekly: pd.DataFrame, quality: pd.DataFrame, summary: pd.DataFrame, out: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    fig.suptitle("Stage 26 — FWD cal + team-prior fade", fontsize=12)

    ax = axes[0]
    colors = {
        "xp_ft": "steelblue",
        "xp_team_fade_ft": "crimson",
        "exp_points_ft": "gray",
    }
    for method, color in colors.items():
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        ax.plot(
            g["gw"],
            g["xi_points_cap"].cumsum(),
            "-o",
            ms=2.5,
            lw=2.2 if method != "exp_points_ft" else 1.3,
            color=color,
            label=method,
        )
    ax.axhline(BASELINE_XP_FT, color="steelblue", ls="--", lw=1, alpha=0.5)
    ax.set_title(f"FT climb (H={FT_HORIZON})")
    ax.set_xlabel("GW")
    ax.set_ylabel("Cumulative pts")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    q = quality.loc[
        quality["universe"].isin(["ALL", "GKP", "DEF", "MID", "FWD"])
        & (quality["predictor"] == "xp")
    ]
    universes = ["ALL", "GKP", "DEF", "MID", "FWD"]
    vals = [
        float(q.loc[q["universe"] == u, "bias"].iloc[0]) if (q["universe"] == u).any() else np.nan
        for u in universes
    ]
    ax.bar(universes, vals, color="steelblue")
    ax.axhline(0, color="gray", lw=0.6)
    ax.axhline(-0.10, color="orange", ls="--", lw=0.8)
    ax.axhline(0.10, color="orange", ls="--", lw=0.8)
    ax.set_ylabel("bias (pred − actual)")
    ax.set_title("xP bias by position (FWD target ±0.10)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    quality: pd.DataFrame,
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    gws: list[int],
    stripped: pd.DataFrame,
    tx_stats: pd.DataFrame,
) -> None:
    xp_ft = float(summary.loc[summary["method"] == "xp_ft", "total_points"].iloc[0])
    fade = summary.loc[summary["method"] == "xp_team_fade_ft"]
    fade_tot = float(fade.iloc[0]["total_points"]) if not fade.empty else float("nan")
    exp_ft = float(
        summary.loc[summary["method"] == "exp_points_ft", "total_points"].iloc[0]
    )
    fwd = quality.loc[(quality.universe == "FWD") & (quality.predictor == "xp")]
    fwd_bias = float(fwd.iloc[0]["bias"]) if not fwd.empty else float("nan")
    fwd_scale = float(fwd.iloc[0]["mean_fwd_scale"]) if not fwd.empty else float("nan")

    weights = ", ".join(
        f"h={h}: {team_prior_fade_weight(h):.2f}" for h in range(FT_HORIZON)
    )

    xp_tx = tx_stats.loc[tx_stats["method"] == "xp_ft"]
    fade_tx = tx_stats.loc[tx_stats["method"] == "xp_team_fade_ft"]
    base_tx = float(xp_tx.iloc[0]["n_transfers"]) if not xp_tx.empty else float("nan")
    fade_n_tx = float(fade_tx.iloc[0]["n_transfers"]) if not fade_tx.empty else float("nan")
    tx_ratio = fade_n_tx / base_tx if base_tx and base_tx > 0 else float("nan")

    lines = [
        "# Stage 26 — FWD goal calibration + team-prior horizon fade",
        "",
        "Keeps linear **xP v2** as the core score. Adds:",
        "1. Leakage-free **FWD goal scale** = expanding `mean(goals)/mean(e_goals)` "
        "(clipped), applied to `xp_goals` before BPS.",
        "2. Optional FT-V blend toward **team-strength prior** (expanding team λ/CS), "
        f"with `w(h)=1` for h≤{TEAM_FADE_HOLD_H}, fading to {TEAM_FADE_END_W:.2f} at h=5.",
        "",
        f"- Eval: **{EVAL_SEASON}**, FT GWs **{gws[0]}–{gws[-1]}** (n={len(gws)})",
        f"- Team-fade weights: {weights}",
        f"- Stage-25 pure xP FT baseline: **{BASELINE_XP_FT:.0f}**",
        "",
        "## FWD calibration",
        "",
        f"- Mean FWD `fwd_goal_scale`: **{fwd_scale:.3f}**",
        f"- FWD bias: **{fwd_bias:+.2f}** (stage-25 was −0.38; target ∈ [−0.10, +0.10])",
        "",
        "## Point-level quality",
        "",
        "| universe | predictor | n | Spearman | MAE | bias |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in quality.itertuples():
        lines.append(
            f"| {r.universe} | {r.label} | {r.n} | {r.spearman:.3f} | "
            f"{r.mae:.2f} | {r.bias:+.2f} |"
        )

    lines += [
        "",
        "## Stripped climb",
        "",
        "| method | total | mean/GW |",
        "|---|---:|---:|",
    ]
    for r in stripped.itertuples():
        lines.append(f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} |")

    lines += [
        "",
        f"## FT climb (H={FT_HORIZON})",
        "",
        "| method | total | mean/GW | vs xp_ft | vs stage25 baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - xp_ft:+.0f} | {r.total_points - BASELINE_XP_FT:+.0f} |"
        )

    lines += [
        "",
        "## Transfer stability",
        "",
        "| method | total transfers | mean/GW | hits |",
        "|---|---:|---:|---:|",
    ]
    for r in tx_stats.itertuples():
        lines.append(
            f"| {r.method} | {r.n_transfers:.0f} | {r.mean_tx:.2f} | {r.hits:.0f} |"
        )
    if np.isfinite(tx_ratio):
        lines.append(
            f"\n- Team-fade / pure-xP transfer ratio: **{tx_ratio:.3f}** "
            f"(cap ≤ {1 + MAX_TRANSFER_INFLATION:.2f})"
        )

    # Verdicts
    if abs(fwd_bias) <= 0.10:
        fwd_v = f"PASS — FWD bias {fwd_bias:+.2f}"
    elif fwd_bias > -0.38:
        fwd_v = f"PARTIAL — FWD bias {fwd_bias:+.2f} (improved from −0.38)"
    else:
        fwd_v = f"FAIL — FWD bias {fwd_bias:+.2f}"

    best_tot = max(xp_ft, fade_tot if np.isfinite(fade_tot) else -1e9)
    if best_tot >= BASELINE_XP_FT + 10:
        climb_v = (
            f"PASS — best FT {best_tot:.0f} beats stage-25 baseline by "
            f"{best_tot - BASELINE_XP_FT:+.0f}"
        )
    elif best_tot >= BASELINE_XP_FT:
        climb_v = (
            f"WEAK — best FT {best_tot:.0f} edges baseline by "
            f"{best_tot - BASELINE_XP_FT:+.0f}"
        )
    else:
        climb_v = (
            f"FAIL — best FT {best_tot:.0f} trails baseline by "
            f"{best_tot - BASELINE_XP_FT:+.0f}"
        )

    if np.isfinite(tx_ratio) and tx_ratio <= 1 + MAX_TRANSFER_INFLATION:
        tx_v = f"PASS — transfer inflation {tx_ratio - 1:+.1%}"
    elif np.isfinite(tx_ratio):
        tx_v = f"FAIL — transfer inflation {tx_ratio - 1:+.1%}"
    else:
        tx_v = "n/a"

    lines += [
        "",
        "## Gate verdicts",
        "",
        f"- **FWD bias:** {fwd_v}",
        f"- **FT climb:** {climb_v}",
        f"- **Transfers:** {tx_v}",
        "",
        f"vs exp_ft: xp {xp_ft - exp_ft:+.0f}, team_fade "
        f"{(fade_tot - exp_ft) if np.isfinite(fade_tot) else float('nan'):+.0f}",
        "",
        "## Outputs",
        "",
        "- `data/plots/stage_26_fwd_team_fade.png`",
        "- `data/processed/stage_26_quality.csv`",
        "- `data/processed/stage_26_ft_climb.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print(f"Building {EVAL_SEASON} (FWD cal + team prior)…", flush=True)
    feat = build_one_season(EVAL_SEASON, _fd_code(EVAL_SEASON))
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0

    quality = point_quality(feat)
    print(quality.to_string(index=False), flush=True)

    print("Stripped climb…", flush=True)
    weekly_s = run_season(
        feat,
        extra_score_cols={
            "xp": "score_xp",
            "exp_points": "score_exp_points",
            "team_prior": "score_team_prior",
        },
    )
    weekly_s = weekly_s.loc[
        weekly_s["method"].isin(["xp", "exp_points", "team_prior", "blend_xp_exp"])
    ]
    stripped = summarize(weekly_s)
    print(stripped.to_string(index=False), flush=True)

    roster = load_vaastav_roster(EVAL_SEASON)
    gws = [g for g in sorted(int(x) for x in feat["gw"].unique()) if g >= 4]
    print(f"FT climbs GWs {gws[0]}–{gws[-1]} H={FT_HORIZON}…", flush=True)

    frames: list[pd.DataFrame] = []
    print("  xp_ft…", flush=True)
    frames.append(
        run_ft_season(
            feat, {"xp": "score_xp"}, gws, roster=roster, horizon=FT_HORIZON
        )
    )
    print("  xp_team_fade_ft…", flush=True)
    frames.append(
        run_ft_season(
            feat,
            {"xp_team_fade": "score_xp"},
            gws,
            roster=roster,
            horizon=FT_HORIZON,
            exp_score_col="score_team_prior",
            blend_schedule="team_fade",
            method_suffix="_ft",
        )
    )
    print("  exp_points_ft…", flush=True)
    frames.append(
        run_ft_season(
            feat,
            {"exp_points": "score_exp_points"},
            gws,
            roster=roster,
            horizon=FT_HORIZON,
        )
    )
    weekly = pd.concat(frames, ignore_index=True)
    summary = summarize(weekly)

    tx_stats = (
        weekly.groupby("method", as_index=False)
        .agg(n_transfers=("n_transfers", "sum"), hits=("hits", "sum"), n_gw=("gw", "nunique"))
    )
    tx_stats["mean_tx"] = tx_stats["n_transfers"] / tx_stats["n_gw"]

    print(summary.to_string(index=False), flush=True)
    print(tx_stats.to_string(index=False), flush=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    quality.to_csv(PROCESSED / "stage_26_quality.csv", index=False)
    weekly.to_csv(PROCESSED / "stage_26_ft_climb.csv", index=False)
    tx_stats.to_csv(PROCESSED / "stage_26_transfers.csv", index=False)
    plot_gate(weekly, quality, summary, PLOTS / "stage_26_fwd_team_fade.png")
    write_report(
        REPORTS / "stage_26_fwd_team_fade.md",
        quality,
        summary,
        weekly,
        gws,
        stripped,
        tx_stats,
    )
    return {
        "quality": quality,
        "summary": summary,
        "stripped": stripped,
        "tx_stats": tx_stats,
        "gws": gws,
    }


if __name__ == "__main__":
    out = run()
    print(f"\nWrote {REPORTS}/stage_26_fwd_team_fade.md")
