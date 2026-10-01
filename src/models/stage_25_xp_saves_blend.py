"""Stage 25 — GKP saves patch + horizon xp/exp blend on FT climb.

1. ``xp_saves`` in ``compute_xp`` (already in xp_engine; re-evaluated here).
2. Point-level GKP bias check vs stage-24 baseline.
3. FT climb ablations: pure xP vs ``w_h·xp + (1−w_h)·exp`` with
   ``w_h = max(floor, γ^h)`` at H=5.

Writes:
  data/processed/stage_25_quality.csv
  data/processed/stage_25_ft_climb.csv
  data/plots/stage_25_xp_saves_blend.png
  reports/stage_25_xp_saves_blend.md
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
from src.models.xp_engine import BLEND_FLOOR, BLEND_GAMMA, horizon_blend_weight

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

FT_HORIZON = 5
ABLATIONS: list[dict[str, Any]] = [
    {"name": "xp", "blend_gamma": None, "blend_floor": 0.5, "suffix": "_ft"},
    {
        "name": "xp_blend85",
        "blend_gamma": 0.85,
        "blend_floor": 0.5,
        "suffix": "_ft",
    },
    {"name": "exp_points", "blend_gamma": None, "blend_floor": 0.5, "suffix": "_ft"},
]


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
        for col, label in (
            ("xp", "xP v2+saves"),
            ("exp_points", "exp points"),
            ("xp_saves", "xp_saves only"),
        ):
            if col not in chunk.columns:
                continue
            p = chunk[col].to_numpy(float)
            mask = np.isfinite(y) & np.isfinite(p)
            if mask.sum() < 40 and col != "xp_saves":
                continue
            if mask.sum() < 20:
                continue
            yy, pp = y[mask], p[mask]
            err = pp - yy
            rows.append(
                {
                    "universe": univ,
                    "predictor": col,
                    "label": label,
                    "n": int(mask.sum()),
                    "spearman": float(
                        pd.Series(pp).corr(pd.Series(yy), method="spearman")
                    ),
                    "mae": float(np.mean(np.abs(err))),
                    "bias": float(np.mean(err)),
                    "mean_pred": float(pp.mean()),
                    "mean_y": float(yy.mean()),
                    "mean_saves_term": float(chunk.loc[mask, "xp_saves"].mean())
                    if "xp_saves" in chunk.columns
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_gate(
    weekly: pd.DataFrame, quality: pd.DataFrame, out: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    fig.suptitle("Stage 25 — GKP saves + horizon xp/exp blend", fontsize=12)

    ax = axes[0]
    colors = {
        "xp_blend85_ft": "crimson",
        "xp_blend80_ft": "darkorange",
        "xp_ft": "steelblue",
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
            lw=2.2 if "blend" in method or method == "xp_ft" else 1.3,
            color=color,
            label=method,
        )
    ax.set_title(f"FT climb (H={FT_HORIZON}, autosubs+VC)")
    ax.set_xlabel("GW")
    ax.set_ylabel("Cumulative pts")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    q = quality.loc[
        quality["universe"].isin(["ALL", "GKP", "DEF", "MID", "FWD"])
        & quality["predictor"].isin(["xp", "exp_points"])
    ]
    universes = ["ALL", "GKP", "DEF", "MID", "FWD"]
    x = np.arange(len(universes))
    width = 0.35
    for i, (pred, color) in enumerate((("xp", "steelblue"), ("exp_points", "gray"))):
        vals = []
        for u in universes:
            hit = q.loc[(q["universe"] == u) & (q["predictor"] == pred)]
            vals.append(float(hit.iloc[0]["bias"]) if not hit.empty else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=pred)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(universes)
    ax.set_ylabel("bias (pred − actual)")
    ax.set_title("Level bias by position")
    ax.legend(fontsize=7)

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
) -> None:
    xp_ft = float(summary.loc[summary["method"] == "xp_ft", "total_points"].iloc[0])
    exp_ft = float(
        summary.loc[summary["method"] == "exp_points_ft", "total_points"].iloc[0]
    )
    gkp = quality.loc[(quality.universe == "GKP") & (quality.predictor == "xp")]
    gkp_bias = float(gkp.iloc[0]["bias"]) if not gkp.empty else float("nan")
    gkp_sp = float(gkp.iloc[0]["mean_saves_term"]) if not gkp.empty else float("nan")

    weights = ", ".join(
        f"h={h}: {horizon_blend_weight(h):.2f}" for h in range(FT_HORIZON)
    )

    lines = [
        "# Stage 25 — GKP saves + horizon xp/exp blend",
        "",
        "Keeps linear **xP v2** as the core score. Adds deterministic GKP "
        "`xp_saves` and optional multi-GW blend "
        "`score_h = w_h·xp + (1−w_h)·exp_points` with "
        f"`w_h = max({BLEND_FLOOR}, {BLEND_GAMMA}^h)` on the FT transfer V "
        f"(lookahead H={FT_HORIZON}). XI/captain still use pure decision-GW score.",
        "",
        f"- Eval season: **{EVAL_SEASON}**",
        f"- FT GWs: **{gws[0]}–{gws[-1]}** (n={len(gws)})",
        f"- Default blend weights: {weights}",
        "",
        "## GKP saves term",
        "",
        "```",
        "λ_conceded = clip(e_total − λ_scored, 0, 5)",
        "E[saves]   = λ_conceded · 2.0     # empirical saves/GC",
        "xp_saves   = p60 · E[saves]/3    # GKP only",
        "```",
        "",
        f"- GKP mean `xp_saves`: **{gkp_sp:.2f}**",
        f"- GKP level bias (pred−actual): **{gkp_bias:+.2f}** (stage-24 was −0.72)",
        "",
        "## Point-level quality",
        "",
        "| universe | predictor | n | Spearman | MAE | bias |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in quality.itertuples():
        if r.predictor not in ("xp", "exp_points"):
            continue
        if r.universe not in ("ALL", "GKP", "DEF", "MID", "FWD"):
            continue
        lines.append(
            f"| {r.universe} | {r.label} | {r.n} | {r.spearman:.3f} | "
            f"{r.mae:.2f} | {r.bias:+.2f} |"
        )

    lines += [
        "",
        "## Stripped climb (captain ×2, no FT)",
        "",
        "| method | total | mean/GW |",
        "|---|---:|---:|",
    ]
    for r in stripped.itertuples():
        lines.append(f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} |")

    lines += [
        "",
        f"## FT climb (H={FT_HORIZON}, autosubs + VC − hits)",
        "",
        "| method | total | mean/GW | vs xp_ft | vs exp_ft |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - xp_ft:+.0f} | {r.total_points - exp_ft:+.0f} |"
        )

    blend_rows = summary.loc[summary["method"].str.contains("blend", na=False)]
    best_blend = None
    best_tot = -1e18
    for r in blend_rows.itertuples():
        if r.total_points > best_tot:
            best_tot = float(r.total_points)
            best_blend = str(r.method)

    if np.isfinite(gkp_bias) and abs(gkp_bias) < 0.35:
        saves_verdict = f"PASS — GKP bias {gkp_bias:+.2f} (was −0.72)"
    elif np.isfinite(gkp_bias) and gkp_bias > -0.72:
        saves_verdict = f"PARTIAL — GKP bias improved to {gkp_bias:+.2f}"
    else:
        saves_verdict = f"FAIL — GKP bias {gkp_bias:+.2f}"

    if best_blend is None:
        blend_verdict = "NO BLEND RESULTS"
    elif best_tot > xp_ft + 10:
        blend_verdict = (
            f"PASS — {best_blend} beats xp_ft by {best_tot - xp_ft:+.0f}"
        )
    elif best_tot >= xp_ft:
        blend_verdict = (
            f"WEAK — {best_blend} edges xp_ft by {best_tot - xp_ft:+.0f}"
        )
    else:
        blend_verdict = (
            f"FAIL — best blend trails xp_ft by {best_tot - xp_ft:+.0f}"
        )

    lines += [
        "",
        "## Gate verdicts",
        "",
        f"- **Saves:** {saves_verdict}",
        f"- **Horizon blend:** {blend_verdict}",
        "",
        "## Plots / outputs",
        "",
        "- `data/plots/stage_25_xp_saves_blend.png`",
        "- `data/processed/stage_25_quality.csv`",
        "- `data/processed/stage_25_ft_climb.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print(f"Building {EVAL_SEASON} features (xP v2 + saves)…", flush=True)
    feat = build_one_season(EVAL_SEASON, _fd_code(EVAL_SEASON))
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0

    quality = point_quality(feat)
    print("\nPoint-level bias:", flush=True)
    print(
        quality.loc[
            quality.predictor.isin(["xp", "exp_points"]),
            ["universe", "label", "spearman", "mae", "bias"],
        ].to_string(index=False),
        flush=True,
    )

    # Stripped climb on same season
    print("Stripped climb…", flush=True)
    weekly_s = run_season(
        feat,
        extra_score_cols={"xp": "score_xp", "exp_points": "score_exp_points"},
    )
    # keep core methods only
    weekly_s = weekly_s.loc[
        weekly_s["method"].isin(["xp", "exp_points", "blend_xp_exp", "roll3_points"])
    ]
    stripped = summarize(weekly_s)
    print(stripped.to_string(index=False), flush=True)

    roster = load_vaastav_roster(EVAL_SEASON)
    gws = sorted(int(g) for g in feat["gw"].unique())
    # skip early GWs with thin priors (match FT warm feel)
    gws = [g for g in gws if g >= 4]
    print(f"\nFT climbs on GWs {gws[0]}–{gws[-1]} (H={FT_HORIZON})…", flush=True)

    frames: list[pd.DataFrame] = []
    for ab in ABLATIONS:
        col = "score_exp_points" if ab["name"] == "exp_points" else "score_xp"
        print(f"  {ab['name']} γ={ab['blend_gamma']} …", flush=True)
        frames.append(
            run_ft_season(
                feat,
                {ab["name"]: col},
                gws,
                roster=roster,
                horizon=FT_HORIZON,
                exp_score_col="score_exp_points" if ab["blend_gamma"] else None,
                blend_gamma=ab["blend_gamma"],
                blend_floor=float(ab["blend_floor"]),
                method_suffix=ab["suffix"],
            )
        )
        print(f"    done {ab['name']}", flush=True)
    weekly = pd.concat(frames, ignore_index=True)
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    quality.to_csv(PROCESSED / "stage_25_quality.csv", index=False)
    weekly.to_csv(PROCESSED / "stage_25_ft_climb.csv", index=False)
    plot_gate(weekly, quality, PLOTS / "stage_25_xp_saves_blend.png")
    write_report(
        REPORTS / "stage_25_xp_saves_blend.md",
        quality,
        summary,
        weekly,
        gws,
        stripped,
    )
    return {
        "quality": quality,
        "summary": summary,
        "stripped": stripped,
        "gws": gws,
    }


if __name__ == "__main__":
    out = run()
    print("\nStripped:")
    print(out["stripped"].to_string(index=False))
    print("\nFT climb:")
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_25_xp_saves_blend.md")
