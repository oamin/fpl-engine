"""Stage 5 — assemble player μ from baseline + market terms that passed stage 3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

MIN_MINUTES = 60.0
MIN_PRIOR = 3


def _multi_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask].astype(float)
    X = X[mask].astype(float)
    if y.size < X.shape[1] + 5:
        return np.full(X.shape[1], np.nan), float("nan"), np.full(0, np.nan)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return coef, r2, pred


def load_go_features(eval_path: Path) -> dict[str, list[str]]:
    ev = pd.read_csv(eval_path)
    go: dict[str, list[str]] = {p: [] for p in ("GKP", "DEF", "MID", "FWD")}
    for _, r in ev.iterrows():
        if bool(r["go"]):
            go[str(r["position"])].append(str(r["feature"]))
    # Dedupe preserve order
    for p in go:
        seen: set[str] = set()
        uniq = []
        for f in go[p]:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        go[p] = uniq
    return go


def assemble_mu(df: pd.DataFrame, go_features: dict[str, list[str]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit residual ~ market features (+ prior share) per position; μ = baseline + pred_resid."""
    out = df.copy()
    out["mu_points"] = np.nan
    out["pred_residual"] = np.nan
    rows: list[dict[str, Any]] = []

    sub0 = out.loc[
        (out["minutes"] >= MIN_MINUTES) & (out["n_prior_apps"] >= MIN_PRIOR)
    ].copy()

    for pos in ("GKP", "DEF", "MID", "FWD"):
        feats = go_features.get(pos) or []
        # Always allow prior minutes share as involvement proxy.
        feat_cols = list(feats)
        if "prior_share_minutes" in out.columns and "prior_share_minutes" not in feat_cols:
            feat_cols.append("prior_share_minutes")
        if "prior_share_xg" in out.columns and pos in ("MID", "FWD"):
            if "prior_share_xg" not in feat_cols:
                feat_cols.append("prior_share_xg")

        chunk = sub0.loc[sub0["position"] == pos].copy()
        if len(chunk) < 40:
            rows.append({"position": pos, "n": len(chunk), "error": "too few"})
            continue

        y = chunk["residual_points"].to_numpy(float)
        # Baseline-only null model for residual (predict 0) vs market model.
        X_cols = []
        for c in feat_cols:
            if c not in chunk.columns:
                continue
            X_cols.append(c)
        if not X_cols:
            # No market go — μ = baseline only
            out.loc[chunk.index, "pred_residual"] = 0.0
            out.loc[chunk.index, "mu_points"] = chunk["baseline_points"]
            rows.append(
                {
                    "position": pos,
                    "n": int(len(chunk)),
                    "features": [],
                    "r2_resid": 0.0,
                    "delta_vs_zero": 0.0,
                    "corr_mu_points": float(
                        chunk["baseline_points"].corr(chunk["total_points"])
                    ),
                    "top_bottom_lift_mu": float("nan"),
                }
            )
            continue

        X = np.column_stack(
            [np.ones(len(chunk))]
            + [pd.to_numeric(chunk[c], errors="coerce").to_numpy(float) for c in X_cols]
        )
        coef, r2, pred = _multi_ols(y, X)
        # Align pred back — _multi_ols drops non-finite rows
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        pred_full = np.full(len(chunk), np.nan)
        pred_full[mask] = pred

        out.loc[chunk.index, "pred_residual"] = pred_full
        out.loc[chunk.index, "mu_points"] = chunk["baseline_points"].to_numpy(float) + pred_full

        # Lift: top vs bottom decile of μ on realised points
        eval_df = chunk.copy()
        eval_df["mu"] = chunk["baseline_points"].to_numpy(float) + pred_full
        eval_df = eval_df.dropna(subset=["mu"])
        lift = float("nan")
        if len(eval_df) >= 50:
            try:
                eval_df["bin"] = pd.qcut(eval_df["mu"], 10, duplicates="drop")
                g = eval_df.groupby("bin", observed=True)["total_points"].mean()
                lift = float(g.iloc[-1] - g.iloc[0])
            except ValueError:
                pass

        rows.append(
            {
                "position": pos,
                "n": int(mask.sum()),
                "features": X_cols,
                "r2_resid": float(r2),
                "coefs": {c: float(coef[i + 1]) for i, c in enumerate(X_cols)}
                if np.all(np.isfinite(coef))
                else {},
                "corr_mu_points": float(eval_df["mu"].corr(eval_df["total_points"]))
                if len(eval_df)
                else float("nan"),
                "corr_base_points": float(
                    eval_df["baseline_points"].corr(eval_df["total_points"])
                )
                if len(eval_df)
                else float("nan"),
                "top_bottom_lift_mu": lift,
            }
        )
    return out, rows


def plot_mu(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Stage 5 — μ vs realised points (by position)", fontsize=13)
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        sub = df.loc[
            (df["position"] == pos)
            & (df["minutes"] >= MIN_MINUTES)
            & (df["n_prior_apps"] >= MIN_PRIOR)
            & df["mu_points"].notna()
        ]
        ax.scatter(sub["mu_points"], sub["total_points"], s=12, alpha=0.3, edgecolors="none")
        if len(sub) > 5:
            lims = [
                min(sub["mu_points"].min(), sub["total_points"].min()),
                max(sub["mu_points"].max(), sub["total_points"].max()),
            ]
            ax.plot(lims, lims, color="gray", ls="--", lw=1)
        corr = (
            float(sub["mu_points"].corr(sub["total_points"])) if len(sub) > 5 else float("nan")
        )
        ax.set_title(f"{pos}  corr(μ,pts)={corr:.3f}  n={len(sub)}")
        ax.set_xlabel("μ points")
        ax.set_ylabel("realised points")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_stage5_report(path: Path, rows: list[dict[str, Any]], go_features: dict[str, list[str]]) -> None:
    lines = [
        "# Stage 5 — Assemble player μ",
        "",
        "μ = baseline_points + predicted residual from GO market features + prior shares.",
        "",
        "## GO features used (from stage 3)",
        "",
    ]
    for pos, feats in go_features.items():
        lines.append(f"- {pos}: {', '.join(feats) if feats else '(baseline only)'}")
    lines += [
        "",
        "| pos | n | features | R²(resid) | corr(μ,pts) | corr(base,pts) | top−bot μ lift |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        feats = ",".join(r.get("features") or []) or "—"
        lines.append(
            f"| {r['position']} | {r.get('n', 0)} | {feats} | "
            f"{r.get('r2_resid', float('nan')):.3f} | "
            f"{r.get('corr_mu_points', float('nan')):.3f} | "
            f"{r.get('corr_base_points', float('nan')):.3f} | "
            f"{r.get('top_bottom_lift_mu', float('nan')):.2f} |"
        )
    lines += [
        "",
        "Plot: `data/plots/stage_5_player_mu.png`",
        "",
        "Output: `data/processed/player_mu.csv`",
        "",
        "## Success read",
        "",
        "- corr(μ, pts) ≥ corr(base, pts) and top−bot lift > 0 → market terms help.",
        "- If R²(resid) ≈ 0, μ collapses to baseline (still useful for ranking by level).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_player_mu(
    baseline_path: Path | None = None,
    eval_path: Path | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(baseline_path or (PROCESSED / "player_baseline.csv"))
    go = load_go_features(eval_path or (PROCESSED / "team_market_eval.csv"))
    out, rows = assemble_mu(df, go)
    out.to_csv(PROCESSED / "player_mu.csv", index=False)
    plot_mu(out, PLOTS / "stage_5_player_mu.png")
    write_stage5_report(REPORTS / "stage_5_player_mu.md", rows, go)
    return {"go_features": go, "by_position": rows}
