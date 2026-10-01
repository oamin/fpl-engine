"""Stage 24 — xP v2 backtest + LightGBM non-linearity gate.

1. Point-level quality of the refactored equation (Spearman / MAE / bias /
   decile calibration / IC horizons).
2. Walk-forward LightGBM to capture residual non-linearities:
     - lgbm_direct  — points ~ features (incl. xP components)
     - lgbm_resid   — score = xP + LGBM(points − xP)   ← preferred
     - lgbm_nl      — score = Σ ŵ_c · xp_c  (learned component weights
                       via LGBM leaf predictions on component-only design)
3. Season climb (stripped XI) vs exp_points / xP on the aligned GW window.

Writes:
  data/processed/xp_v2_quality.csv
  data/processed/xp_v2_lgbm_climb.csv
  data/processed/xp_v2_lgbm_importance.csv
  data/plots/xp_v2_lgbm_gate.png
  reports/stage_24_xp_v2_lgbm.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.season_climb import build_scores, run_season, summarize
from src.models.xp_engine import HORIZONS, add_forward_means, ic_gate

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

MIN_TRAIN_ROWS = 600
MIN_HISTORY = 3

# Full feature set for LGBM (equation inputs + components + priors).
FEATURE_COLS = [
    "xmi",
    "p_play",
    "p60",
    "exp_points",
    "roll3_points",
    "exp_xG",
    "exp_xA",
    "share_xG",
    "share_xA",
    "lam_scored",
    "lam_assist",
    "lam_conceded",
    "p_cs_mkt",
    "p_not_lose",
    "attack_strength",
    "defend_threat",
    "exp_defcon_hit",
    "value",
    "xp",
    "xp_appear",
    "xp_goals",
    "xp_assists",
    "xp_cs",
    "xp_defcon",
    "xp_bps",
    "xp_deductions",
    "xp_gc_loss",
    "xp_card_loss",
    "e_total",
    "p_over",
    "p_under",
]

# Component-only design for learned non-linear reweighting.
COMPONENT_COLS = [
    "xp_appear",
    "xp_goals",
    "xp_assists",
    "xp_cs",
    "xp_defcon",
    "xp_bps",
    "xp_deductions",
    "xmi",
    "p60",
    "lam_scored",
    "lam_conceded",
    "p_cs_mkt",
    "share_xG",
    "share_xA",
    "exp_defcon_hit",
]


def _make_lgbm(**kwargs: Any) -> Any:
    import lightgbm as lgb

    params = dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )
    params.update(kwargs)
    return lgb.LGBMRegressor(**params)


def _design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    pos = pd.get_dummies(df["position"], prefix="pos")
    for p in ("pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"):
        if p not in pos.columns:
            pos[p] = 0
    return pd.concat(
        [x.reset_index(drop=True), pos[["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]].reset_index(drop=True)],
        axis=1,
    )


def _spearman_safe(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 40:
        return float("nan")
    return float(pd.Series(y[mask]).corr(pd.Series(pred[mask]), method="spearman"))


def point_level_quality(feat: pd.DataFrame) -> pd.DataFrame:
    """Pooled next-GW diagnostics for xP / exp / roll3 (and by position)."""
    rows: list[dict[str, Any]] = []
    base = feat.loc[feat["n_prior"] >= MIN_HISTORY].copy()
    predictors = [
        ("xp", "xP v2"),
        ("exp_points", "exp points"),
        ("roll3_points", "roll3 points"),
        ("xmi", "xMi"),
    ]
    universes: list[tuple[str, pd.DataFrame]] = [("ALL", base)]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        universes.append((pos, base.loc[base["position"] == pos]))

    for univ_name, univ in universes:
        if len(univ) < 40:
            continue
        y = univ["total_points"].to_numpy(float)
        for col, label in predictors:
            pred = univ[col].to_numpy(float)
            mask = np.isfinite(y) & np.isfinite(pred)
            if mask.sum() < 40:
                continue
            yy, pp = y[mask], pred[mask]
            err = pp - yy
            rows.append(
                {
                    "universe": univ_name,
                    "predictor": col,
                    "label": label,
                    "n": int(mask.sum()),
                    "spearman": float(pd.Series(pp).corr(pd.Series(yy), method="spearman")),
                    "pearson": float(np.corrcoef(pp, yy)[0, 1]) if np.std(pp) > 0 else float("nan"),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "bias": float(np.mean(err)),
                    "mean_pred": float(pp.mean()),
                    "mean_y": float(yy.mean()),
                }
            )
    return pd.DataFrame(rows)


def decile_calibration(feat: pd.DataFrame, score_col: str = "xp") -> pd.DataFrame:
    """Mean actual points by predicted-score decile (ALL, n_prior≥3)."""
    base = feat.loc[feat["n_prior"] >= MIN_HISTORY].copy()
    pred = pd.to_numeric(base[score_col], errors="coerce")
    y = pd.to_numeric(base["total_points"], errors="coerce")
    mask = pred.notna() & y.notna()
    tmp = pd.DataFrame({"pred": pred[mask], "y": y[mask]})
    try:
        tmp["decile"] = pd.qcut(tmp["pred"], 10, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.DataFrame()
    g = tmp.groupby("decile", as_index=False).agg(
        n=("y", "size"),
        mean_pred=("pred", "mean"),
        mean_y=("y", "mean"),
    )
    g["score"] = score_col
    return g


def walk_forward_lgbm(feat: pd.DataFrame) -> tuple[pd.DataFrame, list[int], pd.DataFrame]:
    """Walk-forward LGBM direct / residual / component-NL; return importance from last fit."""
    out = feat.sort_values(["gw", "player_id"]).copy()
    for col in ("score_lgbm_direct", "score_lgbm_resid", "score_lgbm_nl"):
        out[col] = np.nan

    gws = sorted(int(g) for g in out["gw"].unique())
    scored: list[int] = []
    last_imp: pd.DataFrame | None = None

    for gw in gws:
        train = out.loc[out["gw"] < gw]
        test_idx = out.index[out["gw"] == gw]
        if len(train) < MIN_TRAIN_ROWS or len(test_idx) == 0:
            continue

        x_tr = _design(train, FEATURE_COLS)
        x_te = _design(out.loc[test_idx], FEATURE_COLS).reindex(columns=x_tr.columns, fill_value=0)
        y = train["total_points"].to_numpy(float)
        xp_tr = train["xp"].to_numpy(float)
        xp_te = out.loc[test_idx, "xp"].to_numpy(float)

        # Direct: points ~ features
        m_dir = _make_lgbm()
        m_dir.fit(x_tr, y)
        out.loc[test_idx, "score_lgbm_direct"] = m_dir.predict(x_te)

        # Residual: xP + LGBM(points − xP) — non-linear correction around equation
        m_res = _make_lgbm()
        m_res.fit(x_tr, y - xp_tr)
        out.loc[test_idx, "score_lgbm_resid"] = xp_te + m_res.predict(x_te)

        # Component NL: LGBM on component/market design → absolute points
        x_tr_c = _design(train, COMPONENT_COLS)
        x_te_c = _design(out.loc[test_idx], COMPONENT_COLS).reindex(
            columns=x_tr_c.columns, fill_value=0
        )
        m_nl = _make_lgbm(num_leaves=23, n_estimators=180)
        m_nl.fit(x_tr_c, y)
        out.loc[test_idx, "score_lgbm_nl"] = m_nl.predict(x_te_c)

        last_imp = pd.DataFrame(
            {
                "feature": list(x_tr.columns),
                "gain_resid": m_res.feature_importances_,
                "gain_direct": m_dir.feature_importances_,
            }
        ).sort_values("gain_resid", ascending=False)
        scored.append(gw)

    return out, scored, last_imp if last_imp is not None else pd.DataFrame()


def plot_gate(
    weekly: pd.DataFrame,
    calib: pd.DataFrame,
    quality: pd.DataFrame,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    fig.suptitle("Stage 24 — xP v2 quality + LightGBM non-linearity", fontsize=12)

    # Climb
    ax = axes[0]
    colors = {
        "lgbm_resid": "crimson",
        "lgbm_direct": "darkorange",
        "lgbm_nl": "orchid",
        "xp": "steelblue",
        "exp_points": "gray",
        "roll3_points": "seagreen",
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
            lw=2.2 if method.startswith("lgbm") or method == "xp" else 1.3,
            color=color,
            label=method,
        )
    ax.set_title("Season climb (captain ×2)")
    ax.set_xlabel("GW")
    ax.set_ylabel("Cumulative XI pts")
    ax.legend(fontsize=7, loc="upper left")

    # Calibration
    ax = axes[1]
    if not calib.empty:
        ax.plot(calib["mean_pred"], calib["mean_y"], "o-", color="steelblue", label="xP v2")
        lo = min(calib["mean_pred"].min(), calib["mean_y"].min())
        hi = max(calib["mean_pred"].max(), calib["mean_y"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, label="ideal")
    ax.set_title("Decile calibration (xP)")
    ax.set_xlabel("Mean predicted")
    ax.set_ylabel("Mean actual pts")
    ax.legend(fontsize=7)

    # Spearman by pos
    ax = axes[2]
    q = quality.loc[quality["universe"].isin(["ALL", "GKP", "DEF", "MID", "FWD"])]
    width = 0.35
    universes = ["ALL", "GKP", "DEF", "MID", "FWD"]
    x = np.arange(len(universes))
    for i, (pred, color) in enumerate((("xp", "steelblue"), ("exp_points", "gray"))):
        vals = []
        for u in universes:
            hit = q.loc[(q["universe"] == u) & (q["predictor"] == pred)]
            vals.append(float(hit.iloc[0]["spearman"]) if not hit.empty else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=pred)
    ax.set_xticks(x)
    ax.set_xticklabels(universes)
    ax.set_ylabel("Spearman")
    ax.set_title("Point-level IC by position")
    ax.legend(fontsize=7)
    ax.axhline(0, color="gray", lw=0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    quality: pd.DataFrame,
    ic_rows: list[dict[str, Any]],
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    scored_gws: list[int],
    importance: pd.DataFrame,
    ml_point: pd.DataFrame,
) -> None:
    ic = pd.DataFrame(ic_rows)
    base = float(summary.loc[summary["method"] == "exp_points", "total_points"].iloc[0])
    xp_total = float(summary.loc[summary["method"] == "xp", "total_points"].iloc[0])

    lines = [
        "# Stage 24 — xP v2 backtest + LightGBM non-linearity",
        "",
        "Refactored equation (GKP goals=10, no `play_scale`, probabilistic appearance, "
        "BPS proxy, GC/YC deductions) evaluated on **point-level quality** and the "
        "**stripped season climb**. LightGBM learns residual non-linearities "
        "walk-forward (`gw < t` only).",
        "",
        f"- Aligned climb GWs: **{scored_gws[0]}–{scored_gws[-1]}** (n={len(scored_gws)})",
        f"- Train warm-up: ≥ **{MIN_TRAIN_ROWS}** rows",
        "",
        "## Equation (v2)",
        "",
        "```",
        "xP = appear + goals + assists + cs + defcon + bps − deductions",
        "appear = p_play·1 + p60·1",
        "goals/ast = share × λ × pts   (no play_scale)",
        "bps ≈ 0.18·xp_goals + 0.12·xp_assists + 0.08·xp_cs",
        "deductions = p60·λ_conc/2 (GKP/DEF) + (xMi/90)·0.15",
        "```",
        "",
        "## Point-level quality (pooled, n_prior≥3)",
        "",
        "| universe | predictor | n | Spearman | MAE | bias | mean_pred | mean_y |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in quality.itertuples():
        if r.universe not in ("ALL", "GKP", "DEF", "MID", "FWD"):
            continue
        if r.predictor not in ("xp", "exp_points", "roll3_points"):
            continue
        lines.append(
            f"| {r.universe} | {r.label} | {r.n} | {r.spearman:.3f} | "
            f"{r.mae:.2f} | {r.bias:+.2f} | {r.mean_pred:.2f} | {r.mean_y:.2f} |"
        )

    lines += [
        "",
        "## Horizon IC — regulars xMi≥45 (Spearman vs mean pts over next H)",
        "",
        "| predictor | H=1 | H=3 | H=5 | H=8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("xP engine", "exp points", "roll3 points", "xMi"):
        vals = []
        for H in HORIZONS:
            hit = ic.loc[
                (ic["universe"] == "regulars_xmi45")
                & (ic["label"] == label)
                & (ic["horizon"] == H)
            ]
            vals.append(f"{float(hit.iloc[0]['spearman']):.3f}" if not hit.empty else "—")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Walk-forward ML point metrics (aligned GWs)",
        "",
        "| score | Spearman | MAE | bias |",
        "|---|---:|---:|---:|",
    ]
    for r in ml_point.itertuples():
        lines.append(
            f"| {r.score} | {r.spearman:.3f} | {r.mae:.2f} | {r.bias:+.2f} |"
        )

    lines += [
        "",
        "## Season climb (captain ×2, aligned GWs)",
        "",
        "| method | total | mean/GW | vs exp | vs xP |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - base:+.0f} | {r.total_points - xp_total:+.0f} |"
        )

    if not importance.empty:
        lines += [
            "",
            "## LightGBM residual — top features (last GW fit, gain)",
            "",
            "| feature | gain_resid | gain_direct |",
            "|---|---:|---:|",
        ]
        for r in importance.head(12).itertuples():
            lines.append(
                f"| {r.feature} | {r.gain_resid:.0f} | {r.gain_direct:.0f} |"
            )

    # Verdicts
    xp_all = quality.loc[(quality.universe == "ALL") & (quality.predictor == "xp")]
    exp_all = quality.loc[(quality.universe == "ALL") & (quality.predictor == "exp_points")]
    q_verdict = "inconclusive"
    if not xp_all.empty and not exp_all.empty:
        ds = float(xp_all.iloc[0]["spearman"]) - float(exp_all.iloc[0]["spearman"])
        db = abs(float(xp_all.iloc[0]["bias"]))
        if ds >= 0.02 and db < 1.5:
            q_verdict = (
                f"PASS — xP Spearman {float(xp_all.iloc[0]['spearman']):.3f} "
                f"(Δ vs exp {ds:+.3f}), bias {float(xp_all.iloc[0]['bias']):+.2f}"
            )
        elif ds >= 0:
            q_verdict = (
                f"WEAK — xP edges exp on Spearman ({ds:+.3f}); "
                f"bias {float(xp_all.iloc[0]['bias']):+.2f}"
            )
        else:
            q_verdict = f"FAIL — xP trails exp Spearman by {ds:+.3f}"

    best_ml = None
    best_tot = -1e18
    for m in ("lgbm_resid", "lgbm_direct", "lgbm_nl"):
        hit = summary.loc[summary["method"] == m]
        if hit.empty:
            continue
        tot = float(hit.iloc[0]["total_points"])
        if tot > best_tot:
            best_tot = tot
            best_ml = m
    if best_ml is None:
        climb_verdict = "NO LGBM RESULTS"
    elif best_tot > xp_total + 10 and best_tot > base + 10:
        climb_verdict = (
            f"PASS — {best_ml} beats both "
            f"(vs xP {best_tot - xp_total:+.0f}, vs exp {best_tot - base:+.0f})"
        )
    elif best_tot > xp_total:
        climb_verdict = (
            f"PARTIAL — {best_ml} beats xP ({best_tot - xp_total:+.0f}) "
            f"vs exp {best_tot - base:+.0f}"
        )
    elif best_tot > base + 10:
        climb_verdict = (
            f"PARTIAL — {best_ml} beats exp ({best_tot - base:+.0f}) "
            f"but trails xP ({best_tot - xp_total:+.0f})"
        )
    else:
        climb_verdict = (
            f"FAIL — best {best_ml} trails xP by {best_tot - xp_total:+.0f}"
        )

    lines += [
        "",
        "## Gate verdicts",
        "",
        f"- **Quality:** {q_verdict}",
        f"- **Climb + LGBM:** {climb_verdict}",
        "",
        "Pass bar: xP Spearman ≥ exp on pooled appearances with |bias| modest; "
        "best LGBM finishes ≥ **+10** vs both exp and xP on the climb.",
        "",
        "## Plots / outputs",
        "",
        "- `data/plots/xp_v2_lgbm_gate.png`",
        "- `data/processed/xp_v2_quality.csv`",
        "- `data/processed/xp_v2_lgbm_climb.csv`",
        "- `data/processed/xp_v2_lgbm_importance.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _ml_point_metrics(feat: pd.DataFrame, scored_gws: list[int]) -> pd.DataFrame:
    sub = feat.loc[feat["gw"].isin(scored_gws)]
    rows = []
    for col, label in (
        ("xp", "xp"),
        ("exp_points", "exp_points"),
        ("score_lgbm_direct", "lgbm_direct"),
        ("score_lgbm_resid", "lgbm_resid"),
        ("score_lgbm_nl", "lgbm_nl"),
    ):
        y = sub["total_points"].to_numpy(float)
        p = sub[col].to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(p)
        if mask.sum() < 40:
            continue
        err = p[mask] - y[mask]
        rows.append(
            {
                "score": label,
                "spearman": _spearman_safe(y, p),
                "mae": float(np.mean(np.abs(err))),
                "bias": float(np.mean(err)),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    feat = build_scores()
    # Horizon IC needs forward means
    feat_ic = add_forward_means(feat)
    ic_rows = ic_gate(feat_ic)

    quality = point_level_quality(feat)
    calib = decile_calibration(feat, "xp")

    feat, scored_gws, importance = walk_forward_lgbm(feat)
    if not scored_gws:
        raise RuntimeError("No GWs scored — not enough history for LGBM warm-up")

    ml_point = _ml_point_metrics(feat, scored_gws)

    extra = {
        "lgbm_direct": "score_lgbm_direct",
        "lgbm_resid": "score_lgbm_resid",
        "lgbm_nl": "score_lgbm_nl",
    }
    weekly = run_season(feat, extra_score_cols=extra, gws=scored_gws)
    counts = weekly.groupby("method")["gw"].nunique()
    keep = counts[counts >= len(scored_gws)].index.tolist()
    weekly = weekly.loc[weekly["method"].isin(keep)].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    quality.to_csv(PROCESSED / "xp_v2_quality.csv", index=False)
    weekly.to_csv(PROCESSED / "xp_v2_lgbm_climb.csv", index=False)
    if not importance.empty:
        importance.to_csv(PROCESSED / "xp_v2_lgbm_importance.csv", index=False)
    if not calib.empty:
        calib.to_csv(PROCESSED / "xp_v2_calibration.csv", index=False)
    ml_point.to_csv(PROCESSED / "xp_v2_ml_point.csv", index=False)

    plot_gate(weekly, calib, quality, PLOTS / "xp_v2_lgbm_gate.png")
    write_report(
        REPORTS / "stage_24_xp_v2_lgbm.md",
        quality,
        ic_rows,
        summary,
        weekly,
        scored_gws,
        importance,
        ml_point,
    )
    return {
        "quality": quality,
        "summary": summary,
        "scored_gws": scored_gws,
        "ml_point": ml_point,
        "importance": importance,
    }


if __name__ == "__main__":
    out = run()
    print(f"GWs: {out['scored_gws'][0]}–{out['scored_gws'][-1]} (n={len(out['scored_gws'])})")
    print("\nPoint-level (ALL):")
    q = out["quality"]
    print(
        q.loc[
            (q.universe == "ALL") & q.predictor.isin(["xp", "exp_points", "roll3_points"]),
            ["label", "spearman", "mae", "bias", "mean_pred", "mean_y"],
        ].to_string(index=False)
    )
    print("\nWalk-forward ML point metrics:")
    print(out["ml_point"].to_string(index=False))
    print("\nClimb:")
    print(out["summary"].to_string(index=False))
    if not out["importance"].empty:
        print("\nTop residual features:")
        print(out["importance"].head(8).to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_24_xp_v2_lgbm.md")
