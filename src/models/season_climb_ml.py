"""Stage 15 — Walk-forward ML scorers vs xP / exp_points on season climb.

Train only on GWs < t, predict score for GW t, then same stripped XI climb.

Models:
  ml_ridge  — StandardScaler + Ridge
  ml_hgb    — HistGradientBoostingRegressor (sklearn)
  ml_lgbm   — LightGBM regressor (if importable)
  ml_resid  — HGB on (points − xP), score = xP + residual_hat

Writes:
  data/processed/season_climb_ml.csv
  data/plots/season_climb_ml.png
  reports/stage_15_season_climb_ml.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.season_climb import build_scores, run_season, summarize

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

MIN_TRAIN_ROWS = 600
FEATURE_COLS = [
    "xmi",
    "exp_points",
    "roll3_points",
    "exp_xG",
    "exp_xA",
    "share_xG",
    "share_xA",
    "lam_scored",
    "lam_assist",
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
    "e_total",
    "p_over",
    "p_under",
]


def _design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURE_COLS].copy()
    for c in FEATURE_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    # Position one-hot
    pos = pd.get_dummies(df["position"], prefix="pos")
    for p in ("pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"):
        if p not in pos.columns:
            pos[p] = 0
    x = pd.concat([x.reset_index(drop=True), pos[["pos_GKP", "pos_DEF", "pos_MID", "pos_FWD"]].reset_index(drop=True)], axis=1)
    return x


def _make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]
    )


def _make_hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=120,
        learning_rate=0.06,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=42,
    )


def _make_lgbm() -> Any | None:
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    return lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )


def walk_forward_ml(feat: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Fill score_ml_* columns via walk-forward; return feat + GWs with ML scores."""
    out = feat.sort_values(["gw", "player_id"]).copy()
    for col in ("score_ml_ridge", "score_ml_hgb", "score_ml_lgbm", "score_ml_resid"):
        out[col] = np.nan

    gws = sorted(out["gw"].unique())
    scored_gws: list[int] = []
    lgbm_proto = _make_lgbm()

    for gw in gws:
        train = out.loc[out["gw"] < gw]
        test_idx = out.index[out["gw"] == gw]
        if len(train) < MIN_TRAIN_ROWS or len(test_idx) == 0:
            continue

        x_train = _design_matrix(train)
        y_train = train["total_points"].to_numpy(float)
        x_test = _design_matrix(out.loc[test_idx])
        # Align columns
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

        ridge = _make_ridge()
        ridge.fit(x_train, y_train)
        out.loc[test_idx, "score_ml_ridge"] = ridge.predict(x_test)

        hgb = _make_hgb()
        hgb.fit(x_train, y_train)
        out.loc[test_idx, "score_ml_hgb"] = hgb.predict(x_test)

        if lgbm_proto is not None:
            lgbm = _make_lgbm()
            assert lgbm is not None
            lgbm.fit(x_train, y_train)
            out.loc[test_idx, "score_ml_lgbm"] = lgbm.predict(x_test)

        # Residual model: learn points − xP, then xP + hat
        y_resid = y_train - train["xp"].to_numpy(float)
        hgb_r = _make_hgb()
        hgb_r.fit(x_train, y_resid)
        out.loc[test_idx, "score_ml_resid"] = (
            out.loc[test_idx, "xp"].to_numpy(float) + hgb_r.predict(x_test)
        )

        scored_gws.append(int(gw))

    return out, scored_gws


def plot_climb_ml(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    focus = [
        "ml_hgb",
        "ml_lgbm",
        "ml_ridge",
        "ml_resid",
        "xp",
        "exp_points",
        "blend_xp_exp",
        "price",
    ]
    colors = {
        "ml_hgb": "crimson",
        "ml_lgbm": "darkred",
        "ml_ridge": "tomato",
        "ml_resid": "orchid",
        "xp": "steelblue",
        "exp_points": "gray",
        "blend_xp_exp": "darkorange",
        "price": "seagreen",
    }
    for method in focus:
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        cum = g["xi_points_cap"].cumsum()
        thick = method.startswith("ml_") or method == "xp"
        ax.plot(
            g["gw"],
            cum,
            "-o",
            ms=3,
            lw=2.2 if thick else 1.3,
            color=colors.get(method, "black"),
            label=method,
        )
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("Cumulative XI points (captain ×2)")
    ax.set_title("Season climb — walk-forward ML vs xP / exp_points")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    scored_gws: list[int],
) -> None:
    base = float(summary.loc[summary["method"] == "exp_points", "total_points"].iloc[0])
    xp_total = float(summary.loc[summary["method"] == "xp", "total_points"].iloc[0])
    lines = [
        "# Stage 15 — Walk-forward ML vs xP / exp_points (season climb)",
        "",
        "Same stripped XI climb as stage 14. ML scores are **walk-forward**: "
        "train on all player-GWs with `gw < t`, predict GW `t` points, never see the future.",
        "",
        f"- ML warm-up until ≥ **{MIN_TRAIN_ROWS}** training rows",
        f"- Evaluated GWs: **{scored_gws[0]}–{scored_gws[-1]}** (n={len(scored_gws)}) "
        "— baselines restricted to the same window",
        f"- Features: `{', '.join(FEATURE_COLS[:8])}, …` + position one-hot (+ xP components)",
        "",
        "## Models",
        "",
        "| name | model |",
        "|---|---|",
        "| `ml_ridge` | StandardScaler + Ridge(α=5) |",
        "| `ml_hgb` | sklearn HistGradientBoosting |",
        "| `ml_lgbm` | LightGBM regressor |",
        "| `ml_resid` | HGB on (points−xP), score = xP + residual̂ |",
        "",
        "## Final standings (captain ×2, aligned GWs)",
        "",
        "| method | total | mean/GW | vs exp_points | vs xP |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {r.method} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - base:+.0f} | {r.total_points - xp_total:+.0f} |"
        )

    exp = weekly.loc[weekly["method"] == "exp_points"].set_index("gw")["xi_points_cap"]
    lines += [
        "",
        "## vs exp_points by GW",
        "",
        "| method | GWs ahead | GWs behind | mean Δ/GW |",
        "|---|---:|---:|---:|",
    ]
    for method in summary["method"].tolist():
        if method == "exp_points":
            continue
        m = weekly.loc[weekly["method"] == method].set_index("gw")["xi_points_cap"]
        both = pd.concat([m, exp], axis=1, keys=["m", "e"]).dropna()
        if both.empty:
            continue
        d = both["m"] - both["e"]
        lines.append(
            f"| {method} | {int((d > 0).sum())} | {int((d < 0).sum())} | {d.mean():+.2f} |"
        )

    ml_methods = [m for m in summary["method"] if str(m).startswith("ml_")]
    best_ml = None
    best_ml_total = -1e18
    for m in ml_methods:
        tot = float(summary.loc[summary["method"] == m, "total_points"].iloc[0])
        if tot > best_ml_total:
            best_ml_total = tot
            best_ml = m

    if best_ml is None:
        verdict = "NO ML RESULTS"
    elif best_ml_total > xp_total + 10 and best_ml_total > base + 10:
        verdict = (
            f"PASS — {best_ml} beats both "
            f"(vs xP {best_ml_total - xp_total:+.0f}, vs exp {best_ml_total - base:+.0f})"
        )
    elif best_ml_total > base + 10:
        verdict = (
            f"PARTIAL — {best_ml} beats exp_points ({best_ml_total - base:+.0f}) "
            f"but not xP ({best_ml_total - xp_total:+.0f})"
        )
    elif best_ml_total >= xp_total:
        verdict = f"WEAK — {best_ml} ties/edges xP ({best_ml_total - xp_total:+.0f})"
    else:
        verdict = (
            f"FAIL — best ML {best_ml} trails xP by {best_ml_total - xp_total:+.0f} "
            f"and exp by {best_ml_total - base:+.0f}"
        )

    lines += [
        "",
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "Pass bar: best ML finishes ≥ **+10** vs both `exp_points` and `xp` "
        "on the aligned GW window.",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_ml.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_ml.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    feat = build_scores()
    feat, scored_gws = walk_forward_ml(feat)
    if not scored_gws:
        raise RuntimeError("No GWs scored by walk-forward ML — not enough history")

    extra = {
        "ml_ridge": "score_ml_ridge",
        "ml_hgb": "score_ml_hgb",
        "ml_resid": "score_ml_resid",
    }
    if feat["score_ml_lgbm"].notna().any():
        extra["ml_lgbm"] = "score_ml_lgbm"

    weekly = run_season(feat, extra_score_cols=extra, gws=scored_gws)
    # Keep only methods present every scored GW for fair totals
    counts = weekly.groupby("method")["gw"].nunique()
    keep = counts[counts >= len(scored_gws)].index.tolist()
    weekly = weekly.loc[weekly["method"].isin(keep)].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_ml.csv", index=False)
    plot_climb_ml(weekly, PLOTS / "season_climb_ml.png")
    write_report(REPORTS / "stage_15_season_climb_ml.md", summary, weekly, scored_gws)
    return {"weekly": weekly, "summary": summary, "scored_gws": scored_gws}


if __name__ == "__main__":
    out = run()
    print(f"GWs evaluated: {out['scored_gws'][0]}–{out['scored_gws'][-1]} (n={len(out['scored_gws'])})")
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_15_season_climb_ml.md")
