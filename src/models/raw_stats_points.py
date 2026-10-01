"""Player-level: raw stats ↔ FPL points (same-match + predictive).

Finds which observable stats tightly track points, and which prior
rates predict next-match points.

Writes:
  data/plots/raw_stats_same_match.png
  data/plots/raw_stats_predictive.png
  data/plots/raw_stats_decomp.png
  reports/raw_player_stats_points.md
  data/processed/raw_stats_eval.csv
"""

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

# Same-match candidate features (known after the match).
SAME_MATCH = [
    "minutes",
    "goals",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "xG",
    "xA",
    "defcon",
    "yellow_cards",
    "red_cards",
    "starts",
]

# Pre-match / lagged rates we can use for prediction.
PRIOR_CHANNELS = [
    "total_points",
    "minutes",
    "goals",
    "assists",
    "clean_sheets",
    "xG",
    "xA",
    "bps",
    "bonus",
    "defcon",
    "saves",
]


def _ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask].astype(float)
    x = x[mask].astype(float)
    n = int(y.size)
    if n < 20:
        return {"n": n, "slope": float("nan"), "r2": float("nan"), "corr": float("nan")}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    return {"n": n, "slope": float(coef[1]), "r2": r2, "corr": corr}


def _multi_r2(y: np.ndarray, X: np.ndarray) -> float:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[mask].astype(float)
    X = X[mask].astype(float)
    if y.size < X.shape[1] + 5:
        return float("nan")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def load_players() -> pd.DataFrame:
    path = PROCESSED / "player_matches.csv"
    df = pd.read_csv(path)
    for c in SAME_MATCH + ["total_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def same_match_univariate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("ALL", df)] + [(p, df.loc[df["position"] == p]) for p in ("GKP", "DEF", "MID", "FWD")]
    for scope, sub in scopes:
        y = sub["total_points"].to_numpy(float)
        for feat in SAME_MATCH:
            if feat not in sub.columns:
                continue
            # Skip near-constant features in a scope (e.g. saves for FWD).
            x = sub[feat].to_numpy(float)
            if np.nanstd(x) < 1e-9:
                continue
            fit = _ols(y, x)
            rows.append({"scope": scope, "feature": feat, "kind": "same_match", **fit})
    return pd.DataFrame(rows)


def same_match_multivariate(df: pd.DataFrame) -> list[dict[str, Any]]:
    """How much of points is explained by scoring components together."""
    # Rulebook-ish components available in our CSV.
    recipes = {
        "ALL": ["minutes", "goals", "assists", "clean_sheets", "bonus", "bps", "yellow_cards", "red_cards"],
        "GKP": ["minutes", "goals", "assists", "clean_sheets", "saves", "bonus", "bps", "goals_conceded", "yellow_cards"],
        "DEF": ["minutes", "goals", "assists", "clean_sheets", "bonus", "bps", "goals_conceded", "defcon", "yellow_cards"],
        "MID": ["minutes", "goals", "assists", "clean_sheets", "bonus", "bps", "defcon", "yellow_cards"],
        "FWD": ["minutes", "goals", "assists", "bonus", "bps", "yellow_cards"],
    }
    rows: list[dict[str, Any]] = []
    for scope, feats in recipes.items():
        sub = df if scope == "ALL" else df.loc[df["position"] == scope]
        y = sub["total_points"].to_numpy(float)
        X = np.column_stack([np.ones(len(sub))] + [sub[f].to_numpy(float) for f in feats if f in sub.columns])
        used = [f for f in feats if f in sub.columns]
        r2 = _multi_r2(y, X)
        rows.append({"scope": scope, "features": ",".join(used), "r2": r2, "n": int(len(sub))})
    # Minimal attack recipe
    for scope, sub in [("MID", df.loc[df.position == "MID"]), ("FWD", df.loc[df.position == "FWD"])]:
        y = sub["total_points"].to_numpy(float)
        X = np.column_stack(
            [np.ones(len(sub)), sub["goals"].to_numpy(float), sub["assists"].to_numpy(float), sub["bonus"].to_numpy(float)]
        )
        rows.append(
            {
                "scope": scope,
                "features": "goals,assists,bonus",
                "r2": _multi_r2(y, X),
                "n": int(len(sub)),
            }
        )
    return rows


def add_priors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "date", "fixture_id"], kind="mergesort").copy()
    for col in PRIOR_CHANNELS:
        if col not in out.columns:
            continue
        out[f"prior_{col}"] = out.groupby("player_id")[col].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        pos_mean = out.groupby("position")[col].transform("mean")
        out[f"prior_{col}"] = out[f"prior_{col}"].fillna(pos_mean)
    out["n_prior_apps"] = out.groupby("player_id").cumcount()
    # Convenience: prior xGI
    out["prior_xGI"] = out["prior_xG"] + out["prior_xA"]
    out["xGI"] = out["xG"] + out["xA"]
    return out


def predictive_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """Prior expanding rates → next-match total_points (≥60′, ≥3 prior apps)."""
    sub = df.loc[
        (df["minutes"] >= MIN_MINUTES) & (df["n_prior_apps"] >= MIN_PRIOR)
    ].copy()
    rows: list[dict[str, Any]] = []
    prior_feats = [f"prior_{c}" for c in PRIOR_CHANNELS] + ["prior_xGI"]
    scopes = [("ALL", sub)] + [(p, sub.loc[sub["position"] == p]) for p in ("GKP", "DEF", "MID", "FWD")]
    for scope, chunk in scopes:
        y = chunk["total_points"].to_numpy(float)
        for feat in prior_feats:
            if feat not in chunk.columns:
                continue
            x = chunk[feat].to_numpy(float)
            if np.nanstd(x) < 1e-9:
                continue
            fit = _ols(y, x)
            rows.append({"scope": scope, "feature": feat, "kind": "predictive", **fit})
    return pd.DataFrame(rows)


def predictive_multivariate(df: pd.DataFrame) -> list[dict[str, Any]]:
    sub = df.loc[
        (df["minutes"] >= MIN_MINUTES) & (df["n_prior_apps"] >= MIN_PRIOR)
    ].copy()
    recipes = {
        "ALL": ["prior_total_points", "prior_minutes", "prior_xGI"],
        "GKP": ["prior_total_points", "prior_minutes", "prior_saves", "prior_clean_sheets"],
        "DEF": ["prior_total_points", "prior_minutes", "prior_clean_sheets", "prior_defcon", "prior_xGI"],
        "MID": ["prior_total_points", "prior_minutes", "prior_xGI", "prior_goals", "prior_assists"],
        "FWD": ["prior_total_points", "prior_minutes", "prior_xGI", "prior_goals"],
    }
    rows: list[dict[str, Any]] = []
    for scope, feats in recipes.items():
        chunk = sub if scope == "ALL" else sub.loc[sub["position"] == scope]
        y = chunk["total_points"].to_numpy(float)
        used = [f for f in feats if f in chunk.columns]
        X = np.column_stack([np.ones(len(chunk))] + [chunk[f].to_numpy(float) for f in used])
        # Baseline-only for comparison
        Xb = np.column_stack([np.ones(len(chunk)), chunk["prior_total_points"].to_numpy(float)])
        r2 = _multi_r2(y, X)
        r2_base = _multi_r2(y, Xb)
        rows.append(
            {
                "scope": scope,
                "features": ",".join(used),
                "r2": r2,
                "r2_baseline_only": r2_base,
                "delta_r2": (r2 - r2_base) if np.isfinite(r2) and np.isfinite(r2_base) else float("nan"),
                "n": int(len(chunk)),
            }
        )
    return rows


def plot_same_match(df: pd.DataFrame, uni: pd.DataFrame, out: Path) -> None:
    """Best same-match features by position (highest R²)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Same-match: raw stat → FPL points (tightest links)", fontsize=13)
    picks = {
        "GKP": ["saves", "clean_sheets", "bps"],
        "DEF": ["bps", "clean_sheets", "goals"],
        "MID": ["bps", "goals", "assists"],
        "FWD": ["bps", "goals", "xG"],
    }
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        sub = df.loc[df["position"] == pos]
        # Choose single best non-bps if bps dominates — show bps + best event
        pos_uni = uni.loc[(uni["scope"] == pos) & (uni["kind"] == "same_match")].sort_values(
            "r2", ascending=False
        )
        # Prefer an event feature for the main panel: top among goals/assists/cs/saves/xg
        event_prefs = ["goals", "assists", "clean_sheets", "saves", "xG", "bonus", "bps"]
        feat = None
        for f in event_prefs:
            if f in pos_uni["feature"].values:
                # take highest r2 among these that exists — actually take best overall for clarity
                pass
        feat = str(pos_uni.iloc[0]["feature"]) if len(pos_uni) else picks[pos][0]
        # For display: if bps wins, also annotate; plot bps as it's the honest strongest
        x = sub[feat].to_numpy(float)
        y = sub["total_points"].to_numpy(float)
        fit = _ols(y, x)
        ax.scatter(x, y, s=12, alpha=0.25, edgecolors="none", color="steelblue")
        if np.isfinite(fit["slope"]):
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
            # intercept from means
            intercept = float(np.nanmean(y) - fit["slope"] * np.nanmean(x))
            ax.plot(xs, intercept + fit["slope"] * xs, color="crimson", lw=2)
        ax.set_title(f"{pos}: {feat}  R²={fit['r2']:.3f}  corr={fit['corr']:.3f}")
        ax.set_xlabel(feat)
        ax.set_ylabel("total_points")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_decomp(multi_same: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    # Primary recipe rows only (first per scope in multi_same that has many features)
    primary = []
    seen = set()
    for r in multi_same:
        if r["scope"] in seen:
            continue
        if r["scope"] in ("ALL", "GKP", "DEF", "MID", "FWD") and "goals,assists,bonus" not in r["features"]:
            primary.append(r)
            seen.add(r["scope"])
    labels = [r["scope"] for r in primary]
    vals = [r["r2"] for r in primary]
    colors = ["#4c78a8" if v >= 0.7 else "#f58518" if v >= 0.4 else "#e45756" for v in vals]
    ax.bar(labels, vals, color=colors, edgecolor="none")
    ax.axhline(0.7, color="gray", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² (points ~ scoring components)")
    ax.set_title("Same-match decomposition: how deterministic are FPL points?")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_predictive(df: pd.DataFrame, uni_pred: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(
        "Predictive: prior expanding rate → next-match points (≥60′, ≥3 apps)",
        fontsize=13,
    )
    # Best predictive feature per position (exclude prior_total_points for variety in one panel;
    # actually show prior_total_points AND best other)
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        chunk = df.loc[
            (df["position"] == pos)
            & (df["minutes"] >= MIN_MINUTES)
            & (df["n_prior_apps"] >= MIN_PRIOR)
        ]
        pos_uni = uni_pred.loc[uni_pred["scope"] == pos].sort_values("r2", ascending=False)
        feat = str(pos_uni.iloc[0]["feature"]) if len(pos_uni) else "prior_total_points"
        x = chunk[feat].to_numpy(float)
        y = chunk["total_points"].to_numpy(float)
        fit = _ols(y, x)
        ax.scatter(x, y, s=12, alpha=0.25, edgecolors="none", color="steelblue")
        if np.isfinite(fit["slope"]) and np.nanstd(x) > 0:
            xs = np.linspace(np.nanmin(x), np.nanmax(x), 40)
            intercept = float(np.nanmean(y) - fit["slope"] * np.nanmean(x))
            ax.plot(xs, intercept + fit["slope"] * xs, color="crimson", lw=2)
        # Annotate top 3 features
        top3 = pos_uni.head(3)
        ann = "\n".join(f"{r.feature}: R²={r.r2:.3f}" for r in top3.itertuples())
        ax.text(
            0.98,
            0.02,
            ann,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        )
        ax.set_title(f"{pos}: best={feat}  R²={fit['r2']:.3f}")
        ax.set_xlabel(feat)
        ax.set_ylabel("next total_points")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    uni_same: pd.DataFrame,
    multi_same: list[dict[str, Any]],
    uni_pred: pd.DataFrame,
    multi_pred: list[dict[str, Any]],
) -> None:
    lines = [
        "# Raw player stats ↔ FPL points",
        "",
        "Two questions:",
        "1. **Same-match** — given what happened, how tightly do stats map to points?",
        "2. **Predictive** — do prior rates forecast *next* match points?",
        "",
        "## Same-match multivariate (decomposition)",
        "",
        "| scope | features | n | R² |",
        "|---|---|---:|---:|",
    ]
    for r in multi_same:
        lines.append(
            f"| {r['scope']} | `{r['features']}` | {r['n']} | {r['r2']:.3f} |"
        )

    lines += [
        "",
        "## Same-match univariate — top features by scope",
        "",
    ]
    for scope in ("ALL", "GKP", "DEF", "MID", "FWD"):
        top = (
            uni_same.loc[uni_same["scope"] == scope]
            .sort_values("r2", ascending=False)
            .head(6)
        )
        lines.append(f"### {scope}")
        lines.append("")
        lines.append("| feature | n | corr | R² |")
        lines.append("|---|---:|---:|---:|")
        for r in top.itertuples():
            lines.append(
                f"| {r.feature} | {r.n} | {r.corr:.3f} | {r.r2:.3f} |"
            )
        lines.append("")

    lines += [
        "## Predictive univariate — top prior rates by scope",
        "",
        f"Filter: minutes ≥ {MIN_MINUTES}, prior apps ≥ {MIN_PRIOR}.",
        "",
    ]
    for scope in ("ALL", "GKP", "DEF", "MID", "FWD"):
        top = (
            uni_pred.loc[uni_pred["scope"] == scope]
            .sort_values("r2", ascending=False)
            .head(6)
        )
        lines.append(f"### {scope}")
        lines.append("")
        lines.append("| feature | n | corr | R² |")
        lines.append("|---|---:|---:|---:|")
        for r in top.itertuples():
            lines.append(
                f"| {r.feature} | {r.n} | {r.corr:.3f} | {r.r2:.3f} |"
            )
        lines.append("")

    lines += [
        "## Predictive multivariate vs points-baseline",
        "",
        "| scope | features | n | R² | R²(prior points only) | ΔR² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in multi_pred:
        lines.append(
            f"| {r['scope']} | `{r['features']}` | {r['n']} | "
            f"{r['r2']:.3f} | {r['r2_baseline_only']:.3f} | {r['delta_r2']:.3f} |"
        )

    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/raw_stats_same_match.png`",
        "- `data/plots/raw_stats_decomp.png`",
        "- `data/plots/raw_stats_predictive.png`",
        "",
        "## Read",
        "",
        "- High same-match R² ⇒ points are a near-deterministic function of events (BPS/bonus/G/A/CS).",
        "- Low predictive R² ⇒ those events are hard to forecast; baselines beat chasing noise.",
        "- Useful trends are channels we *can* predict (minutes, goal involvement rates), not markets→points clouds.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    raw = load_players()
    uni_same = same_match_univariate(raw)
    multi_same = same_match_multivariate(raw)

    with_priors = add_priors(raw)
    uni_pred = predictive_univariate(with_priors)
    multi_pred = predictive_multivariate(with_priors)

    plot_same_match(raw, uni_same, PLOTS / "raw_stats_same_match.png")
    plot_decomp(multi_same, PLOTS / "raw_stats_decomp.png")
    plot_predictive(with_priors, uni_pred, PLOTS / "raw_stats_predictive.png")

    eval_df = pd.concat(
        [
            uni_same,
            uni_pred,
        ],
        ignore_index=True,
    )
    eval_df.to_csv(PROCESSED / "raw_stats_eval.csv", index=False)
    write_report(
        REPORTS / "raw_player_stats_points.md",
        uni_same,
        multi_same,
        uni_pred,
        multi_pred,
    )
    return {
        "uni_same": uni_same,
        "multi_same": multi_same,
        "uni_pred": uni_pred,
        "multi_pred": multi_pred,
    }


if __name__ == "__main__":
    out = run()
    print("=== Same-match multivariate R² ===")
    for r in out["multi_same"]:
        print(f"  {r['scope']:4s}  R²={r['r2']:.3f}  ({r['features'][:60]})")
    print("=== Predictive top (ALL) ===")
    top = out["uni_pred"].loc[out["uni_pred"]["scope"] == "ALL"].sort_values("r2", ascending=False).head(8)
    for r in top.itertuples():
        print(f"  {r.feature:22s}  R²={r.r2:.3f}  corr={r.corr:.3f}")
    print("=== Predictive multivariate ===")
    for r in out["multi_pred"]:
        print(
            f"  {r['scope']:4s}  R²={r['r2']:.3f}  base={r['r2_baseline_only']:.3f}  "
            f"Δ={r['delta_r2']:.3f}"
        )
    print(f"Wrote {REPORTS}/raw_player_stats_points.md")
