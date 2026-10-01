"""Stage 6 — Shin P(win)/P(lose) only → team pot tilt → player μ.

Drops OU/AH. Uses:
  GKP/DEF: pot ~ p_lose
  MID:     pot ~ p_win
  FWD:     no fixture term (baseline only)

Player μ = baseline + fitted (prior_share × pot_tilt).

Writes:
  data/processed/team_pot_shin.csv
  data/processed/player_mu_shin.csv
  data/plots/stage_6_shin_pot.png
  data/plots/stage_6_shin_mu.png
  reports/stage_6_shin_only_mu.md
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

# Position → 1X2 feature (None = no market term).
POS_FEATURE = {
    "GKP": "p_lose",
    "DEF": "p_lose",
    "MID": "p_win",
    "FWD": None,
}


def _ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask].astype(float)
    x = x[mask].astype(float)
    n = int(y.size)
    if n < 10:
        return {
            "n": n,
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "corr": float("nan"),
        }
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    corr = float(np.corrcoef(x, y)[0, 1]) if n > 2 else float("nan")
    return {
        "n": n,
        "slope": float(coef[1]),
        "intercept": float(coef[0]),
        "r2": r2,
        "corr": corr,
    }


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


def _decile_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame({"x": x[mask], "y": y[mask]})
    if len(df) < 30:
        return float("nan")
    try:
        df["bin"] = pd.qcut(df["x"], 10, duplicates="drop")
    except ValueError:
        return float("nan")
    g = df.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
    if len(g) < 4:
        return float("nan")
    return float(g["x"].corr(g["y"]))


def _top_bottom_lift(score: pd.Series, realised: pd.Series) -> float:
    df = pd.DataFrame({"s": score, "y": realised}).dropna()
    if len(df) < 50:
        return float("nan")
    try:
        df["bin"] = pd.qcut(df["s"], 10, duplicates="drop")
    except ValueError:
        return float("nan")
    g = df.groupby("bin", observed=True)["y"].mean()
    if len(g) < 2:
        return float("nan")
    return float(g.iloc[-1] - g.iloc[0])


def build_side_pots(pots: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    fix = fixtures.copy()
    if "fixture_id" not in fix.columns:
        fix["fixture_id"] = (
            fix["date"].astype(str) + ":" + fix["home_norm"] + ":" + fix["away_norm"]
        )
    cols = ["fixture_id", "p_home", "p_draw", "p_away", "date"]
    have = [c for c in cols if c in fix.columns]
    m = pots.merge(fix[have], on="fixture_id", how="inner", suffixes=("", "_fix"))
    is_home = m["is_home"].astype(int) == 1
    m["p_win"] = np.where(is_home, m["p_home"], m["p_away"])
    m["p_lose"] = np.where(is_home, m["p_away"], m["p_home"])
    return m


def fit_team_pots(side: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit pot ~ 1X2 feature; pot_tilt = pot_hat − prior team×pos pot mean."""
    out = side.sort_values(
        ["team_norm", "position", "date", "fixture_id"], kind="mergesort"
    ).copy()
    out["pot_hat"] = np.nan
    out["pot_tilt"] = np.nan
    out["pot_prior_mean"] = out.groupby(["team_norm", "position"])["pot_points"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    pos_mean = out.groupby("position")["pot_points"].transform("mean")
    out["pot_prior_mean"] = out["pot_prior_mean"].fillna(pos_mean)

    rows: list[dict[str, Any]] = []
    for pos, feat in POS_FEATURE.items():
        mask = out["position"] == pos
        chunk = out.loc[mask]
        y = pd.to_numeric(chunk["pot_points"], errors="coerce").to_numpy(float)
        if feat is None:
            # FWD: pot_hat = prior mean (no Shin term).
            pot_hat = chunk["pot_prior_mean"].to_numpy(float)
            out.loc[mask, "pot_hat"] = pot_hat
            out.loc[mask, "pot_tilt"] = 0.0
            rows.append(
                {
                    "position": pos,
                    "feature": "none",
                    "n": int(len(chunk)),
                    "slope": 0.0,
                    "intercept": float("nan"),
                    "r2": 0.0,
                    "corr": float("nan"),
                    "decile_corr": float("nan"),
                    "use_market": False,
                }
            )
            continue

        x = pd.to_numeric(chunk[feat], errors="coerce").to_numpy(float)
        fit = _ols(y, x)
        pot_hat = fit["intercept"] + fit["slope"] * x
        pot_tilt = pot_hat - chunk["pot_prior_mean"].to_numpy(float)
        out.loc[mask, "pot_hat"] = pot_hat
        out.loc[mask, "pot_tilt"] = pot_tilt
        rows.append(
            {
                "position": pos,
                "feature": feat,
                "n": fit["n"],
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "r2": fit["r2"],
                "corr": fit["corr"],
                "decile_corr": _decile_corr(x, y),
                "use_market": True,
            }
        )
    return out, rows


def assemble_player_mu(
    players: pd.DataFrame, pots_shin: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """μ = baseline + α · (prior_share × pot_tilt) [+ xG share term for MID/FWD]."""
    keys = ["fixture_id", "team_norm", "position", "is_home"]
    tilt_cols = keys + ["p_win", "p_lose", "pot_hat", "pot_tilt", "pot_prior_mean", "pot_points"]
    # Avoid duplicate date columns on merge.
    merge_pots = pots_shin[tilt_cols].drop_duplicates(keys)
    df = players.merge(merge_pots, on=keys, how="inner", suffixes=("", "_pot"))
    df = df.sort_values(["player_id", "date", "fixture_id"], kind="mergesort")

    # Share × tilt interaction (player-scale fixture adjustment).
    share_m = pd.to_numeric(df.get("prior_share_minutes"), errors="coerce").fillna(0.0)
    share_x = pd.to_numeric(df.get("prior_share_xg"), errors="coerce").fillna(0.0)
    tilt = pd.to_numeric(df["pot_tilt"], errors="coerce").fillna(0.0)
    df["share_tilt_minutes"] = share_m * tilt
    df["share_tilt_xg"] = share_x * tilt

    df["mu_points"] = np.nan
    df["mu_baseline_only"] = pd.to_numeric(df["baseline_points"], errors="coerce")
    df["pred_residual"] = np.nan

    rows: list[dict[str, Any]] = []
    eligible = df.loc[
        (pd.to_numeric(df["minutes"], errors="coerce") >= MIN_MINUTES)
        & (pd.to_numeric(df["n_prior_apps"], errors="coerce") >= MIN_PRIOR)
    ].copy()

    for pos in ("GKP", "DEF", "MID", "FWD"):
        chunk = eligible.loc[eligible["position"] == pos].copy()
        if len(chunk) < 40:
            rows.append({"position": pos, "n": len(chunk), "error": "too few"})
            continue

        y = chunk["residual_points"].to_numpy(float)
        base = chunk["baseline_points"].to_numpy(float)
        pts = chunk["total_points"].to_numpy(float)

        if POS_FEATURE[pos] is None:
            # Baseline only.
            pred_resid = np.zeros(len(chunk))
            mu = base
            feat_names: list[str] = []
            r2_resid = 0.0
            coefs: dict[str, float] = {}
        else:
            feat_names = ["share_tilt_minutes"]
            cols = [chunk["share_tilt_minutes"].to_numpy(float)]
            if pos in ("MID", "FWD") and "share_tilt_xg" in chunk.columns:
                feat_names.append("share_tilt_xg")
                cols.append(chunk["share_tilt_xg"].to_numpy(float))
            # Also allow raw pot_tilt as team-level fallback (small weight if share~0).
            feat_names.append("pot_tilt")
            cols.append(chunk["pot_tilt"].to_numpy(float))

            X = np.column_stack([np.ones(len(chunk)), *cols])
            coef, r2_resid, pred = _multi_ols(y, X)
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            pred_resid = np.full(len(chunk), np.nan)
            pred_resid[mask] = pred
            mu = base + pred_resid
            coefs = (
                {name: float(coef[i + 1]) for i, name in enumerate(feat_names)}
                if np.all(np.isfinite(coef))
                else {}
            )

        df.loc[chunk.index, "pred_residual"] = pred_resid
        df.loc[chunk.index, "mu_points"] = mu

        eval_df = pd.DataFrame(
            {
                "mu": mu,
                "base": base,
                "pts": pts,
            }
        ).dropna()
        corr_mu = (
            float(eval_df["mu"].corr(eval_df["pts"])) if len(eval_df) > 5 else float("nan")
        )
        corr_base = (
            float(eval_df["base"].corr(eval_df["pts"])) if len(eval_df) > 5 else float("nan")
        )
        lift_mu = _top_bottom_lift(eval_df["mu"], eval_df["pts"])
        lift_base = _top_bottom_lift(eval_df["base"], eval_df["pts"])

        rows.append(
            {
                "position": pos,
                "feature": POS_FEATURE[pos] or "none",
                "n": int(len(eval_df)),
                "features": feat_names,
                "coefs": coefs,
                "r2_resid": float(r2_resid),
                "corr_mu": corr_mu,
                "corr_base": corr_base,
                "delta_corr": corr_mu - corr_base
                if np.isfinite(corr_mu) and np.isfinite(corr_base)
                else float("nan"),
                "lift_mu": lift_mu,
                "lift_base": lift_base,
                "delta_lift": lift_mu - lift_base
                if np.isfinite(lift_mu) and np.isfinite(lift_base)
                else float("nan"),
                "beats_baseline": bool(
                    (np.isfinite(corr_mu) and np.isfinite(corr_base) and corr_mu > corr_base)
                    or (
                        np.isfinite(lift_mu)
                        and np.isfinite(lift_base)
                        and lift_mu > lift_base
                    )
                ),
            }
        )
    return df, rows


def plot_team_pots(pots: pd.DataFrame, pot_rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Stage 6 — Team pot vs Shin 1X2 only", fontsize=13)
    meta = {r["position"]: r for r in pot_rows}
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        sub = pots.loc[pots["position"] == pos]
        feat = POS_FEATURE[pos]
        info = meta[pos]
        if feat is None:
            ax.text(0.5, 0.5, "FWD: no Shin term\n(baseline pot only)", ha="center", va="center")
            ax.set_axis_off()
            ax.set_title("FWD [off]")
            continue
        x = pd.to_numeric(sub[feat], errors="coerce")
        y = pd.to_numeric(sub["pot_points"], errors="coerce")
        ax.scatter(x, y, s=10, alpha=0.2, edgecolors="none", color="steelblue")
        # Decile means
        df = pd.DataFrame({"x": x, "y": y}).dropna()
        try:
            df["bin"] = pd.qcut(df["x"], 10, duplicates="drop")
            g = df.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
            ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=6)
        except ValueError:
            pass
        if np.isfinite(info["slope"]):
            xs = np.linspace(float(x.min()), float(x.max()), 40)
            ax.plot(
                xs,
                info["intercept"] + info["slope"] * xs,
                color="black",
                ls="--",
                lw=1,
            )
        ax.set_title(
            f"{pos}: {feat}  R²={info['r2']:.3f}  decile_r={info['decile_corr']:.3f}"
        )
        ax.set_xlabel(feat)
        ax.set_ylabel("pot_points")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_player_mu(df: pd.DataFrame, mu_rows: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Stage 6 — Player μ (Shin-only) vs realised points", fontsize=13)
    meta = {r["position"]: r for r in mu_rows if "error" not in r}
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        sub = df.loc[
            (df["position"] == pos)
            & (pd.to_numeric(df["minutes"], errors="coerce") >= MIN_MINUTES)
            & (pd.to_numeric(df["n_prior_apps"], errors="coerce") >= MIN_PRIOR)
            & df["mu_points"].notna()
        ]
        ax.scatter(
            sub["mu_points"],
            sub["total_points"],
            s=12,
            alpha=0.3,
            edgecolors="none",
            color="steelblue",
        )
        if len(sub) > 5:
            lo = float(min(sub["mu_points"].min(), sub["total_points"].min()))
            hi = float(max(sub["mu_points"].max(), sub["total_points"].max()))
            ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
        info = meta.get(pos, {})
        ax.set_title(
            f"{pos}  corr_μ={info.get('corr_mu', float('nan')):.3f}  "
            f"Δcorr={info.get('delta_corr', float('nan')):.3f}  "
            f"Δlift={info.get('delta_lift', float('nan')):.2f}"
        )
        ax.set_xlabel("μ (Shin-only)")
        ax.set_ylabel("realised points")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    pot_rows: list[dict[str, Any]],
    mu_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Stage 6 — Shin P(win)/P(lose) only → player μ",
        "",
        "Market features: **p_win / p_lose only** (no OU, no AH).",
        "",
        "| pos | pot feature |",
        "|---|---|",
        "| GKP | `p_lose` |",
        "| DEF | `p_lose` |",
        "| MID | `p_win` |",
        "| FWD | none (baseline only) |",
        "",
        "## Team pot fits",
        "",
        "| pos | feature | n | β | R² | decile_r |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in pot_rows:
        lines.append(
            f"| {r['position']} | {r['feature']} | {r['n']} | "
            f"{r['slope']:.2f} | {r['r2']:.3f} | {r['decile_corr']:.3f} |"
        )
    lines += [
        "",
        "## Player μ vs baseline-only",
        "",
        "μ = baseline + residual~(share×pot_tilt, pot_tilt).",
        "",
        "| pos | n | R²(resid) | corr_μ | corr_base | Δcorr | lift_μ | lift_base | Δlift | beats? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in mu_rows:
        if "error" in r:
            lines.append(f"| {r['position']} | {r['n']} | — | — | — | — | — | — | — | N |")
            continue
        beats = "Y" if r.get("beats_baseline") else "N"
        lines.append(
            f"| {r['position']} | {r['n']} | {r['r2_resid']:.3f} | "
            f"{r['corr_mu']:.3f} | {r['corr_base']:.3f} | {r['delta_corr']:.3f} | "
            f"{r['lift_mu']:.2f} | {r['lift_base']:.2f} | {r['delta_lift']:.2f} | {beats} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        "- Ship DEF/MID fixture term if Δcorr > 0 or Δlift > 0.",
        "- FWD stays baseline-only by design.",
        "",
        "## Plots",
        "",
        "- `data/plots/stage_6_shin_pot.png`",
        "- `data/plots/stage_6_shin_mu.png`",
        "",
        "## Outputs",
        "",
        "- `data/processed/team_pot_shin.csv`",
        "- `data/processed/player_mu_shin.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_shin_only_mu() -> dict[str, Any]:
    pots = pd.read_csv(PROCESSED / "team_pos_pots.csv")
    fixtures = pd.read_csv(PROCESSED / "fixtures_odds.csv")
    players = pd.read_csv(PROCESSED / "player_baseline.csv")

    side = build_side_pots(pots, fixtures)
    pots_shin, pot_rows = fit_team_pots(side)
    pots_shin.to_csv(PROCESSED / "team_pot_shin.csv", index=False)

    players_mu, mu_rows = assemble_player_mu(players, pots_shin)
    players_mu.to_csv(PROCESSED / "player_mu_shin.csv", index=False)

    plot_team_pots(pots_shin, pot_rows, PLOTS / "stage_6_shin_pot.png")
    plot_player_mu(players_mu, mu_rows, PLOTS / "stage_6_shin_mu.png")
    write_report(REPORTS / "stage_6_shin_only_mu.md", pot_rows, mu_rows)

    return {"pot_rows": pot_rows, "mu_rows": mu_rows}


if __name__ == "__main__":
    out = run_shin_only_mu()
    print("Team pots:")
    for r in out["pot_rows"]:
        print(
            f"  {r['position']}: {r['feature']}  R²={r['r2']:.3f}  "
            f"decile_r={r['decile_corr']:.3f}"
        )
    print("Player μ:")
    for r in out["mu_rows"]:
        if "error" in r:
            print(f"  {r['position']}: too few")
            continue
        print(
            f"  {r['position']}: Δcorr={r['delta_corr']:+.3f}  "
            f"Δlift={r['delta_lift']:+.2f}  beats={r['beats_baseline']}"
        )
    print(f"Wrote {REPORTS}/stage_6_shin_only_mu.md")
