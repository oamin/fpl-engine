"""Stage 3 — do team×pos FPL points track 1X2 / OU / AH?"""

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

# Go/no-go: decile calibration correlation and residual ΔR² vs market feature.
GO_CORR = 0.55
GO_DELTA_R2 = 0.02


def _ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask].astype(float)
    x = x[mask].astype(float)
    n = int(y.size)
    if n < 10:
        return {"n": n, "slope": float("nan"), "r2": float("nan"), "corr": float("nan")}
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    corr = float(np.corrcoef(x, y)[0, 1]) if n > 2 else float("nan")
    return {"n": n, "slope": float(coef[1]), "r2": r2, "corr": corr}


def _decile_means(x: np.ndarray, y: np.ndarray) -> pd.DataFrame | None:
    """Mean x and y per market decile (the gate calibration curve)."""
    mask = np.isfinite(y) & np.isfinite(x)
    df = pd.DataFrame({"y": y[mask], "x": x[mask]})
    if len(df) < 30:
        return None
    try:
        df["bin"] = pd.qcut(df["x"], 10, duplicates="drop")
    except ValueError:
        return None
    g = df.groupby("bin", observed=True).agg(
        x=("x", "mean"), y=("y", "mean"), n=("y", "count")
    )
    if len(g) < 4:
        return None
    return g.reset_index(drop=True)


def _decile_corr(y: np.ndarray, x: np.ndarray) -> float:
    g = _decile_means(x, y)
    if g is None:
        return float("nan")
    return float(g["x"].corr(g["y"]))


def _best_feature_for_position(
    results: list[dict[str, Any]], position: str
) -> dict[str, Any]:
    pos_rows = [r for r in results if r["position"] == position]
    return max(
        pos_rows,
        key=lambda r: abs(r["decile_corr"]) if np.isfinite(r["decile_corr"]) else -1.0,
    )


def _plot_decile_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    decile_r: float,
    match_r2: float,
) -> None:
    g = _decile_means(x, y)
    ax.scatter(x, y, s=8, alpha=0.12, color="steelblue", edgecolors="none", zorder=1)
    if g is not None:
        ax.plot(
            g["x"],
            g["y"],
            "o-",
            color="crimson",
            lw=2,
            ms=7,
            label="decile means",
            zorder=3,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pot_points")
    ax.text(
        0.03,
        0.97,
        f"decile_r={decile_r:.3f}\nR²(match)={match_r2:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def build_side_pots(pots: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """One row per team-side × position with market features."""
    fix = fixtures.copy()
    fix["fixture_id"] = (
        fix["date"].astype(str) + ":" + fix["home_norm"] + ":" + fix["away_norm"]
    )
    # Ensure pots have fixture market cols via merge on fixture_id.
    cols = [
        "fixture_id",
        "p_home",
        "p_draw",
        "p_away",
        "p_over25",
        "p_under25",
        "p_ah_home",
        "p_ah_away",
        "ah_line",
        "home_norm",
        "away_norm",
    ]
    have = [c for c in cols if c in fix.columns]
    merged = pots.merge(fix[have], on="fixture_id", how="inner")
    is_home = merged["is_home"].astype(int) == 1
    merged["p_win"] = np.where(is_home, merged["p_home"], merged["p_away"])
    merged["p_lose"] = np.where(is_home, merged["p_away"], merged["p_home"])
    merged["p_over"] = merged["p_over25"]
    ah = pd.to_numeric(merged["ah_line"], errors="coerce")
    merged["ah_line_own"] = np.where(is_home, ah, -ah)
    merged["p_ah_cover"] = np.where(
        is_home,
        pd.to_numeric(merged["p_ah_home"], errors="coerce"),
        pd.to_numeric(merged["p_ah_away"], errors="coerce"),
    )
    # Strength proxy: higher = better for this side's attack / worse for CS.
    merged["attack_strength"] = merged["p_win"] + 0.5 * merged["p_draw"]
    merged["defend_threat"] = merged["p_lose"] + 0.5 * merged["p_draw"]  # opp win-ish
    return merged


def evaluate_position(df: pd.DataFrame, position: str) -> list[dict[str, Any]]:
    sub = df.loc[df["position"] == position].copy()
    y = pd.to_numeric(sub["pot_points"], errors="coerce").to_numpy(float)
    # Defence cares about opp attack / not covering; attack cares about own strength.
    if position in ("GKP", "DEF"):
        features = [
            ("defend_threat", "opp strength (p_lose+0.5p_d)"),
            ("p_over", "p(over 2.5)"),
            ("ah_line_own", "AH line own"),
            ("p_ah_cover", "p(AH cover)"),
        ]
    else:
        features = [
            ("attack_strength", "attack strength (p_win+0.5p_d)"),
            ("p_over", "p(over 2.5)"),
            ("ah_line_own", "AH line own"),
            ("p_ah_cover", "p(AH cover)"),
        ]
    rows: list[dict[str, Any]] = []
    for col, label in features:
        x = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
        fit = _ols(y, x)
        dcorr = _decile_corr(y, x)
        # Sign: for GKP/DEF, higher defend_threat should mean fewer points → negative corr OK if |corr| high
        go = bool(
            (np.isfinite(dcorr) and abs(dcorr) >= GO_CORR)
            or (np.isfinite(fit["r2"]) and fit["r2"] >= GO_DELTA_R2)
        )
        rows.append(
            {
                "position": position,
                "feature": col,
                "label": label,
                "n": fit["n"],
                "slope": fit["slope"],
                "r2": fit["r2"],
                "corr": fit["corr"],
                "decile_corr": dcorr,
                "go": go,
            }
        )
    return rows


def plot_team_market(df: pd.DataFrame, results: list[dict[str, Any]], out: Path) -> None:
    """Match-level scatter (noisy view)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Stage 3 — Match-level scatter (best |decile_r| feature)", fontsize=13)
    pos_order = ("GKP", "DEF", "MID", "FWD")
    for ax, pos in zip(axes.ravel(), pos_order):
        best = _best_feature_for_position(results, pos)
        sub = df.loc[df["position"] == pos]
        x = pd.to_numeric(sub[best["feature"]], errors="coerce")
        y = pd.to_numeric(sub["pot_points"], errors="coerce")
        ax.scatter(x, y, s=14, alpha=0.35, edgecolors="none")
        mask = x.notna() & y.notna()
        if mask.sum() > 5:
            xx = x[mask].to_numpy(float)
            yy = y[mask].to_numpy(float)
            fit = _ols(yy, xx)
            if np.isfinite(fit["slope"]):
                xs = np.linspace(float(np.nanmin(xx)), float(np.nanmax(xx)), 40)
                intercept = float(np.mean(yy) - fit["slope"] * np.mean(xx))
                ax.plot(xs, fit["slope"] * xs + intercept, color="crimson", lw=2)
        flag = "GO" if best["go"] else "NO-GO"
        ax.set_title(
            f"{pos}: {best['feature']} [{flag}]\n"
            f"R²={best['r2']:.3f} decile_r={best['decile_corr']:.3f}"
        )
        ax.set_xlabel(best["feature"])
        ax.set_ylabel("pot_points")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_gate_deciles(df: pd.DataFrame, results: list[dict[str, Any]], out: Path) -> None:
    """Gate check: decile-mean calibration curves (what decile_r measures)."""
    panels: list[tuple[str, str, float, float, bool]] = []
    for pos in ("GKP", "DEF", "MID", "FWD"):
        best = _best_feature_for_position(results, pos)
        panels.append(
            (pos, best["feature"], best["decile_corr"], best["r2"], best["go"])
        )
    # MID also show 1X2 attack_strength (often clearer than AH line).
    mid_atk = next(
        (r for r in results if r["position"] == "MID" and r["feature"] == "attack_strength"),
        None,
    )
    if mid_atk is not None:
        panels.append(
            (
                "MID",
                "attack_strength",
                mid_atk["decile_corr"],
                mid_atk["r2"],
                mid_atk["go"],
            )
        )

    n = len(panels)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.2 * nrows), constrained_layout=True)
    fig.suptitle(
        f"Stage 3 — Gate checks (decile means, pass if |decile_r| ≥ {GO_CORR})",
        fontsize=13,
    )
    flat = np.atleast_1d(axes).ravel()
    for ax, (pos, feat, dcorr, r2, go) in zip(flat, panels):
        sub = df.loc[df["position"] == pos]
        x = pd.to_numeric(sub[feat], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["pot_points"], errors="coerce").to_numpy(float)
        label = f"{pos}: {feat} [{'GO' if go else 'NO-GO'}]"
        if feat == "attack_strength" and pos == "MID":
            label = f"MID: attack_strength (1X2) [{'GO' if go else 'NO-GO'}]"
        _plot_decile_panel(
            ax, x, y, title=label, xlabel=feat, decile_r=dcorr, match_r2=r2
        )
    for ax in flat[len(panels) :]:
        ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_gate_combined(df: pd.DataFrame, results: list[dict[str, Any]], out: Path) -> None:
    """Side-by-side: match scatter vs decile curve per position."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 14), constrained_layout=True)
    fig.suptitle("Stage 3 — Scatter vs decile gate check (by position)", fontsize=13)
    for row, pos in enumerate(("GKP", "DEF", "MID", "FWD")):
        best = _best_feature_for_position(results, pos)
        feat = best["feature"]
        sub = df.loc[df["position"] == pos]
        x = pd.to_numeric(sub[feat], errors="coerce")
        y = pd.to_numeric(sub["pot_points"], errors="coerce")
        ax_s, ax_d = axes[row, 0], axes[row, 1]

        ax_s.scatter(x, y, s=14, alpha=0.35, edgecolors="none")
        mask = x.notna() & y.notna()
        if mask.sum() > 5:
            xx = x[mask].to_numpy(float)
            yy = y[mask].to_numpy(float)
            fit = _ols(yy, xx)
            if np.isfinite(fit["slope"]):
                xs = np.linspace(float(np.nanmin(xx)), float(np.nanmax(xx)), 40)
                intercept = float(np.mean(yy) - fit["slope"] * np.mean(xx))
                ax_s.plot(xs, fit["slope"] * xs + intercept, color="crimson", lw=2)
        ax_s.set_title(f"{pos} scatter — {feat}")
        ax_s.set_xlabel(feat)
        ax_s.set_ylabel("pot_points")

        _plot_decile_panel(
            ax_d,
            x.to_numpy(float),
            y.to_numpy(float),
            title=f"{pos} gate — {feat}",
            xlabel=feat,
            decile_r=best["decile_corr"],
            match_r2=best["r2"],
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_stage3_report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 3 — Team pots vs 1X2 / OU / AH",
        "",
        f"Go thresholds: |decile corr| ≥ {GO_CORR} **or** match R² ≥ {GO_DELTA_R2}.",
        "",
        "| pos | feature | n | slope | R² | corr | decile_corr | go |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['position']} | {r['feature']} | {r['n']} | {r['slope']:.3f} | "
            f"{r['r2']:.3f} | {r['corr']:.3f} | {r['decile_corr']:.3f} | "
            f"{'Y' if r['go'] else 'N'} |"
        )
    # Summary go per position (any feature)
    lines += ["", "## Position go/no-go (any feature passes)", ""]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        ok = any(r["go"] for r in results if r["position"] == pos)
        lines.append(f"- **{pos}**: {'GO' if ok else 'NO-GO'}")
    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/stage_3_team_market.png` — match-level scatter",
        "- `data/plots/stage_3_gate_deciles.png` — **gate check** (decile means)",
        "- `data/plots/stage_3_gate_combined.png` — scatter vs decile by position",
        "",
        "Output: `data/processed/team_market_eval.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_team_market(
    pots_path: Path | None = None,
    fixtures_path: Path | None = None,
) -> dict[str, Any]:
    pots = pd.read_csv(pots_path or (PROCESSED / "team_pos_pots.csv"))
    fixtures = pd.read_csv(fixtures_path or (PROCESSED / "fixtures_odds.csv"))
    # Pots may lack fixture market columns — merge from fixtures.
    side = build_side_pots(pots, fixtures)
    results: list[dict[str, Any]] = []
    for pos in ("GKP", "DEF", "MID", "FWD"):
        results.extend(evaluate_position(side, pos))
    plot_team_market(side, results, PLOTS / "stage_3_team_market.png")
    plot_gate_deciles(side, results, PLOTS / "stage_3_gate_deciles.png")
    plot_gate_combined(side, results, PLOTS / "stage_3_gate_combined.png")
    pd.DataFrame(results).to_csv(PROCESSED / "team_market_eval.csv", index=False)
    write_stage3_report(REPORTS / "stage_3_team_market.md", results)
    go_pos = {
        pos: any(r["go"] for r in results if r["position"] == pos)
        for pos in ("GKP", "DEF", "MID", "FWD")
    }
    return {"results": results, "go_positions": go_pos}
