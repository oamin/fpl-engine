"""Shin P(win) vs team×pos pot points (pre-match 1X2 signal).

Writes:
  data/plots/shin_pwin_vs_pot.png
  data/plots/shin_pwin_vs_pot_deciles.png
  reports/shin_pwin_pots.md
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


def _ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask].astype(float)
    x = x[mask].astype(float)
    n = int(y.size)
    if n < 10:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "corr": float("nan")}
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


def _decile_means(x: np.ndarray, y: np.ndarray) -> pd.DataFrame | None:
    mask = np.isfinite(y) & np.isfinite(x)
    df = pd.DataFrame({"x": x[mask], "y": y[mask]})
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


def build_side_pots(pots: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    fix = fixtures.copy()
    if "fixture_id" not in fix.columns:
        fix["fixture_id"] = (
            fix["date"].astype(str) + ":" + fix["home_norm"] + ":" + fix["away_norm"]
        )
    cols = [
        "fixture_id",
        "p_home",
        "p_draw",
        "p_away",
        "home_goals",
        "away_goals",
        "home_norm",
        "away_norm",
    ]
    have = [c for c in cols if c in fix.columns]
    m = pots.merge(fix[have], on="fixture_id", how="inner")
    is_home = m["is_home"].astype(int) == 1
    m["p_win"] = np.where(is_home, m["p_home"], m["p_away"])
    m["p_lose"] = np.where(is_home, m["p_away"], m["p_home"])
    m["p_draw_side"] = m["p_draw"]
    own_g = np.where(is_home, m["home_goals"], m["away_goals"])
    opp_g = np.where(is_home, m["away_goals"], m["home_goals"])
    m["wdl"] = np.where(own_g > opp_g, "W", np.where(own_g < opp_g, "L", "D"))
    return m


def plot_scatter(df: pd.DataFrame, out: Path) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(
        "Shin P(win) → team×pos pot points (pre-match 1X2)",
        fontsize=13,
    )
    rows: list[dict[str, Any]] = []
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        sub = df.loc[df["position"] == pos]
        x = pd.to_numeric(sub["p_win"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["pot_points"], errors="coerce").to_numpy(float)
        fit = _ols(y, x)
        ax.scatter(x, y, s=14, alpha=0.3, edgecolors="none", color="steelblue")
        if np.isfinite(fit["slope"]):
            xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 40)
            ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color="crimson", lw=2)
        ax.set_title(f"{pos}")
        ax.set_xlabel("Shin P(win)")
        ax.set_ylabel("pot_points")
        ax.text(
            0.03,
            0.97,
            f"n={fit['n']}\nβ={fit['slope']:.1f}\n"
            f"corr={fit['corr']:.3f}\nR²={fit['r2']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        rows.append({"position": pos, "view": "match_scatter", **fit})
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return rows


def plot_deciles(df: pd.DataFrame, out: Path) -> list[dict[str, Any]]:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(
        "Shin P(win) deciles → mean pot points (gate-style calibration)",
        fontsize=13,
    )
    rows: list[dict[str, Any]] = []
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        sub = df.loc[df["position"] == pos]
        x = pd.to_numeric(sub["p_win"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["pot_points"], errors="coerce").to_numpy(float)
        fit = _ols(y, x)
        g = _decile_means(x, y)
        ax.scatter(x, y, s=8, alpha=0.12, color="steelblue", edgecolors="none")
        dcorr = float("nan")
        if g is not None:
            ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=7, label="decile mean")
            dcorr = float(g["x"].corr(g["y"]))
            for _, row in g.iterrows():
                ax.annotate(
                    f"{row['y']:.1f}",
                    (row["x"], row["y"]),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7,
                    color="0.3",
                )
        ax.set_title(f"{pos}")
        ax.set_xlabel("Shin P(win)")
        ax.set_ylabel("pot_points")
        ax.text(
            0.03,
            0.97,
            f"decile_r={dcorr:.3f}\nmatch R²={fit['r2']:.3f}\nβ={fit['slope']:.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        rows.append(
            {
                "position": pos,
                "view": "decile",
                "decile_corr": dcorr,
                "r2": fit["r2"],
                "slope": fit["slope"],
                "n": fit["n"],
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return rows


def plot_compare_wdl(df: pd.DataFrame, out: Path) -> None:
    """Left: realised W/D/L mean pots. Right: Shin P(win) decile means for MID+DEF."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    fig.suptitle(
        "What Shin P(win) is pricing: result pots vs pre-match P(win)",
        fontsize=13,
    )

    # Left: WDL bars for DEF/MID
    ax = axes[0]
    positions = ["GKP", "DEF", "MID", "FWD"]
    wdl_order = ["W", "D", "L"]
    colors = {"W": "#2ca02c", "D": "#ffbf00", "L": "#8c564b"}
    x = np.arange(len(positions))
    width = 0.25
    for i, wdl in enumerate(wdl_order):
        means = []
        for pos in positions:
            sub = df.loc[(df["position"] == pos) & (df["wdl"] == wdl), "pot_points"]
            means.append(float(sub.mean()) if len(sub) else float("nan"))
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            label=wdl,
            color=colors[wdl],
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel("mean pot_points")
    ax.set_title("Realised: mean pot after W / D / L")
    ax.legend(title="result", fontsize=8)

    # Right: DEF + MID decile curves for P(win)
    ax = axes[1]
    for pos, color in (("DEF", "steelblue"), ("MID", "crimson")):
        sub = df.loc[df["position"] == pos]
        xx = pd.to_numeric(sub["p_win"], errors="coerce").to_numpy(float)
        yy = pd.to_numeric(sub["pot_points"], errors="coerce").to_numpy(float)
        g = _decile_means(xx, yy)
        if g is not None:
            ax.plot(g["x"], g["y"], "o-", color=color, lw=2, ms=6, label=pos)
    ax.set_xlabel("Shin P(win)")
    ax.set_ylabel("mean pot_points")
    ax.set_title("Pre-match: P(win) decile → mean pot")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    scatter_rows: list[dict[str, Any]],
    decile_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Shin P(win) vs team×pos pots",
        "",
        "Shin P(win) = side-specific win probability from Shin-devigged 1X2 "
        "(home uses `p_home`, away uses `p_away`).",
        "",
        "## Match-level",
        "",
        "| pos | n | slope β | corr | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in scatter_rows:
        lines.append(
            f"| {r['position']} | {r['n']} | {r['slope']:.2f} | "
            f"{r['corr']:.3f} | {r['r2']:.3f} |"
        )
    lines += [
        "",
        "## Decile calibration",
        "",
        "| pos | n | slope β | match R² | decile_r |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in decile_rows:
        lines.append(
            f"| {r['position']} | {r['n']} | {r['slope']:.2f} | "
            f"{r['r2']:.3f} | {r['decile_corr']:.3f} |"
        )
    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/shin_pwin_vs_pot.png` — match scatter",
        "- `data/plots/shin_pwin_vs_pot_deciles.png` — decile means",
        "- `data/plots/shin_pwin_vs_wdl_compare.png` — WDL bars vs P(win) curves",
        "",
        "## Read",
        "",
        "- Use **Shin P(win)** as the pre-match fixture feature (no outcome leakage).",
        "- DEF/MID show clear positive slopes; FWD is nearly flat.",
        "- Decile curves are the fair visual for “does the market move the pot?”",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    pots = pd.read_csv(PROCESSED / "team_pos_pots.csv")
    fixtures = pd.read_csv(PROCESSED / "fixtures_odds.csv")
    side = build_side_pots(pots, fixtures)
    scatter = plot_scatter(side, PLOTS / "shin_pwin_vs_pot.png")
    deciles = plot_deciles(side, PLOTS / "shin_pwin_vs_pot_deciles.png")
    plot_compare_wdl(side, PLOTS / "shin_pwin_vs_wdl_compare.png")
    write_report(REPORTS / "shin_pwin_pots.md", scatter, deciles)
    return {"scatter": scatter, "deciles": deciles, "n": len(side)}


if __name__ == "__main__":
    out = run()
    for r in out["deciles"]:
        print(
            f"{r['position']}: β={r['slope']:.1f}  "
            f"R²={r['r2']:.3f}  decile_r={r['decile_corr']:.3f}"
        )
    print(f"Wrote plots under {PLOTS}/shin_pwin_*.png")
