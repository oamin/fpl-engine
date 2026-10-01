"""Shin de-vigged 1X2 / OU implied probs vs realised match outcomes.

Writes:
  data/plots/market_cal_1x2.png
  data/plots/market_cal_ou.png
  data/plots/market_cal_compare.png
  reports/market_calibration.md
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


def _decile_calibration(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame | None:
    mask = np.isfinite(p) & np.isfinite(y)
    df = pd.DataFrame({"p": p[mask], "y": y[mask]})
    if len(df) < 30:
        return None
    try:
        df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    except ValueError:
        return None
    g = df.groupby("bin", observed=True).agg(
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
        n=("y", "count"),
    )
    if len(g) < 4:
        return None
    return g.reset_index(drop=True)


def _cal_stats(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask].astype(float)
    y = y[mask].astype(float)
    g = _decile_calibration(p, y)
    if g is None or len(g) < 4:
        return {
            "n": float(p.size),
            "base_rate": float(y.mean()) if p.size else float("nan"),
            "mean_p": float(p.mean()) if p.size else float("nan"),
            "decile_corr": float("nan"),
            "decile_r2": float("nan"),
            "brier": float("nan"),
            "mae_decile": float("nan"),
        }
    # Decile R² of y_rate ~ p_mean (calibration quality).
    xm = g["p_mean"].to_numpy(float)
    ym = g["y_rate"].to_numpy(float)
    ss_res = float(np.sum((ym - xm) ** 2))
    ss_tot = float(np.sum((ym - ym.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "n": float(p.size),
        "base_rate": float(y.mean()),
        "mean_p": float(p.mean()),
        "decile_corr": float(np.corrcoef(xm, ym)[0, 1]),
        "decile_r2": float(r2),
        "brier": float(np.mean((p - y) ** 2)),
        "mae_decile": float(np.mean(np.abs(ym - xm))),
    }


def _draw_cal_panel(
    ax: plt.Axes,
    p: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    color: str = "crimson",
) -> dict[str, float]:
    stats = _cal_stats(p, y)
    g = _decile_calibration(p, y)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.2, label="perfect (y = p)")
    if g is not None:
        ax.plot(
            g["p_mean"],
            g["y_rate"],
            "o-",
            color=color,
            lw=2,
            ms=7,
            label="decile empirical",
        )
        for _, row in g.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["p_mean"], row["y_rate"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color="0.35",
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Shin de-vig implied p")
    ax.set_ylabel("Empirical rate")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.98,
        0.02,
        f"n={int(stats['n'])}\n"
        f"mean p={stats['mean_p']:.3f}\n"
        f"rate={stats['base_rate']:.3f}\n"
        f"decile r={stats['decile_corr']:.3f}\n"
        f"cal R²={stats['decile_r2']:.3f}\n"
        f"Brier={stats['brier']:.3f}\n"
        f"MAE_bin={stats['mae_decile']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    return stats


def plot_1x2(df: pd.DataFrame, out: Path) -> list[dict[str, Any]]:
    home_win = (df["home_goals"] > df["away_goals"]).astype(float).to_numpy()
    draw = (df["home_goals"] == df["away_goals"]).astype(float).to_numpy()
    away_win = (df["home_goals"] < df["away_goals"]).astype(float).to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    fig.suptitle("1X2 — Shin de-vig implied p vs realised outcome (decile calibration)", fontsize=13)
    specs = [
        (axes[0], df["p_home"].to_numpy(float), home_win, "Home win", "steelblue"),
        (axes[1], df["p_draw"].to_numpy(float), draw, "Draw", "darkorange"),
        (axes[2], df["p_away"].to_numpy(float), away_win, "Away win", "seagreen"),
    ]
    rows: list[dict[str, Any]] = []
    for ax, p, y, title, color in specs:
        stats = _draw_cal_panel(ax, p, y, title=title, color=color)
        rows.append({"market": "1X2", "outcome": title, **stats})
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return rows


def plot_ou(df: pd.DataFrame, out: Path) -> list[dict[str, Any]]:
    over = ((df["home_goals"] + df["away_goals"]) >= 3).astype(float).to_numpy()
    under = 1.0 - over
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), constrained_layout=True)
    fig.suptitle("OU 2.5 — Shin de-vig implied p vs realised outcome", fontsize=13)
    rows: list[dict[str, Any]] = []
    specs = [
        (axes[0], df["p_over25"].to_numpy(float), over, "Over 2.5", "crimson"),
        (axes[1], df["p_under25"].to_numpy(float), under, "Under 2.5", "purple"),
    ]
    for ax, p, y, title, color in specs:
        stats = _draw_cal_panel(ax, p, y, title=title, color=color)
        rows.append({"market": "OU2.5", "outcome": title, **stats})
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return rows


def plot_compare(df: pd.DataFrame, out: Path) -> None:
    """Side-by-side home-win 1X2 vs over 2.5 for easy visual compare."""
    home_win = (df["home_goals"] > df["away_goals"]).astype(float).to_numpy()
    over = ((df["home_goals"] + df["away_goals"]) >= 3).astype(float).to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    fig.suptitle("Market calibration compare — 1X2 home vs OU over 2.5", fontsize=13)
    _draw_cal_panel(
        axes[0],
        df["p_home"].to_numpy(float),
        home_win,
        title="1X2: P(home) → home win rate",
        color="steelblue",
    )
    _draw_cal_panel(
        axes[1],
        df["p_over25"].to_numpy(float),
        over,
        title="OU: P(over 2.5) → over rate",
        color="crimson",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], n_fixtures: int) -> None:
    lines = [
        "# Market calibration — 1X2 & OU vs outcomes",
        "",
        f"- Fixtures: **{n_fixtures}** (Shin de-vig from football-data closing-ish prices)",
        "",
        "| market | outcome | n | mean p | rate | decile r | cal R² | Brier | MAE_bin |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['market']} | {r['outcome']} | {int(r['n'])} | "
            f"{r['mean_p']:.3f} | {r['base_rate']:.3f} | "
            f"{r['decile_corr']:.3f} | {r['decile_r2']:.3f} | "
            f"{r['brier']:.3f} | {r['mae_decile']:.3f} |"
        )
    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/market_cal_1x2.png`",
        "- `data/plots/market_cal_ou.png`",
        "- `data/plots/market_cal_compare.png`",
        "",
        "## Read",
        "",
        "- Points near the diagonal ⇒ well-calibrated implied probs.",
        "- High **cal R² / decile r** with low match Brier is normal for binary events.",
        "- Compare 1X2 home vs OU over: both should track y=p if markets are sharp.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_market_calibration(
    fixtures_path: Path | None = None,
) -> dict[str, Any]:
    path = fixtures_path or (PROCESSED / "fixtures_odds.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=["p_home", "p_draw", "p_away", "p_over25", "home_goals", "away_goals"])
    rows_1x2 = plot_1x2(df, PLOTS / "market_cal_1x2.png")
    rows_ou = plot_ou(df, PLOTS / "market_cal_ou.png")
    plot_compare(df, PLOTS / "market_cal_compare.png")
    rows = rows_1x2 + rows_ou
    write_report(REPORTS / "market_calibration.md", rows, len(df))
    return {"n": len(df), "rows": rows}


if __name__ == "__main__":
    out = run_market_calibration()
    for r in out["rows"]:
        print(
            f"{r['market']:6s} {r['outcome']:12s} "
            f"decile_r={r['decile_corr']:.3f} cal_R2={r['decile_r2']:.3f} "
            f"Brier={r['brier']:.3f}"
        )
    print(f"Wrote {PLOTS}/market_cal_*.png and {REPORTS}/market_calibration.md")
