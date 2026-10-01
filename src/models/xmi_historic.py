"""Historic expected minutes (xMi) with leakage-free backtest.

Uses Vaastav merged_gw including 0-minute rows. Features are rolling /
expanding stats from *prior* GWs only.

Predictors (simple, interpretable):
  - **primary:** roll3 mean minutes (``xmi``)
  - comparators: roll2 / roll5 / roll8, expanding, start/bench mixture, blend

Backtest: predict minutes at GW t from history < t; MAE / RMSE / R²,
plus start classification from P(start).

Writes:
  data/processed/xmi_historic.csv
  data/plots/xmi_backtest.png
  data/plots/xmi_by_position.png
  reports/stage_7_xmi_historic.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

POS_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
ROLL_WINDOWS = (2, 3, 5, 8)
MIN_HISTORY = 3  # prior GWs before scoring a prediction
PRIMARY_WINDOW = 3


def load_gw_minutes(season: str = "2025_26") -> pd.DataFrame:
    path = CACHE / f"merged_gw_{season}.csv"
    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "player_id": raw["element"].astype(str),
            "player_name": raw["name"].astype(str),
            "team": raw["team"].astype(str),
            "position": raw["position"].map(lambda p: POS_MAP.get(str(p).upper())),
            "gw": pd.to_numeric(raw.get("GW", raw.get("round")), errors="coerce"),
            "minutes": pd.to_numeric(raw["minutes"], errors="coerce").fillna(0.0),
            "starts": pd.to_numeric(raw["starts"], errors="coerce").fillna(0.0),
            "kickoff_time": raw["kickoff_time"].astype(str),
        }
    )
    df = df.dropna(subset=["position", "gw"]).copy()
    df["gw"] = df["gw"].astype(int)
    df["started"] = (df["starts"] >= 1).astype(float)
    df["played"] = (df["minutes"] > 0).astype(float)
    df["full_90"] = (df["minutes"] >= 90).astype(float)
    df["early_sub"] = ((df["started"] == 1) & (df["minutes"] < 70)).astype(float)
    df["late_sub"] = (
        (df["started"] == 1) & (df["minutes"] >= 70) & (df["minutes"] < 90)
    ).astype(float)
    # One row per player-GW (merged_gw can rarely duplicate; keep first).
    df = df.sort_values(["player_id", "gw", "kickoff_time"], kind="mergesort")
    df = df.drop_duplicates(["player_id", "gw"], keep="first")
    return df.reset_index(drop=True)


def _roll_mean(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def add_historic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)

    out["n_prior"] = g.cumcount()
    out["prior_minutes_mean"] = g["minutes"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    out["prior_start_rate"] = g["started"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    out["prior_play_rate"] = g["played"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    for w in ROLL_WINDOWS:
        out[f"roll{w}_minutes"] = g["minutes"].transform(lambda s: _roll_mean(s, w))
        out[f"roll{w}_start_rate"] = g["started"].transform(lambda s: _roll_mean(s, w))

    # Conditional prior means: mins|start, mins|bench, early-sub rate|start.
    start_vals: list[pd.Series] = []
    bench_vals: list[pd.Series] = []
    early_vals: list[pd.Series] = []
    for _, grp in out.groupby("player_id", sort=False):
        started = grp["started"].to_numpy(float)
        mins = grp["minutes"].to_numpy(float)
        early = grp["early_sub"].to_numpy(float)
        n = len(grp)
        mins_start = np.full(n, np.nan)
        mins_bench = np.full(n, np.nan)
        early_rate = np.full(n, np.nan)
        s_num = s_den = 0.0
        b_num = b_den = 0.0
        e_num = e_den = 0.0
        for i in range(n):
            mins_start[i] = (s_num / s_den) if s_den > 0 else np.nan
            mins_bench[i] = (b_num / b_den) if b_den > 0 else np.nan
            early_rate[i] = (e_num / e_den) if e_den > 0 else np.nan
            if started[i] >= 1:
                s_num += float(mins[i])
                s_den += 1.0
                e_num += float(early[i])
                e_den += 1.0
            else:
                b_num += float(mins[i])
                b_den += 1.0
        start_vals.append(pd.Series(mins_start, index=grp.index))
        bench_vals.append(pd.Series(mins_bench, index=grp.index))
        early_vals.append(pd.Series(early_rate, index=grp.index))

    out["prior_mins_if_start"] = pd.concat(start_vals).sort_index()
    out["prior_mins_if_bench"] = pd.concat(bench_vals).sort_index()
    out["prior_early_sub_rate"] = pd.concat(early_vals).sort_index()

    # Position cold-start fills
    pos_min = out.groupby("position")["minutes"].transform("mean")
    pos_start = out.groupby("position")["started"].transform("mean")
    out["prior_minutes_mean"] = out["prior_minutes_mean"].fillna(pos_min)
    out["prior_start_rate"] = out["prior_start_rate"].fillna(pos_start)
    out["prior_play_rate"] = out["prior_play_rate"].fillna(pos_start)
    for w in ROLL_WINDOWS:
        out[f"roll{w}_minutes"] = out[f"roll{w}_minutes"].fillna(out["prior_minutes_mean"])
        out[f"roll{w}_start_rate"] = out[f"roll{w}_start_rate"].fillna(
            out["prior_start_rate"]
        )
    out["prior_mins_if_start"] = out["prior_mins_if_start"].fillna(80.0)
    out["prior_mins_if_bench"] = out["prior_mins_if_bench"].fillna(5.0)
    out["prior_early_sub_rate"] = out["prior_early_sub_rate"].fillna(0.15)

    # Mixture / blend kept as comparators; canonical xMi = 3-GW rolling mean.
    p_s = out["roll3_start_rate"].clip(0, 1)
    out["xmi_mixture"] = (
        p_s * out["prior_mins_if_start"] + (1.0 - p_s) * out["prior_mins_if_bench"]
    )
    out["xmi_roll2"] = out["roll2_minutes"]
    out["xmi_roll3"] = out["roll3_minutes"]
    out["xmi_roll5"] = out["roll5_minutes"]
    out["xmi_roll8"] = out["roll8_minutes"]
    out["xmi_expanding"] = out["prior_minutes_mean"]
    out["xmi_blend"] = 0.5 * out["xmi_roll3"] + 0.5 * out["xmi_mixture"]
    out["xmi"] = out["xmi_roll3"]  # locked primary
    return out


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(float)
    pred = pred[mask].astype(float)
    if y.size < 10:
        return {"n": float(y.size), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "corr": float("nan")}
    err = pred - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else float("nan")
    return {
        "n": float(y.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": r2,
        "corr": corr,
        "mean_y": float(y.mean()),
        "mean_pred": float(pred.mean()),
    }


def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score models on rows with enough history."""
    sub = df.loc[df["n_prior"] >= MIN_HISTORY].copy()
    models = [
        "xmi",
        "xmi_roll2",
        "xmi_roll3",
        "xmi_roll5",
        "xmi_roll8",
        "xmi_expanding",
        "xmi_mixture",
        "xmi_blend",
    ]
    rows: list[dict[str, Any]] = []
    y = sub["minutes"].to_numpy(float)
    for name in models:
        m = _metrics(y, sub[name].to_numpy(float))
        rows.append({"model": name, "scope": "ALL", **m})
        for pos in ("GKP", "DEF", "MID", "FWD"):
            chunk = sub.loc[sub["position"] == pos]
            mp = _metrics(chunk["minutes"].to_numpy(float), chunk[name].to_numpy(float))
            rows.append({"model": name, "scope": pos, **mp})

    # Start classification from roll3_start_rate (aligned with primary window)
    p_start = sub["roll3_start_rate"].to_numpy(float)
    started = sub["started"].to_numpy(float)
    # Accuracy at 0.5 threshold
    pred_start = (p_start >= 0.5).astype(float)
    mask = np.isfinite(p_start) & np.isfinite(started)
    acc = float((pred_start[mask] == started[mask]).mean()) if mask.any() else float("nan")
    # Brier
    brier = float(np.mean((p_start[mask] - started[mask]) ** 2)) if mask.any() else float("nan")
    rows.append(
        {
            "model": "roll3_start_rate",
            "scope": "ALL_start_clf",
            "n": float(mask.sum()),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "corr": float("nan"),
            "accuracy": acc,
            "brier": brier,
            "mean_y": float(started[mask].mean()) if mask.any() else float("nan"),
            "mean_pred": float(p_start[mask].mean()) if mask.any() else float("nan"),
        }
    )
    return sub, rows


def plot_backtest(sub: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    fig.suptitle("Historic xMi backtest — 3-GW rolling mean", fontsize=13)

    ax = axes[0]
    ax.scatter(
        sub["xmi"],
        sub["minutes"],
        s=8,
        alpha=0.15,
        edgecolors="none",
        color="steelblue",
    )
    ax.plot([0, 90], [0, 90], color="gray", ls="--", lw=1)
    m = _metrics(sub["minutes"].to_numpy(float), sub["xmi"].to_numpy(float))
    ax.set_xlabel("xMi (roll3)")
    ax.set_ylabel("realised minutes")
    ax.set_title("Primary: 3-GW rolling mean")
    ax.text(
        0.03,
        0.97,
        f"n={int(m['n'])}\nMAE={m['mae']:.1f}\nRMSE={m['rmse']:.1f}\n"
        f"R²={m['r2']:.3f}\ncorr={m['corr']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )

    ax = axes[1]
    tmp = sub[["xmi", "minutes"]].dropna().copy()
    try:
        tmp["bin"] = pd.qcut(tmp["xmi"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(
            x=("xmi", "mean"), y=("minutes", "mean"), n=("minutes", "count")
        )
        ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=7, label="decile mean")
        ax.plot([0, 90], [0, 90], color="gray", ls="--", lw=1, label="y = x")
        dcorr = float(g["x"].corr(g["y"]))
    except ValueError:
        dcorr = float("nan")
    ax.set_xlabel("xMi (roll3)")
    ax.set_ylabel("mean realised minutes")
    ax.set_title(f"Decile calibration (r={dcorr:.3f})")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_by_position(sub: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("xMi (roll3) by position — predicted vs realised", fontsize=13)
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        chunk = sub.loc[sub["position"] == pos]
        ax.scatter(
            chunk["xmi"],
            chunk["minutes"],
            s=10,
            alpha=0.2,
            edgecolors="none",
            color="steelblue",
        )
        ax.plot([0, 90], [0, 90], color="gray", ls="--", lw=1)
        m = _metrics(chunk["minutes"].to_numpy(float), chunk["xmi"].to_numpy(float))
        ax.set_title(f"{pos}  MAE={m['mae']:.1f}  R²={m['r2']:.3f}  corr={m['corr']:.3f}")
        ax.set_xlabel("xMi (roll3)")
        ax.set_ylabel("minutes")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, metrics: list[dict[str, Any]], n_raw: int) -> None:
    lines = [
        "# Stage 7 — Historic xMi backtest",
        "",
        "**Canonical xMi = 3-GW rolling mean of minutes** (history only).",
        "LLM / team-news overlay is deferred; keep backtesting other point drivers first.",
        "",
        "Includes 0-minute GW rows from Vaastav. Features use prior GWs only.",
        "",
        f"- Raw player-GW rows: **{n_raw}**",
        f"- Backtest rows (prior GWs ≥ {MIN_HISTORY}): see tables",
        "",
        "## Models",
        "",
        "| name | definition |",
        "|---|---|",
        "| `xmi` / `xmi_roll3` | **primary** — mean minutes over last 3 GWs |",
        "| `xmi_roll2` | comparator (lowest MAE; more twitchy) |",
        "| `xmi_roll5` | comparator |",
        "| `xmi_roll8` | comparator |",
        "| `xmi_expanding` | comparator |",
        "| `xmi_mixture` | comparator — P₃(start)·E[mins\\|start] + (1−P₃)·E[mins\\|bench] |",
        "| `xmi_blend` | comparator — 0.5·roll3 + 0.5·mixture |",
        "",
        "## Overall minutes backtest",
        "",
        "| model | n | MAE | RMSE | R² | corr | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics:
        if r["scope"] != "ALL":
            continue
        if r["model"] == "xmi_roll3":
            continue  # identical to xmi; avoid duplicate row
        lines.append(
            f"| {r['model']} | {int(r['n'])} | {r['mae']:.2f} | {r['rmse']:.2f} | "
            f"{r['r2']:.3f} | {r['corr']:.3f} | {r['mean_y']:.1f} | {r['mean_pred']:.1f} |"
        )

    lines += [
        "",
        "## Primary (`xmi` = roll3) by position",
        "",
        "| pos | n | MAE | RMSE | R² | corr |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics:
        if r["model"] == "xmi" and r["scope"] in ("GKP", "DEF", "MID", "FWD"):
            lines.append(
                f"| {r['scope']} | {int(r['n'])} | {r['mae']:.2f} | {r['rmse']:.2f} | "
                f"{r['r2']:.3f} | {r['corr']:.3f} |"
            )

    start_row = next((r for r in metrics if r["scope"] == "ALL_start_clf"), None)
    if start_row:
        lines += [
            "",
            "## Start classification (`roll3_start_rate` ≥ 0.5)",
            "",
            f"- n={int(start_row['n'])}",
            f"- accuracy=**{start_row.get('accuracy', float('nan')):.3f}**",
            f"- Brier=**{start_row.get('brier', float('nan')):.3f}**",
            f"- base start rate={start_row.get('mean_y', float('nan')):.3f}",
            "",
        ]

    lines += [
        "## Plots",
        "",
        "- `data/plots/xmi_backtest.png`",
        "- `data/plots/xmi_by_position.png`",
        "",
        "## Output",
        "",
        "- `data/processed/xmi_historic.csv` — use column **`xmi`**",
        "",
        "## Decision",
        "",
        "- Locked **roll3** as xMi (better than roll5; less twitchy than roll2).",
        "- Do **not** add LLM/news until other scoring channels are backtested historically.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    raw = load_gw_minutes()
    feat = add_historic_features(raw)
    sub, metrics = backtest(feat)
    # Persist full feature table (including cold-start rows)
    keep = [
        "player_id",
        "player_name",
        "team",
        "position",
        "gw",
        "minutes",
        "started",
        "played",
        "n_prior",
        "roll2_minutes",
        "roll3_minutes",
        "roll5_minutes",
        "roll8_minutes",
        "roll3_start_rate",
        "roll5_start_rate",
        "prior_start_rate",
        "prior_mins_if_start",
        "prior_mins_if_bench",
        "prior_early_sub_rate",
        "xmi",
        "xmi_roll2",
        "xmi_roll3",
        "xmi_roll5",
        "xmi_roll8",
        "xmi_expanding",
        "xmi_mixture",
        "xmi_blend",
    ]
    feat[keep].to_csv(PROCESSED / "xmi_historic.csv", index=False)
    plot_backtest(sub, PLOTS / "xmi_backtest.png")
    plot_by_position(sub, PLOTS / "xmi_by_position.png")
    write_report(REPORTS / "stage_7_xmi_historic.md", metrics, len(raw))
    return {"metrics": metrics, "n_raw": len(raw), "n_bt": len(sub)}


if __name__ == "__main__":
    out = run()
    print(f"Rows: raw={out['n_raw']} backtest={out['n_bt']}")
    for r in out["metrics"]:
        if r["scope"] == "ALL" and r["model"] in (
            "xmi",
            "xmi_roll2",
            "xmi_roll5",
            "xmi_roll8",
            "xmi_expanding",
            "xmi_mixture",
            "xmi_blend",
        ):
            print(
                f"  {r['model']:16s}  MAE={r['mae']:.2f}  R²={r['r2']:.3f}  "
                f"corr={r['corr']:.3f}"
            )
        if r["scope"] == "ALL_start_clf":
            print(
                f"  start_clf        acc={r['accuracy']:.3f}  Brier={r['brier']:.3f}"
            )
    print(f"Wrote {REPORTS}/stage_7_xmi_historic.md")
