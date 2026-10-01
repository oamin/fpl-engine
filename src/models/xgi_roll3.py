"""Roll-3 xG / xA / xGI → next-match goals / assists / GI / points.

Leakage-free: prior 3 GWs only (shift+rolling mean), including 0-minute rows.

Writes:
  data/processed/xgi_roll3_backtest.csv
  data/plots/xgi_roll3_backtest.png
  data/plots/xgi_roll3_by_position.png
  reports/stage_8_xgi_roll3.md
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
MIN_HISTORY = 3
ROLL = 3


def _roll_mean(s: pd.Series, window: int = ROLL) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(float)
    pred = pred[mask].astype(float)
    n = int(y.size)
    if n < 20:
        return {
            "n": float(n),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "corr": float("nan"),
            "mean_y": float("nan"),
            "mean_pred": float("nan"),
        }
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    corr = (
        float(np.corrcoef(pred, y)[0, 1])
        if np.std(pred) > 0 and np.std(y) > 0
        else float("nan")
    )
    return {
        "n": float(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": r2,
        "corr": corr,
        "mean_y": float(y.mean()),
        "mean_pred": float(pred.mean()),
    }


def load_gw(season: str = "2025_26") -> pd.DataFrame:
    raw = pd.read_csv(CACHE / f"merged_gw_{season}.csv")
    df = pd.DataFrame(
        {
            "player_id": raw["element"].astype(str),
            "player_name": raw["name"].astype(str),
            "team": raw["team"].astype(str),
            "position": raw["position"].map(lambda p: POS_MAP.get(str(p).upper())),
            "gw": pd.to_numeric(raw.get("GW", raw.get("round")), errors="coerce"),
            "minutes": pd.to_numeric(raw["minutes"], errors="coerce").fillna(0.0),
            "goals": pd.to_numeric(raw["goals_scored"], errors="coerce").fillna(0.0),
            "assists": pd.to_numeric(raw["assists"], errors="coerce").fillna(0.0),
            "xG": pd.to_numeric(raw["expected_goals"], errors="coerce").fillna(0.0),
            "xA": pd.to_numeric(raw["expected_assists"], errors="coerce").fillna(0.0),
            "xGI_vaastav": pd.to_numeric(
                raw["expected_goal_involvements"], errors="coerce"
            ).fillna(0.0),
            "total_points": pd.to_numeric(raw["total_points"], errors="coerce").fillna(
                0.0
            ),
        }
    )
    df = df.dropna(subset=["position", "gw"]).copy()
    df["gw"] = df["gw"].astype(int)
    df["GI"] = df["goals"] + df["assists"]
    df["xGI"] = df["xG"] + df["xA"]  # consistent sum; vaastav field kept for check
    df = df.sort_values(["player_id", "gw"], kind="mergesort")
    df = df.drop_duplicates(["player_id", "gw"], keep="first")
    return df.reset_index(drop=True)


def add_roll3(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    out["n_prior"] = g.cumcount()

    for col in ("xG", "xA", "xGI", "goals", "assists", "GI", "total_points", "minutes"):
        out[f"roll3_{col}"] = g[col].transform(_roll_mean)
        # cold start: position mean of that channel
        pos_mean = out.groupby("position")[col].transform("mean")
        out[f"roll3_{col}"] = out[f"roll3_{col}"].fillna(pos_mean)

    return out


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Pairs: predictor → target."""
    sub_all = df.loc[df["n_prior"] >= MIN_HISTORY].copy()
    sub_60 = sub_all.loc[sub_all["minutes"] >= 60].copy()

    pairs = [
        ("roll3_xG", "goals", "xG → goals"),
        ("roll3_xA", "assists", "xA → assists"),
        ("roll3_xGI", "GI", "xGI → G+A"),
        ("roll3_xGI", "total_points", "xGI → points"),
        ("roll3_goals", "goals", "prior goals → goals"),
        ("roll3_assists", "assists", "prior assists → assists"),
        ("roll3_GI", "GI", "prior G+A → G+A"),
        ("roll3_total_points", "total_points", "prior points → points"),
        ("roll3_minutes", "minutes", "roll3 mins → mins (ref)"),
    ]

    rows: list[dict[str, Any]] = []
    for pred_col, y_col, label in pairs:
        for scope_name, chunk in (
            ("ALL", sub_all),
            ("mins>=60", sub_60),
        ):
            m = _metrics(
                chunk[y_col].to_numpy(float), chunk[pred_col].to_numpy(float)
            )
            rows.append(
                {
                    "label": label,
                    "predictor": pred_col,
                    "target": y_col,
                    "filter": scope_name,
                    "scope": "ALL",
                    **m,
                }
            )
            for pos in ("GKP", "DEF", "MID", "FWD"):
                pos_chunk = chunk.loc[chunk["position"] == pos]
                if len(pos_chunk) < 30:
                    continue
                mp = _metrics(
                    pos_chunk[y_col].to_numpy(float),
                    pos_chunk[pred_col].to_numpy(float),
                )
                rows.append(
                    {
                        "label": label,
                        "predictor": pred_col,
                        "target": y_col,
                        "filter": scope_name,
                        "scope": pos,
                        **mp,
                    }
                )
    return sub_all, rows


def plot_main(sub: pd.DataFrame, out: Path) -> None:
    """xG→goals, xA→assists, xGI→GI, xGI→points for mins>=60 MID+FWD emphasis — show all."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Roll-3 xG/xA/xGI → next-match output (all positions, all rows)", fontsize=13)
    panels = [
        (0, 0, "roll3_xG", "goals", "roll3 xG → goals"),
        (0, 1, "roll3_xA", "assists", "roll3 xA → assists"),
        (1, 0, "roll3_xGI", "GI", "roll3 xGI → G+A"),
        (1, 1, "roll3_xGI", "total_points", "roll3 xGI → points"),
    ]
    for r, c, px, py, title in panels:
        ax = axes[r][c]
        x = sub[px].to_numpy(float)
        y = sub[py].to_numpy(float)
        m = _metrics(y, x)
        ax.scatter(x, y, s=8, alpha=0.12, edgecolors="none", color="steelblue")
        # Decile means
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        try:
            tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
            g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
            ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=6, label="decile")
            # y ≈ x for event rates (goals scale)
            if py in ("goals", "assists", "GI"):
                lo = float(min(g["x"].min(), g["y"].min()))
                hi = float(max(g["x"].max(), g["y"].max()))
                ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1, label="y=x")
        except ValueError:
            pass
        ax.set_title(f"{title}\nR²={m['r2']:.3f} corr={m['corr']:.3f} MAE={m['mae']:.3f}")
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        ax.legend(fontsize=7, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_by_position(sub: pd.DataFrame, out: Path) -> None:
    """xGI → GI for MID/FWD (and DEF) with mins>=60."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    fig.suptitle("Roll-3 xGI → next G+A (minutes ≥ 60)", fontsize=13)
    chunk0 = sub.loc[sub["minutes"] >= 60]
    for ax, pos in zip(axes, ("DEF", "MID", "FWD")):
        chunk = chunk0.loc[chunk0["position"] == pos]
        x = chunk["roll3_xGI"].to_numpy(float)
        y = chunk["GI"].to_numpy(float)
        m = _metrics(y, x)
        ax.scatter(x, y, s=12, alpha=0.25, edgecolors="none", color="steelblue")
        tmp = pd.DataFrame({"x": x, "y": y}).dropna()
        try:
            tmp["bin"] = pd.qcut(tmp["x"], 8, duplicates="drop")
            g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
            ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=6)
            lo = float(min(g["x"].min(), g["y"].min()))
            hi = float(max(g["x"].max(), g["y"].max()))
            ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
        except ValueError:
            pass
        ax.set_title(f"{pos}  R²={m['r2']:.3f}  corr={m['corr']:.3f}")
        ax.set_xlabel("roll3 xGI")
        ax.set_ylabel("next G+A")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]], n_raw: int) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage 8 — Roll-3 xG / xA / xGI → next-match output",
        "",
        "Prior **3-GW rolling mean** of xG, xA, xGI (and naive goal/assist rates) "
        "vs realised next-GW events. Leakage-free (`shift` + rolling).",
        "",
        f"- Raw player-GW rows: **{n_raw}**",
        f"- Backtest requires ≥ {MIN_HISTORY} prior GWs",
        "",
        "## Headline (scope=ALL)",
        "",
        "### All rows (incl. 0 minutes)",
        "",
        "| pair | n | MAE | R² | corr | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[(df["scope"] == "ALL") & (df["filter"] == "ALL")].itertuples():
        lines.append(
            f"| {r.label} | {int(r.n)} | {r.mae:.3f} | {r.r2:.3f} | "
            f"{r.corr:.3f} | {r.mean_y:.3f} | {r.mean_pred:.3f} |"
        )
    lines += [
        "",
        "### Minutes ≥ 60 only",
        "",
        "| pair | n | MAE | R² | corr | mean y | mean pred |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[(df["scope"] == "ALL") & (df["filter"] == "mins>=60")].itertuples():
        lines.append(
            f"| {r.label} | {int(r.n)} | {r.mae:.3f} | {r.r2:.3f} | "
            f"{r.corr:.3f} | {r.mean_y:.3f} | {r.mean_pred:.3f} |"
        )

    lines += [
        "",
        "## MID / FWD — xGI → G+A (mins ≥ 60)",
        "",
        "| pos | n | MAE | R² | corr |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in df.loc[
        (df["label"] == "xGI → G+A")
        & (df["filter"] == "mins>=60")
        & (df["scope"].isin(["MID", "FWD", "DEF"]))
    ].itertuples():
        lines.append(
            f"| {r.scope} | {int(r.n)} | {r.mae:.3f} | {r.r2:.3f} | {r.corr:.3f} |"
        )

    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/xgi_roll3_backtest.png`",
        "- `data/plots/xgi_roll3_by_position.png`",
        "",
        "## Output",
        "",
        "- `data/processed/xgi_roll3_backtest.csv`",
        "",
        "## Read",
        "",
        "- Compare **xG → goals** vs **prior goals → goals** (does xG beat counting?).",
        "- High corr + low R² is common for rare events; decile curves matter more than R².",
        "- If xGI ≉ prior GI for next G+A, prefer the simpler counting rate.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    raw = load_gw()
    feat = add_roll3(raw)
    sub, rows = evaluate(feat)
    pd.DataFrame(rows).to_csv(PROCESSED / "xgi_roll3_backtest.csv", index=False)
    # Also persist feature sample for reuse
    keep = [
        "player_id",
        "player_name",
        "team",
        "position",
        "gw",
        "minutes",
        "goals",
        "assists",
        "GI",
        "xG",
        "xA",
        "xGI",
        "total_points",
        "n_prior",
        "roll3_xG",
        "roll3_xA",
        "roll3_xGI",
        "roll3_goals",
        "roll3_assists",
        "roll3_GI",
        "roll3_total_points",
        "roll3_minutes",
    ]
    feat[keep].to_csv(PROCESSED / "xgi_roll3_features.csv", index=False)
    plot_main(sub, PLOTS / "xgi_roll3_backtest.png")
    plot_by_position(sub, PLOTS / "xgi_roll3_by_position.png")
    write_report(REPORTS / "stage_8_xgi_roll3.md", rows, len(raw))
    return {"rows": rows, "n": len(raw), "n_bt": len(sub)}


if __name__ == "__main__":
    out = run()
    df = pd.DataFrame(out["rows"])
    print(f"raw={out['n']} backtest={out['n_bt']}")
    show = df.loc[(df.scope == "ALL") & (df["filter"].isin(["ALL", "mins>=60"]))]
    for filt in ("ALL", "mins>=60"):
        print(f"\n=== filter={filt} ===")
        for r in show.loc[show["filter"] == filt].itertuples():
            print(
                f"  {r.label:28s}  R²={r.r2:.3f}  corr={r.corr:.3f}  "
                f"MAE={r.mae:.3f}  ȳ={r.mean_y:.3f}  pred̄={r.mean_pred:.3f}"
            )
    print(f"\nWrote {REPORTS}/stage_8_xgi_roll3.md")
