"""Stage 10 — Gate: roll3 xMi → next-match points.

Does the only reliable forecast (minutes) move FPL points on its own?
Compare vs roll3 prior points and a trivial linear map xMi → points.

Writes:
  data/processed/xmi_points_gate.csv
  data/plots/xmi_points_gate.png
  data/plots/xmi_points_gate_by_pos.png
  reports/stage_10_xmi_points_gate.md
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
            "decile_corr": float("nan"),
            "mean_y": float("nan"),
            "mean_pred": float("nan"),
            "top_bottom": float("nan"),
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
    decile_corr = float("nan")
    top_bottom = float("nan")
    tmp = pd.DataFrame({"x": pred, "y": y})
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        if len(g) >= 4 and g["x"].std() > 0 and g["y"].std() > 0:
            decile_corr = float(g["x"].corr(g["y"]))
        if len(g) >= 2:
            top_bottom = float(g["y"].iloc[-1] - g["y"].iloc[0])
    except ValueError:
        pass
    return {
        "n": float(n),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": r2,
        "corr": corr,
        "decile_corr": decile_corr,
        "mean_y": float(y.mean()),
        "mean_pred": float(pred.mean()),
        "top_bottom": top_bottom,
    }


def _ols_fit_predict(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """In-sample OLS y ~ a + b x (scale calibration only; ranking unchanged)."""
    mask = np.isfinite(x) & np.isfinite(y)
    pred = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 20:
        return pred, {"intercept": float("nan"), "slope": float("nan")}
    X = np.column_stack([np.ones(int(mask.sum())), x[mask]])
    coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    pred[mask] = coef[0] + coef[1] * x[mask]
    return pred, {"intercept": float(coef[0]), "slope": float(coef[1])}


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
            "total_points": pd.to_numeric(raw["total_points"], errors="coerce").fillna(
                0.0
            ),
            "starts": pd.to_numeric(raw["starts"], errors="coerce").fillna(0.0),
        }
    )
    df = df.dropna(subset=["position", "gw"]).copy()
    df["gw"] = df["gw"].astype(int)
    df["started"] = (df["starts"] >= 1).astype(float)
    df = df.sort_values(["player_id", "gw"], kind="mergesort")
    df = df.drop_duplicates(["player_id", "gw"], keep="first")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    out["n_prior"] = g.cumcount()
    out["xmi"] = g["minutes"].transform(_roll_mean)  # locked roll3
    out["roll3_points"] = g["total_points"].transform(_roll_mean)
    out["roll3_start_rate"] = g["started"].transform(_roll_mean)
    # Cold start → position means
    for col, src in (
        ("xmi", "minutes"),
        ("roll3_points", "total_points"),
        ("roll3_start_rate", "started"),
    ):
        pos_mean = out.groupby("position")[src].transform("mean")
        out[col] = out[col].fillna(pos_mean)
    # ppm (points per minute) from prior history only — expanding
    prior_pts = g["total_points"].transform(lambda s: s.shift(1).expanding().sum())
    prior_mins = g["minutes"].transform(lambda s: s.shift(1).expanding().sum())
    out["prior_ppm"] = prior_pts / prior_mins.replace(0, np.nan)
    pos_ppm = out.groupby("position").apply(
        lambda d: d["total_points"].sum() / max(d["minutes"].sum(), 1.0),
        include_groups=False,
    )
    out["prior_ppm"] = out["prior_ppm"].fillna(out["position"].map(pos_ppm))
    out["xmi_x_ppm"] = out["xmi"] * out["prior_ppm"]
    return out


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    sub = df.loc[df["n_prior"] >= MIN_HISTORY].copy()
    # Calibrated xMi → points (in-sample scale only)
    cal_pred, cal_coef = _ols_fit_predict(
        sub["xmi"].to_numpy(float), sub["total_points"].to_numpy(float)
    )
    sub["xmi_cal_points"] = cal_pred
    ppm_pred, ppm_coef = _ols_fit_predict(
        sub["xmi_x_ppm"].to_numpy(float), sub["total_points"].to_numpy(float)
    )
    sub["xmi_ppm_cal"] = ppm_pred

    pairs = [
        ("xmi", "total_points", "xMi → points (raw mins scale)"),
        ("xmi_cal_points", "total_points", "xMi → points (OLS calibrated)"),
        ("xmi_x_ppm", "total_points", "xMi×prior_ppm → points"),
        ("xmi_ppm_cal", "total_points", "xMi×ppm OLS → points"),
        ("roll3_points", "total_points", "roll3 points → points"),
        ("roll3_start_rate", "total_points", "roll3 start rate → points"),
        ("xmi", "minutes", "xMi → minutes (ref)"),
    ]

    rows: list[dict[str, Any]] = []
    scopes = [
        ("ALL", sub),
        ("mins>0", sub.loc[sub["minutes"] > 0]),
        ("mins>=60", sub.loc[sub["minutes"] >= 60]),
        ("xmi>=45", sub.loc[sub["xmi"] >= 45]),  # predicted regulars
    ]
    for pred, y, label in pairs:
        for filt_name, chunk in scopes:
            if len(chunk) < 30:
                continue
            m = _metrics(chunk[y].to_numpy(float), chunk[pred].to_numpy(float))
            rows.append(
                {
                    "label": label,
                    "predictor": pred,
                    "target": y,
                    "filter": filt_name,
                    "scope": "ALL",
                    **m,
                }
            )
            if filt_name in ("ALL", "mins>=60") and y == "total_points":
                for pos in ("GKP", "DEF", "MID", "FWD"):
                    pc = chunk.loc[chunk["position"] == pos]
                    if len(pc) < 30:
                        continue
                    mp = _metrics(pc[y].to_numpy(float), pc[pred].to_numpy(float))
                    rows.append(
                        {
                            "label": label,
                            "predictor": pred,
                            "target": y,
                            "filter": filt_name,
                            "scope": pos,
                            **mp,
                        }
                    )
    return sub, rows, {**cal_coef, **{f"ppm_{k}": v for k, v in ppm_coef.items()}}


def _panel(ax, x, y, title, xlabel, ylabel="next points") -> None:
    m = _metrics(y, x)
    ax.scatter(x, y, s=8, alpha=0.1, edgecolors="none", color="steelblue")
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        ax.plot(g["x"], g["y"], "o-", color="crimson", lw=2, ms=6, label="decile")
    except ValueError:
        pass
    ax.axhline(float(np.nanmean(y)), color="gray", ls=":", lw=1)
    ax.set_title(
        f"{title}\nR²={m['r2']:.3f} corr={m['corr']:.3f} "
        f"Δtop-bot={m['top_bottom']:.2f}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_main(sub: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("Gate: xMi → next-match points (all rows)", fontsize=13)
    panels = [
        (0, 0, "xmi", "xMi → points", "xMi (roll3 mins)"),
        (0, 1, "xmi_cal_points", "xMi OLS → points", "calibrated points"),
        (1, 0, "roll3_points", "roll3 points → points", "roll3 prior points"),
        (1, 1, "xmi_x_ppm", "xMi×ppm → points", "xMi × prior ppm"),
    ]
    for r, c, col, title, xlab in panels:
        _panel(
            axes[r][c],
            sub[col].to_numpy(float),
            sub["total_points"].to_numpy(float),
            title,
            xlab,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_by_pos(sub: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle("xMi (OLS-calibrated) → points by position", fontsize=13)
    for ax, pos in zip(axes.ravel(), ("GKP", "DEF", "MID", "FWD")):
        chunk = sub.loc[sub["position"] == pos]
        _panel(
            ax,
            chunk["xmi_cal_points"].to_numpy(float),
            chunk["total_points"].to_numpy(float),
            pos,
            "xMi-cal points",
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    n_raw: int,
    coef: dict[str, float],
) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage 10 — Gate: xMi → next-match points",
        "",
        "Does **roll3 xMi** alone forecast next-GW `total_points`? "
        "Compared to roll3 prior points and xMi×prior points-per-minute.",
        "",
        f"- Raw rows: **{n_raw}**",
        f"- Backtest: prior GWs ≥ {MIN_HISTORY}",
        f"- OLS xMi→pts: intercept={coef.get('intercept', float('nan')):.4f}, "
        f"slope={coef.get('slope', float('nan')):.4f}",
        "",
        "## Headline (scope=ALL)",
        "",
    ]
    for filt in ("ALL", "mins>0", "mins>=60", "xmi>=45"):
        chunk = df.loc[(df["scope"] == "ALL") & (df["filter"] == filt) & (df["target"] == "total_points")]
        if chunk.empty:
            continue
        lines += [
            f"### Filter: `{filt}`",
            "",
            "| pair | n | MAE | R² | corr | decile_corr | Δ top−bot | mean y |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in chunk.itertuples():
            lines.append(
                f"| {r.label} | {int(r.n)} | {r.mae:.3f} | {r.r2:.3f} | "
                f"{r.corr:.3f} | {r.decile_corr:.3f} | {r.top_bottom:.3f} | "
                f"{r.mean_y:.3f} |"
            )
        lines.append("")

    # Reference minutes row
    ref = df.loc[
        (df["scope"] == "ALL") & (df["filter"] == "ALL") & (df["label"] == "xMi → minutes (ref)")
    ]
    if not ref.empty:
        r = ref.iloc[0]
        lines += [
            "## Minutes reference (sanity)",
            "",
            f"- xMi → minutes: R²={float(r['r2']):.3f}, corr={float(r['corr']):.3f}, "
            f"MAE={float(r['mae']):.2f}",
            "",
        ]

    lines += [
        "## By position — xMi OLS → points (ALL rows)",
        "",
        "| pos | n | MAE | R² | corr | Δ top−bot |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in df.loc[
        (df["label"] == "xMi → points (OLS calibrated)")
        & (df["filter"] == "ALL")
        & (df["scope"].isin(["GKP", "DEF", "MID", "FWD"]))
    ].itertuples():
        lines.append(
            f"| {r.scope} | {int(r.n)} | {r.mae:.3f} | {r.r2:.3f} | "
            f"{r.corr:.3f} | {r.top_bottom:.3f} |"
        )

    lines += [
        "",
        "## Plots",
        "",
        "- `data/plots/xmi_points_gate.png`",
        "- `data/plots/xmi_points_gate_by_pos.png`",
        "",
        "## Output",
        "",
        "- `data/processed/xmi_points_gate.csv`",
        "",
        "## Gate read",
        "",
        "- If xMi→points corr ≪ xMi→minutes, minutes forecast well but **points ceiling is thin**.",
        "- If roll3 points ≫ xMi for points, historic scoring rate dominates appearance.",
        "- If both weak once mins≥60, appearance is the whole story and event noise remains.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    raw = load_gw()
    feat = add_features(raw)
    sub, rows, coef = evaluate(feat)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PROCESSED / "xmi_points_gate.csv", index=False)
    plot_main(sub, PLOTS / "xmi_points_gate.png")
    plot_by_pos(sub, PLOTS / "xmi_points_gate_by_pos.png")
    write_report(REPORTS / "stage_10_xmi_points_gate.md", rows, len(raw), coef)
    return {"rows": rows, "n": len(raw), "n_bt": len(sub), "coef": coef}


if __name__ == "__main__":
    out = run()
    df = pd.DataFrame(out["rows"])
    print(f"raw={out['n']} backtest={out['n_bt']}")
    print(f"OLS xMi→pts: a={out['coef']['intercept']:.4f} b={out['coef']['slope']:.4f}")
    for filt in ("ALL", "mins>=60", "xmi>=45"):
        print(f"\n=== filter={filt} ===")
        show = df.loc[
            (df.scope == "ALL")
            & (df["filter"] == filt)
            & (df.target == "total_points")
        ]
        for r in show.itertuples():
            print(
                f"  {r.label:36s}  R²={r.r2:.3f}  corr={r.corr:.3f}  "
                f"dcorr={r.decile_corr:.3f}  Δ={r.top_bottom:.2f}  MAE={r.mae:.3f}"
            )
    print(f"\nWrote {REPORTS}/stage_10_xmi_points_gate.md")
