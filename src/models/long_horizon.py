"""Long-horizon backtest — do priors hold when outcomes are aggregated?

Same predictors as before, but y = sum/mean over the next H gameweeks
(not only GW t). If signal is real but match-noisy, corr should rise with H.

Horizons: 1, 3, 5, 8.

Writes:
  data/processed/long_horizon_backtest.csv
  data/plots/long_horizon_corr.png
  reports/stage_12_long_horizon.md
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
HORIZONS = (1, 3, 5, 8)
MIN_HISTORY = 3
ROLL = 3
MIN_MINUTES = 60.0
DEFCON_THRESH = {"DEF": 10.0, "MID": 12.0, "FWD": 12.0}


def _roll_mean(s: pd.Series, window: int = ROLL) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=1).mean()


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(float)
    pred = pred[mask].astype(float)
    n = int(y.size)
    out = {
        "n": float(n),
        "corr": float("nan"),
        "spearman": float("nan"),
        "r2": float("nan"),
        "mae": float("nan"),
        "decile_corr": float("nan"),
        "top_bottom": float("nan"),
        "mean_y": float("nan"),
        "mean_pred": float("nan"),
    }
    if n < 40:
        return out
    err = pred - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    out["mae"] = float(np.mean(np.abs(err)))
    out["r2"] = 0.0 if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    out["mean_y"] = float(y.mean())
    out["mean_pred"] = float(pred.mean())
    if np.std(pred) > 0 and np.std(y) > 0:
        out["corr"] = float(np.corrcoef(pred, y)[0, 1])
        out["spearman"] = float(pd.Series(pred).corr(pd.Series(y), method="spearman"))
    tmp = pd.DataFrame({"x": pred, "y": y})
    try:
        tmp["bin"] = pd.qcut(tmp["x"], 10, duplicates="drop")
        g = tmp.groupby("bin", observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        if len(g) >= 4 and g["x"].std() > 0 and g["y"].std() > 0:
            out["decile_corr"] = float(g["x"].corr(g["y"]))
        if len(g) >= 2:
            out["top_bottom"] = float(g["y"].iloc[-1] - g["y"].iloc[0])
    except ValueError:
        pass
    return out


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
            "total_points": pd.to_numeric(raw["total_points"], errors="coerce").fillna(0.0),
            "goals": pd.to_numeric(raw["goals_scored"], errors="coerce").fillna(0.0),
            "assists": pd.to_numeric(raw["assists"], errors="coerce").fillna(0.0),
            "cs": pd.to_numeric(raw["clean_sheets"], errors="coerce").fillna(0.0),
            "xG": pd.to_numeric(raw["expected_goals"], errors="coerce").fillna(0.0),
            "xA": pd.to_numeric(raw["expected_assists"], errors="coerce").fillna(0.0),
            "xGC": pd.to_numeric(raw["expected_goals_conceded"], errors="coerce").fillna(0.0),
            "defcon": pd.to_numeric(raw["defensive_contribution"], errors="coerce").fillna(0.0),
            "starts": pd.to_numeric(raw["starts"], errors="coerce").fillna(0.0),
        }
    )
    df = df.dropna(subset=["position", "gw"]).copy()
    df["gw"] = df["gw"].astype(int)
    df = df.sort_values(["player_id", "gw"], kind="mergesort")
    df = df.drop_duplicates(["player_id", "gw"], keep="first")
    thr = df["position"].map(DEFCON_THRESH)
    df["defcon_hit"] = (
        df["position"].isin(DEFCON_THRESH)
        & (df["minutes"] >= MIN_MINUTES)
        & (df["defcon"] >= thr)
    ).astype(float)
    df["gi"] = df["goals"] + df["assists"]
    df["played60"] = (df["minutes"] >= MIN_MINUTES).astype(float)
    return df.reset_index(drop=True)


def add_priors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    out["n_prior"] = g.cumcount()
    for col in (
        "minutes",
        "total_points",
        "goals",
        "assists",
        "gi",
        "cs",
        "xG",
        "xA",
        "defcon_hit",
        "played60",
    ):
        out[f"roll3_{col}"] = g[col].transform(_roll_mean)
        pos_mean = out.groupby("position")[col].transform("mean")
        out[f"roll3_{col}"] = out[f"roll3_{col}"].fillna(pos_mean)
    # Expanding rates (longer memory)
    for col in ("goals", "assists", "gi", "cs", "xG", "xA", "defcon_hit", "total_points", "minutes"):
        out[f"exp_{col}"] = g[col].transform(lambda s: s.shift(1).expanding().mean())
        pos_mean = out.groupby("position")[col].transform("mean")
        out[f"exp_{col}"] = out[f"exp_{col}"].fillna(pos_mean)
    out["xmi"] = out["roll3_minutes"]
    return out


def add_forward_windows(df: pd.DataFrame) -> pd.DataFrame:
    """For each horizon H: sum of y over GWs [t, t+H) — requires H future rows present."""
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    g = out.groupby("player_id", sort=False)
    targets = ("total_points", "goals", "assists", "gi", "cs", "defcon_hit", "minutes", "played60")
    for H in HORIZONS:
        for col in targets:
            # sum of current + next H-1 (inclusive window starting at t)
            # use reverse rolling trick: rolling on reversed then reverse back
            fwd = (
                g[col]
                .transform(lambda s: s.iloc[::-1].rolling(H, min_periods=H).sum().iloc[::-1])
            )
            out[f"fwd{H}_{col}"] = fwd
            out[f"fwd{H}_mean_{col}"] = fwd / H
    # Only keep rows where longest horizon is complete
    out["has_fwd8"] = out["fwd8_total_points"].notna()
    return out


def evaluate(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Regulars: prior history + (optional) currently in form as starter proxy via xmi."""
    rows: list[dict[str, Any]] = []
    base = df.loc[(df["n_prior"] >= MIN_HISTORY) & df["has_fwd8"]].copy()

    # Universes
    universes = {
        "ALL": base,
        "regulars_xmi45": base.loc[base["xmi"] >= 45],
        "regulars_xmi60": base.loc[base["xmi"] >= 60],
    }

    # Predictor → target family
    # For horizon H, compare prior rate * H vs fwd sum (scale), or prior vs fwd mean (rank)
    specs = [
        # minutes / points
        ("xmi", "minutes", "xMi → minutes"),
        ("xmi", "total_points", "xMi → points"),
        ("roll3_total_points", "total_points", "roll3 pts → points"),
        ("exp_total_points", "total_points", "exp pts → points"),
        # G/A
        ("roll3_xG", "goals", "roll3 xG → goals"),
        ("exp_xG", "goals", "exp xG → goals"),
        ("roll3_goals", "goals", "roll3 goals → goals"),
        ("exp_goals", "goals", "exp goals → goals"),
        ("roll3_xG", "gi", "roll3 xG → G+A"),
        ("exp_xG", "gi", "exp xG → G+A"),
        ("roll3_xA", "assists", "roll3 xA → assists"),
        # CS (among DEF/GKP scopes below)
        ("roll3_cs", "cs", "roll3 CS → CS"),
        ("exp_cs", "cs", "exp CS → CS"),
        # DefCon
        ("roll3_defcon_hit", "defcon_hit", "roll3 DefCon hit → hit"),
        ("exp_defcon_hit", "defcon_hit", "exp DefCon hit → hit"),
    ]

    pos_scopes = {
        "ALL": None,
        "MID+FWD": ["MID", "FWD"],
        "GKP+DEF": ["GKP", "DEF"],
        "DEF+MID": ["DEF", "MID"],
        "GKP": ["GKP"],
        "DEF": ["DEF"],
        "MID": ["MID"],
        "FWD": ["FWD"],
    }

    for univ_name, univ in universes.items():
        for H in HORIZONS:
            for pred, target, label in specs:
                ycol = f"fwd{H}_{target}"
                # Use mean target so pred scale (rate) is comparable across H
                ycol_mean = f"fwd{H}_mean_{target}"
                for scope_name, positions in pos_scopes.items():
                    # Restrict which specs apply to which scopes
                    if target in ("goals", "assists", "gi") and scope_name not in (
                        "ALL",
                        "MID+FWD",
                        "MID",
                        "FWD",
                    ):
                        continue
                    if target == "cs" and scope_name not in ("ALL", "GKP+DEF", "GKP", "DEF"):
                        continue
                    if target == "defcon_hit" and scope_name not in (
                        "ALL",
                        "DEF+MID",
                        "DEF",
                        "MID",
                    ):
                        continue
                    if target in ("minutes", "total_points") and scope_name not in (
                        "ALL",
                        "MID+FWD",
                        "GKP+DEF",
                        "MID",
                        "FWD",
                        "DEF",
                        "GKP",
                    ):
                        continue

                    chunk = univ if positions is None else univ.loc[univ["position"].isin(positions)]
                    if len(chunk) < 40:
                        continue
                    # Rank metrics on per-GW mean over horizon (longevity of rate)
                    m = _metrics(
                        chunk[ycol_mean].to_numpy(float),
                        chunk[pred].to_numpy(float),
                    )
                    # Also sum-scale: pred * H vs fwd sum (calibration of level)
                    m_sum = _metrics(
                        chunk[ycol].to_numpy(float),
                        (chunk[pred] * H).to_numpy(float),
                    )
                    rows.append(
                        {
                            "universe": univ_name,
                            "scope": scope_name,
                            "horizon": H,
                            "label": label,
                            "predictor": pred,
                            "target": target,
                            "corr_mean": m["corr"],
                            "spearman_mean": m["spearman"],
                            "decile_corr_mean": m["decile_corr"],
                            "top_bottom_mean": m["top_bottom"],
                            "r2_mean": m["r2"],
                            "corr_sum_scaled": m_sum["corr"],
                            "r2_sum_scaled": m_sum["r2"],
                            "mae_sum_scaled": m_sum["mae"],
                            "n": m["n"],
                            "mean_y_per_gw": m["mean_y"],
                            "mean_pred": m["mean_pred"],
                        }
                    )
    return rows


def plot_corr_curves(rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(
        "Long-horizon: predictor ↔ next-H-GW mean outcome (regulars xmi≥45)",
        fontsize=12,
    )
    panels = [
        (0, 0, "MID+FWD", "roll3 xG → goals", "Goals"),
        (0, 1, "GKP+DEF", "roll3 CS → CS", "Clean sheets"),
        (1, 0, "DEF+MID", "exp DefCon hit → hit", "DefCon hit"),
        (1, 1, "ALL", "xMi → points", "Points"),
    ]
    univ = "regulars_xmi45"
    for r, c, scope, label, title in panels:
        ax = axes[r][c]
        sub = df.loc[
            (df["universe"] == univ) & (df["scope"] == scope) & (df["label"] == label)
        ].sort_values("horizon")
        # also plot a baseline comparator if exists
        ax.plot(sub["horizon"], sub["corr_mean"], "o-", color="crimson", lw=2, label=label)
        # comparators
        comps = {
            "Goals": ["exp xG → goals", "roll3 goals → goals"],
            "Clean sheets": ["exp CS → CS"],
            "DefCon hit": ["roll3 DefCon hit → hit"],
            "Points": ["roll3 pts → points", "exp pts → points", "xMi → minutes"],
        }
        colors = ["steelblue", "seagreen", "gray"]
        for lab, col in zip(comps.get(title, []), colors):
            s2 = df.loc[
                (df["universe"] == univ) & (df["scope"] == scope) & (df["label"] == lab)
            ].sort_values("horizon")
            if s2.empty and title == "Points" and lab == "xMi → minutes":
                s2 = df.loc[
                    (df["universe"] == univ)
                    & (df["scope"] == "ALL")
                    & (df["label"] == lab)
                ].sort_values("horizon")
            if not s2.empty:
                ax.plot(s2["horizon"], s2["corr_mean"], "o-", color=col, lw=1.5, label=lab)
        ax.set_title(title)
        ax.set_xlabel("horizon H (GWs)")
        ax.set_ylabel("Pearson corr (pred vs mean y over H)")
        ax.set_xticks(list(HORIZONS))
        ax.set_ylim(-0.05, 0.85)
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=7, loc="lower right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    lines = [
        "# Stage 12 — Long-horizon prior longevity",
        "",
        "Question: do the same priors **rank / track rates over the next H gameweeks**, "
        "not only the immediate next match?",
        "",
        "For each player-GW t (with ≥3 prior GWs and a full next-8 window):",
        "- predictor = leakage-free roll3 / expanding rate at t",
        "- outcome = **mean** of the target over GWs `[t, t+H)`",
        "- horizons H ∈ {1, 3, 5, 8}",
        "",
        "Universe focus: **regulars with xMi ≥ 45** (known-ish minutes).",
        "",
    ]

    focus = [
        ("MID+FWD", "roll3 xG → goals"),
        ("MID+FWD", "exp xG → goals"),
        ("MID+FWD", "roll3 goals → goals"),
        ("GKP+DEF", "roll3 CS → CS"),
        ("GKP+DEF", "exp CS → CS"),
        ("DEF+MID", "exp DefCon hit → hit"),
        ("DEF+MID", "roll3 DefCon hit → hit"),
        ("ALL", "xMi → points"),
        ("ALL", "xMi → minutes"),
        ("ALL", "roll3 pts → points"),
        ("ALL", "exp pts → points"),
    ]
    lines += [
        "## Corr vs horizon (regulars xmi≥45)",
        "",
        "| scope | pair | H=1 | H=3 | H=5 | H=8 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope, label in focus:
        vals = []
        for H in HORIZONS:
            hit = df.loc[
                (df["universe"] == "regulars_xmi45")
                & (df["scope"] == scope)
                & (df["label"] == label)
                & (df["horizon"] == H)
            ]
            if hit.empty:
                vals.append("—")
            else:
                vals.append(f"{float(hit.iloc[0]['corr_mean']):.3f}")
        lines.append(f"| {scope} | {label} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Read",
        "",
        "- If corr **rises** with H → real rate signal buried in match noise (longevity OK).",
        "- If corr **flat/falls** → prior doesn’t persist; not useful even as a medium-term rate.",
        "- xMi → minutes should stay high; xMi → points may rise slowly via appearance.",
        "",
        "## Plots",
        "",
        "- `data/plots/long_horizon_corr.png`",
        "",
        "## Output",
        "",
        "- `data/processed/long_horizon_backtest.csv`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    raw = load_gw()
    feat = add_priors(raw)
    feat = add_forward_windows(feat)
    rows = evaluate(feat)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PROCESSED / "long_horizon_backtest.csv", index=False)
    plot_corr_curves(rows, PLOTS / "long_horizon_corr.png")
    write_report(REPORTS / "stage_12_long_horizon.md", rows)
    return {"rows": rows, "n": len(feat)}


if __name__ == "__main__":
    out = run()
    df = pd.DataFrame(out["rows"])
    print("regulars xmi≥45 — corr(pred, mean y over next H)\n")
    focus = [
        ("MID+FWD", "roll3 xG → goals"),
        ("MID+FWD", "exp xG → goals"),
        ("GKP+DEF", "roll3 CS → CS"),
        ("GKP+DEF", "exp CS → CS"),
        ("DEF+MID", "exp DefCon hit → hit"),
        ("ALL", "xMi → minutes"),
        ("ALL", "xMi → points"),
        ("ALL", "exp pts → points"),
    ]
    for scope, label in focus:
        xs = []
        for H in HORIZONS:
            hit = df.loc[
                (df.universe == "regulars_xmi45")
                & (df.scope == scope)
                & (df.label == label)
                & (df.horizon == H)
            ]
            xs.append(f"H{H}={float(hit.iloc[0]['corr_mean']):.3f}" if not hit.empty else f"H{H}=—")
        print(f"  {scope:8s}  {label:28s}  " + "  ".join(xs))
    print(f"\nWrote {REPORTS}/stage_12_long_horizon.md")
