"""Calibration of the published score against 2025/26 points.

No formula change. Gameweeks 5–38, players the climb can buy
(at least 3 prior appearances, and a minutes prior of at least 45).

Mean error can cancel. The table also records absolute error, root mean
square error, Pearson and Spearman correlation, and the mean actual points
in each decile of the predicted score. A component is left alone when its
mean error is inside 0.10 and removing it moves Spearman correlation with
actual points by less than 0.01.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.xp_engine import (
    MIN_HISTORY,
    add_market_pots,
    add_player_priors,
    compute_xp,
    load_joined,
)
from src.rules.fpl_2026 import (
    APPEARANCE_60,
    APPEARANCE_UNDER_60,
    ASSIST_POINTS,
    CS_POINTS,
    DEFCON_POINTS,
    DEFCON_THRESHOLD,
    GOAL_POINTS,
    GOALS_CONCEDED_PER_DEDUCTION,
    MINUTES_FOR_APPEARANCE_BONUS,
    SAVES_PER_POINT,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Predicted column, actual column. Deductions are stored as positive losses.
PAIRS = (
    ("appear", "xp_appear", "act_appear"),
    ("goals", "xp_goals", "act_goals"),
    ("assists", "xp_assists", "act_assists"),
    ("clean_sheets", "xp_cs", "act_cs"),
    ("defcon", "xp_defcon", "act_defcon"),
    ("saves", "xp_saves", "act_saves"),
    ("bonus_proxy", "xp_bps", "act_bonus"),
    ("goals_conceded", "xp_gc_loss", "act_gc"),
    ("cards", "xp_card_loss", "act_cards"),
)

BIAS_KEEP = 0.10
IC_KEEP = 0.01


def actual_components(df: pd.DataFrame) -> pd.DataFrame:
    """Official awards for the events on the row. Not the model's scale."""
    out = df.copy()
    minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
    pos = out["position"].astype(str)
    played = minutes > 0
    full = minutes >= MINUTES_FOR_APPEARANCE_BONUS
    out["act_appear"] = np.where(full, APPEARANCE_60, np.where(played, APPEARANCE_UNDER_60, 0))
    goal_pts = pos.map(GOAL_POINTS).fillna(0.0)
    out["act_goals"] = pd.to_numeric(out["goals"], errors="coerce").fillna(0.0) * goal_pts
    out["act_assists"] = pd.to_numeric(out["assists"], errors="coerce").fillna(0.0) * ASSIST_POINTS
    cs_flag = (pd.to_numeric(out["clean_sheets"], errors="coerce").fillna(0.0) > 0) & full
    out["act_cs"] = np.where(cs_flag, pos.map(CS_POINTS).fillna(0.0), 0.0)
    thr = pos.map(DEFCON_THRESHOLD)
    raw = pd.to_numeric(out.get("defcon_raw", out.get("defcon")), errors="coerce").fillna(0.0)
    hit = pos.isin(DEFCON_THRESHOLD) & full & (raw >= thr)
    out["act_defcon"] = np.where(hit, DEFCON_POINTS, 0.0)
    saves = pd.to_numeric(out["saves"], errors="coerce").fillna(0.0)
    out["act_saves"] = np.where(pos.eq("GKP"), np.floor(saves / SAVES_PER_POINT), 0.0)
    out["act_bonus"] = pd.to_numeric(out["bonus"], errors="coerce").fillna(0.0)
    conceded = pd.to_numeric(out["goals_conceded"], errors="coerce").fillna(0.0)
    gc_units = np.floor(conceded / GOALS_CONCEDED_PER_DEDUCTION)
    out["act_gc"] = np.where(pos.isin(["GKP", "DEF"]) & played, gc_units, 0.0)
    yellows = pd.to_numeric(out["yellow_cards"], errors="coerce").fillna(0.0)
    reds = pd.to_numeric(out["red_cards"], errors="coerce").fillna(0.0)
    out["act_cards"] = yellows + 3.0 * reds
    return out


def fit_row(pred: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Error and rank agreement. Empty input returns zeros and a zero count."""
    p = pd.to_numeric(pred, errors="coerce")
    a = pd.to_numeric(actual, errors="coerce")
    ok = p.notna() & a.notna()
    p, a = p[ok], a[ok]
    n = int(len(p))
    if n == 0:
        return {"n": 0, "bias": 0.0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}
    err = p - a
    pearson = float(p.corr(a, method="pearson")) if n > 2 and p.std() > 0 and a.std() > 0 else 0.0
    spearman = float(p.corr(a, method="spearman")) if n > 2 and p.nunique() > 1 else 0.0
    return {
        "n": n,
        "bias": float(err.mean()),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "pearson": pearson,
        "spearman": spearman,
    }


def deciles(pred: pd.Series, actual: pd.Series) -> pd.DataFrame:
    """Mean predicted score and mean actual points in each decile of the prediction."""
    frame = pd.DataFrame({"pred": pd.to_numeric(pred, errors="coerce"), "actual": pd.to_numeric(actual, errors="coerce")})
    frame = frame.dropna()
    if frame.empty:
        return pd.DataFrame(columns=["decile", "n", "mean_pred", "mean_actual"])
    frame["decile"] = pd.qcut(frame["pred"], 10, labels=False, duplicates="drop") + 1
    out = (
        frame.groupby("decile", as_index=False)
        .agg(n=("actual", "size"), mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
    )
    return out


def ablation(df: pd.DataFrame, score: str, component: str, actual: str) -> float:
    """How much Spearman correlation falls when the component is removed."""
    full = fit_row(df[score], df[actual])["spearman"]
    reduced = fit_row(df[score] - df[component], df[actual])["spearman"]
    return float(full - reduced)


def decision_pool(feat: pd.DataFrame) -> pd.DataFrame:
    """Gameweeks 5–38, the same buy rule as the climb."""
    xmi = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0.0)
    prior = pd.to_numeric(feat["n_prior"], errors="coerce").fillna(0.0)
    return feat.loc[(feat["gw"] >= 5) & (feat["gw"] <= 38) & (prior >= MIN_HISTORY) & (xmi >= 45.0)].copy()


def build_table(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Position fit, component fit, and deciles."""
    pool = actual_components(pool)
    positions = ["ALL", "GKP", "DEF", "MID", "FWD"]
    pos_rows = []
    for pos in positions:
        block = pool if pos == "ALL" else pool.loc[pool["position"] == pos]
        row = fit_row(block["xp"], block["total_points"])
        row["position"] = pos
        pos_rows.append(row)
    comp_rows = []
    for name, pred_col, act_col in PAIRS:
        row = fit_row(pool[pred_col], pool[act_col])
        row["component"] = name
        row["delta_spearman"] = ablation(pool, "xp", pred_col, "total_points")
        row["keep"] = int(abs(row["bias"]) < BIAS_KEEP and abs(row["delta_spearman"]) < IC_KEEP)
        comp_rows.append(row)
    return pd.DataFrame(pos_rows), pd.DataFrame(comp_rows), deciles(pool["xp"], pool["total_points"])


def load_pool() -> pd.DataFrame:
    feat = compute_xp(add_player_priors(add_market_pots(load_joined())))
    return decision_pool(feat)


def write_report(path: Path, positions: pd.DataFrame, components: pd.DataFrame, bins: pd.DataFrame, tie: float) -> None:
    lines = [
        "# Score calibration, 2025/26",
        "",
        "Published `compute_xp`, Gameweeks 5–38, players with at least 3 prior appearances and a minutes prior of at least 45. No coefficient was changed.",
        "",
        f"The official awards reconstructed from the sheet (appearance, goals, assists, clean sheets, defensive contributions, saves, bonus, goals conceded, cards) differ from `total_points` by **{tie:.3f}** points per row on average. Own goals, penalty saves, and missed penalties are not on this reconstruction.",
        "",
        "A goalkeeper goal is worth 6 in the official total and 10 inside `xp_goals`. That scale was left as it is so earlier xP totals stay reproducible.",
        "",
        "## Score against actual points",
        "",
        "| position | n | bias | MAE | RMSE | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in positions.itertuples(index=False):
        lines.append(
            f"| {row.position} | {int(row.n)} | {row.bias:+.3f} | {row.mae:.3f} | {row.rmse:.3f} | {row.pearson:.3f} | {row.spearman:.3f} |"
        )
    lines += [
        "",
        "Bias is predicted score minus actual points. A positive bias is a score that sits above the points.",
        "",
        "## Components",
        "",
        "Bias is the predicted component minus the official points from that event. `delta Spearman` is how much rank agreement with total points falls when the component is removed from the score. Keep means the mean error is inside 0.10 and the rank change is under 0.01.",
        "",
        "| component | n | bias | MAE | RMSE | Spearman vs its own points | delta Spearman | keep |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in components.itertuples(index=False):
        flag = "yes" if int(row.keep) else "no"
        lines.append(
            f"| {row.component} | {int(row.n)} | {row.bias:+.3f} | {row.mae:.3f} | {row.rmse:.3f} | {row.spearman:.3f} | {row.delta_spearman:+.3f} | {flag} |"
        )
    lines += [
        "",
        "## Deciles of the predicted score",
        "",
        "| decile | n | mean predicted | mean actual |",
        "|---:|---:|---:|---:|",
    ]
    for row in bins.itertuples(index=False):
        lines.append(f"| {int(row.decile)} | {int(row.n)} | {row.mean_pred:.3f} | {row.mean_actual:.3f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    pool = load_pool()
    pool = actual_components(pool)
    built = (
        pool["act_appear"]
        + pool["act_goals"]
        + pool["act_assists"]
        + pool["act_cs"]
        + pool["act_defcon"]
        + pool["act_saves"]
        + pool["act_bonus"]
        - pool["act_gc"]
        - pool["act_cards"]
    )
    tie = float((built - pool["total_points"]).mean())
    positions, components, bins = build_table(pool)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    positions.to_csv(PROCESSED / "stage_39_calibration_position.csv", index=False)
    components.to_csv(PROCESSED / "stage_39_calibration_component.csv", index=False)
    bins.to_csv(PROCESSED / "stage_39_calibration_decile.csv", index=False)
    write_report(REPORTS / "stage_39_calibration.md", positions, components, bins, tie)
    return {"n": int(len(pool)), "tie": tie}


if __name__ == "__main__":
    print(run())
