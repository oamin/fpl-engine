"""Stage 33 — twelve player scores, fast XI only.

Locked with Gemini before the totals were read. No grid. None of these
scores take a free-transfer climb in this batch. A score is clearly
behind when it trails expected points by 100 or more on 2025/26, GW5–38.
Survivors are queued.

``within_pos`` rescales each position to mean 0 and sd 1 inside the
gameweek. That can pick a formation for tail shape rather than points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.stage_29_batch import KILL_GAP, fast_xi

FORMULAS = (
    "per_million",
    "minutes",
    "upside",
    "attack",
    "no_deduction",
    "agree_min",
    "agree_max",
    "within_pos",
    "premium",
    "starter",
    "split",
    "goals_tilt",
)


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def apply_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """Decision-time score. Missing inputs stay missing."""
    xp = _num(df, "score_xp")
    if formula == "per_million":
        price_m = _num(df, "value") / 10.0
        return xp / price_m.where(price_m > 0)
    if formula == "minutes":
        scale = (_num(df, "xmi") / 90.0).clip(lower=0.0, upper=1.0)
        return xp * scale
    if formula == "upside":
        return xp + 0.25 * _num(df, "sigma_xp")
    if formula == "attack":
        return _num(df, "xp_goals") + _num(df, "xp_assists")
    if formula == "no_deduction":
        return xp + _num(df, "xp_deductions")
    if formula == "agree_min":
        return np.minimum(xp, _num(df, "score_exp_points"))
    if formula == "agree_max":
        return np.maximum(xp, _num(df, "score_exp_points"))
    if formula == "within_pos":
        return df.groupby(["gw", "position"], sort=False)["score_xp"].transform(_zscore)
    if formula == "premium":
        return xp * xp.abs()
    if formula == "starter":
        scale = ((_num(df, "xmi") - 45.0) / 45.0).clip(lower=0.0, upper=1.0)
        return xp * scale
    if formula == "split":
        defensive = _num(df, "xp_cs") + _num(df, "xp_saves") + _num(df, "xp_appear")
        attacking = _num(df, "xp_goals") + _num(df, "xp_assists")
        pos = df["position"].astype(str)
        return defensive.where(pos.isin(["GKP", "DEF"]), attacking)
    if formula == "goals_tilt":
        return xp + _num(df, "xp_goals")
    raise ValueError(f"unknown formula: {formula}")


def _zscore(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    std = float(clean.std(ddof=0))
    if not np.isfinite(std) or std == 0.0:
        return pd.Series(0.0, index=values.index)
    return (clean - float(clean.mean())) / std


def attach_screen_scores(feat: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    out = feat.copy()
    cols = {"xp": "score_xp"}
    for formula in FORMULAS:
        column = f"screen__{formula}"
        out[column] = apply_formula(out, formula)
        cols[formula] = column
    return out, cols


def screen_table(weekly: pd.DataFrame, kill_gap: float = KILL_GAP) -> pd.DataFrame:
    totals = weekly.groupby("method", sort=False)["xi_points_cap"].sum()
    base = float(totals["xp"])
    rows = []
    for name, total in totals.items():
        delta = float(total) - base
        rows.append(
            {
                "candidate": name,
                "xi_points": float(total),
                "delta_vs_xp": delta,
                "killed": bool(name != "xp" and delta <= -float(kill_gap)),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["delta_vs_xp", "candidate"], ascending=[False, True])


def run_screen(feat: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    scored, cols = attach_screen_scores(feat)
    weekly = fast_xi(scored, gws, cols)
    return screen_table(weekly)
