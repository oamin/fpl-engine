"""Causal Sharpe utility for the FT climb.

u = μ / (σ + 1), with μ = score_xp and σ the expanding residual
std of (total_points − score_xp) using only earlier gameweeks.
A missing σ is the position median of those earlier residuals, then 3.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1.0
LAMBDA_RISK = 0.25
SIGMA_LO = 0.5
SIGMA_HI = 8.0
MIN_PERIODS = 3
SIGMA_FILL = 3.0  # UNC_TAU


def add_causal_sharpe_u(
    df: pd.DataFrame,
    mu_col: str = "score_xp",
) -> pd.DataFrame:
    """Attach ``sigma_xp`` and ``score_u``. Later gameweeks cannot move earlier rows."""
    out = df.sort_values(["player_id", "gw"], kind="mergesort").copy()
    out["position"] = out["position"].astype(str).replace({"GK": "GKP"})
    mu = pd.to_numeric(out[mu_col], errors="coerce")
    pts = pd.to_numeric(out["total_points"], errors="coerce")
    resid = pts - mu
    prior = resid.groupby(out["player_id"], sort=False).shift(1)
    raw = prior.groupby(out["player_id"], sort=False).transform(
        lambda s: s.expanding(min_periods=MIN_PERIODS).std()
    )
    raw = pd.to_numeric(raw, errors="coerce")

    filled = pd.Series(np.nan, index=out.index, dtype=float)
    gw = out["gw"].astype(int)
    pos = out["position"]
    history: dict[str, list[float]] = {}
    for t in sorted(gw.unique()):
        med = {p: float(np.median(vals)) for p, vals in history.items() if vals}
        mask = gw == int(t)
        for idx in out.index[mask]:
            sigma = raw.at[idx]
            if pd.isna(sigma):
                sigma = med.get(pos.at[idx], SIGMA_FILL)
            filled.at[idx] = float(sigma)
        for idx in out.index[mask]:
            sigma = raw.at[idx]
            if pd.notna(sigma):
                history.setdefault(pos.at[idx], []).append(float(sigma))

    out["sigma_xp"] = filled.clip(SIGMA_LO, SIGMA_HI)
    out["score_u"] = mu.to_numpy(float) / (out["sigma_xp"].to_numpy(float) + EPS)
    out["score_risk"] = mu.to_numpy(float) - LAMBDA_RISK * out["sigma_xp"].to_numpy(float)
    return out
