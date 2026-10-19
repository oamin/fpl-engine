"""Player G/A finishing scales (item 4 candidate).

Baseline score uses ``e_goals = share_xG · λ`` (and the assist analogue).
This module attaches leakage-free player finishing factors:

  finish_g = clip( (Σ goals + κ) / (Σ xG + κ) , lo, hi )   # shift-1 expanding
  finish_a = clip( (Σ assists + κ) / (Σ xA + κ) , lo, hi )

so the candidate rate is ``share · λ · finish``. Team pot / shares stay as
exposure; the finish factor is the player skill residual over xG/xA.

Not promoted until the gated screen clears.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FINISH_PRIOR = 2.0
FINISH_LO = 0.65
FINISH_HI = 1.40


def add_finish_scales(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``finish_g`` / ``finish_a`` from shift-1 expanding conversion."""
    out = df.copy()
    if "assists" not in out.columns:
        out["assists"] = 0.0
    goals = pd.to_numeric(out.get("goals", 0.0), errors="coerce").fillna(0.0)
    assists = pd.to_numeric(out["assists"], errors="coerce").fillna(0.0)
    xg = pd.to_numeric(out.get("xG", 0.0), errors="coerce").fillna(0.0)
    xa = pd.to_numeric(out.get("xA", 0.0), errors="coerce").fillna(0.0)
    out["_g"] = goals
    out["_a"] = assists
    out["_xg"] = xg
    out["_xa"] = xa

    order = out.sort_values(["player_id", "gw", "date"], kind="mergesort").index.to_numpy()
    ordered = out.loc[order]
    g = ordered.groupby("player_id", sort=False)
    # Shift-1 expanding sums (no current row).
    sum_g = g["_g"].cumsum().shift(1)
    sum_a = g["_a"].cumsum().shift(1)
    sum_xg = g["_xg"].cumsum().shift(1)
    sum_xa = g["_xa"].cumsum().shift(1)

    fin_g = (sum_g.fillna(0.0) + FINISH_PRIOR) / (sum_xg.fillna(0.0) + FINISH_PRIOR)
    fin_a = (sum_a.fillna(0.0) + FINISH_PRIOR) / (sum_xa.fillna(0.0) + FINISH_PRIOR)
    fin_g = fin_g.clip(FINISH_LO, FINISH_HI)
    fin_a = fin_a.clip(FINISH_LO, FINISH_HI)

    out.loc[order, "finish_g"] = fin_g.to_numpy(float)
    out.loc[order, "finish_a"] = fin_a.to_numpy(float)
    out["finish_g"] = pd.to_numeric(out["finish_g"], errors="coerce").fillna(1.0)
    out["finish_a"] = pd.to_numeric(out["finish_a"], errors="coerce").fillna(1.0)
    return out.drop(columns=["_g", "_a", "_xg", "_xa"])
