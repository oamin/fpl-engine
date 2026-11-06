"""Deadline-causal minutes for path FT planning.

At deadline ``d`` the information set is:

- **this GW** — oracle/actual minutes for GW ``d`` (team news in);
- **later GWs** — as-of rolling mean from history with ``GW < d`` only.

Future GW oracle minutes must not enter the plan until their own deadline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.markets.cascade_score_v2 import rolling_m_sigma
from src.markets import numagent_xp


def asof_minutes_by_element(
    merged: pd.DataFrame,
    *,
    before_gw: int,
    season: str,
    window: int = 3,
) -> dict[int, float]:
    """Rolling mean minutes from history strictly before ``before_gw``."""
    hist = numagent_xp.asof_share_history(
        merged, before_gw=int(before_gw), season=str(season)
    )
    out: dict[int, float] = {}
    if hist.empty or "element" not in hist.columns:
        return out
    for el in pd.to_numeric(hist["element"], errors="coerce").dropna().unique():
        m, _ = rolling_m_sigma(hist, int(el), window=int(window))
        out[int(el)] = float(m)
    return out


def oracle_minutes_by_element(merged: pd.DataFrame, gw: int) -> dict[int, float]:
    """Actual minutes in GW ``gw`` (deadline team-news proxy)."""
    if merged is None or merged.empty or "GW" not in merged.columns:
        return {}
    g = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
    if g.empty:
        return {}
    return {
        int(r.element): float(r.minutes or 0.0)
        for r in g.itertuples(index=False)
        if pd.notna(r.element)
    }


def minutes_vector_for_pool(
    pool: pd.DataFrame,
    by_el: dict[int, float],
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Align element→minutes onto ``pool`` row order."""
    n = len(pool)
    if fallback is None:
        base = np.zeros(n, dtype=float)
        if "m" in pool.columns:
            base = pd.to_numeric(pool["m"], errors="coerce").fillna(0.0).to_numpy(float)
        elif "xmi" in pool.columns:
            base = pd.to_numeric(pool["xmi"], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        base = np.asarray(fallback, dtype=float)
        if len(base) != n:
            raise ValueError(f"fallback length {len(base)} != pool {n}")
    els = pd.to_numeric(pool["element"], errors="coerce")
    out = base.copy()
    for i, el in enumerate(els):
        if pd.isna(el):
            continue
        if int(el) in by_el:
            out[i] = float(by_el[int(el)])
    return out


def pool_with_minutes(pool: pd.DataFrame, minutes: np.ndarray) -> pd.DataFrame:
    """Copy of ``pool`` with ``m`` / ``xmi`` / ``sigma_m`` set from ``minutes``."""
    out = pool.copy()
    m = np.asarray(minutes, dtype=float)
    if len(m) != len(out):
        raise ValueError(f"minutes length {len(m)} != pool {len(out)}")
    out["m"] = m
    out["xmi"] = m
    out["sigma_m"] = 0.0
    return out


def deadline_minute_views(
    pool: pd.DataFrame,
    merged: pd.DataFrame,
    *,
    deadline_gw: int,
    season: str,
) -> dict[str, Any]:
    """As-of and oracle minute vectors for one deadline (pool-aligned)."""
    asof = asof_minutes_by_element(
        merged, before_gw=int(deadline_gw), season=str(season)
    )
    oracle = oracle_minutes_by_element(merged, int(deadline_gw))
    asof_vec = minutes_vector_for_pool(pool, asof)
    # This-GW board: oracle where known, else as-of.
    this_vec = minutes_vector_for_pool(pool, oracle, fallback=asof_vec)
    return {
        "deadline_gw": int(deadline_gw),
        "asof": asof_vec,
        "this_gw": this_vec,
    }
