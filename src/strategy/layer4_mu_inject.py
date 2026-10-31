"""Inject external μ boards into Layer-4 without Odds/FD projection.

``solve_horizon_transfers`` is already column-agnostic (``mu_col``). The
priced factory ``build_mupost_priced_horizon`` builds t+1 via
``project_week_score_xp_v2``. Product centreline horizon (priced near +
Opta blind far) is ``src.strategy.l3_centreline_horizon.build_l3_centreline_horizon``
under ``protocol.l3_centreline_forecast``. Cascade / scenario paths should
**not** go through the priced-only factory — they pass frozen μ frames here,
then call L4 (or use the centreline factory).

Universe, prices, and club/position meta come from ``week0_pool`` only.
Missing players get μ = 0 (same convention as ``_align_to_week0``).

See ``reports/CASCADE_CONTRACT.md`` · ``reports/L3_CENTRELINE_FORECAST_LOCK.md``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd


def _element_key(frame: pd.DataFrame) -> pd.Series:
    if "element" in frame.columns:
        return pd.to_numeric(frame["element"], errors="coerce")
    if "player_id" in frame.columns:
        return pd.to_numeric(
            frame["player_id"].astype(str).str.split(":").str[-1],
            errors="coerce",
        )
    raise ValueError("μ frame needs element or player_id")


def _mu_series(frame: pd.DataFrame) -> pd.Series:
    for col in ("mu_post", "mu", "score_xp_v2"):
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    raise ValueError("μ frame needs mu_post, mu, or score_xp_v2")


def _se_series(frame: pd.DataFrame) -> pd.Series | None:
    if "xp_se" in frame.columns:
        return pd.to_numeric(frame["xp_se"], errors="coerce").fillna(0.0)
    if "se" in frame.columns:
        return pd.to_numeric(frame["se"], errors="coerce").fillna(0.0)
    return None


def mu_frame_to_week_pool(
    week0_pool: pd.DataFrame,
    mu_frame: pd.DataFrame,
    *,
    out_col: str = "mu_post",
    raw_col: str = "score_xp_v2",
) -> pd.DataFrame:
    """Map element→μ (+ optional SE) onto the week-0 player universe.

    Returns a copy of ``week0_pool`` with ``out_col`` / ``raw_col`` / ``xp_se``
    overwritten from ``mu_frame``. Players absent from ``mu_frame`` get 0.
    """
    if week0_pool.empty:
        raise ValueError("week0_pool is empty")
    if "player_id" not in week0_pool.columns and "element" not in week0_pool.columns:
        raise ValueError("week0_pool needs player_id or element")

    w0 = week0_pool.copy()
    if "element" not in w0.columns:
        w0["element"] = pd.to_numeric(
            w0["player_id"].astype(str).str.split(":").str[-1], errors="coerce"
        )

    mu_f = mu_frame.copy()
    mu_f["element"] = _element_key(mu_f)
    mu_f = mu_f.dropna(subset=["element"])
    mu_f["element"] = mu_f["element"].astype(int)
    mu_f["_mu"] = _mu_series(mu_f)
    se = _se_series(mu_f)
    if se is not None:
        mu_f["_se"] = se
    # last row wins on duplicate elements
    mu_f = mu_f.drop_duplicates("element", keep="last")

    by_el = mu_f.set_index("element")
    els = w0["element"].astype(int)
    w0[out_col] = els.map(lambda e: float(by_el["_mu"].get(int(e), 0.0))).astype(float)
    w0[raw_col] = w0[out_col]
    if "_se" in by_el.columns:
        w0["xp_se"] = els.map(lambda e: float(by_el["_se"].get(int(e), 0.0))).astype(float)
    elif "xp_se" not in w0.columns:
        w0["xp_se"] = 0.0
    w0["total_points"] = 0.0
    return w0.reset_index(drop=True)


def pools_from_mu_boards(
    week0_pool: pd.DataFrame,
    mu_boards: Sequence[pd.DataFrame] | Mapping[int, pd.DataFrame],
    *,
    out_col: str = "mu_post",
) -> list[pd.DataFrame]:
    """Build an L4 pool list from week-0 meta + per-step μ boards.

    ``mu_boards`` is either:

    - a sequence ``[μ_t, μ_t+1, …]`` (index 0 = decision week), or
    - a mapping ``{0: μ_t, 1: μ_t+1, …}`` (step offsets).

    Length-1 recovers an L3-shaped single-week board. Pass the result to
    ``solve_horizon_transfers(..., mu_col=out_col)``.
    """
    if isinstance(mu_boards, Mapping):
        if not mu_boards:
            raise ValueError("mu_boards is empty")
        steps = sorted(int(k) for k in mu_boards.keys())
        ordered = [mu_boards[k] for k in steps]
    else:
        ordered = list(mu_boards)
    if not ordered:
        raise ValueError("mu_boards is empty")

    pools: list[pd.DataFrame] = []
    for board in ordered:
        pools.append(
            mu_frame_to_week_pool(week0_pool, board, out_col=out_col)
        )
    return pools
