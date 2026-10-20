"""L4 μ_post boards with expanding vs rolling-window history.

Expanding arm = ``build_mupost_priced_horizon`` (week-t = L3 board).
Window arm W: rates / priors / minutes and cohort moments on
``GW ∈ [t−W, t)`` (fallback: all ``GW < t``), then project t:t+2 onto FD
pots and constant-K shrink. Week-t μ is the windowed projection (aligned
onto the L3 player universe for prices/eligibility).

  python -m src.strategy.layer4_mupost_window --season 2025-26 --gw 15 --window 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.markets.numagent_xp_pilot import _asof_alphas
from src.rules.fpl_2026 import normalize_position
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer4_bookie_path import BOOKIE_DEPTH, _fd_index, _load_merged_enriched
from src.strategy.layer4_mupost_priced import (
    MupostPricedHorizon,
    _align_to_week0,
    _attach_store_ids,
    build_mupost_priced_horizon,
    project_week_score_xp_v2,
)
from src.strategy.shrinkage import CohortMoments, estimate_cohort_moments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

WINDOWS = (3, 5)


@dataclass(frozen=True)
class WindowMoments:
    moments: dict[str, CohortMoments]
    n_hist_gw: int
    window: int | None  # None = expanding


def hist_window(merged: pd.DataFrame, decision_gw: int, window: int | None) -> pd.DataFrame:
    """As-of history: expanding (``window is None``) or last ``window`` GWs."""
    past = merged.loc[merged["GW"] < int(decision_gw)]
    if window is None or past.empty:
        return past
    gws = sorted(int(g) for g in past["GW"].unique())
    keep = set(gws[-int(window) :])
    return past.loc[past["GW"].isin(keep)].copy()


def moments_from_v2(
    v2: pd.DataFrame,
    decision_gw: int,
    *,
    window: int | None,
) -> WindowMoments:
    """Cohort moments from closed score_xp_v2 rows strictly before ``decision_gw``."""
    hist = v2.loc[v2["gw"] < int(decision_gw)].copy()
    if hist.empty:
        return WindowMoments({}, 0, window)
    if window is not None:
        gws = sorted(int(g) for g in hist["gw"].unique())
        keep = set(gws[-int(window) :])
        hist = hist.loc[hist["gw"].isin(keep)]
    n_gw = int(hist["gw"].nunique()) if not hist.empty else 0
    moments: dict[str, CohortMoments] = {}
    for p, g in hist.groupby("position"):
        moments[str(normalize_position(p) or p)] = estimate_cohort_moments(
            g["score_xp_v2"].to_numpy(float),
            g["xp_se"].to_numpy(float),
            g["total_points"].to_numpy(float),
        )
    return WindowMoments(moments, n_gw, window)


def build_mupost_windowed_horizon(
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    v2_season: pd.DataFrame,
    window: int,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> MupostPricedHorizon:
    """Project t:t+2 from windowed history; shrink with windowed cohort moments."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if merged is None:
        merged = _load_merged_enriched(season)
    if fd is None:
        fd = _fd_index(season)
    if prices is None:
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)

    w0_meta = week0_pool.copy()
    if "mu_post" not in w0_meta.columns:
        raise ValueError("week0_pool needs mu_post (universe / prices)")
    if "element" not in w0_meta.columns and "player_id" in w0_meta.columns:
        w0_meta["element"] = pd.to_numeric(
            w0_meta["player_id"].astype(str).str.split(":").str[-1], errors="coerce"
        )

    hist = hist_window(merged, decision_gw, window)
    wm = moments_from_v2(v2_season, decision_gw, window=window)
    moments = wm.moments or None
    alpha_g, alpha_a = _asof_alphas(hist) if not hist.empty else _asof_alphas(
        merged.loc[merged["GW"] < int(decision_gw)]
    )
    avail = set(int(g) for g in merged["GW"].unique())

    pools: list[pd.DataFrame] = []
    priced_gws: list[int] = []
    n_fx = 0
    for step in range(0, BOOKIE_DEPTH):
        tgt = int(decision_gw) + step
        if tgt not in avail:
            continue
        raw, n = project_week_score_xp_v2(
            hist,
            merged,
            fd,
            decision_gw=decision_gw,
            target_gw=tgt,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
        )
        if raw.empty:
            continue
        n_fx += n
        board = _attach_store_ids(raw, prices, decision_gw, season)
        if moments:
            board, _ = attach_mu_post(
                board,
                mu_col="score_xp_v2",
                se_col="xp_se",
                mode="none",
                moments_by_pos=moments,
                out_col="mu_post",
            )
        else:
            board = board.copy()
            board["mu_post"] = pd.to_numeric(board["score_xp_v2"], errors="coerce").fillna(0.0)
        pools.append(_align_to_week0(w0_meta, board))
        priced_gws.append(tgt)

    if not pools:
        # Degenerate: keep L3 week0 only
        pools = [w0_meta]
        priced_gws = [int(decision_gw)]

    return MupostPricedHorizon(
        season=season,
        decision_gw=int(decision_gw),
        pools=pools,
        priced_gws=tuple(priced_gws),
        n_projected_fixtures=n_fx,
        meta={
            "window": int(window),
            "n_hist_gw": wm.n_hist_gw,
            "alpha_g": alpha_g,
            "alpha_a": alpha_a,
            "H": len(pools),
            "path": "windowed",
        },
    )


def build_horizon_for_arm(
    arm: str,
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    moments_expanding: dict[str, CohortMoments] | None,
    v2_season: pd.DataFrame,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
) -> MupostPricedHorizon:
    if arm == "l4_exp":
        hz = build_mupost_priced_horizon(
            season,
            decision_gw,
            week0_pool=week0_pool,
            moments=moments_expanding,
            merged=merged,
            fd=fd,
            prices=prices,
        )
        meta = {**hz.meta, "path": "expanding", "window": None}
        return MupostPricedHorizon(
            season=hz.season,
            decision_gw=hz.decision_gw,
            pools=hz.pools,
            priced_gws=hz.priced_gws,
            n_projected_fixtures=hz.n_projected_fixtures,
            meta=meta,
        )
    if arm == "l4_w3":
        return build_mupost_windowed_horizon(
            season,
            decision_gw,
            week0_pool=week0_pool,
            v2_season=v2_season,
            window=3,
            merged=merged,
            fd=fd,
            prices=prices,
        )
    if arm == "l4_w5":
        return build_mupost_windowed_horizon(
            season,
            decision_gw,
            week0_pool=week0_pool,
            v2_season=v2_season,
            window=5,
            merged=merged,
            fd=fd,
            prices=prices,
        )
    raise ValueError(f"unknown arm {arm!r}")


def main(argv: list[str] | None = None) -> int:
    from src.strategy.shrink_climb import SeasonStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=15)
    p.add_argument("--window", type=int, default=3)
    args = p.parse_args(argv)
    v2 = pd.read_csv(PROCESSED / "score_xp_v2_closed.csv")
    store = SeasonStore(args.season, v2)
    pool = store.pool(args.gw)
    moments = store.moments_asof(args.gw)
    work, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    hz = build_mupost_windowed_horizon(
        args.season,
        args.gw,
        week0_pool=work,
        v2_season=store.v2,
        window=args.window,
    )
    print(
        json.dumps(
            {
                "H": len(hz.pools),
                "priced_gws": hz.priced_gws,
                "meta": hz.meta,
                "top": hz.pools[0]
                .sort_values("mu_post", ascending=False)
                .head(5)[["name", "mu_post"]]
                .to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
