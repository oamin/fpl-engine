"""L4 priced μ_post + chase tilt from cov upside semi-SD.

Mean boards = ``build_mupost_priced_horizon``. Decision:

    mu_dec = mu_post + λ · z(upside_semi_sd)

``upside_semi_sd`` from fixture-covariance draws (as-of decision GW).
μ_post never replaced. No SE blend.

  python -m src.strategy.layer4_mupost_chase --season 2025-26 --gw 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.markets.numagent_xp_pilot import _asof_alphas
from src.strategy.layer4_bookie_path import (
    _fd_index,
    _load_merged_enriched,
    score_week_cov_risk,
)
from src.strategy.layer4_mupost_priced import (
    MupostPricedHorizon,
    build_mupost_priced_horizon,
)
from src.strategy.risk_mu import DEFAULT_CHASE_LAMBDA, attach_chase_mu
from src.strategy.shrinkage import CohortMoments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

N_SCENARIOS = 32
CHASE_SEED = 19


@dataclass(frozen=True)
class MupostChaseHorizon:
    season: str
    decision_gw: int
    pools: list[pd.DataFrame]
    priced_gws: tuple[int, ...]
    n_projected_fixtures: int
    n_upside_players: int
    meta: dict[str, Any]


def _merge_upside(pool: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    if risk.empty or "element" not in out.columns:
        out["upside_semi_sd"] = 0.0
        return out
    el = pd.to_numeric(out["element"], errors="coerce")
    col = "upside_semi_sd" if "upside_semi_sd" in risk.columns else "sd_draw"
    u_map = dict(
        zip(
            risk["element"].astype(int),
            pd.to_numeric(risk[col], errors="coerce").fillna(0.0),
        )
    )
    out["upside_semi_sd"] = el.map(
        lambda e: float(u_map.get(int(e), 0.0)) if pd.notna(e) else 0.0
    )
    return out


def build_mupost_chase_horizon(
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    lambda_chase: float = DEFAULT_CHASE_LAMBDA,
    n_scenarios: int = N_SCENARIOS,
    seed: int = CHASE_SEED,
) -> MupostChaseHorizon:
    if merged is None:
        merged = _load_merged_enriched(season)
    if fd is None:
        fd = _fd_index(season)

    base: MupostPricedHorizon = build_mupost_priced_horizon(
        season,
        decision_gw,
        week0_pool=week0_pool,
        moments=moments,
        merged=merged,
        fd=fd,
        prices=prices,
    )
    hist = merged.loc[merged["GW"] < int(decision_gw)]
    alpha_g, alpha_a = _asof_alphas(hist)

    pools: list[pd.DataFrame] = []
    n_up = 0
    for j, (pool, tgt) in enumerate(zip(base.pools, base.priced_gws)):
        try:
            risk = score_week_cov_risk(
                merged,
                fd,
                decision_gw=int(decision_gw),
                target_gw=int(tgt),
                alpha_g=alpha_g,
                alpha_a=alpha_a,
                n_scenarios=int(n_scenarios),
                seed=int(seed) + 1009 * int(tgt) + 17 * j,
            )
        except (ValueError, KeyError, RuntimeError):
            risk = pd.DataFrame()
        board = _merge_upside(pool, risk)
        if "upside_semi_sd" in board.columns:
            n_up += int((board["upside_semi_sd"] > 0).sum())
        board = attach_chase_mu(
            board,
            mu_col="mu_post",
            upside_col="upside_semi_sd",
            lambda_chase=lambda_chase,
            out_col="mu_dec",
        )
        pools.append(board)

    return MupostChaseHorizon(
        season=season,
        decision_gw=int(decision_gw),
        pools=pools,
        priced_gws=base.priced_gws,
        n_projected_fixtures=base.n_projected_fixtures,
        n_upside_players=n_up,
        meta={
            **base.meta,
            "lambda_chase": float(lambda_chase),
            "n_scenarios": int(n_scenarios),
            "posture": "chase",
            "upside": "cov_semi_sd",
        },
    )


def main(argv: list[str] | None = None) -> int:
    from src.strategy.decision_value import attach_mu_post
    from src.strategy.shrink_climb import SeasonStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=12)
    p.add_argument("--lambda-chase", type=float, default=DEFAULT_CHASE_LAMBDA)
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
    hz = build_mupost_chase_horizon(
        args.season,
        args.gw,
        week0_pool=work,
        moments=moments,
        lambda_chase=args.lambda_chase,
    )
    w0 = hz.pools[0]
    top = w0.sort_values("mu_dec", ascending=False).head(8)
    cols = [c for c in ("name", "mu_post", "mu_dec", "upside_semi_sd", "chase_unit") if c in top.columns]
    print(
        json.dumps(
            {
                "H": len(hz.pools),
                "priced_gws": hz.priced_gws,
                "n_upside_players": hz.n_upside_players,
                "meta": hz.meta,
                "top_mu_dec": top[cols].to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
