"""L4 μ_post priced path + risk-seeking decision μ.

Mean board unchanged from ``build_mupost_priced_horizon``.
Decision column:

    mu_dec = mu_post + λ · (w_se · z(se_post) + w_cov · z(sd_cov))

``sd_cov`` from fixture-covariance draws (as-of decision GW; risk only).
Hard stop at priced window. Cov mean never replaces μ_post.

  python -m src.strategy.layer4_mupost_risk --season 2025-26 --gw 12
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
    BOOKIE_DEPTH,
    _fd_index,
    _load_merged_enriched,
    score_week_cov_risk,
)
from src.strategy.layer4_mupost_priced import (
    MupostPricedHorizon,
    build_mupost_priced_horizon,
)
from src.strategy.risk_mu import (
    DEFAULT_LAMBDA,
    DEFAULT_W_COV,
    DEFAULT_W_SE,
    attach_risk_mu,
    risk_defs,
)
from src.strategy.shrinkage import CohortMoments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

N_SCENARIOS = 32
RISK_SEED = 17


@dataclass(frozen=True)
class MupostRiskHorizon:
    season: str
    decision_gw: int
    pools: list[pd.DataFrame]
    priced_gws: tuple[int, ...]
    n_projected_fixtures: int
    n_cov_players: int
    meta: dict[str, Any]


def _merge_sd_cov(pool: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    if risk.empty or "element" not in out.columns:
        out["sd_cov"] = 0.0
        return out
    el = pd.to_numeric(out["element"], errors="coerce")
    sd_map = dict(
        zip(
            risk["element"].astype(int),
            pd.to_numeric(risk["sd_draw"], errors="coerce").fillna(0.0),
        )
    )
    out["sd_cov"] = el.map(lambda e: float(sd_map.get(int(e), 0.0)) if pd.notna(e) else 0.0)
    return out


def build_mupost_risk_horizon(
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    lambda_risk: float = DEFAULT_LAMBDA,
    w_se: float = DEFAULT_W_SE,
    w_cov: float = DEFAULT_W_COV,
    n_scenarios: int = N_SCENARIOS,
    seed: int = RISK_SEED,
) -> MupostRiskHorizon:
    """Priced μ_post pools with ``mu_dec`` for the MILP (μ_post left intact)."""
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
    n_cov = 0
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
        board = _merge_sd_cov(pool, risk)
        if not risk.empty:
            n_cov += int((board["sd_cov"] > 0).sum())
        board = attach_risk_mu(
            board,
            mu_col="mu_post",
            se_col="xp_se",
            cov_sd_col="sd_cov",
            lambda_risk=lambda_risk,
            w_se=w_se,
            w_cov=w_cov,
            moments_by_pos=moments,
            out_col="mu_dec",
        )
        pools.append(board)

    return MupostRiskHorizon(
        season=season,
        decision_gw=int(decision_gw),
        pools=pools,
        priced_gws=base.priced_gws,
        n_projected_fixtures=base.n_projected_fixtures,
        n_cov_players=n_cov,
        meta={
            **base.meta,
            **risk_defs(
                lambda_risk=lambda_risk,
                w_se=w_se,
                w_cov=w_cov,
                n_scenarios=n_scenarios,
            ),
            "bookie_depth": BOOKIE_DEPTH,
        },
    )


def main(argv: list[str] | None = None) -> int:
    from src.strategy.decision_value import attach_mu_post
    from src.strategy.shrink_climb import SeasonStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=12)
    p.add_argument("--lambda-risk", type=float, default=DEFAULT_LAMBDA)
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
    hz = build_mupost_risk_horizon(
        args.season,
        args.gw,
        week0_pool=work,
        moments=moments,
        lambda_risk=args.lambda_risk,
    )
    w0 = hz.pools[0]
    top = w0.sort_values("mu_dec", ascending=False).head(8)
    print(
        json.dumps(
            {
                "H": len(hz.pools),
                "priced_gws": hz.priced_gws,
                "n_cov_players": hz.n_cov_players,
                "meta": hz.meta,
                "top_mu_dec": top[
                    [c for c in ("name", "mu_post", "mu_dec", "risk_unit", "se_post", "sd_cov") if c in top.columns]
                ].to_dict(orient="records"),
                "corr_mu_risk": float(w0["mu_post"].corr(w0["risk_unit"]))
                if "risk_unit" in w0.columns
                else None,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
