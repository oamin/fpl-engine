"""L4 priced path with adaptive chase/consolidate from gap-driven λ.

  mu_dec = mu_post + λ_chase · z(upside_semi_sd) − λ_cons · z(downside_semi_sd)

λ from as-of cumulative gap (L3 − L4 through GW t−1). Cov draws supply
semi-SDs only; μ_post intact.

  python -m src.strategy.layer4_mupost_posture --season 2025-26 --gw 15 --gap 40
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
from src.strategy.risk_mu import (
    DEFAULT_GAP_SCALE,
    DEFAULT_LAMBDA_MAX,
    attach_posture_mu,
    gap_to_lambdas,
)
from src.strategy.shrinkage import CohortMoments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

N_SCENARIOS = 32
POSTURE_SEED = 23


@dataclass(frozen=True)
class MupostPostureHorizon:
    season: str
    decision_gw: int
    pools: list[pd.DataFrame]
    priced_gws: tuple[int, ...]
    n_projected_fixtures: int
    lambda_chase: float
    lambda_cons: float
    posture: str
    gap: float
    meta: dict[str, Any]


def _merge_semi(pool: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    if risk.empty or "element" not in out.columns:
        out["upside_semi_sd"] = 0.0
        out["downside_semi_sd"] = 0.0
        return out
    el = pd.to_numeric(out["element"], errors="coerce")

    def _map(col: str) -> pd.Series:
        if col not in risk.columns:
            return el.map(lambda _: 0.0)
        m = dict(
            zip(
                risk["element"].astype(int),
                pd.to_numeric(risk[col], errors="coerce").fillna(0.0),
            )
        )
        return el.map(lambda e: float(m.get(int(e), 0.0)) if pd.notna(e) else 0.0)

    out["upside_semi_sd"] = _map("upside_semi_sd")
    out["downside_semi_sd"] = _map("downside_semi_sd")
    return out


def build_mupost_posture_horizon(
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    gap: float,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
    gap_scale: float = DEFAULT_GAP_SCALE,
    chase_only: bool = False,
    n_scenarios: int = N_SCENARIOS,
    seed: int = POSTURE_SEED,
) -> MupostPostureHorizon:
    if merged is None:
        merged = _load_merged_enriched(season)
    if fd is None:
        fd = _fd_index(season)

    lam_c, lam_k, posture = gap_to_lambdas(
        gap, lambda_max=lambda_max, scale=gap_scale, chase_only=chase_only
    )
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
        board = _merge_semi(pool, risk)
        if posture == "neutral" or (lam_c == 0.0 and lam_k == 0.0):
            board = board.copy()
            board["mu_dec"] = pd.to_numeric(board["mu_post"], errors="coerce").fillna(0.0)
            board["lambda_chase"] = 0.0
            board["lambda_cons"] = 0.0
        else:
            board = attach_posture_mu(
                board,
                lambda_chase=lam_c,
                lambda_cons=lam_k,
                out_col="mu_dec",
            )
        pools.append(board)

    return MupostPostureHorizon(
        season=season,
        decision_gw=int(decision_gw),
        pools=pools,
        priced_gws=base.priced_gws,
        n_projected_fixtures=base.n_projected_fixtures,
        lambda_chase=lam_c,
        lambda_cons=lam_k,
        posture=posture,
        gap=float(gap),
        meta={
            **base.meta,
            "lambda_max": float(lambda_max),
            "gap_scale": float(gap_scale),
            "chase_only": bool(chase_only),
            "n_scenarios": int(n_scenarios),
            "lambda_chase": lam_c,
            "lambda_cons": lam_k,
            "posture": posture,
            "gap": float(gap),
        },
    )


def main(argv: list[str] | None = None) -> int:
    from src.strategy.decision_value import attach_mu_post
    from src.strategy.shrink_climb import SeasonStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=15)
    p.add_argument("--gap", type=float, default=40.0, help="L3−L4 cumulative pts")
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
    hz = build_mupost_posture_horizon(
        args.season,
        args.gw,
        week0_pool=work,
        moments=moments,
        gap=args.gap,
    )
    print(
        json.dumps(
            {
                "gap": hz.gap,
                "posture": hz.posture,
                "lambda_chase": hz.lambda_chase,
                "lambda_cons": hz.lambda_cons,
                "H": len(hz.pools),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
