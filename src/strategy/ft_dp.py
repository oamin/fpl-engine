"""FT bank DP: as-of thresholds for roll vs spend (vs hit).

State (week, bank b∈0..MAX_FT). Opportunities: 1-FT gain g1 and 2-FT gain g2
(as-of swap edges). Using 2 FTs can lower the bank — that creates option value.
Continuation follows advance_ft. Leave-one-season-out gains.

Threshold τ(t,b) = min g1 to prefer a 1-FT spend over roll.
  τ → 0 near the cap (unused grant wasted at MAX_FT).

  python -m src.strategy.ft_dp --season 2025-26
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS
from src.rules.fpl_2026 import HIT_COST, MAX_FT
from src.strategy.decision_value import attach_mu_post
from src.strategy.shrink_climb import SeasonStore

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
GW_END = 38
N_SWAP_SAMPLES = 200


@dataclass(frozen=True)
class FtDpPolicy:
    """τ[gw][b] = min gain to spend rather than roll; b=0..MAX_FT."""

    tau: dict[int, dict[int, float]]
    n_gains: int
    mean_g: float
    hit_cost: float = float(HIT_COST)

    def threshold(self, gw: int, bank: int) -> float:
        b = int(np.clip(bank, 0, MAX_FT))
        gws = sorted(self.tau)
        if not gws:
            return float(HIT_COST) if b == 0 else 0.0
        t = int(gw)
        if t not in self.tau:
            t = min(gws, key=lambda x: abs(x - t))
        return float(self.tau[t][b])


def _pseudo_owned(df: pd.DataFrame, start: int) -> set[str]:
    owned: list[str] = []
    for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        sub = (
            df.loc[df["pos"] == pos]
            .sort_values("mu", ascending=False)
            .reset_index(drop=True)
        )
        if len(sub) < n:
            continue
        s = min(int(start), max(0, len(sub) - n))
        owned.extend(sub.iloc[s : s + n]["pid"].tolist())
    return set(owned)


def _k_swap_gain(work: pd.DataFrame, *, k: int, start: int, mu_col: str = "mu_post") -> float:
    """Sum of top-k same-position swap edges on a pseudo-15 offset ``start``."""
    df = work.copy()
    df["pid"] = df["player_id"].astype(str)
    df["mu"] = pd.to_numeric(df[mu_col], errors="coerce").fillna(0.0)
    df["pos"] = df["position"].astype(str)
    own_set = _pseudo_owned(df, start)
    if len(own_set) < 10:
        return 0.0
    edges: list[float] = []
    for pos in ("GKP", "DEF", "MID", "FWD"):
        o = df.loc[(df["pos"] == pos) & df["pid"].isin(own_set)].sort_values("mu")
        u = (
            df.loc[(df["pos"] == pos) & ~df["pid"].isin(own_set)]
            .sort_values("mu", ascending=False)
        )
        for i in range(min(len(o), len(u))):
            edges.append(float(u.iloc[i]["mu"] - o.iloc[i]["mu"]))
    edges = sorted((e for e in edges if e > 0), reverse=True)
    if not edges:
        return 0.0
    return float(sum(edges[: int(k)]))


def collect_gains(
    *,
    exclude_season: str | None = None,
    gw_start: int = GW_START,
    gw_end: int = GW_END,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """As-of (g1, g2) swap gains; leave-one-season-out capable."""
    rng = np.random.default_rng(seed)
    v2 = pd.read_csv(CLOSED_V2)
    g1s: list[float] = []
    g2s: list[float] = []
    for season in CLOSED_SEASONS:
        if exclude_season and season == exclude_season:
            continue
        store = SeasonStore(season, v2)
        gws = store.gws(gw_start, gw_end)
        if len(gws) > N_SWAP_SAMPLES:
            pick = set(rng.choice(gws, size=N_SWAP_SAMPLES, replace=False).tolist())
        else:
            pick = set(gws)
        for gw in gws:
            if gw not in pick:
                continue
            moments = store.moments_asof(gw)
            pool = store.pool(gw)
            work, _ = attach_mu_post(
                pool,
                mu_col="score_xp_v2",
                se_col="xp_se",
                mode="none",
                moments_by_pos=moments,
                out_col="mu_post",
            )
            start = int(rng.choice([0, 0, 1, 1, 2, 3]))
            g1s.append(_k_swap_gain(work, k=1, start=start))
            g2s.append(_k_swap_gain(work, k=2, start=start))
    return np.asarray(g1s, float), np.asarray(g2s, float)


def _next_bank(b: int, used: int) -> int:
    """advance_ft without chip: used transfers this week."""
    if used <= b:
        return min(MAX_FT, b - used + 1)
    return min(MAX_FT, 1)


def solve_ft_dp(
    g1: np.ndarray,
    g2: np.ndarray | None = None,
    *,
    gw_start: int = GW_START,
    gw_end: int = GW_END,
    hit_cost: float = float(HIT_COST),
) -> FtDpPolicy:
    """Backward induction with 1-FT and 2-FT opportunities."""
    gws = list(range(int(gw_start), int(gw_end) + 1))
    if g1.size == 0:
        g1 = np.asarray([0.0])
    if g2 is None or g2.size != g1.size:
        g2 = g1.copy()
    V = np.zeros((len(gws) + 1, MAX_FT + 1), dtype=float)
    tau: dict[int, dict[int, float]] = {}

    for ti in range(len(gws) - 1, -1, -1):
        tau_b: dict[int, float] = {}
        for b in range(MAX_FT + 1):
            v_roll = float(V[ti + 1, _next_bank(b, 0)])
            if b >= 1:
                thr = v_roll - float(V[ti + 1, _next_bank(b, 1)])
                tau_b[b] = float(max(0.0, thr))
            else:
                thr = hit_cost + (v_roll - float(V[ti + 1, _next_bank(0, 1)]))
                tau_b[b] = float(max(0.0, thr))

            total = 0.0
            for ga, gb in zip(g1, g2, strict=True):
                ga, gb = float(ga), float(gb)
                best = v_roll
                if b >= 1:
                    best = max(best, ga + float(V[ti + 1, _next_bank(b, 1)]))
                else:
                    best = max(
                        best, ga - hit_cost + float(V[ti + 1, _next_bank(0, 1)])
                    )
                need = 2
                hits = max(0, need - b)
                best = max(
                    best,
                    gb - hit_cost * hits + float(V[ti + 1, _next_bank(b, need)]),
                )
                total += best
            V[ti, b] = total / float(g1.size)
        tau[int(gws[ti])] = tau_b

    return FtDpPolicy(
        tau=tau,
        n_gains=int(g1.size),
        mean_g=float(np.mean(g1)),
        hit_cost=float(hit_cost),
    )


def fit_policy_leave_one_out(exclude_season: str, **kwargs: Any) -> FtDpPolicy:
    g1, g2 = collect_gains(exclude_season=exclude_season, **kwargs)
    return solve_ft_dp(
        g1,
        g2,
        gw_start=kwargs.get("gw_start", GW_START),
        gw_end=kwargs.get("gw_end", GW_END),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26", help="leave-one-out season")
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=GW_END)
    args = ap.parse_args(argv)
    pol = fit_policy_leave_one_out(
        args.season, gw_start=args.gw_start, gw_end=args.gw_end
    )
    mid = 20
    print(
        json.dumps(
            {
                "exclude_season": args.season,
                "n_gains": pol.n_gains,
                "mean_g1": pol.mean_g,
                f"tau_gw{mid}": {
                    str(b): round(pol.threshold(mid, b), 4) for b in range(MAX_FT + 1)
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
