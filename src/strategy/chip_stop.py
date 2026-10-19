"""Chip half-stopping: TC/BB leave-one-out DP + fixed WC/FH margins.

Single-use options expire at half-end. Thresholds τ(r) for remaining weeks r
in the half: play TC/BB this week iff week gain ≥ τ(r). τ_1=0 (use-or-lose);
τ_r = E[max(G, τ_{r-1})] from as-of pseudo-XI gains (leave-one-season-out).

WC / FH keep locked live margins (16 / 12) vs none on the commit week.

  python -m src.strategy.chip_stop --season 2025-26
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS
from src.live.policy import FH_MARGIN, WC_MARGIN
from src.rules.fpl_2026 import FIRST_HALF_END_GW, N_GAMEWEEKS, half_for_gw
from src.strategy.decision_value import attach_mu_post
from src.strategy.shrink_climb import SeasonStore

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
GW_END = 38
N_SAMPLES = 250
MAX_R = 19  # longest half span


def weeks_left_in_half(gw: int) -> int:
    end = FIRST_HALF_END_GW if half_for_gw(gw) == "H1" else N_GAMEWEEKS
    return max(1, int(end) - int(gw) + 1)


def _pseudo_xi_bench(work: pd.DataFrame, mu_col: str = "mu_post") -> tuple[float, float]:
    """As-of TC/BB gains from a greedy legal 15 → XI: captain μ, bench Σμ."""
    df = work.copy()
    df["pid"] = df["player_id"].astype(str)
    df["mu"] = pd.to_numeric(df[mu_col], errors="coerce").fillna(0.0)
    df["pos"] = df["position"].astype(str)
    parts: list[pd.DataFrame] = []
    for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        sub = df.loc[df["pos"] == pos].sort_values("mu", ascending=False).head(n)
        if len(sub) < n:
            return 0.0, 0.0
        parts.append(sub)
    squad = pd.concat(parts, ignore_index=True)
    gkp = squad.loc[squad["pos"] == "GKP"].sort_values("mu", ascending=False).head(1)
    outfield = squad.loc[squad["pos"] != "GKP"]
    best_xi: pd.DataFrame | None = None
    best_sum = -1.0
    for nd, nm, nf in (
        (3, 4, 3),
        (3, 5, 2),
        (4, 4, 2),
        (4, 5, 1),
        (5, 3, 2),
        (5, 4, 1),
        (4, 3, 3),
        (5, 2, 3),
    ):
        d = outfield.loc[outfield["pos"] == "DEF"].sort_values("mu", ascending=False).head(nd)
        m = outfield.loc[outfield["pos"] == "MID"].sort_values("mu", ascending=False).head(nm)
        f = outfield.loc[outfield["pos"] == "FWD"].sort_values("mu", ascending=False).head(nf)
        if len(d) < nd or len(m) < nm or len(f) < nf:
            continue
        xi = pd.concat([gkp, d, m, f], ignore_index=True)
        s = float(xi["mu"].sum())
        if s > best_sum:
            best_sum = s
            best_xi = xi
    if best_xi is None or best_xi.empty:
        return 0.0, 0.0
    xi_ids = set(best_xi["pid"].astype(str))
    bench = (
        squad.loc[~squad["pid"].astype(str).isin(xi_ids)]
        .sort_values("mu", ascending=False)
        .head(4)
    )
    captain_mu = float(best_xi["mu"].max())
    bench_mu = float(bench["mu"].sum()) if len(bench) else 0.0
    return captain_mu, bench_mu


def collect_scoring_gains(
    *,
    exclude_season: str | None = None,
    gw_start: int = GW_START,
    gw_end: int = GW_END,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """As-of (tc_gain, bb_gain) pairs; leave-one-season-out capable."""
    rng = np.random.default_rng(seed)
    v2 = pd.read_csv(CLOSED_V2)
    tc: list[float] = []
    bb: list[float] = []
    for season in CLOSED_SEASONS:
        if exclude_season and season == exclude_season:
            continue
        store = SeasonStore(season, v2)
        gws = store.gws(gw_start, gw_end)
        if len(gws) > N_SAMPLES:
            pick = set(rng.choice(gws, size=N_SAMPLES, replace=False).tolist())
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
            g_tc, g_bb = _pseudo_xi_bench(work)
            tc.append(g_tc)
            bb.append(g_bb)
    return np.asarray(tc, float), np.asarray(bb, float)


def solve_stopping_tau(gains: np.ndarray, *, max_r: int = MAX_R) -> dict[int, float]:
    """Finite-horizon optimal-stopping thresholds from empirical gains."""
    g = np.asarray(gains, float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {r: 0.0 for r in range(1, max_r + 1)}
    tau: dict[int, float] = {1: 0.0}
    for r in range(2, max_r + 1):
        tau[r] = float(np.mean(np.maximum(g, tau[r - 1])))
    return tau


@dataclass(frozen=True)
class ChipStopPolicy:
    """τ_tc / τ_bb keyed by weeks left in half; WC/FH fixed margins."""

    tau_tc: dict[int, float]
    tau_bb: dict[int, float]
    wc_margin: float = float(WC_MARGIN)
    fh_margin: float = float(FH_MARGIN)
    n_gains: int = 0
    mean_tc: float = 0.0
    mean_bb: float = 0.0

    def threshold_tc(self, gw: int) -> float:
        r = min(MAX_R, weeks_left_in_half(gw))
        return float(self.tau_tc.get(r, 0.0))

    def threshold_bb(self, gw: int) -> float:
        r = min(MAX_R, weeks_left_in_half(gw))
        return float(self.tau_bb.get(r, 0.0))


def fit_policy_leave_one_out(
    exclude_season: str,
    *,
    gw_start: int = GW_START,
    gw_end: int = GW_END,
    seed: int = 0,
) -> ChipStopPolicy:
    tc, bb = collect_scoring_gains(
        exclude_season=exclude_season, gw_start=gw_start, gw_end=gw_end, seed=seed
    )
    return ChipStopPolicy(
        tau_tc=solve_stopping_tau(tc),
        tau_bb=solve_stopping_tau(bb),
        n_gains=int(tc.size),
        mean_tc=float(tc.mean()) if tc.size else 0.0,
        mean_bb=float(bb.mean()) if bb.size else 0.0,
    )


def choose_chip(
    *,
    gw: int,
    available: tuple[str, ...],
    tc_gain: float,
    bb_gain: float,
    wc_gain: float | None,
    fh_gain: float | None,
    policy: ChipStopPolicy,
) -> tuple[str, dict[str, Any]]:
    """Pick one chip or ``none``. Clears hurdle with largest excess wins."""
    from src.strategy.layer5_week import NO_CHIP

    avail = set(available)
    cands: list[tuple[str, float, float]] = []  # chip, gain, hurdle
    if "triple_captain" in avail:
        h = policy.threshold_tc(gw)
        if tc_gain >= h and tc_gain > 0:
            cands.append(("triple_captain", float(tc_gain), h))
    if "bench_boost" in avail:
        h = policy.threshold_bb(gw)
        if bb_gain >= h and bb_gain > 0:
            cands.append(("bench_boost", float(bb_gain), h))
    if "wildcard" in avail and wc_gain is not None:
        h = policy.wc_margin
        if float(wc_gain) >= h:
            cands.append(("wildcard", float(wc_gain), h))
    if "free_hit" in avail and fh_gain is not None:
        h = policy.fh_margin
        if float(fh_gain) >= h:
            cands.append(("free_hit", float(fh_gain), h))
    meta = {
        "tc_gain": float(tc_gain),
        "bb_gain": float(bb_gain),
        "wc_gain": None if wc_gain is None else float(wc_gain),
        "fh_gain": None if fh_gain is None else float(fh_gain),
        "tau_tc": policy.threshold_tc(gw),
        "tau_bb": policy.threshold_bb(gw),
        "weeks_left": weeks_left_in_half(gw),
        "n_clear": len(cands),
    }
    if not cands:
        return NO_CHIP, meta
    cands.sort(key=lambda x: x[1] - x[2], reverse=True)
    return cands[0][0], meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=GW_END)
    args = ap.parse_args(argv)
    pol = fit_policy_leave_one_out(
        args.season, gw_start=args.gw_start, gw_end=args.gw_end
    )
    print(
        f"n={pol.n_gains} mean_tc={pol.mean_tc:.3f} mean_bb={pol.mean_bb:.3f}",
        flush=True,
    )
    for r in (1, 2, 4, 8, 12, 19):
        print(
            f"r={r:2d} τ_tc={pol.tau_tc[r]:.3f} τ_bb={pol.tau_bb[r]:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
