"""Stage 19b — FT climb with XI-horizon transfer valuation.

Rules modelled (chips only when the caller passes a week → chip map):
  - GW1 of the climb: free 15-man build under £100.0m (wildcard-like)
  - Thereafter: 1 FT / GW, stack to 5; extras −4 each
  - Sell price = purchase + ⌊rise/2⌋; full fall to current
  - Transfer policy: enumerate hold / swaps; maximise
      V = Σ_{h<H} γ^h · XI_score_h − 4·hits
    with current scores carried forward (leakage-free) and blanks → 0.
  - Hold unless best V beats hold by HOLD_EPS (stops FT churn)
  - Scoring: FPL autosubs (0-min starters ← ordered bench) + captain/VC
  - An empty chip map plays nothing. The map does not choose the week.

Primary scorer: stage-17 ``ridge_global_starters``. Baselines: xp, exp_points.

Writes:
  data/processed/season_climb_ft.csv
  data/plots/season_climb_ft.png
  reports/stage_19_season_climb_ft.md
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.ridge_multiseason import CACHE
from src.models.ridge_starters import (
    EVAL_SEASON,
    build_fresh_seasons,
    walk_forward_global_ridge_starters,
)
from src.models.season_climb import bank_squad_gw, pick_xi, summarize
from src.models.season_climb_budget import (
    BUDGET,
    MAX_PER_CLUB,
    SQUAD_QUOTA,
    pick_squad,
    run_budgeted_season,
)
from src.rules.fpl_2026 import (
    FREE_TRANSFER_CHIPS,
    HIT_COST,
    MAX_FT,
    advance_ft,
    captain_extra_points,
    sell_price,
    validate_chip_map,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"

MAX_HITS = 2  # allow up to ft+2 transfers
HORIZON = 3
GAMMA = 0.9
HOLD_EPS = 1.25  # need ~0.4+ XI pts/GW over hold (horizon-scaled)
SWITCH_PENALTY = 1.0  # hysteresis per transfer in V (on top of hits)
# Horizon uncertainty: σ_h = σ0·√(1+β·h); w = τ²/(τ²+σ_h²)
UNC_BETA = 1.0
UNC_TAU = 3.0
# Multi-transfer beam: sequential best-1 from top partials
BEAM_WIDTH = 6
FALLBACK_SCORE_COLS = ("score_xp", "score_exp_points", "value")


def load_vaastav_roster(season: str) -> pd.DataFrame:
    """Full Vaastav player×GW grid (survives odds-join drops in feat)."""
    raw = pd.read_csv(CACHE / f"merged_gw_{season.replace('-', '_')}.csv")
    pos = raw["position"].astype(str).replace({"GK": "GKP"})
    out = pd.DataFrame(
        {
            "player_id": season + ":" + raw["element"].astype(str),
            "gw": pd.to_numeric(raw["GW"], errors="coerce").astype(int),
            "player_name": raw["name"],
            "position": pos,
            "team": raw["team"].astype(str),
            "team_norm": raw["team"].astype(str).str.lower(),
            "value": pd.to_numeric(raw["value"], errors="coerce").fillna(50).astype(int),
            "total_points": pd.to_numeric(raw["total_points"], errors="coerce").fillna(0),
            "minutes": pd.to_numeric(raw["minutes"], errors="coerce").fillna(0),
        }
    )
    return out.drop_duplicates(["player_id", "gw"], keep="first")


def load_fixture_counts(season: str) -> dict[int, dict[str, int]]:
    """Per-GW fixture multiplicity from Vaastav (1=SGW, 2=DGW). Missing ⇒ blank."""
    raw = pd.read_csv(CACHE / f"merged_gw_{season.replace('-', '_')}.csv")
    pid = season + ":" + raw["element"].astype(str)
    gw = pd.to_numeric(raw["GW"], errors="coerce").astype(int)
    counts = (
        pd.DataFrame({"player_id": pid, "gw": gw})
        .groupby(["gw", "player_id"], sort=False)
        .size()
    )
    out: dict[int, dict[str, int]] = {}
    for (g, p), n in counts.items():
        out.setdefault(int(g), {})[str(p)] = int(min(int(n), 2))
    return out


def _carry_forward_owned(
    roster: pd.DataFrame, owned: set[str], gw: int
) -> pd.DataFrame:
    """Stub rows for owned players blanked/missing in Vaastav this GW (0 pts)."""
    have = set(roster.loc[roster["gw"] == gw, "player_id"])
    need = owned - have
    if not need:
        return roster.iloc[0:0].copy()
    hist = (
        roster.loc[roster["player_id"].isin(need) & (roster["gw"] < gw)]
        .sort_values("gw")
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )
    if hist.empty:
        return roster.iloc[0:0].copy()
    hist["gw"] = gw
    hist["total_points"] = 0.0
    hist["minutes"] = 0.0
    return hist


@dataclass
class SquadState:
    """Persistent FPL squad state."""

    purchase: dict[str, int] = field(default_factory=dict)  # player_id → buy price
    bank: int = 0
    ft: int = 1

    def ids(self) -> set[str]:
        return set(self.purchase.keys())


def _fill_score(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Score for optimisation; fall back so owned non-eligible rows stay usable."""
    s = pd.to_numeric(df[score_col], errors="coerce")
    for col in FALLBACK_SCORE_COLS:
        if col == score_col or col not in df.columns:
            continue
        s = s.fillna(pd.to_numeric(df[col], errors="coerce"))
    return s.fillna(-1e6)


def _gw_pool(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    gw: int,
    owned: set[str],
) -> pd.DataFrame:
    """Eligible buy pool + owned rows (roster-filled when odds-join dropped them)."""
    pool = feat.loc[feat["gw"] == gw].copy()
    pool["player_id"] = pool["player_id"].astype(str)
    pool = pool.drop_duplicates("player_id", keep="first")
    keep = pool["eligible"].astype(bool) | pool["player_id"].isin(owned)
    pool = pool.loc[keep].copy()

    missing = owned - set(pool["player_id"])
    if missing:
        stubs = roster.loc[
            (roster["gw"] == gw) & (roster["player_id"].isin(missing))
        ].copy()
        carried = _carry_forward_owned(roster, missing, gw)
        if not carried.empty:
            stubs = pd.concat([stubs, carried], ignore_index=True)
            stubs = stubs.drop_duplicates("player_id", keep="first")
        if not stubs.empty:
            # Carry the latest score from before this deadline. A later
            # week in the frame is not known yet.
            hist = feat.loc[
                feat["player_id"].isin(stubs["player_id"])
                & (pd.to_numeric(feat["gw"], errors="coerce") < gw)
            ]
            hist = (
                hist.sort_values("gw")
                .groupby("player_id", as_index=False)
                .tail(1)
            )
            score_cols = [
                c
                for c in feat.columns
                if c.startswith("score_") or c in ("exp_points", "xp", "value")
            ]
            if len(hist) and score_cols:
                stubs = stubs.merge(
                    hist[["player_id", *score_cols]].drop_duplicates("player_id"),
                    on="player_id",
                    how="left",
                )
            stubs["eligible"] = False
            stubs["position"] = stubs["position"].replace({"GK": "GKP"})
            for c in pool.columns:
                if c not in stubs.columns:
                    stubs[c] = np.nan
            stubs = stubs[pool.columns]
            pool = pd.concat([pool, stubs], ignore_index=True)

    pool["position"] = pool["position"].replace({"GK": "GKP"})
    return pool.drop_duplicates("player_id", keep="first").reset_index(drop=True)


def _as_int_value(v: object, default: int = 50) -> int:
    x = pd.to_numeric(v, errors="coerce")
    return int(x) if pd.notna(x) else default


def _squad_legal(positions: list[str], clubs: list[str]) -> bool:
    if len(positions) != 15:
        return False
    from collections import Counter

    pc = Counter(positions)
    for pos, need in SQUAD_QUOTA.items():
        if pc.get(pos, 0) != need:
            return False
    for c, n in Counter(clubs).items():
        if n > MAX_PER_CLUB:
            return False
    return True


def _xi_score_sum(squad_df: pd.DataFrame, score_col: str) -> float:
    if len(squad_df) < 11:
        return -1e9
    df = squad_df.copy()
    df[score_col] = _fill_score(df, score_col)
    try:
        xi, _ = pick_xi(df, score_col)
    except RuntimeError:
        return -1e9
    return float(xi[score_col].sum())


def precision_weight(
    sigma0: float,
    h: int,
    *,
    beta: float = UNC_BETA,
    tau: float = UNC_TAU,
) -> float:
    """w_σ = τ² / (τ² + σ_h²) with σ_h = σ0·√(1+β·h)."""
    sig_h = float(sigma0) * np.sqrt(1.0 + beta * float(h))
    return float((tau * tau) / (tau * tau + sig_h * sig_h))


def transfer_value(
    squad_ids: set[str],
    score_now: dict[str, float],
    meta: dict[str, dict[str, Any]],
    hits: int,
    gw: int,
    future_gws: list[int],
    roster_by_gw: dict[int, set[str]],
    score_col: str,
    *,
    horizon: int | None = None,
    fixture_counts: dict[int, dict[str, int]] | None = None,
    sigma0: dict[str, float] | None = None,
    fdr_by_gw: dict[int, dict[str, float]] | None = None,
    flow_by_pid: dict[str, float] | None = None,
    unc_beta: float = UNC_BETA,
    unc_tau: float = UNC_TAU,
    exp_score_now: dict[str, float] | None = None,
    blend_gamma: float | None = None,
    blend_floor: float = 0.5,
    blend_schedule: str = "gamma",
    score_by_gw: dict[int, dict[str, float]] | None = None,
) -> float:
    """V = Σ γ^h XI_score_h − 4·hits; optional σ / FDR / flow / xp–prior blend.

    ``score_by_gw`` replaces the frozen decision-week score at that horizon
    step. The published path leaves it empty and keeps the freeze.
    """
    from src.models.xp_engine import blend_xp_exp

    h_len = HORIZON if horizon is None else int(horizon)
    v = -float(HIT_COST * hits)
    horizon_gws = [gw] + [g for g in future_gws if g > gw][: h_len - 1]
    for h, g in enumerate(horizon_gws):
        playing = roster_by_gw.get(g, set())
        fc_gw = fixture_counts.get(g, {}) if fixture_counts is not None else None
        fdr_gw = fdr_by_gw.get(g, {}) if fdr_by_gw is not None else None
        rows = []
        for pid in squad_ids:
            m = meta[pid]
            # Leakage-free: freeze decision-GW scores; blank → 0; DGW × fixtures.
            # A supplied step score is already that week's total, including a blank.
            step_scores = score_by_gw.get(g) if score_by_gw is not None else None
            if step_scores is not None and pid in step_scores:
                sc = float(step_scores[pid])
                sc_exp = sc
            elif fc_gw is not None:
                n_fix = int(fc_gw.get(pid, 0))
                sc = float(score_now.get(pid, 0.0)) * n_fix if n_fix > 0 else 0.0
                sc_exp = (
                    float(exp_score_now.get(pid, 0.0)) * n_fix
                    if exp_score_now is not None and n_fix > 0
                    else 0.0
                )
            elif g != gw and pid not in playing:
                sc = 0.0
                sc_exp = 0.0
            else:
                sc = float(score_now.get(pid, -1e6))
                sc_exp = (
                    float(exp_score_now.get(pid, sc))
                    if exp_score_now is not None
                    else sc
                )
            if (
                blend_gamma is not None or blend_schedule == "team_fade"
            ) and exp_score_now is not None and sc != 0.0 and sc > -1e5:
                sc = blend_xp_exp(
                    sc,
                    sc_exp,
                    h,
                    gamma=float(blend_gamma if blend_gamma is not None else 0.9),
                    floor=float(blend_floor),
                    schedule=blend_schedule,
                )
            if sc != 0.0 and sc > -1e5:
                if sigma0 is not None:
                    sc *= precision_weight(
                        float(sigma0.get(pid, UNC_TAU)),
                        h,
                        beta=unc_beta,
                        tau=unc_tau,
                    )
                if fdr_gw is not None:
                    sc *= float(fdr_gw.get(pid, 0.0 if fc_gw is not None else 1.0))
                if flow_by_pid is not None:
                    sc *= float(flow_by_pid.get(pid, 1.0))
            rows.append(
                {
                    "player_id": pid,
                    "position": m["position"],
                    "team_norm": m["team_norm"],
                    score_col: sc,
                }
            )
        sdf = pd.DataFrame(rows)
        v += (GAMMA**h) * _xi_score_sum(sdf, score_col)
    return v


def _apply_swaps(
    state: SquadState,
    sells: list[str],
    buys: list[str],
    price: dict[str, int],
) -> SquadState | None:
    if len(sells) != len(buys):
        return None
    owned = state.ids()
    if any(s not in owned for s in sells):
        return None
    if any(b in owned for b in buys):
        return None
    revenue = sum(sell_price(state.purchase[s], price[s]) for s in sells)
    spend = sum(price[b] for b in buys)
    new_bank = state.bank + revenue - spend
    if new_bank < 0:
        return None
    new_purchase = dict(state.purchase)
    for s in sells:
        del new_purchase[s]
    for b in buys:
        new_purchase[b] = price[b]
    if len(new_purchase) != 15:
        return None
    return SquadState(purchase=new_purchase, bank=int(new_bank), ft=state.ft)


def _one_swap_candidates(
    state: SquadState,
    pool: pd.DataFrame,
    score_col: str,
    *,
    top_n: int = 40,
) -> list[tuple[str, str, SquadState, float]]:
    """Top legal 1-for-1 swaps by raw score Δ (same position only)."""
    df = pool.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["xfer_score"] = _fill_score(df, score_col)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(50).astype(int)
    df = df.drop_duplicates("player_id", keep="first")

    by_id = {str(r.player_id): r for r in df.itertuples()}
    owned = [pid for pid in state.ids() if pid in by_id]
    price = {pid: int(by_id[pid].value) for pid in by_id}
    pos = {pid: str(by_id[pid].position) for pid in by_id}
    club = {pid: str(by_id[pid].team_norm) for pid in by_id}
    sc = {pid: float(by_id[pid].xfer_score) for pid in by_id}

    buys_by_pos: dict[str, list[str]] = {p: [] for p in SQUAD_QUOTA}
    for pid, r in by_id.items():
        if pid in state.ids() or not bool(getattr(r, "eligible", False)):
            continue
        buys_by_pos.setdefault(pos[pid], []).append(pid)
    for p, lst in buys_by_pos.items():
        lst.sort(key=lambda x: sc[x], reverse=True)
        buys_by_pos[p] = lst[:20]

    raw: list[tuple[float, str, str]] = []
    for s in owned:
        for b in buys_by_pos.get(pos[s], []):
            raw.append((sc[b] - sc[s], s, b))
    raw.sort(reverse=True)

    out: list[tuple[str, str, SquadState, float]] = []
    for delta, s, b in raw:
        if len(out) >= top_n:
            break
        new_ids = (state.ids() - {s}) | {b}
        if not _squad_legal([pos[p] for p in new_ids], [club[p] for p in new_ids]):
            continue
        ns = _apply_swaps(state, [s], [b], price)
        if ns is None:
            continue
        out.append((s, b, ns, delta))
    return out


def _two_swap_candidates(
    state: SquadState,
    pool: pd.DataFrame,
    score_col: str,
    *,
    top_n: int = 40,
) -> list[tuple[list[str], list[str], SquadState, float]]:
    """Simultaneous 2-transfers preserving 2/5/5/3 (enables structural cross-pos)."""
    df = pool.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["xfer_score"] = _fill_score(df, score_col)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(50).astype(int)
    df = df.drop_duplicates("player_id", keep="first")

    by_id = {str(r.player_id): r for r in df.itertuples()}
    owned = [pid for pid in state.ids() if pid in by_id]
    price = {pid: int(by_id[pid].value) for pid in by_id}
    pos = {pid: str(by_id[pid].position) for pid in by_id}
    club = {pid: str(by_id[pid].team_norm) for pid in by_id}
    sc = {pid: float(by_id[pid].xfer_score) for pid in by_id}

    buys_by_pos: dict[str, list[str]] = {p: [] for p in SQUAD_QUOTA}
    for pid, r in by_id.items():
        if pid in state.ids() or not bool(getattr(r, "eligible", False)):
            continue
        buys_by_pos.setdefault(pos[pid], []).append(pid)
    for p, lst in buys_by_pos.items():
        lst.sort(key=lambda x: sc[x], reverse=True)
        buys_by_pos[p] = lst[:12]

    # Per-position same-pos legs (seed pairs)
    legs: list[tuple[float, str, str]] = []
    for s in owned:
        for b in buys_by_pos.get(pos[s], []):
            legs.append((sc[b] - sc[s], s, b))
    legs.sort(reverse=True)
    legs = legs[:30]

    # Cross-pos structural: sell two different positions, buy matching counts
    # e.g. sell MID+DEF → buy FWD+DEF (net MID−1 FWD+1) is illegal under fixed quotas.
    # Legal cross only when buy positions == sell positions (permutation).
    # So structural value = sell expensive MID + cheap FWD, buy cheap MID + expensive FWD.
    owned_by_pos: dict[str, list[str]] = {p: [] for p in SQUAD_QUOTA}
    for s in owned:
        owned_by_pos.setdefault(pos[s], []).append(s)
    for p, lst in owned_by_pos.items():
        lst.sort(key=lambda x: sc[x])  # weakest first

    cross_legs: list[tuple[float, str, str, str, str]] = []
    pos_pairs = [("MID", "FWD"), ("MID", "DEF"), ("DEF", "FWD"), ("GKP", "DEF")]
    for p1, p2 in pos_pairs:
        sells1 = owned_by_pos.get(p1, [])[:4]
        sells2 = owned_by_pos.get(p2, [])[:4]
        buys1 = buys_by_pos.get(p1, [])[:8]
        buys2 = buys_by_pos.get(p2, [])[:8]
        for s1 in sells1:
            for s2 in sells2:
                if s1 == s2:
                    continue
                for b1 in buys1:
                    for b2 in buys2:
                        if b1 == b2:
                            continue
                        delta = (sc[b1] - sc[s1]) + (sc[b2] - sc[s2])
                        if delta <= 0:
                            continue
                        cross_legs.append((delta, s1, s2, b1, b2))
    cross_legs.sort(reverse=True)
    cross_legs = cross_legs[:50]

    raw_pairs: list[tuple[float, list[str], list[str]]] = []
    # Same-pos sequential pairs from top legs
    for i, (d1, s1, b1) in enumerate(legs):
        for d2, s2, b2 in legs[i + 1 :]:
            if s1 == s2 or b1 == b2:
                continue
            raw_pairs.append((d1 + d2, [s1, s2], [b1, b2]))
    for delta, s1, s2, b1, b2 in cross_legs:
        raw_pairs.append((delta, [s1, s2], [b1, b2]))
    raw_pairs.sort(reverse=True)

    out: list[tuple[list[str], list[str], SquadState, float]] = []
    seen: set[tuple[str, ...]] = set()
    for delta, sells, buys in raw_pairs:
        if len(out) >= top_n:
            break
        key = tuple(sorted(sells) + sorted(buys))
        if key in seen:
            continue
        seen.add(key)
        new_ids = (state.ids() - set(sells)) | set(buys)
        if len(new_ids) != 15:
            continue
        if not _squad_legal([pos[p] for p in new_ids], [club[p] for p in new_ids]):
            continue
        ns = _apply_swaps(state, sells, buys, price)
        if ns is None:
            continue
        out.append((sells, buys, ns, delta))
    return out


def choose_transfers(
    state: SquadState,
    gw_df: pd.DataFrame,
    score_col: str,
    gw: int,
    future_gws: list[int],
    roster_by_gw: dict[int, set[str]],
    *,
    horizon: int | None = None,
    fixture_counts: dict[int, dict[str, int]] | None = None,
    structural_2tx: bool = False,
    sigma0: dict[str, float] | None = None,
    fdr_by_gw: dict[int, dict[str, float]] | None = None,
    flow_by_pid: dict[str, float] | None = None,
    unc_beta: float = UNC_BETA,
    unc_tau: float = UNC_TAU,
    exp_score_col: str | None = None,
    blend_gamma: float | None = None,
    blend_floor: float = 0.5,
    blend_schedule: str = "gamma",
    hold_eps: float | None = None,
    switch_penalty: float | None = None,
    score_by_gw: dict[int, dict[str, float]] | None = None,
) -> tuple[SquadState, int, int]:
    """Argmax V over hold / 1-swaps / optional structural 2-transfers."""
    eps = HOLD_EPS if hold_eps is None else float(hold_eps)
    pen = SWITCH_PENALTY if switch_penalty is None else float(switch_penalty)
    df = gw_df.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["xfer_score"] = _fill_score(df, score_col)
    if exp_score_col is not None and exp_score_col in df.columns:
        df["xfer_exp"] = _fill_score(df, exp_score_col)
    else:
        df["xfer_exp"] = df["xfer_score"]
    df = df.drop_duplicates("player_id", keep="first")
    by_id = {str(r.player_id): r for r in df.itertuples()}

    use_blend = blend_schedule == "team_fade" or blend_gamma is not None

    def meta_for(ids: set[str]) -> dict[str, dict[str, Any]]:
        return {
            pid: {
                "position": str(by_id[pid].position),
                "team_norm": str(by_id[pid].team_norm),
            }
            for pid in ids
            if pid in by_id
        }

    def scores_for(ids: set[str]) -> dict[str, float]:
        return {pid: float(by_id[pid].xfer_score) for pid in ids if pid in by_id}

    def exp_scores_for(ids: set[str]) -> dict[str, float] | None:
        if not use_blend:
            return None
        return {pid: float(by_id[pid].xfer_exp) for pid in ids if pid in by_id}

    def value_of(st: SquadState, n_tx: int) -> float | None:
        ids = st.ids()
        if len(ids) != 15 or any(pid not in by_id for pid in ids):
            return None
        hits = max(0, n_tx - state.ft)
        if hits > MAX_HITS:
            return None
        base = transfer_value(
            ids,
            scores_for(ids),
            meta_for(ids),
            hits,
            gw,
            future_gws,
            roster_by_gw,
            score_col,
            horizon=horizon,
            fixture_counts=fixture_counts,
            sigma0=sigma0,
            fdr_by_gw=fdr_by_gw,
            flow_by_pid=flow_by_pid,
            unc_beta=unc_beta,
            unc_tau=unc_tau,
            exp_score_now=exp_scores_for(ids),
            blend_gamma=blend_gamma if blend_schedule != "team_fade" else 0.9,
            blend_floor=blend_floor,
            blend_schedule=blend_schedule,
            score_by_gw=score_by_gw,
        )
        return base - pen * n_tx

    hold_v = value_of(state, 0)
    if hold_v is None:
        return state, 0, 0
    hold_legal = _squad_legal(
        [str(by_id[pid].position) for pid in state.ids()],
        [str(by_id[pid].team_norm) for pid in state.ids()],
    )
    # A legal hold is the baseline. An illegal hold (a player changed club and
    # the squad is now over the cap) is not a baseline: any legal squad beats it.
    if hold_legal:
        best_st, best_n, best_hits, best_v = state, 0, 0, hold_v
    else:
        best_st, best_n, best_hits, best_v = None, 0, 0, -1e18
    max_tx = min(state.ft + MAX_HITS, 3)

    # Depth 1: evaluate top score-Δ swaps under full V
    depth1: list[tuple[float, SquadState, int]] = []
    for _s, _b, st1, _delta in _one_swap_candidates(state, gw_df, score_col, top_n=35):
        val = value_of(st1, 1)
        if val is None:
            continue
        depth1.append((val, st1, 1))
        if val > best_v:
            best_st, best_n, best_hits, best_v = st1, 1, max(0, 1 - state.ft), val

    if max_tx >= 2:
        # Simultaneous structural / same-pos 2-transfers
        if structural_2tx:
            for _sells, _buys, st2, _d in _two_swap_candidates(
                state, gw_df, score_col, top_n=40
            ):
                val = value_of(st2, 2)
                if val is None:
                    continue
                if val > best_v:
                    best_st, best_n, best_hits, best_v = (
                        st2,
                        2,
                        max(0, 2 - state.ft),
                        val,
                    )

        if depth1:
            depth1.sort(key=lambda x: x[0], reverse=True)
            for _val0, st0, n0 in depth1[:BEAM_WIDTH]:
                for _s, _b, st1, _d in _one_swap_candidates(
                    st0, gw_df, score_col, top_n=20
                ):
                    n_tx = n0 + 1
                    if n_tx > max_tx:
                        continue
                    val = value_of(st1, n_tx)
                    if val is None:
                        continue
                    if val > best_v:
                        best_st, best_n, best_hits, best_v = (
                            st1,
                            n_tx,
                            max(0, n_tx - state.ft),
                            val,
                        )

    if max_tx >= 3 and best_n >= 2:
        st0 = best_st if best_n == 2 else None
        if st0 is not None:
            for _s, _b, st1, _d in _one_swap_candidates(
                st0, gw_df, score_col, top_n=15
            ):
                val = value_of(st1, 3)
                if val is not None and val > best_v:
                    best_st, best_n, best_hits, best_v = (
                        st1,
                        3,
                        max(0, 3 - state.ft),
                        val,
                    )

    if not hold_legal:
        if best_st is None:
            return state, 0, 0
        return best_st, best_n, best_hits
    if best_n > 0 and best_v < hold_v + eps:
        return state, 0, 0
    return best_st, best_n, best_hits


def initial_squad(
    gw_df: pd.DataFrame, score_col: str
) -> SquadState:
    pool = gw_df.loc[gw_df["eligible"]].copy()
    pool[score_col] = _fill_score(pool, score_col)
    squad = pick_squad(pool, score_col, budget=BUDGET)
    purchase = {
        str(r.player_id): _as_int_value(r.value)
        for r in squad.itertuples()
    }
    spent = sum(purchase.values())
    return SquadState(purchase=purchase, bank=max(0, BUDGET - spent), ft=1)


def rebuild_squad(state: SquadState, pool: pd.DataFrame, score_col: str) -> SquadState:
    """Wildcard or Free Hit squad, paid from the bank plus sell prices.

    A kept player's purchase price stays. A new player is bought at the
    current price. This is not a fresh £100.0m.
    """
    df = pool.copy()
    df["player_id"] = df["player_id"].astype(str)
    df = df.drop_duplicates("player_id", keep="first")
    owned = state.ids()
    missing = owned - set(df["player_id"])
    if missing:
        raise RuntimeError("owned players missing from the chip pool")
    market = {
        str(r.player_id): _as_int_value(r.value) for r in df.itertuples()
    }
    sell = {pid: sell_price(state.purchase[pid], market[pid]) for pid in owned}
    budget = int(state.bank + sum(sell.values()))
    df["value"] = [sell[pid] if pid in sell else market[pid] for pid in df["player_id"]]
    df[score_col] = _fill_score(df, score_col)
    chosen = pick_squad(df, score_col, budget=budget)
    purchase: dict[str, int] = {}
    spent = 0
    for row in chosen.itertuples():
        pid = str(row.player_id)
        if pid in state.purchase:
            purchase[pid] = state.purchase[pid]
            spent += sell[pid]
        else:
            purchase[pid] = market[pid]
            spent += market[pid]
    return SquadState(purchase=purchase, bank=int(budget - spent), ft=state.ft)


def _chip_additions(
    squad_df: pd.DataFrame, banked: dict[str, Any], chip: str
) -> tuple[float, float]:
    """Captain extra and bench-boost extra, on top of the XI sum."""
    mins = {
        str(r.player_id): float(getattr(r, "minutes") or 0) for r in squad_df.itertuples()
    }
    pts = {
        str(r.player_id): float(getattr(r, "total_points") or 0)
        for r in squad_df.itertuples()
    }
    cap_id = str(banked["captain_id"])
    vc_id = str(banked["vice_id"])
    extra = captain_extra_points(
        pts.get(cap_id, 0.0),
        pts.get(vc_id, 0.0),
        captain_played=mins.get(cap_id, 0.0) > 0,
        vice_played=mins.get(vc_id, 0.0) > 0,
        chip=chip,
    )
    bench = 0.0
    if chip == "bench_boost":
        final_ids = set(banked["xi"]["player_id"].astype(str))
        bench = float(sum(points for pid, points in pts.items() if pid not in final_ids))
    return extra, bench


def run_ft_season(
    feat: pd.DataFrame,
    score_cols: dict[str, str],
    gws: list[int],
    roster: pd.DataFrame | None = None,
    *,
    horizon: int | None = None,
    exp_score_col: str | None = None,
    blend_gamma: float | None = None,
    blend_floor: float = 0.5,
    blend_schedule: str = "gamma",
    method_suffix: str = "_ft",
    hold_eps: float | None = None,
    trace: list[dict[str, Any]] | None = None,
    value_col: str | None = None,
    switch_penalty: float | None = None,
    chips: dict[int, str] | None = None,
    opening: SquadState | None = None,
    horizon_scores: Any | None = None,
) -> pd.DataFrame:
    """``chips`` maps a gameweek to one chip name. None and {} play nothing.

    ``opening`` is a 15-man squad already owned at the first gameweek.
    That week is scored with no transfers. Later weeks use the normal rule.

    ``horizon_scores(gw, pool, gws)`` returns a step-score map for that
    deadline. None keeps the frozen decision-week score.
    """
    rows: list[dict[str, Any]] = []
    plan = validate_chip_map(chips)
    if roster is None:
        roster = load_vaastav_roster(EVAL_SEASON)
    if opening is not None and len(opening.purchase) != 15:
        raise RuntimeError("opening squad must contain 15 players")

    roster_by_gw: dict[int, set[str]] = {
        int(g): set(gdf["player_id"].astype(str))
        for g, gdf in roster.groupby("gw")
    }

    for method, col in score_cols.items():
        state: SquadState | None = deepcopy(opening) if opening is not None else None
        for i, gw in enumerate(gws):
            owned = state.ids() if state else set()
            pool = _gw_pool(feat, roster, gw, owned)
            if pool["position"].nunique() < 4:
                continue

            chip = plan.get(int(gw))
            restore: SquadState | None = None
            if state is not None and i == 0 and opening is not None and chip not in FREE_TRANSFER_CHIPS:
                n_tx, hits = 0, 0
                ft_before = 0
            elif state is None:
                try:
                    state = initial_squad(pool, col)
                except RuntimeError:
                    break
                n_tx, hits = 0, 0
                ft_before = 0
            elif chip in FREE_TRANSFER_CHIPS:
                ft_before = state.ft
                previous = state.ids()
                if chip == "free_hit":
                    restore = deepcopy(state)
                try:
                    state = rebuild_squad(state, pool, value_col or col)
                except RuntimeError:
                    break
                n_tx = len(state.ids() - previous)
                hits = 0
            else:
                ft_before = state.ft
                step_scores = None
                if horizon_scores is not None:
                    step_scores = horizon_scores(int(gw), pool, list(gws))
                new_state, n_tx, hits = choose_transfers(
                    state,
                    pool,
                    value_col or col,
                    gw=int(gw),
                    future_gws=list(gws),
                    roster_by_gw=roster_by_gw,
                    horizon=horizon,
                    exp_score_col=exp_score_col,
                    blend_gamma=blend_gamma,
                    blend_floor=blend_floor,
                    blend_schedule=blend_schedule,
                    hold_eps=hold_eps,
                    switch_penalty=switch_penalty,
                    score_by_gw=step_scores,
                )
                state = new_state

            squad_df = pool.loc[pool["player_id"].isin(state.ids())].copy()
            squad_df = squad_df.drop_duplicates("player_id", keep="first")
            if len(squad_df) < 11 or len(state.purchase) != 15:
                break
            squad_df[col] = _fill_score(squad_df, col)
            try:
                banked = bank_squad_gw(squad_df, col, use_autosubs=True)
            except RuntimeError:
                break
            if trace is not None:
                xi_intended, _form_intended = pick_xi(squad_df, col)
                trace.append(
                    {
                        "gw": int(gw),
                        "method": f"{method}{method_suffix}",
                        "xi": xi_intended.copy(),
                        "squad": squad_df.copy(),
                        "final_xi": banked["xi"].copy(),
                        "captain_id": banked["captain_id"],
                        "vice_id": banked["vice_id"],
                        "cap_extra": float(banked["cap_extra"]),
                    }
                )
            form = banked["form"]
            if chip is None:
                cap_pts = float(banked["cap_extra"])
                bench_pts = 0.0
            else:
                cap_pts, bench_pts = _chip_additions(squad_df, banked, chip)
            pts_sum = float(banked["xi_points"]) + bench_pts
            hit_pts = 0.0 if chip in FREE_TRANSFER_CHIPS else float(HIT_COST * hits)
            if i == 0:
                ft_after = 1
            elif chip in FREE_TRANSFER_CHIPS:
                ft_after = advance_ft(ft_before, 0, chip=chip)
            else:
                ft_after = advance_ft(ft_before, n_tx)
            squad_val = float(
                sum(
                    sell_price(
                        state.purchase[str(r.player_id)],
                        _as_int_value(r.value),
                    )
                    for r in squad_df.itertuples()
                )
            )

            rows.append(
                {
                    "gw": int(gw),
                    "method": f"{method}{method_suffix}",
                    "mode": "ft",
                    "xi_points": pts_sum,
                    "xi_points_cap": pts_sum + cap_pts - hit_pts,
                    "hit_cost": hit_pts,
                    "n_transfers": n_tx,
                    "hits": hits,
                    "ft_before": ft_before,
                    "ft_after": ft_after,
                    "chip": chip,
                    "bank": state.bank,
                    "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
                    "squad_sell_value": squad_val,
                    "n_eligible": int(pool["eligible"].sum()),
                    "n_autosubs": int(banked["n_autosubs"]),
                    "n_blank_intended": int(banked["n_blank_intended"]),
                    "n_blank_final": int(banked["n_blank_final"]),
                    "sub_points": float(banked["sub_points"]),
                }
            )
            if restore is not None:
                state = restore
            state.ft = ft_after

    return pd.DataFrame(rows)


def plot_climb(weekly: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    style = {
        "ridge_global_starters_ft": ("ridge_ft", "crimson", 2.6),
        "ridge_global_starters_budget": ("ridge_budget", "tomato", 1.6),
        "xp_ft": ("xp_ft", "steelblue", 2.0),
        "xp_budget": ("xp_budget", "lightblue", 1.4),
        "exp_points_ft": ("exp_ft", "gray", 1.8),
        "exp_points_budget": ("exp_budget", "silver", 1.3),
    }
    for method, (label, color, lw) in style.items():
        g = weekly.loc[weekly["method"] == method].sort_values("gw")
        if g.empty:
            continue
        ax.plot(
            g["gw"],
            g["xi_points_cap"].cumsum(),
            "-o",
            ms=3,
            lw=lw,
            color=color,
            label=label,
        )
    ax.set_xlabel("Gameweek (2025/26)")
    ax.set_ylabel("Cumulative XI points (cap×2 − hits)")
    ax.set_title(
        f"FT climb — XI-horizon V (H={HORIZON}) vs budget rebuild"
    )
    ax.legend(fontsize=8, loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def write_report(
    path: Path, summary: pd.DataFrame, weekly: pd.DataFrame, scored: list[int]
) -> None:
    def tot(name: str) -> float:
        hit = summary.loc[summary["method"] == name, "total_points"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    ridge_ft = tot("ridge_global_starters_ft")
    ridge_b = tot("ridge_global_starters_budget")
    xp_ft = tot("xp_ft")
    exp_ft = tot("exp_points_ft")

    ft_rows = weekly.loc[weekly["method"] == "ridge_global_starters_ft"]
    mean_tx = float(ft_rows["n_transfers"].mean()) if len(ft_rows) else float("nan")
    total_hits = float(ft_rows["hits"].sum()) if len(ft_rows) else float("nan")
    hit_pts = float(ft_rows["hit_cost"].sum()) if len(ft_rows) else float("nan")

    lines = [
        "# Stage 19 — Transfer-constrained season climb",
        "",
        "FPL rules (chips off unless a week map is passed):",
        "",
        "- **First scored GW:** free 15 under £100.0m (wildcard-like)",
        f"- **Thereafter:** 1 FT / GW, stack to **{MAX_FT}**; extras **−{HIT_COST}** each "
        f"(explore 0–{MAX_HITS} hits)",
        "- **Sell price:** purchase + ⌊rise/2⌋; full fall to current (`value`)",
        "- **Transfer policy:** enumerate hold / swaps; "
        f"V = Σ γ^h XI_score (H={HORIZON}, γ={GAMMA}) − 4·hits; "
        f"hold unless ΔV ≥ {HOLD_EPS}",
        "- Scores frozen from decision GW; future blanks (missing Vaastav row) → 0",
        "- Buy pool = eligible ∪ currently owned; XI from squad by score",
        "",
        f"- GWs: **{scored[0]}–{scored[-1]}** (n={len(scored)})",
        "",
        "## Final standings (captain ×2 − hits)",
        "",
        "| method | mode | total | mean/GW | vs ridge_ft |",
        "|---|---|---:|---:|---:|",
    ]
    for r in summary.itertuples():
        mode = "ft" if str(r.method).endswith("_ft") else "budget"
        lines.append(
            f"| {r.method} | {mode} | {r.total_points:.0f} | {r.mean_gw:.1f} | "
            f"{r.total_points - ridge_ft:+.0f} |"
        )

    lines += [
        "",
        "## Transfer friction",
        "",
        f"- Ridge FT mean transfers/GW: **{mean_tx:.2f}**",
        f"- Ridge FT hold weeks: **{int((ft_rows['n_transfers'] == 0).sum())}** "
        f"/ {len(ft_rows)}",
        f"- Ridge FT max ft_before seen: **{ft_rows['ft_before'].max():.0f}**",
        f"- Ridge FT total hits: **{total_hits:.0f}** (−{hit_pts:.0f} pts)",
        f"- Ridge FT vs budget rebuild: **{ridge_ft - ridge_b:+.0f}**",
        f"- Ridge FT vs xp FT: **{ridge_ft - xp_ft:+.0f}**",
        f"- Ridge FT vs exp FT: **{ridge_ft - exp_ft:+.0f}**",
        "",
    ]

    if ridge_ft > exp_ft + 10 and ridge_ft > xp_ft:
        verdict = (
            f"PASS — ridge_ft beats exp_ft ({ridge_ft - exp_ft:+.0f}) "
            f"and xp_ft ({ridge_ft - xp_ft:+.0f}); "
            f"vs budget rebuild {ridge_ft - ridge_b:+.0f}"
        )
    elif ridge_ft > exp_ft + 10:
        verdict = (
            f"PARTIAL — beats exp_ft but not xp_ft "
            f"(vs xp {ridge_ft - xp_ft:+.0f})"
        )
    else:
        verdict = "FAIL — ridge_ft trails baselines under FT constraints"

    lines += [
        "## Gate verdict",
        "",
        f"**{verdict}**",
        "",
        "## Plots",
        "",
        "- `data/plots/season_climb_ft.png`",
        "",
        "## Output",
        "",
        "- `data/processed/season_climb_ft.csv`",
        "",
        "## Read",
        "",
        "- FT stack + hits is the real decision surface; budget rebuild was an upper bound.",
        "- Large FT≪budget gap ⇒ value is locked in transfer timing, not just ranking.",
        f"- Policy uses XI-horizon V (not Σ15); hold unless ΔV ≥ {HOLD_EPS}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    print("Building features + walk-forward ridge_global_starters…")
    all_feat = build_fresh_seasons()
    pred, scored = walk_forward_global_ridge_starters(all_feat)
    if not scored:
        raise RuntimeError("No scored GWs")

    eval_feat = all_feat.loc[all_feat["season"] == EVAL_SEASON].copy()
    eval_feat["score_ridge_global_starters"] = pred.loc[eval_feat.index]

    score_cols = {
        "ridge_global_starters": "score_ridge_global_starters",
        "xp": "score_xp",
        "exp_points": "score_exp_points",
    }

    roster = load_vaastav_roster(EVAL_SEASON)
    print(f"FT climb on GWs {scored[0]}–{scored[-1]} (n={len(scored)})…")
    weekly_ft = run_ft_season(eval_feat, score_cols, scored, roster=roster)
    print("Budget rebuild climb for comparison…")
    weekly_b = run_budgeted_season(eval_feat, score_cols, scored)
    weekly = pd.concat([weekly_ft, weekly_b], ignore_index=True)

    counts = weekly.groupby("method")["gw"].nunique()
    weekly = weekly.loc[weekly["method"].isin(counts[counts >= len(scored)].index)].copy()
    summary = summarize(weekly)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "season_climb_ft.csv", index=False)
    plot_climb(weekly, PLOTS / "season_climb_ft.png")
    write_report(REPORTS / "stage_19_season_climb_ft.md", summary, weekly, scored)
    return {"summary": summary, "weekly": weekly, "scored_gws": scored}


if __name__ == "__main__":
    out = run()
    print(out["summary"].to_string(index=False))
    print(f"\nWrote {REPORTS}/stage_19_season_climb_ft.md")
