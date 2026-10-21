"""Lineup / minutes worlds before attack dice (short-horizon).

Layer 1 of the generative stack:

1. Detect competitor pairs from same-club minutes correlation.
2. Per path, for each pair either:
   - **exclusive:** winner 90′, loser 0′ (rotation / either-or XI), or
   - **starter_sub:** winner plays part of the match, loser comes on for the
     rest (Gyökeres 60′ / Havertz 30′). Both can have minutes on one path.
3. CS eligibility stays ``minutes ≥ 60``; attack dice gate on ``minutes > 0``.

When the XI is uncertain, both members of a pair still appear across paths.
When one is clearly nailed in the window, the mixture still skews toward them
via mean-minute weights (plus a small prior so a cold striker is not wiped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.rules.fpl_2026 import normalize_position

START_GATE = 60.0  # CS / 60′ appearance bonus gate
CORR_THRESH = -0.40
PRIOR_MINUTES = 30.0  # softens cold-start wipe in exclusive mixtures
MIN_PAIR_MEAN = 5.0
# P(starter+sub share the match | pair rolled) vs exclusive 90/0.
# Fixed prior for lock; as-of pair rates can replace later.
P_SUB_ON = 0.45
SUB_STARTER_LO = 55.0  # starter minutes when sub comes on
SUB_STARTER_HI = 80.0
MATCH_MINUTES = 90.0


@dataclass(frozen=True)
class CompetitorPair:
    i: int  # row index on the board
    j: int
    corr: float
    name_i: str
    name_j: str


@dataclass(frozen=True)
class LineupWorld:
    """Minutes for one club on one path."""

    minutes: np.ndarray
    eligible: np.ndarray  # bool, minutes >= START_GATE (CS gate)
    pair_choices: tuple[tuple[str, str, str], ...]  # (winner, loser, mode|pair)

    def as_dict(self, names: Sequence[str]) -> dict[str, Any]:
        return {
            "minutes": {
                str(names[i]): float(self.minutes[i]) for i in range(len(names))
            },
            "eligible": [str(names[i]) for i in range(len(names)) if self.eligible[i]],
            "pair_choices": [
                {"winner": w, "loser": lo, "pair": lab} for w, lo, lab in self.pair_choices
            ],
        }


def minutes_corr_pairs(
    history: pd.DataFrame,
    *,
    id_col: str = "player_id",
    name_col: str = "player_name",
    minutes_col: str = "minutes",
    gw_col: str = "gw",
    corr_thresh: float = CORR_THRESH,
) -> pd.DataFrame:
    """Off-diagonal minutes correlations for players in ``history`` (one club)."""
    if history.empty:
        return pd.DataFrame(
            columns=["id_a", "id_b", "name_a", "name_b", "pos_a", "pos_b", "corr", "mean_a", "mean_b"]
        )
    h = history.copy()
    h[minutes_col] = pd.to_numeric(h[minutes_col], errors="coerce").fillna(0.0)
    panel = h.pivot_table(
        index=gw_col, columns=id_col, values=minutes_col, aggfunc="sum"
    ).fillna(0.0)
    if panel.shape[1] < 2 or panel.shape[0] < 3:
        return pd.DataFrame(
            columns=["id_a", "id_b", "name_a", "name_b", "pos_a", "pos_b", "corr", "mean_a", "mean_b"]
        )
    meta = (
        h.groupby(id_col)
        .agg(name=(name_col, "last"), position=("position", "last"))
        .reset_index()
    )
    meta["position"] = meta["position"].map(
        lambda p: normalize_position("GKP" if str(p) == "GK" else str(p))
    )
    name_of = dict(zip(meta[id_col].astype(int), meta["name"].astype(str)))
    pos_of = dict(zip(meta[id_col].astype(int), meta["position"].astype(str)))
    C = panel.corr()
    ids = [int(x) for x in panel.columns.tolist()]
    rows = []
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1 :]:
            c = float(C.loc[a, b]) if a in C.index and b in C.columns else float("nan")
            if not np.isfinite(c) or c > float(corr_thresh):
                continue
            ma = float(panel[a].mean())
            mb = float(panel[b].mean())
            if max(ma, mb) < MIN_PAIR_MEAN:
                continue
            pa, pb = pos_of.get(a, ""), pos_of.get(b, "")
            if not _valid_competitor_positions(pa, pb):
                continue
            rows.append(
                {
                    "id_a": a,
                    "id_b": b,
                    "name_a": name_of.get(a, str(a)),
                    "name_b": name_of.get(b, str(b)),
                    "pos_a": pa,
                    "pos_b": pb,
                    "corr": c,
                    "mean_a": ma,
                    "mean_b": mb,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("corr").reset_index(drop=True)


def _valid_competitor_positions(pa: str, pb: str) -> bool:
    """Same-role competition only (avoid DEF↔FWD artefacts)."""
    pa = normalize_position(pa)
    pb = normalize_position(pb)
    return pa == pb


def select_disjoint_pairs(
    board_ids: Sequence[int],
    pair_table: pd.DataFrame,
    *,
    max_pairs: int = 8,
) -> list[CompetitorPair]:
    """Greedy strongest-|corr| matching; each player in at most one pair.

    Same-position pairs only (GK↔GK, FWD↔FWD, …). Prefer more negative corr.
    """
    id_to_row = {int(pid): i for i, pid in enumerate(board_ids)}
    used: set[int] = set()
    chosen: list[CompetitorPair] = []
    if pair_table is None or pair_table.empty:
        return chosen
    ordered = pair_table.sort_values("corr")
    for _, r in ordered.iterrows():
        a, b = int(r["id_a"]), int(r["id_b"])
        if a not in id_to_row or b not in id_to_row:
            continue
        if a in used or b in used:
            continue
        if str(r.get("pos_a")) != str(r.get("pos_b")):
            continue
        chosen.append(
            CompetitorPair(
                i=id_to_row[a],
                j=id_to_row[b],
                corr=float(r["corr"]),
                name_i=str(r["name_a"]),
                name_j=str(r["name_b"]),
            )
        )
        used.add(a)
        used.add(b)
        if len(chosen) >= int(max_pairs):
            break
    return chosen


def _rng_uniform(rng: Any, lo: float, hi: float) -> float:
    if hasattr(rng, "uniform"):
        return float(rng.uniform(lo, hi))
    # random.Random
    return float(lo) + (float(hi) - float(lo)) * float(rng.random())


def _rng_random(rng: Any) -> float:
    if hasattr(rng, "random"):
        return float(rng.random())
    return float(np.asarray(rng.random()))


def roll_lineup_world(
    *,
    mean_m: np.ndarray | Sequence[float],
    names: Sequence[str],
    pairs: Sequence[CompetitorPair],
    rng: Any,
    start_minutes: float = MATCH_MINUTES,
    bench_minutes: float = 0.0,
    prior_minutes: float = PRIOR_MINUTES,
    start_gate: float = START_GATE,
    p_sub_on: float = P_SUB_ON,
    sub_starter_lo: float = SUB_STARTER_LO,
    sub_starter_hi: float = SUB_STARTER_HI,
) -> LineupWorld:
    """Roll competitor-pair minutes: exclusive XI or starter+sub share.

    Non-paired players keep mean minutes. ``p_sub_on=0`` recovers the old
    exclusive-only behaviour (winner ``start_minutes``, loser ``bench_minutes``).
    """
    m0 = np.asarray(mean_m, dtype=float).copy()
    n = m0.size
    minutes = m0.copy()
    choices: list[tuple[str, str, str]] = []
    p_sub = float(np.clip(p_sub_on, 0.0, 1.0))
    lo = float(min(sub_starter_lo, sub_starter_hi))
    hi = float(max(sub_starter_lo, sub_starter_hi))
    match_m = float(start_minutes)

    for pair in pairs:
        i, j = int(pair.i), int(pair.j)
        if not (0 <= i < n and 0 <= j < n):
            continue
        w_i = max(float(m0[i]), 0.0) + float(prior_minutes)
        w_j = max(float(m0[j]), 0.0) + float(prior_minutes)
        p_i = w_i / (w_i + w_j) if (w_i + w_j) > 0 else 0.5
        if _rng_random(rng) < p_i:
            winner, loser = i, j
            w_name, l_name = pair.name_i, pair.name_j
        else:
            winner, loser = j, i
            w_name, l_name = pair.name_j, pair.name_i

        if p_sub > 0.0 and _rng_random(rng) < p_sub:
            # Starter plays most of the match; pair comes on for the rest.
            starter_m = _rng_uniform(rng, lo, hi)
            starter_m = float(np.clip(starter_m, 0.0, match_m))
            sub_m = float(max(0.0, match_m - starter_m))
            minutes[winner] = starter_m
            minutes[loser] = sub_m
            mode = f"starter_sub:{pair.name_i}/{pair.name_j}"
        else:
            minutes[winner] = float(start_minutes)
            minutes[loser] = float(bench_minutes)
            mode = f"exclusive:{pair.name_i}/{pair.name_j}"
        choices.append((w_name, l_name, mode))

    eligible = minutes >= float(start_gate)
    return LineupWorld(
        minutes=minutes,
        eligible=eligible,
        pair_choices=tuple(choices),
    )


def gate_dice_weights(
    weights: np.ndarray,
    eligible: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Zero ineligible players and renormalise (compositional)."""
    w = np.asarray(weights, dtype=float).copy()
    el = np.asarray(eligible, dtype=bool)
    w = np.where(el, np.maximum(w, 0.0), 0.0)
    s = float(w.sum())
    if s <= eps:
        # Nobody cleared the gate — fall back to uniform on eligible, else uniform all
        if el.any():
            w = el.astype(float)
            w /= float(w.sum())
        else:
            w[:] = 1.0 / max(w.size, 1)
        return w
    return w / s
