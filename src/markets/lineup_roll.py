"""Lineup / minutes worlds before attack dice (short-horizon).

Layer 1 of the generative stack:

1. Detect competitor pairs from same-club minutes correlation.
2. Per path, resample each pair's minutes from their **as-of historical
   joint** ``(m_i, m_j)`` on gameweeks where either played. That carries
   rotation (90/0), starter+sub (60/30), and other patterns without a
   hand-tuned ``P_SUB_ON``. Thin history leaves the pair on mean minutes.
3. CS eligibility stays ``minutes ≥ 60``; attack dice gate on ``minutes > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.rules.fpl_2026 import normalize_position

START_GATE = 60.0  # CS / 60′ appearance bonus gate
CORR_THRESH = -0.40
PRIOR_MINUTES = 30.0  # unused for joint resample; kept for callers/tests
MIN_PAIR_MEAN = 5.0
MIN_JOINT_OBS = 3  # need this many past (m_i, m_j) draws to resample
MATCH_MINUTES = 90.0


@dataclass(frozen=True)
class CompetitorPair:
    i: int  # row index on the board
    j: int
    corr: float
    name_i: str
    name_j: str
    id_i: int = 0
    id_j: int = 0
    # Historical joint minutes (m_i, m_j) for GWs where either played.
    joints: tuple[tuple[float, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LineupWorld:
    """Minutes for one club on one path."""

    minutes: np.ndarray
    eligible: np.ndarray  # bool, minutes >= START_GATE (CS gate)
    pair_choices: tuple[tuple[str, str, str], ...]  # (name_a, name_b, mode)

    def as_dict(self, names: Sequence[str]) -> dict[str, Any]:
        return {
            "minutes": {
                str(names[i]): float(self.minutes[i]) for i in range(len(names))
            },
            "eligible": [str(names[i]) for i in range(len(names)) if self.eligible[i]],
            "pair_choices": [
                {"a": a, "b": b, "mode": lab} for a, b, lab in self.pair_choices
            ],
        }


def minutes_panel(
    history: pd.DataFrame,
    *,
    id_col: str = "player_id",
    minutes_col: str = "minutes",
    gw_col: str = "gw",
) -> pd.DataFrame:
    """GW × player minutes matrix (0-filled)."""
    if history is None or history.empty:
        return pd.DataFrame()
    h = history.copy()
    h[minutes_col] = pd.to_numeric(h[minutes_col], errors="coerce").fillna(0.0)
    return (
        h.pivot_table(index=gw_col, columns=id_col, values=minutes_col, aggfunc="sum")
        .fillna(0.0)
    )


def joint_minutes_for_ids(
    panel: pd.DataFrame,
    id_a: int,
    id_b: int,
) -> tuple[tuple[float, float], ...]:
    """As-of joint ``(m_a, m_b)`` on GWs where either player had minutes > 0."""
    if panel is None or panel.empty:
        return ()
    if id_a not in panel.columns or id_b not in panel.columns:
        return ()
    a = panel[id_a].to_numpy(float)
    b = panel[id_b].to_numpy(float)
    out: list[tuple[float, float]] = []
    for ma, mb in zip(a, b):
        if ma > 0.0 or mb > 0.0:
            out.append((float(ma), float(mb)))
    return tuple(out)


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
    panel = minutes_panel(h, id_col=id_col, minutes_col=minutes_col, gw_col=gw_col)
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
    panel: pd.DataFrame | None = None,
) -> list[CompetitorPair]:
    """Greedy strongest-|corr| matching; each player in at most one pair.

    Attaches as-of joint minute samples from ``panel`` when provided.
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
        joints = joint_minutes_for_ids(panel, a, b) if panel is not None else ()
        chosen.append(
            CompetitorPair(
                i=id_to_row[a],
                j=id_to_row[b],
                corr=float(r["corr"]),
                name_i=str(r["name_a"]),
                name_j=str(r["name_b"]),
                id_i=a,
                id_j=b,
                joints=joints,
            )
        )
        used.add(a)
        used.add(b)
        if len(chosen) >= int(max_pairs):
            break
    return chosen


def _rng_choice_index(rng: Any, n: int) -> int:
    if n <= 0:
        raise ValueError("empty choice")
    if hasattr(rng, "integers"):
        return int(rng.integers(0, n))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(n))
    return int(np.asarray(rng.integers(0, n)))


def roll_lineup_world(
    *,
    mean_m: np.ndarray | Sequence[float],
    names: Sequence[str],
    pairs: Sequence[CompetitorPair],
    rng: Any,
    start_gate: float = START_GATE,
    min_joint_obs: int = MIN_JOINT_OBS,
) -> LineupWorld:
    """Resample competitor-pair minutes from as-of historical joints.

    Non-paired players (and pairs with thin joint history) keep mean minutes.
    """
    m0 = np.asarray(mean_m, dtype=float).copy()
    n = m0.size
    minutes = m0.copy()
    choices: list[tuple[str, str, str]] = []
    min_obs = int(min_joint_obs)

    for pair in pairs:
        i, j = int(pair.i), int(pair.j)
        if not (0 <= i < n and 0 <= j < n):
            continue
        joints = tuple(pair.joints or ())
        if len(joints) >= min_obs:
            ma, mb = joints[_rng_choice_index(rng, len(joints))]
            minutes[i] = float(ma)
            minutes[j] = float(mb)
            mode = f"joint:{pair.name_i}/{pair.name_j}"
        else:
            # Thin as-of history — leave mean minutes for both.
            mode = f"mean_fallback:{pair.name_i}/{pair.name_j}"
        choices.append((pair.name_i, pair.name_j, mode))

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
        if el.any():
            w = el.astype(float)
            w /= float(w.sum())
        else:
            w[:] = 1.0 / max(w.size, 1)
        return w
    return w / s
