"""Lineup / minutes worlds before attack dice (short-horizon).

Default (``lineup_cov=True``): **club matchday resample** — draw one as-of
gameweek for the club and use every player's minutes from that GW. No
correlation threshold, no competitor-pair graph.

- CS eligibility: ``minutes ≥ 60``.
- Attack dice gate: ``minutes > 0``.
- Thin club history: fall back to as-of mean minutes.

Legacy pair utilities (corr threshold, joint pair resample) stay importable
for archaeology/tests but are **not** on the generative default path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.rules.fpl_2026 import normalize_position

START_GATE = 60.0  # CS / 60′ appearance bonus gate
MIN_CLUB_GWS = 3  # need this many as-of club GWs to resample a matchday
MATCH_MINUTES = 90.0

# --- Legacy pair constants (parked; not used by default lineup_cov path) ---
CORR_THRESH = -0.40  # retired from default path — do not re-enable without PI lock
PRIOR_MINUTES = 30.0
MIN_PAIR_MEAN = 5.0
MIN_JOINT_OBS = 3


@dataclass(frozen=True)
class CompetitorPair:
    """Legacy pair record (parked)."""

    i: int
    j: int
    corr: float
    name_i: str
    name_j: str
    id_i: int = 0
    id_j: int = 0
    joints: tuple[tuple[float, float], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LineupWorld:
    """Minutes for one club on one path."""

    minutes: np.ndarray
    eligible: np.ndarray  # bool, minutes >= START_GATE (CS gate)
    pair_choices: tuple[tuple[str, str, str], ...]  # diagnostic labels

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


def _rng_choice_index(rng: Any, n: int) -> int:
    if n <= 0:
        raise ValueError("empty choice")
    if hasattr(rng, "integers"):
        return int(rng.integers(0, n))
    if hasattr(rng, "randrange"):
        return int(rng.randrange(n))
    return int(np.asarray(rng.integers(0, n)))


def _hist_gw_col(hist: pd.DataFrame) -> str | None:
    if "gw" in hist.columns:
        return "gw"
    if "GW" in hist.columns:
        return "GW"
    return None


def _hist_id_col(hist: pd.DataFrame) -> str | None:
    for c in ("element", "player_id"):
        if c in hist.columns:
            return c
    return None


def club_matchday_gws(
    hist: pd.DataFrame,
    *,
    id_col: str | None = None,
    minutes_col: str = "minutes",
    gw_col: str | None = None,
) -> list[Any]:
    """As-of club GWs that have any positive minutes."""
    if hist is None or hist.empty:
        return []
    gw_c = gw_col or _hist_gw_col(hist)
    if gw_c is None:
        return []
    h = hist.copy()
    h[minutes_col] = pd.to_numeric(h[minutes_col], errors="coerce").fillna(0.0)
    gws = []
    for gw, block in h.groupby(gw_c, sort=True):
        if float(block[minutes_col].sum()) > 0:
            gws.append(gw)
    return gws


def roll_club_gw_lineup(
    club: pd.DataFrame,
    hist: pd.DataFrame | None,
    rng: Any,
    *,
    id_col_club: str = "element",
    start_gate: float = START_GATE,
    min_club_gws: int = MIN_CLUB_GWS,
) -> LineupWorld:
    """Draw one past club GW; assign each roster player's minutes from that GW.

    Players missing from the drawn GW get 0′. If history is too thin, keep
    as-of mean minutes on the club frame (``m`` / ``xmi``).
    """
    if "m" in club.columns:
        mean_m = pd.to_numeric(club["m"], errors="coerce").fillna(0.0).to_numpy(float)
    elif "xmi" in club.columns:
        mean_m = pd.to_numeric(club["xmi"], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        mean_m = np.zeros(len(club), dtype=float)

    names_col = "name" if "name" in club.columns else (
        "web_name" if "web_name" in club.columns else None
    )
    names = (
        club[names_col].astype(str).tolist()
        if names_col
        else [str(i) for i in range(len(club))]
    )

    if hist is None or hist.empty or club.empty:
        return LineupWorld(
            minutes=mean_m.copy(),
            eligible=mean_m >= float(start_gate),
            pair_choices=(("club", "mean_fallback", "no_hist"),),
        )

    gw_c = _hist_gw_col(hist)
    id_h = _hist_id_col(hist)
    id_c = id_col_club if id_col_club in club.columns else (
        "player_id" if "player_id" in club.columns else None
    )
    if gw_c is None or id_h is None or id_c is None:
        return LineupWorld(
            minutes=mean_m.copy(),
            eligible=mean_m >= float(start_gate),
            pair_choices=(("club", "mean_fallback", "bad_cols"),),
        )

    gws = club_matchday_gws(hist, minutes_col="minutes", gw_col=gw_c)
    if len(gws) < int(min_club_gws):
        return LineupWorld(
            minutes=mean_m.copy(),
            eligible=mean_m >= float(start_gate),
            pair_choices=(("club", "mean_fallback", f"thin_gws={len(gws)}"),),
        )

    gw = gws[_rng_choice_index(rng, len(gws))]
    block = hist.loc[hist[gw_c] == gw].copy()
    block["minutes"] = pd.to_numeric(block["minutes"], errors="coerce").fillna(0.0)
    mins_by_id = (
        block.groupby(id_h, sort=False)["minutes"].sum().to_dict()
    )
    # keys may be int-like
    mins_by_id = {int(k): float(v) for k, v in mins_by_id.items()}

    minutes = np.zeros(len(club), dtype=float)
    for i, raw in enumerate(club[id_c].tolist()):
        try:
            eid = int(str(raw).split(":")[-1])
        except (TypeError, ValueError):
            minutes[i] = 0.0
            continue
        minutes[i] = float(mins_by_id.get(eid, 0.0))

    return LineupWorld(
        minutes=minutes,
        eligible=minutes >= float(start_gate),
        pair_choices=(("club", f"gw={gw}", "matchday_resample"),),
    )


def trivial_lineup_from_means(
    mean_m: np.ndarray | Sequence[float],
    *,
    start_gate: float = START_GATE,
) -> LineupWorld:
    minutes = np.asarray(mean_m, dtype=float).copy()
    return LineupWorld(
        minutes=minutes,
        eligible=minutes >= float(start_gate),
        pair_choices=(("club", "trivial", "mean_minutes"),),
    )


# ---------------------------------------------------------------------------
# Legacy pair path (parked) — kept for unit tests / archaeology only
# ---------------------------------------------------------------------------


def minutes_panel(
    history: pd.DataFrame,
    *,
    id_col: str = "player_id",
    minutes_col: str = "minutes",
    gw_col: str = "gw",
) -> pd.DataFrame:
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
    """Legacy: off-diagonal minutes correlations (not on default path)."""
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
            if normalize_position(pa) != normalize_position(pb):
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


def select_disjoint_pairs(
    board_ids: Sequence[int],
    pair_table: pd.DataFrame,
    *,
    max_pairs: int = 8,
    panel: pd.DataFrame | None = None,
) -> list[CompetitorPair]:
    """Legacy greedy pair matching (not on default path)."""
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


def roll_lineup_world(
    *,
    mean_m: np.ndarray | Sequence[float],
    names: Sequence[str],
    pairs: Sequence[CompetitorPair],
    rng: Any,
    start_gate: float = START_GATE,
    min_joint_obs: int = MIN_JOINT_OBS,
) -> LineupWorld:
    """Legacy pair-joint resample (not on default path)."""
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
            mode = f"mean_fallback:{pair.name_i}/{pair.name_j}"
        choices.append((pair.name_i, pair.name_j, mode))

    return LineupWorld(
        minutes=minutes,
        eligible=minutes >= float(start_gate),
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
