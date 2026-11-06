"""Historical player-rate baselines for attack shares (cascade / cold-start).

Same-season-only ``rG``/``rA`` collapses early when one club-mate scores first
(Matty Cash / Villa GW1–5). Attack rates instead:

1. **Season start** — every player begins at a prior rate:
   - **Player prior:** appearance counts from all closed seasons before the
     scored season (name-matched; element ids are not stable across seasons).
   - **Else positional prior:** appearance per-90 G/A by position from those
     same prior seasons.
2. **Within-season update** — as GWs complete, same-season appearance
   goals/assists/minutes (GW ``< t``) are added to the prior counts so rates
   move with the season.

Does **not** reopen share-seed MC jitter (``keep_frozen_shares``). Soft terms
(bonus / DefCon / cards) stay same-season. Protocol:
``score_xp_v2_rate_baseline_prior``.
"""

from __future__ import annotations

import math
import unicodedata
from functools import lru_cache
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import load_merged
from src.rules.fpl_2026 import normalize_position

# Closed seasons available under data/cache/merged_gw_*.csv
CLOSED_SEASONS: tuple[str, ...] = ("2022-23", "2023-24", "2024-25", "2025-26")
_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss"})


def norm_player_name(name: str) -> str:
    """Lowercase, strip accents/apostrophes — cross-season name key."""
    text = str(name).translate(_TRANSLIT).lower().replace(".", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("'", "")
    return " ".join(text.split())


def prior_seasons_for(season: str) -> tuple[str, ...]:
    """Closed seasons strictly before ``season`` (never includes holdout logic)."""
    want = str(season).replace("_", "-")
    return tuple(s for s in CLOSED_SEASONS if s < want)


def _rate_row(
    *,
    goals: float,
    assists: float,
    minutes: float,
    m0: float,
    c_floor: float,
) -> dict[str, float]:
    m = float(max(minutes, 0.0))
    g = float(max(goals, 0.0))
    a = float(max(assists, 0.0))
    denom = m + float(m0)
    if denom <= 0:
        denom = float(m0) if m0 > 0 else 1.0
    return {
        "rG": g / denom,
        "drG": max(math.sqrt(g) if g > 0 else 0.0, c_floor) / denom,
        "rA": a / denom,
        "drA": max(math.sqrt(a) if a > 0 else 0.0, c_floor) / denom,
        "rate_minutes": m,
        "rate_goals": g,
        "rate_assists": a,
    }


@lru_cache(maxsize=8)
def _prior_appearance_frame(before_season: str) -> pd.DataFrame:
    """Stack appearance rows (minutes>0) from all seasons before ``before_season``."""
    seasons = prior_seasons_for(before_season)
    parts: list[pd.DataFrame] = []
    for season in seasons:
        m = load_merged(season)
        if m is None or m.empty:
            continue
        h = m.copy()
        h["minutes"] = pd.to_numeric(h["minutes"], errors="coerce").fillna(0.0)
        h = h.loc[h["minutes"] > 0].copy()
        if h.empty:
            continue
        h["goals_scored"] = pd.to_numeric(
            h.get("goals_scored", h.get("goals", 0)), errors="coerce"
        ).fillna(0.0)
        h["assists"] = pd.to_numeric(h["assists"], errors="coerce").fillna(0.0)
        h["name_key"] = h["name"].map(norm_player_name)
        h["position"] = h["position"].map(
            lambda p: normalize_position(str(p)) if pd.notna(p) else "MID"
        )
        h["season"] = season
        parts.append(h[["name_key", "name", "position", "minutes", "goals_scored", "assists", "season"]])
    if not parts:
        return pd.DataFrame(
            columns=[
                "name_key",
                "name",
                "position",
                "minutes",
                "goals_scored",
                "assists",
                "season",
            ]
        )
    return pd.concat(parts, ignore_index=True)


@lru_cache(maxsize=8)
def player_prior_rates(before_season: str) -> pd.DataFrame:
    """One row per ``name_key``: rG/rA from all prior-season appearances."""
    h = _prior_appearance_frame(before_season)
    if h.empty:
        return pd.DataFrame()
    m0 = float(numagent_xp.RATE_M0)
    c_floor = float(numagent_xp.RATE_SE_FLOOR)
    rows: list[dict[str, Any]] = []
    for key, g in h.groupby("name_key", sort=False):
        # Modal position for disambiguation / reporting
        pos = (
            g["position"].mode().iloc[0]
            if not g["position"].mode().empty
            else str(g["position"].iloc[0])
        )
        rr = _rate_row(
            goals=float(g["goals_scored"].sum()),
            assists=float(g["assists"].sum()),
            minutes=float(g["minutes"].sum()),
            m0=m0,
            c_floor=c_floor,
        )
        rows.append(
            {
                "name_key": key,
                "name": str(g["name"].iloc[0]),
                "position": pos,
                "rate_source": "player_prior",
                "rate_n_seasons": int(g["season"].nunique()),
                "rate_n": int(len(g)),
                "rate_m0": m0,
                "rate_se_floor": c_floor,
                **rr,
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def positional_prior_rates(before_season: str) -> dict[str, dict[str, float]]:
    """Position → rG/rA from pooled prior-season appearances (per-90 via M0)."""
    h = _prior_appearance_frame(before_season)
    m0 = float(numagent_xp.RATE_M0)
    c_floor = float(numagent_xp.RATE_SE_FLOOR)
    out: dict[str, dict[str, float]] = {}
    if h.empty:
        # Hard fallback if no prior data on disk
        defaults = {
            "GKP": (0.0, 0.0002),
            "DEF": (0.0008, 0.0010),
            "MID": (0.0025, 0.0020),
            "FWD": (0.0045, 0.0015),
        }
        for pos, (rg, ra) in defaults.items():
            out[pos] = {
                "rG": rg,
                "drG": c_floor / m0,
                "rA": ra,
                "drA": c_floor / m0,
                "rate_minutes": 0.0,
                "rate_goals": 0.0,
                "rate_assists": 0.0,
                "rate_source": "positional_hardcoded",
            }
        return out
    for pos, g in h.groupby("position", sort=False):
        key = normalize_position(str(pos))
        rr = _rate_row(
            goals=float(g["goals_scored"].sum()),
            assists=float(g["assists"].sum()),
            minutes=float(g["minutes"].sum()),
            m0=m0,
            c_floor=c_floor,
        )
        out[key] = {**rr, "rate_source": "positional_prior"}
    return out


def _season_appearance_counts(
    hist: pd.DataFrame | None,
    *,
    name_col: str = "name",
    element_col: str = "element",
    goals_col: str = "goals_scored",
    assists_col: str = "assists",
    minutes_col: str = "minutes",
) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, float]]]:
    """Same-season appearance totals keyed by name_key and element."""
    by_name: dict[str, dict[str, float]] = {}
    by_el: dict[int, dict[str, float]] = {}
    if hist is None or len(hist) == 0:
        return by_name, by_el
    h = hist.copy()
    h[minutes_col] = pd.to_numeric(h[minutes_col], errors="coerce").fillna(0.0)
    h = h.loc[h[minutes_col] > 0].copy()
    if h.empty:
        return by_name, by_el
    if goals_col not in h.columns and "goals" in h.columns:
        goals_col = "goals"
    h[goals_col] = pd.to_numeric(h[goals_col], errors="coerce").fillna(0.0)
    h[assists_col] = pd.to_numeric(h[assists_col], errors="coerce").fillna(0.0)
    if name_col in h.columns:
        h["_nk"] = h[name_col].map(norm_player_name)
        for key, g in h.groupby("_nk", sort=False):
            by_name[str(key)] = {
                "goals": float(g[goals_col].sum()),
                "assists": float(g[assists_col].sum()),
                "minutes": float(g[minutes_col].sum()),
            }
    if element_col in h.columns:
        h[element_col] = pd.to_numeric(h[element_col], errors="coerce")
        for eid, g in h.dropna(subset=[element_col]).groupby(element_col, sort=False):
            by_el[int(eid)] = {
                "goals": float(g[goals_col].sum()),
                "assists": float(g[assists_col].sum()),
                "minutes": float(g[minutes_col].sum()),
            }
    return by_name, by_el


def _combine_rate(
    *,
    g_prior: float,
    a_prior: float,
    m_prior: float,
    g_season: float,
    a_season: float,
    m_season: float,
    m0: float,
    c_floor: float,
    prior_kind: str,
) -> dict[str, Any]:
    """Player prior: (G0+Gs)/(M0+Ms+M0_pseudo). Positional: (r_pos·κ+Gs)/(κ+Ms)."""
    gs, as_, ms = float(g_season), float(a_season), float(m_season)
    if prior_kind == "player_prior":
        g = float(g_prior) + gs
        a = float(a_prior) + as_
        m = float(m_prior) + ms
        denom = m + float(m0)
        src = "player_prior+season" if ms > 0 else "player_prior"
    else:
        # positional: κ minutes of weight at r_pos (= g_prior/m_prior style via r)
        kappa = float(m0)
        # g_prior/a_prior here are κ·r_pos (pseudo counts); m_prior unused
        g = float(g_prior) + gs
        a = float(a_prior) + as_
        denom = kappa + ms
        m = ms
        src = "positional_prior+season" if ms > 0 else "positional_prior"
    if denom <= 0:
        denom = float(m0) if m0 > 0 else 1.0
    return {
        "rG": g / denom,
        "drG": max(math.sqrt(g) if g > 0 else 0.0, c_floor) / denom,
        "rA": a / denom,
        "drA": max(math.sqrt(a) if a > 0 else 0.0, c_floor) / denom,
        "rate_minutes": float(m_prior) + ms if prior_kind == "player_prior" else ms,
        "rate_goals": g,
        "rate_assists": a,
        "rate_source": src,
        "rate_season_minutes": ms,
    }


def rates_for_players(
    players: pd.DataFrame,
    *,
    season: str,
    name_col: str = "name",
    position_col: str = "position",
    element_col: str = "element",
    hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prior rate at season start, updated with same-season appearances in ``hist``.

    Prior lookup: player prior by normalised name → positional prior.
    ``hist`` should be same-season GW ``< t`` (``asof_share_history``).
    """
    if players is None or players.empty:
        return players.copy() if isinstance(players, pd.DataFrame) else pd.DataFrame()
    season_key = str(season).replace("_", "-")
    priors = player_prior_rates(season_key)
    pos_map = positional_prior_rates(season_key)
    by_key = (
        priors.set_index("name_key")
        if priors is not None and not priors.empty
        else None
    )
    season_by_name, season_by_el = _season_appearance_counts(
        hist, name_col=name_col, element_col=element_col
    )
    m0 = float(numagent_xp.RATE_M0)
    c_floor = float(numagent_xp.RATE_SE_FLOOR)

    out = players.copy()
    n = len(out)
    rG = np.zeros(n, dtype=float)
    drG = np.zeros(n, dtype=float)
    rA = np.zeros(n, dtype=float)
    drA = np.zeros(n, dtype=float)
    sources: list[str] = []
    minutes = np.zeros(n, dtype=float)
    season_minutes = np.zeros(n, dtype=float)

    for i in range(n):
        row = out.iloc[i]
        key = norm_player_name(str(row.get(name_col, "") or ""))
        pos_raw = row.get(position_col, "MID")
        try:
            pos = normalize_position(str(pos_raw)) if pd.notna(pos_raw) else "MID"
        except ValueError:
            pos = "MID"

        # Same-season counts (element preferred when present).
        sc = {"goals": 0.0, "assists": 0.0, "minutes": 0.0}
        if element_col in out.columns and pd.notna(row.get(element_col)):
            sc = season_by_el.get(int(row[element_col]), sc)
        if sc["minutes"] <= 0 and key:
            sc = season_by_name.get(key, sc)

        if by_key is not None and key and key in by_key.index:
            pr = by_key.loc[key]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            combined = _combine_rate(
                g_prior=float(pr["rate_goals"]),
                a_prior=float(pr["rate_assists"]),
                m_prior=float(pr["rate_minutes"]),
                g_season=sc["goals"],
                a_season=sc["assists"],
                m_season=sc["minutes"],
                m0=m0,
                c_floor=c_floor,
                prior_kind="player_prior",
            )
        else:
            pos_r = pos_map.get(pos) or pos_map.get("MID") or {
                "rG": 0.0,
                "rA": 0.0,
            }
            # κ·r_pos pseudo-counts (κ = RATE_M0)
            combined = _combine_rate(
                g_prior=float(pos_r["rG"]) * m0,
                a_prior=float(pos_r["rA"]) * m0,
                m_prior=m0,
                g_season=sc["goals"],
                a_season=sc["assists"],
                m_season=sc["minutes"],
                m0=m0,
                c_floor=c_floor,
                prior_kind="positional_prior",
            )
        rG[i] = float(combined["rG"])
        drG[i] = float(combined["drG"])
        rA[i] = float(combined["rA"])
        drA[i] = float(combined["drA"])
        minutes[i] = float(combined["rate_minutes"])
        season_minutes[i] = float(combined["rate_season_minutes"])
        sources.append(str(combined["rate_source"]))

    out["rG"] = rG
    out["drG"] = drG
    out["rA"] = rA
    out["drA"] = drA
    out["rate_minutes"] = minutes
    out["rate_season_minutes"] = season_minutes
    out["rate_source"] = sources
    out["rate_m0"] = m0
    out["rate_se_floor"] = c_floor
    return out


def apply_historical_rate_baseline(
    frame: pd.DataFrame,
    *,
    season: str,
    name_col: str = "name",
    position_col: str = "position",
    element_col: str = "element",
    hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Overwrite ``rG``/``rA``/``dr*``: prior at start, updated with ``hist``."""
    if frame is None or frame.empty:
        return frame
    if name_col not in frame.columns:
        raise ValueError(f"apply_historical_rate_baseline needs {name_col!r}")
    cols = [name_col]
    if position_col in frame.columns:
        cols.append(position_col)
    if element_col in frame.columns:
        cols.append(element_col)
    slim = frame[cols].copy()
    if position_col not in slim.columns:
        slim[position_col] = "MID"
    rates = rates_for_players(
        slim,
        season=season,
        name_col=name_col,
        position_col=position_col,
        element_col=element_col,
        hist=hist,
    )
    out = frame.copy()
    for c in (
        "rG",
        "drG",
        "rA",
        "drA",
        "rate_minutes",
        "rate_season_minutes",
        "rate_source",
        "rate_m0",
        "rate_se_floor",
    ):
        if c in rates.columns:
            out[c] = rates[c].to_numpy()
    return out


def baseline_summary(season: str) -> dict[str, Any]:
    """Small diagnostic dict for reports."""
    season_key = str(season).replace("_", "-")
    priors = player_prior_rates(season_key)
    pos = positional_prior_rates(season_key)
    return {
        "season": season_key,
        "prior_seasons": list(prior_seasons_for(season_key)),
        "n_player_priors": int(len(priors)) if priors is not None else 0,
        "positional": {
            k: {"rG": float(v["rG"]), "rA": float(v["rA"])} for k, v in pos.items()
        },
    }
