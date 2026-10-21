"""Historical player-rate baselines for attack shares (cascade / cold-start).

Same-season-only ``rG``/``rA`` collapses early when one club-mate scores first
(Matty Cash / Villa GW1–5). This module supplies:

1. **Player prior** — appearance rates from **all closed seasons before** the
   scored season (name-matched; element ids are not stable across seasons).
2. **Positional prior** — appearance per-90 G/A by position from those same
   prior seasons, for players with no prior history.

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


def rates_for_players(
    players: pd.DataFrame,
    *,
    season: str,
    name_col: str = "name",
    position_col: str = "position",
) -> pd.DataFrame:
    """Attach historical baseline rates onto ``players`` (one row per player).

    Lookup order: player prior by normalised name → positional prior.
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

    out = players.copy()
    n = len(out)
    rG = np.zeros(n, dtype=float)
    drG = np.zeros(n, dtype=float)
    rA = np.zeros(n, dtype=float)
    drA = np.zeros(n, dtype=float)
    sources: list[str] = []
    minutes = np.zeros(n, dtype=float)

    for i in range(n):
        row = out.iloc[i]
        key = norm_player_name(str(row.get(name_col, "") or ""))
        pos_raw = row.get(position_col, "MID")
        try:
            pos = normalize_position(str(pos_raw)) if pd.notna(pos_raw) else "MID"
        except ValueError:
            pos = "MID"
        used = None
        if by_key is not None and key and key in by_key.index:
            pr = by_key.loc[key]
            if isinstance(pr, pd.DataFrame):
                pr = pr.iloc[0]
            used = pr
            src = "player_prior"
        else:
            used = pos_map.get(pos) or pos_map.get("MID") or {
                "rG": 0.0,
                "drG": 0.0,
                "rA": 0.0,
                "drA": 0.0,
                "rate_minutes": 0.0,
            }
            src = str(used.get("rate_source", "positional_prior"))
        rG[i] = float(used["rG"])
        drG[i] = float(used["drG"])
        rA[i] = float(used["rA"])
        drA[i] = float(used["drA"])
        minutes[i] = float(used.get("rate_minutes", 0.0) or 0.0)
        sources.append(src)

    out["rG"] = rG
    out["drG"] = drG
    out["rA"] = rA
    out["drA"] = drA
    out["rate_minutes"] = minutes
    out["rate_source"] = sources
    out["rate_m0"] = float(numagent_xp.RATE_M0)
    out["rate_se_floor"] = float(numagent_xp.RATE_SE_FLOOR)
    return out


def apply_historical_rate_baseline(
    frame: pd.DataFrame,
    *,
    season: str,
    name_col: str = "name",
    position_col: str = "position",
) -> pd.DataFrame:
    """Overwrite ``rG``/``rA``/``dr*`` on ``frame`` with historical baselines."""
    if frame is None or frame.empty:
        return frame
    if name_col not in frame.columns:
        raise ValueError(f"apply_historical_rate_baseline needs {name_col!r}")
    rates = rates_for_players(
        frame[[name_col, position_col]].copy()
        if position_col in frame.columns
        else frame[[name_col]].assign(**{position_col: "MID"}),
        season=season,
        name_col=name_col,
        position_col=position_col,
    )
    out = frame.copy()
    for c in ("rG", "drG", "rA", "drA", "rate_minutes", "rate_source", "rate_m0", "rate_se_floor"):
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
