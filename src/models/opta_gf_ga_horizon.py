"""Long-horizon match pots from Opta predicted ranks → season GF/GA.

Opta publishes finishing place / points only. We map place → season GF/GA
(with uncertainty bands) on closed seasons, then build match λ pots in the
same shape as ``forecast_xp.side_pot`` / ``strength_match_pots`` so the
existing player-share path (``xp_on_pot``) can compute ``forecast_xp``.

Does not fit on the live 2026/27 holdout.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.models.forecast_xp import NEUTRAL_TOTAL, _clip, blend_pots, neutral_pot
from src.teams import norm_team


class _Rng(Protocol):
    def random(self) -> float: ...

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
OPTA_RANKS_PATH = ROOT / "data/predictions/2026-27/opta_ranks_20261012.json"

CLOSED = [
    ("2022/23", "E0_2223.csv"),
    ("2023/24", "E0_2324.csv"),
    ("2024/25", "E0_2425.csv"),
    ("2025/26", "E0_2526.csv"),
]

N_MATCHES = 38.0
# Neutral venue factor. The previous 1.08 home boost was never PI-approved
# (agent-era spice, same class as λ ±15% jitter) — retired 2026-10-24.
HOME_ADV = 1.0
# OLS on closed seasons (rank → season totals); also used as default se.
OLS_GF = (79.896, -2.264)  # intercept, slope
OLS_GA = (36.297, 1.888)
OLS_MAE_GF = 7.29
OLS_MAE_GA = 6.77
Z_BAND = 1.0  # ±1 MAE ≈ ~68% band under Laplace / rough 1σ proxy


def load_opta_ranks(path: Path | None = None) -> list[dict[str, Any]]:
    raw = json.loads((path or OPTA_RANKS_PATH).read_text())
    rows = []
    for item in raw["clubs"]:
        rows.append(
            {
                "rank": int(item["rank"]),
                "club": str(item["club"]),
                "club_norm": norm_team(item["club"]),
                "pred_pts": float(item["pred_pts"]),
                "as_of": raw.get("as_of"),
                "source": raw.get("source"),
            }
        )
    return sorted(rows, key=lambda r: r["rank"])


def _season_table(path: Path) -> list[dict[str, Any]]:
    import csv

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"pts": 0, "gf": 0, "ga": 0})
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row.get("FTHG"):
                continue
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            for team, gf, ga in (
                (row["HomeTeam"], hg, ag),
                (row["AwayTeam"], ag, hg),
            ):
                s = stats[team]
                s["gf"] += gf
                s["ga"] += ga
                s["pts"] += 3 if gf > ga else (1 if gf == ga else 0)
    rows = [
        {
            "team": t,
            "pts": s["pts"],
            "gf": s["gf"],
            "ga": s["ga"],
            "gd": s["gf"] - s["ga"],
        }
        for t, s in stats.items()
    ]
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows


def _profile_from_files(fnames: list[str]) -> dict[int, dict[str, float]]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"gf": [], "ga": []}
    )
    for fname in fnames:
        for r in _season_table(CACHE / fname):
            buckets[int(r["rank"])]["gf"].append(float(r["gf"]))
            buckets[int(r["rank"])]["ga"].append(float(r["ga"]))
    out: dict[int, dict[str, float]] = {}
    for rank, d in buckets.items():
        out[rank] = {
            "mean_gf": statistics.fmean(d["gf"]),
            "mean_ga": statistics.fmean(d["ga"]),
            "sd_gf": statistics.stdev(d["gf"]) if len(d["gf"]) > 1 else OLS_MAE_GF,
            "sd_ga": statistics.stdev(d["ga"]) if len(d["ga"]) > 1 else OLS_MAE_GA,
            "n": float(len(d["gf"])),
        }
    return out


def rank_gf_ga_profile() -> dict[int, dict[str, float]]:
    """Mean/sd of season GF/GA by finishing place (closed seasons)."""
    return _profile_from_files([fname for _, fname in CLOSED])


def rank_gf_ga_profile_excluding(exclude_fname: str) -> dict[int, dict[str, float]]:
    """Place profile fit on closed seasons other than ``exclude_fname``."""
    fnames = [fname for _, fname in CLOSED if fname != exclude_fname]
    if not fnames:
        raise ValueError("no seasons left after exclude")
    return _profile_from_files(fnames)


def _ols_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (intercept, slope, MAE)."""
    n = len(xs)
    if n < 2:
        raise ValueError("need at least two points for OLS")
    xm = statistics.fmean(xs)
    ym = statistics.fmean(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    if denom <= 0:
        raise ValueError("degenerate OLS design")
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys, strict=True)) / denom
    intercept = ym - slope * xm
    mae = statistics.fmean(abs(y - (intercept + slope * x)) for x, y in zip(xs, ys, strict=True))
    return float(intercept), float(slope), float(mae)


def fit_rank_ols(fnames: list[str] | None = None) -> dict[str, Any]:
    """OLS rank → season GF/GA on the listed E0 files (default: all closed)."""
    files = fnames if fnames is not None else [fname for _, fname in CLOSED]
    ranks: list[float] = []
    gfs: list[float] = []
    gas: list[float] = []
    for fname in files:
        for row in _season_table(CACHE / fname):
            ranks.append(float(row["rank"]))
            gfs.append(float(row["gf"]))
            gas.append(float(row["ga"]))
    a_gf, b_gf, mae_gf = _ols_fit(ranks, gfs)
    a_ga, b_ga, mae_ga = _ols_fit(ranks, gas)
    return {
        "files": list(files),
        "n_rows": len(ranks),
        "ols_gf": (a_gf, b_gf),
        "ols_ga": (a_ga, b_ga),
        "mae_gf": mae_gf,
        "mae_ga": mae_ga,
    }


def fit_rank_ols_excluding(exclude_fname: str) -> dict[str, Any]:
    """OLS on closed seasons other than ``exclude_fname`` (oracle-map fit)."""
    files = [fname for _, fname in CLOSED if fname != exclude_fname]
    if len(files) < 2:
        raise ValueError("need at least two prior seasons for the oracle map")
    out = fit_rank_ols(files)
    out["exclude"] = exclude_fname
    return out


def season_final_ranks(fname: str) -> dict[str, float]:
    """``club_norm → finishing rank`` from one E0 season file."""
    return {norm_team(r["team"]): float(r["rank"]) for r in _season_table(CACHE / fname)}


def gf_ga_from_rank(
    rank: float,
    *,
    profile: dict[int, dict[str, float]] | None = None,
    method: str = "ols",
    ols_gf: tuple[float, float] | None = None,
    ols_ga: tuple[float, float] | None = None,
    mae_gf: float | None = None,
    mae_ga: float | None = None,
    z_band: float | None = None,
) -> dict[str, float]:
    """Season GF/GA + uncertainty bands for a predicted finishing place."""
    r = max(1.0, min(20.0, float(rank)))
    z = float(Z_BAND if z_band is None else z_band)
    prof = profile if profile is not None else rank_gf_ga_profile()
    if method == "rankmean":
        lo = int(math.floor(r))
        hi = int(math.ceil(r))
        if lo == hi:
            gf, ga = prof[lo]["mean_gf"], prof[lo]["mean_ga"]
            sd_gf, sd_ga = prof[lo]["sd_gf"], prof[lo]["sd_ga"]
        else:
            w = r - lo
            gf = (1 - w) * prof[lo]["mean_gf"] + w * prof[hi]["mean_gf"]
            ga = (1 - w) * prof[lo]["mean_ga"] + w * prof[hi]["mean_ga"]
            sd_gf = (1 - w) * prof[lo]["sd_gf"] + w * prof[hi]["sd_gf"]
            sd_ga = (1 - w) * prof[lo]["sd_ga"] + w * prof[hi]["sd_ga"]
    else:
        a_gf, b_gf = ols_gf if ols_gf is not None else OLS_GF
        a_ga, b_ga = ols_ga if ols_ga is not None else OLS_GA
        band_gf = float(mae_gf) if mae_gf is not None else OLS_MAE_GF
        band_ga = float(mae_ga) if mae_ga is not None else OLS_MAE_GA
        gf = a_gf + b_gf * r
        ga = a_ga + b_ga * r
        # Blend OLS MAE with place-specific sd when available.
        lo = int(round(r))
        lo = min(20, max(1, lo))
        sd_gf = max(band_gf, prof[lo]["sd_gf"] * 0.5)
        sd_ga = max(band_ga, prof[lo]["sd_ga"] * 0.5)

    return {
        "rank": r,
        "gf": float(gf),
        "ga": float(ga),
        "gf_lo": float(gf - z * sd_gf),
        "gf_hi": float(gf + z * sd_gf),
        "ga_lo": float(max(0.0, ga - z * sd_ga)),
        "ga_hi": float(ga + z * sd_ga),
        "sd_gf": float(sd_gf),
        "sd_ga": float(sd_ga),
        "method": method,
        "z_band": z,
    }


def league_avg_gf(profile: dict[int, dict[str, float]] | None = None) -> float:
    prof = profile if profile is not None else rank_gf_ga_profile()
    return statistics.fmean(v["mean_gf"] for v in prof.values())


def pots_from_season_gf_ga(
    home_gf: float,
    home_ga: float,
    away_gf: float,
    away_ga: float,
    *,
    league_gf: float | None = None,
    home_adv: float = HOME_ADV,
    shrink: float = 1.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Match pots from season GF/GA totals (attack × opp defence / league).

    ``λ_H = (GF_H/38)·((GA_A/38) / ḡ)·home_adv`` (clipped). Default
    ``home_adv`` is 1.0 (no venue tilt). Returns pots compatible with
    ``xp_on_pot``.
    """
    lavg = float(league_gf if league_gf is not None else 52.0) / N_MATCHES
    lavg = lavg if lavg > 0 else NEUTRAL_TOTAL / 2.0
    rh_att = float(home_gf) / N_MATCHES
    ra_att = float(away_gf) / N_MATCHES
    rh_def = float(home_ga) / N_MATCHES
    ra_def = float(away_ga) / N_MATCHES
    lam_h = _clip(rh_att * (ra_def / lavg) * home_adv, 0.20, 3.50)
    lam_a = _clip(ra_att * (rh_def / lavg) / home_adv, 0.20, 3.50)
    e_total = lam_h + lam_a
    home = {
        "lam_scored": lam_h,
        "lam_assist": _clip(lam_h * 0.75, 0.1, 3.0),
        "e_total": e_total,
        "p_cs_mkt": _clip(0.90 * math.exp(-lam_a), 0.05, 0.55),
        "attack": lam_h,
        "threat": lam_a,
        "source": "opta_gf_ga",
    }
    away = {
        "lam_scored": lam_a,
        "lam_assist": _clip(lam_a * 0.75, 0.1, 3.0),
        "e_total": e_total,
        "p_cs_mkt": _clip(0.90 * math.exp(-lam_h), 0.04, 0.50),
        "attack": lam_a,
        "threat": lam_h,
        "source": "opta_gf_ga",
    }
    if shrink >= 1.0 - 1e-12:
        return home, away
    neutral = neutral_pot()
    return blend_pots(home, neutral, shrink), blend_pots(away, neutral, shrink)


def sample_within_band(
    lo: float,
    mid: float,
    hi: float,
    rng: _Rng,
    *,
    centre_blend: float = 0.5,
) -> float:
    """Centreline mid mixed with a uniform draw on ``[lo, hi]``."""
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    u = a + (b - a) * float(rng.random())
    w = float(min(1.0, max(0.0, centre_blend)))
    return w * float(mid) + (1.0 - w) * u


def sample_season_gf_ga_for_rank(
    rank: float,
    rng: _Rng,
    *,
    centre_blend: float = 0.5,
    profile: dict[int, dict[str, float]] | None = None,
    z_band: float | None = None,
    method: str = "ols",
) -> dict[str, float]:
    """Sample season GF/GA inside the Opta-rank band (baseline for one club).

    1st place uses the 1st-place GF/GA centre ± MAE; 20th uses 20th. No path.
    """
    rates = gf_ga_from_rank(
        rank, profile=profile, method=method, z_band=z_band
    )
    gf = sample_within_band(
        rates["gf_lo"], rates["gf"], rates["gf_hi"], rng, centre_blend=centre_blend
    )
    ga = sample_within_band(
        rates["ga_lo"], rates["ga"], rates["ga_hi"], rng, centre_blend=centre_blend
    )
    return {
        "rank": float(rates["rank"]),
        "gf": float(max(0.0, gf)),
        "ga": float(max(0.0, ga)),
        "gf_centre": float(rates["gf"]),
        "ga_centre": float(rates["ga"]),
        "gf_lo": float(rates["gf_lo"]),
        "gf_hi": float(rates["gf_hi"]),
        "ga_lo": float(rates["ga_lo"]),
        "ga_hi": float(rates["ga_hi"]),
    }


def sample_match_from_opta_ranks(
    home_club: str,
    away_club: str,
    rng: _Rng,
    *,
    ranks: Mapping[str, float] | None = None,
    centre_blend: float = 0.5,
    profile: dict[int, dict[str, float]] | None = None,
    league_gf: float | None = None,
    z_band: float | None = None,
    method: str = "ols",
) -> dict[str, Any]:
    """Baseline match λ from Opta place GF/GA ± bands (no ClubPath blend).

    Each side rolls season GF/GA inside its rank band, then
    ``pots_from_season_gf_ga`` → match λ. 1st vs 20th keep distinct rate
    centres for every fixture on the horizon.
    """
    if ranks is None:
        ranks = {r["club_norm"]: float(r["rank"]) for r in load_opta_ranks()}
    h = norm_team(home_club)
    a = norm_team(away_club)
    if h not in ranks or a not in ranks:
        missing = [c for c in (h, a) if c not in ranks]
        raise KeyError(f"missing Opta rank for {missing}")
    prof = profile if profile is not None else rank_gf_ga_profile()
    lavg = float(league_gf) if league_gf is not None else league_avg_gf(prof)
    home_s = sample_season_gf_ga_for_rank(
        ranks[h],
        rng,
        centre_blend=centre_blend,
        profile=prof,
        z_band=z_band,
        method=method,
    )
    away_s = sample_season_gf_ga_for_rank(
        ranks[a],
        rng,
        centre_blend=centre_blend,
        profile=prof,
        z_band=z_band,
        method=method,
    )
    pot_h, pot_a = pots_from_season_gf_ga(
        home_s["gf"],
        home_s["ga"],
        away_s["gf"],
        away_s["ga"],
        league_gf=lavg,
    )
    # Band endpoints from rank centres (for diagnostics / sample_scoreline).
    centre = match_prediction(
        h,
        a,
        ranks=ranks,
        method=method,
        profile=prof,
        league_gf=lavg,
        z_band=z_band,
    )
    return {
        "home": h,
        "away": a,
        "home_rank": float(ranks[h]),
        "away_rank": float(ranks[a]),
        "home_season_draw": home_s,
        "away_season_draw": away_s,
        "lam_home": float(pot_h["lam_scored"]),
        "lam_away": float(pot_a["lam_scored"]),
        "lam_home_lo": float(centre["lam_home_lo"]),
        "lam_home_hi": float(centre["lam_home_hi"]),
        "lam_away_lo": float(centre["lam_away_lo"]),
        "lam_away_hi": float(centre["lam_away_hi"]),
        "lam_home_centre": float(centre["lam_home"]),
        "lam_away_centre": float(centre["lam_away"]),
        "league_avg_gf": lavg,
    }


def match_prediction(
    home_club: str,
    away_club: str,
    *,
    ranks: Mapping[str, float] | None = None,
    method: str = "ols",
    profile: dict[int, dict[str, float]] | None = None,
    ols_gf: tuple[float, float] | None = None,
    ols_ga: tuple[float, float] | None = None,
    mae_gf: float | None = None,
    mae_ga: float | None = None,
    league_gf: float | None = None,
    z_band: float | None = None,
) -> dict[str, Any]:
    """Full scoreline forecast for one fixture from Opta ranks."""
    if ranks is None:
        ranks = {r["club_norm"]: float(r["rank"]) for r in load_opta_ranks()}
    h = norm_team(home_club)
    a = norm_team(away_club)
    if h not in ranks or a not in ranks:
        missing = [c for c in (h, a) if c not in ranks]
        raise KeyError(f"missing Opta rank for {missing}")
    prof = profile if profile is not None else rank_gf_ga_profile()
    lavg = float(league_gf) if league_gf is not None else league_avg_gf(prof)
    rate_kw = {
        "profile": prof,
        "method": method,
        "ols_gf": ols_gf,
        "ols_ga": ols_ga,
        "mae_gf": mae_gf,
        "mae_ga": mae_ga,
        "z_band": z_band,
    }
    home_rates = gf_ga_from_rank(ranks[h], **rate_kw)
    away_rates = gf_ga_from_rank(ranks[a], **rate_kw)

    pot_h, pot_a = pots_from_season_gf_ga(
        home_rates["gf"],
        home_rates["ga"],
        away_rates["gf"],
        away_rates["ga"],
        league_gf=lavg,
    )
    # Uncertainty: evaluate λ at band corners (home/away GF/GA lo/hi).
    corners = []
    for hgf in (home_rates["gf_lo"], home_rates["gf"], home_rates["gf_hi"]):
        for hga in (home_rates["ga_lo"], home_rates["ga"], home_rates["ga_hi"]):
            for agf in (away_rates["gf_lo"], away_rates["gf"], away_rates["gf_hi"]):
                for aga in (away_rates["ga_lo"], away_rates["ga"], away_rates["ga_hi"]):
                    ph, pa = pots_from_season_gf_ga(
                        hgf, hga, agf, aga, league_gf=lavg
                    )
                    corners.append((ph["lam_scored"], pa["lam_scored"]))
    lam_h_vals = [c[0] for c in corners]
    lam_a_vals = [c[1] for c in corners]

    return {
        "home": h,
        "away": a,
        "home_rank": ranks[h],
        "away_rank": ranks[a],
        "home_season": home_rates,
        "away_season": away_rates,
        "pot_home": pot_h,
        "pot_away": pot_a,
        "lam_home": pot_h["lam_scored"],
        "lam_away": pot_a["lam_scored"],
        "lam_home_lo": min(lam_h_vals),
        "lam_home_hi": max(lam_h_vals),
        "lam_away_lo": min(lam_a_vals),
        "lam_away_hi": max(lam_a_vals),
        "e_total": pot_h["e_total"],
        "p_cs_home": pot_h["p_cs_mkt"],
        "p_cs_away": pot_a["p_cs_mkt"],
        "league_avg_gf": lavg,
    }


def forecast_pots_from_opta_gf_ga(
    fixtures: list[Mapping[str, Any]],
    team_names: Mapping[int, str],
    *,
    start: int,
    end: int,
    priced_weeks: set[int] | None = None,
    ranks_path: Path | None = None,
    method: str = "ols",
    shrink: float = 1.0,
) -> dict[tuple[int, str], list[dict[str, float]]]:
    """Unpriced-week pots from Opta GF/GA (drop-in beside outrights pots).

    Only scheduled fixtures emit pots. Weeks in ``priced_weeks`` are skipped
    so opening/Betfair odds keep the imminent slate.
    """
    priced = priced_weeks or set()
    opta = load_opta_ranks(ranks_path)
    ranks = {r["club_norm"]: float(r["rank"]) for r in opta}
    profile = rank_gf_ga_profile()
    lavg = league_avg_gf(profile)
    rates = {
        r["club_norm"]: gf_ga_from_rank(r["rank"], profile=profile, method=method)
        for r in opta
    }
    out: dict[tuple[int, str], list[dict[str, float]]] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None:
            continue
        gw = int(event)
        if gw < int(start) or gw > int(end) or gw in priced:
            continue
        home = team_names.get(int(fixture["team_h"]))
        away = team_names.get(int(fixture["team_a"]))
        if not home or not away:
            continue
        h = norm_team(home)
        a = norm_team(away)
        if h not in rates or a not in rates:
            continue
        pot_h, pot_a = pots_from_season_gf_ga(
            rates[h]["gf"],
            rates[h]["ga"],
            rates[a]["gf"],
            rates[a]["ga"],
            league_gf=lavg,
            shrink=shrink,
        )
        pot_h = {**pot_h, "gw": gw, "opponent": a}
        pot_a = {**pot_a, "gw": gw, "opponent": h}
        out.setdefault((gw, h), []).append(pot_h)
        out.setdefault((gw, a), []).append(pot_a)
    return out


def bootstrap_shares(
    bootstrap: Mapping[str, Any],
    *,
    team_id: int,
) -> list[dict[str, Any]]:
    """Deadline-style shares from bootstrap expected goals within a club."""
    pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    players = [
        e for e in bootstrap.get("elements") or [] if int(e.get("team") or 0) == team_id
    ]
    xg = [max(float(p.get("expected_goals") or 0.0), 0.0) for p in players]
    xa = [max(float(p.get("expected_assists") or 0.0), 0.0) for p in players]
    sum_xg = sum(xg) or 1.0
    sum_xa = sum(xa) or 1.0
    # Soft floors by position so unused players keep a tiny share.
    floors = {"GKP": 0.005, "DEF": 0.02, "MID": 0.05, "FWD": 0.08}
    rows = []
    for p, gx, ax in zip(players, xg, xa):
        pos = pos_map.get(int(p.get("element_type") or 3), "MID")
        minutes = float(p.get("minutes") or 0.0)
        starts = max(int(p.get("starts") or 0), 1)
        xmi = _clip(minutes / starts, 0.0, 90.0) if minutes > 0 else 0.0
        share_g = max(gx / sum_xg, floors[pos] * 0.05)
        share_a = max(ax / sum_xa, floors[pos] * 0.05)
        rows.append(
            {
                "player_id": str(p["id"]),
                "name": str(p.get("web_name") or p.get("second_name") or ""),
                "position": pos,
                "team_id": team_id,
                "share_xG": share_g,
                "share_xA": share_a,
                "xmi": xmi,
                "exp_defcon_hit": 0.15 if pos == "DEF" else (0.08 if pos == "MID" else 0.0),
                "expected_goals": gx,
                "minutes_season": minutes,
            }
        )
    # Renormalise shares after floors.
    sg = sum(r["share_xG"] for r in rows) or 1.0
    sa = sum(r["share_xA"] for r in rows) or 1.0
    for r in rows:
        r["share_xG"] /= sg
        r["share_xA"] /= sa
    return rows


def player_forecast_xp(
    players: list[Mapping[str, Any]],
    pot: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Apply ``xp_on_pot`` with deadline shares (standard procedure)."""
    from src.models.forecast_xp import xp_on_pot

    out = []
    for p in players:
        xp = xp_on_pot(
            position=str(p.get("position") or "MID"),
            xmi=float(p.get("xmi") or 0.0),
            share_xg=float(p.get("share_xG") or 0.0),
            share_xa=float(p.get("share_xA") or 0.0),
            exp_defcon_hit=float(p.get("exp_defcon_hit") or 0.0),
            fwd_goal_scale=1.0,
            pot=dict(pot),
        )
        out.append({**dict(p), "forecast_xp": round(float(xp), 3)})
    out.sort(key=lambda r: -r["forecast_xp"])
    return out
