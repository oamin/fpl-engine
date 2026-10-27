"""Long-horizon match pots from Opta predicted ranks → season GF/GA.

PI-approved cascade_long strat (2026-10-24,
``protocol.cascade_long_match_outcome``):

- Place map: empirical mean GF/GA by finishing place + sample SD
  (``method="rankmean"``).
- Latents: per strand, draw GF/GA/ḡ inside ±z·SD with
  ``centre_blend=0.5``.
- ``ḡ``: assessed expanding mean goals/team/match (live/assessed E0).
- λ: ``(GF/38)·((GA_opp/38)/ḡ)``; hierarchical ``λ → M|λ → k∼M``
  (no second λ jitter). δ-method ``sd_lam`` is diagnostic only.

OLS place→GF/GA is archaeology only (``method="ols"``). Does not fit the
rank map on the live 2026/27 holdout.
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
# Default place→GF/GA map (cascade_long): empirical place means + SD.
DEFAULT_RANK_METHOD = "rankmean"
# OLS place→GF/GA — retired from the live path; only for method="ols".
OLS_GF = (79.896, -2.264)  # intercept, slope
OLS_GA = (36.297, 1.888)
OLS_MAE_GF = 7.29
OLS_MAE_GA = 6.77
Z_BAND = 1.0  # ±z · place SD (rankmean) or ±z · MAE (ols archaeology)
# Live / assessed E0 for expanding league ḡ (goals per team per match).
LIVE_E0 = "E0_2627.csv"
# Pseudo-matches of closed prior mixed into early-season expanding ḡ.
PRIOR_PSEUDO_MATCHES = 38.0


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
    method: str = DEFAULT_RANK_METHOD,
    ols_gf: tuple[float, float] | None = None,
    ols_ga: tuple[float, float] | None = None,
    mae_gf: float | None = None,
    mae_ga: float | None = None,
    z_band: float | None = None,
) -> dict[str, float]:
    """Season GF/GA + uncertainty bands for a predicted finishing place.

    Default ``rankmean``: place-bucket mean GF/GA and sample SD from closed
    seasons (interpolated for non-integer ranks). ``ols`` keeps the retired
    linear map for explicit archaeology only.
    """
    r = max(1.0, min(20.0, float(rank)))
    z = float(Z_BAND if z_band is None else z_band)
    prof = profile if profile is not None else rank_gf_ga_profile()
    mode = str(method or DEFAULT_RANK_METHOD)
    if mode == "rankmean":
        lo = int(math.floor(r))
        hi = int(math.ceil(r))
        lo = min(20, max(1, lo))
        hi = min(20, max(1, hi))
        if lo == hi:
            gf, ga = prof[lo]["mean_gf"], prof[lo]["mean_ga"]
            sd_gf, sd_ga = prof[lo]["sd_gf"], prof[lo]["sd_ga"]
        else:
            w = r - lo
            gf = (1 - w) * prof[lo]["mean_gf"] + w * prof[hi]["mean_gf"]
            ga = (1 - w) * prof[lo]["mean_ga"] + w * prof[hi]["mean_ga"]
            sd_gf = (1 - w) * prof[lo]["sd_gf"] + w * prof[hi]["sd_gf"]
            sd_ga = (1 - w) * prof[lo]["sd_ga"] + w * prof[hi]["sd_ga"]
    elif mode == "ols":
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
    else:
        raise ValueError(f"unknown rank GF/GA method={method!r}")

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
        "method": mode,
        "z_band": z,
    }


def league_avg_gf(profile: dict[int, dict[str, float]] | None = None) -> float:
    """Closed-season prior: mean place ``mean_gf`` (season-total units)."""
    prof = profile if profile is not None else rank_gf_ga_profile()
    return statistics.fmean(v["mean_gf"] for v in prof.values())


def _e0_team_goals_per_match(
    path: Path, *, max_matches: int | None = None
) -> list[float]:
    """Per completed E0 fixture: mean goals scored per team ``(FTHG+FTAG)/2``."""
    import csv

    if not path.is_file():
        return []
    out: list[float] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw_h, raw_a = row.get("FTHG"), row.get("FTAG")
            if raw_h is None or raw_h == "" or raw_a is None or raw_a == "":
                continue
            try:
                hg, ag = int(raw_h), int(raw_a)
            except (TypeError, ValueError):
                continue
            out.append(0.5 * float(hg + ag))
            if max_matches is not None and len(out) >= int(max_matches):
                break
    return out


def closed_league_gf_prior(
    profile: dict[int, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Prior ḡ from closed E0 seasons (centre + cross-season SD)."""
    centre = float(league_avg_gf(profile))
    season_pm: list[float] = []
    for _, fname in CLOSED:
        xs = _e0_team_goals_per_match(CACHE / fname)
        if xs:
            season_pm.append(statistics.fmean(xs))
    sd_pm = (
        statistics.stdev(season_pm)
        if len(season_pm) > 1
        else max(0.05, centre / N_MATCHES * 0.05)
    )
    return {
        "league_gf": centre,
        "g_pm": centre / N_MATCHES,
        "sd_league_gf": float(sd_pm * N_MATCHES),
        "sd_g_pm": float(sd_pm),
        "se_league_gf": float(sd_pm * N_MATCHES),
        "se_g_pm": float(sd_pm),
        "n_matches": 0.0,
        "source": "closed_prior",
    }


def assessed_league_gf(
    e0_fname: str | Path | None = None,
    *,
    max_matches: int | None = None,
    profile: dict[int, dict[str, float]] | None = None,
    prior_matches: float = PRIOR_PSEUDO_MATCHES,
) -> dict[str, float]:
    """Live / assessed expanding mean goals per team per match → season GF units.

    ``ḡ`` is the as-of mean of ``(FTHG+FTAG)/2`` on completed E0 rows, blended
    with the closed-season prior (``prior_matches`` pseudo-games) so early
    weeks stay stable. ``sd_*`` is the match-level sample SD; ``se_*`` is the
    SE of the expanding mean (uncertainty on ḡ itself).
    """
    prior = closed_league_gf_prior(profile)
    if e0_fname is None:
        return prior
    path = Path(e0_fname) if isinstance(e0_fname, Path) else CACHE / str(e0_fname)
    xs = _e0_team_goals_per_match(path, max_matches=max_matches)
    n = float(len(xs))
    if n <= 0:
        return prior
    hat_pm = float(statistics.fmean(xs))
    sd_pm = float(statistics.stdev(xs)) if n > 1 else float(prior["sd_g_pm"])
    n0 = max(0.0, float(prior_matches))
    g_pm = (n0 * float(prior["g_pm"]) + n * hat_pm) / (n0 + n)
    # SE of blended mean: match SD / sqrt(effective n).
    se_pm = sd_pm / math.sqrt(n0 + n)
    return {
        "league_gf": float(g_pm * N_MATCHES),
        "g_pm": float(g_pm),
        "hat_g_pm": hat_pm,
        "sd_league_gf": float(sd_pm * N_MATCHES),
        "sd_g_pm": sd_pm,
        "se_league_gf": float(se_pm * N_MATCHES),
        "se_g_pm": float(se_pm),
        "n_matches": n,
        "prior_matches": n0,
        "e0": str(path.name),
        "source": "expanding_assessed",
    }


def lambdas_from_gf_ga(
    home_gf: float,
    home_ga: float,
    away_gf: float,
    away_ga: float,
    *,
    league_gf: float,
    sd_home_gf: float = 0.0,
    sd_home_ga: float = 0.0,
    sd_away_gf: float = 0.0,
    sd_away_ga: float = 0.0,
    sd_league_gf: float = 0.0,
    home_adv: float = HOME_ADV,
    z_band: float | None = None,
) -> dict[str, float]:
    """Dixon–Coles λ with delta-method SDs from place GF/GA and ḡ uncertainty.

    ``λ_H = (GF_H/38)·((GA_A/38)/ḡ)·h`` with ``ḡ = league_gf/38``. Relative
    variances add under independence: GF_H, GA_A, and league_gf for ``λ_H``
    (symmetrically GF_A, GA_H, league_gf for ``λ_A``).
    """
    z = float(Z_BAND if z_band is None else z_band)
    L = max(float(league_gf), 1e-9)
    h_adv = float(home_adv)
    gf_h, ga_h = max(float(home_gf), 0.0), max(float(home_ga), 0.0)
    gf_a, ga_a = max(float(away_gf), 0.0), max(float(away_ga), 0.0)
    lam_h_raw = (gf_h * ga_a * h_adv) / (N_MATCHES * L)
    lam_a_raw = (gf_a * ga_h / h_adv) / (N_MATCHES * L) if h_adv > 0 else 0.0

    def _rel_var(val: float, sd: float) -> float:
        if val <= 1e-12:
            return 0.0
        return (float(sd) / val) ** 2

    sd_h = abs(lam_h_raw) * math.sqrt(
        _rel_var(gf_h, sd_home_gf)
        + _rel_var(ga_a, sd_away_ga)
        + _rel_var(L, sd_league_gf)
    )
    sd_a = abs(lam_a_raw) * math.sqrt(
        _rel_var(gf_a, sd_away_gf)
        + _rel_var(ga_h, sd_home_ga)
        + _rel_var(L, sd_league_gf)
    )
    lam_h = _clip(lam_h_raw, 0.20, 3.50)
    lam_a = _clip(lam_a_raw, 0.20, 3.50)
    return {
        "lam_home": float(lam_h),
        "lam_away": float(lam_a),
        "lam_home_raw": float(lam_h_raw),
        "lam_away_raw": float(lam_a_raw),
        "sd_lam_home": float(sd_h),
        "sd_lam_away": float(sd_a),
        "lam_home_lo": float(_clip(lam_h_raw - z * sd_h, 0.20, 3.50)),
        "lam_home_hi": float(_clip(lam_h_raw + z * sd_h, 0.20, 3.50)),
        "lam_away_lo": float(_clip(lam_a_raw - z * sd_a, 0.20, 3.50)),
        "lam_away_hi": float(_clip(lam_a_raw + z * sd_a, 0.20, 3.50)),
        "league_gf": float(L),
        "g_pm": float(L / N_MATCHES),
        "z_band": z,
    }


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

    ``λ_H = (GF_H/38)·((GA_A/38) / ḡ)·home_adv`` (clipped). ``league_gf`` is
    season-total units (``ḡ_pm × 38``); prefer ``assessed_league_gf`` for the
    live / assessed expanding mean. Default ``home_adv`` is 1.0.
    """
    if league_gf is None:
        league_gf = float(closed_league_gf_prior()["league_gf"])
    lavg = float(league_gf) / N_MATCHES
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
    method: str = DEFAULT_RANK_METHOD,
) -> dict[str, float]:
    """Sample season GF/GA inside the Opta-rank band (baseline for one club).

    1st place uses the 1st-place empirical GF/GA mean ± z·SD; 20th uses 20th. No path.
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
    sd_league_gf: float | None = None,
    e0_fname: str | Path | None = None,
    z_band: float | None = None,
    method: str = DEFAULT_RANK_METHOD,
    home_adv: float = HOME_ADV,
) -> dict[str, Any]:
    """Baseline match λ from Opta place GF/GA ± bands (no ClubPath blend).

    Each side rolls season GF/GA inside its rank band; ḡ is the assessed
    expanding league mean (or an explicit ``league_gf``), also sampled inside
    its SE band; then ``pots_from_season_gf_ga`` → match λ.
    """
    if ranks is None:
        ranks = {r["club_norm"]: float(r["rank"]) for r in load_opta_ranks()}
    h = norm_team(home_club)
    a = norm_team(away_club)
    if h not in ranks or a not in ranks:
        missing = [c for c in (h, a) if c not in ranks]
        raise KeyError(f"missing Opta rank for {missing}")
    prof = profile if profile is not None else rank_gf_ga_profile()
    z = float(Z_BAND if z_band is None else z_band)
    if league_gf is None:
        assessed = assessed_league_gf(e0_fname, profile=prof)
        l_mid = float(assessed["league_gf"])
        l_sd = float(assessed["se_league_gf"])
    else:
        l_mid = float(league_gf)
        l_sd = float(sd_league_gf) if sd_league_gf is not None else 0.0
    lavg = sample_within_band(
        l_mid - z * l_sd,
        l_mid,
        l_mid + z * l_sd,
        rng,
        centre_blend=centre_blend,
    )
    lavg = max(lavg, 1e-6)
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
        home_adv=float(home_adv),
    )
    # Diagnostic bands: delta-method through place SD + ḡ SE at centres.
    centre = match_prediction(
        h,
        a,
        ranks=ranks,
        method=method,
        profile=prof,
        league_gf=l_mid,
        sd_league_gf=l_sd,
        e0_fname=e0_fname,
        z_band=z_band,
        home_adv=float(home_adv),
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
        "sd_lam_home": float(centre["sd_lam_home"]),
        "sd_lam_away": float(centre["sd_lam_away"]),
        "lam_home_centre": float(centre["lam_home"]),
        "lam_away_centre": float(centre["lam_away"]),
        "league_avg_gf": float(l_mid),
        "league_gf_draw": float(lavg),
        "sd_league_gf": float(l_sd),
    }


def match_prediction(
    home_club: str,
    away_club: str,
    *,
    ranks: Mapping[str, float] | None = None,
    method: str = DEFAULT_RANK_METHOD,
    profile: dict[int, dict[str, float]] | None = None,
    ols_gf: tuple[float, float] | None = None,
    ols_ga: tuple[float, float] | None = None,
    mae_gf: float | None = None,
    mae_ga: float | None = None,
    league_gf: float | None = None,
    sd_league_gf: float | None = None,
    e0_fname: str | Path | None = None,
    z_band: float | None = None,
    home_adv: float = HOME_ADV,
) -> dict[str, Any]:
    """Full scoreline forecast for one fixture from Opta ranks.

    λ bands use delta-method propagation of place GF/GA SDs and the SE of
    assessed expanding ḡ (not a corner envelope of independent lo/hi).
    """
    if ranks is None:
        ranks = {r["club_norm"]: float(r["rank"]) for r in load_opta_ranks()}
    h = norm_team(home_club)
    a = norm_team(away_club)
    if h not in ranks or a not in ranks:
        missing = [c for c in (h, a) if c not in ranks]
        raise KeyError(f"missing Opta rank for {missing}")
    prof = profile if profile is not None else rank_gf_ga_profile()
    if league_gf is None:
        assessed = assessed_league_gf(e0_fname, profile=prof)
        lavg = float(assessed["league_gf"])
        l_sd = float(assessed["se_league_gf"])
        league_meta = assessed
    else:
        lavg = float(league_gf)
        l_sd = float(sd_league_gf) if sd_league_gf is not None else 0.0
        league_meta = {
            "league_gf": lavg,
            "se_league_gf": l_sd,
            "source": "explicit",
        }
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
        home_adv=float(home_adv),
    )
    prop = lambdas_from_gf_ga(
        home_rates["gf"],
        home_rates["ga"],
        away_rates["gf"],
        away_rates["ga"],
        league_gf=lavg,
        sd_home_gf=home_rates["sd_gf"],
        sd_home_ga=home_rates["sd_ga"],
        sd_away_gf=away_rates["sd_gf"],
        sd_away_ga=away_rates["sd_ga"],
        sd_league_gf=l_sd,
        z_band=z_band,
        home_adv=float(home_adv),
    )

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
        "lam_home_lo": prop["lam_home_lo"],
        "lam_home_hi": prop["lam_home_hi"],
        "lam_away_lo": prop["lam_away_lo"],
        "lam_away_hi": prop["lam_away_hi"],
        "sd_lam_home": prop["sd_lam_home"],
        "sd_lam_away": prop["sd_lam_away"],
        "e_total": pot_h["e_total"],
        "p_cs_home": pot_h["p_cs_mkt"],
        "p_cs_away": pot_a["p_cs_mkt"],
        "league_avg_gf": lavg,
        "sd_league_gf": l_sd,
        "league": league_meta,
    }


def forecast_pots_from_opta_gf_ga(
    fixtures: list[Mapping[str, Any]],
    team_names: Mapping[int, str],
    *,
    start: int,
    end: int,
    priced_weeks: set[int] | None = None,
    ranks_path: Path | None = None,
    method: str = DEFAULT_RANK_METHOD,
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
    lavg = float(assessed_league_gf(LIVE_E0, profile=profile)["league_gf"])
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
