"""Bookie-informed L4 horizon boards: GW(t:t+1) + beyond arms A/B/C.

At decision GW ``t``:
  - Freeze rates / minutes priors / α / competition pairs on merged GW < t.
  - For h ∈ {t, t+1}: project onto opening book moments (live Odds API /
    historical FD AH+OU); score fixtures; board μ as configured by caller.
  - PI 2026-10-16: ``BOOKIE_DEPTH=2`` — live Odds API lists ~9d of EPL only
    (this GW + next). No forecast week without bookie pricing.

Arms
  A_hard_stop — H = n_priced only
  B_crude_fdr — extend to H_extend; μ = mean(priced) × as-of FDR vs opponent
  C_decay     — extend to H_extend; decay last priced μ toward position mean

  python -m src.strategy.layer4_bookie_path --season 2025-26 --gw 10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.markets.fixture_covariance import build_club_side, draw_fixture_covariant
from src.markets.fixture_covariance_audit import asof_groups_for_team
from src.markets.minutes_covariance import load_merged, merged_path
from src.markets.numagent_xp_pilot import _asof_alphas, load_fd_moments
from src.rules.fpl_2026 import normalize_position
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

BOOKIE_DEPTH = 2  # t, t+1 only (live book horizon; no unpriced weeks)
H_EXTEND = 5
N_SCENARIOS = 48
GAMMA_FILL = 0.9
MIN_MEAN_M = 15.0

ArmName = Literal["A_hard_stop", "B_crude_fdr", "C_decay"]
ARMS: tuple[ArmName, ...] = ("A_hard_stop", "B_crude_fdr", "C_decay")


@dataclass(frozen=True)
class BookieHorizon:
    season: str
    decision_gw: int
    arm: ArmName
    pools: list[pd.DataFrame]  # length H; columns include player_id, mu_post, …
    priced_gws: tuple[int, ...]
    filler_gws: tuple[int, ...]
    n_fixtures_scored: int
    n_scenarios: int
    meta: dict[str, Any]


def _load_merged_enriched(season: str) -> pd.DataFrame:
    df = load_merged(season=season)
    # Need kickoff / was_home for FD join + schedule
    raw = pd.read_csv(merged_path(season))
    extra = ["kickoff_time", "was_home", "goals_scored", "assists", "bonus",
             "yellow_cards", "red_cards", "team_h_score", "team_a_score",
             "defensive_contribution", "saves", "opponent_team"]
    keep = ["element", "fixture"] + [c for c in extra if c in raw.columns]
    raw = raw[keep].drop_duplicates(["element", "fixture"], keep="first")
    out = df.merge(raw, on=["element", "fixture"], how="left", suffixes=("", "_raw"))
    if "was_home" not in out.columns and "was_home_raw" in out.columns:
        out["was_home"] = out["was_home_raw"]
    out["team_n"] = out["team"].map(norm_team)
    if "kickoff_time" in out.columns:
        out["date_iso"] = (
            pd.to_datetime(out["kickoff_time"], utc=True)
            .dt.tz_convert(None)
            .dt.date
        )
    else:
        out["date_iso"] = pd.NaT
    for col, default in (
        ("defensive_contribution", 0.0),
        ("saves", 0.0),
        ("bonus", 0.0),
        ("yellow_cards", 0.0),
        ("red_cards", 0.0),
        ("goals_scored", 0.0),
        ("assists", 0.0),
    ):
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)
    def _pos(p: object) -> str:
        key = str(p).strip().upper()
        if key in ("GK", "GKP"):
            return "GKP"
        if key in ("AM", "MID"):  # 2024/25 Vaastav AM → MID
            return "MID"
        return normalize_position(key)

    out["position"] = out["position"].map(_pos)
    return out


def _fd_index(season: str) -> pd.DataFrame:
    fd = load_fd_moments(season)
    # Also keep lam for cov draws
    raw = pd.read_csv(
        PROCESSED / f"scoreline_fd_moments_{season.replace('-', '_')}.csv"
    )
    raw["team_n"] = raw["team"].map(norm_team)
    raw["date_iso"] = pd.to_datetime(raw["date"], dayfirst=True).dt.date
    cols = ["team_n", "date_iso", "xg", "xcs", "xgc", "lam_h", "lam_a", "rho", "home", "away", "side"]
    have = [c for c in cols if c in raw.columns]
    return raw[have].drop_duplicates(["team_n", "date_iso"], keep="first")


def _schedule(merged: pd.DataFrame) -> pd.DataFrame:
    """Unique fixtures with home/away team names + GW + date."""
    cols = ["fixture", "GW", "team", "team_n", "date_iso"]
    if "was_home" in merged.columns:
        cols.append("was_home")
    sub = merged[cols].drop_duplicates()
    if "was_home" in sub.columns:
        home = sub.loc[sub["was_home"] == True][  # noqa: E712
            ["fixture", "GW", "team", "team_n", "date_iso"]
        ].rename(columns={"team": "home_team", "team_n": "home_n"})
        away = sub.loc[sub["was_home"] == False][  # noqa: E712
            ["fixture", "team", "team_n"]
        ].rename(columns={"team": "away_team", "team_n": "away_n"})
        return home.merge(away, on="fixture", how="inner")
    rows = []
    for (fx, gw), g in sub.groupby(["fixture", "GW"]):
        teams = sorted(g["team"].astype(str).unique())
        if len(teams) != 2:
            continue
        tn = {str(r.team): str(r.team_n) for r in g.itertuples(index=False)}
        d = g["date_iso"].iloc[0]
        rows.append(
            {
                "fixture": fx,
                "GW": gw,
                "home_team": teams[0],
                "away_team": teams[1],
                "home_n": tn[teams[0]],
                "away_n": tn[teams[1]],
                "date_iso": d,
            }
        )
    return pd.DataFrame(rows)


def _team_strength_asof(hist: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Crude as-of attack / concede rates per team (per fixture)."""
    if hist.empty:
        return {}
    rows: list[dict[str, float | str]] = []
    has_score = "team_h_score" in hist.columns and "team_a_score" in hist.columns
    for (team, fx), g in hist.groupby(["team", "fixture"], sort=False):
        gf = float(pd.to_numeric(g["goals_scored"], errors="coerce").fillna(0).sum())
        ga = float("nan")
        if has_score and "was_home" in g.columns:
            wh = bool(g["was_home"].iloc[0])
            hs, aus = g["team_h_score"].iloc[0], g["team_a_score"].iloc[0]
            if pd.notna(hs) and pd.notna(aus):
                ga = float(aus if wh else hs)
        rows.append({"team": str(team), "gf": gf, "ga": ga})
    panel = pd.DataFrame(rows)
    out: dict[str, dict[str, float]] = {}
    for team, tg in panel.groupby("team"):
        gf = tg["gf"].to_numpy(float)
        ga = pd.to_numeric(tg["ga"], errors="coerce").dropna().to_numpy(float)
        out[str(team)] = {
            "att": float(gf.mean()) if len(gf) else 1.2,
            "defn": float(ga.mean()) if len(ga) else 1.2,
            "n": int(len(tg)),
        }
    return out


def crude_fdr_scale(
    team: str,
    opp: str,
    position: str,
    strengths: dict[str, dict[str, float]],
) -> float:
    """Position-aware scale in ~[0.55, 1.35] from as-of gf/ga (no opening line)."""
    league_att = float(np.mean([v["att"] for v in strengths.values()])) if strengths else 1.2
    league_def = float(np.mean([v["defn"] for v in strengths.values()])) if strengths else 1.2
    ta = strengths.get(team, {}).get("att", league_att)
    td = strengths.get(team, {}).get("defn", league_def)
    oa = strengths.get(opp, {}).get("att", league_att)
    od = strengths.get(opp, {}).get("defn", league_def)
    pos = normalize_position(str(position))
    if pos in ("GKP", "DEF"):
        # Easier when opponent attack is weak / we concede little
        raw = 0.65 + 0.7 * ((league_att / max(oa, 0.4)) * (league_def / max(td, 0.4))) ** 0.5 / 1.2
    else:
        raw = 0.65 + 0.7 * ((ta / max(league_att, 0.4)) * (league_def / max(od, 0.4))) ** 0.5 / 1.2
    return float(np.clip(raw, 0.55, 1.35))


def _score_fixture_cov(
    hist: pd.DataFrame,
    home_team: str,
    away_team: str,
    gw: int,
    fd_row_home: pd.Series,
    *,
    alpha_g: float,
    alpha_a: float,
    n_scenarios: int,
    seed: int,
) -> pd.DataFrame:
    """Return per-player draw-mean xP for one fixture (both sides)."""
    home, _ = build_club_side(hist, home_team, min_mean_minutes=MIN_MEAN_M)
    away, _ = build_club_side(hist, away_team, min_mean_minutes=MIN_MEAN_M)
    h_groups = asof_groups_for_team(hist, home_team, gw)
    a_groups = asof_groups_for_team(hist, away_team, gw)
    h_keep = set(home["element"].astype(int))
    a_keep = set(away["element"].astype(int))
    h_groups = [g for g in h_groups if set(g.elements) <= h_keep]
    a_groups = [g for g in a_groups if set(g.elements) <= a_keep]

    lam_h = float(fd_row_home.get("lam_h", 1.5))
    lam_a = float(fd_row_home.get("lam_a", 1.2))
    rho = float(fd_row_home.get("rho", -0.05))
    if not np.isfinite(lam_h):
        lam_h = 1.5
    if not np.isfinite(lam_a):
        lam_a = 1.2
    if not np.isfinite(rho):
        rho = -0.05

    draws = draw_fixture_covariant(
        lam_h,
        lam_a,
        rho,
        home,
        away,
        home_groups=h_groups,
        away_groups=a_groups,
        n_scenarios=n_scenarios,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        seed=seed,
    )
    def _upside_semi_sd(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        mu = float(x.mean())
        excess = np.maximum(x - mu, 0.0)
        return float(np.sqrt(np.mean(excess**2)))

    def _downside_semi_sd(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        mu = float(x.mean())
        short = np.maximum(mu - x, 0.0)
        return float(np.sqrt(np.mean(short**2)))

    rows = []
    for side_df, side_name, offset in (
        (home, "home", 0),
        (away, "away", len(home)),
    ):
        for j, r in enumerate(side_df.itertuples(index=False)):
            col = offset + j
            col_xp = draws.xp[:, col]
            rows.append(
                {
                    "element": int(r.element),
                    "player_id": str(r.element),
                    "name": str(r.name),
                    "position": str(r.position),
                    "team": home_team if side_name == "home" else away_team,
                    "club": home_team if side_name == "home" else away_team,
                    "mu_draw": float(col_xp.mean()),
                    "sd_draw": float(col_xp.std(ddof=1)) if col_xp.size > 1 else 0.0,
                    "upside_semi_sd": _upside_semi_sd(col_xp),
                    "downside_semi_sd": _downside_semi_sd(col_xp),
                    "mu_point": float(draws.mu_point[col]),
                    "side": side_name,
                }
            )
    return pd.DataFrame(rows)


def score_priced_week(
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    decision_gw: int,
    target_gw: int,
    alpha_g: float,
    alpha_a: float,
    n_scenarios: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """Project frozen priors at decision_gw onto target_gw fixtures via FD + cov draws."""
    hist = merged.loc[merged["GW"] < int(decision_gw)].copy()
    if hist.empty:
        return pd.DataFrame(), 0
    sched = _schedule(merged)
    week_fx = sched.loc[sched["GW"] == int(target_gw)]
    if week_fx.empty:
        return pd.DataFrame(), 0

    parts: list[pd.DataFrame] = []
    n_ok = 0
    for i, row in enumerate(week_fx.itertuples(index=False)):
        # FD home-side row for λ + moments
        fd_h = fd.loc[
            (fd["team_n"] == row.home_n) & (fd["date_iso"] == row.date_iso)
        ]
        if fd_h.empty:
            continue
        try:
            scored = _score_fixture_cov(
                hist,
                str(row.home_team),
                str(row.away_team),
                int(decision_gw),
                fd_h.iloc[0],
                alpha_g=alpha_g,
                alpha_a=alpha_a,
                n_scenarios=n_scenarios,
                seed=seed + 17 * i + 1009 * int(target_gw),
            )
        except (ValueError, KeyError, RuntimeError):
            continue
        scored["fixture"] = int(row.fixture)
        scored["gw"] = int(target_gw)
        parts.append(scored)
        n_ok += 1

    if not parts:
        return pd.DataFrame(), 0
    all_p = pd.concat(parts, ignore_index=True)
    # DGW: sum μ across fixtures that week
    board = (
        all_p.groupby("element", as_index=False)
        .agg(
            player_id=("player_id", "first"),
            name=("name", "first"),
            position=("position", "first"),
            team=("team", "first"),
            club=("club", "first"),
            mu_draw=("mu_draw", "sum"),
            # Independent fixtures → RSS of per-fixture draw SDs / upside semi-SDs.
            sd_draw=(
                "sd_draw",
                lambda s: float(np.sqrt(np.sum(np.asarray(s, dtype=float) ** 2))),
            ),
            upside_semi_sd=(
                "upside_semi_sd",
                lambda s: float(np.sqrt(np.sum(np.asarray(s, dtype=float) ** 2))),
            ),
            downside_semi_sd=(
                "downside_semi_sd",
                lambda s: float(np.sqrt(np.sum(np.asarray(s, dtype=float) ** 2))),
            ),
            mu_point=("mu_point", "sum"),
            n_fixtures=("fixture", "nunique"),
        )
    )
    board["mu_post"] = board["mu_draw"]
    board["gw"] = int(target_gw)
    return board, n_ok


def score_week_cov_risk(
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    decision_gw: int,
    target_gw: int,
    alpha_g: float,
    alpha_a: float,
    n_scenarios: int,
    seed: int,
) -> pd.DataFrame:
    """Per-element fixture-covariance risk columns for ``target_gw`` (as-of ``decision_gw``).

    Does **not** replace μ — risk columns only. Empty if no fixtures score.
    """
    board, _ = score_priced_week(
        merged,
        fd,
        decision_gw=decision_gw,
        target_gw=target_gw,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        n_scenarios=n_scenarios,
        seed=seed,
    )
    if board.empty:
        return board
    cols = [
        "element",
        "sd_draw",
        "upside_semi_sd",
        "downside_semi_sd",
        "mu_draw",
        "n_fixtures",
    ]
    return board[[c for c in cols if c in board.columns]].copy()


def _attach_prices(
    board: pd.DataFrame, prices: pd.DataFrame, gw: int, *, season: str
) -> pd.DataFrame:
    pr = prices.loc[prices["gw"] == int(gw)].copy()
    pr["element"] = pd.to_numeric(pr["element"], errors="coerce")
    out = board.merge(
        pr[["element", "value", "team_norm", "player_id"]],
        on="element",
        how="left",
        suffixes=("_el", "_pr"),
    )
    # Prefer store/season player_id (e.g. 2025-26:123); fall back to element
    if "player_id_pr" in out.columns:
        out["player_id"] = out["player_id_pr"].astype(str)
    elif "player_id" in out.columns:
        out["player_id"] = out["player_id"].astype(str)
    else:
        out["player_id"] = out["element"].astype(str)
    miss_pid = out["player_id"].isna() | (out["player_id"] == "nan")
    out.loc[miss_pid, "player_id"] = (
        season + ":" + out.loc[miss_pid, "element"].astype(int).astype(str)
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    miss = out["value"].isna()
    if miss.any():
        earlier = prices.loc[prices["gw"] <= int(gw)].sort_values("gw")
        last = earlier.drop_duplicates("element", keep="last")
        last["element"] = pd.to_numeric(last["element"], errors="coerce")
        fill = out.loc[miss, ["element"]].merge(
            last[["element", "value", "player_id"]], on="element", how="left"
        )
        out.loc[miss, "value"] = fill["value"].to_numpy()
        still = out["player_id"].isna() | (out["player_id"] == "nan")
        if still.any() and "player_id" in fill.columns:
            out.loc[still, "player_id"] = fill["player_id"].astype(str).to_numpy()
    out["value"] = out["value"].fillna(45).astype(int)
    out["club"] = out.get("team_norm", out["club"]).fillna(out["club"]).astype(str)
    if "team_norm" in out.columns:
        out["club"] = out["team_norm"].fillna(out["club"]).astype(str)
    out["position"] = out["position"].map(normalize_position)
    out["score_xp_v2"] = out["mu_post"]
    out["xp_se"] = 0.0
    out["total_points"] = 0.0
    return out


def _position_means(board: pd.DataFrame) -> dict[str, float]:
    out = {}
    for p, g in board.groupby("position"):
        out[str(p)] = float(g["mu_post"].mean())
    return out


def _opp_map(merged: pd.DataFrame, gw: int) -> dict[str, str]:
    sched = _schedule(merged)
    week = sched.loc[sched["GW"] == int(gw)]
    m: dict[str, str] = {}
    for r in week.itertuples(index=False):
        m[str(r.home_team)] = str(r.away_team)
        m[str(r.away_team)] = str(r.home_team)
    return m


@dataclass
class PricedCache:
    """Shared t:t+1 boards for one decision GW (reuse across arms)."""

    pools: list[pd.DataFrame]
    universe: pd.DataFrame
    priced_gws: tuple[int, ...]
    n_fixtures_scored: int
    alpha_g: float
    alpha_a: float
    hist: pd.DataFrame
    avail: list[int]


def build_priced_cache(
    season: str,
    decision_gw: int,
    *,
    n_scenarios: int = N_SCENARIOS,
    seed: int = 0,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> PricedCache:
    if merged is None:
        merged = _load_merged_enriched(season)
    if fd is None:
        fd = _fd_index(season)
    if prices is None:
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)

    hist = merged.loc[merged["GW"] < int(decision_gw)]
    alpha_g, alpha_a = _asof_alphas(hist)
    avail = sorted(int(g) for g in merged["GW"].unique())
    priced_gws = [g for g in avail if decision_gw <= g < decision_gw + BOOKIE_DEPTH]
    priced_boards: list[pd.DataFrame] = []
    n_fx = 0
    for g in priced_gws:
        board, n = score_priced_week(
            merged,
            fd,
            decision_gw=decision_gw,
            target_gw=g,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            n_scenarios=n_scenarios,
            seed=seed,
        )
        if board.empty:
            continue
        board = _attach_prices(board, prices, decision_gw, season=season)
        priced_boards.append(board)
        n_fx += n
    if not priced_boards:
        raise RuntimeError(f"no priced boards for {season} GW{decision_gw}")

    universe = priced_boards[0].copy()

    def _reindex(board: pd.DataFrame) -> pd.DataFrame:
        by = board.set_index(board["player_id"].astype(str))
        rows = []
        for pid in universe["player_id"].astype(str):
            u = universe.loc[universe["player_id"].astype(str) == pid].iloc[0]
            if pid in by.index:
                r = by.loc[pid]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                mu = float(r["mu_post"])
            else:
                mu = 0.0
            rows.append(
                {
                    "element": int(u["element"]),
                    "player_id": pid,
                    "name": str(u["name"]),
                    "position": str(u["position"]),
                    "club": str(u["club"]),
                    "team": str(u["team"]),
                    "value": int(u["value"]),
                    "mu_post": mu,
                    "score_xp_v2": mu,
                    "xp_se": 0.0,
                    "total_points": 0.0,
                }
            )
        return pd.DataFrame(rows)

    return PricedCache(
        pools=[_reindex(b) for b in priced_boards],
        universe=universe,
        priced_gws=tuple(int(g) for g in priced_gws[: len(priced_boards)]),
        n_fixtures_scored=n_fx,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        hist=hist,
        avail=avail,
    )


def extend_arm(
    cache: PricedCache,
    arm: ArmName,
    *,
    season: str,
    decision_gw: int,
    merged: pd.DataFrame,
    h_extend: int = H_EXTEND,
    gamma_fill: float = GAMMA_FILL,
    n_scenarios: int = N_SCENARIOS,
) -> BookieHorizon:
    """Apply beyond-t+2 policy onto a shared priced cache."""
    pools = [p.copy() for p in cache.pools]
    filler_gws: list[int] = []
    universe = cache.universe

    if arm != "A_hard_stop":
        strengths = _team_strength_asof(cache.hist)
        last = pools[-1].copy()
        pos_mean = _position_means(last)
        stack = pd.concat(pools, ignore_index=True)
        mean_mu = stack.groupby("player_id", as_index=False).agg(mu_mean=("mu_post", "mean"))
        mean_map = dict(zip(mean_mu["player_id"].astype(str), mean_mu["mu_mean"], strict=True))
        last_map = dict(
            zip(last["player_id"].astype(str), last["mu_post"].astype(float), strict=True)
        )
        n_fill = max(0, int(h_extend) - len(pools))
        for k in range(n_fill):
            gw_f = int(decision_gw) + BOOKIE_DEPTH + k
            filler_gws.append(gw_f)
            opp = _opp_map(merged, gw_f) if gw_f in cache.avail else {}
            rows = []
            for r in universe.itertuples(index=False):
                pid = str(r.player_id)
                team = str(r.team)
                if arm == "B_crude_fdr":
                    o = opp.get(team, team)
                    scale = crude_fdr_scale(team, o, str(r.position), strengths)
                    mu = float(mean_map.get(pid, 0.0)) * scale
                else:
                    w = float(gamma_fill) ** (k + 1)
                    mu = w * float(last_map.get(pid, 0.0)) + (1.0 - w) * float(
                        pos_mean.get(str(r.position), 0.0)
                    )
                rows.append(
                    {
                        "element": int(r.element),
                        "player_id": pid,
                        "name": str(r.name),
                        "position": str(r.position),
                        "club": str(r.club),
                        "team": team,
                        "value": int(r.value),
                        "mu_post": mu,
                        "score_xp_v2": mu,
                        "xp_se": 0.0,
                        "total_points": 0.0,
                    }
                )
            pools.append(pd.DataFrame(rows))

    return BookieHorizon(
        season=season,
        decision_gw=int(decision_gw),
        arm=arm,
        pools=pools,
        priced_gws=cache.priced_gws,
        filler_gws=tuple(filler_gws),
        n_fixtures_scored=cache.n_fixtures_scored,
        n_scenarios=n_scenarios,
        meta={
            "alpha_g": cache.alpha_g,
            "alpha_a": cache.alpha_a,
            "h": len(pools),
            "arm": arm,
        },
    )


def build_horizon(
    season: str,
    decision_gw: int,
    arm: ArmName,
    *,
    n_scenarios: int = N_SCENARIOS,
    h_extend: int = H_EXTEND,
    gamma_fill: float = GAMMA_FILL,
    seed: int = 0,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> BookieHorizon:
    """Build aligned weekly pools for one decision GW and arm."""
    if merged is None:
        merged = _load_merged_enriched(season)
    cache = build_priced_cache(
        season,
        decision_gw,
        n_scenarios=n_scenarios,
        seed=seed,
        merged=merged,
        fd=fd,
        prices=prices,
    )
    return extend_arm(
        cache,
        arm,
        season=season,
        decision_gw=decision_gw,
        merged=merged,
        h_extend=h_extend,
        gamma_fill=gamma_fill,
        n_scenarios=n_scenarios,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=10)
    p.add_argument("--arm", default="A_hard_stop", choices=list(ARMS))
    p.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    args = p.parse_args(argv)
    hz = build_horizon(
        args.season, args.gw, args.arm, n_scenarios=args.n_scenarios  # type: ignore[arg-type]
    )
    top = hz.pools[0].sort_values("mu_post", ascending=False).head(8)
    print(
        json.dumps(
            {
                "season": hz.season,
                "gw": hz.decision_gw,
                "arm": hz.arm,
                "H": len(hz.pools),
                "priced": hz.priced_gws,
                "filler": hz.filler_gws,
                "n_fixtures": hz.n_fixtures_scored,
                "top": top[["name", "position", "mu_post"]].to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
