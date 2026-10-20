"""L4 priced path: decision μ at t, bookie-projected μ at t+1 only.

Week ``t`` uses the same board as Layer 3 (``mu_post`` ≡ ``score_xp_v2``;
shrink retired 2026-10-16). Week ``t+1`` rebuilds ``score_xp_v2`` with
priors/α/minutes frozen on GW ``< t`` and opening book pots for that
fixture (identity attach into ``mu_post``).

PI 2026-10-16: ``BOOKIE_DEPTH=2`` — do not forecast weeks without bookie
lines (live Odds API covers ~t and t+1). Hard stop; no t+2 fill.

Covariance draws are **not** used here (reserved for risk analysis).

  python -m src.strategy.layer4_mupost_priced --season 2025-26 --gw 12
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_fd_moments
from src.rules.fpl_2026 import normalize_position
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer4_bookie_path import (
    BOOKIE_DEPTH,
    MIN_MEAN_M,
    _fd_index,
    _load_merged_enriched,
    _schedule,
)
from src.strategy.shrinkage import CohortMoments
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


@dataclass(frozen=True)
class MupostPricedHorizon:
    season: str
    decision_gw: int
    pools: list[pd.DataFrame]
    priced_gws: tuple[int, ...]
    n_projected_fixtures: int
    meta: dict[str, Any]


def _pos_norm(p: object) -> str:
    key = str(p).strip().upper()
    if key in ("GK", "GKP"):
        return "GKP"
    # AM (Assistant Manager) already dropped in load_merged.
    return normalize_position(key)


def _mean_minutes_asof(
    hist: pd.DataFrame,
    *,
    rho: float = 1.0,
    window: int = 3,
) -> pd.DataFrame:
    """As-of minutes roster for projection.

    ``rho=1`` (default): expanding mean of all GW ``< t`` minutes (legacy L4).
    ``rho<1``: geometric decay on the last ``window`` GW rows (zeros included) —
    same law as ``protocol.score_xp_v2_minutes_decay`` / ``rolling_game_minutes``.
    """
    if hist.empty:
        return pd.DataFrame()
    rho = float(rho)
    if rho >= 1.0 - 1e-12:
        g = (
            hist.groupby("element", as_index=False)
            .agg(
                name=("name", "first"),
                position=("position", "first"),
                team=("team", "first"),
                m=("minutes", "mean"),
                sigma_m=("minutes", "std"),
            )
        )
        g["sigma_m"] = g["sigma_m"].fillna(10.0).clip(lower=5.0)
        g = g.loc[g["m"] >= MIN_MEAN_M].copy()
        g["position"] = g["position"].map(_pos_norm)
        return g

    from src.live.news_packets import rolling_game_minutes, rolling_game_minutes_sigma

    rows: list[dict[str, Any]] = []
    h = hist.copy()
    h["minutes"] = pd.to_numeric(h["minutes"], errors="coerce").fillna(0.0)
    h["GW"] = pd.to_numeric(h["GW"], errors="coerce")
    h = h.dropna(subset=["element", "GW"])
    for eid, g in h.groupby("element", sort=False):
        by_gw = g.sort_values("GW").groupby("GW", sort=True)["minutes"].max()
        series = [float(v) for v in by_gw.tolist()]
        m_hat = rolling_game_minutes(series, window=int(window), rho=rho)
        if m_hat is None or float(m_hat) < MIN_MEAN_M:
            continue
        sig = rolling_game_minutes_sigma(series, window=int(window))
        rows.append(
            {
                "element": eid,
                "name": g["name"].iloc[0] if "name" in g.columns else str(eid),
                "position": g["position"].iloc[0],
                "team": g["team"].iloc[0],
                "m": float(m_hat),
                "sigma_m": float(max(sig, 5.0)),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["position"] = out["position"].map(_pos_norm)
    return out


def project_week_score_xp_v2(
    hist: pd.DataFrame,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    *,
    decision_gw: int,
    target_gw: int,
    alpha_g: float,
    alpha_a: float,
    minutes_rho: float = 1.0,
    minutes_window: int = 3,
) -> tuple[pd.DataFrame, int]:
    """Point ``score_xp_v2`` for target_gw with frozen priors + FD pots.

    Player shares use ``hist`` only (same season, GW ``< decision_gw``) —
    no prior-season fill.
    """
    hist = numagent_xp.asof_share_history(hist, before_gw=int(decision_gw))
    roster = _mean_minutes_asof(hist, rho=minutes_rho, window=minutes_window)
    if roster.empty:
        return pd.DataFrame(), 0

    rates = numagent_xp.player_rates_from_history(hist, id_col="element")
    priors = numagent_xp.history_priors_for_xp(hist, id_col="element")
    side = roster.merge(rates, on="element", how="left").merge(
        priors.drop(columns=["position"], errors="ignore"),
        on="element",
        how="left",
        suffixes=("", "_pr"),
    )
    for c, default in (
        ("rG", 0.0),
        ("drG", 0.0),
        ("rA", 0.0),
        ("drA", 0.0),
        ("bonus_mu", 0.2),
        ("bonus_se", 0.1),
        ("card_mu", -0.1),
        ("card_se", 0.05),
        ("saves_mu", 0.0),
        ("saves_se", 0.0),
        ("defcon_mu", 0.0),
        ("defcon_sig", numagent_xp.SIG_FLOOR),
        ("defcon_n", 5),
    ):
        if c not in side.columns:
            side[c] = default
        else:
            side[c] = pd.to_numeric(side[c], errors="coerce").fillna(default)
    side["player_id"] = side["element"].astype(str)
    side["team_n"] = side["team"].map(norm_team)

    sched = _schedule(merged)
    week_fx = sched.loc[sched["GW"] == int(target_gw)]
    if week_fx.empty:
        return pd.DataFrame(), 0

    parts: list[pd.DataFrame] = []
    n_ok = 0
    for row in week_fx.itertuples(index=False):
        for team, team_n, side_name in (
            (str(row.home_team), str(row.home_n), "home"),
            (str(row.away_team), str(row.away_n), "away"),
        ):
            fd_row = fd.loc[(fd["team_n"] == team_n) & (fd["date_iso"] == row.date_iso)]
            if fd_row.empty:
                continue
            club = side.loc[side["team"].astype(str) == team].copy()
            if club.empty:
                continue
            fr = fd_row.iloc[0]
            scored = numagent_xp.score_team_side(
                club,
                xg=float(fr["xg"]),
                xcs=float(fr["xcs"]),
                xgc=float(fr["xgc"]),
                alpha_g=alpha_g,
                alpha_a=alpha_a,
            )
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["club"] = team
            scored["fixture"] = int(row.fixture)
            scored["gw"] = int(target_gw)
            scored["side"] = side_name
            parts.append(scored)
            n_ok += 1

    if not parts:
        return pd.DataFrame(), 0
    all_p = pd.concat(parts, ignore_index=True)
    board = (
        all_p.groupby("element", as_index=False)
        .agg(
            name=("name", "first") if "name" in all_p.columns else ("element", "first"),
            position=("position", "first"),
            team=("team", "first"),
            club=("club", "first"),
            score_xp_v2=("xp", "sum"),
            xp_se=("xp_se", lambda s: float(np.sqrt((s.astype(float) ** 2).sum()))),
        )
    )
    if "name" not in board.columns or board["name"].isna().all():
        board = board.drop(columns=["name"], errors="ignore")
        board = board.merge(
            roster[["element", "name"]], on="element", how="left"
        )
    board["gw"] = int(target_gw)
    return board, n_ok // 2  # counted sides; report fixtures ≈ sides/2


def _attach_store_ids(
    board: pd.DataFrame, prices: pd.DataFrame, gw: int, season: str
) -> pd.DataFrame:
    pr = prices.loc[prices["gw"] == int(gw)].copy()
    pr["element"] = pd.to_numeric(pr["element"], errors="coerce")
    out = board.merge(
        pr[["element", "value", "team_norm", "player_id"]],
        on="element",
        how="left",
        suffixes=("", "_pr"),
    )
    if "player_id" in out.columns and out["player_id"].notna().any():
        out["player_id"] = out["player_id"].astype(str)
    else:
        out["player_id"] = season + ":" + out["element"].astype(int).astype(str)
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
        if still.any():
            out.loc[still, "player_id"] = fill["player_id"].astype(str).to_numpy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(45).astype(int)
    if "team_norm" in out.columns:
        out["club"] = out["team_norm"].fillna(out.get("club", out["team"])).astype(str)
    else:
        out["club"] = out["team"].astype(str)
    out["position"] = out["position"].map(normalize_position)
    out["total_points"] = 0.0
    return out


def _align_to_week0(week0: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Same player universe / meta as week-0; μ from other (0 if missing)."""
    by = other.set_index(other["player_id"].astype(str))
    mu_map = {}
    se_map = {}
    raw_map = {}
    for pid, o in by.iterrows():
        mu_map[str(pid)] = float(o["mu_post"])
        se_map[str(pid)] = float(o["xp_se"]) if "xp_se" in o.index else 0.0
        raw_map[str(pid)] = float(o["score_xp_v2"]) if "score_xp_v2" in o.index else mu_map[str(pid)]
    out = week0.copy()
    pids = out["player_id"].astype(str)
    out["mu_post"] = pids.map(lambda p: mu_map.get(p, 0.0)).astype(float)
    out["score_xp_v2"] = pids.map(lambda p: raw_map.get(p, 0.0)).astype(float)
    out["xp_se"] = pids.map(lambda p: se_map.get(p, 0.0)).astype(float)
    out["total_points"] = 0.0
    return out.reset_index(drop=True)


def build_mupost_priced_horizon(
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    minutes_rho: float = 1.0,
    minutes_window: int = 3,
) -> MupostPricedHorizon:
    """Return pools length ≤ 2: week0 mu_post + projected mu_post for t+1."""
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

    # Week 0: caller supplies L3 board (already has mu_post)
    w0 = week0_pool.copy()
    if "mu_post" not in w0.columns:
        raise ValueError("week0_pool needs mu_post")
    if "element" not in w0.columns and "player_id" in w0.columns:
        # try parse
        w0["element"] = (
            pd.to_numeric(w0["player_id"].astype(str).str.split(":").str[-1], errors="coerce")
        )
    pools = [w0]
    priced_gws = [int(decision_gw)]

    hist = merged.loc[merged["GW"] < int(decision_gw)]
    alpha_g, alpha_a = _asof_alphas(hist)
    avail = set(int(g) for g in merged["GW"].unique())
    n_fx = 0

    for step in range(1, BOOKIE_DEPTH):
        tgt = int(decision_gw) + step
        if tgt not in avail:
            continue
        raw, n = project_week_score_xp_v2(
            hist,
            merged,
            fd,
            decision_gw=decision_gw,
            target_gw=tgt,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            minutes_rho=minutes_rho,
            minutes_window=minutes_window,
        )
        if raw.empty:
            continue
        n_fx += n
        board = _attach_store_ids(raw, prices, decision_gw, season)
        board, _ = attach_mu_post(
            board,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            out_col="mu_post",
        )
        pools.append(_align_to_week0(w0, board))
        priced_gws.append(tgt)

    return MupostPricedHorizon(
        season=season,
        decision_gw=int(decision_gw),
        pools=pools,
        priced_gws=tuple(priced_gws),
        n_projected_fixtures=n_fx,
        meta={
            "alpha_g": alpha_g,
            "alpha_a": alpha_a,
            "H": len(pools),
            "minutes_rho": float(minutes_rho),
            "minutes_window": int(minutes_window),
        },
    )


def main(argv: list[str] | None = None) -> int:
    from src.strategy.decision_value import attach_mu_post as _attach
    from src.strategy.shrink_climb import SeasonStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=12)
    args = p.parse_args(argv)
    v2 = pd.read_csv(PROCESSED / "score_xp_v2_closed.csv")
    store = SeasonStore(args.season, v2)
    pool = store.pool(args.gw)
    moments = store.moments_asof(args.gw)
    work, _ = _attach(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    hz = build_mupost_priced_horizon(
        args.season, args.gw, week0_pool=work, moments=moments
    )
    print(
        json.dumps(
            {
                "H": len(hz.pools),
                "priced_gws": hz.priced_gws,
                "n_projected_fixtures": hz.n_projected_fixtures,
                "week0_top": work.sort_values("mu_post", ascending=False)
                .head(5)[["name", "mu_post"]]
                .to_dict(orient="records"),
                "week1_top": hz.pools[1].sort_values("mu_post", ascending=False)
                .head(5)[["name", "mu_post"]]
                .to_dict(orient="records")
                if len(hz.pools) > 1
                else [],
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
