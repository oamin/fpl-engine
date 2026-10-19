"""Extended as-of μ boards for a longer chip calendar (horizon foundation).

Policy (locked 2026-10-16 — live-legal; BOOKIE_DEPTH=2 as of same day):
  1. Week t = caller L3 ``mu_post`` board.
  2. Week t+1 only: FD / bookie opening pots (same as L4 ``BOOKIE_DEPTH=2``).
  3. Weeks t+2 … t+H−1: **no bookie data** — frozen rates + schedule + as-of
     team-strength pots only (live we do not have those openings).
  4. L4 priced path stays hard-stop H≤2; this module is calendar-only.

  python -m src.strategy.layer5_calendar_boards --season 2025-26 --gw 12 --depth 8
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
from src.markets.numagent_xp_pilot import _asof_alphas
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer4_bookie_path import (
    BOOKIE_DEPTH,
    _fd_index,
    _load_merged_enriched,
    _schedule,
    _team_strength_asof,
)
from src.strategy.layer4_mupost_priced import (
    _attach_store_ids,
    _align_to_week0,
    _mean_minutes_asof,
    project_week_score_xp_v2,
)
from src.strategy.shrinkage import CohortMoments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# Calendar depth (weeks including decision week). Not the L4 BOOKIE_DEPTH=2.
DEFAULT_CALENDAR_DEPTH = 8


def _bookie_allowed(decision_gw: int, target_gw: int) -> bool:
    """True iff target is inside the live-priced window [t, t+BOOKIE_DEPTH)."""
    return int(decision_gw) <= int(target_gw) < int(decision_gw) + int(BOOKIE_DEPTH)

# Neutral-ish pots when team has no as-of history (goals per match scale).
_FALLBACK_XG = 1.3
_FALLBACK_XGC = 1.3
_FALLBACK_XCS = 0.28


@dataclass(frozen=True)
class CalendarHorizon:
    season: str
    decision_gw: int
    pools: list[pd.DataFrame]
    gws: tuple[int, ...]
    source_by_gw: dict[int, str]  # "week0" | "fd_priced" | "strength_fill"
    n_fd_fixtures: int
    n_fill_fixtures: int
    meta: dict[str, Any]


def _pots_from_strength(
    strengths: dict[str, dict[str, float]],
    team: str,
    *,
    opp: str,
    home: bool,
) -> tuple[float, float, float]:
    """Crude xG / xGC / xCS for one side from as-of attack/concede rates.

    ``strengths`` keys are raw Vaastav team names (same as ``_team_strength_asof``).
    """
    league_att = (
        float(np.mean([v["att"] for v in strengths.values()])) if strengths else _FALLBACK_XG
    )
    league_def = (
        float(np.mean([v["defn"] for v in strengths.values()])) if strengths else _FALLBACK_XGC
    )
    ta = float(strengths.get(team, {}).get("att", league_att))
    td = float(strengths.get(team, {}).get("defn", league_def))
    oa = float(strengths.get(opp, {}).get("att", league_att))
    od = float(strengths.get(opp, {}).get("defn", league_def))
    # Blend own attack with opp concede; mild home bump
    xg = 0.5 * (ta + od)
    xgc = 0.5 * (td + oa)
    if home:
        xg *= 1.08
        xgc *= 0.95
    else:
        xg *= 0.95
        xgc *= 1.05
    xg = float(np.clip(xg, 0.4, 3.0))
    xgc = float(np.clip(xgc, 0.4, 3.0))
    xcs = float(np.clip(np.exp(-xgc), 0.05, 0.55))
    return xg, xgc, xcs


def project_week_calendar(
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
) -> tuple[pd.DataFrame, int, int, str]:
    """Project one future GW under the live-legal bookie window.

    Inside ``BOOKIE_DEPTH`` (t…t+2): FD pots when joined (L4-identical).
    Beyond t+2: **strength fill only** — never read FD / bookie rows.

    Returns (board, n_fd_fixtures, n_fill_fixtures, dominant_source).
    """
    sched = _schedule(merged)
    week_fx = sched.loc[sched["GW"] == int(target_gw)]
    if week_fx.empty:
        return pd.DataFrame(), 0, 0, "empty"

    use_bookie = _bookie_allowed(decision_gw, target_gw)
    n_fx_sched = int(len(week_fx))

    if use_bookie:
        n_fd_sides = 0
        for row in week_fx.itertuples(index=False):
            for team_n in (str(row.home_n), str(row.away_n)):
                fd_row = fd.loc[
                    (fd["team_n"] == team_n) & (fd["date_iso"] == row.date_iso)
                ]
                if not fd_row.empty:
                    n_fd_sides += 1
        n_fd_fx = n_fd_sides // 2
        if n_fd_fx >= n_fx_sched:
            raw_fd, n_fd = project_week_score_xp_v2(
                hist,
                merged,
                fd,
                decision_gw=decision_gw,
                target_gw=target_gw,
                alpha_g=alpha_g,
                alpha_a=alpha_a,
                minutes_rho=minutes_rho,
                minutes_window=minutes_window,
            )
            if not raw_fd.empty:
                return raw_fd, int(n_fd), 0, "fd_priced"

    roster = _mean_minutes_asof(hist, rho=minutes_rho, window=minutes_window)
    if roster.empty:
        return pd.DataFrame(), 0, 0, "empty"

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
    strengths = _team_strength_asof(hist)

    parts: list[pd.DataFrame] = []
    n_fill_sides = 0
    n_fd_side = 0
    for row in week_fx.itertuples(index=False):
        fx = int(row.fixture)
        for team, team_n, opp, side_name, is_home in (
            (str(row.home_team), str(row.home_n), str(row.away_team), "home", True),
            (str(row.away_team), str(row.away_n), str(row.home_team), "away", False),
        ):
            club = side.loc[side["team"].astype(str) == team].copy()
            if club.empty:
                continue
            fd_row = pd.DataFrame()
            if use_bookie:
                fd_row = fd.loc[
                    (fd["team_n"] == team_n) & (fd["date_iso"] == row.date_iso)
                ]
            if use_bookie and not fd_row.empty:
                fr = fd_row.iloc[0]
                xg, xcs, xgc = float(fr["xg"]), float(fr["xcs"]), float(fr["xgc"])
                n_fd_side += 1
            else:
                xg, xgc, xcs = _pots_from_strength(
                    strengths, team, opp=opp, home=is_home
                )
                n_fill_sides += 1
            scored = numagent_xp.score_team_side(
                club, xg=xg, xcs=xcs, xgc=xgc, alpha_g=alpha_g, alpha_a=alpha_a
            )
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["club"] = team
            scored["fixture"] = fx
            scored["gw"] = int(target_gw)
            scored["side"] = side_name
            parts.append(scored)

    if not parts:
        return pd.DataFrame(), 0, 0, "empty"

    all_p = pd.concat(parts, ignore_index=True)
    board = all_p.groupby("element", as_index=False).agg(
        name=("name", "first") if "name" in all_p.columns else ("element", "first"),
        position=("position", "first"),
        team=("team", "first"),
        club=("club", "first"),
        score_xp_v2=("xp", "sum"),
        xp_se=("xp_se", lambda s: float(np.sqrt((s.astype(float) ** 2).sum()))),
    )
    if "name" not in board.columns or board["name"].isna().all():
        board = board.drop(columns=["name"], errors="ignore")
        board = board.merge(roster[["element", "name"]], on="element", how="left")
    board["gw"] = int(target_gw)
    n_fd_out = n_fd_side // 2
    n_fill_fx = n_fill_sides // 2
    if not use_bookie:
        src = "strength_fill"
    else:
        src = (
            "fd_priced"
            if n_fill_fx == 0
            else ("strength_fill" if n_fd_out == 0 else "mixed")
        )
    return board, n_fd_out, n_fill_fx, src


def build_mupost_calendar_horizon(
    season: str,
    decision_gw: int,
    *,
    week0_pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    depth: int = DEFAULT_CALENDAR_DEPTH,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> CalendarHorizon:
    """Week-0 mu_post + projected mu_post for the next ``depth-1`` schedule weeks."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
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

    w0 = week0_pool.copy()
    if "mu_post" not in w0.columns:
        raise ValueError("week0_pool needs mu_post")
    pools = [w0]
    gws = [int(decision_gw)]
    source_by_gw = {int(decision_gw): "week0"}
    n_fd = 0
    n_fill = 0

    hist = merged.loc[merged["GW"] < int(decision_gw)]
    alpha_g, alpha_a = _asof_alphas(hist)
    avail = set(int(g) for g in merged["GW"].unique())

    for step in range(1, int(depth)):
        tgt = int(decision_gw) + step
        if tgt not in avail:
            break
        raw, n_fd_w, n_fill_w, src = project_week_calendar(
            hist,
            merged,
            fd,
            decision_gw=decision_gw,
            target_gw=tgt,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
        )
        if raw.empty:
            break
        n_fd += n_fd_w
        n_fill += n_fill_w
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
        gws.append(tgt)
        source_by_gw[tgt] = src

    return CalendarHorizon(
        season=season,
        decision_gw=int(decision_gw),
        pools=pools,
        gws=tuple(gws),
        source_by_gw=source_by_gw,
        n_fd_fixtures=int(n_fd),
        n_fill_fixtures=int(n_fill),
        meta={
            "depth_requested": int(depth),
            "H": len(pools),
            "alpha_g": alpha_g,
            "alpha_a": alpha_a,
            "hist_max_gw": int(hist["GW"].max()) if len(hist) else None,
        },
    )


def audit_calendar_boards(
    *,
    season: str = "2025-26",
    gw: int = 12,
    depth: int = DEFAULT_CALENDAR_DEPTH,
) -> dict[str, Any]:
    """Smoke: depth, as-of hist, sources, no crash."""
    from src.strategy.shrink_climb import SeasonStore

    v2 = pd.read_csv(PROCESSED / "score_xp_v2_closed.csv")
    store = SeasonStore(season, v2)
    pool = store.pool(gw)
    moments = store.moments_asof(gw)
    work, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    hz = build_mupost_calendar_horizon(
        season, gw, week0_pool=work, moments=moments, depth=depth
    )
    checks = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    add("H_ge_3", len(hz.pools) >= 3, f"H={len(hz.pools)}")
    add(
        "H_reaches_or_stops",
        len(hz.pools) <= depth,
        f"H={len(hz.pools)} depth={depth}",
    )
    add("week0_first", hz.gws[0] == int(gw), f"gws={hz.gws}")
    add(
        "asof_hist",
        hz.meta.get("hist_max_gw") is not None
        and int(hz.meta["hist_max_gw"]) < int(gw),
        f"hist_max={hz.meta.get('hist_max_gw')}",
    )
    add("sources_tagged", all(g in hz.source_by_gw for g in hz.gws), "")
    # Align: every pool same player_id universe as week0
    ids0 = set(hz.pools[0]["player_id"].astype(str))
    align_ok = all(set(p["player_id"].astype(str)) == ids0 for p in hz.pools)
    add("aligned_universe", align_ok, "")
    # Live law: no FD/bookie source beyond t+2
    beyond = [
        g
        for g in hz.gws
        if not _bookie_allowed(gw, g) and hz.source_by_gw.get(g) in ("fd_priced", "mixed")
    ]
    add(
        "no_bookie_beyond_t2",
        len(beyond) == 0,
        f"violations={beyond} sources={hz.source_by_gw}",
    )
    beyond_ok = [
        g
        for g in hz.gws
        if not _bookie_allowed(gw, g) and g != int(gw)
    ]
    add(
        "beyond_is_strength",
        all(hz.source_by_gw.get(g) == "strength_fill" for g in beyond_ok),
        f"beyond={ {g: hz.source_by_gw.get(g) for g in beyond_ok} }",
    )
    n_pass = sum(1 for c in checks if c["pass"])
    return {
        "passed": n_pass == len(checks),
        "n_pass": n_pass,
        "n_checks": len(checks),
        "checks": checks,
        "horizon": {
            "gws": list(hz.gws),
            "H": len(hz.pools),
            "source_by_gw": {str(k): v for k, v in hz.source_by_gw.items()},
            "n_fd_fixtures": hz.n_fd_fixtures,
            "n_fill_fixtures": hz.n_fill_fixtures,
            "bookie_depth": BOOKIE_DEPTH,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=12)
    p.add_argument("--depth", type=int, default=DEFAULT_CALENDAR_DEPTH)
    p.add_argument("--audit", action="store_true")
    args = p.parse_args(argv)

    if args.audit:
        doc = audit_calendar_boards(season=args.season, gw=args.gw, depth=args.depth)
        out = PROCESSED / "layer5_calendar_boards_audit.json"
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        md = ROOT / "reports" / "layer5_calendar_boards_audit.md"
        lines = [
            "# Layer 5 calendar boards audit",
            "",
            "Live-legal boards: FD/bookie pots **only** inside `BOOKIE_DEPTH` (t:t+2). "
            "Beyond t+2 = as-of team-strength fill (no bookie). L4 priced path unchanged.",
            "",
            f"Cases: **{doc['n_pass']}/{doc['n_checks']}**",
            "",
            "| check | pass | detail |",
            "| --- | --- | --- |",
        ]
        for c in doc["checks"]:
            lines.append(
                f"| `{c['name']}` | {'PASS' if c['pass'] else 'FAIL'} | {c.get('detail','')} |"
            )
        lines += [
            "",
            f"Horizon GWs: `{doc['horizon']['gws']}`",
            f"Sources: `{doc['horizon']['source_by_gw']}`",
            f"FD fixtures: {doc['horizon']['n_fd_fixtures']} · fill: {doc['horizon']['n_fill_fixtures']}",
            f"Bookie depth: {doc['horizon'].get('bookie_depth')}",
            "",
            "JSON: `data/processed/layer5_calendar_boards_audit.json`",
        ]
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"passed": doc["passed"], "H": doc["horizon"]["H"]}, indent=2))
        print(f"wrote {md}")
        return 0 if doc["passed"] else 1

    from src.strategy.shrink_climb import SeasonStore

    v2 = pd.read_csv(PROCESSED / "score_xp_v2_closed.csv")
    store = SeasonStore(args.season, v2)
    pool = store.pool(args.gw)
    moments = store.moments_asof(args.gw)
    work, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    hz = build_mupost_calendar_horizon(
        args.season, args.gw, week0_pool=work, moments=moments, depth=args.depth
    )
    print(
        json.dumps(
            {
                "H": len(hz.pools),
                "gws": hz.gws,
                "source_by_gw": hz.source_by_gw,
                "n_fd": hz.n_fd_fixtures,
                "n_fill": hz.n_fill_fixtures,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
