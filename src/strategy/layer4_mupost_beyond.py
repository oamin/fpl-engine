"""Beyond-t+2 fillers on the priced μ_post path (arms A/B/C/D*).

A_hard_stop — H = n_priced (typically 3); no filler
B_crude_fdr — extend to H=5; μ = mean(priced μ)×as-of gf/ga FDR vs opponent
C_decay     — extend to H=5; μ = γ^k·last_priced + (1−γ^k)·pos mean
D_team_form — fixed amp×tanh FDR (parked — lost to A)
D_adapt_const — as-of OLS amp; scale = 1 + amp·tanh(FDR/τ) on fill weeks
D_adapt_decay — as-of amp + ρ^k (k=0 at t+3); stronger shrink further out

Priced window t:t+2 unchanged (FD pots → score_xp_v2 → constant-K shrink).
Fillers never use that week's opening line. Scale fit uses only outcomes
with target_gw < decision_gw.

  python -m src.strategy.layer4_mupost_beyond --season 2025-26 --gw 12 --arm D_adapt_decay
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.strategy.layer4_bookie_path import (
    BOOKIE_DEPTH,
    GAMMA_FILL,
    H_EXTEND,
    _fd_index,
    _load_merged_enriched,
    _opp_map,
    _position_means,
    _schedule,
    _team_strength_asof,
    crude_fdr_scale,
)
from src.strategy.layer4_mupost_priced import (
    MupostPricedHorizon,
    build_mupost_priced_horizon,
)
from src.strategy.shrinkage import CohortMoments
from src.strategy.team_form_fdr import _actual_map, _form_asof, _team_gw_actuals
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# Multiplicative fill scale from team-form FDR (same clip band as crude_fdr_scale).
TEAM_FORM_FDR_TAU = 8.0
TEAM_FORM_FDR_AMP = 0.35
TEAM_FORM_SCALE_LO = 0.55
TEAM_FORM_SCALE_HI = 1.35
ADAPT_MIN_PAIRS = 40
ADAPT_RHO_GRID = tuple(round(x, 2) for x in np.linspace(0.4, 1.0, 13))

BeyondArm = Literal[
    "A_hard_stop",
    "B_crude_fdr",
    "C_decay",
    "D_team_form",
    "D_adapt_const",
    "D_adapt_decay",
    "E_strength_gamma",
    "E_strength_gamma_decay",
]
BEYOND_ARMS: tuple[BeyondArm, ...] = (
    "A_hard_stop",
    "B_crude_fdr",
    "C_decay",
    "D_team_form",
    "D_adapt_const",
    "D_adapt_decay",
    "E_strength_gamma",
    "E_strength_gamma_decay",
)
# Combined treatment: strength×γ fills + last-3 minutes decay on projections.
E_DECAY_MINUTES_RHO = 0.7
E_DECAY_MINUTES_WINDOW = 3
E_STRENGTH_ARMS: tuple[BeyondArm, ...] = ("E_strength_gamma", "E_strength_gamma_decay")
D_ADAPT_ARMS: tuple[BeyondArm, ...] = ("D_adapt_const", "D_adapt_decay")


@dataclass(frozen=True)
class MupostBeyondHorizon:
    season: str
    decision_gw: int
    arm: BeyondArm
    pools: list[pd.DataFrame]
    priced_gws: tuple[int, ...]
    filler_gws: tuple[int, ...]
    n_projected_fixtures: int
    meta: dict[str, Any]


def _team_col(df: pd.DataFrame) -> pd.Series:
    if "team" in df.columns:
        return df["team"].astype(str)
    if "club" in df.columns:
        return df["club"].astype(str)
    raise ValueError("pool needs team or club")


def team_form_mu_asof(merged: pd.DataFrame, decision_gw: int) -> dict[str, float]:
    """Expanding club-pts mean (managers stripped), keyed by norm_team."""
    panel = _team_gw_actuals(merged)
    form = _form_asof(panel, int(decision_gw))
    if form.empty:
        return {}
    return dict(zip(form["team"].astype(str), form["team_mu"].astype(float), strict=True))


@dataclass(frozen=True)
class TeamFormScaleFit:
    """As-of multiplicative FDR scale: 1 + amp · ρ^k · tanh(FDR/τ)."""

    amp: float
    rho: float
    tau: float
    n_pairs: int
    league_mu: float
    mode: str  # const | decay | fallback
    sse: float


def _fdr_from_form(team: str, opp: str, form_mu: dict[str, float]) -> float:
    if not form_mu:
        return 0.0
    league = float(np.mean(list(form_mu.values())))
    own = float(form_mu.get(norm_team(str(team)), league))
    omu = float(form_mu.get(norm_team(str(opp)), league))
    return own - omu


def team_form_fdr_scale(
    team: str,
    opp: str,
    form_mu: dict[str, float],
    *,
    tau: float = TEAM_FORM_FDR_TAU,
    amp: float = TEAM_FORM_FDR_AMP,
    rho: float = 1.0,
    fill_k: int = 0,
) -> float:
    """Map FDR = μ_own − μ_opp → multiplicative scale in ~[0.55, 1.35]."""
    fdr = _fdr_from_form(team, opp, form_mu)
    z = float(np.tanh(fdr / max(float(tau), 1e-6)))
    raw = 1.0 + float(amp) * (float(rho) ** int(fill_k)) * z
    return float(np.clip(raw, TEAM_FORM_SCALE_LO, TEAM_FORM_SCALE_HI))


def _asof_fdr_pairs(
    merged: pd.DataFrame,
    decision_gw: int,
    *,
    lead_min: int = BOOKIE_DEPTH,
    lead_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Past (z, k, y_rel) with target_gw < decision_gw only.

    z = tanh(FDR/τ), k = lead − lead_min, y_rel = pts_margin / league_mu.
    """
    if lead_max is None:
        lead_max = lead_min + max(0, H_EXTEND - BOOKIE_DEPTH) - 1
    panel = _team_gw_actuals(merged)
    sched = _schedule(merged)
    t_dec = int(decision_gw)
    zs: list[float] = []
    ks: list[float] = []
    ys: list[float] = []
    league_vals: list[float] = []

    for t in sorted(int(g) for g in panel["gw"].unique()):
        if t >= t_dec:
            break
        form = _form_asof(panel, t)
        if form.empty:
            continue
        mu_map = dict(zip(form["team"].astype(str), form["team_mu"].astype(float), strict=True))
        league_vals.append(float(np.mean(list(mu_map.values()))))
        for lead in range(int(lead_min), int(lead_max) + 1):
            tgt = t + lead
            if tgt >= t_dec:
                continue
            week = sched.loc[sched["GW"] == tgt]
            if week.empty:
                continue
            actual = _actual_map(panel, tgt)
            if not actual:
                continue
            for r in week.itertuples(index=False):
                home = norm_team(str(r.home_team))
                away = norm_team(str(r.away_team))
                if home not in mu_map or away not in mu_map:
                    continue
                for team, opp in ((home, away), (away, home)):
                    fdr = float(mu_map[team] - mu_map[opp])
                    y = float(actual.get(team, 0.0) - actual.get(opp, 0.0))
                    zs.append(float(np.tanh(fdr / TEAM_FORM_FDR_TAU)))
                    ks.append(float(lead - lead_min))
                    ys.append(y)

    if not ys:
        return (
            np.asarray([], float),
            np.asarray([], float),
            np.asarray([], float),
            45.0,
        )
    league_mu = float(np.mean(league_vals)) if league_vals else 45.0
    league_mu = max(league_mu, 1.0)
    y_rel = np.asarray(ys, float) / league_mu
    return np.asarray(zs, float), np.asarray(ks, float), y_rel, league_mu


def _ols_amp(predictor: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Through-origin OLS amp; returns (amp, sse)."""
    denom = float(np.dot(predictor, predictor))
    if denom < 1e-12:
        return 0.0, float(np.dot(y, y))
    amp = float(np.dot(predictor, y) / denom)
    resid = y - amp * predictor
    return amp, float(np.dot(resid, resid))


def team_form_terminal_credits(
    pools: list[pd.DataFrame],
    merged: pd.DataFrame,
    decision_gw: int,
    *,
    n_beyond: int = 2,
    mode: Literal["const", "decay"] = "const",
    mu_col: str = "mu_post",
) -> tuple[dict[str, float], TeamFormScaleFit]:
    """Per-player hold credit from beyond FDR — no fill boards.

    credit_i = amp · Σ_k ρ^k · tanh(FDR_{t+3+k}) · mean_priced_μ_i
    amp (and ρ) from as-of fit; priced pools unchanged.
    """
    n_beyond = max(0, int(n_beyond))
    lead_max = BOOKIE_DEPTH + n_beyond - 1 if n_beyond else BOOKIE_DEPTH
    fit = fit_team_form_scale_asof(
        merged, int(decision_gw), mode=mode, lead_max=lead_max
    )
    form_mu = team_form_mu_asof(merged, int(decision_gw))
    if not pools or n_beyond == 0:
        return {}, fit
    stack = pd.concat(pools, ignore_index=True)
    mean_mu = stack.groupby(stack["player_id"].astype(str), as_index=False).agg(
        mu_mean=(mu_col, "mean")
    )
    mean_map = dict(
        zip(mean_mu["player_id"].astype(str), mean_mu["mu_mean"].astype(float), strict=True)
    )
    universe = pools[0]
    team_map = dict(
        zip(universe["player_id"].astype(str), _team_col(universe), strict=True)
    )
    avail = set(int(g) for g in merged["GW"].unique())
    opp_by_gw: dict[int, dict[str, str]] = {}
    for k in range(n_beyond):
        gw_f = int(decision_gw) + BOOKIE_DEPTH + k
        opp_by_gw[gw_f] = _opp_map(merged, gw_f) if gw_f in avail else {}

    credits: dict[str, float] = {}
    for pid, mu_ref in mean_map.items():
        team = team_map.get(pid, "")
        acc = 0.0
        for k in range(n_beyond):
            gw_f = int(decision_gw) + BOOKIE_DEPTH + k
            opp = opp_by_gw[gw_f].get(team, team)
            fdr = _fdr_from_form(team, opp, form_mu)
            z = float(np.tanh(fdr / max(fit.tau, 1e-6)))
            acc += (float(fit.rho) ** k) * z
        credits[pid] = float(fit.amp) * acc * float(mu_ref)
    return credits, fit


def fit_team_form_scale_asof(
    merged: pd.DataFrame,
    decision_gw: int,
    *,
    mode: Literal["const", "decay"] = "const",
    lead_min: int = BOOKIE_DEPTH,
    lead_max: int | None = None,
) -> TeamFormScaleFit:
    """Tune amp (and ρ for decay) from as-of FDR → relative pts margin."""
    z, k, y_rel, league_mu = _asof_fdr_pairs(
        merged, decision_gw, lead_min=lead_min, lead_max=lead_max
    )
    n = int(z.size)
    if n < ADAPT_MIN_PAIRS:
        rho = 1.0 if mode == "const" else 0.75
        return TeamFormScaleFit(
            amp=TEAM_FORM_FDR_AMP,
            rho=rho,
            tau=TEAM_FORM_FDR_TAU,
            n_pairs=n,
            league_mu=league_mu,
            mode="fallback",
            sse=float("nan"),
        )
    if mode == "const":
        amp, sse = _ols_amp(z, y_rel)
        amp = float(np.clip(amp, 0.0, 1.0))
        return TeamFormScaleFit(
            amp=amp,
            rho=1.0,
            tau=TEAM_FORM_FDR_TAU,
            n_pairs=n,
            league_mu=league_mu,
            mode="const",
            sse=sse,
        )
    best: TeamFormScaleFit | None = None
    for rho in ADAPT_RHO_GRID:
        pred = z * (float(rho) ** k)
        amp, sse = _ols_amp(pred, y_rel)
        amp = float(np.clip(amp, 0.0, 1.0))
        cand = TeamFormScaleFit(
            amp=amp,
            rho=float(rho),
            tau=TEAM_FORM_FDR_TAU,
            n_pairs=n,
            league_mu=league_mu,
            mode="decay",
            sse=sse,
        )
        if best is None or sse < best.sse:
            best = cand
    assert best is not None
    return best


def extend_mupost_beyond(
    base: MupostPricedHorizon,
    arm: BeyondArm,
    *,
    merged: pd.DataFrame,
    h_extend: int = H_EXTEND,
    gamma_fill: float = GAMMA_FILL,
) -> MupostBeyondHorizon:
    """Apply beyond-t+2 policy onto a priced μ_post horizon."""
    pools = [p.copy() for p in base.pools]
    filler_gws: list[int] = []
    if arm == "A_hard_stop" or not pools:
        return MupostBeyondHorizon(
            season=base.season,
            decision_gw=base.decision_gw,
            arm=arm,
            pools=pools,
            priced_gws=base.priced_gws,
            filler_gws=(),
            n_projected_fixtures=base.n_projected_fixtures,
            meta={**base.meta, "arm": arm, "H": len(pools), "filler_gws": []},
        )

    hist = merged.loc[merged["GW"] < int(base.decision_gw)]
    strengths = _team_strength_asof(hist) if arm == "B_crude_fdr" else {}
    use_form = arm in ("D_team_form", *D_ADAPT_ARMS)
    form_mu = team_form_mu_asof(merged, int(base.decision_gw)) if use_form else {}
    n_fill = max(0, int(h_extend) - len(pools))
    lead_max = BOOKIE_DEPTH + n_fill - 1 if n_fill else BOOKIE_DEPTH
    scale_fit: TeamFormScaleFit | None = None
    if arm == "D_adapt_const":
        scale_fit = fit_team_form_scale_asof(
            merged, int(base.decision_gw), mode="const", lead_max=lead_max
        )
    elif arm == "D_adapt_decay":
        scale_fit = fit_team_form_scale_asof(
            merged, int(base.decision_gw), mode="decay", lead_max=lead_max
        )
    avail = set(int(g) for g in merged["GW"].unique())
    universe = pools[0]
    last = pools[-1]
    pos_mean = _position_means(last)
    stack = pd.concat(pools, ignore_index=True)
    mean_mu = stack.groupby(stack["player_id"].astype(str), as_index=False).agg(
        mu_mean=("mu_post", "mean")
    )
    mean_map = dict(zip(mean_mu["player_id"].astype(str), mean_mu["mu_mean"], strict=True))
    last_map = dict(
        zip(last["player_id"].astype(str), last["mu_post"].astype(float), strict=True)
    )
    team_map = dict(
        zip(universe["player_id"].astype(str), _team_col(universe), strict=True)
    )

    for k in range(n_fill):
        gw_f = int(base.decision_gw) + BOOKIE_DEPTH + k
        filler_gws.append(gw_f)
        opp = _opp_map(merged, gw_f) if gw_f in avail else {}
        rows = []
        for r in universe.itertuples(index=False):
            pid = str(r.player_id)
            team = team_map.get(pid, str(getattr(r, "club", getattr(r, "team", ""))))
            pos = str(r.position)
            if arm == "B_crude_fdr":
                o = opp.get(team, team)
                scale = crude_fdr_scale(team, o, pos, strengths)
                mu = float(mean_map.get(pid, 0.0)) * scale
            elif arm == "D_team_form":
                o = opp.get(team, team)
                scale = team_form_fdr_scale(team, o, form_mu)
                mu = float(mean_map.get(pid, 0.0)) * scale
            elif arm in D_ADAPT_ARMS:
                assert scale_fit is not None
                o = opp.get(team, team)
                scale = team_form_fdr_scale(
                    team,
                    o,
                    form_mu,
                    tau=scale_fit.tau,
                    amp=scale_fit.amp,
                    rho=scale_fit.rho,
                    fill_k=k,
                )
                mu = float(mean_map.get(pid, 0.0)) * scale
            else:  # C_decay
                w = float(gamma_fill) ** (k + 1)
                mu = w * float(last_map.get(pid, 0.0)) + (1.0 - w) * float(
                    pos_mean.get(pos, 0.0)
                )
            row = {
                "player_id": pid,
                "position": pos,
                "club": str(getattr(r, "club", team)),
                "team": team,
                "value": int(getattr(r, "value", 45)),
                "mu_post": mu,
                "score_xp_v2": mu,
                "xp_se": 0.0,
                "total_points": 0.0,
                "name": str(getattr(r, "name", pid)),
            }
            if hasattr(r, "element") and pd.notna(getattr(r, "element")):
                try:
                    row["element"] = int(r.element)
                except (TypeError, ValueError):
                    pass
            rows.append(row)
        pools.append(pd.DataFrame(rows))

    meta: dict[str, Any] = {
        **base.meta,
        "arm": arm,
        "H": len(pools),
        "h_extend": int(h_extend),
        "gamma_fill": float(gamma_fill) if arm == "C_decay" else None,
        "filler_gws": filler_gws,
    }
    if arm == "D_team_form":
        meta.update(
            {
                "team_form_fdr_tau": TEAM_FORM_FDR_TAU,
                "team_form_fdr_amp": TEAM_FORM_FDR_AMP,
                "n_form_teams": len(form_mu),
            }
        )
    if scale_fit is not None:
        meta["team_form_scale_fit"] = asdict(scale_fit)
        meta["n_form_teams"] = len(form_mu)

    return MupostBeyondHorizon(
        season=base.season,
        decision_gw=base.decision_gw,
        arm=arm,
        pools=pools,
        priced_gws=base.priced_gws,
        filler_gws=tuple(filler_gws),
        n_projected_fixtures=base.n_projected_fixtures,
        meta=meta,
    )


def extend_strength_gamma(
    base: MupostPricedHorizon,
    *,
    season: str,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    h_extend: int = H_EXTEND,
    gamma_fill: float = GAMMA_FILL,
    arm: BeyondArm = "E_strength_gamma",
    minutes_rho: float = 1.0,
    minutes_window: int = 3,
) -> MupostBeyondHorizon:
    """Fill weeks: strength-only score_xp_v2 pots, then μ *= γ^{k+1}.

    Not FDR rescale of priced μ. Live-legal: no book lines beyond BOOKIE_DEPTH.
    ``E_strength_gamma_decay`` also uses last-W minutes decay on fill boards
    (priced t+1 already decayed in ``base`` when built with the same rho).
    """
    from src.markets.numagent_xp_pilot import _asof_alphas
    from src.strategy.decision_value import attach_mu_post
    from src.strategy.layer4_mupost_priced import _align_to_week0, _attach_store_ids
    from src.strategy.layer5_calendar_boards import project_week_calendar

    pools = [p.copy() for p in base.pools]
    filler_gws: list[int] = []
    n_fill = max(0, int(h_extend) - len(pools))
    if n_fill == 0 or not pools:
        return MupostBeyondHorizon(
            season=base.season,
            decision_gw=base.decision_gw,
            arm=arm,
            pools=pools,
            priced_gws=base.priced_gws,
            filler_gws=(),
            n_projected_fixtures=base.n_projected_fixtures,
            meta={
                **base.meta,
                "arm": arm,
                "H": len(pools),
                "filler_gws": [],
                "minutes_rho": float(minutes_rho),
            },
        )

    hist = merged.loc[merged["GW"] < int(base.decision_gw)]
    alpha_g, alpha_a = _asof_alphas(hist)
    avail = set(int(g) for g in merged["GW"].unique())
    n_strength_fx = 0
    for k in range(n_fill):
        gw_f = int(base.decision_gw) + BOOKIE_DEPTH + k
        filler_gws.append(gw_f)
        if gw_f not in avail:
            break
        raw, _n_fd, n_fill_fx, src = project_week_calendar(
            hist,
            merged,
            fd,
            decision_gw=int(base.decision_gw),
            target_gw=gw_f,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            minutes_rho=minutes_rho,
            minutes_window=minutes_window,
        )
        if raw.empty:
            break
        n_strength_fx += int(n_fill_fx)
        board = _attach_store_ids(raw, prices, int(base.decision_gw), season)
        board, _ = attach_mu_post(
            board,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            out_col="mu_post",
        )
        scale = float(gamma_fill) ** (k + 1)
        board["mu_post"] = board["mu_post"].astype(float) * scale
        board["score_xp_v2"] = board["score_xp_v2"].astype(float) * scale
        board["source"] = src
        board["fill_gamma"] = scale
        pools.append(_align_to_week0(pools[0], board))

    fill_label = (
        "strength_pots_x_gamma_minutes_decay"
        if float(minutes_rho) < 1.0 - 1e-12
        else "strength_pots_x_gamma"
    )
    return MupostBeyondHorizon(
        season=base.season,
        decision_gw=base.decision_gw,
        arm=arm,
        pools=pools,
        priced_gws=base.priced_gws,
        filler_gws=tuple(filler_gws),
        n_projected_fixtures=base.n_projected_fixtures + n_strength_fx,
        meta={
            **base.meta,
            "arm": arm,
            "H": len(pools),
            "h_extend": int(h_extend),
            "gamma_fill": float(gamma_fill),
            "filler_gws": filler_gws,
            "n_strength_fixtures": n_strength_fx,
            "fill": fill_label,
            "minutes_rho": float(minutes_rho),
            "minutes_window": int(minutes_window),
        },
    )


def build_mupost_beyond_horizon(
    season: str,
    decision_gw: int,
    arm: BeyondArm,
    *,
    week0_pool: pd.DataFrame,
    moments: dict[str, CohortMoments] | None,
    merged: pd.DataFrame | None = None,
    fd: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    h_extend: int = H_EXTEND,
    gamma_fill: float = GAMMA_FILL,
    minutes_rho: float | None = None,
    minutes_window: int | None = None,
) -> MupostBeyondHorizon:
    if merged is None:
        merged = _load_merged_enriched(season)
    if arm == "E_strength_gamma_decay":
        rho = float(E_DECAY_MINUTES_RHO if minutes_rho is None else minutes_rho)
        win = int(E_DECAY_MINUTES_WINDOW if minutes_window is None else minutes_window)
    elif arm == "E_strength_gamma":
        rho = float(1.0 if minutes_rho is None else minutes_rho)
        win = int(3 if minutes_window is None else minutes_window)
    else:
        rho = float(1.0 if minutes_rho is None else minutes_rho)
        win = int(3 if minutes_window is None else minutes_window)

    base = build_mupost_priced_horizon(
        season,
        decision_gw,
        week0_pool=week0_pool,
        moments=moments,
        merged=merged,
        fd=fd,
        prices=prices,
        minutes_rho=rho if arm in E_STRENGTH_ARMS else 1.0,
        minutes_window=win if arm in E_STRENGTH_ARMS else 3,
    )
    if arm in E_STRENGTH_ARMS:
        if fd is None:
            fd = _fd_index(season)
        if prices is None:
            prices = pd.read_csv(
                PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
                usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
            )
            prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        return extend_strength_gamma(
            base,
            season=season,
            merged=merged,
            fd=fd,
            prices=prices,
            moments=moments,
            h_extend=h_extend,
            gamma_fill=gamma_fill,
            arm=arm,
            minutes_rho=rho,
            minutes_window=win,
        )
    return extend_mupost_beyond(
        base, arm, merged=merged, h_extend=h_extend, gamma_fill=gamma_fill
    )


def main(argv: list[str] | None = None) -> int:
    from src.strategy.decision_value import attach_mu_post
    from src.strategy.shrink_climb import SeasonStore

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default="2025-26")
    p.add_argument("--gw", type=int, default=12)
    p.add_argument("--arm", default="C_decay", choices=list(BEYOND_ARMS))
    args = p.parse_args(argv)
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
    hz = build_mupost_beyond_horizon(
        args.season,
        args.gw,
        args.arm,  # type: ignore[arg-type]
        week0_pool=work,
        moments=moments,
    )
    print(
        json.dumps(
            {
                "arm": hz.arm,
                "H": len(hz.pools),
                "priced_gws": hz.priced_gws,
                "filler_gws": hz.filler_gws,
                "meta": hz.meta,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
