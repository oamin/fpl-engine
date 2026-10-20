"""Long-horizon multistage MC cascade + MILP selection frequency.

Numagent 2.0 scoring (locked ``score_xp_v2``)
--------------------------------------------
From decision gameweek ``t``, for each of ``N`` Monte Carlo strands:

1. **Match scenario (GW t+2…t+n).** For each fixture, take Opta predicted
   ranks → season GF/GA (±MAE bands from closed-season place history) →
   match λ (``match_prediction`` / path blend). Sample λ inside those
   bands, build Dixon–Coles ``M`` from the sampled λ
   (``match_outcome_probs_from_lambdas``), then ``k ∼ M``
   (``match_outcome_roll``). Sampled goals update the strand's club rates
   so later weeks stay coherent.

2. **Player xP (``score_xp_v2``).** At each snapshot, freeze same-season
   as-of rates / priors / rolling minutes (``numagent_xp``) and score
   each club side with ``score_team_side`` on Dixon–Coles moments from
   the scenario λ pair — not ``xp_on_pot`` / share×λ. Generative
   attack/defence on long strands stays deferred (cascade_short owns that
   for priced t:t+1).

3. **MILP + frequency.** Selectable-pool squad MILP on discounted path
   xP; aggregate selection frequency across strands × snapshots.

Objective modes
---------------
- ``squad_sum`` (original): maximise Σ score over the 15 (bench = full weight).
- ``xi_plus_captain``: maximise Σ score over a legal XI **plus** one captain
  double (``c ≤ y ≤ x``), so premiums are valued as captain candidates.
- ``xi_plus_captain_ft``: first snapshot free rebuild; later snapshots keep a
  path squad under FT/hit cost (``V = XI + captain − 4·hits``).

Optional **minutes overlay** (``--minutes-overlay``): scale xmi/shares by FPL
``chance_of_playing_*`` / status so injured or doubtful pool players are not
full-weight. Graded — not a second hard filter.

Share seeding is **off** in v1 (see module note below). Horizon-widening
of bands is deferred to a later systematic screen. Chips off.
Not a gated likelihood promotion.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.cascade_score_v2 import (
    SCORER as SCORE_XP_V2,
    build_frozen_pool,
    score_pool_on_lam_pots,
)
from src.markets.match_outcome import (
    match_outcome_probs_from_lambdas,
    match_outcome_roll,
)
from src.models.opta_gf_ga_horizon import (
    N_MATCHES,
    gf_ga_from_rank,
    league_avg_gf,
    load_opta_ranks,
    match_prediction,
    pots_from_season_gf_ga,
    rank_gf_ga_profile,
)
from src.models.minutes_overlay import apply_availability_overlay, overlay_summary
from src.models.player_pool import POOL_JSON as DEFAULT_POOL_JSON, load_player_pool_ids
from src.teams import norm_team


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

ROOT = Path(__file__).resolve().parents[2]
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DEFAULT_GAMMA = 0.9
LOG_CSV = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
# v1: do not sample shares. Match λ already moves team-level uncertainty;
# share jitter is a second axis — open only after a calibration screen.
SEED_SHARES = False
OBJECTIVE_SQUAD_SUM = "squad_sum"
OBJECTIVE_XI_CAPTAIN = "xi_plus_captain"
OBJECTIVE_XI_CAPTAIN_FT = "xi_plus_captain_ft"
OBJECTIVES = (OBJECTIVE_SQUAD_SUM, OBJECTIVE_XI_CAPTAIN, OBJECTIVE_XI_CAPTAIN_FT)
# Path FT mode: allow this many hits beyond the free-transfer bank at a snapshot.
MAX_HITS_SNAPSHOT = 2


@dataclass
class ClubPath:
    """Per-match attack / concede rates along one MC strand.

    Initialised from Opta season GF/GA ÷ 38. Each sampled scoreline updates
    the expanding mean so later fixtures stay coherent with earlier rolls.
    """

    att: float  # goals scored per match
    dfn: float  # goals conceded per match
    n: float = 0.0  # matches observed on this strand

    def season_gf(self) -> float:
        return float(self.att) * N_MATCHES

    def season_ga(self) -> float:
        return float(self.dfn) * N_MATCHES

    def observe(self, goals_for: int, goals_against: int) -> None:
        self.n += 1.0
        self.att += (float(goals_for) - self.att) / self.n
        self.dfn += (float(goals_against) - self.dfn) / self.n


@dataclass
class StrandResult:
    strand_id: int
    scorelines: list[dict[str, Any]] = field(default_factory=list)
    squads: list[dict[str, Any]] = field(default_factory=list)


def sample_match_lams(
    lam_h: float,
    lam_a: float,
    *,
    lam_h_lo: float,
    lam_h_hi: float,
    lam_a_lo: float,
    lam_a_hi: float,
    rng: random.Random,
    centre_blend: float = 0.5,
) -> tuple[float, float]:
    """Sample λ inside Opta/path predicted bands (match-band centre_blend)."""
    lo_h, hi_h = min(lam_h_lo, lam_h_hi), max(lam_h_lo, lam_h_hi)
    lo_a, hi_a = min(lam_a_lo, lam_a_hi), max(lam_a_lo, lam_a_hi)
    u_h = lo_h + (hi_h - lo_h) * rng.random()
    u_a = lo_a + (hi_a - lo_a) * rng.random()
    w = float(min(1.0, max(0.0, centre_blend)))
    sh = max(1e-6, w * float(lam_h) + (1.0 - w) * u_h)
    sa = max(1e-6, w * float(lam_a) + (1.0 - w) * u_a)
    return sh, sa


def sample_scoreline(
    lam_h: float,
    lam_a: float,
    *,
    lam_h_lo: float,
    lam_h_hi: float,
    lam_a_lo: float,
    lam_a_hi: float,
    rng: random.Random,
    centre_blend: float = 0.5,
    rho: float = 0.0,
) -> tuple[int, int, float, float]:
    """Sample λ inside predicted bands, build ``M``, then ``k ∼ M``.

    Match outcome distribution comes from Opta rank → GF/GA → λ (caller),
    not independent Poissons. ``rho`` defaults to 0 on the long path
    (no AH/OU fit); Dixon–Coles low-score mass stays available if set.
    """
    sh, sa = sample_match_lams(
        lam_h,
        lam_a,
        lam_h_lo=lam_h_lo,
        lam_h_hi=lam_h_hi,
        lam_a_lo=lam_a_lo,
        lam_a_hi=lam_a_hi,
        rng=rng,
        centre_blend=centre_blend,
    )
    probs = match_outcome_probs_from_lambdas(sh, sa, float(rho))
    roll = match_outcome_roll(probs, rng)
    return int(roll.goals_home), int(roll.goals_away), float(sh), float(sa)


def init_paths_from_opta(
    ranks: Mapping[str, float] | None = None,
) -> dict[str, ClubPath]:
    opta = load_opta_ranks() if ranks is None else None
    rank_map = (
        {r["club_norm"]: float(r["rank"]) for r in opta}
        if ranks is None
        else {norm_team(k): float(v) for k, v in ranks.items()}
    )
    profile = rank_gf_ga_profile()
    paths: dict[str, ClubPath] = {}
    for club, rk in rank_map.items():
        rates = gf_ga_from_rank(rk, profile=profile, method="ols")
        paths[club] = ClubPath(
            att=float(rates["gf"]) / N_MATCHES,
            dfn=float(rates["ga"]) / N_MATCHES,
            n=0.0,
        )
    return paths


def fixtures_by_gw(
    fixtures: Sequence[Mapping[str, Any]],
    team_names: Mapping[int, str],
    *,
    start: int,
    end: int,
) -> dict[int, list[dict[str, str]]]:
    out: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in fixtures:
        gw = row.get("event")
        if gw is None:
            continue
        g = int(gw)
        if g < int(start) or g > int(end):
            continue
        if row.get("finished") and g < int(start):
            continue
        th = int(row["team_h"])
        ta = int(row["team_a"])
        out[g].append(
            {
                "home": norm_team(team_names[th]),
                "away": norm_team(team_names[ta]),
            }
        )
    return dict(out)


def path_match_pots(
    paths: Mapping[str, ClubPath],
    home: str,
    away: str,
    *,
    league_gf: float,
) -> dict[str, Any]:
    """Build pots + λ bands from the strand's current season-rate projection."""
    h, a = paths[home], paths[away]
    pot_h, pot_a = pots_from_season_gf_ga(
        h.season_gf(),
        h.season_ga(),
        a.season_gf(),
        a.season_ga(),
        league_gf=league_gf,
    )
    # Band: ±10% around path λ (path already carries Opta MAE via init; keep thin)
    return {
        "home": home,
        "away": away,
        "pot_home": pot_h,
        "pot_away": pot_a,
        "lam_home": pot_h["lam_scored"],
        "lam_away": pot_a["lam_scored"],
        "lam_home_lo": pot_h["lam_scored"] * 0.85,
        "lam_home_hi": pot_h["lam_scored"] * 1.15,
        "lam_away_lo": pot_a["lam_scored"] * 0.85,
        "lam_away_hi": pot_a["lam_scored"] * 1.15,
    }


def pot_from_sampled_lam(lam_scored: float, lam_opp: float) -> dict[str, float]:
    """Scenario λ pot for ``score_pool_on_lam_pots`` (Numagent 2.0)."""
    lam = _clip(float(lam_scored), 0.2, 3.5)
    opp = _clip(float(lam_opp), 0.05, 3.5)
    e_total = lam + opp
    return {
        "lam_scored": lam,
        "lam_assist": _clip(lam * 0.75, 0.1, 3.0),
        "e_total": float(e_total),
        "p_cs_mkt": _clip(0.90 * math.exp(-opp), 0.05, 0.55),
        "attack": lam,
        "threat": opp,
        "source": "opta_mc_scenario",
    }


def score_pool_on_pots(
    pool: pd.DataFrame,
    pots: Mapping[tuple[int, str], list[dict[str, float]]],
    *,
    gws: Sequence[int],
    gamma: float = DEFAULT_GAMMA,
    alpha_g: float | None = None,
    alpha_a: float | None = None,
) -> pd.DataFrame:
    """Discounted multi-GW ``score_xp_v2`` for MILP (club-side assembler)."""
    return score_pool_on_lam_pots(
        pool,
        pots,
        gws=gws,
        gamma=gamma,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
    )


def pots_for_gw_from_paths(
    paths: Mapping[str, ClubPath],
    fixtures: Sequence[Mapping[str, str]],
    gw: int,
    *,
    league_gf: float,
    scenario_lams: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> dict[tuple[int, str], list[dict[str, float]]]:
    """Path-mean pots, overridden by this strand's sampled λ when present."""
    pots: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    for fx in fixtures:
        key = (fx["home"], fx["away"])
        if scenario_lams is not None and key in scenario_lams:
            sh, sa = scenario_lams[key]
            pot_h = pot_from_sampled_lam(sh, sa)
            pot_a = pot_from_sampled_lam(sa, sh)
        else:
            pred = path_match_pots(paths, fx["home"], fx["away"], league_gf=league_gf)
            pot_h, pot_a = pred["pot_home"], pred["pot_away"]
        pots[(int(gw), fx["home"])].append(pot_h)
        pots[(int(gw), fx["away"])].append(pot_a)
    return dict(pots)


def sample_opta_scenario(
    home: str,
    away: str,
    *,
    paths: Mapping[str, ClubPath],
    league_gf: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Sample λ from Opta rank→GF/GA bands, build ``M``, roll scoreline.

    Falls back to path ±15% bands only if a club is missing from Opta ranks.
    """
    try:
        pred = match_prediction(home, away)
        # Light blend toward path rates after early rolls (coherence), still
        # anchored on Opta mean/bands for the fixture quality (1st vs 20th).
        path_pred = path_match_pots(paths, home, away, league_gf=league_gf)
        lam_h = 0.7 * pred["lam_home"] + 0.3 * path_pred["lam_home"]
        lam_a = 0.7 * pred["lam_away"] + 0.3 * path_pred["lam_away"]
        lo_h, hi_h = pred["lam_home_lo"], pred["lam_home_hi"]
        lo_a, hi_a = pred["lam_away_lo"], pred["lam_away_hi"]
        home_rank = pred["home_rank"]
        away_rank = pred["away_rank"]
    except KeyError:
        path_pred = path_match_pots(paths, home, away, league_gf=league_gf)
        lam_h = path_pred["lam_home"]
        lam_a = path_pred["lam_away"]
        lo_h, hi_h = path_pred["lam_home_lo"], path_pred["lam_home_hi"]
        lo_a, hi_a = path_pred["lam_away_lo"], path_pred["lam_away_hi"]
        home_rank = away_rank = None

    gh, ga, sh, sa = sample_scoreline(
        lam_h,
        lam_a,
        lam_h_lo=lo_h,
        lam_h_hi=hi_h,
        lam_a_lo=lo_a,
        lam_a_hi=hi_a,
        rng=rng,
    )
    return {
        "home": home,
        "away": away,
        "home_rank": home_rank,
        "away_rank": away_rank,
        "hg": gh,
        "ag": ga,
        "lam_h": sh,
        "lam_a": sa,
        "lam_h_lo": lo_h,
        "lam_h_hi": hi_h,
        "lam_a_lo": lo_a,
        "lam_a_hi": hi_a,
    }


def run_strand(
    *,
    strand_id: int,
    fixtures_gw: Mapping[int, list[dict[str, str]]],
    snapshot_gws: Sequence[int],
    horizon_gws: Sequence[int],
    pool: pd.DataFrame,
    paths0: Mapping[str, ClubPath],
    league_gf: float,
    rng: random.Random,
    gamma: float = DEFAULT_GAMMA,
    milp_time_limit: float = 8.0,
    objective: str = OBJECTIVE_SQUAD_SUM,
    alpha_g: float | None = None,
    alpha_a: float | None = None,
) -> StrandResult:
    """One coherent MC strand: Opta scenarios → score_xp_v2 → MILP."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective!r}")
    if alpha_g is None or alpha_a is None:
        ag, aa, _, _ = numagent_xp.load_alphas()
        alpha_g = ag if alpha_g is None else alpha_g
        alpha_a = aa if alpha_a is None else alpha_a
    paths = {
        c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()
    }
    result = StrandResult(strand_id=strand_id)
    # Sampled λ by (home, away) for each GW — used when scoring that GW's pot.
    scenario_by_gw: dict[int, dict[tuple[str, str], tuple[float, float]]] = {}
    owned_ids: set[str] | None = None
    owned_frame: pd.DataFrame | None = None
    ft_bank = 1
    prev_snapshot_gw: int | None = None

    from src.rules.fpl_2026 import advance_ft

    for gw in horizon_gws:
        week_fix = fixtures_gw.get(int(gw), [])
        scenario_by_gw[int(gw)] = {}
        for fx in week_fix:
            if fx["home"] not in paths or fx["away"] not in paths:
                continue
            sc = sample_opta_scenario(
                fx["home"],
                fx["away"],
                paths=paths,
                league_gf=league_gf,
                rng=rng,
            )
            paths[fx["home"]].observe(sc["hg"], sc["ag"])
            paths[fx["away"]].observe(sc["ag"], sc["hg"])
            scenario_by_gw[int(gw)][(fx["home"], fx["away"])] = (
                float(sc["lam_h"]),
                float(sc["lam_a"]),
            )
            result.scorelines.append(
                {
                    "gw": int(gw),
                    "home": fx["home"],
                    "away": fx["away"],
                    "home_rank": sc["home_rank"],
                    "away_rank": sc["away_rank"],
                    "hg": sc["hg"],
                    "ag": sc["ag"],
                    "lam_h": round(sc["lam_h"], 3),
                    "lam_a": round(sc["lam_a"], 3),
                }
            )

        if int(gw) not in set(int(x) for x in snapshot_gws):
            continue

        # Remaining horizon: this GW uses sampled scenario λ; later GWs use
        # path-updated Opta-style rates (mean path, not re-sampled here).
        remaining = [g for g in horizon_gws if int(g) >= int(gw)]
        pots: dict[tuple[int, str], list[dict[str, float]]] = {}
        for g2 in remaining:
            pots.update(
                pots_for_gw_from_paths(
                    paths,
                    fixtures_gw.get(int(g2), []),
                    int(g2),
                    league_gf=league_gf,
                    scenario_lams=scenario_by_gw.get(int(g2))
                    if int(g2) == int(gw)
                    else None,
                )
            )
        scored = score_pool_on_pots(
            pool,
            pots,
            gws=remaining,
            gamma=gamma,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
        )
        scored = scored.loc[np.isfinite(scored["mc_score"])]
        if len(scored) < 15 or scored["position"].nunique() < 4:
            continue

        n_transfers = 0
        hits = 0
        ft_before = int(ft_bank)
        try:
            if objective == OBJECTIVE_XI_CAPTAIN_FT:
                # Hold weeks between snapshots still roll the FT bank.
                if prev_snapshot_gw is not None:
                    for _gap in range(int(prev_snapshot_gw) + 1, int(gw)):
                        ft_bank = advance_ft(ft_bank, 0)
                    ft_before = int(ft_bank)
                if owned_ids is None:
                    picked = _pick_squad_xi_captain(
                        scored, "mc_score", time_limit=milp_time_limit
                    )
                    # Opening rebuild is free (wildcard-like); bank starts at 1.
                    n_transfers, hits = 0, 0
                    ft_bank = 1
                    obj_value = float(picked["objective"])
                else:
                    scored_ft = _merge_owned_into_scored(scored, owned_frame)
                    picked = _pick_squad_xi_captain_ft(
                        scored_ft,
                        "mc_score",
                        owned_ids=owned_ids,
                        ft=ft_before,
                        time_limit=milp_time_limit,
                    )
                    n_transfers = int(picked["n_transfers"])
                    hits = int(picked["hits"])
                    obj_value = float(picked["net_objective"])
                    ft_bank = advance_ft(ft_before, n_transfers)
                squad = picked["squad"]
                captain_id = picked["captain_id"]
                xi_ids = picked["xi_ids"]
                owned_ids = {str(x) for x in squad["player_id"].tolist()}
                owned_frame = squad.copy()
                prev_snapshot_gw = int(gw)
            elif objective == OBJECTIVE_XI_CAPTAIN:
                picked = _pick_squad_xi_captain(
                    scored, "mc_score", time_limit=milp_time_limit
                )
                squad = picked["squad"]
                captain_id = picked["captain_id"]
                xi_ids = picked["xi_ids"]
                obj_value = picked["objective"]
            else:
                squad = _pick_squad_fast(
                    scored, "mc_score", time_limit=milp_time_limit
                )
                captain_id = None
                xi_ids = []
                obj_value = float(squad["mc_score"].sum())
        except RuntimeError:
            continue
        result.squads.append(
            {
                "gw": int(gw),
                "player_ids": [str(x) for x in squad["player_id"].tolist()],
                "names": [str(x) for x in squad["name"].tolist()],
                "score_sum": float(squad["mc_score"].sum()),
                "objective": objective,
                "objective_value": float(obj_value),
                "captain_id": captain_id,
                "xi_ids": xi_ids,
                "n_transfers": int(n_transfers),
                "hits": int(hits),
                "ft_before": int(ft_before),
                "ft_after": int(ft_bank) if objective == OBJECTIVE_XI_CAPTAIN_FT else None,
            }
        )
    return result


def _candidate_frame(gw_df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df = gw_df.loc[np.isfinite(gw_df[score_col])].copy()
    df = df.drop_duplicates("player_id", keep="first").reset_index(drop=True)
    if len(df) < 15:
        raise RuntimeError(f"Need ≥15 candidates, got {len(df)}")
    return df


def _merge_owned_into_scored(
    scored: pd.DataFrame, owned_frame: pd.DataFrame | None
) -> pd.DataFrame:
    """Keep owned players as candidates with updated scores when available."""
    if owned_frame is None or owned_frame.empty:
        return scored
    out = scored.copy()
    have = set(out["player_id"].astype(str))
    extras = []
    for row in owned_frame.itertuples(index=False):
        pid = str(row.player_id)
        if pid in have:
            continue
        extras.append(
            {
                "player_id": pid,
                "name": getattr(row, "name", pid),
                "team_norm": getattr(row, "team_norm", "owned"),
                "position": getattr(row, "position", "MID"),
                "value": getattr(row, "value", 45),
                "mc_score": float(getattr(row, "mc_score", 0.0) or 0.0),
            }
        )
    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)
    return out


def _pick_squad_fast(
    gw_df: pd.DataFrame, score_col: str, *, time_limit: float = 8.0
) -> pd.DataFrame:
    """Maximise Σ score over a legal 15 (bench weighted equally)."""
    from scipy.optimize import Bounds, LinearConstraint, milp

    from src.rules.fpl_2026 import BUDGET_TENTHS as BUDGET
    from src.rules.fpl_2026 import MAX_PER_CLUB, SQUAD_QUOTA

    df = _candidate_frame(gw_df, score_col)
    n = len(df)
    scores = df[score_col].to_numpy(float)
    values = pd.to_numeric(df["value"], errors="coerce").fillna(50.0).to_numpy(float)
    positions = df["position"].astype(str).to_numpy()
    clubs = df["team_norm"].astype(str).to_numpy()
    constraints: list[LinearConstraint] = [
        LinearConstraint(values, -np.inf, float(BUDGET)),
        LinearConstraint(np.ones(n), 15.0, 15.0),
    ]
    for pos, need in SQUAD_QUOTA.items():
        constraints.append(
            LinearConstraint((positions == pos).astype(float), float(need), float(need))
        )
    for club in sorted(set(clubs.tolist())):
        constraints.append(
            LinearConstraint(
                (clubs == club).astype(float), -np.inf, float(MAX_PER_CLUB)
            )
        )
    res = milp(
        c=-scores,
        constraints=constraints,
        bounds=Bounds(0, 1),
        integrality=np.ones(n, dtype=int),
        options={"time_limit": float(time_limit)},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"Squad MILP failed: {res.message}")
    chosen = np.where(res.x > 0.5)[0]
    if len(chosen) != 15:
        raise RuntimeError(f"Expected 15 players, got {len(chosen)}")
    return df.iloc[chosen].copy()


def _pick_squad_xi_captain(
    gw_df: pd.DataFrame,
    score_col: str,
    *,
    time_limit: float = 12.0,
    owned_ids: set[str] | None = None,
    n_transfers: int | None = None,
) -> dict[str, Any]:
    """Maximise XI score + captain double under FPL squad / formation rules.

    Decision variables (length 3n): squad ``x``, XI ``y``, captain ``c`` with
    ``c ≤ y ≤ x``. Objective ``Σ s·y + Σ s·c``. Bench is selected only to fill
    quotas and budget — it does not enter the objective.

    When ``owned_ids`` and ``n_transfers`` are set, force exactly that many
    buys (players not in the owned set).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    from src.rules.fpl_2026 import (
        BUDGET_TENTHS as BUDGET,
        MAX_DEF,
        MAX_FWD,
        MAX_MID,
        MAX_PER_CLUB,
        MIN_DEF,
        MIN_FWD,
        MIN_MID,
        SQUAD_QUOTA,
    )

    df = _candidate_frame(gw_df, score_col)
    n = len(df)
    scores = df[score_col].to_numpy(float)
    values = pd.to_numeric(df["value"], errors="coerce").fillna(50.0).to_numpy(float)
    positions = df["position"].astype(str).to_numpy()
    clubs = df["team_norm"].astype(str).to_numpy()
    ids = df["player_id"].astype(str).to_numpy()
    owned = {str(x) for x in (owned_ids or set())}
    is_buy = np.array([pid not in owned for pid in ids], dtype=float)

    constraints: list[LinearConstraint] = []

    # Budget and squad size on x.
    ax_budget = np.zeros(3 * n)
    ax_budget[:n] = values
    constraints.append(LinearConstraint(ax_budget, -np.inf, float(BUDGET)))
    ax_size = np.zeros(3 * n)
    ax_size[:n] = 1.0
    constraints.append(LinearConstraint(ax_size, 15.0, 15.0))

    for pos, need in SQUAD_QUOTA.items():
        row = np.zeros(3 * n)
        row[:n] = (positions == pos).astype(float)
        constraints.append(LinearConstraint(row, float(need), float(need)))
    for club in sorted(set(clubs.tolist())):
        row = np.zeros(3 * n)
        row[:n] = (clubs == club).astype(float)
        constraints.append(LinearConstraint(row, -np.inf, float(MAX_PER_CLUB)))

    if n_transfers is not None:
        if owned_ids is None:
            raise ValueError("n_transfers requires owned_ids")
        row = np.zeros(3 * n)
        row[:n] = is_buy
        k = float(n_transfers)
        constraints.append(LinearConstraint(row, k, k))

    # XI size and formation bounds on y.
    ay_size = np.zeros(3 * n)
    ay_size[n : 2 * n] = 1.0
    constraints.append(LinearConstraint(ay_size, 11.0, 11.0))
    gkp_y = np.zeros(3 * n)
    gkp_y[n : 2 * n] = (positions == "GKP").astype(float)
    constraints.append(LinearConstraint(gkp_y, 1.0, 1.0))
    for pos, lo, hi in (
        ("DEF", MIN_DEF, MAX_DEF),
        ("MID", MIN_MID, MAX_MID),
        ("FWD", MIN_FWD, MAX_FWD),
    ):
        row = np.zeros(3 * n)
        row[n : 2 * n] = (positions == pos).astype(float)
        constraints.append(LinearConstraint(row, float(lo), float(hi)))

    # Exactly one captain; c_i ≤ y_i ≤ x_i.
    acap = np.zeros(3 * n)
    acap[2 * n :] = 1.0
    constraints.append(LinearConstraint(acap, 1.0, 1.0))
    for i in range(n):
        row_yx = np.zeros(3 * n)
        row_yx[i] = -1.0
        row_yx[n + i] = 1.0
        constraints.append(LinearConstraint(row_yx, -np.inf, 0.0))
        row_cy = np.zeros(3 * n)
        row_cy[n + i] = -1.0
        row_cy[2 * n + i] = 1.0
        constraints.append(LinearConstraint(row_cy, -np.inf, 0.0))

    c_obj = np.zeros(3 * n)
    c_obj[n : 2 * n] = -scores
    c_obj[2 * n :] = -scores

    res = milp(
        c=c_obj,
        constraints=constraints,
        bounds=Bounds(0, 1),
        integrality=np.ones(3 * n, dtype=int),
        options={"time_limit": float(time_limit)},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"XI+captain MILP failed: {res.message}")
    x = res.x[:n]
    y = res.x[n : 2 * n]
    c = res.x[2 * n :]
    chosen = np.where(x > 0.5)[0]
    xi_idx = np.where(y > 0.5)[0]
    cap_idx = np.where(c > 0.5)[0]
    if len(chosen) != 15 or len(xi_idx) != 11 or len(cap_idx) != 1:
        raise RuntimeError(
            f"Bad XI+captain solution: squad={len(chosen)} xi={len(xi_idx)} cap={len(cap_idx)}"
        )
    squad = df.iloc[chosen].copy()
    obj = float(scores[xi_idx].sum() + scores[cap_idx[0]])
    n_buys = int(sum(1 for i in chosen if is_buy[i] > 0.5)) if owned else 0
    return {
        "squad": squad,
        "xi_ids": [str(ids[i]) for i in xi_idx],
        "captain_id": str(ids[cap_idx[0]]),
        "objective": obj,
        "n_transfers": n_buys,
    }


def _pick_squad_xi_captain_ft(
    gw_df: pd.DataFrame,
    score_col: str,
    *,
    owned_ids: set[str],
    ft: int,
    max_hits: int = MAX_HITS_SNAPSHOT,
    time_limit: float = 8.0,
) -> dict[str, Any]:
    """Choose transfers maximising XI+captain − 4·hits given the FT bank."""
    from src.rules.fpl_2026 import hit_cost

    owned = {str(x) for x in owned_ids}
    frame = gw_df
    best: dict[str, Any] | None = None
    best_v = -1e18
    k_max = int(ft) + int(max_hits)
    per_k = max(2.0, float(time_limit) / float(k_max + 1))
    for k in range(0, k_max + 1):
        try:
            picked = _pick_squad_xi_captain(
                frame,
                score_col,
                time_limit=per_k,
                owned_ids=owned,
                n_transfers=k,
            )
        except RuntimeError:
            continue
        hits = int(hit_cost(int(ft), k))
        value = float(picked["objective"]) - float(hits)
        if value > best_v:
            best_v = value
            best = {
                **picked,
                "n_transfers": k,
                "hits": hits,
                "ft_before": int(ft),
                "net_objective": value,
            }
    if best is None:
        raise RuntimeError("FT path MILP found no feasible transfer plan")
    return best


def aggregate_frequencies(
    strands: Sequence[StrandResult],
    pool: pd.DataFrame,
) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    cap_counts: Counter[str] = Counter()
    xi_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    keep_denom: Counter[str] = Counter()
    n_squads = 0
    n_with_cap = 0
    transfer_rows = 0
    transfer_sum = 0
    hits_sum = 0
    for s in strands:
        prev_ids: set[str] | None = None
        for squad in s.squads:
            n_squads += 1
            ids = [str(x) for x in squad["player_ids"]]
            counts.update(ids)
            xi = squad.get("xi_ids") or []
            if xi:
                xi_counts.update(str(x) for x in xi)
            cap = squad.get("captain_id")
            if cap:
                n_with_cap += 1
                cap_counts[str(cap)] += 1
            if squad.get("n_transfers") is not None and prev_ids is not None:
                transfer_rows += 1
                transfer_sum += int(squad.get("n_transfers") or 0)
                hits_sum += int(squad.get("hits") or 0)
            if prev_ids is not None:
                for pid in ids:
                    keep_denom[pid] += 1
                    if pid in prev_ids:
                        kept_counts[pid] += 1
            prev_ids = set(ids)
    meta = pool.set_index("player_id")
    rows = []
    for pid, n in counts.most_common():
        if pid not in meta.index:
            continue
        r = meta.loc[pid]
        denom = keep_denom.get(pid, 0)
        rows.append(
            {
                "player_id": pid,
                "name": r["name"],
                "team_norm": r["team_norm"],
                "position": r["position"],
                "value": int(r["value"]),
                "n_selected": int(n),
                "freq": float(n) / float(n_squads) if n_squads else 0.0,
                "n_xi": int(xi_counts.get(pid, 0)),
                "freq_xi": float(xi_counts.get(pid, 0)) / float(n_squads) if n_squads else 0.0,
                "n_captain": int(cap_counts.get(pid, 0)),
                "freq_captain": (
                    float(cap_counts.get(pid, 0)) / float(n_with_cap) if n_with_cap else 0.0
                ),
                "stick_rate": (
                    float(kept_counts.get(pid, 0)) / float(denom) if denom else None
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_squads"] = n_squads
    out.attrs["n_strands"] = len(strands)
    out.attrs["n_with_captain"] = n_with_cap
    out.attrs["mean_transfers"] = (
        float(transfer_sum) / float(transfer_rows) if transfer_rows else None
    )
    out.attrs["mean_hits"] = (
        float(hits_sum) / float(transfer_rows) if transfer_rows else None
    )
    return out


def run_multistage_trial(
    *,
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    decision_gw: int,
    horizon_end: int,
    n_strands: int = 20,
    seed: int = 7,
    gamma: float = DEFAULT_GAMMA,
    pool_path: Path | None = None,
    logs_path: Path | None = None,
    start_offset: int = 2,
    snapshot_every: int = 1,
    objective: str = OBJECTIVE_SQUAD_SUM,
    minutes_overlay: bool = True,
) -> dict[str, Any]:
    """Run N coherent strands and return frequency table + metadata."""
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective: {objective!r}")
    team_names = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    selectable = load_player_pool_ids(pool_path or DEFAULT_POOL_JSON)
    logs = pd.read_csv(logs_path or LOG_CSV)
    pool = build_frozen_pool(
        bootstrap,
        selectable,
        logs,
        before_gw=int(decision_gw),
        season="2026-27",
    )
    overlay_meta: dict[str, Any] = {"overlay": False, "n": int(len(pool))}
    if minutes_overlay:
        pool = apply_availability_overlay(pool, bootstrap, prefer_next=True)
        overlay_meta = overlay_summary(pool)
    if len(pool) < 15:
        raise RuntimeError("selectable pool too small for MILP")
    n_with_share = int((pd.to_numeric(pool["rG"], errors="coerce").fillna(0) > 0).sum())
    alpha_g, alpha_a, _, _ = numagent_xp.load_alphas()

    start = int(decision_gw) + int(start_offset)
    end = int(horizon_end)
    by_gw = fixtures_by_gw(fixtures, team_names, start=start, end=end)
    horizon_gws = sorted(g for g in by_gw if start <= g <= end)
    if not horizon_gws:
        raise RuntimeError("no fixtures in horizon window")
    snapshot_gws = [
        g for i, g in enumerate(horizon_gws) if i % max(1, int(snapshot_every)) == 0
    ]

    profile = rank_gf_ga_profile()
    league_gf = league_avg_gf(profile)
    paths0 = init_paths_from_opta()
    rng = random.Random(int(seed))

    strands: list[StrandResult] = []
    for i in range(int(n_strands)):
        strand_rng = random.Random(rng.randint(0, 2**31 - 1))
        strands.append(
            run_strand(
                strand_id=i,
                fixtures_gw=by_gw,
                snapshot_gws=snapshot_gws,
                horizon_gws=horizon_gws,
                pool=pool,
                paths0=paths0,
                league_gf=league_gf,
                rng=strand_rng,
                gamma=gamma,
                objective=objective,
                alpha_g=alpha_g,
                alpha_a=alpha_a,
            )
        )

    freq = aggregate_frequencies(strands, pool)
    n_squads = int(freq.attrs.get("n_squads") or 0)
    return {
        "decision_gw": int(decision_gw),
        "objective": objective,
        "scorer": SCORE_XP_V2,
        "assembler": "src/markets/numagent_xp.py",
        "horizon": {"start": start, "end": end, "gws": horizon_gws},
        "snapshot_gws": snapshot_gws,
        "n_strands": int(n_strands),
        "n_squads_solved": n_squads,
        "n_selectable": int(len(pool)),
        "n_with_nonzero_rate": n_with_share,
        "n_with_nonzero_share": n_with_share,  # back-compat key
        "seed_shares": SEED_SHARES,
        "minutes_overlay": bool(minutes_overlay),
        "overlay": overlay_meta,
        "alpha_g": float(alpha_g),
        "alpha_a": float(alpha_a),
        "scoring": (
            "Opta match_prediction λ bands → Dixon–Coles M → "
            f"numagent_xp.score_team_side ({SCORE_XP_V2}); "
            "same-season as-of rates/priors/rolling m frozen at decision GW"
            + ("; FPL availability overlay on m" if minutes_overlay else "")
        ),
        "seed": int(seed),
        "gamma": float(gamma),
        "frequencies": freq.to_dict("records"),
        "top15_mean_freq": float(freq.head(15)["freq"].mean()) if len(freq) >= 15 else None,
        "mean_transfers": freq.attrs.get("mean_transfers"),
        "mean_hits": freq.attrs.get("mean_hits"),
        "strands_with_squads": sum(1 for s in strands if s.squads),
        "sample_strand_scorelines": strands[0].scorelines[:12] if strands else [],
    }


def write_trial_outputs(result: dict[str, Any], *, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    obj = str(result.get("objective") or OBJECTIVE_SQUAD_SUM)
    if obj == OBJECTIVE_XI_CAPTAIN_FT:
        stem = "multistage_xi_captain_ft"
    elif obj == OBJECTIVE_XI_CAPTAIN:
        stem = "multistage_xi_captain"
    else:
        stem = "multistage_mc_milp"
    if result.get("minutes_overlay"):
        stem = f"{stem}_overlay"
    json_path = out_dir / f"{stem}_trial.json"
    csv_path = out_dir / f"{stem}_player_freq.csv"
    report = ROOT / "reports" / f"{stem}_trial.md"

    json_path.write_text(json.dumps(result, indent=2) + "\n")
    freq = result.get("frequencies") or []
    if freq:
        pd.DataFrame(freq).to_csv(csv_path, index=False)
    else:
        csv_path.write_text("player_id,name,freq\n")

    lines = [
        "# Multistage MC + MILP — long-horizon selection frequency",
        "",
        f"Trial (diagnostic). Objective: **`{obj}`**. Not a gated likelihood promotion. Chips off.",
        "",
        "## Design",
        "",
        f"- Decision GW **{result['decision_gw']}**; roll fixtures "
        f"GW{result['horizon']['start']}–{result['horizon']['end']}.",
        f"- **{result['n_strands']}** coherent strands: each fixture samples λ "
        "from **Opta rank→GF/GA bands** (`match_prediction`, e.g. 2nd vs 20th), "
        "builds Dixon–Coles `M`, rolls `k ∼ M`, path rate update.",
        f"- Player xP: **`{result.get('scorer', SCORE_XP_V2)}`** via "
        "`numagent_xp.score_team_side` on Dixon–Coles moments from scenario λ; "
        f"same-season as-of rates frozen at decision GW. Share seeding: "
        f"**{result.get('seed_shares')}**.",
        f"- Minutes overlay: **{result.get('minutes_overlay')}** "
        f"({result.get('overlay')}).",
        f"- Snapshots at GW {result['snapshot_gws']} → selectable-pool MILP "
        f"(`{obj}`).",
        f"- Solved squads: **{result['n_squads_solved']}** · selectable "
        f"**{result['n_selectable']}** ({result.get('n_with_nonzero_rate', result.get('n_with_nonzero_share'))} with "
        f"nonzero rG) · seed {result['seed']}.",
        "",
        f"Scoring: `{result.get('scoring')}`",
        "",
        "## Player stability (selection frequency)",
        "",
        "| rank | id | name | team | pos | £ | freq | xi | cap | n |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(freq[:40], start=1):
        lines.append(
            f"| {i} | {row['player_id']} | {row['name']} | {row['team_norm']} | "
            f"{row['position']} | {row['value']/10:.1f} | {row['freq']:.2f} | "
            f"{row.get('freq_xi', 0):.2f} | {row.get('freq_captain', 0):.2f} | "
            f"{row['n_selected']} |"
        )
    if len(freq) > 40:
        lines.append(f"| … | ({len(freq) - 40} more) | | | | | | | | |")

    lines += [
        "",
        "## Read",
        "",
        "High ``freq`` across strands × snapshots = appears in MILP squads "
        "under many coherent future paths → candidate for long-horizon hold. "
        "Under ``xi_plus_captain``, ``freq_captain`` is the share of solved "
        "squads where that player is the named captain.",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20 --objective {obj}",
        "```",
        "",
    ]
    report.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path), "csv": str(csv_path), "report": str(report)}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision-gw", type=int, default=6)
    p.add_argument("--horizon-end", type=int, default=19)
    p.add_argument("--n", type=int, default=12, help="Number of MC strands")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--snapshot-every", type=int, default=2)
    p.add_argument(
        "--bootstrap",
        type=Path,
        default=ROOT
        / "data/predictions/2026-27/gw06/betfair_20261008/bootstrap.json",
    )
    p.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT
        / "data/predictions/2026-27/gw06/betfair_20261008/fixtures.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/predictions/2026-27/multistage_mc_milp",
    )
    p.add_argument("--logs", type=Path, default=LOG_CSV)
    p.add_argument(
        "--objective",
        choices=list(OBJECTIVES),
        default=OBJECTIVE_SQUAD_SUM,
        help="squad_sum = equal 15; xi_plus_captain = XI + captain double",
    )
    p.add_argument(
        "--minutes-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale xmi/shares by FPL chance_of_playing / status (default on)",
    )
    args = p.parse_args()
    bootstrap = json.loads(args.bootstrap.read_text())
    fixtures = json.loads(args.fixtures.read_text())
    result = run_multistage_trial(
        bootstrap=bootstrap,
        fixtures=fixtures,
        decision_gw=args.decision_gw,
        horizon_end=args.horizon_end,
        n_strands=args.n,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
        logs_path=args.logs,
        objective=args.objective,
        minutes_overlay=bool(args.minutes_overlay),
    )
    paths = write_trial_outputs(result, out_dir=args.out)
    print(
        json.dumps(
            {
                "objective": result.get("objective"),
                "n_strands": result["n_strands"],
                "n_squads_solved": result["n_squads_solved"],
                "strands_with_squads": result["strands_with_squads"],
                "top5": [
                    {
                        "name": r["name"],
                        "freq": round(r["freq"], 3),
                        "freq_captain": round(r.get("freq_captain") or 0.0, 3),
                    }
                    for r in (result["frequencies"] or [])[:5]
                ],
                **paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
