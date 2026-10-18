"""Monte Carlo: Shin opening 1X2 → scoreline → player goal scorers (2025/26).

Each fixture:
  1. Opening odds → Shin de-vigged (p_home, p_draw, p_away) and OU → λ_h, λ_a
     (same pot map as ``forecast_xp.side_pot``).
  2. Draw W/D/A from the Shin multinomial, then (hg, ag) ~ independent
     Poisson(λ_h)×Poisson(λ_a) conditioned on that outcome (rejection).
  3. Assign each goal to a squad player with probability proportional to
     max(prior xG, position floor), times a soft minutes rate.

Player prior at GW1: 2024/25 per-appearance xG mean (and variance for an
optional Negative-Binomial / Gamma-Poisson draw of the rate). After each
GW the prior expands with that week's observed xG.

Residuals:
  match:  actual (hg−ag) − E_sim[hg−ag]
  player: actual goals − E_sim[goals]

Writes under ``reports/`` and ``data/processed/``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets.shin import shin_1x2, shin_pair
from src.models.forecast_xp import NEUTRAL_TOTAL
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEASON = "2025_26"
PRIOR_SEASON = "2024_25"
N_SIMS_DEFAULT = 1000
RNG_SEED = 20251018

# Baseline xG floors so every squad player has weight > 0.
# Ordering: FWD > MID > DEF > GK.
POS_FLOOR = {"FWD": 0.22, "MID": 0.08, "DEF": 0.025, "GKP": 0.008, "GK": 0.008}
POS_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


@dataclass
class PlayerState:
    """Expanding xG history for one player (prior season prepended)."""

    xg: list[float] = field(default_factory=list)
    minutes: list[float] = field(default_factory=list)

    def mean_xg(self) -> float:
        return float(np.mean(self.xg)) if self.xg else 0.0

    def var_xg(self) -> float:
        if len(self.xg) < 2:
            return float(self.mean_xg())  # Poisson-ish default
        return float(np.var(self.xg, ddof=1))

    def mean_minutes(self) -> float:
        return float(np.mean(self.minutes)) if self.minutes else 0.0

    def append(self, xg: float, minutes: float) -> None:
        self.xg.append(float(xg))
        self.minutes.append(float(minutes))


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def lambdas_from_shin(
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float,
    p_under: float,
) -> tuple[float, float, float]:
    """Home/away goal λ and expected total from Shin probs (forecast_xp map)."""
    e_total = NEUTRAL_TOTAL + 1.10 * (p_over - p_under)
    home_att = p_home + 0.5 * p_draw
    away_att = p_away + 0.5 * p_draw
    denom = home_att + away_att
    if denom <= 0:
        lam_h = lam_a = e_total * 0.5
    else:
        lam_h = e_total * home_att / denom
        lam_a = e_total * away_att / denom
    return _clip(lam_h, 0.2, 3.5), _clip(lam_a, 0.2, 3.5), float(e_total)


def sample_scorelines_batch(
    rng: np.random.Generator,
    p_home: float,
    p_draw: float,
    p_away: float,
    lam_h: float,
    lam_a: float,
    n: int,
    *,
    pool: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shin-conditioned Poisson scorelines via rejection from a shared pool.

    Returns hg, ag, outcome_code (0=H, 1=D, 2=A).
    """
    outcomes = rng.choice(3, size=n, p=[p_home, p_draw, p_away])
    hg_out = np.empty(n, dtype=np.int16)
    ag_out = np.empty(n, dtype=np.int16)

    # Build candidate pools per outcome class.
    pools: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {0: [], 1: [], 2: []}
    need = {0: int((outcomes == 0).sum()), 1: int((outcomes == 1).sum()), 2: int((outcomes == 2).sum())}
    have = {0: 0, 1: 0, 2: 0}
    guard = 0
    while any(have[k] < need[k] for k in (0, 1, 2)) and guard < 40:
        guard += 1
        ch = rng.poisson(lam_h, size=pool).astype(np.int16)
        ca = rng.poisson(lam_a, size=pool).astype(np.int16)
        masks = {
            0: ch > ca,
            1: ch == ca,
            2: ca > ch,
        }
        for k, m in masks.items():
            if have[k] >= need[k]:
                continue
            take = int(m.sum())
            if take == 0:
                continue
            pools[k].append((ch[m], ca[m]))
            have[k] += take

    for k in (0, 1, 2):
        idx = np.flatnonzero(outcomes == k)
        if idx.size == 0:
            continue
        if have[k] == 0:
            hg_out[idx] = rng.poisson(lam_h, size=idx.size).astype(np.int16)
            ag_out[idx] = rng.poisson(lam_a, size=idx.size).astype(np.int16)
            continue
        hs = np.concatenate([p[0] for p in pools[k]])
        as_ = np.concatenate([p[1] for p in pools[k]])
        pick = rng.integers(0, len(hs), size=idx.size)
        hg_out[idx] = hs[pick]
        ag_out[idx] = as_[pick]
    return hg_out, ag_out, outcomes


def nb_rate(rng: np.random.Generator, mean: float, var: float) -> float:
    """Draw a Gamma rate with E=mean, Var≈var (NB mixture). Floor at tiny ε."""
    mean = max(mean, 1e-6)
    var = max(var, mean)  # at least Poisson
    # Gamma shape/scale: mean = kθ, var = kθ² → θ = var/mean, k = mean/θ
    theta = var / mean
    k = mean / theta
    if k <= 0 or not math.isfinite(k):
        return mean
    return float(max(rng.gamma(k, theta), 1e-8))


def scorer_weights(
    players: list[dict[str, Any]],
    states: dict[str, PlayerState],
    rng: np.random.Generator,
    *,
    use_nb: bool = True,
) -> np.ndarray:
    """Positive weights: max(μ_xG, pos floor) × minutes prior, optional NB draw.

    Minutes prior is the expanding mean minutes (ε=1 for unknowns) so unused
    squad players do not soak ~30% of the team λ. Position floors keep
    FWD > MID > DEF > GKP when xG history is empty.
    """
    w = np.zeros(len(players), dtype=float)
    for i, p in enumerate(players):
        st = states.get(p["player_id"])
        mu = st.mean_xg() if st is not None else 0.0
        var = st.var_xg() if st is not None else mu
        floor = POS_FLOOR.get(p["position"], 0.01)
        base = max(mu, floor)
        if use_nb and st is not None and len(st.xg) >= 2:
            base = max(nb_rate(rng, base, max(var, floor)), floor)
        mins = st.mean_minutes() if st is not None else 0.0
        # ε=1 for never-seen; regulars (e.g. 70 mins) dominate unused (ε).
        soft = max(mins, 1.0)
        w[i] = base * soft
    s = float(w.sum())
    if s <= 0:
        w[:] = 1.0 / len(players)
    else:
        w /= s
    return w


def allocate_goals(
    rng: np.random.Generator,
    n_goals: int,
    weights: np.ndarray,
) -> np.ndarray:
    """Multinomial goal counts for one team in one sim."""
    if n_goals <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=int)
    return rng.multinomial(n_goals, weights)


def load_player_gw(season: str) -> pd.DataFrame:
    raw = pd.read_csv(CACHE / f"merged_gw_{season}.csv")
    df = pd.DataFrame(
        {
            "player_id": raw["element"].astype(str),
            "name": raw["name"].astype(str),
            "team": raw["team"].astype(str),
            "team_norm": raw["team"].map(norm_team),
            "position": raw["position"].map(
                lambda p: POS_MAP.get(str(p).upper(), str(p).upper())
            ),
            "gw": pd.to_numeric(raw.get("GW", raw.get("round")), errors="coerce"),
            "fixture": raw["fixture"].astype(str),
            "date": raw["kickoff_time"].astype(str).str.slice(0, 10),
            "kickoff": raw["kickoff_time"].astype(str),
            "is_home": raw["was_home"]
            .map(lambda x: str(x).lower() in {"true", "1", "yes"})
            .astype(int),
            "minutes": pd.to_numeric(raw["minutes"], errors="coerce").fillna(0.0),
            "goals": pd.to_numeric(raw["goals_scored"], errors="coerce").fillna(0.0),
            "xg": pd.to_numeric(raw["expected_goals"], errors="coerce").fillna(0.0),
            "hg": pd.to_numeric(raw["team_h_score"], errors="coerce"),
            "ag": pd.to_numeric(raw["team_a_score"], errors="coerce"),
        }
    )
    df = df.dropna(subset=["gw", "hg", "ag"]).copy()
    df["gw"] = df["gw"].astype(int)
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    return df


def build_prior_states(prior: pd.DataFrame) -> dict[str, PlayerState]:
    """Seed states from prior-season appearances (minutes > 0 preferred)."""
    states: dict[str, PlayerState] = {}
    # Map by name for cross-season id continuity; also keep prior ids.
    by_name: dict[str, PlayerState] = {}
    sub = prior.sort_values(["name", "gw"], kind="mergesort")
    for _, r in sub.iterrows():
        # Use all GW rows; 0-minute weeks still update minutes rate.
        key = str(r["name"]).strip().lower()
        st = by_name.get(key)
        if st is None:
            st = PlayerState()
            by_name[key] = st
        st.append(float(r["xg"]), float(r["minutes"]))
    # Index by 2025/26 id filled later; keep name map separate.
    for key, st in by_name.items():
        states[f"name:{key}"] = st
    return states


def load_odds() -> pd.DataFrame:
    """Prefer processed Shin fixtures; rebuild from E0 openings if needed."""
    path = PROCESSED / "fixtures_odds.csv"
    if path.exists():
        odds = pd.read_csv(path)
        odds["home_norm"] = odds["home"].map(norm_team)
        odds["away_norm"] = odds["away"].map(norm_team)
        return odds

    # Rebuild from football-data opening columns.
    from src.ingest.fpl_odds import opening_prices

    rows = []
    raw = pd.read_csv(CACHE / "E0_2526.csv")
    for _, row in raw.iterrows():
        d = {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}
        prices = opening_prices(d)
        if prices is None:
            continue
        p_h, p_d, p_a = shin_1x2(
            float(prices["home_odds"]),
            float(prices["draw_odds"]),
            float(prices["away_odds"]),
        )
        p_o, p_u = shin_pair(
            float(prices["over25_odds"]), float(prices["under25_odds"])
        )
        date = pd.to_datetime(row["Date"], dayfirst=True).strftime("%Y-%m-%d")
        rows.append(
            {
                "date": date,
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "home_norm": norm_team(row["HomeTeam"]),
                "away_norm": norm_team(row["AwayTeam"]),
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "p_over25": p_o,
                "p_under25": p_u,
                "home_goals": float(row["FTHG"]),
                "away_goals": float(row["FTAG"]),
            }
        )
    return pd.DataFrame(rows)


def join_fixtures(players: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture with Shin probs and λ."""
    sides = (
        players.groupby("fixture", as_index=False)
        .agg(
            gw=("gw", "first"),
            date=("date", "first"),
            kickoff=("kickoff", "first"),
            hg=("hg", "first"),
            ag=("ag", "first"),
        )
        .copy()
    )
    home = (
        players.loc[players["is_home"] == 1, ["fixture", "team", "team_norm"]]
        .drop_duplicates("fixture")
        .rename(columns={"team": "home", "team_norm": "home_norm"})
    )
    away = (
        players.loc[players["is_home"] == 0, ["fixture", "team", "team_norm"]]
        .drop_duplicates("fixture")
        .rename(columns={"team": "away", "team_norm": "away_norm"})
    )
    sides = sides.merge(home, on="fixture").merge(away, on="fixture")
    odds_key = odds.copy()
    odds_key["okey"] = (
        odds_key["date"].astype(str)
        + ":"
        + odds_key["home_norm"]
        + ":"
        + odds_key["away_norm"]
    )
    sides["okey"] = (
        sides["date"].astype(str)
        + ":"
        + sides["home_norm"]
        + ":"
        + sides["away_norm"]
    )
    cols = [
        "okey",
        "p_home",
        "p_draw",
        "p_away",
        "p_over25",
        "p_under25",
    ]
    m = sides.merge(odds_key[cols], on="okey", how="inner")
    lams = [
        lambdas_from_shin(
            float(r.p_home),
            float(r.p_draw),
            float(r.p_away),
            float(r.p_over25),
            float(r.p_under25),
        )
        for r in m.itertuples()
    ]
    m["lam_h"] = [t[0] for t in lams]
    m["lam_a"] = [t[1] for t in lams]
    m["e_total"] = [t[2] for t in lams]
    m["gd_act"] = m["hg"] - m["ag"]
    return m.sort_values(["kickoff", "fixture"], kind="mergesort").reset_index(drop=True)


def run_season(
    *,
    n_sims: int = N_SIMS_DEFAULT,
    seed: int = RNG_SEED,
    use_nb: bool = True,
) -> dict[str, Any]:
    players = load_player_gw(SEASON)
    prior = load_player_gw(PRIOR_SEASON)
    odds = load_odds()
    fixtures = join_fixtures(players, odds)

    name_states = build_prior_states(prior)
    # Bind 2025/26 player_ids to name-seeded states (copy so shared names OK).
    states: dict[str, PlayerState] = {}
    name_of: dict[str, str] = {}
    for _, r in players[["player_id", "name"]].drop_duplicates().iterrows():
        pid = str(r["player_id"])
        key = f"name:{str(r['name']).strip().lower()}"
        name_of[pid] = key
        src = name_states.get(key)
        if src is None:
            states[pid] = PlayerState()
        else:
            states[pid] = PlayerState(xg=list(src.xg), minutes=list(src.minutes))

    rng = np.random.default_rng(seed)
    match_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []

    # Process GW order; within GW by kickoff (already sorted).
    by_fixture_players = {
        fid: g.to_dict("records")
        for fid, g in players.groupby("fixture", sort=False)
    }

    for fr in fixtures.itertuples(index=False):
        fid = str(fr.fixture)
        plist = by_fixture_players[fid]
        # Scorer rolls among players who appeared (minutes > 0). Scoreline λ
        # stays pre-match from Shin; this isolates the share model given the XI.
        home_p = [
            p
            for p in plist
            if int(p["is_home"]) == 1 and float(p["minutes"]) > 0
        ]
        away_p = [
            p
            for p in plist
            if int(p["is_home"]) == 0 and float(p["minutes"]) > 0
        ]
        if not home_p or not away_p:
            continue

        # Scoreline MC (Shin-conditioned). Player E[goals] = E[team goals] × share.
        # Shares use expanding xG mean (position floor); NB averages μ over jitter.
        hg_s, ag_s, out_s = sample_scorelines_batch(
            rng,
            float(fr.p_home),
            float(fr.p_draw),
            float(fr.p_away),
            float(fr.lam_h),
            float(fr.lam_a),
            n_sims,
        )
        e_hg = float(hg_s.mean())
        e_ag = float(ag_s.mean())
        e_gd = e_hg - e_ag
        gd_resid = float(fr.gd_act) - e_gd
        sim_p = (
            float((out_s == 0).mean()),
            float((out_s == 1).mean()),
            float((out_s == 2).mean()),
        )

        if use_nb:
            wh_acc = np.zeros(len(home_p), dtype=float)
            wa_acc = np.zeros(len(away_p), dtype=float)
            n_w = min(64, n_sims)
            for _ in range(n_w):
                wh_acc += scorer_weights(home_p, states, rng, use_nb=True)
                wa_acc += scorer_weights(away_p, states, rng, use_nb=True)
            wh = wh_acc / n_w
            wa = wa_acc / n_w
        else:
            wh = scorer_weights(home_p, states, rng, use_nb=False)
            wa = scorer_weights(away_p, states, rng, use_nb=False)
        # E[player goals] = E[team goals] × share; multinomial averages to this.
        home_exp = wh * e_hg
        away_exp = wa * e_ag

        match_rows.append(
            {
                "gw": int(fr.gw),
                "fixture": fid,
                "date": fr.date,
                "home": fr.home,
                "away": fr.away,
                "hg": int(fr.hg),
                "ag": int(fr.ag),
                "gd_act": int(fr.gd_act),
                "e_hg": round(e_hg, 4),
                "e_ag": round(e_ag, 4),
                "e_gd": round(e_gd, 4),
                "gd_resid": round(gd_resid, 4),
                "hg_resid": round(int(fr.hg) - e_hg, 4),
                "ag_resid": round(int(fr.ag) - e_ag, 4),
                "lam_h": round(float(fr.lam_h), 4),
                "lam_a": round(float(fr.lam_a), 4),
                "p_home": round(float(fr.p_home), 4),
                "p_draw": round(float(fr.p_draw), 4),
                "p_away": round(float(fr.p_away), 4),
                "sim_p_home": round(sim_p[0], 4),
                "sim_p_draw": round(sim_p[1], 4),
                "sim_p_away": round(sim_p[2], 4),
            }
        )

        for i, p in enumerate(home_p):
            player_rows.append(
                {
                    "gw": int(fr.gw),
                    "fixture": fid,
                    "player_id": p["player_id"],
                    "name": p["name"],
                    "team": p["team"],
                    "position": p["position"],
                    "is_home": 1,
                    "minutes": float(p["minutes"]),
                    "goals_act": float(p["goals"]),
                    "xg_act": float(p["xg"]),
                    "goals_exp": round(float(home_exp[i]), 5),
                    "goal_resid": round(float(p["goals"]) - float(home_exp[i]), 5),
                    "prior_xg_mean": round(states[p["player_id"]].mean_xg(), 5),
                    "prior_xg_var": round(states[p["player_id"]].var_xg(), 5),
                }
            )
        for i, p in enumerate(away_p):
            player_rows.append(
                {
                    "gw": int(fr.gw),
                    "fixture": fid,
                    "player_id": p["player_id"],
                    "name": p["name"],
                    "team": p["team"],
                    "position": p["position"],
                    "is_home": 0,
                    "minutes": float(p["minutes"]),
                    "goals_act": float(p["goals"]),
                    "xg_act": float(p["xg"]),
                    "goals_exp": round(float(away_exp[i]), 5),
                    "goal_resid": round(float(p["goals"]) - float(away_exp[i]), 5),
                    "prior_xg_mean": round(states[p["player_id"]].mean_xg(), 5),
                    "prior_xg_var": round(states[p["player_id"]].var_xg(), 5),
                }
            )

        # Expand priors with this GW's observed xG / minutes (as-of next GW).
        for p in plist:
            states[p["player_id"]].append(float(p["xg"]), float(p["minutes"]))

    match_df = pd.DataFrame(match_rows)
    player_df = pd.DataFrame(player_rows)
    summary = summarise(match_df, player_df, n_sims=n_sims, use_nb=use_nb)
    return {
        "match": match_df,
        "player": player_df,
        "summary": summary,
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return None
    if float(np.std(x[mask])) == 0 or float(np.std(y[mask])) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def summarise(
    match: pd.DataFrame, player: pd.DataFrame, *, n_sims: int, use_nb: bool
) -> dict[str, Any]:
    gd_act = match["gd_act"].to_numpy(float)
    e_gd = match["e_gd"].to_numpy(float)
    gd_resid = match["gd_resid"].to_numpy(float)

    # Players who appeared (fairer residual pool).
    played = player.loc[player["minutes"] > 0].copy()
    g_act = played["goals_act"].to_numpy(float)
    g_exp = played["goals_exp"].to_numpy(float)
    g_resid = played["goal_resid"].to_numpy(float)

    pos_rows = []
    for pos, g in played.groupby("position"):
        pos_rows.append(
            {
                "position": pos,
                "n": int(len(g)),
                "mean_resid": round(float(g["goal_resid"].mean()), 4),
                "mae_resid": round(float(np.mean(np.abs(g["goal_resid"]))), 4),
                "corr_exp_act": (
                    None
                    if (c := _corr(g["goals_exp"].to_numpy(), g["goals_act"].to_numpy()))
                    is None
                    else round(c, 4)
                ),
            }
        )

    gw_rows = []
    for gw, g in match.groupby("gw"):
        c = _corr(g["e_gd"].to_numpy(), g["gd_act"].to_numpy())
        gw_rows.append(
            {
                "gw": int(gw),
                "n": int(len(g)),
                "mean_gd_resid": round(float(g["gd_resid"].mean()), 4),
                "mae_gd_resid": round(float(np.mean(np.abs(g["gd_resid"]))), 4),
                "corr_egd_act": None if c is None else round(c, 4),
            }
        )

    return {
        "season": "2025/26",
        "n_sims": n_sims,
        "use_nb_rates": use_nb,
        "n_fixtures": int(len(match)),
        "n_player_rows": int(len(player)),
        "n_player_played": int(len(played)),
        "match": {
            "mean_gd_resid": round(float(gd_resid.mean()), 4),
            "mae_gd_resid": round(float(np.mean(np.abs(gd_resid))), 4),
            "rmse_gd_resid": round(float(np.sqrt(np.mean(gd_resid**2))), 4),
            "corr_egd_act": (
                None if (c := _corr(e_gd, gd_act)) is None else round(c, 4)
            ),
            "mean_hg_resid": round(float(match["hg_resid"].mean()), 4),
            "mean_ag_resid": round(float(match["ag_resid"].mean()), 4),
            "mae_hg_resid": round(float(np.mean(np.abs(match["hg_resid"]))), 4),
            "mae_ag_resid": round(float(np.mean(np.abs(match["ag_resid"]))), 4),
        },
        "player_played": {
            "mean_goal_resid": round(float(g_resid.mean()), 4),
            "mae_goal_resid": round(float(np.mean(np.abs(g_resid))), 4),
            "rmse_goal_resid": round(float(np.sqrt(np.mean(g_resid**2))), 4),
            "corr_exp_act": (
                None if (c := _corr(g_exp, g_act)) is None else round(c, 4)
            ),
            "sum_goals_act": round(float(g_act.sum()), 1),
            "sum_goals_exp": round(float(g_exp.sum()), 1),
        },
        "by_position": pos_rows,
        "by_gw": gw_rows,
    }


def write_outputs(result: dict[str, Any]) -> dict[str, str]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    match_path = PROCESSED / "mc_shin_match_residuals_2025_26.csv"
    player_path = PROCESSED / "mc_shin_player_residuals_2025_26.csv"
    summary_path = FIGURES / "mc_shin_player_goals_2025_26.json"
    report_path = REPORTS / "mc_shin_player_goals_2025_26.md"

    result["match"].to_csv(match_path, index=False)
    result["player"].to_csv(player_path, index=False)
    summary_path.write_text(json.dumps(result["summary"], indent=2) + "\n")

    s = result["summary"]
    lines = [
        "# Monte Carlo Shin scoreline + player goal allocation (2025/26)",
        "",
        "## Method",
        "",
        "1. **Opening odds** per fixture from football-data (`AvgH` → `B365H` → `PSH`, "
        "plus OU 2.5), Shin-devigged to `(p_home, p_draw, p_away)` and `(p_over, p_under)` "
        "(same map as `fixtures_odds.csv` / `forecast_xp.side_pot`).",
        "2. **Team λ:** `e_total = 2.45 + 1.10(p_over − p_under)`; "
        "`λ_side = e_total · attack_strength / Σ attack` with "
        "`attack = p_win + 0.5 p_draw`.",
        "3. **Scoreline MC:** draw W/D/A from the Shin multinomial; sample "
        "`(hg, ag) ~ Poisson(λ_h)×Poisson(λ_a)` conditioned on that outcome "
        f"({s['n_sims']} sims / fixture).",
        "4. **Player prior:** at GW1 seed each player with 2024/25 per-GW xG mean "
        "and variance (name-matched). After each GW the prior expands with that "
        "week’s observed xG. Empty history gets a position floor "
        "`FWD 0.22 > MID 0.08 > DEF 0.025 > GKP 0.008`. Optional Gamma–Poisson "
        "(NB) jitter of the rate uses the expanding variance.",
        "5. **Scorer roll:** among players with minutes > 0, "
        "`E[goals_i] = E[team goals] × w_i` with "
        "`w_i ∝ max(μ_xG, floor) × max(μ_minutes, 1)`.",
        "6. **Residuals:** match `GD_act − E_sim[GD]`; player `goals_act − E_sim[goals]`.",
        "",
        f"- Fixtures: **{s['n_fixtures']}**",
        f"- Sims per fixture: **{s['n_sims']}**",
        f"- NB/Gamma rate draws: **{s['use_nb_rates']}**",
        "",
        "## Match scoreline residuals (home − away)",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mean GD residual (act − E) | {s['match']['mean_gd_resid']} |",
        f"| MAE GD residual | {s['match']['mae_gd_resid']} |",
        f"| RMSE GD residual | {s['match']['rmse_gd_resid']} |",
        f"| corr(E[GD], actual GD) | {s['match']['corr_egd_act']} |",
        f"| MAE home-goals residual | {s['match']['mae_hg_resid']} |",
        f"| MAE away-goals residual | {s['match']['mae_ag_resid']} |",
        "",
        "## Player goal residuals (appeared)",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| n player–matches | {s['n_player_played']} |",
        f"| mean residual | {s['player_played']['mean_goal_resid']} |",
        f"| MAE residual | {s['player_played']['mae_goal_resid']} |",
        f"| RMSE residual | {s['player_played']['rmse_goal_resid']} |",
        f"| corr(E[goals], actual) | {s['player_played']['corr_exp_act']} |",
        f"| Σ goals act / exp | {s['player_played']['sum_goals_act']} / {s['player_played']['sum_goals_exp']} |",
        "",
        "## By position",
        "",
        "| pos | n | mean resid | MAE | corr(E, act) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in s["by_position"]:
        lines.append(
            f"| {r['position']} | {r['n']} | {r['mean_resid']} | {r['mae_resid']} | {r['corr_exp_act']} |"
        )
    lines += [
        "",
        "## Artefacts",
        "",
        f"- `{match_path.relative_to(ROOT)}`",
        f"- `{player_path.relative_to(ROOT)}`",
        f"- `{summary_path.relative_to(ROOT)}`",
        f"- `{FIGURES.relative_to(ROOT)}/mc_shin_player_goals_2025_26.png`",
        "",
        "Reproduce: `python3 -m src.models.mc_shin_player_goals`",
        "",
    ]
    report_path.write_text("\n".join(lines))

    # Compact residual plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        m = result["match"]
        axes[0].scatter(m["e_gd"], m["gd_act"], alpha=0.35, s=12, c="#1f4e79")
        lim = max(3, float(np.nanmax(np.abs(m["gd_act"]))))
        axes[0].plot([-lim, lim], [-lim, lim], "k--", lw=1)
        axes[0].set_xlabel("E_sim[home − away]")
        axes[0].set_ylabel("Actual home − away")
        axes[0].set_title("Match GD")
        played = result["player"].loc[result["player"]["minutes"] > 0]
        axes[1].scatter(
            played["goals_exp"],
            played["goals_act"],
            alpha=0.15,
            s=8,
            c="#8b2942",
        )
        axes[1].set_xlabel("E_sim[goals]")
        axes[1].set_ylabel("Actual goals")
        axes[1].set_title("Player goals (mins>0)")
        fig.tight_layout()
        plot_path = FIGURES / "mc_shin_player_goals_2025_26.png"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    except Exception:
        plot_path = FIGURES / "mc_shin_player_goals_2025_26.png"

    return {
        "match": str(match_path),
        "player": str(player_path),
        "summary": str(summary_path),
        "report": str(report_path),
        "plot": str(plot_path),
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sims", type=int, default=N_SIMS_DEFAULT)
    p.add_argument("--seed", type=int, default=RNG_SEED)
    p.add_argument("--no-nb", action="store_true", help="Use mean xG weights only")
    args = p.parse_args()
    result = run_season(n_sims=args.sims, seed=args.seed, use_nb=not args.no_nb)
    paths = write_outputs(result)
    print(json.dumps(result["summary"]["match"], indent=2))
    print(json.dumps(result["summary"]["player_played"], indent=2))
    print("wrote", paths)


if __name__ == "__main__":
    main()
