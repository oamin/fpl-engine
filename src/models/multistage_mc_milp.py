"""Long-horizon multistage MC strands + MILP selection frequency.

Trial design (not a gated likelihood promotion)
----------------------------------------------
From decision gameweek ``t``, for each of ``N`` Monte Carlo strands:

1. At every future GW ``g ∈ {t+2, …, t+n}``, for each fixture sample a
   scoreline from the Opta-derived λ bounds (Poisson, clipped to the
   predicted lam band). Sampled goals update that strand's club GF/GA path
   so later weeks stay coherent with earlier rolls.
2. At each snapshot GW on the strand, score the season-long selectable
   pool with ``xp_on_pot`` on the path-implied pots and solve the squad
   MILP (``pick_squad``).
3. Aggregate how often each player is selected across strands × snapshots
   → stability / long-horizon investment signal.

The imminent week ``t`` / ``t+1`` stays on the priced path (not rolled here).
Chips are not modelled. 2026/27 holdout is not fit; Opta rank→GF/GA bands
come from closed-season OLS already in ``opta_gf_ga_horizon``.
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

from src.models.forecast_xp import xp_on_pot
from src.models.opta_gf_ga_horizon import (
    N_MATCHES,
    gf_ga_from_rank,
    league_avg_gf,
    load_opta_ranks,
    match_prediction,
    pots_from_season_gf_ga,
    rank_gf_ga_profile,
)
from src.models.selectable_pool import DEFAULT_POOL_JSON, load_selectable_ids
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
POS_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DEFAULT_GAMMA = 0.9


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


def _poisson(rng: random.Random, lam: float) -> int:
    # Knuth for small λ; normal approx avoided — PL λ are small.
    lam = max(0.05, float(lam))
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def sample_scoreline(
    lam_h: float,
    lam_a: float,
    *,
    lam_h_lo: float,
    lam_h_hi: float,
    lam_a_lo: float,
    lam_a_hi: float,
    rng: random.Random,
) -> tuple[int, int, float, float]:
    """Sample λ inside predicted bounds, then Poisson goals."""
    lo_h, hi_h = min(lam_h_lo, lam_h_hi), max(lam_h_lo, lam_h_hi)
    lo_a, hi_a = min(lam_a_lo, lam_a_hi), max(lam_a_lo, lam_a_hi)
    # Shrink band toward centre slightly so samples stay "within predicted bounds"
    # without always hitting corners.
    sh = lo_h + (hi_h - lo_h) * rng.random()
    sa = lo_a + (hi_a - lo_a) * rng.random()
    # Blend with mean λ so the strand stays near the Opta centre.
    sh = 0.5 * float(lam_h) + 0.5 * sh
    sa = 0.5 * float(lam_a) + 0.5 * sa
    gh = _poisson(rng, sh)
    ga = _poisson(rng, sa)
    # Soft clip: reject extreme outliers beyond ceil(hi)+2 once
    if gh > math.ceil(hi_h) + 2:
        gh = int(round(sh))
    if ga > math.ceil(hi_a) + 2:
        ga = int(round(sa))
    return gh, ga, sh, sa


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


def bootstrap_candidate_frame(
    bootstrap: Mapping[str, Any],
    selectable_ids: set[str],
) -> pd.DataFrame:
    """Selectable players with share proxies from bootstrap season xG."""
    teams = {int(t["id"]): norm_team(t["name"]) for t in bootstrap["teams"]}
    elements = [el for el in bootstrap["elements"] if str(el["id"]) in selectable_ids]
    # Team xG totals for share
    team_xg: dict[str, float] = defaultdict(float)
    team_xa: dict[str, float] = defaultdict(float)
    for el in elements:
        club = teams[int(el["team"])]
        team_xg[club] += float(el.get("expected_goals") or 0.0)
        team_xa[club] += float(el.get("expected_assists") or 0.0)

    rows = []
    for el in elements:
        club = teams[int(el["team"])]
        xg = float(el.get("expected_goals") or 0.0)
        xa = float(el.get("expected_assists") or 0.0)
        share_xg = xg / team_xg[club] if team_xg[club] > 1e-9 else 0.0
        share_xa = xa / team_xa[club] if team_xa[club] > 1e-9 else 0.0
        starts = max(int(el.get("starts") or 0), 1)
        minutes = float(el.get("minutes") or 0.0)
        xmi = minutes / starts if starts else 0.0
        rows.append(
            {
                "player_id": str(el["id"]),
                "name": el.get("web_name"),
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "team_norm": club,
                "value": int(el.get("now_cost") or 50),
                "xmi": float(min(90.0, max(0.0, xmi))),
                "share_xG": float(min(0.55, max(0.0, share_xg))),
                "share_xA": float(min(0.55, max(0.0, share_xa))),
                "exp_defcon_hit": 0.15 if int(el["element_type"]) in {2, 3} else 0.05,
                "fwd_goal_scale": 1.0,
                "eligible": True,
            }
        )
    return pd.DataFrame(rows)


def score_pool_on_pots(
    pool: pd.DataFrame,
    pots: Mapping[tuple[int, str], list[dict[str, float]]],
    *,
    gws: Sequence[int],
    gamma: float = DEFAULT_GAMMA,
) -> pd.DataFrame:
    """Discounted multi-GW xp_on_pot score for MILP."""
    out = pool.copy()
    scores = []
    for row in out.itertuples(index=False):
        total = 0.0
        club = str(row.team_norm)
        for h, gw in enumerate(gws):
            quotes = pots.get((int(gw), club)) or []
            if not quotes:
                continue
            pot = quotes[0]
            xp = xp_on_pot(
                position=str(row.position),
                xmi=float(row.xmi),
                share_xg=float(row.share_xG),
                share_xa=float(row.share_xA),
                exp_defcon_hit=float(row.exp_defcon_hit),
                fwd_goal_scale=float(row.fwd_goal_scale),
                pot=pot,
            )
            total += (float(gamma) ** h) * xp
        scores.append(total)
    out["mc_score"] = scores
    return out


def pots_for_gw_from_paths(
    paths: Mapping[str, ClubPath],
    fixtures: Sequence[Mapping[str, str]],
    gw: int,
    *,
    league_gf: float,
) -> dict[tuple[int, str], list[dict[str, float]]]:
    pots: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    for fx in fixtures:
        pred = path_match_pots(paths, fx["home"], fx["away"], league_gf=league_gf)
        pots[(int(gw), fx["home"])].append(pred["pot_home"])
        pots[(int(gw), fx["away"])].append(pred["pot_away"])
    return dict(pots)


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
) -> StrandResult:
    """One coherent MC strand: sample scores, update paths, MILP at snapshots."""
    paths = {
        c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()
    }
    result = StrandResult(strand_id=strand_id)
    for gw in horizon_gws:
        week_fix = fixtures_gw.get(int(gw), [])
        for fx in week_fix:
            if fx["home"] not in paths or fx["away"] not in paths:
                continue
            # Prefer Opta band corners when available; else path ±15%
            try:
                pred = match_prediction(fx["home"], fx["away"])
                lam_h, lam_a = pred["lam_home"], pred["lam_away"]
                # After updates, blend Opta centre with path centre
                path_pred = path_match_pots(
                    paths, fx["home"], fx["away"], league_gf=league_gf
                )
                lam_h = 0.5 * lam_h + 0.5 * path_pred["lam_home"]
                lam_a = 0.5 * lam_a + 0.5 * path_pred["lam_away"]
                lo_h, hi_h = pred["lam_home_lo"], pred["lam_home_hi"]
                lo_a, hi_a = pred["lam_away_lo"], pred["lam_away_hi"]
            except KeyError:
                path_pred = path_match_pots(
                    paths, fx["home"], fx["away"], league_gf=league_gf
                )
                lam_h = path_pred["lam_home"]
                lam_a = path_pred["lam_away"]
                lo_h, hi_h = path_pred["lam_home_lo"], path_pred["lam_home_hi"]
                lo_a, hi_a = path_pred["lam_away_lo"], path_pred["lam_away_hi"]

            gh, ga, sh, sa = sample_scoreline(
                lam_h,
                lam_a,
                lam_h_lo=lo_h,
                lam_h_hi=hi_h,
                lam_a_lo=lo_a,
                lam_a_hi=hi_a,
                rng=rng,
            )
            paths[fx["home"]].observe(gh, ga)
            paths[fx["away"]].observe(ga, gh)
            result.scorelines.append(
                {
                    "gw": int(gw),
                    "home": fx["home"],
                    "away": fx["away"],
                    "hg": gh,
                    "ag": ga,
                    "lam_h": round(sh, 3),
                    "lam_a": round(sa, 3),
                }
            )

        if int(gw) not in set(int(x) for x in snapshot_gws):
            continue

        # Pots for remaining horizon from current path state
        remaining = [g for g in horizon_gws if int(g) >= int(gw)]
        pots: dict[tuple[int, str], list[dict[str, float]]] = {}
        for g2 in remaining:
            pots.update(
                pots_for_gw_from_paths(
                    paths, fixtures_gw.get(int(g2), []), int(g2), league_gf=league_gf
                )
            )
        scored = score_pool_on_pots(pool, pots, gws=remaining, gamma=gamma)
        scored = scored.loc[np.isfinite(scored["mc_score"])]
        if len(scored) < 15 or scored["position"].nunique() < 4:
            continue
        try:
            # Temporarily lower milp time via monkeypatch options — pick_squad
            # hardcodes 20s; call scipy here for tighter limit on MC volume.
            squad = _pick_squad_fast(scored, "mc_score", time_limit=milp_time_limit)
        except RuntimeError:
            continue
        result.squads.append(
            {
                "gw": int(gw),
                "player_ids": [str(x) for x in squad["player_id"].tolist()],
                "names": [str(x) for x in squad["name"].tolist()],
                "score_sum": float(squad["mc_score"].sum()),
            }
        )
    return result


def _pick_squad_fast(
    gw_df: pd.DataFrame, score_col: str, *, time_limit: float = 8.0
) -> pd.DataFrame:
    from scipy.optimize import Bounds, LinearConstraint, milp

    from src.rules.fpl_2026 import BUDGET_TENTHS as BUDGET
    from src.rules.fpl_2026 import MAX_PER_CLUB, SQUAD_QUOTA

    df = gw_df.loc[np.isfinite(gw_df[score_col])].copy()
    df = df.drop_duplicates("player_id", keep="first").reset_index(drop=True)
    if len(df) < 15:
        raise RuntimeError(f"Need ≥15 candidates, got {len(df)}")
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


def aggregate_frequencies(
    strands: Sequence[StrandResult],
    pool: pd.DataFrame,
) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    n_squads = 0
    for s in strands:
        for squad in s.squads:
            n_squads += 1
            counts.update(squad["player_ids"])
    meta = pool.set_index("player_id")
    rows = []
    for pid, n in counts.most_common():
        if pid not in meta.index:
            continue
        r = meta.loc[pid]
        rows.append(
            {
                "player_id": pid,
                "name": r["name"],
                "team_norm": r["team_norm"],
                "position": r["position"],
                "value": int(r["value"]),
                "n_selected": int(n),
                "freq": float(n) / float(n_squads) if n_squads else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["n_squads"] = n_squads
    out.attrs["n_strands"] = len(strands)
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
    start_offset: int = 2,
    snapshot_every: int = 1,
) -> dict[str, Any]:
    """Run N coherent strands and return frequency table + metadata."""
    team_names = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    selectable = load_selectable_ids(pool_path or DEFAULT_POOL_JSON)
    pool = bootstrap_candidate_frame(bootstrap, selectable)
    if len(pool) < 15:
        raise RuntimeError("selectable pool too small for MILP")

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
            )
        )

    freq = aggregate_frequencies(strands, pool)
    n_squads = int(freq.attrs.get("n_squads") or 0)
    return {
        "decision_gw": int(decision_gw),
        "horizon": {"start": start, "end": end, "gws": horizon_gws},
        "snapshot_gws": snapshot_gws,
        "n_strands": int(n_strands),
        "n_squads_solved": n_squads,
        "n_selectable": int(len(pool)),
        "seed": int(seed),
        "gamma": float(gamma),
        "frequencies": freq.to_dict("records"),
        "top15_mean_freq": float(freq.head(15)["freq"].mean()) if len(freq) >= 15 else None,
        "strands_with_squads": sum(1 for s in strands if s.squads),
        "sample_strand_scorelines": strands[0].scorelines[:12] if strands else [],
    }


def write_trial_outputs(result: dict[str, Any], *, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "multistage_mc_milp_trial.json"
    csv_path = out_dir / "multistage_player_freq.csv"
    report = ROOT / "reports" / "multistage_mc_milp_trial.md"

    json_path.write_text(json.dumps(result, indent=2) + "\n")
    freq = result.get("frequencies") or []
    if freq:
        pd.DataFrame(freq).to_csv(csv_path, index=False)
    else:
        csv_path.write_text("player_id,name,freq\n")

    lines = [
        "# Multistage MC + MILP — long-horizon selection frequency",
        "",
        "Trial (diagnostic). Not a gated likelihood promotion. Chips off.",
        "",
        "## Design",
        "",
        f"- Decision GW **{result['decision_gw']}**; roll fixtures "
        f"GW{result['horizon']['start']}–{result['horizon']['end']}.",
        f"- **{result['n_strands']}** coherent Monte Carlo strands: each GW "
        "samples scorelines inside Opta λ bounds, updates club GF/GA path.",
        f"- Snapshots at GW {result['snapshot_gws']}: score season selectable "
        "pool (`xp_on_pot` on path pots) → squad MILP.",
        f"- Solved squads: **{result['n_squads_solved']}** · selectable "
        f"**{result['n_selectable']}** · seed {result['seed']}.",
        "",
        "## Player stability (selection frequency)",
        "",
        "| rank | id | name | team | pos | £ | freq | n |",
        "|---:|---:|---|---|---|---:|---:|---:|",
    ]
    for i, row in enumerate(freq[:40], start=1):
        lines.append(
            f"| {i} | {row['player_id']} | {row['name']} | {row['team_norm']} | "
            f"{row['position']} | {row['value']/10:.1f} | {row['freq']:.2f} | "
            f"{row['n_selected']} |"
        )
    if len(freq) > 40:
        lines.append(f"| … | ({len(freq) - 40} more) | | | | | | |")

    lines += [
        "",
        "## Read",
        "",
        "High ``freq`` across strands × snapshots = appears in MILP squads "
        "under many coherent future paths → candidate for long-horizon hold. "
        "Low freq = path-sensitive / avoid locking capital.",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.models.multistage_mc_milp --n 20",
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
    )
    paths = write_trial_outputs(result, out_dir=args.out)
    print(
        json.dumps(
            {
                "n_strands": result["n_strands"],
                "n_squads_solved": result["n_squads_solved"],
                "strands_with_squads": result["strands_with_squads"],
                "top5": [
                    {"name": r["name"], "freq": round(r["freq"], 3)}
                    for r in (result["frequencies"] or [])[:5]
                ],
                **paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
