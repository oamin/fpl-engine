"""Probe: does xi_ft swap Saka → Groß by GW10 on mean short+long boards?

1. Roll N strands (priced GW8 + Opta long), score full pool generative each GW.
2. Average → per-GW μ boards.
3. Build a legal opening 15 that owns Saka.
4. Joint path plan ``value_captain=False``, ``ft_pe_nu=0`` (bank pot separate).
5. Report ownership / transfers for Saka and Groß.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets import numagent_xp
from src.markets.cascade_score_v2 import build_frozen_pool, normalize_live_history
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import (
    DEFAULT_LINEUP_COV,
    LOG_CSV,
    ClubPath,
    _pick_squad_fast,
    _priced_index,
    _undiscounted_scores_by_gw,
    fixtures_by_gw,
    init_paths_from_opta,
    load_priced_week_fits,
    sample_week_scenario,
)
from src.models.opta_gf_ga_horizon import LIVE_E0, assessed_league_gf
from src.models.path_transfer_plan import (
    pick_path_transfer_plan,
    score_matrix_for_candidates,
    select_plan_candidates,
)
from src.models.player_pool import POOL_JSON, load_player_pool_ids
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[1]


def _find_player(pool: pd.DataFrame, needle: str) -> pd.Series:
    hit = pool[pool["name"].astype(str).str.contains(needle, case=False, na=False)]
    if hit.empty:
        raise SystemExit(f"no pool player matching {needle!r}")
    return hit.iloc[0]


def mean_boards(
    *,
    n_strands: int,
    seed0: int,
    decision_gw: int,
    horizon_end: int,
    priced_gw: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[str, Any]]:
    bootstrap = json.loads((ROOT / "data/live/bootstrap.json").read_text())
    fixtures = json.loads((ROOT / "data/live/fixtures.json").read_text())
    team_names = {int(t["id"]): str(t["name"]) for t in bootstrap["teams"]}
    selectable = load_player_pool_ids(POOL_JSON)
    logs = pd.read_csv(LOG_CSV)
    hist = numagent_xp.asof_share_history(
        normalize_live_history(logs), before_gw=int(decision_gw), season="2026-27"
    )
    team_hist = hist_by_team_norm(hist)
    pool = build_frozen_pool(
        bootstrap, selectable, logs, before_gw=int(decision_gw), season="2026-27"
    )
    by_gw = fixtures_by_gw(
        fixtures, team_names, start=int(decision_gw), end=int(horizon_end)
    )
    plan_gws = sorted(g for g in by_gw if int(decision_gw) <= g <= int(horizon_end))

    odds_path = ROOT / "data/scratch/gw08_odds_full.json"
    priced_by_gw: dict[int, list] = {}
    if odds_path.is_file() and int(priced_gw) in set(plan_gws):
        fits = load_priced_week_fits(odds_path, bootstrap, gw=int(priced_gw))
        if fits:
            priced_by_gw[int(priced_gw)] = fits
    priced_index = _priced_index(priced_by_gw)
    league_state = assessed_league_gf(LIVE_E0)
    league_gf = float(league_state["league_gf"])
    sd_league_gf = float(league_state["se_league_gf"])
    alpha_g, alpha_a, _, _ = numagent_xp.load_alphas()

    acc = {g: np.zeros(len(pool), dtype=float) for g in plan_gws}
    for s in range(int(n_strands)):
        rng = random.Random(int(seed0) + s)
        paths: dict[str, ClubPath] = {
            c: ClubPath(att=p.att, dfn=p.dfn, n=p.n)
            for c, p in init_paths_from_opta().items()
        }
        rolled: dict[int, list[dict[str, Any]]] = {}
        for gw in plan_gws:
            week = []
            for fx in by_gw.get(int(gw), []):
                if fx["home"] not in paths or fx["away"] not in paths:
                    continue
                sc = sample_week_scenario(
                    fx["home"],
                    fx["away"],
                    gw=int(gw),
                    priced_index=priced_index,
                    paths=paths,
                    league_gf=league_gf,
                    rng=rng,
                    sd_league_gf=sd_league_gf,
                )
                paths[fx["home"]].observe(sc["hg"], sc["ag"])
                paths[fx["away"]].observe(sc["ag"], sc["hg"])
                week.append(sc)
            rolled[int(gw)] = week
        scores = _undiscounted_scores_by_gw(
            pool,
            plan_gws=plan_gws,
            rolled_by_gw=rolled,
            fixtures_gw=by_gw,
            paths=paths,
            league_gf=league_gf,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            rng=random.Random(int(seed0) + 10_000 + s),
            hist_by_team=team_hist,
            generative_attack=True,
            lineup_cov=DEFAULT_LINEUP_COV,
        )
        for g in plan_gws:
            acc[int(g)] += scores[int(g)]
    mean = {g: acc[g] / float(n_strands) for g in plan_gws}
    meta = {
        "n_strands": int(n_strands),
        "seed0": int(seed0),
        "plan_gws": plan_gws,
        "league_gf": league_gf,
        "n_priced_fits": int(sum(len(v) for v in priced_by_gw.values())),
    }
    return pool, mean, meta


def opening_15_with(
    pool: pd.DataFrame,
    scores_by_gw: dict[int, np.ndarray],
    gws: list[int],
    force_id: str,
) -> list[str]:
    """Legal 15 maximizing mean horizon score, forcing ``force_id`` in."""
    mean = np.mean([scores_by_gw[g] for g in gws], axis=0)
    tmp = pool.copy()
    tmp["mc_score"] = mean
    # Soft-force: huge boost so the squad MILP keeps them.
    idx = tmp.index[tmp["player_id"].astype(str) == str(force_id)]
    if len(idx) != 1:
        raise RuntimeError(f"force id {force_id} not unique in pool")
    tmp.loc[idx, "mc_score"] = float(tmp["mc_score"].max()) + 100.0
    squad = _pick_squad_fast(tmp, "mc_score", time_limit=20.0)
    ids = [str(x) for x in squad["player_id"].tolist()]
    if str(force_id) not in ids:
        raise RuntimeError("failed to force player into opening 15")
    return ids


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-strands", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/opt/cursor/artifacts/saka_gross_swap_probe.json"),
    )
    args = ap.parse_args()
    n_strands = int(args.n_strands)
    seed0 = int(args.seed0)
    decision_gw, horizon_end, priced_gw = 8, 14, 8
    pool, scores_by_gw, meta = mean_boards(
        n_strands=n_strands,
        seed0=seed0,
        decision_gw=decision_gw,
        horizon_end=horizon_end,
        priced_gw=priced_gw,
    )
    gws = list(meta["plan_gws"])
    saka = _find_player(pool, "Saka")
    gross = _find_player(pool, "Groß")
    saka_id, gross_id = str(saka["player_id"]), str(gross["player_id"])

    # Board snapshot for the two players (matches the earlier plot idea).
    pid_index = {str(p): i for i, p in enumerate(pool["player_id"].astype(str))}
    board_rows = []
    for g in gws:
        board_rows.append(
            {
                "gw": g,
                "saka": float(scores_by_gw[g][pid_index[saka_id]]),
                "gross": float(scores_by_gw[g][pid_index[gross_id]]),
                "gross_minus_saka": float(
                    scores_by_gw[g][pid_index[gross_id]]
                    - scores_by_gw[g][pid_index[saka_id]]
                ),
            }
        )
    board = pd.DataFrame(board_rows)

    owned = opening_15_with(pool, scores_by_gw, gws, saka_id)
    cands = select_plan_candidates(
        pool,
        scores_by_gw,
        gws=gws,
        buy_ids=None,
        force_ids=set(owned) | {gross_id},
        top_n=48,
    )
    # Ensure both assets are choosable.
    for need in (saka_id, gross_id):
        if need not in set(cands["player_id"].astype(str)):
            raise RuntimeError(f"{need} missing from plan candidates")
    S = score_matrix_for_candidates(cands, pool, scores_by_gw, gws)
    plan = pick_path_transfer_plan(
        cands,
        S,
        gws,
        gamma=1.0,
        ft_pe_nu=0.0,
        value_captain=False,
        time_limit=120.0,
        start_owned_ids=owned,
        start_ft=1,
    )

    weeks = plan["weeks"]
    print(f"=== mean board ({n_strands} strands) Saka vs Groß ===")
    print(board.to_string(index=False))
    print()
    print(
        f"Opening 15 forced Saka={saka_id}; Groß id={gross_id}; "
        f"candidates={plan['n_candidates']}; value_captain={plan['value_captain']}; "
        f"bank_pot.in_objective={plan['bank_pot']['in_objective']}"
    )
    print()
    print("=== joint xi_ft plan (from owned, ft=1) ===")
    for w in weeks:
        ids = set(w["player_ids"])
        xi = set(w["xi_ids"])
        print(
            f"GW{w['gw']}: Saka_owned={saka_id in ids} Saka_XI={saka_id in xi} | "
            f"Groß_owned={gross_id in ids} Groß_XI={gross_id in xi} | "
            f"transfers={w['n_transfers']} hits={w['hits']} "
            f"ft {w['ft_before']}→{w['ft_after']} | "
            f"XI={w['xi_score']:.2f}"
        )
        if w["n_transfers"]:
            # names of buys: in current not in previous
            pass

    # Transfer narrative vs previous week
    print()
    print("=== transfers (buys into the 15) ===")
    id_to_name = {
        str(r.player_id): str(r.name) for r in pool.itertuples()
    }
    prev = set(owned)
    for w in weeks:
        cur = set(w["player_ids"])
        buys = sorted(cur - prev)
        sells = sorted(prev - cur)
        if buys or sells:
            bnames = [id_to_name.get(x, x) for x in buys]
            snames = [id_to_name.get(x, x) for x in sells]
            print(f"GW{w['gw']}: buy {bnames} / sell {snames}")
        prev = cur

    saka_own = [saka_id in set(w["player_ids"]) for w in weeks]
    gross_own = [gross_id in set(w["player_ids"]) for w in weeks]
    gw10_i = gws.index(10) if 10 in gws else None
    swapped = (
        gw10_i is not None
        and (not saka_own[gw10_i])
        and gross_own[gw10_i]
    )
    held_gross_after = all(gross_own[i] for i in range(gw10_i, len(gws))) if gw10_i is not None else False
    print()
    print(
        json.dumps(
            {
                "swap_saka_to_gross_by_gw10": swapped,
                "gross_held_from_gw10_on": held_gross_after,
                "saka_owned_by_gw": dict(zip(gws, saka_own)),
                "gross_owned_by_gw": dict(zip(gws, gross_own)),
                "terminal_ft": plan["terminal_ft"],
                "objective": plan["objective"],
            },
            indent=2,
        )
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "board": board.to_dict("records"),
                "opening_15": owned,
                "weeks": weeks,
                "bank_pot": plan["bank_pot"],
                "meta": meta,
                "verdict": {
                    "swap_saka_to_gross_by_gw10": swapped,
                    "gross_held_from_gw10_on": held_gross_after,
                    "total_transfers": int(
                        sum(int(w["n_transfers"]) for w in weeks)
                    ),
                    "total_hits": int(sum(int(w["hits"]) for w in weeks)),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
