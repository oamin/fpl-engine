"""Trial: Opta place → GF/GA → λ → k ∼ M → generative player rolls.

Default fixture: Man City (Opta 1st) v Fulham (Opta 20th). Props off
(cascade_long). Same-season rates from ``player_gw_2026_27``.

  PYTHONPATH=. python3 -m src.markets.opta_player_roll_trial [--n 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markets import numagent_xp as nx
from src.markets.attack_roll_trial import _side_frame
from src.markets.short_horizon_generative import score_fixture_generative
from src.models.opta_gf_ga_horizon import (
    load_opta_ranks,
    match_prediction,
    sample_match_from_opta_ranks,
)
from src.markets.numagent_xp_pilot import norm_team

ROOT = Path(__file__).resolve().parents[2]
PLAYER_GW = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
DEFAULT_OUT = ROOT / "reports" / "opta_player_roll_mancity_fulham.json"


def _hist_for_team(history: pd.DataFrame, team: str, before_gw: int) -> pd.DataFrame:
    return history[
        (history["team_norm"] == norm_team(team)) & (history["gw"] < before_gw)
    ].copy()


def run_trial(
    *,
    home: str = "Man City",
    away: str = "Fulham",
    n_paths: int = 2000,
    seed: int = 0,
    before_gw: int = 6,
    lineup_cov: bool = True,
    centre_blend: float = 0.5,
) -> dict[str, Any]:
    ranks = {r["club_norm"]: float(r["rank"]) for r in load_opta_ranks()}
    home_n, away_n = norm_team(home), norm_team(away)
    if home_n not in ranks or away_n not in ranks:
        raise KeyError(f"missing Opta rank for {[home_n, away_n]}")

    centre = match_prediction(home, away, ranks=ranks)
    history = pd.read_csv(PLAYER_GW)
    history["team_norm"] = history["team"].map(norm_team)
    home_board = _side_frame(history, team=home, before_gw=before_gw)
    away_board = _side_frame(history, team=away, before_gw=before_gw)
    home_hist = _hist_for_team(history, home, before_gw)
    away_hist = _hist_for_team(history, away, before_gw)

    ag, aa, _, _ = nx.load_alphas()
    rng = random.Random(seed)

    score_hist: Counter[str] = Counter()
    scorer_hist: Counter[str] = Counter()
    assist_hist: Counter[str] = Counter()
    xp_sum: dict[str, float] = Counter()
    xp_n: dict[str, int] = Counter()
    cs_home = cs_away = 0
    lam_h_sum = lam_a_sum = 0.0
    paths_sample: list[dict[str, Any]] = []

    for i in range(int(n_paths)):
        draw = sample_match_from_opta_ranks(
            home, away, rng, ranks=ranks, centre_blend=centre_blend
        )
        lam_h = float(draw["lam_home"])
        lam_a = float(draw["lam_away"])
        lam_h_sum += lam_h
        lam_a_sum += lam_a
        home_xp, away_xp, meta = score_fixture_generative(
            home_board,
            away_board,
            lam_h=lam_h,
            lam_a=lam_a,
            rho=0.0,
            alpha_g=float(ag),
            alpha_a=float(aa),
            rng=rng,
            home_hist=home_hist,
            away_hist=away_hist,
            home_props=None,
            away_props=None,
            lineup_cov=lineup_cov,
        )
        label = f"{meta['goals_home']}-{meta['goals_away']}"
        score_hist[label] += 1
        if meta["home_cs"]:
            cs_home += 1
        if meta["away_cs"]:
            cs_away += 1

        for side_label, board in ((home, home_xp), (away, away_xp)):
            if board is None or board.empty:
                continue
            for _, row in board.iterrows():
                name = f"{side_label}:{row['player_name']}"
                g = float(row.get("E_G", 0.0) or 0.0)
                a = float(row.get("E_A", 0.0) or 0.0)
                if g > 0:
                    scorer_hist[name] += int(round(g))
                if a > 0:
                    assist_hist[name] += int(round(a))
                xp_sum[name] += float(row["xp"])
                xp_n[name] += 1

        if i < 8:
            paths_sample.append(
                {
                    "scoreline": label,
                    "lam_h": lam_h,
                    "lam_a": lam_a,
                    "home_season_gf": draw["home_season_draw"]["gf"],
                    "away_season_gf": draw["away_season_draw"]["gf"],
                    "top_home_xp": (
                        home_xp.nlargest(3, "xp")[["player_name", "xp", "E_G", "E_A"]]
                        .to_dict("records")
                        if home_xp is not None and not home_xp.empty
                        else []
                    ),
                    "top_away_xp": (
                        away_xp.nlargest(3, "xp")[["player_name", "xp", "E_G", "E_A"]]
                        .to_dict("records")
                        if away_xp is not None and not away_xp.empty
                        else []
                    ),
                }
            )

    n = float(n_paths)
    mean_xp = sorted(
        (
            (name, xp_sum[name] / max(xp_n[name], 1))
            for name in xp_sum
        ),
        key=lambda x: -x[1],
    )
    return {
        "n": int(n_paths),
        "seed": int(seed),
        "props": False,
        "lineup_cov": bool(lineup_cov),
        "centre_blend": float(centre_blend),
        "before_gw": int(before_gw),
        "fixture": {
            "home": home,
            "away": away,
            "home_rank": ranks[home_n],
            "away_rank": ranks[away_n],
            "source": "opta_place_gf_ga",
            "lam_home_centre": float(centre["lam_home"]),
            "lam_away_centre": float(centre["lam_away"]),
            "lam_home_lo": float(centre["lam_home_lo"]),
            "lam_home_hi": float(centre["lam_home_hi"]),
            "lam_away_lo": float(centre["lam_away_lo"]),
            "lam_away_hi": float(centre["lam_away_hi"]),
            "home_season_gf_centre": float(centre["home_season"]["gf"]),
            "home_season_ga_centre": float(centre["home_season"]["ga"]),
            "away_season_gf_centre": float(centre["away_season"]["gf"]),
            "away_season_ga_centre": float(centre["away_season"]["ga"]),
            "mean_sampled_lam_h": lam_h_sum / n,
            "mean_sampled_lam_a": lam_a_sum / n,
            "alpha_g": float(ag),
            "alpha_a": float(aa),
            "n_home_players": int(len(home_board)),
            "n_away_players": int(len(away_board)),
        },
        "mean_scoreline": [
            sum(int(k.split("-")[0]) * v for k, v in score_hist.items()) / n,
            sum(int(k.split("-")[1]) * v for k, v in score_hist.items()) / n,
        ],
        "p_cs_home": cs_home / n,
        "p_cs_away": cs_away / n,
        "scoreline_freq": dict(score_hist.most_common(40)),
        "top_scorers": scorer_hist.most_common(20),
        "top_assisters": assist_hist.most_common(20),
        "top_mean_xp": mean_xp[:25],
        "sample_paths": paths_sample,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--home", default="Man City")
    p.add_argument("--away", default="Fulham")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--before-gw", type=int, default=6)
    p.add_argument("--no-lineup-cov", action="store_true")
    p.add_argument("--centre-blend", type=float, default=0.5)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    out = run_trial(
        home=args.home,
        away=args.away,
        n_paths=args.n,
        seed=args.seed,
        before_gw=args.before_gw,
        lineup_cov=not args.no_lineup_cov,
        centre_blend=args.centre_blend,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    fx = out["fixture"]
    print(
        f"{fx['home']} (#{int(fx['home_rank'])}) v {fx['away']} "
        f"(#{int(fx['away_rank'])})  N={out['n']}  props=off"
    )
    print(
        f"centre λ {fx['lam_home_centre']:.3f}/{fx['lam_away_centre']:.3f}  "
        f"sampled mean λ {fx['mean_sampled_lam_h']:.3f}/{fx['mean_sampled_lam_a']:.3f}"
    )
    print(
        f"mean scoreline {out['mean_scoreline'][0]:.2f}-{out['mean_scoreline'][1]:.2f}  "
        f"P(CS home)={out['p_cs_home']:.3f}  P(CS away)={out['p_cs_away']:.3f}"
    )
    print("top scorelines:", list(out["scoreline_freq"].items())[:8])
    print("top scorers:", out["top_scorers"][:8])
    print("top assisters:", out["top_assisters"][:8])
    print("top mean xP:", out["top_mean_xp"][:10])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
