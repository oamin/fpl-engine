"""Trial: one event through M → k → scorer/assist dice.

Default fixture: Arsenal v Everton from ``data/scratch/gw08_odds_full.json``
(props + fitted λ). Same-season rates from ``player_gw_2026_27`` (GW < 6).

  PYTHONPATH=. python3 -m src.markets.attack_roll_trial [--n 12] [--seed 0]
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

from src.live.odds_props import match_anytime_props
from src.markets import numagent_xp as nx
from src.markets.attack_roll import EMPTY_A, EMPTY_G, build_side_dice, roll_side_attack
from src.markets.lineup_roll import (
    minutes_corr_pairs,
    roll_lineup_world,
    select_disjoint_pairs,
)
from src.markets.match_outcome import match_outcome_probs_from_lambdas, match_outcome_roll
from src.markets.numagent_xp_pilot import norm_team
from src.markets.prop_shares import anytime_p_to_lambda
from src.rules.fpl_2026 import normalize_position

ROOT = Path(__file__).resolve().parents[2]
ODDS = ROOT / "data" / "scratch" / "gw08_odds_full.json"
PLAYER_GW = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
def _find_bootstrap() -> Path:
    for folder in (
        ROOT / "data" / "predictions" / "2026-27" / "gw07",
        ROOT / "data" / "predictions" / "2026-27" / "gw06",
    ):
        cands = sorted(folder.glob("bootstrap_*.json"))
        if cands:
            return cands[-1]
        plain = folder / "bootstrap.json"
        if plain.is_file():
            return plain
    raise FileNotFoundError("no bootstrap_*.json under predictions/2026-27/gw06|gw07")


BOOTSTRAP = _find_bootstrap()


def _load_bootstrap() -> dict[str, Any]:
    return json.loads(BOOTSTRAP.read_text(encoding="utf-8"))


def _side_frame(
    history: pd.DataFrame,
    *,
    team: str,
    before_gw: int,
    minutes_floor: float = 1.0,
) -> pd.DataFrame:
    """Appearance rates + rolling minutes for one club."""
    hist = history[
        (history["team_norm"] == norm_team(team))
        & (history["gw"] < before_gw)
        & (pd.to_numeric(history["minutes"], errors="coerce").fillna(0) > 0)
    ].copy()
    week_ids = (
        history[(history["team_norm"] == norm_team(team)) & (history["gw"] == before_gw - 1)][
            ["player_id", "player_name", "position"]
        ]
        .drop_duplicates("player_id")
    )
    # Fallback: anyone with history on the club
    if week_ids.empty:
        week_ids = hist[["player_id", "player_name", "position"]].drop_duplicates("player_id")

    rates = nx.player_rates_from_history(
        hist.rename(columns={"goals": "goals_scored"}),
        id_col="player_id",
        goals_col="goals_scored",
    )
    rows = []
    for _, p in week_ids.iterrows():
        pid = int(p["player_id"])
        series = history[(history["player_id"] == pid) & (history["gw"] < before_gw)].sort_values(
            "gw"
        )
        last3 = series.tail(3)
        m = float(last3["minutes"].mean()) if len(last3) else 0.0
        sm = float(last3["minutes"].std(ddof=1)) if len(last3) >= 2 else 0.0
        if m != m:  # NaN
            m = 0.0
        if sm != sm:
            sm = 0.0
        rows.append(
            {
                "player_id": pid,
                "player_name": str(p["player_name"]),
                "position": str(p["position"]),
                "m": m,
                "sigma_m": sm,
            }
        )
    board = pd.DataFrame(rows)
    if rates.empty:
        board["rG"] = board["rA"] = board["drG"] = board["drA"] = 0.0
    else:
        board = board.merge(rates, on="player_id", how="left")
        for c in ("rG", "drG", "rA", "drA"):
            board[c] = board[c].fillna(0.0)
    # Playing set for dice: projected minutes > 0
    board = board[board["m"] >= minutes_floor].reset_index(drop=True)
    if board.empty:
        return board
    # Compositional baseline shares via weights (no shrink for trial clarity)
    wG = np.array(
        [nx.weight_and_delta(r, dr, m, sm)[0] for r, dr, m, sm in board[["rG", "drG", "m", "sigma_m"]].to_numpy()]
    )
    wA = np.array(
        [nx.weight_and_delta(r, dr, m, sm)[0] for r, dr, m, sm in board[["rA", "drA", "m", "sigma_m"]].to_numpy()]
    )
    sG, _ = nx.shares_and_deltas(wG, np.zeros_like(wG))
    sA, _ = nx.shares_and_deltas(wA, np.zeros_like(wA))
    m_arr = board["m"].to_numpy(float)
    pos_list = board["position"].tolist()
    # Early-season: if a channel has no rate mass, fall back to position×minutes prior
    # so the assist dice is not empty for non-scorers.
    def _pos_prior(channel: str) -> np.ndarray:
        weights = nx.SHARE_POS_WEIGHT_G if channel == "G" else nx.SHARE_POS_WEIGHT_A
        raw = np.array(
            [
                float(m_arr[i])
                * float(weights.get(normalize_position(str(pos_list[i])), 0.0))
                for i in range(len(board))
            ],
            dtype=float,
        )
        z = float(raw.sum())
        return raw / z if z > 0 else np.full(len(board), 1.0 / max(len(board), 1))

    if float(wG.sum()) <= 1e-12:
        sG = _pos_prior("G")
    else:
        sG, _ = nx.shrink_shares_by_se(
            sG, np.zeros_like(sG), m_arr, positions=pos_list, channel="G"
        )
    if float(wA.sum()) <= 1e-12:
        sA = _pos_prior("A")
    else:
        # Blend sparse early-season assist mass with role prior (trial only).
        sA_shrunk, _ = nx.shrink_shares_by_se(
            sA, np.zeros_like(sA), m_arr, positions=pos_list, channel="A"
        )
        prior = _pos_prior("A")
        sA = 0.5 * sA_shrunk + 0.5 * prior
        sA = sA / float(sA.sum())
    board["sG"] = sG
    board["sA"] = sA
    return board


def run_trial(
    *,
    n_paths: int = 12,
    seed: int = 0,
    before_gw: int = 6,
    lineup_cov: bool = True,
) -> dict[str, Any]:
    odds = json.loads(ODDS.read_text(encoding="utf-8"))
    match = next(m for m in odds["matches"] if m.get("home") == "Arsenal" and m.get("away") == "Everton")
    props_raw = odds["props"]["Arsenal v Everton"]["all"]
    boot = _load_bootstrap()
    matched = match_anytime_props(props_raw, home="Arsenal", away="Everton", bootstrap=boot)

    history = pd.read_csv(PLAYER_GW)
    history["team_norm"] = history["team"].map(norm_team)
    home_board = _side_frame(history, team="Arsenal", before_gw=before_gw)
    away_board = _side_frame(history, team="Everton", before_gw=before_gw)

    ag, aa, _, _ = nx.load_alphas()
    probs = match_outcome_probs_from_lambdas(
        float(match["lam_h"]), float(match["lam_a"]), float(match.get("rho") or 0.0)
    )
    g_home = ag * float(probs.moments["xg_home"])
    g_away = ag * float(probs.moments["xg_away"])

    # Map props → row index on each board
    by_id = {int(k): float(v) for k, v in (matched.get("by_id") or {}).items()}

    def prop_map(board: pd.DataFrame) -> dict[int, float]:
        out: dict[int, float] = {}
        for i, pid in enumerate(board["player_id"].tolist()):
            if int(pid) in by_id and float(board.iloc[i]["m"]) > 0:
                out[i] = by_id[int(pid)]
        return out

    home_dice = build_side_dice(
        home_board["player_name"].tolist(),
        home_board["sG"].to_numpy(),
        home_board["sA"].to_numpy(),
        alpha_g=ag,
        alpha_a=aa,
        G_team=g_home,
        prop_p=prop_map(home_board) or None,
    )
    away_dice = build_side_dice(
        away_board["player_name"].tolist(),
        away_board["sG"].to_numpy(),
        away_board["sA"].to_numpy(),
        alpha_g=ag,
        alpha_a=aa,
        G_team=g_away,
        prop_p=prop_map(away_board) or None,
    )

    def _pairs_for(board: pd.DataFrame, team: str):
        club_hist = history[
            (history["team_norm"] == norm_team(team)) & (history["gw"] < before_gw)
        ]
        table = minutes_corr_pairs(club_hist)
        return select_disjoint_pairs(board["player_id"].tolist(), table), table

    home_pairs, home_pair_table = _pairs_for(home_board, "Arsenal")
    away_pairs, away_pair_table = _pairs_for(away_board, "Everton")

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    paths = []
    score_hist: Counter[str] = Counter()
    scorer_hist: Counter[str] = Counter()
    assist_hist: Counter[str] = Counter()
    eligible_hist: Counter[str] = Counter()
    pair_winner_hist: Counter[str] = Counter()
    for _ in range(int(n_paths)):
        roll = match_outcome_roll(probs, rng)
        home_lineup = away_lineup = None
        if lineup_cov:
            home_lineup = roll_lineup_world(
                mean_m=home_board["m"].to_numpy(float),
                names=home_board["player_name"].tolist(),
                pairs=home_pairs,
                rng=rng,
            )
            away_lineup = roll_lineup_world(
                mean_m=away_board["m"].to_numpy(float),
                names=away_board["player_name"].tolist(),
                pairs=away_pairs,
                rng=rng,
            )
            for name, ok in zip(home_dice.names, home_lineup.eligible):
                if ok:
                    eligible_hist[f"Arsenal:{name}"] += 1
            for name, ok in zip(away_dice.names, away_lineup.eligible):
                if ok:
                    eligible_hist[f"Everton:{name}"] += 1
            for w, _lo, lab in home_lineup.pair_choices:
                pair_winner_hist[f"Arsenal:{lab}→{w}"] += 1
            for w, _lo, lab in away_lineup.pair_choices:
                pair_winner_hist[f"Everton:{lab}→{w}"] += 1
        home_att = roll_side_attack(
            home_dice, roll.goals_home, np_rng, team="Arsenal", lineup=home_lineup
        )
        away_att = roll_side_attack(
            away_dice, roll.goals_away, np_rng, team="Everton", lineup=away_lineup
        )
        label = f"{roll.goals_home}-{roll.goals_away}"
        score_hist[label] += 1
        for s in home_att.scorers:
            scorer_hist[f"Arsenal:{s}"] += 1
        for s in away_att.scorers:
            scorer_hist[f"Everton:{s}"] += 1
        for a in home_att.assisters:
            assist_hist[f"Arsenal:{a}"] += 1
        for a in away_att.assisters:
            assist_hist[f"Everton:{a}"] += 1
        paths.append(
            {
                "scoreline": label,
                "p_cell": roll.p,
                "home": home_att.as_dict(),
                "away": away_att.as_dict(),
                "home_lineup": None
                if home_lineup is None
                else {
                    "eligible": [
                        n
                        for n, ok in zip(home_dice.names, home_lineup.eligible)
                        if ok
                    ],
                    "pair_choices": [
                        {"winner": w, "loser": lo, "pair": lab}
                        for w, lo, lab in home_lineup.pair_choices
                    ],
                },
                "away_lineup": None
                if away_lineup is None
                else {
                    "eligible": [
                        n
                        for n, ok in zip(away_dice.names, away_lineup.eligible)
                        if ok
                    ],
                    "pair_choices": [
                        {"winner": w, "loser": lo, "pair": lab}
                        for w, lo, lab in away_lineup.pair_choices
                    ],
                },
            }
        )

    # Highlight π for Saka / Gyökeres-style names
    def pi_row(dice, needle: str) -> dict[str, float] | None:
        for i, name in enumerate(dice.names):
            if needle.lower() in name.lower():
                return {
                    "player": name,
                    "pi": float(dice.pi[i]),
                    "rho": float(dice.rho[i]),
                    "lambda_if_prop": float(
                        anytime_p_to_lambda(prop_map(home_board).get(i, 0.0))
                    )
                    if dice is home_dice
                    else None,
                }
        return None

    payload = {
        "fixture": {
            "home": "Arsenal",
            "away": "Everton",
            "source": str(ODDS.relative_to(ROOT)),
            "before_gw": before_gw,
            "lam_h": float(match["lam_h"]),
            "lam_a": float(match["lam_a"]),
            "rho": float(match.get("rho") or 0.0),
            "xg_home": float(probs.moments["xg_home"]),
            "xg_away": float(probs.moments["xg_away"]),
            "G_team_home": g_home,
            "G_team_away": g_away,
            "alpha_g": ag,
            "alpha_a": aa,
        },
        "props_match": {
            "n_matched": matched.get("n_matched"),
            "n_unmatched": matched.get("n_unmatched"),
            "unmatched_sample": (matched.get("unmatched") or [])[:8],
            "home_covered": len(prop_map(home_board)),
            "away_covered": len(prop_map(away_board)),
        },
        "dice": {
            "home": home_dice.summary(),
            "away": away_dice.summary(),
            "empty_tokens": {"goal": EMPTY_G, "assist": EMPTY_A},
        },
        "n_paths": n_paths,
        "seed": seed,
        "lineup_cov": bool(lineup_cov),
        "competitor_pairs": {
            "home": [
                {
                    "a": p.name_i,
                    "b": p.name_j,
                    "corr": p.corr,
                }
                for p in home_pairs
            ],
            "away": [
                {
                    "a": p.name_i,
                    "b": p.name_j,
                    "corr": p.corr,
                }
                for p in away_pairs
            ],
            "home_table_n": int(len(home_pair_table)) if home_pair_table is not None else 0,
            "away_table_n": int(len(away_pair_table)) if away_pair_table is not None else 0,
        },
        "scoreline_freq": dict(score_hist.most_common()),
        "scorer_freq": dict(scorer_hist.most_common(20)),
        "assist_freq": dict(assist_hist.most_common(20)),
        "eligible_freq": dict(eligible_hist.most_common(20)),
        "pair_winner_freq": dict(pair_winner_hist.most_common(20)),
        "paths": paths,
    }
    return payload


def _print_trial(doc: dict[str, Any]) -> None:
    fx = doc["fixture"]
    print(f"=== Trial: {fx['home']} v {fx['away']} ===")
    print(
        f"M moments: xG {fx['xg_home']:.3f}–{fx['xg_away']:.3f}  "
        f"G_team={fx['G_team_home']:.3f}/{fx['G_team_away']:.3f}  "
        f"α_G={fx['alpha_g']:.3f} α_A={fx['alpha_a']:.3f}"
    )
    print(
        f"Props matched: {doc['props_match']['n_matched']}  "
        f"home covered={doc['props_match']['home_covered']}  "
        f"away covered={doc['props_match']['away_covered']}  "
        f"lineup_cov={doc.get('lineup_cov')}"
    )
    if doc.get("competitor_pairs"):
        print("\nCompetitor pairs (Arsenal):")
        for p in doc["competitor_pairs"]["home"]:
            print(f"  {p['a']} ↔ {p['b']}  corr={p['corr']:+.3f}")
    print("\nArsenal top π (goal dice, pre-lineup):")
    for r in doc["dice"]["home"]["top_pi"][:6]:
        print(f"  {r['player']:20s}  π={r['pi']:.3f}  ρ={r['rho']:.3f}")
    n_show = min(12, int(doc["n_paths"]))
    print(f"\nFirst {n_show} of {doc['n_paths']} paths (seed={doc['seed']}):")
    for i, path in enumerate(doc["paths"][:n_show], 1):
        h = path["home"]
        a = path["away"]
        hev = ", ".join(
            f"{e['scorer']}" + (f"←{e['assister']}" if e["assister"] else "")
            for e in h["events"]
        ) or "—"
        aev = ", ".join(
            f"{e['scorer']}" + (f"←{e['assister']}" if e["assister"] else "")
            for e in a["events"]
        ) or "—"
        mutex = ""
        if path.get("home_lineup") and path["home_lineup"].get("pair_choices"):
            mutex = " | " + ", ".join(
                f"{c['winner']}▶{c['loser']}" for c in path["home_lineup"]["pair_choices"][:3]
            )
        print(f"  [{i:02d}] {path['scoreline']:5s}  ARS: {hev}{mutex}")
        print(f"         {'':5s}  EVE: {aev}")
    print("\nScoreline freq:", dict(list(doc["scoreline_freq"].items())[:8]))
    print("Top FPL scorers:", dict(list(doc["scorer_freq"].items())[:10]))
    if doc.get("pair_winner_freq"):
        print("Pair winners:", dict(list(doc["pair_winner_freq"].items())[:8]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=12, help="number of scenario paths")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--before-gw", type=int, default=6)
    p.add_argument(
        "--no-lineup-cov",
        action="store_true",
        help="disable minutes-covariance lineup worlds",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "attack_roll_trial_arsenal_everton.json",
    )
    args = p.parse_args(argv)
    doc = run_trial(
        n_paths=args.n,
        seed=args.seed,
        before_gw=args.before_gw,
        lineup_cov=not args.no_lineup_cov,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    _print_trial(doc)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
