"""Mean-board ``xi_ft`` path on a closed season (oracle ranks + LOSO GF/GA).

Diagnostic follow-on to the live Saka–Groß mean-strand probe:

1. Roll N generative strands (FD short on priced head GWs; oracle-rank long).
2. Average → per-GW μ boards.
3. One joint ``xi_ft`` plan (no captain in V; bank pot ledger; ``ft_pe_nu=0``).

Baselines (PI note):

- **Player rG/rA:** historical rate baseline (prior-season player/positional
  prior, then same-season GW ``< decision`` update).
- **Place → GF/GA:** rankmean±SD fit on closed seasons **excluding** the
  scored season (LOSO).
- **Club ranks:** oracle post-GW38 finishing places for the scored season
  (not live Opta).

  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_hist
  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_hist --season 2025-26 --n 500
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cascade_mc_vs_mean_hist import (
    DEFAULT_SEASON,
    SEASON_E0,
    SEED,
    _build_hist_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import (
    DEFAULT_LINEUP_COV,
    ClubPath,
    _pick_squad_fast,
    _priced_index,
    _undiscounted_scores_by_gw,
    init_paths_from_opta,
    sample_week_scenario,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.path_transfer_plan import (
    pick_path_transfer_plan,
    score_matrix_for_candidates,
    select_plan_candidates,
)
from src.models.short_horizon_cascade import load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_mean_xi_ft_hist.md"
STATE_JSON = ROOT / "data/processed/cascade_mean_xi_ft_hist" / "state.json"

# Path starts at decision GW (as-of = GW < decision). Default matches the
# live probe window shape: plan GW8–19 on 2025-26.
DECISION_GW = 8
HORIZON_END = 19
N_STRANDS = 500
CAND_TOP_N = 48
MILP_TIME = 180.0


def _now() -> float:
    return time.time()


def _out_dir(season: str) -> Path:
    return ROOT / "data" / "predictions" / str(season) / "cascade_mean_xi_ft_hist"


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(
        json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8"
    )


def mean_boards_hist(
    *,
    pool: pd.DataFrame,
    fixtures_gw: dict[int, list[dict[str, str]]],
    priced_by_gw: dict[int, list[dict[str, Any]]],
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    hist_by_team: dict[str, pd.DataFrame],
    alpha_g: float,
    alpha_a: float,
    plan_gws: list[int],
    n_strands: int,
    seed0: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Average generative player scores across strands (no per-strand MILP)."""
    priced_index = _priced_index(priced_by_gw)
    paths0 = init_paths_from_opta(ranks, profile=profile)
    acc = {g: np.zeros(len(pool), dtype=float) for g in plan_gws}
    t0 = _now()
    sources: set[str] = set()
    for s in range(int(n_strands)):
        if s == 0 or (s + 1) % 25 == 0 or s + 1 == int(n_strands):
            print(f"  [mean] strand {s + 1}/{n_strands}", flush=True)
        rng = random.Random(int(seed0) + s)
        paths: dict[str, ClubPath] = {
            c: ClubPath(att=p.att, dfn=p.dfn, n=p.n) for c, p in paths0.items()
        }
        rolled: dict[int, list[dict[str, Any]]] = {}
        for gw in plan_gws:
            week: list[dict[str, Any]] = []
            for fx in fixtures_gw.get(int(gw), []):
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
                    ranks=ranks,
                    profile=profile,
                    sd_league_gf=0.0,
                )
                sources.add(str(sc.get("source") or ""))
                paths[fx["home"]].observe(sc["hg"], sc["ag"])
                paths[fx["away"]].observe(sc["ag"], sc["hg"])
                week.append(sc)
            rolled[int(gw)] = week
        scores = _undiscounted_scores_by_gw(
            pool,
            plan_gws=plan_gws,
            rolled_by_gw=rolled,
            fixtures_gw=fixtures_gw,
            paths=paths,
            league_gf=league_gf,
            alpha_g=float(alpha_g),
            alpha_a=float(alpha_a),
            rng=random.Random(int(seed0) + 10_000 + s),
            hist_by_team=hist_by_team,
            generative_attack=True,
            lineup_cov=DEFAULT_LINEUP_COV,
        )
        for g in plan_gws:
            acc[int(g)] += scores[int(g)]
    mean = {g: acc[g] / float(n_strands) for g in plan_gws}
    meta = {
        "n_strands": int(n_strands),
        "seed0": int(seed0),
        "elapsed_s": _now() - t0,
        "sources": sorted(x for x in sources if x),
        "n_priced_fits": int(sum(len(v) for v in priced_by_gw.values())),
    }
    return mean, meta


def opening_15(
    pool: pd.DataFrame,
    scores_by_gw: dict[int, np.ndarray],
    gws: list[int],
) -> list[str]:
    mean = np.mean([scores_by_gw[g] for g in gws], axis=0)
    tmp = pool.copy()
    tmp["mc_score"] = mean
    squad = _pick_squad_fast(tmp, "mc_score", time_limit=30.0)
    return [str(x) for x in squad["player_id"].tolist()]


def plan_xi_ft(
    pool: pd.DataFrame,
    scores_by_gw: dict[int, np.ndarray],
    gws: list[int],
    owned: list[str],
    *,
    top_n: int = CAND_TOP_N,
    time_limit: float = MILP_TIME,
) -> dict[str, Any]:
    cands = select_plan_candidates(
        pool,
        scores_by_gw,
        gws=gws,
        buy_ids=None,
        force_ids=set(owned),
        top_n=int(top_n),
    )
    S = score_matrix_for_candidates(cands, pool, scores_by_gw, gws)
    return pick_path_transfer_plan(
        cands,
        S,
        gws,
        gamma=1.0,
        ft_pe_nu=0.0,
        value_captain=False,
        time_limit=float(time_limit),
        start_owned_ids=owned,
        start_ft=1,
    )


def _transfer_narrative(
    weeks: list[dict[str, Any]],
    owned0: list[str],
    pool: pd.DataFrame,
) -> list[dict[str, Any]]:
    id_to_name = {
        str(r.player_id): str(r.name) for r in pool.itertuples(index=False)
    }
    prev = set(owned0)
    out: list[dict[str, Any]] = []
    for w in weeks:
        cur = set(w["player_ids"])
        buys = sorted(cur - prev)
        sells = sorted(prev - cur)
        if buys or sells:
            out.append(
                {
                    "gw": int(w["gw"]),
                    "buys": [id_to_name.get(x, x) for x in buys],
                    "sells": [id_to_name.get(x, x) for x in sells],
                    "n_transfers": int(w["n_transfers"]),
                    "hits": int(w["hits"]),
                    "ft_before": int(w["ft_before"]),
                    "ft_after": int(w["ft_after"]),
                }
            )
        prev = cur
    return out


def write_report(state: dict[str, Any]) -> None:
    plan = state["plan"]
    weeks = plan["weeks"]
    xfers = state["transfers"]
    board = state.get("board_top") or []
    lines = [
        "# Mean-board xi_ft hist path — oracle ranks + LOSO GF/GA",
        "",
        f"Protocol: `cascade_mean_xi_ft_hist` · season **{state['season']}** · "
        f"decision GW{state['decision_gw']} → horizon "
        f"{state['horizon_start']}–{state['horizon_end']} · "
        f"N={state['n_strands']} · seed={state['seed']}",
        "",
        "**Diagnostic.** Mean of N generative strands → one joint `xi_ft` plan "
        "(no captain in V; bank pot separate; `ft_pe_nu=0`).",
        "",
        "## Baselines",
        "",
        "- Club ranks: **oracle** post-GW38 finishing places",
        "- Place→GF/GA: **rankmean±SD LOSO** (exclude scored-season E0)",
        "- League ḡ: LOSO closed prior (`league_avg_gf`)",
        "- Player rG/rA: **historical rate baseline** (prior seasons + "
        f"same-season GW `< {state['decision_gw']}`)",
        f"- Short priced head: FD AH+OU GWs {state['priced_gws']}",
        f"- Long: oracle-rank generative (`sources={state['board_meta'].get('sources')}`)",
        "",
        "## Path summary",
        "",
        f"- Objective (Σ XI − hits): **{plan['objective']:.2f}**",
        f"- Total transfers / hits: **{state['total_transfers']}** / "
        f"**{state['total_hits']}**",
        f"- Terminal FT: **{plan['terminal_ft']}** "
        f"(bank_pot.in_objective={plan['bank_pot']['in_objective']})",
        f"- Opening rebuild free: week-0 from owned 15 with start_ft=1",
        f"- Board roll wall: {state['board_meta']['elapsed_s']:.1f}s · "
        f"total wall: {state['elapsed_s']:.1f}s",
        "",
        "## FT ledger",
        "",
        "| GW | FT before→after | transfers | hits | XI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for w in weeks:
        lines.append(
            f"| {w['gw']} | {w['ft_before']}→{w['ft_after']} | "
            f"{w['n_transfers']} | {w['hits']} | {w['xi_score']:.2f} |"
        )
    lines += ["", "## Transfers", ""]
    if not xfers:
        lines.append("_none_")
    else:
        for t in xfers:
            lines.append(
                f"- GW{t['gw']}: buy {t['buys']} / sell {t['sells']} "
                f"(ft {t['ft_before']}→{t['ft_after']}, hits={t['hits']})"
            )
    if board:
        lines += [
            "",
            "## Top mean xP (horizon average)",
            "",
            "| player | team | pos | mean xP |",
            "| --- | --- | --- | ---: |",
        ]
        for r in board[:15]:
            lines.append(
                f"| {r['name']} | {r['team']} | {r['position']} | "
                f"{r['mean_xp']:.2f} |"
            )
    lines += [
        "",
        "## Artefacts",
        "",
        f"- `{STATE_JSON.relative_to(ROOT)}`",
        f"- `{_out_dir(state['season']).relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_hist "
        f"--season {state['season']} --n {state['n_strands']} "
        f"--decision-gw {state['decision_gw']} "
        f"--horizon-end {state['horizon_end']}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--decision-gw", type=int, default=DECISION_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cand-top-n", type=int, default=CAND_TOP_N)
    ap.add_argument("--milp-time", type=float, default=MILP_TIME)
    ap.add_argument(
        "--artifact",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_hist.json"),
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    if season not in SEASON_E0:
        raise SystemExit(f"unknown season {season!r}; choose from {sorted(SEASON_E0)}")
    e0 = SEASON_E0[season]
    decision_gw = int(args.decision_gw)
    horizon_end = int(args.horizon_end)
    # Horizon starts at the decision GW (as-of history is GW < decision).
    start = decision_gw

    print(
        f"[mean-xi-ft] season={season} e0={e0} asof=<GW{decision_gw} "
        f"horizon={start}-{horizon_end} N={args.n} seed={args.seed}",
        flush=True,
    )
    t_all = _now()

    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    print(
        f"[mean-xi-ft] oracle ranks={len(ranks)} LOSO places={len(profile)} "
        f"league_gf={league_gf:.2f}",
        flush=True,
    )

    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    hist = numagent_xp.asof_share_history(
        merged, before_gw=decision_gw, season=season
    )
    if hist.empty:
        raise RuntimeError(f"no history before GW{decision_gw}")
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)

    pool = _build_hist_pool(
        hist, prices, decision_gw=decision_gw, season=season
    )
    print(
        f"[mean-xi-ft] pool={len(pool)} alphas=({alpha_g:.3f},{alpha_a:.3f}) "
        "(rate baseline prior+season)",
        flush=True,
    )

    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start, end=horizon_end)
    plan_gws = sorted(fixtures_gw)
    if not plan_gws:
        raise RuntimeError("no fixtures in horizon")
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    priced_gws = sorted(priced_by_gw)
    print(
        f"[mean-xi-ft] plan_gws={plan_gws} priced={priced_gws}",
        flush=True,
    )

    print("[mean-xi-ft] rolling mean boards…", flush=True)
    scores_by_gw, board_meta = mean_boards_hist(
        pool=pool,
        fixtures_gw=fixtures_gw,
        priced_by_gw=priced_by_gw,
        ranks=ranks,
        profile=profile,
        league_gf=league_gf,
        hist_by_team=team_hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        plan_gws=plan_gws,
        n_strands=int(args.n),
        seed0=int(args.seed),
    )
    print(
        f"[mean-xi-ft] boards done ({board_meta['elapsed_s']:.1f}s) "
        f"sources={board_meta['sources']}",
        flush=True,
    )

    owned = opening_15(pool, scores_by_gw, plan_gws)
    print(f"[mean-xi-ft] opening 15 n={len(owned)}", flush=True)

    print("[mean-xi-ft] xi_ft joint plan…", flush=True)
    t_plan = _now()
    plan = plan_xi_ft(
        pool,
        scores_by_gw,
        plan_gws,
        owned,
        top_n=int(args.cand_top_n),
        time_limit=float(args.milp_time),
    )
    plan_elapsed = _now() - t_plan
    weeks = plan["weeks"]
    total_tr = int(sum(int(w["n_transfers"]) for w in weeks))
    total_hits = int(sum(int(w["hits"]) for w in weeks))
    xfers = _transfer_narrative(weeks, owned, pool)

    # Horizon-mean board ranking for the report.
    mean_xp = np.mean([scores_by_gw[g] for g in plan_gws], axis=0)
    board_top = (
        pool.assign(mean_xp=mean_xp)
        .sort_values("mean_xp", ascending=False)
        .loc[:, ["name", "team", "position", "player_id", "mean_xp"]]
        .head(25)
        .assign(mean_xp=lambda d: d["mean_xp"].round(3))
        .to_dict("records")
    )

    id_to_name = {
        str(r.player_id): str(r.name) for r in pool.itertuples(index=False)
    }
    opening_names = [id_to_name.get(x, x) for x in owned]

    state = {
        "task": "cascade_mean_xi_ft_hist",
        "season": season,
        "e0": e0,
        "decision_gw": decision_gw,
        "horizon_start": start,
        "horizon_end": horizon_end,
        "n_strands": int(args.n),
        "seed": int(args.seed),
        "priced_gws": priced_gws,
        "n_pool": int(len(pool)),
        "ranks": "oracle_post_gw38",
        "gf_ga_profile": "loso_prior_seasons_rankmean",
        "rate_baseline": "prior_plus_season_before_decision",
        "objective": "xi_ft",
        "value_captain": False,
        "ft_pe_nu": 0.0,
        "board_meta": board_meta,
        "opening_15": owned,
        "opening_names": opening_names,
        "plan": {
            "objective": plan["objective"],
            "terminal_ft": plan["terminal_ft"],
            "n_candidates": plan["n_candidates"],
            "value_captain": plan["value_captain"],
            "bank_pot": plan["bank_pot"],
            "weeks": weeks,
        },
        "transfers": xfers,
        "total_transfers": total_tr,
        "total_hits": total_hits,
        "board_top": board_top,
        "plan_elapsed_s": plan_elapsed,
        "elapsed_s": _now() - t_all,
        "report": str(REPORT_MD),
    }
    _save(state)
    out_dir = _out_dir(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "state.json").write_text(
        json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (out_dir / "transfers.json").write_text(
        json.dumps(xfers, indent=2) + "\n", encoding="utf-8"
    )
    write_report(state)

    art = Path(args.artifact)
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")

    summary = {
        "elapsed_s": state["elapsed_s"],
        "board_s": board_meta["elapsed_s"],
        "plan_s": plan_elapsed,
        "total_transfers": total_tr,
        "total_hits": total_hits,
        "terminal_ft": plan["terminal_ft"],
        "objective": plan["objective"],
        "sources": board_meta["sources"],
        "report": str(REPORT_MD),
        "artifact": str(art),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
