"""Rolling mean-board ``xi_ft``: stick to the open-loop plan?

Short diagnostic trial:

1. **Open-loop** at GW8: mean boards GW8–19 → one ``xi_ft`` plan.
2. **Rolling:** at each of GW8, GW9, GW10, refresh as-of history / rates /
   mean boards for the remaining horizon, replan from the committed 15 + FT
   bank, commit week-0 only.

Compares committed ownership / transfers to the open-loop intention.

  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_rolling --n 100
  PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_rolling --n 100 --steps 8,9,10
"""

from __future__ import annotations

import argparse
import json
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
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_xi_ft_hist import (
    CAND_TOP_N,
    HORIZON_END,
    MILP_TIME,
    mean_boards_hist,
    opening_15,
    _transfer_narrative,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
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
REPORT_MD = ROOT / "reports" / "cascade_mean_xi_ft_rolling.md"
STATE_JSON = ROOT / "data/processed/cascade_mean_xi_ft_rolling" / "state.json"

N_STRANDS = 100
START_GW = 8
STEP_GWS = (8, 9, 10)


def _now() -> float:
    return time.time()


def _save(state: dict[str, Any]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(
        json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _out_dir(season: str) -> Path:
    return ROOT / "data" / "predictions" / str(season) / "cascade_mean_xi_ft_rolling"


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return float(len(a & b) / len(u)) if u else 1.0


def _ensure_owned_in_pool(
    pool: pd.DataFrame, stash: pd.DataFrame | None, owned: list[str]
) -> pd.DataFrame:
    """Keep committed players scoreable even if they drop the minutes filter."""
    have = set(pool["player_id"].astype(str))
    missing = [str(o) for o in owned if str(o) not in have]
    if not missing:
        return pool
    if stash is None or stash.empty:
        raise RuntimeError(f"owned missing from pool with no stash: {missing}")
    add = stash.loc[stash["player_id"].astype(str).isin(missing)].copy()
    if len(add) < len(missing):
        raise RuntimeError(
            f"stash missing owned rows: need {missing}, got "
            f"{sorted(add['player_id'].astype(str).unique())}"
        )
    return pd.concat([pool, add], ignore_index=True)


def _plan_from_owned(
    pool: pd.DataFrame,
    scores_by_gw: dict[int, np.ndarray],
    gws: list[int],
    owned: list[str],
    *,
    start_ft: int,
    top_n: int,
    time_limit: float,
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
        start_ft=int(start_ft),
    )


def _week_by_gw(weeks: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(w["gw"]): w for w in weeks}


def _xfer_sets(prev: set[str], cur: set[str]) -> tuple[set[str], set[str]]:
    return cur - prev, prev - cur


def build_asof_bundle(
    *,
    season: str,
    decision_gw: int,
    horizon_end: int,
    merged: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    ranks: dict[str, float],
    profile: dict[int, dict[str, float]],
    league_gf: float,
    owned: list[str] | None,
    stash: pd.DataFrame | None,
    n_strands: int,
    seed0: int,
) -> dict[str, Any]:
    """As-of pool + mean boards for ``decision_gw … horizon_end``."""
    hist = numagent_xp.asof_share_history(
        merged, before_gw=int(decision_gw), season=season
    )
    if hist.empty:
        raise RuntimeError(f"no history before GW{decision_gw}")
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)
    pool = _build_hist_pool(
        hist, prices, decision_gw=int(decision_gw), season=season
    )
    share_pool = _build_hist_share_pool(
        hist, prices, decision_gw=int(decision_gw), season=season
    )
    if owned:
        pool = _ensure_owned_in_pool(pool, stash, owned)
        share_pool = _ensure_owned_in_pool(share_pool, stash, owned)
    fixtures_gw = _hist_fixtures_by_gw(
        enriched, start=int(decision_gw), end=int(horizon_end)
    )
    plan_gws = sorted(fixtures_gw)
    if not plan_gws:
        raise RuntimeError(f"no fixtures from GW{decision_gw}")
    priced_by_gw = _fd_priced_by_gw(enriched, fd, plan_gws[:2])
    scores_by_gw, board_meta = mean_boards_hist(
        pool=pool,
        share_pool=share_pool,
        fixtures_gw=fixtures_gw,
        priced_by_gw=priced_by_gw,
        ranks=ranks,
        profile=profile,
        league_gf=league_gf,
        hist_by_team=team_hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        plan_gws=plan_gws,
        n_strands=int(n_strands),
        seed0=int(seed0),
    )
    return {
        "decision_gw": int(decision_gw),
        "pool": pool,
        "share_pool": share_pool,
        "plan_gws": plan_gws,
        "priced_gws": sorted(priced_by_gw),
        "scores_by_gw": scores_by_gw,
        "board_meta": board_meta,
        "alpha_g": float(alpha_g),
        "alpha_a": float(alpha_a),
        "n_pool": int(len(pool)),
        "n_share_pool": int(len(share_pool)),
    }


def write_report(state: dict[str, Any]) -> None:
    ol = state["open_loop"]
    roll = state["rolling"]
    stick = state["stick"]
    lines = [
        "# Rolling mean xi_ft — stick to the open-loop plan?",
        "",
        f"Protocol: `cascade_mean_xi_ft_rolling` · season **{state['season']}** · "
        f"horizon {state['horizon_start']}–{state['horizon_end']} · "
        f"steps {state['step_gws']} · N={state['n_strands']} · seed={state['seed']}",
        "",
        "**Diagnostic.** Open-loop mean-board `xi_ft` at GW8; commit that "
        "week-0, then at GW9/GW10 refresh as-of rates + mean boards and replan "
        "from the committed 15 (receding stick test).",
        "",
        "## Baselines (each step)",
        "",
        "- Oracle post-GW38 ranks; LOSO rankmean GF/GA; prior rate baseline "
        "+ same-season GW `< t`",
        "- Mean of N generative strands → one `xi_ft` plan "
        f"(value_captain=false, ft_pe_nu=0)",
        f"- Wall: {state['elapsed_s']:.1f}s",
        "",
        "## Open-loop (GW8 → 19)",
        "",
        f"- Objective: **{ol['objective']:.2f}** · transfers/hits "
        f"**{ol['total_transfers']}/{ol['total_hits']}** · "
        f"terminal FT {ol['terminal_ft']}",
        f"- Board roll: {ol['board_s']:.1f}s · sources {ol['sources']}",
        "",
        "| GW | FT before→after | transfers | hits | XI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for w in ol["weeks"]:
        if int(w["gw"]) > max(state["step_gws"]) + 2:
            # keep table short but show through early horizon
            pass
        lines.append(
            f"| {w['gw']} | {w['ft_before']}→{w['ft_after']} | "
            f"{w['n_transfers']} | {w['hits']} | {w['xi_score']:.2f} |"
        )
    lines += ["", "### Open-loop transfers (full)", ""]
    if not ol["transfers"]:
        lines.append("_none_")
    else:
        for t in ol["transfers"]:
            lines.append(
                f"- GW{t['gw']}: buy {t['buys']} / sell {t['sells']} "
                f"(ft {t['ft_before']}→{t['ft_after']})"
            )

    lines += [
        "",
        "## Rolling commits (GW8 → GW10)",
        "",
        "| step | commit transfers | hits | FT before→after | owned Jaccard vs OL | "
        "sells match | buys match | replan future transfers |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for row in stick["per_step"]:
        lines.append(
            f"| GW{row['gw']} | {row['commit_n_transfers']} | {row['commit_hits']} | "
            f"{row['ft_before']}→{row['ft_after']} | {row['owned_jaccard_vs_ol']:.2f} | "
            f"{row['sells_match']} | {row['buys_match']} | "
            f"{row['replan_future_transfers']} |"
        )
    lines += [
        "",
        f"- Mean owned Jaccard (steps): **{stick['mean_owned_jaccard']:.3f}**",
        f"- Transfer-set exact match rate: **{stick['xfer_exact_match_rate']:.2f}** "
        f"({stick['n_xfer_exact']}/{stick['n_steps']})",
        f"- Rolling committed transfers / hits over steps: "
        f"**{stick['rolling_transfers']}/{stick['rolling_hits']}**",
        f"- Open-loop transfers / hits on the same steps: "
        f"**{stick['ol_transfers_on_steps']}/{stick['ol_hits_on_steps']}**",
        "",
        "## Commit narrative",
        "",
    ]
    for step in roll["steps"]:
        c = step["commit"]
        lines.append(
            f"- **GW{step['gw']}:** buy {c['buys']} / sell {c['sells']} · "
            f"ft {c['ft_before']}→{c['ft_after']} · hits={c['hits']} · "
            f"board {step['board_s']:.1f}s · pool={step['n_pool']}"
        )
        if step.get("replan_transfers_preview"):
            prev = step["replan_transfers_preview"]
            lines.append(
                f"  - replan still wants later: "
                + ", ".join(
                    f"GW{t['gw']} buy {t['buys']}/sell {t['sells']}" for t in prev[:4]
                )
                + (" …" if len(prev) > 4 else "")
            )

    lines += [
        "",
        "## Verdict",
        "",
        state["verdict_md"],
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
        f"PYTHONPATH=. python3 -m src.eval.cascade_mean_xi_ft_rolling "
        f"--season {state['season']} --n {state['n_strands']} "
        f"--steps {','.join(str(g) for g in state['step_gws'])}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--start-gw", type=int, default=START_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument(
        "--steps",
        type=str,
        default=",".join(str(g) for g in STEP_GWS),
        help="comma-separated decision GWs to commit",
    )
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cand-top-n", type=int, default=CAND_TOP_N)
    ap.add_argument("--milp-time", type=float, default=MILP_TIME)
    ap.add_argument(
        "--artifact",
        type=Path,
        default=Path("/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.json"),
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    if season not in SEASON_E0:
        raise SystemExit(f"unknown season {season!r}")
    e0 = SEASON_E0[season]
    start_gw = int(args.start_gw)
    horizon_end = int(args.horizon_end)
    step_gws = [int(x) for x in str(args.steps).split(",") if x.strip()]
    if not step_gws or step_gws[0] != start_gw:
        raise SystemExit("steps must start at --start-gw")
    n_strands = int(args.n)
    seed0 = int(args.seed)

    print(
        f"[roll] season={season} open-loop {start_gw}-{horizon_end} "
        f"steps={step_gws} N={n_strands}",
        flush=True,
    )
    t_all = _now()

    ranks = season_final_ranks(e0)
    profile = rank_gf_ga_profile_excluding(e0)
    league_gf = league_avg_gf(profile)
    merged = load_merged(season)
    enriched = _load_merged_enriched(season)
    fd = _fd_index(season)
    prices = load_prices(season)

    # --- Open-loop at start_gw ---
    print(f"[roll] open-loop boards @ GW{start_gw}…", flush=True)
    ol_bundle = build_asof_bundle(
        season=season,
        decision_gw=start_gw,
        horizon_end=horizon_end,
        merged=merged,
        enriched=enriched,
        fd=fd,
        prices=prices,
        ranks=ranks,
        profile=profile,
        league_gf=league_gf,
        owned=None,
        stash=None,
        n_strands=n_strands,
        seed0=seed0,
    )
    ol_pool = ol_bundle["pool"]
    ol_owned0 = opening_15(ol_pool, ol_bundle["scores_by_gw"], ol_bundle["plan_gws"])
    print(f"[roll] open-loop plan from opening 15…", flush=True)
    ol_plan = _plan_from_owned(
        ol_pool,
        ol_bundle["scores_by_gw"],
        ol_bundle["plan_gws"],
        ol_owned0,
        start_ft=1,
        top_n=int(args.cand_top_n),
        time_limit=float(args.milp_time),
    )
    ol_weeks = ol_plan["weeks"]
    ol_by_gw = _week_by_gw(ol_weeks)
    ol_xfers = _transfer_narrative(ol_weeks, ol_owned0, ol_pool)
    id_to_name = {
        str(r.player_id): str(r.name) for r in ol_pool.itertuples(index=False)
    }

    open_loop = {
        "decision_gw": start_gw,
        "opening_15": ol_owned0,
        "opening_names": [id_to_name.get(x, x) for x in ol_owned0],
        "objective": float(ol_plan["objective"]),
        "terminal_ft": int(ol_plan["terminal_ft"]),
        "total_transfers": int(sum(int(w["n_transfers"]) for w in ol_weeks)),
        "total_hits": int(sum(int(w["hits"]) for w in ol_weeks)),
        "weeks": ol_weeks,
        "transfers": ol_xfers,
        "board_s": float(ol_bundle["board_meta"]["elapsed_s"]),
        "sources": ol_bundle["board_meta"]["sources"],
        "priced_gws": ol_bundle["priced_gws"],
        "n_pool": ol_bundle["n_pool"],
    }
    print(
        f"[roll] open-loop done tr={open_loop['total_transfers']} "
        f"hits={open_loop['total_hits']} ({open_loop['board_s']:.1f}s boards)",
        flush=True,
    )

    # --- Rolling steps ---
    # GW8 commits the open-loop week-0 (same information set). Later steps
    # refresh as-of history + mean boards, then replan from the committed 15.
    owned = list(ol_owned0)
    ft_bank = 1
    stash = ol_pool.copy()
    roll_steps: list[dict[str, Any]] = []
    stick_rows: list[dict[str, Any]] = []

    for step_i, gw in enumerate(step_gws):
        ol_w = ol_by_gw.get(int(gw))
        if ol_w is None:
            raise RuntimeError(f"open-loop missing GW{gw}")
        ol_ids = set(ol_w["player_ids"])
        if int(gw) == start_gw:
            ol_prev = set(ol_owned0)
        else:
            ol_prev_w = ol_by_gw.get(int(gw) - 1)
            ol_prev = set(ol_prev_w["player_ids"]) if ol_prev_w else set(ol_owned0)
        ol_buys, ol_sells = _xfer_sets(ol_prev, ol_ids)

        if int(gw) == start_gw:
            print(f"[roll] step GW{gw}: commit open-loop week-0 (no re-roll)", flush=True)
            week0 = ol_w
            plan_weeks = ol_weeks
            plan_obj = float(ol_plan["objective"])
            plan_term_ft = int(ol_plan["terminal_ft"])
            board_s = 0.0
            sources = open_loop["sources"]
            priced_gws = open_loop["priced_gws"]
            n_pool = open_loop["n_pool"]
            alphas = [ol_bundle["alpha_g"], ol_bundle["alpha_a"]]
            name_pool = ol_pool
        else:
            step_seed = seed0 + 10_000 * int(gw)
            print(f"[roll] step GW{gw} boards (seed={step_seed})…", flush=True)
            bundle = build_asof_bundle(
                season=season,
                decision_gw=int(gw),
                horizon_end=horizon_end,
                merged=merged,
                enriched=enriched,
                fd=fd,
                prices=prices,
                ranks=ranks,
                profile=profile,
                league_gf=league_gf,
                owned=owned,
                stash=stash,
                n_strands=n_strands,
                seed0=step_seed,
            )
            stash = bundle["pool"].copy()
            for r in bundle["pool"].itertuples(index=False):
                id_to_name[str(r.player_id)] = str(r.name)
            plan = _plan_from_owned(
                bundle["pool"],
                bundle["scores_by_gw"],
                bundle["plan_gws"],
                owned,
                start_ft=int(ft_bank),
                top_n=int(args.cand_top_n),
                time_limit=float(args.milp_time),
            )
            week0 = plan["weeks"][0]
            if int(week0["gw"]) != int(gw):
                raise RuntimeError(f"expected commit GW{gw}, got {week0['gw']}")
            plan_weeks = plan["weeks"]
            plan_obj = float(plan["objective"])
            plan_term_ft = int(plan["terminal_ft"])
            board_s = float(bundle["board_meta"]["elapsed_s"])
            sources = bundle["board_meta"]["sources"]
            priced_gws = bundle["priced_gws"]
            n_pool = bundle["n_pool"]
            alphas = [bundle["alpha_g"], bundle["alpha_a"]]
            name_pool = bundle["pool"]

        prev = set(owned)
        cur = set(week0["player_ids"])
        buys, sells = _xfer_sets(prev, cur)
        commit = {
            "gw": int(gw),
            "player_ids": list(week0["player_ids"]),
            "names": [id_to_name.get(x, x) for x in week0["player_ids"]],
            "buys": sorted(id_to_name.get(x, x) for x in buys),
            "sells": sorted(id_to_name.get(x, x) for x in sells),
            "buy_ids": sorted(buys),
            "sell_ids": sorted(sells),
            "n_transfers": int(week0["n_transfers"]),
            "hits": int(week0["hits"]),
            "ft_before": int(week0["ft_before"]),
            "ft_after": int(week0["ft_after"]),
            "xi_score": float(week0["xi_score"]),
        }

        replan_xfers = _transfer_narrative(plan_weeks, owned, name_pool)
        future_tr = int(sum(int(w["n_transfers"]) for w in plan_weeks[1:]))

        stick_rows.append(
            {
                "gw": int(gw),
                "commit_n_transfers": commit["n_transfers"],
                "commit_hits": commit["hits"],
                "ft_before": commit["ft_before"],
                "ft_after": commit["ft_after"],
                "owned_jaccard_vs_ol": _jaccard(cur, ol_ids),
                "sells_match": sells == ol_sells,
                "buys_match": buys == ol_buys,
                "xfer_exact": sells == ol_sells and buys == ol_buys,
                "ol_n_transfers": int(ol_w["n_transfers"]),
                "ol_hits": int(ol_w["hits"]),
                "replan_future_transfers": future_tr,
                "commit_buys": commit["buys"],
                "commit_sells": commit["sells"],
                "ol_buys": sorted(id_to_name.get(x, x) for x in ol_buys),
                "ol_sells": sorted(id_to_name.get(x, x) for x in ol_sells),
                "reused_open_loop": int(gw) == start_gw,
            }
        )

        roll_steps.append(
            {
                "gw": int(gw),
                "n_pool": n_pool,
                "priced_gws": priced_gws,
                "board_s": board_s,
                "sources": sources,
                "alphas": alphas,
                "commit": commit,
                "replan_objective": plan_obj,
                "replan_terminal_ft": plan_term_ft,
                "replan_transfers_preview": replan_xfers,
                "replan_weeks_head": plan_weeks[:4],
                "reused_open_loop": int(gw) == start_gw,
            }
        )

        owned = list(week0["player_ids"])
        ft_bank = int(week0["ft_after"])
        print(
            f"[roll] commit GW{gw}: tr={commit['n_transfers']} hits={commit['hits']} "
            f"ft {commit['ft_before']}→{commit['ft_after']} "
            f"J={stick_rows[-1]['owned_jaccard_vs_ol']:.2f} "
            f"xfer_exact={stick_rows[-1]['xfer_exact']}",
            flush=True,
        )

    n_steps = len(stick_rows)
    n_exact = sum(1 for r in stick_rows if r["xfer_exact"])
    mean_j = float(np.mean([r["owned_jaccard_vs_ol"] for r in stick_rows]))
    roll_tr = int(sum(r["commit_n_transfers"] for r in stick_rows))
    roll_hits = int(sum(r["commit_hits"] for r in stick_rows))
    ol_tr_steps = int(sum(r["ol_n_transfers"] for r in stick_rows))
    ol_hits_steps = int(sum(r["ol_hits"] for r in stick_rows))

    if mean_j >= 0.95 and n_exact == n_steps:
        verdict = (
            f"**Sticks.** Rolling commits match the open-loop transfer sets on "
            f"all {n_steps} steps (mean ownership Jaccard {mean_j:.3f})."
        )
    elif mean_j >= 0.80 and roll_hits == 0 and ol_hits_steps == 0:
        verdict = (
            f"**Mostly sticks / same regime.** Mean ownership Jaccard "
            f"{mean_j:.3f}; exact transfer match {n_exact}/{n_steps}; "
            f"both arms hit-free on the stepped GWs "
            f"(OL {ol_tr_steps} FT vs rolling {roll_tr} FT)."
        )
    else:
        verdict = (
            f"**Drifts.** Mean ownership Jaccard {mean_j:.3f}; exact transfer "
            f"match {n_exact}/{n_steps}; rolling {roll_tr} FT / {roll_hits} hits "
            f"vs open-loop {ol_tr_steps} FT / {ol_hits_steps} hits on stepped GWs."
        )

    stick = {
        "per_step": stick_rows,
        "mean_owned_jaccard": mean_j,
        "xfer_exact_match_rate": float(n_exact / max(n_steps, 1)),
        "n_xfer_exact": n_exact,
        "n_steps": n_steps,
        "rolling_transfers": roll_tr,
        "rolling_hits": roll_hits,
        "ol_transfers_on_steps": ol_tr_steps,
        "ol_hits_on_steps": ol_hits_steps,
    }

    # Lightweight state (drop score arrays)
    state = {
        "task": "cascade_mean_xi_ft_rolling",
        "season": season,
        "e0": e0,
        "horizon_start": start_gw,
        "horizon_end": horizon_end,
        "step_gws": step_gws,
        "n_strands": n_strands,
        "seed": seed0,
        "ranks": "oracle_post_gw38",
        "gf_ga_profile": "loso_prior_seasons_rankmean",
        "rate_baseline": "prior_plus_season_before_each_step",
        "objective": "xi_ft",
        "open_loop": open_loop,
        "rolling": {"steps": roll_steps, "final_owned": owned, "final_ft": ft_bank},
        "stick": stick,
        "verdict_md": verdict,
        "elapsed_s": _now() - t_all,
        "report": str(REPORT_MD),
    }
    _save(state)
    out_dir = _out_dir(season)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "state.json").write_text(
        json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(state)

    art = Path(args.artifact)
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")

    # Summary plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gws = [r["gw"] for r in stick_rows]
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), sharex=True)
        ax = axes[0]
        x = np.arange(len(gws))
        w = 0.35
        ax.bar(x - w / 2, [r["ol_n_transfers"] for r in stick_rows], width=w,
               color="#7f8c8d", label="open-loop")
        ax.bar(x + w / 2, [r["commit_n_transfers"] for r in stick_rows], width=w,
               color="#2980b9", label="rolling commit")
        ax.set_ylabel("transfers")
        ax.set_title(
            f"Rolling vs open-loop · N={n_strands} · "
            f"J̄={mean_j:.2f} · exact={n_exact}/{n_steps}"
        )
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        ax2 = axes[1]
        ax2.plot(x, [r["owned_jaccard_vs_ol"] for r in stick_rows], "o-",
                 color="#1a5276", lw=2)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("owned Jaccard vs OL")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"GW{g}" for g in gws])
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        plot = Path("/opt/cursor/artifacts/cascade_mean_xi_ft_rolling.png")
        fig.savefig(plot, dpi=140)
        print("wrote", plot, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[roll] plot skipped: {exc}", flush=True)

    print(json.dumps({
        "elapsed_s": state["elapsed_s"],
        "mean_owned_jaccard": mean_j,
        "xfer_exact_match_rate": stick["xfer_exact_match_rate"],
        "rolling_transfers": roll_tr,
        "rolling_hits": roll_hits,
        "ol_transfers_on_steps": ol_tr_steps,
        "verdict": verdict,
        "report": str(REPORT_MD),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
