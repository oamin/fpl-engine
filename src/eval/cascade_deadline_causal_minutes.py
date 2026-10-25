"""Open-loop frozen minutes vs deadline-causal rolling path plan.

Protocol: ``cascade_deadline_causal_minutes`` (diagnostic).

Motivation: FT banking is insurance for unexpected minutes drops. Open-loop
plans see a frozen minutes vector for the whole horizon, so there is nothing
to insure against. Candidate updates oracle minutes only at each GW deadline
and replans (as-of minutes for later weeks).

  PYTHONPATH=. python3 -m src.eval.cascade_deadline_causal_minutes
  PYTHONPATH=. python3 -m src.eval.cascade_deadline_causal_minutes --n 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.cascade_ft_pe_path import _path_resid_table, _transfer_stats
from src.eval.cascade_mc_vs_mean_hist import (
    DECISION_GW,
    DEFAULT_SEASON,
    HORIZON_END,
    N_STRANDS,
    SEASON_E0,
    SEED,
    SNAPSHOT_EVERY,
    START_OFFSET,
    _build_hist_pool,
    _build_hist_share_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_path_gw_resid import (
    run_path_strands,
    semi_oracle_elements,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import (
    DEFAULT_FT_PE_NU,
    PATH_PLAN_DEADLINE_CAUSAL,
    PATH_PLAN_OPEN_LOOP,
)
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import build_frozen_side_table, load_prices
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_deadline_causal_minutes.md"
STATE_JSON = ROOT / "data/processed/cascade_deadline_causal_minutes" / "state.json"


def _now() -> float:
    return time.time()


def _out_dir(season: str) -> Path:
    return ROOT / "data" / "predictions" / str(season) / "cascade_deadline_causal_minutes"


def _mean_terminal_ft(path: dict[str, Any]) -> float:
    vals: list[float] = []
    for st in path.get("strands") or []:
        sqs = st.get("squads") or []
        if sqs:
            vals.append(float(sqs[-1].get("ft_after") or 0))
    return float(np.mean(vals)) if vals else float("nan")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--decision-gw", type=int, default=DECISION_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    ap.add_argument(
        "--ft-pe-nu",
        type=float,
        default=float(DEFAULT_FT_PE_NU),
        help="terminal FT PE ν (default HIT_COST)",
    )
    args = ap.parse_args(argv)

    season = str(args.season)
    if season not in SEASON_E0:
        raise SystemExit(f"unknown season {season!r}")
    e0 = SEASON_E0[season]
    decision_gw = int(args.decision_gw)
    start = decision_gw + START_OFFSET
    pe_nu = float(args.ft_pe_nu)

    print(
        f"[causal_m] season={season} decision={decision_gw} "
        f"horizon={start}-{args.horizon_end} N={args.n} "
        f"control={PATH_PLAN_OPEN_LOOP} vs candidate={PATH_PLAN_DEADLINE_CAUSAL} "
        f"ft_pe_nu={pe_nu:g}",
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
    hist = numagent_xp.asof_share_history(
        merged, before_gw=decision_gw, season=season
    )
    alpha_g, alpha_a = _asof_alphas(hist)
    team_hist = hist_by_team_norm(hist)

    score_pool = _build_hist_pool(
        hist, prices, decision_gw=decision_gw, season=season
    )
    share_pool = _build_hist_share_pool(
        hist, prices, decision_gw=decision_gw, season=season
    )
    oracle_els = semi_oracle_elements(merged, decision_gw)
    buy_pool = score_pool.loc[
        pd.to_numeric(score_pool["element"], errors="coerce").isin(oracle_els)
    ].copy()
    pool_ids = set(buy_pool["player_id"].astype(str))
    pool_ids |= {
        str(int(e))
        for e in pd.to_numeric(buy_pool["element"], errors="coerce").dropna().astype(int)
    }

    # Control open-loop freeze: as-of + GW-t overlay on buy-eligible (legacy).
    gw_t = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == decision_gw]
    m_t = {
        int(r.element): float(r.minutes or 0.0)
        for r in gw_t.itertuples(index=False)
        if pd.notna(r.element)
    }
    if "m" not in score_pool.columns:
        score_pool["m"] = pd.to_numeric(
            score_pool["xmi"] if "xmi" in score_pool.columns else 0.0,
            errors="coerce",
        ).fillna(0.0)
    score_pool["m"] = [
        float(m_t[int(e)]) if int(e) in oracle_els and int(e) in m_t else float(m)
        for e, m in zip(
            pd.to_numeric(score_pool["element"], errors="coerce").fillna(-1),
            pd.to_numeric(score_pool["m"], errors="coerce").fillna(0.0),
            strict=True,
        )
    ]
    score_pool["xmi"] = score_pool["m"]
    score_pool["sigma_m"] = 0.0

    fixtures_gw = _hist_fixtures_by_gw(enriched, start=start, end=int(args.horizon_end))
    horizon_gws = sorted(fixtures_gw)
    snapshot_gws = [
        g for i, g in enumerate(horizon_gws) if i % max(1, int(args.snapshot_every)) == 0
    ]
    priced_by_gw = _fd_priced_by_gw(enriched, fd, horizon_gws[:2])
    side, _, _ = build_frozen_side_table(
        hist,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        season=season,
        historical_rate_baseline=True,
    )

    arms: dict[str, Any] = {}
    for label, mode in (
        ("control", PATH_PLAN_OPEN_LOOP),
        ("candidate", PATH_PLAN_DEADLINE_CAUSAL),
    ):
        print(f"[causal_m] path arm {label} ({mode})…", flush=True)
        t0 = _now()
        path = run_path_strands(
            pool=score_pool,
            share_pool=share_pool,
            fixtures_gw=fixtures_gw,
            priced_by_gw=priced_by_gw,
            ranks=ranks,
            profile=profile,
            league_gf=league_gf,
            hist_by_team=team_hist,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            horizon_gws=horizon_gws,
            snapshot_gws=snapshot_gws,
            n_strands=int(args.n),
            seed=int(args.seed),
            selectable_ids=pool_ids,
            ft_pe_nu=pe_nu,
            path_plan_mode=mode,
            merged=merged if mode == PATH_PLAN_DEADLINE_CAUSAL else None,
            season=season if mode == PATH_PLAN_DEADLINE_CAUSAL else None,
        )
        xfer = _transfer_stats(path)
        resid = _path_resid_table(
            path=path,
            side=side,
            merged=merged,
            enriched=enriched,
            fd=fd,
            prices=prices,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            season=season,
            decision_gw=decision_gw,
            horizon_gws=horizon_gws,
            pool=score_pool,
            pool_ids=pool_ids,
        )
        term = _mean_terminal_ft(path)
        arms[label] = {
            "path_plan_mode": mode,
            "elapsed_s": _now() - t0,
            "transfers": xfer,
            "mean_terminal_ft": term,
            "path_actual_sum": float(resid["path_actual"].sum())
            if not resid.empty
            else float("nan"),
            "path_resid_bias": float(resid["path_resid"].mean())
            if not resid.empty
            else float("nan"),
            "path_resid_mae": float(resid["path_resid"].abs().mean())
            if not resid.empty
            else float("nan"),
            "strands": path.get("strands"),
        }
        print(
            f"  [{label}] transfers/strand={xfer['mean_transfers_per_strand']:.2f} "
            f"paid={xfer['mean_paid_per_strand']:.2f} "
            f"hit_pts={xfer['mean_hit_pts_per_strand']:.1f} "
            f"terminal_ft={term:.2f} "
            f"actual_sum={arms[label]['path_actual_sum']:.1f} "
            f"({arms[label]['elapsed_s']:.1f}s)",
            flush=True,
        )

    ctrl, cand = arms["control"], arms["candidate"]
    delta = {
        "transfers_per_strand": cand["transfers"]["mean_transfers_per_strand"]
        - ctrl["transfers"]["mean_transfers_per_strand"],
        "paid_per_strand": cand["transfers"]["mean_paid_per_strand"]
        - ctrl["transfers"]["mean_paid_per_strand"],
        "hit_pts_per_strand": cand["transfers"]["mean_hit_pts_per_strand"]
        - ctrl["transfers"]["mean_hit_pts_per_strand"],
        "terminal_ft": cand["mean_terminal_ft"] - ctrl["mean_terminal_ft"],
        "path_actual_sum": cand["path_actual_sum"] - ctrl["path_actual_sum"],
        "path_resid_bias": cand["path_resid_bias"] - ctrl["path_resid_bias"],
    }
    # Heuristic decision for the diagnostic batch.
    if delta["terminal_ft"] >= 0.25 and delta["path_actual_sum"] >= -2.0:
        decision = "keep_watching_causal_helps_bank"
    elif abs(delta["transfers_per_strand"]) < 0.2 and abs(delta["terminal_ft"]) < 0.15:
        decision = "keep_watching_causal_small"
    else:
        decision = "keep_watching_causal_moves"

    state = {
        "task": "cascade_deadline_causal_minutes",
        "season": season,
        "decision_gw": decision_gw,
        "horizon": [start, int(args.horizon_end)],
        "n": int(args.n),
        "seed": int(args.seed),
        "ft_pe_nu": pe_nu,
        "control": PATH_PLAN_OPEN_LOOP,
        "candidate": PATH_PLAN_DEADLINE_CAUSAL,
        "decision": decision,
        "delta_candidate_minus_control": delta,
        "elapsed_s": _now() - t_all,
        "arms": {
            k: {kk: vv for kk, vv in v.items() if kk != "strands"}
            for k, v in arms.items()
        },
        "strands": {k: v.get("strands") for k, v in arms.items()},
    }
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")
    out = _out_dir(season)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "delta": delta,
                "decision": decision,
                "control": state["arms"]["control"],
                "candidate": state["arms"]["candidate"],
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# Deadline-causal minutes vs open-loop path plan",
        "",
        f"Protocol: `cascade_deadline_causal_minutes` · season **{season}** · "
        f"decision GW{decision_gw} → {start}–{args.horizon_end} · N={args.n} · seed={args.seed}",
        "",
        "FT banking is insurance for unexpected minutes/rotations. "
        "Control freezes minutes for the whole horizon (open-loop joint MILP). "
        "Candidate reveals oracle minutes only at each GW deadline and replans "
        "(as-of minutes for later weeks).",
        "",
        "| arm | minutes / planner |",
        "| --- | --- |",
        f"| control | frozen as-of (+ GW-t buy overlay); `{PATH_PLAN_OPEN_LOOP}` joint MILP |",
        f"| candidate | oracle @ deadline GW only; as-of for later; `{PATH_PLAN_DEADLINE_CAUSAL}` rolling replan |",
        "",
        f"Both arms use terminal FT PE ν={pe_nu:g}. Wall: {state['elapsed_s']:.1f}s",
        "",
        "## Transfer behaviour (mean over strands; excl. opening free rebuild)",
        "",
        "| arm | transfers | free used | paid | hit pts | mean terminal ft |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| control | {ctrl['transfers']['mean_transfers_per_strand']:.2f} | "
            f"{ctrl['transfers']['mean_free_used_per_strand']:.2f} | "
            f"{ctrl['transfers']['mean_paid_per_strand']:.2f} | "
            f"{ctrl['transfers']['mean_hit_pts_per_strand']:.1f} | "
            f"{ctrl['mean_terminal_ft']:.2f} |"
        ),
        (
            f"| candidate | {cand['transfers']['mean_transfers_per_strand']:.2f} | "
            f"{cand['transfers']['mean_free_used_per_strand']:.2f} | "
            f"{cand['transfers']['mean_paid_per_strand']:.2f} | "
            f"{cand['transfers']['mean_hit_pts_per_strand']:.1f} | "
            f"{cand['mean_terminal_ft']:.2f} |"
        ),
        (
            f"| Δ (cand−ctrl) | {delta['transfers_per_strand']:+.2f} |  | "
            f"{delta['paid_per_strand']:+.2f} | {delta['hit_pts_per_strand']:+.1f} | "
            f"{delta['terminal_ft']:+.2f} |"
        ),
        "",
        "## Path residual vs actual",
        "",
        "| arm | sum actual | resid bias | MAE |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| control | {ctrl['path_actual_sum']:.1f} | "
            f"{ctrl['path_resid_bias']:+.2f} | {ctrl['path_resid_mae']:.2f} |"
        ),
        (
            f"| candidate | {cand['path_actual_sum']:.1f} | "
            f"{cand['path_resid_bias']:+.2f} | {cand['path_resid_mae']:.2f} |"
        ),
        f"| Δ | {delta['path_actual_sum']:+.1f} | {delta['path_resid_bias']:+.2f} |  |",
        "",
        "## Decision",
        "",
        f"**`{decision}`** — diagnostic only (not a four-season promote).",
        "",
        "## Artefacts",
        "",
        f"- `{STATE_JSON.relative_to(ROOT)}`",
        f"- `{out.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"PYTHONPATH=. python3 -m src.eval.cascade_deadline_causal_minutes "
        f"--season {season} --n {args.n}",
        "```",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))
    print(json.dumps({"delta": delta, "decision": decision, "elapsed_s": state["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
