"""Path FT potential-energy trial vs legacy ``S − hit_pts``.

Protocol: ``cascade_ft_pe_path`` (diagnostic). Joint multi-GW transfer plan;
control ``ft_pe_nu=0``, candidate ``ft_pe_nu=HIT_COST`` with terminal
``U(f_H)=ν·f_H`` (each stored FT = +4, five = +20).

  PYTHONPATH=. python3 -m src.eval.cascade_ft_pe_path
  PYTHONPATH=. python3 -m src.eval.cascade_ft_pe_path --season 2025-26 --n 12
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
    DECISION_GW,
    DEFAULT_SEASON,
    HORIZON_END,
    N_STRANDS,
    SEASON_E0,
    SEED,
    SNAPSHOT_EVERY,
    START_OFFSET,
    _build_hist_pool,
    _fd_priced_by_gw,
    _hist_fixtures_by_gw,
)
from src.eval.cascade_mean_path_gw_resid import (
    MIN_ORACLE_MINUTES,
    _mean_board_oracle,
    _squad_frame_for_bank,
    _xi_pred_from_board,
    bank_locked_actual,
    run_path_strands,
    semi_oracle_elements,
)
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.short_horizon_generative import hist_by_team_norm
from src.models.multistage_mc_milp import ft_bank_potential
from src.models.opta_gf_ga_horizon import (
    league_avg_gf,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.models.short_horizon_cascade import build_frozen_side_table, load_prices
from src.rules.fpl_2026 import HIT_COST
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched

ROOT = Path(__file__).resolve().parents[2]
REPORT_MD = ROOT / "reports" / "cascade_ft_pe_path.md"
STATE_JSON = ROOT / "data/processed/cascade_ft_pe_path" / "state.json"
# Simple PE: each stored FT worth one hit.
FT_PE_NU = float(HIT_COST)


def _now() -> float:
    return time.time()


def _out_dir(season: str) -> Path:
    return ROOT / "data" / "predictions" / str(season) / "cascade_ft_pe_path"


def _transfer_stats(path: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for st in path.get("strands") or []:
        for sq in st.get("squads") or []:
            nt = int(sq.get("n_transfers") or 0)
            ft_b = int(sq.get("ft_before") or 0)
            rows.append(
                {
                    "strand": st.get("strand_id"),
                    "gw": int(sq["gw"]),
                    "n_transfers": nt,
                    "hits": int(sq.get("hits") or 0),
                    "ft_before": ft_b,
                    "ft_after": int(sq["ft_after"])
                    if sq.get("ft_after") is not None
                    else None,
                    "paid": max(0, nt - ft_b),
                    "free_used": min(nt, ft_b),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "n_squad_rows": 0,
            "mean_transfers_per_strand": 0.0,
            "mean_hit_pts_per_strand": 0.0,
            "mean_paid_per_strand": 0.0,
            "mean_free_used_per_strand": 0.0,
            "per_gw": [],
        }
    later = df.loc[df["gw"] > df["gw"].min()]  # exclude opening free rebuild
    by_strand = later.groupby("strand").agg(
        transfers=("n_transfers", "sum"),
        hit_pts=("hits", "sum"),
        paid=("paid", "sum"),
        free_used=("free_used", "sum"),
    )
    per_gw = (
        later.groupby("gw")
        .agg(
            mean_transfers=("n_transfers", "mean"),
            mean_hit_pts=("hits", "mean"),
            mean_paid=("paid", "mean"),
            mean_free_used=("free_used", "mean"),
            mean_ft_after=("ft_after", "mean"),
        )
        .reset_index()
    )
    return {
        "n_squad_rows": int(len(df)),
        "mean_transfers_per_strand": float(by_strand["transfers"].mean()),
        "mean_hit_pts_per_strand": float(by_strand["hit_pts"].mean()),
        "mean_paid_per_strand": float(by_strand["paid"].mean()),
        "mean_free_used_per_strand": float(by_strand["free_used"].mean()),
        "per_gw": per_gw.round(3).to_dict(orient="records"),
    }


def _path_resid_table(
    *,
    path: dict[str, Any],
    side: pd.DataFrame,
    merged: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    alpha_g: float,
    alpha_a: float,
    season: str,
    decision_gw: int,
    horizon_gws: list[int],
    pool: pd.DataFrame,
    pool_ids: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gw in horizon_gws:
        board = _mean_board_oracle(
            side,
            merged,
            enriched,
            fd,
            prices,
            season=season,
            decision_gw=decision_gw,
            gw=int(gw),
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            pool_ids=pool_ids,
        )
        gw_rows = merged.loc[pd.to_numeric(merged["GW"], errors="coerce") == int(gw)]
        preds: list[float] = []
        acts: list[float] = []
        for st in path["strands"]:
            held = (st.get("held") or {}).get(int(gw))
            if not held:
                continue
            xi = list(held["xi_ids"])
            cap = held.get("captain_id")
            pids = list(held.get("player_ids") or xi)
            preds.append(_xi_pred_from_board(board, xi, cap, season))
            sdf = _squad_frame_for_bank(
                player_ids=pids,
                xi_ids=xi,
                captain_id=cap,
                board=board,
                merged_gw=gw_rows,
                season=season,
                pool=pool,
            )
            bank = bank_locked_actual(sdf, xi_ids=xi, captain_id=cap, season=season)
            acts.append(float(bank["actual"]))
        if not preds:
            continue
        pred = float(np.mean(preds))
        act = float(np.mean(acts))
        rows.append(
            {
                "gw": int(gw),
                "path_pred": pred,
                "path_actual": act,
                "path_resid": pred - act,
                "n_strands": int(len(preds)),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default=DEFAULT_SEASON)
    ap.add_argument("--decision-gw", type=int, default=DECISION_GW)
    ap.add_argument("--horizon-end", type=int, default=HORIZON_END)
    ap.add_argument("--n", type=int, default=N_STRANDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY)
    ap.add_argument("--nu", type=float, default=FT_PE_NU, help="candidate ft_pe_nu")
    args = ap.parse_args(argv)

    season = str(args.season)
    if season not in SEASON_E0:
        raise SystemExit(f"unknown season {season!r}")
    e0 = SEASON_E0[season]
    decision_gw = int(args.decision_gw)
    start = decision_gw + START_OFFSET
    nu = float(args.nu)

    print(
        f"[ft_pe] season={season} decision={decision_gw} "
        f"horizon={start}-{args.horizon_end} N={args.n} "
        f"control nu=0 vs candidate nu={nu}",
        flush=True,
    )
    print(
        f"[ft_pe] joint-plan terminal U(b)=ν·b (ν={nu:g}=HIT_COST): "
        + ", ".join(f"U({b})={ft_bank_potential(b, nu=nu):.0f}" for b in range(0, 6)),
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
    oracle_els = semi_oracle_elements(merged, decision_gw)
    buy_pool = score_pool.loc[
        pd.to_numeric(score_pool["element"], errors="coerce").isin(oracle_els)
    ].copy()
    pool_ids = set(buy_pool["player_id"].astype(str))
    pool_ids |= {
        str(int(e))
        for e in pd.to_numeric(buy_pool["element"], errors="coerce").dropna().astype(int)
    }

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
    for label, pe_nu in (("control", 0.0), ("candidate", nu)):
        print(f"[ft_pe] path arm {label} (nu={pe_nu})…", flush=True)
        t0 = _now()
        path = run_path_strands(
            pool=score_pool,
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
            ft_pe_nu=float(pe_nu),
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
        arms[label] = {
            "ft_pe_nu": float(pe_nu),
            "elapsed_s": _now() - t0,
            "path_elapsed_s": float(path.get("elapsed_s") or 0.0),
            "transfers": xfer,
            "per_gw": resid.to_dict(orient="records"),
            "path_actual_sum": float(resid["path_actual"].sum()) if len(resid) else 0.0,
            "path_resid_bias": float(resid["path_resid"].mean()) if len(resid) else 0.0,
            "path_resid_mae": float(resid["path_resid"].abs().mean())
            if len(resid)
            else 0.0,
            # keep strand squads for audit (compact: drop names)
            "strands": [
                {
                    "strand_id": s["strand_id"],
                    "squads": [
                        {
                            k: sq.get(k)
                            for k in (
                                "gw",
                                "n_transfers",
                                "hits",
                                "ft_before",
                                "ft_after",
                                "captain_id",
                            )
                        }
                        for sq in s.get("squads") or []
                    ],
                }
                for s in path.get("strands") or []
            ],
        }
        print(
            f"  [{label}] transfers/strand={xfer['mean_transfers_per_strand']:.2f} "
            f"paid={xfer['mean_paid_per_strand']:.2f} "
            f"hit_pts={xfer['mean_hit_pts_per_strand']:.1f} "
            f"actual_sum={arms[label]['path_actual_sum']:.1f} "
            f"resid_bias={arms[label]['path_resid_bias']:+.2f} "
            f"({arms[label]['elapsed_s']:.1f}s)",
            flush=True,
        )

    ctrl = arms["control"]
    cand = arms["candidate"]
    delta = {
        "transfers_per_strand": cand["transfers"]["mean_transfers_per_strand"]
        - ctrl["transfers"]["mean_transfers_per_strand"],
        "paid_per_strand": cand["transfers"]["mean_paid_per_strand"]
        - ctrl["transfers"]["mean_paid_per_strand"],
        "hit_pts_per_strand": cand["transfers"]["mean_hit_pts_per_strand"]
        - ctrl["transfers"]["mean_hit_pts_per_strand"],
        "path_actual_sum": cand["path_actual_sum"] - ctrl["path_actual_sum"],
        "path_resid_bias": cand["path_resid_bias"] - ctrl["path_resid_bias"],
    }

    state = {
        "task": "cascade_ft_pe_path",
        "planner": "joint_path_transfer_plan",
        "season": season,
        "decision_gw": decision_gw,
        "horizon": [start, int(args.horizon_end)],
        "n": int(args.n),
        "seed": int(args.seed),
        "snapshot_gws": snapshot_gws,
        "candidate_nu": nu,
        "pe_form": "terminal U(f_H)=ν·f_H with ν=HIT_COST (+4 per stored FT, +20 at cap)",
        "U": {str(b): ft_bank_potential(b, nu=nu) for b in range(0, 6)},
        "arms": {
            k: {kk: vv for kk, vv in v.items() if kk != "strands"}
            for k, v in arms.items()
        },
        "strands": {k: v["strands"] for k, v in arms.items()},
        "delta_candidate_minus_control": delta,
        "elapsed_s": _now() - t_all,
    }
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    out = _out_dir(season)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(
            {
                "delta": delta,
                "control": {
                    "transfers": ctrl["transfers"],
                    "path_actual_sum": ctrl["path_actual_sum"],
                    "path_resid_bias": ctrl["path_resid_bias"],
                },
                "candidate": {
                    "transfers": cand["transfers"],
                    "path_actual_sum": cand["path_actual_sum"],
                    "path_resid_bias": cand["path_resid_bias"],
                    "ft_pe_nu": nu,
                },
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    def _mean_terminal_ft(strand_rows: list[dict[str, Any]]) -> float:
        vals: list[float] = []
        for st in strand_rows:
            sqs = st.get("squads") or []
            if not sqs:
                continue
            fa = sqs[-1].get("ft_after")
            if fa is not None:
                vals.append(float(fa))
        return float(np.mean(vals)) if vals else float("nan")

    ctrl_term = _mean_terminal_ft(arms["control"]["strands"])
    cand_term = _mean_terminal_ft(arms["candidate"]["strands"])

    lines = [
        "# Path FT potential energy vs legacy objective (joint plan)",
        "",
        f"Protocol: `cascade_ft_pe_path` · season **{season}** · decision GW{decision_gw} "
        f"→ {start}–{args.horizon_end} · N={args.n} · seed={args.seed}",
        "",
        "**Diagnostic retest under `cascade_joint_path_plan`.** Same path setup as "
        "`cascade_mean_path_gw_resid` (full-side shares, semi-oracle buy, generative "
        "attack, autosub actuals). Planner is one joint multi-GW FT schedule per strand "
        "(not the retired myopic snapshot FT MILP).",
        "",
        "| arm | objective |",
        "| --- | --- |",
        "| control | `V = Σ (XI+cap) − hit_pts` (`ft_pe_nu=0`, γ=1) |",
        (
            f"| candidate | `V + U(f_H)`, `U(b)=ν·b`, **ν={nu:g}** (=HIT_COST): "
            "each stored FT = +4, five = +20 (terminal bank) |"
        ),
        "",
        "Bank dynamics: hold below cap → next deadline bank +1 → PE +4; "
        "at five FTs the weekly grant adds +0 PE unless a transfer is used first. "
        "Retired: concave φ and multi-week W scaling.",
        "",
        f"Wall: {state['elapsed_s']:.1f}s",
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
            f"{ctrl_term:.2f} |"
        ),
        (
            f"| candidate | {cand['transfers']['mean_transfers_per_strand']:.2f} | "
            f"{cand['transfers']['mean_free_used_per_strand']:.2f} | "
            f"{cand['transfers']['mean_paid_per_strand']:.2f} | "
            f"{cand['transfers']['mean_hit_pts_per_strand']:.1f} | "
            f"{cand_term:.2f} |"
        ),
        (
            f"| Δ (cand−ctrl) | {delta['transfers_per_strand']:+.2f} |  | "
            f"{delta['paid_per_strand']:+.2f} | {delta['hit_pts_per_strand']:+.1f} | "
            f"{cand_term - ctrl_term:+.2f} |"
        ),
        "",
        "### Per GW (candidate joint schedule)",
        "",
        "| GW | transfers | free | paid | hit pts | mean ft_after |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cand["transfers"]["per_gw"]:
        lines.append(
            f"| {int(row['gw'])} | {row['mean_transfers']:.2f} | "
            f"{row['mean_free_used']:.2f} | {row['mean_paid']:.2f} | "
            f"{row['mean_hit_pts']:.1f} | {row.get('mean_ft_after', float('nan')):.2f} |"
        )
    lines += [
        "",
        "## Path residual vs actual (oracle minutes + autosub)",
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
        (
            f"| Δ | {delta['path_actual_sum']:+.1f} | "
            f"{delta['path_resid_bias']:+.2f} |  |"
        ),
        "",
        "## Decision",
        "",
    ]
    # Fill decision from observed deltas (diagnostic, not a promote gate).
    if delta["path_actual_sum"] < -5.0 and delta["paid_per_strand"] > -0.5:
        decision = (
            "**`reject_pe_hurts_actual`** — terminal PE did not cut burn enough "
            "and path actual fell."
        )
    elif abs(delta["paid_per_strand"]) < 0.5 and abs(delta["transfers_per_strand"]) < 0.5:
        decision = (
            "**`keep_watching_pe_small`** — under the joint plan, terminal PE "
            f"(ν={nu:g}) barely moves transfer behaviour; path actual "
            f"Δ={delta['path_actual_sum']:+.1f}."
        )
    elif delta["paid_per_strand"] <= -0.5 and delta["path_actual_sum"] >= -5.0:
        decision = (
            "**`keep_watching_pe_helps_burn`** — paid transfers fall without a "
            "clear actual drop; still diagnostic only (not a four-season promote)."
        )
    else:
        decision = (
            "**`keep_watching`** — mixed PE effect under joint plan; diagnostic "
            "only, not a four-season promote."
        )
    lines += [
        decision,
        "",
        "Prior myopic-snapshot PE screen is superseded by this joint-plan retest.",
        "",
    ]
    lines += [
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
        f"PYTHONPATH=. python3 -m src.eval.cascade_ft_pe_path --season {season} --n {args.n}",
        "```",
        "",
    ]
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"delta": delta, "elapsed_s": state["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
