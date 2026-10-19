"""Step 4: chip half-stopping on L4 priced boards vs L4 no-chip.

Arms (priced t:t+2, γ=1, unconstrained hits):
  l4         — no chips (locked baseline)
  chip_stop  — TC/BB leave-one-out DP thresholds; WC≥16 / FH≥12 vs none

Transfers always from L4 ``solve_horizon_transfers`` (none path). Scoring chips
reuse that XI/bench; WC/FH rebuild then score. Promote only if paired Δ clears 0.

  python -m src.strategy.layer4_chip_stop_trial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.rules.fpl_2026 import BUDGET_TENTHS, ChipWallet, advance_ft
from src.strategy.chip_stop import ChipStopPolicy, choose_chip, fit_policy_leave_one_out
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import DEFAULT_MAX_HITS, HeldState, held_from_frame
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_mupost_priced import build_mupost_priced_horizon
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.layer5_beam import (
    _commit_from_week0,
    _rebuild_then_score,
    _with_mu,
)
from src.strategy.layer5_week import (
    NO_CHIP,
    PlannerState,
    advance_planner_state,
    bench_from_squad,
    score_commit_week,
)
from src.strategy.shrink_climb import SeasonStore, _realised

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
N_BOOT = 1000
SEED = 0
TIME_LIMIT = 45.0
GAMMA = 1.0
MAX_HITS = DEFAULT_MAX_HITS

ARM_L4 = "l4"
ARM_STOP = "chip_stop"
ARMS = (ARM_L4, ARM_STOP)


def _prep(store: SeasonStore, gw: int):
    moments = store.moments_asof(gw)
    pool = store.pool(gw)
    work, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    return work, moments


def _solve_hz(pools: list[pd.DataFrame], held: HeldState, *, time_limit: float):
    return solve_horizon_transfers(
        pools,
        held,
        mu_col="mu_post",
        horizon=len(pools),
        gamma=GAMMA,
        terminal_mu_weight=0.0,
        terminal_value_weight=0.0,
        max_hits=MAX_HITS,
        hit_penalty=None,
        time_limit=time_limit,
    )


def _realised_chip(
    xi: pd.DataFrame,
    captain_id: str,
    bench: pd.DataFrame | None,
    chip: str,
    pts_map: pd.Series,
    hit_points: float,
) -> float:
    xi_ids = xi["player_id"].astype(str)
    pts = float(xi_ids.map(pts_map).astype(float).sum())
    cap_pts = float(pts_map.get(str(captain_id), 0.0))
    if chip == "triple_captain":
        pts = pts + 2.0 * cap_pts
    else:
        pts = pts + cap_pts
    if chip == "bench_boost" and bench is not None:
        pts += float(bench["player_id"].astype(str).map(pts_map).fillna(0.0).sum())
    return pts - float(hit_points)


def climb_season(
    store: SeasonStore,
    *,
    arm: str,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    time_limit: float,
    policy: ChipStopPolicy | None = None,
) -> list[dict[str, Any]]:
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    state: PlannerState | None = None
    prev_meta: pd.DataFrame | None = None

    for gw in gws:
        work, moments = _prep(store, gw)

        if state is None:
            splan = solve_week_sigma(
                work,
                mu_col="score_xp_v2",
                se_col="xp_se",
                mode="none",
                moments_by_pos=moments,
                budget=BUDGET_TENTHS,
                time_limit=time_limit,
            )
            squad = splan.plan.squad.copy()
            squad["purchase"] = squad["value"].astype(int)
            bank = int(BUDGET_TENTHS - squad["purchase"].sum())
            held = held_from_frame(squad, bank=max(0, bank), ft=1)
            held = HeldState(
                held.player_ids, held.purchase, held.bank, advance_ft(1, 0)
            )
            state = PlannerState(held=held, wallet=ChipWallet())
            pts_map = work.set_index(work["player_id"].astype(str))["total_points"]
            pts = _realised(splan.plan.xi, splan.plan.captain_id, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "phase": "wildcard",
                    "chip": NO_CHIP,
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "ft_before": held.ft,
                    "pts_net": pts,
                    "choice": "open_wc",
                }
            )
            prev_meta = work.copy()
            print(f"  {store.season} {arm} GW{gw:02d} openWC pts={pts:.0f}", flush=True)
            continue

        assert prev_meta is not None
        work = _ensure_held_rows(work, state.held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)
        ft_before = int(state.held.ft)

        hz = build_mupost_priced_horizon(
            store.season,
            int(gw),
            week0_pool=work,
            moments=moments,
            merged=merged,
            fd=fd,
            prices=prices,
        )
        pools = [_ensure_held_rows(b, state.held, prev_meta) for b in hz.pools]

        hplan = _solve_hz(pools, state.held, time_limit=time_limit)
        week0 = hplan.week0
        xi = week0.xi.copy()
        if "mu" not in xi.columns:
            xi["mu"] = pd.to_numeric(xi.get("mu_post", 0.0), errors="coerce").fillna(0.0)
        squad = week0.squad.copy()
        if "mu" not in squad.columns:
            meta = work.set_index(work["player_id"].astype(str))
            squad["mu"] = (
                squad["player_id"]
                .astype(str)
                .map(meta["mu_post"] if "mu_post" in meta.columns else meta.get("mu", 0.0))
                .astype(float)
            )
        bench = bench_from_squad(squad, xi)
        none_score = score_commit_week(
            xi=xi,
            captain_id=week0.captain_id,
            bench=bench,
            chip=NO_CHIP,
            hit_points=float(week0.hit_points),
            mu_col="mu",
        )
        commit_none = _commit_from_week0(state.held, week0, pools[0])

        chip = NO_CHIP
        choice = "l4"
        n_tx = int(week0.n_transfers)
        hit_pts = int(week0.hit_points)
        commit = commit_none
        xi_play, cap_play, bench_play = xi, str(week0.captain_id), bench
        meta_choice: dict[str, Any] = {}

        if arm == ARM_STOP:
            assert policy is not None
            avail = state.wallet.available(int(gw))
            tc_gain = float(none_score.captain_mu)
            # score_commit_week leaves bench_mu=0 under none — sum explicitly
            bb_gain = float(pd.to_numeric(bench["mu"], errors="coerce").fillna(0.0).sum())

            wc_gain = None
            fh_gain = None
            wc_act = fh_act = None
            if "wildcard" in avail or "free_hit" in avail:
                pool0 = _with_mu(pools[0].copy(), "mu_post")
                if "wildcard" in avail:
                    wc_act = _rebuild_then_score(
                        pool0,
                        state.held,
                        chip="wildcard",
                        moments=moments,
                        time_limit=time_limit,
                    )
                    wc_gain = float(wc_act.week_mu - none_score.objective)
                if "free_hit" in avail:
                    fh_act = _rebuild_then_score(
                        pool0,
                        state.held,
                        chip="free_hit",
                        moments=moments,
                        time_limit=time_limit,
                    )
                    fh_gain = float(fh_act.week_mu - none_score.objective)

            chip, meta_choice = choose_chip(
                gw=int(gw),
                available=avail,
                tc_gain=tc_gain,
                bb_gain=bb_gain,
                wc_gain=wc_gain,
                fh_gain=fh_gain,
                policy=policy,
            )
            choice = chip if chip != NO_CHIP else "none"

            if chip == "wildcard" and wc_act is not None:
                commit = wc_act.commit_held
                xi_play, cap_play, bench_play = wc_act.xi, wc_act.captain_id, wc_act.bench
                n_tx, hit_pts = 0, 0
            elif chip == "free_hit" and fh_act is not None:
                commit = fh_act.commit_held
                xi_play, cap_play, bench_play = fh_act.xi, fh_act.captain_id, fh_act.bench
                n_tx, hit_pts = 0, 0
            elif chip in ("triple_captain", "bench_boost", NO_CHIP):
                commit = commit_none
                # xi/cap/bench already from none path
            else:
                chip = NO_CHIP
                choice = "none"

        net = _realised_chip(
            xi_play, cap_play, bench_play, chip, pts_map, hit_pts
        )
        if arm == ARM_L4:
            # Match prior L4 climbs: ordinary realised (no chip)
            net = _realised(week0.xi, week0.captain_id, pts_map) - float(week0.hit_points)
            chip = NO_CHIP
            choice = "l4"
            n_tx = int(week0.n_transfers)
            hit_pts = int(week0.hit_points)
            commit = commit_none

        trans = advance_planner_state(
            state,
            gw=int(gw),
            chip=chip if arm == ARM_STOP else NO_CHIP,
            commit_held=commit,
            n_transfers=n_tx,
        )
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "phase": "transfer",
                "chip": chip if arm == ARM_STOP else NO_CHIP,
                "n_transfers": n_tx,
                "hits": hit_pts // 4 if hit_pts else 0,
                "hit_points": hit_pts,
                "ft_before": ft_before,
                "pts_net": net,
                "choice": choice,
                **{f"m_{k}": v for k, v in meta_choice.items()},
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} chip={chip} T={n_tx} "
            f"hits={hit_pts} pts={net:.0f}",
            flush=True,
        )
        state = trans.next_state
        prev_meta = pd.concat(
            [work, xi_play], ignore_index=True
        ).drop_duplicates("player_id", keep="last")

    return rows


def _interval(df: pd.DataFrame, arm_a: str, arm_b: str) -> dict[str, Any] | None:
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if arm_a not in block.columns or arm_b not in block.columns:
            continue
        d = (block[arm_a] - block[arm_b]).dropna().to_numpy(float)
        if d.size:
            deltas[season] = d
    if set(deltas) != set(CLOSED_SEASONS):
        return None
    iv = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
    iv["covers_zero"] = bool(iv["lo"] <= 0.0 <= iv["hi"])
    iv["by_season_mean"] = {s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS}
    return iv


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    vs = _interval(df, ARM_STOP, ARM_L4)
    by_season = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        row: dict[str, Any] = {"season": season}
        for arm in ARMS:
            a = sub.loc[sub["arm"] == arm]
            row[f"{arm}_pts"] = float(a["pts_net"].sum())
            row[f"{arm}_hit_pts"] = int(a["hit_points"].sum())
            chips = a["chip"].fillna(NO_CHIP)
            row[f"{arm}_chip_weeks"] = int((chips != NO_CHIP).sum())
        by_season.append(row)
    stop = df.loc[df["arm"] == ARM_STOP]
    chip_counts = (
        stop.loc[stop["chip"] != NO_CHIP, "chip"].value_counts().to_dict()
        if not stop.empty
        else {}
    )
    return {
        "by_season": by_season,
        "total_l4_pts": float(sum(r["l4_pts"] for r in by_season)),
        "total_chip_stop_pts": float(sum(r["chip_stop_pts"] for r in by_season)),
        "total_l4_hit_pts": int(sum(r["l4_hit_pts"] for r in by_season)),
        "total_chip_stop_hit_pts": int(sum(r["chip_stop_hit_pts"] for r in by_season)),
        "paired_vs_l4": {ARM_STOP: vs},
        "chip_counts": chip_counts,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    vs = (doc.get("paired_vs_l4") or {}).get(ARM_STOP)
    lines = [
        "# L4 chip half-stopping vs L4 no-chip",
        "",
        "Priced t:t+2, γ=1. `chip_stop` = TC/BB leave-one-out DP τ(r) + WC≥16 / FH≥12.",
        "",
        "## Pooled",
        "",
        f"| arm | pts | hit pts |",
        f"| --- | ---: | ---: |",
        f"| l4 | {doc['total_l4_pts']:.0f} | {doc['total_l4_hit_pts']} |",
        f"| chip_stop | {doc['total_chip_stop_pts']:.0f} | {doc['total_chip_stop_hit_pts']} |",
        "",
        f"Chip plays: `{doc.get('chip_counts')}`",
        "",
        "## By season",
        "",
        "| season | L4 | stop | L4h | sh | chip weeks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['l4_pts']:.0f} | {r['chip_stop_pts']:.0f} | "
            f"{r['l4_hit_pts']} | {r['chip_stop_hit_pts']} | {r['chip_stop_chip_weeks']} |"
        )
    lines += ["", "## chip_stop − l4", ""]
    if not vs:
        lines += ["- Interval unavailable.", ""]
    else:
        lines += [
            f"- Mean Δ/GW: **{vs['mean']:+.3f}** [{vs['lo']:+.3f}, {vs['hi']:+.3f}]",
            f"- Covers zero: **{vs.get('covers_zero')}**",
            "",
            "| season | mean Δ/GW |",
            "| --- | ---: |",
            *[
                f"| {s} | {m:+.3f} |"
                for s, m in (vs.get("by_season_mean") or {}).items()
            ],
            "",
        ]
    lines += [
        "## Verdict",
        "",
        "Promote only if paired interval clears 0 vs L4. Else park.",
        "",
        "JSON: `data/processed/layer4_chip_stop.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    print("loading …", flush=True)
    v2 = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []
    policies: dict[str, Any] = {}

    for season in seasons:
        print(f"fitting chip-stop leave-out {season} …", flush=True)
        pol = fit_policy_leave_one_out(
            season, gw_start=args.gw_start, gw_end=args.gw_end
        )
        policies[season] = {
            "n_gains": pol.n_gains,
            "mean_tc": pol.mean_tc,
            "mean_bb": pol.mean_bb,
            "tau_tc_r": {str(r): pol.tau_tc[r] for r in (1, 2, 4, 8, 12)},
            "tau_bb_r": {str(r): pol.tau_bb[r] for r in (1, 2, 4, 8, 12)},
        }
        print(
            f"  n={pol.n_gains} mean_tc={pol.mean_tc:.2f} mean_bb={pol.mean_bb:.2f} "
            f"τ_tc={policies[season]['tau_tc_r']} τ_bb={policies[season]['tau_bb_r']}",
            flush=True,
        )
        merged = _load_merged_enriched(season)
        fd = _fd_index(season)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        store = SeasonStore(season, v2)
        for arm in ARMS:
            print(f"climbing {season} arm={arm} …", flush=True)
            all_rows.extend(
                climb_season(
                    store,
                    arm=arm,
                    merged=merged,
                    fd=fd,
                    prices=prices,
                    gw_start=args.gw_start,
                    gw_end=args.gw_end,
                    time_limit=args.time_limit,
                    policy=pol if arm == ARM_STOP else None,
                )
            )

    doc = summarise(all_rows)
    doc["defs"] = {
        "gamma": GAMMA,
        "wc_margin": float(pol.wc_margin) if seasons else 16.0,
        "fh_margin": float(pol.fh_margin) if seasons else 12.0,
        "policies": policies,
    }
    out_json = PROCESSED / "layer4_chip_stop.json"
    out_md = REPORTS / "layer4_chip_stop.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    vs = (doc.get("paired_vs_l4") or {}).get(ARM_STOP) or {}
    print(
        json.dumps(
            {
                "totals": {
                    "l4": doc["total_l4_pts"],
                    "chip_stop": doc["total_chip_stop_pts"],
                },
                "chip_counts": doc.get("chip_counts"),
                "vs_l4": {
                    k: vs.get(k) for k in ("mean", "lo", "hi", "covers_zero")
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
