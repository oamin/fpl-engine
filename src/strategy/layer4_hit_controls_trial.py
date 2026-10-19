"""Step 2 controls vs locked L4: hard hit budget + simple hit gate.

Arms (priced t:t+2, γ=1):
  l4      — unconstrained hits (locked baseline)
  budget  — season hit_points ≤ 16 (≤4 hits of −4); then max_hits=0
  gate    — objective hit penalty = HIT_COST + κ (κ=4 → 8); realised still −4/hit

  python -m src.strategy.layer4_hit_controls_trial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.rules.fpl_2026 import BUDGET_TENTHS, HIT_COST, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import DEFAULT_MAX_HITS, HeldState, held_from_frame
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_priced import build_mupost_priced_horizon
from src.strategy.layer4_week import solve_horizon_transfers
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
SEASON_HIT_BUDGET = 16  # 4 × HIT_COST
GATE_KAPPA = 4  # objective penalty = 4 + κ

ARM_L4 = "l4"
ARM_BUDGET = "budget"
ARM_GATE = "gate"
ARMS = (ARM_L4, ARM_BUDGET, ARM_GATE)


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
) -> list[dict[str, Any]]:
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    held: HeldState | None = None
    prev_meta: pd.DataFrame | None = None
    season_hit_pts = 0

    for gw in gws:
        work, moments = _prep(store, gw)

        if held is None:
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
            pts_map = work.set_index(work["player_id"].astype(str))["total_points"]
            pts = _realised(splan.plan.xi, splan.plan.captain_id, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "phase": "wildcard",
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "ft_before": held.ft,
                    "pts_net": pts,
                }
            )
            prev_meta = work.copy()
            print(f"  {store.season} {arm} GW{gw:02d} WC pts={pts:.0f}", flush=True)
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)
        ft_before = int(held.ft)

        if arm == ARM_BUDGET:
            remain = max(0, SEASON_HIT_BUDGET - season_hit_pts)
            max_hits = remain // HIT_COST
            hit_penalty = None
        elif arm == ARM_GATE:
            max_hits = MAX_HITS
            hit_penalty = float(HIT_COST + GATE_KAPPA)
        else:
            max_hits = MAX_HITS
            hit_penalty = None

        hz = build_mupost_priced_horizon(
            store.season,
            int(gw),
            week0_pool=work,
            moments=moments,
            merged=merged,
            fd=fd,
            prices=prices,
        )
        pools = [_ensure_held_rows(b, held, prev_meta) for b in hz.pools]
        hplan = solve_horizon_transfers(
            pools,
            held,
            mu_col="mu_post",
            horizon=len(pools),
            gamma=GAMMA,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=int(max_hits),
            hit_penalty=hit_penalty,
            time_limit=time_limit,
        )
        week0 = hplan.week0
        pts = _realised(week0.xi, week0.captain_id, pts_map)
        net = pts - float(week0.hit_points)
        season_hit_pts += int(week0.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "phase": "transfer",
                "n_transfers": int(week0.n_transfers),
                "hits": int(week0.hits),
                "hit_points": int(week0.hit_points),
                "ft_before": ft_before,
                "pts_net": net,
                "max_hits_allowed": int(max_hits),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} ft={ft_before} pts={net:.0f}",
            flush=True,
        )
        held = _step_held(held, week0)
        prev_meta = pd.concat([work, week0.squad], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
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
    vs_l4 = {a: _interval(df, a, ARM_L4) for a in (ARM_BUDGET, ARM_GATE)}
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
            row[f"{arm}_hit_weeks"] = int((a["hit_points"] > 0).sum())
            row[f"{arm}_mean_ft"] = float(a["ft_before"].mean())
        by_season.append(row)
    totals = {
        f"total_{arm}_pts": float(sum(r[f"{arm}_pts"] for r in by_season)) for arm in ARMS
    }
    totals.update(
        {
            f"total_{arm}_hit_pts": int(sum(r[f"{arm}_hit_pts"] for r in by_season))
            for arm in ARMS
        }
    )
    return {
        "by_season": by_season,
        **totals,
        "paired_vs_l4": vs_l4,
        "weeks": df.to_dict(orient="records"),
    }


def _fmt_iv(iv: dict[str, Any] | None) -> list[str]:
    if not iv:
        return ["- Interval unavailable.", ""]
    return [
        f"- Mean Δ/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
        f"- Covers zero: **{iv.get('covers_zero')}**",
        "",
        "| season | mean Δ/GW |",
        "| --- | ---: |",
        *[f"| {s} | {m:+.3f} |" for s, m in (iv.get("by_season_mean") or {}).items()],
        "",
    ]


def write_report(doc: dict[str, Any], path: Path) -> None:
    defs = doc.get("defs") or {}
    vs = doc.get("paired_vs_l4") or {}
    lines = [
        "# L4 hit controls — budget + gate vs unconstrained",
        "",
        "Priced t:t+2, γ=1. Baseline `l4` = locked unconstrained hits.",
        "",
        f"| arm | rule |",
        f"| --- | --- |",
        f"| `l4` | unconstrained (max_hits={defs.get('max_hits')}) |",
        f"| `budget` | season hit_points ≤ {defs.get('season_hit_budget')} "
        f"(≤{int(defs.get('season_hit_budget', 16)) // HIT_COST} hits) |",
        f"| `gate` | objective hit penalty = {HIT_COST}+{defs.get('gate_kappa')} "
        f"= {HIT_COST + int(defs.get('gate_kappa', 0))}; realised still −{HIT_COST}/hit |",
        "",
        "## Hits + FT (pooled)",
        "",
        "| arm | pts | hit pts |",
        "| --- | ---: | ---: |",
        f"| l4 | {doc['total_l4_pts']:.0f} | {doc['total_l4_hit_pts']} |",
        f"| budget | {doc['total_budget_pts']:.0f} | {doc['total_budget_hit_pts']} |",
        f"| gate | {doc['total_gate_pts']:.0f} | {doc['total_gate_hit_pts']} |",
        "",
        "## By season",
        "",
        "| season | L4 pts | L4 hits | bud pts | bud hits | gate pts | gate hits | mean FT L4 | mean FT bud | mean FT gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['l4_pts']:.0f} | {r['l4_hit_pts']} | "
            f"{r['budget_pts']:.0f} | {r['budget_hit_pts']} | "
            f"{r['gate_pts']:.0f} | {r['gate_hit_pts']} | "
            f"{r['l4_mean_ft']:.2f} | {r['budget_mean_ft']:.2f} | {r['gate_mean_ft']:.2f} |"
        )
    lines += ["", "## Points vs L4", "", "### budget − l4", ""] + _fmt_iv(
        vs.get(ARM_BUDGET)
    )
    lines += ["### gate − l4", ""] + _fmt_iv(vs.get(ARM_GATE))
    lines += [
        "## Verdict",
        "",
        "Controls are baselines for the FT-DP arm — not promote candidates.",
        "Prefer lower hits / higher mean FT without a collapse in points.",
        "",
        "JSON: `data/processed/layer4_hit_controls.json`",
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
    for season in seasons:
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
                )
            )
    doc = summarise(all_rows)
    doc["defs"] = {
        "season_hit_budget": SEASON_HIT_BUDGET,
        "gate_kappa": GATE_KAPPA,
        "max_hits": MAX_HITS,
        "gamma": GAMMA,
        "hit_cost": HIT_COST,
    }
    out_json = PROCESSED / "layer4_hit_controls.json"
    out_md = REPORTS / "layer4_hit_controls.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(
        json.dumps(
            {
                "totals": {
                    a: {
                        "pts": doc[f"total_{a}_pts"],
                        "hit_pts": doc[f"total_{a}_hit_pts"],
                    }
                    for a in ARMS
                },
                "vs_l4": {
                    a: {
                        k: ((doc["paired_vs_l4"] or {}).get(a) or {}).get(k)
                        for k in ("mean", "lo", "hi", "covers_zero")
                    }
                    for a in (ARM_BUDGET, ARM_GATE)
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
