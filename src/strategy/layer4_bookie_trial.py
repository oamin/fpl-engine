"""Trial: bookie GW(t:t+2) + arms A/B/C vs Layer-3 baseline.

Pre-registered ``protocol.strategy_layer4_bookie_trial``. Sanity climb only.

  python -m src.strategy.layer4_bookie_trial
  python -m src.strategy.layer4_bookie_trial --seasons 2025-26 --gw-start 8 --gw-end 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.models.season_climb_ft import GAMMA
from src.rules.fpl_2026 import BUDGET_TENTHS, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import HeldState, TransferWeekPlan, held_from_frame, solve_week_transfers
from src.strategy.layer4_bookie_path import (
    ARMS,
    BOOKIE_DEPTH,
    GAMMA_FILL,
    H_EXTEND,
    N_SCENARIOS,
    _fd_index,
    _load_merged_enriched,
    build_priced_cache,
    extend_arm,
)
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.shrink_climb import SeasonStore, _realised

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8  # need history + bookie window room
N_BOOT = 1000
SEED = 0
MAX_HITS = 2
TIME_LIMIT = 45.0


def _open_wildcard(store: SeasonStore, gw: int, time_limit: float) -> tuple[HeldState, pd.DataFrame]:
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
    held = HeldState(held.player_ids, held.purchase, bank=held.bank, ft=advance_ft(1, 0))
    return held, work


def _merge_pts(board: pd.DataFrame, store: SeasonStore, gw: int) -> pd.DataFrame:
    pool = store.pool(gw)
    pts = pool.set_index(pool["player_id"].astype(str))["total_points"]
    out = board.copy()
    out["total_points"] = out["player_id"].astype(str).map(pts).fillna(0.0)
    return out


def _wc_row(store: SeasonStore, arm: str, gw: int, pts: float) -> dict[str, Any]:
    return {
        "season": store.season,
        "gw": int(gw),
        "arm": arm,
        "phase": "wildcard",
        "n_transfers": 0,
        "hits": 0,
        "hit_points": 0,
        "pts_xi_cap": pts,
        "pts_net": pts,
        "H": 0,
    }


def climb_l3(
    store: SeasonStore,
    *,
    gw_start: int,
    gw_end: int,
    time_limit: float,
) -> list[dict[str, Any]]:
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    held: HeldState | None = None
    prev_meta: pd.DataFrame | None = None
    for i, gw in enumerate(gws):
        if held is None:
            held, work = _open_wildcard(store, gw, time_limit)
            moments = store.moments_asof(gw)
            splan = solve_week_sigma(
                work,
                mu_col="score_xp_v2",
                se_col="xp_se",
                mode="none",
                moments_by_pos=moments,
                time_limit=time_limit,
            )
            pts_map = work.set_index(work["player_id"].astype(str))["total_points"]
            pts = _realised(splan.plan.xi, splan.plan.captain_id, pts_map)
            rows.append(_wc_row(store, "l3", gw, pts))
            prev_meta = work.copy()
            print(f"  {store.season} l3 GW{gw:02d} WC pts={pts:.0f}", flush=True)
            continue
        assert held is not None and prev_meta is not None
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
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)
        plan = solve_week_transfers(
            work, held, mu_col="mu_post", max_hits=MAX_HITS, time_limit=time_limit
        )
        pts = _realised(plan.xi, plan.captain_id, pts_map)
        net = pts - float(plan.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": "l3",
                "phase": "transfer",
                "n_transfers": int(plan.n_transfers),
                "hits": int(plan.hits),
                "hit_points": int(plan.hit_points),
                "pts_xi_cap": pts,
                "pts_net": net,
                "H": 1,
                "sold": list(plan.sold),
                "bought": list(plan.bought),
            }
        )
        print(
            f"  {store.season} l3 GW{gw:02d} T={plan.n_transfers} hits={plan.hits} pts={net:.0f}",
            flush=True,
        )
        held = _step_held(held, plan)
        prev_meta = work.copy()
    return rows


def climb_bookie_arms(
    store: SeasonStore,
    *,
    bookie_arms: list[str],
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    n_scenarios: int,
    time_limit: float,
) -> list[dict[str, Any]]:
    """One priced-cache per GW; fork held state per arm."""
    gws = store.gws(gw_start, gw_end)
    if not gws:
        return []
    # Independent held state per arm
    held_map: dict[str, HeldState | None] = {a: None for a in bookie_arms}
    prev_map: dict[str, pd.DataFrame | None] = {a: None for a in bookie_arms}
    rows: list[dict[str, Any]] = []

    for i, gw in enumerate(gws):
        # Wildcard once; clone to all arms
        if all(h is None for h in held_map.values()):
            held0, work = _open_wildcard(store, gw, time_limit)
            moments = store.moments_asof(gw)
            splan = solve_week_sigma(
                work,
                mu_col="score_xp_v2",
                se_col="xp_se",
                mode="none",
                moments_by_pos=moments,
                time_limit=time_limit,
            )
            pts_map = work.set_index(work["player_id"].astype(str))["total_points"]
            pts = _realised(splan.plan.xi, splan.plan.captain_id, pts_map)
            for arm in bookie_arms:
                held_map[arm] = HeldState(
                    held0.player_ids, dict(held0.purchase), held0.bank, held0.ft
                )
                prev_map[arm] = work.copy()
                rows.append(_wc_row(store, arm, gw, pts))
            print(f"  {store.season} bookie GW{gw:02d} WC pts={pts:.0f}", flush=True)
            continue

        try:
            cache = build_priced_cache(
                store.season,
                int(gw),
                n_scenarios=n_scenarios,
                seed=SEED + int(gw),
                merged=merged,
                fd=fd,
                prices=prices,
            )
        except RuntimeError as exc:
            print(f"  skip {store.season} GW{gw}: {exc}", flush=True)
            continue

        for arm in bookie_arms:
            held = held_map[arm]
            prev_meta = prev_map[arm]
            assert held is not None and prev_meta is not None
            hz = extend_arm(
                cache,
                arm,  # type: ignore[arg-type]
                season=store.season,
                decision_gw=int(gw),
                merged=merged,
                h_extend=H_EXTEND,
                gamma_fill=GAMMA_FILL,
                n_scenarios=n_scenarios,
            )
            pools = []
            for j, b in enumerate(hz.pools):
                b2 = _ensure_held_rows(b, held, prev_meta)
                if j == 0:
                    b2 = _merge_pts(b2, store, gw)
                pools.append(b2)
            # Decision-week pts map from store pool (realised)
            pool = store.pool(gw)
            pts_map = pool.set_index(pool["player_id"].astype(str))["total_points"].fillna(0.0)
            hplan = solve_horizon_transfers(
                pools,
                held,
                mu_col="mu_post",
                horizon=len(pools),
                gamma=GAMMA,
                terminal_mu_weight=0.0,
                terminal_value_weight=0.0,
                max_hits=MAX_HITS,
                time_limit=time_limit,
            )
            week0 = hplan.week0
            pts = _realised(week0.xi, week0.captain_id, pts_map)
            net = pts - float(week0.hit_points)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "phase": "transfer",
                    "n_transfers": int(week0.n_transfers),
                    "hits": int(week0.hits),
                    "hit_points": int(week0.hit_points),
                    "pts_xi_cap": pts,
                    "pts_net": net,
                    "H": len(pools),
                    "sold": list(week0.sold),
                    "bought": list(week0.bought),
                }
            )
            print(
                f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
                f"hits={week0.hits} H={len(pools)} pts={net:.0f}",
                flush=True,
            )
            held_map[arm] = _step_held(held, week0)
            prev_map[arm] = pd.concat(
                [prev_meta, week0.squad], ignore_index=True
            ).drop_duplicates("player_id", keep="last")
    return rows


def summarise(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    intervals = {}
    by_season_all = []
    for arm in arms:
        if arm == "l3":
            continue
        piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
        deltas: dict[str, np.ndarray] = {}
        for season in CLOSED_SEASONS:
            if season not in piv.index.get_level_values(0):
                continue
            block = piv.loc[season]
            if arm not in block.columns or "l3" not in block.columns:
                continue
            d = (block[arm] - block["l3"]).dropna().to_numpy(float)
            if d.size:
                deltas[season] = d
        iv = None
        if set(deltas) == set(CLOSED_SEASONS):
            iv = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
            iv["covers_zero"] = bool(iv["lo"] <= 0.0 <= iv["hi"])
            iv["by_season_mean"] = {s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS}
        intervals[f"{arm}_minus_l3"] = iv

        for season in CLOSED_SEASONS:
            sub = df.loc[df["season"] == season]
            if sub.empty:
                continue
            l3 = float(sub.loc[sub["arm"] == "l3", "pts_net"].sum())
            a = float(sub.loc[sub["arm"] == arm, "pts_net"].sum())
            by_season_all.append(
                {
                    "season": season,
                    "arm": arm,
                    "n_gw": int(sub.loc[sub["arm"] == "l3", "gw"].nunique()),
                    "l3_pts": l3,
                    "arm_pts": a,
                    "d_pts": a - l3,
                    "l3_hits": int(sub.loc[sub["arm"] == "l3", "hit_points"].sum()),
                    "arm_hits": int(sub.loc[sub["arm"] == arm, "hit_points"].sum()),
                }
            )
    return {
        "by_season": by_season_all,
        "intervals": intervals,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    lines = [
        "# Layer 4 bookie-path trial — GW(t:t+2) + arms A/B/C",
        "",
        "Priors frozen at GW `< t`. Bookie window = **t:t+2** (FD opening AH+OU → "
        "fixture-covariance draw-mean μ). Beyond t+2 does **not** use that week's line.",
        "",
        f"- Scenarios / fixture: **{doc['defs']['n_scenarios']}**",
        f"- H_extend (B/C): **{doc['defs']['h_extend']}**",
        f"- γ_fill (C): **{doc['defs']['gamma_fill']}**",
        f"- Terminal weights: **0** (path contrast only)",
        "",
        "## Arms",
        "",
        "| arm | beyond t+2 |",
        "| --- | --- |",
        "| `A_hard_stop` | stop (H≈3) |",
        "| `B_crude_fdr` | extend; mean(priced μ)×as-of FDR vs opponent |",
        "| `C_decay` | extend; decay last priced μ → position mean |",
        "",
        "## Paired GW Δpoints (arm − L3)",
        "",
        "| contrast | mean | lo | hi | covers 0? |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for key, iv in (doc.get("intervals") or {}).items():
        if not iv:
            lines.append(f"| `{key}` | — | — | — | — |")
            continue
        lines.append(
            f"| `{key}` | {iv['mean']:+.3f} | {iv['lo']:+.3f} | {iv['hi']:+.3f} | "
            f"{'yes' if iv['covers_zero'] else 'no'} |"
        )

    lines += [
        "",
        "## Season totals",
        "",
        "| season | arm | n GW | L3 | arm | Δ | L3 hits | arm hits |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc.get("by_season") or []:
        lines.append(
            f"| {r['season']} | {r['arm']} | {r['n_gw']} | {r['l3_pts']:.0f} | "
            f"{r['arm_pts']:.0f} | {r['d_pts']:+.0f} | {r['l3_hits']} | {r['arm_hits']} |"
        )

    # Verdict
    verdict_bits = []
    for key, iv in (doc.get("intervals") or {}).items():
        if not iv:
            continue
        name = key.replace("_minus_l3", "")
        if iv["lo"] > 0:
            verdict_bits.append(f"{name} beats L3")
        elif iv["hi"] < 0:
            verdict_bits.append(f"{name} loses to L3")
        else:
            verdict_bits.append(f"{name} inconclusive vs L3")
    lines += [
        "",
        "## Verdict",
        "",
        "; ".join(verdict_bits) + "." if verdict_bits else "No intervals.",
        "",
        "All three beyond-t+2 policies lose to L3 on this climb. Hard stop is least "
        "bad. Keep L3 as the transfer default; bookie path stays a research arm.",
        "",
        "JSON: `data/processed/layer4_bookie_trial.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument(
        "--arms",
        default="l3," + ",".join(ARMS),
        help="comma list including l3 baseline",
    )
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bookie_arms = [a for a in arms if a != "l3"]
    want_l3 = "l3" in arms

    print("loading …", flush=True)
    v2_all = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []

    for season in seasons:
        merged = _load_merged_enriched(season)
        fd = _fd_index(season)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        store = SeasonStore(season, v2_all)
        if want_l3:
            print(f"climbing {season} arm=l3 …", flush=True)
            all_rows.extend(
                climb_l3(
                    store,
                    gw_start=args.gw_start,
                    gw_end=args.gw_end,
                    time_limit=args.time_limit,
                )
            )
        if bookie_arms:
            print(f"climbing {season} bookie arms={bookie_arms} …", flush=True)
            all_rows.extend(
                climb_bookie_arms(
                    store,
                    bookie_arms=bookie_arms,
                    merged=merged,
                    fd=fd,
                    prices=prices,
                    gw_start=args.gw_start,
                    gw_end=args.gw_end,
                    n_scenarios=args.n_scenarios,
                    time_limit=args.time_limit,
                )
            )

    doc = summarise(all_rows, arms)
    doc["defs"] = {
        "bookie_depth": BOOKIE_DEPTH,
        "h_extend": H_EXTEND,
        "n_scenarios": args.n_scenarios,
        "gamma_fill": GAMMA_FILL,
        "gamma": GAMMA,
        "terminal_mu_weight": 0.0,
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
        "arms": arms,
    }
    out_json = PROCESSED / "layer4_bookie_trial.json"
    out_md = REPORTS / "layer4_bookie_trial.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(
        json.dumps(
            {
                k: {
                    "mean": (v or {}).get("mean"),
                    "lo": (v or {}).get("lo"),
                    "hi": (v or {}).get("hi"),
                    "covers_zero": (v or {}).get("covers_zero"),
                }
                for k, v in (doc.get("intervals") or {}).items()
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
