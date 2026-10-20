"""First-order: why does L5 ``no_chips`` lose to L4?

The ablation's ``no_chips`` arm is **not** L4 without chips — it is the L5
**beam** (myopic ``solve_week_transfers`` each depth) on calendar boards.
L4 uses ``solve_horizon_transfers`` on priced H≤3.

2×2 screen (shared μ-WC open, unconstrained hits, γ=1):

  l4_priced      — horizon MILP + priced boards (production L4)
  l4_calendar    — horizon MILP + calendar depth=8
  beam_priced    — L5 beam, chips locked, priced boards
  beam_calendar  — L5 beam, chips locked, calendar depth=8

If ``beam_priced`` ≪ ``l4_priced`` → beam ≠ L4 (search bug / design gap).
If ``l4_calendar`` ≪ ``l4_priced`` → calendar boards hurt even under L4 policy.
If only ``beam_calendar`` loses → interaction.

  python -m src.strategy.layer5_nochips_diagnose
  python -m src.strategy.layer5_nochips_diagnose --seasons 2025-26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS, cluster_interval
from src.rules.fpl_2026 import BUDGET_TENTHS, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import DEFAULT_MAX_HITS, HeldState, held_from_frame
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_priced import build_mupost_priced_horizon
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.layer5_beam import DEFAULT_BEAM_K
from src.strategy.layer5_beam_k_trial import climb_k
from src.strategy.layer5_calendar_boards import (
    DEFAULT_CALENDAR_DEPTH,
    build_mupost_calendar_horizon,
)
from src.strategy.layer5_week import NO_CHIP
from src.strategy.shrink_climb import SeasonStore, _realised, _stem

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
N_BOOT = 1000
SEED = 0
TIME_LIMIT = 10.0
MAX_HITS = DEFAULT_MAX_HITS


def _prep(store: SeasonStore, gw: int) -> tuple[pd.DataFrame, Any]:
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


def climb_l4_boards(
    store: SeasonStore,
    *,
    board_mode: str,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    time_limit: float,
    calendar_depth: int = DEFAULT_CALENDAR_DEPTH,
    arm_name: str = "l4",
) -> list[dict[str, Any]]:
    """L4 horizon-MILP climb on priced or calendar boards."""
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    held: HeldState | None = None
    prev_meta: pd.DataFrame | None = None

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
                    "arm": arm_name,
                    "phase": "wildcard",
                    "pts_net": pts,
                    "hit_points": 0,
                    "n_transfers": 0,
                    "H": 0,
                }
            )
            prev_meta = work.copy()
            print(f"  {store.season} {arm_name} GW{gw:02d} WC pts={pts:.0f}", flush=True)
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)

        if board_mode == "calendar":
            hz = build_mupost_calendar_horizon(
                store.season,
                int(gw),
                week0_pool=work,
                moments=moments,
                depth=int(calendar_depth),
                merged=merged,
                fd=fd,
                prices=prices,
            )
            raw_pools = list(hz.pools)
        else:
            hz = build_mupost_priced_horizon(
                store.season,
                int(gw),
                week0_pool=work,
                moments=moments,
                merged=merged,
                fd=fd,
                prices=prices,
            )
            raw_pools = list(hz.pools)

        pools = [_ensure_held_rows(b, held, prev_meta) for b in raw_pools]
        hplan = solve_horizon_transfers(
            pools,
            held,
            mu_col="mu_post",
            horizon=len(pools),
            gamma=1.0,
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
                "arm": arm_name,
                "phase": "horizon",
                "pts_net": net,
                "hit_points": int(week0.hit_points),
                "n_transfers": int(week0.n_transfers),
                "H": len(pools),
            }
        )
        print(
            f"  {store.season} {arm_name} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} H={len(pools)} pts={net:.0f}",
            flush=True,
        )
        held = _step_held(held, week0)
        prev_meta = pd.concat([work, week0.squad], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
    return rows


def summarise(rows: list[dict[str, Any]], arms: list[str], baseline: str = "l4_priced") -> dict[str, Any]:
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    by_arm: dict[str, Any] = {}
    for arm in arms:
        if arm == baseline:
            continue
        deltas: dict[str, np.ndarray] = {}
        for season in CLOSED_SEASONS:
            if season not in piv.index.get_level_values(0):
                continue
            block = piv.loc[season]
            if arm not in block.columns or baseline not in block.columns:
                continue
            d = (block[arm] - block[baseline]).dropna().to_numpy(float)
            if d.size:
                deltas[season] = d
        interval = None
        if set(deltas) == set(CLOSED_SEASONS):
            interval = cluster_interval(
                deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED
            )
            interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
            interval["by_season_mean"] = {
                s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS
            }
        season_rows = []
        for season in CLOSED_SEASONS:
            sub = df.loc[df["season"] == season]
            if sub.empty or arm not in sub["arm"].values:
                continue
            base = float(sub.loc[sub["arm"] == baseline, "pts_net"].sum())
            a = float(sub.loc[sub["arm"] == arm, "pts_net"].sum())
            season_rows.append(
                {
                    "season": season,
                    "baseline_pts": base,
                    "arm_pts": a,
                    "d_pts": a - base,
                }
            )
        by_arm[arm] = {
            "by_season": season_rows,
            "total_d": float(sum(r["d_pts"] for r in season_rows)),
            "paired_gw_interval": interval,
        }
    # raw season totals for all arms
    totals = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        row = {"season": season}
        for arm in arms:
            row[arm] = float(sub.loc[sub["arm"] == arm, "pts_net"].sum()) if arm in sub["arm"].values else None
        totals.append(row)
    return {"baseline": baseline, "by_arm_vs_baseline": by_arm, "season_totals": totals, "weeks": df.to_dict(orient="records")}


def write_report(doc: dict[str, Any], path: Path, arms: list[str]) -> None:
    base = doc["baseline"]
    lines = [
        "# L5 no_chips first-order diagnose",
        "",
        "Ablation stopped: ``no_chips`` was L5 **beam** (not L4 horizon MILP). "
        "This 2×2 separates **search** (horizon MILP vs beam) from **boards** "
        "(priced H≤3 vs calendar depth=8).",
        "",
        f"Baseline: **`{base}`**. Hits unconstrained; γ=1; shared μ-WC open.",
        "",
        "## Season totals",
        "",
        "| season | " + " | ".join(f"`{a}`" for a in arms) + " |",
        "| --- | " + " | ".join("---:" for _ in arms) + " |",
    ]
    for r in doc["season_totals"]:
        cells = []
        for a in arms:
            v = r.get(a)
            cells.append(f"{v:.0f}" if v is not None else "—")
        lines.append(f"| {r['season']} | " + " | ".join(cells) + " |")

    lines += ["", f"## Δ vs `{base}` (paired GW)", "",
              "| arm | mean Δ/GW | lo | hi | covers 0 | pooled Δ |",
              "| --- | ---: | ---: | ---: | --- | ---: |"]
    for arm in arms:
        if arm == base:
            continue
        block = doc["by_arm_vs_baseline"].get(arm) or {}
        iv = block.get("paired_gw_interval") or {}
        lines.append(
            f"| `{arm}` | {iv.get('mean', float('nan')):+.3f} | "
            f"{iv.get('lo', float('nan')):+.3f} | {iv.get('hi', float('nan')):+.3f} | "
            f"{iv.get('covers_zero', 'n/a')} | {block.get('total_d', float('nan')):+.0f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- `beam_priced` ≪ 0 → beam search is not L4; chip ablation was confounded.",
        "- `l4_calendar` ≪ 0 → longer boards hurt under the **same** L4 policy.",
        "- Only `beam_calendar` bad → interaction; fix search before chips.",
        "",
        "JSON: `data/processed/layer5_nochips_diagnose.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--depth", type=int, default=DEFAULT_CALENDAR_DEPTH)
    ap.add_argument("--beam-k", type=int, default=DEFAULT_BEAM_K)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    arms = ["l4_priced", "l4_calendar", "beam_priced", "beam_calendar"]
    print("loading …", flush=True)
    v2 = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []
    none_only = frozenset({NO_CHIP})

    for season in seasons:
        merged = _load_merged_enriched(season)
        fd = _fd_index(season)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{_stem(season)}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        store = SeasonStore(season, v2)

        for mode, name in (("priced", "l4_priced"), ("calendar", "l4_calendar")):
            print(f"climbing {season} {name} …", flush=True)
            rows = climb_l4_boards(
                store,
                board_mode=mode,
                merged=merged,
                fd=fd,
                prices=prices,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
                calendar_depth=args.depth,
                arm_name=name,
            )
            all_rows.extend(rows)

        for mode, name in (("priced", "beam_priced"), ("calendar", "beam_calendar")):
            print(f"climbing {season} {name} …", flush=True)
            rows = climb_k(
                store,
                K=int(args.beam_k),
                merged=merged,
                fd=fd,
                prices=prices,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
                board_mode=mode,
                calendar_depth=args.depth,
                allowed_chips=none_only,
            )
            for r in rows:
                all_rows.append(
                    {
                        "season": r["season"],
                        "gw": r["gw"],
                        "arm": name,
                        "phase": r.get("phase"),
                        "pts_net": r["pts_net"],
                        "hit_points": r.get("hit_points", 0),
                        "n_transfers": r.get("n_transfers", 0),
                        "H": r.get("H"),
                    }
                )

    doc = summarise(all_rows, arms)
    doc["defs"] = {
        "role": "first-order no_chips confound — not promote",
        "beam_K": int(args.beam_k),
        "calendar_depth": int(args.depth),
        "arms": arms,
        "ablation_aborted": True,
    }
    out_j = PROCESSED / "layer5_nochips_diagnose.json"
    out_m = REPORTS / "layer5_nochips_diagnose.md"
    out_j.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_m, arms)
    brief = {
        a: (doc["by_arm_vs_baseline"].get(a) or {}).get("total_d")
        for a in arms
        if a != "l4_priced"
    }
    print(json.dumps({"pooled_d_vs_l4_priced": brief, "seasons": seasons}, indent=2))
    print(f"wrote {out_m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
