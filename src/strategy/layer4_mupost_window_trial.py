"""Trial: L4 expanding μ_post vs windowed W=3 / W=5 vs L3.

  python -m src.strategy.layer4_mupost_window_trial
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
from src.strategy.layer3_week import HeldState, held_from_frame, solve_week_transfers
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_window import build_horizon_for_arm
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.shrink_climb import SeasonStore, _realised

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
N_BOOT = 1000
SEED = 0
MAX_HITS = 2
TIME_LIMIT = 45.0

L4_ARMS = ("l4_exp", "l4_w3", "l4_w5")
ALL_ARMS = ("l3",) + L4_ARMS


def _prep_week0(store: SeasonStore, gw: int):
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

    for gw in gws:
        work, moments = _prep_week0(store, gw)

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
                    "pts_xi_cap": pts,
                    "pts_net": pts,
                    "H": 0,
                }
            )
            prev_meta = work.copy()
            print(f"  {store.season} {arm} GW{gw:02d} WC pts={pts:.0f}", flush=True)
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)

        if arm == "l3":
            plan = solve_week_transfers(
                work, held, mu_col="mu_post", max_hits=MAX_HITS, time_limit=time_limit
            )
            week0, H = plan, 1
        else:
            hz = build_horizon_for_arm(
                arm,
                store.season,
                int(gw),
                week0_pool=work,
                moments_expanding=moments,
                v2_season=store.v2,
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
                max_hits=MAX_HITS,
                time_limit=time_limit,
            )
            week0, H = hplan.week0, len(pools)

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
                "H": H,
                "sold": list(week0.sold),
                "bought": list(week0.bought),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} H={H} pts={net:.0f}",
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
    paired = {arm: _interval(df, arm, "l3") for arm in L4_ARMS}
    vs_exp = {arm: _interval(df, arm, "l4_exp") for arm in ("l4_w3", "l4_w5")}
    by_season = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "season": season,
            "n_gw": int(sub.loc[sub["arm"] == "l3", "gw"].nunique()),
            "l3_pts": float(sub.loc[sub["arm"] == "l3", "pts_net"].sum()),
        }
        for arm in L4_ARMS:
            pts = float(sub.loc[sub["arm"] == arm, "pts_net"].sum())
            row[f"{arm}_pts"] = pts
            row[f"{arm}_d"] = pts - row["l3_pts"]
            row[f"{arm}_hits"] = int(sub.loc[sub["arm"] == arm, "hit_points"].sum())
        by_season.append(row)
    totals = {
        "total_l3": float(sum(r["l3_pts"] for r in by_season)),
    }
    for arm in L4_ARMS:
        totals[f"total_{arm}"] = float(sum(r[f"{arm}_pts"] for r in by_season))
        totals[f"total_d_{arm}"] = float(sum(r[f"{arm}_d"] for r in by_season))
    return {
        "by_season": by_season,
        **totals,
        "paired_vs_l3": paired,
        "paired_vs_exp": vs_exp,
        "weeks": df.to_dict(orient="records"),
    }


def _fmt_iv(iv: dict[str, Any] | None) -> list[str]:
    if not iv:
        return ["- Interval unavailable.", ""]
    lines = [
        f"- Mean Δ/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
        f"- Covers zero: **{iv.get('covers_zero')}**",
        "",
        "| season | mean Δ/GW |",
        "| --- | ---: |",
    ]
    for s, m in (iv.get("by_season_mean") or {}).items():
        lines.append(f"| {s} | {m:+.3f} |")
    lines.append("")
    return lines


def _verdict(iv: dict[str, Any] | None, name: str) -> str:
    if not iv:
        return f"{name}: no interval."
    if iv["lo"] > 0:
        return f"{name} **beats** baseline."
    if iv["hi"] < 0:
        return f"{name} **loses** to baseline."
    return f"{name} **inconclusive**."


def write_report(doc: dict[str, Any], path: Path) -> None:
    lines = [
        "# Layer 4 expanding vs windowed μ_post",
        "",
        "Priced path t:t+2. **Expanding** = locked L4 priced (week-t = L3 `mu_post`). "
        "**Window W** = rates/priors/minutes + cohort moments on last W GWs; project "
        "t:t+2 then same constant-K shrink (week-t also windowed projection).",
        "",
        f"- γ = {doc['defs']['gamma']}; terminal = 0; windows = {doc['defs']['windows']}",
        "",
        "## Vs L3",
        "",
    ]
    for arm, label in (
        ("l4_exp", "Expanding − L3"),
        ("l4_w3", "W=3 − L3"),
        ("l4_w5", "W=5 − L3"),
    ):
        lines += [f"### {label}", ""] + _fmt_iv((doc.get("paired_vs_l3") or {}).get(arm))

    lines += ["## Vs expanding L4", ""]
    for arm, label in (("l4_w3", "W=3 − expanding"), ("l4_w5", "W=5 − expanding")):
        lines += [f"### {label}", ""] + _fmt_iv((doc.get("paired_vs_exp") or {}).get(arm))

    lines += [
        "## Season totals",
        "",
        "| season | L3 | exp | Δexp | W3 | ΔW3 | W5 | ΔW5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['l3_pts']:.0f} | {r['l4_exp_pts']:.0f} | "
            f"{r['l4_exp_d']:+.0f} | {r['l4_w3_pts']:.0f} | {r['l4_w3_d']:+.0f} | "
            f"{r['l4_w5_pts']:.0f} | {r['l4_w5_d']:+.0f} |"
        )
    lines += [
        "",
        f"**Pooled Δ vs L3:** exp **{doc['total_d_l4_exp']:+.0f}** · "
        f"W3 **{doc['total_d_l4_w3']:+.0f}** · W5 **{doc['total_d_l4_w5']:+.0f}**",
        "",
        "## Verdict",
        "",
        _verdict((doc.get("paired_vs_l3") or {}).get("l4_exp"), "Expanding vs L3"),
        "",
        _verdict((doc.get("paired_vs_l3") or {}).get("l4_w3"), "W=3 vs L3"),
        "",
        _verdict((doc.get("paired_vs_l3") or {}).get("l4_w5"), "W=5 vs L3"),
        "",
        _verdict((doc.get("paired_vs_exp") or {}).get("l4_w3"), "W=3 vs expanding"),
        "",
        _verdict((doc.get("paired_vs_exp") or {}).get("l4_w5"), "W=5 vs expanding"),
        "",
        "JSON: `data/processed/layer4_mupost_window.json`",
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
        for arm in ALL_ARMS:
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
        "gamma": GAMMA,
        "terminal_mu_weight": 0.0,
        "windows": [3, 5],
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
    }
    out_json = PROCESSED / "layer4_mupost_window.json"
    out_md = REPORTS / "layer4_mupost_window.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(
        json.dumps(
            {
                "totals_d": {
                    "exp": doc["total_d_l4_exp"],
                    "w3": doc["total_d_l4_w3"],
                    "w5": doc["total_d_l4_w5"],
                },
                "vs_l3": {
                    a: {
                        k: ((doc["paired_vs_l3"] or {}).get(a) or {}).get(k)
                        for k in ("mean", "lo", "hi", "covers_zero")
                    }
                    for a in L4_ARMS
                },
                "vs_exp": {
                    a: {
                        k: ((doc["paired_vs_exp"] or {}).get(a) or {}).get(k)
                        for k in ("mean", "lo", "hi", "covers_zero")
                    }
                    for a in ("l4_w3", "l4_w5")
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
