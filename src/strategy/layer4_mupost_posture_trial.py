"""Lockstep climb: L4 adaptive λ from as-of L3−L4 cumulative gap.

Gap uses points through GW t−1 only (no same-week leakage).
Arms: l3, l4_exp (neutral priced), l4_adapt (neutral/chase/consolidate).

  python -m src.strategy.layer4_mupost_posture_trial
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
from src.strategy.layer4_mupost_posture import N_SCENARIOS, build_mupost_posture_horizon
from src.strategy.layer4_mupost_priced import build_mupost_priced_horizon
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.risk_mu import DEFAULT_GAP_SCALE, DEFAULT_LAMBDA_MAX
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

L4_ARMS = ("l4_exp", "l4_adapt")


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


def _open_held(work: pd.DataFrame, moments, time_limit: float) -> tuple[HeldState, float, Any]:
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
    held = HeldState(held.player_ids, held.purchase, held.bank, advance_ft(1, 0))
    pts_map = work.set_index(work["player_id"].astype(str))["total_points"]
    pts = _realised(splan.plan.xi, splan.plan.captain_id, pts_map)
    return held, pts, splan


def climb_season_lockstep(
    store: SeasonStore,
    *,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    time_limit: float,
    lambda_max: float,
    gap_scale: float,
    n_scenarios: int,
    chase_only: bool = False,
) -> list[dict[str, Any]]:
    """One pass: L3, expanding L4, and adaptive L4 share the GW calendar."""
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    held: dict[str, HeldState | None] = {"l3": None, "l4_exp": None, "l4_adapt": None}
    prev: dict[str, pd.DataFrame | None] = {"l3": None, "l4_exp": None, "l4_adapt": None}
    cum = {"l3": 0.0, "l4_exp": 0.0, "l4_adapt": 0.0}

    for gw in gws:
        work, moments = _prep_week0(store, gw)

        if held["l3"] is None:
            # Shared WC from expanding μ_post board (same open for all arms).
            h0, pts, _ = _open_held(work, moments, time_limit)
            for arm in ("l3", "l4_exp", "l4_adapt"):
                held[arm] = h0
                prev[arm] = work.copy()
                cum[arm] = float(pts)
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
                        "gap": 0.0,
                        "lambda_chase": 0.0,
                        "lambda_cons": 0.0,
                        "posture": "wildcard",
                    }
                )
            print(
                f"  {store.season} WC GW{gw:02d} pts={pts:.0f} (shared)",
                flush=True,
            )
            continue

        # --- L3 ---
        w3 = _ensure_held_rows(work, held["l3"], prev["l3"])
        pts_map = w3.set_index(w3["player_id"].astype(str))["total_points"].fillna(0.0)
        plan3 = solve_week_transfers(
            w3, held["l3"], mu_col="mu_post", max_hits=MAX_HITS, time_limit=time_limit
        )
        pts3 = _realised(plan3.xi, plan3.captain_id, pts_map)
        net3 = pts3 - float(plan3.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": "l3",
                "phase": "transfer",
                "n_transfers": int(plan3.n_transfers),
                "hits": int(plan3.hits),
                "hit_points": int(plan3.hit_points),
                "pts_xi_cap": pts3,
                "pts_net": net3,
                "H": 1,
                "gap": 0.0,
                "lambda_chase": 0.0,
                "lambda_cons": 0.0,
                "posture": "l3",
                "sold": list(plan3.sold),
                "bought": list(plan3.bought),
            }
        )

        # --- expanding L4 (neutral) ---
        we = _ensure_held_rows(work, held["l4_exp"], prev["l4_exp"])
        pts_map_e = we.set_index(we["player_id"].astype(str))["total_points"].fillna(0.0)
        hz_e = build_mupost_priced_horizon(
            store.season,
            int(gw),
            week0_pool=we,
            moments=moments,
            merged=merged,
            fd=fd,
            prices=prices,
        )
        pools_e = [_ensure_held_rows(b, held["l4_exp"], prev["l4_exp"]) for b in hz_e.pools]
        hp_e = solve_horizon_transfers(
            pools_e,
            held["l4_exp"],
            mu_col="mu_post",
            horizon=len(pools_e),
            gamma=GAMMA,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=MAX_HITS,
            time_limit=time_limit,
        )
        pts_e = _realised(hp_e.week0.xi, hp_e.week0.captain_id, pts_map_e)
        net_e = pts_e - float(hp_e.week0.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": "l4_exp",
                "phase": "transfer",
                "n_transfers": int(hp_e.week0.n_transfers),
                "hits": int(hp_e.week0.hits),
                "hit_points": int(hp_e.week0.hit_points),
                "pts_xi_cap": pts_e,
                "pts_net": net_e,
                "H": len(pools_e),
                "gap": 0.0,
                "lambda_chase": 0.0,
                "lambda_cons": 0.0,
                "posture": "neutral",
                "sold": list(hp_e.week0.sold),
                "bought": list(hp_e.week0.bought),
            }
        )

        # --- adaptive L4: gap = L3 cum − adapt cum through t−1 ---
        gap = float(cum["l3"] - cum["l4_adapt"])
        wa = _ensure_held_rows(work, held["l4_adapt"], prev["l4_adapt"])
        pts_map_a = wa.set_index(wa["player_id"].astype(str))["total_points"].fillna(0.0)
        hz_a = build_mupost_posture_horizon(
            store.season,
            int(gw),
            week0_pool=wa,
            moments=moments,
            gap=gap,
            merged=merged,
            fd=fd,
            prices=prices,
            lambda_max=lambda_max,
            gap_scale=gap_scale,
            chase_only=chase_only,
            n_scenarios=n_scenarios,
        )
        pools_a = [_ensure_held_rows(b, held["l4_adapt"], prev["l4_adapt"]) for b in hz_a.pools]
        mu_col = "mu_dec" if hz_a.posture != "neutral" else "mu_post"
        # boards always have mu_dec; use it (equals mu_post when neutral)
        if "mu_dec" in pools_a[0].columns:
            mu_col = "mu_dec"
        hp_a = solve_horizon_transfers(
            pools_a,
            held["l4_adapt"],
            mu_col=mu_col,
            horizon=len(pools_a),
            gamma=GAMMA,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=MAX_HITS,
            time_limit=time_limit,
        )
        pts_a = _realised(hp_a.week0.xi, hp_a.week0.captain_id, pts_map_a)
        net_a = pts_a - float(hp_a.week0.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": "l4_adapt",
                "phase": "transfer",
                "n_transfers": int(hp_a.week0.n_transfers),
                "hits": int(hp_a.week0.hits),
                "hit_points": int(hp_a.week0.hit_points),
                "pts_xi_cap": pts_a,
                "pts_net": net_a,
                "H": len(pools_a),
                "gap": gap,
                "lambda_chase": hz_a.lambda_chase,
                "lambda_cons": hz_a.lambda_cons,
                "posture": hz_a.posture,
                "sold": list(hp_a.week0.sold),
                "bought": list(hp_a.week0.bought),
            }
        )
        print(
            f"  {store.season} GW{gw:02d} gap={gap:+.0f} {hz_a.posture} "
            f"λc={hz_a.lambda_chase:.2f} λk={hz_a.lambda_cons:.2f} | "
            f"L3={net3:.0f} exp={net_e:.0f} adapt={net_a:.0f}",
            flush=True,
        )

        # step held + cum (after this GW's realised)
        held["l3"] = _step_held(held["l3"], plan3)
        prev["l3"] = pd.concat([w3, plan3.squad], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
        cum["l3"] += net3

        held["l4_exp"] = _step_held(held["l4_exp"], hp_e.week0)
        prev["l4_exp"] = pd.concat([we, hp_e.week0.squad], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
        cum["l4_exp"] += net_e

        held["l4_adapt"] = _step_held(held["l4_adapt"], hp_a.week0)
        prev["l4_adapt"] = pd.concat(
            [wa, hp_a.week0.squad], ignore_index=True
        ).drop_duplicates("player_id", keep="last")
        cum["l4_adapt"] += net_a

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
    paired = {
        "adapt_vs_l3": _interval(df, "l4_adapt", "l3"),
        "exp_vs_l3": _interval(df, "l4_exp", "l3"),
        "adapt_vs_exp": _interval(df, "l4_adapt", "l4_exp"),
    }
    by_season = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        adapt = sub.loc[sub["arm"] == "l4_adapt"]
        tr = adapt.loc[adapt["phase"] == "transfer"]
        row = {
            "season": season,
            "n_gw": int(sub.loc[sub["arm"] == "l3", "gw"].nunique()),
            "l3_pts": float(sub.loc[sub["arm"] == "l3", "pts_net"].sum()),
            "exp_pts": float(sub.loc[sub["arm"] == "l4_exp", "pts_net"].sum()),
            "adapt_pts": float(adapt["pts_net"].sum()),
            "frac_chase": float((tr["posture"] == "chase").mean()) if len(tr) else 0.0,
            "frac_cons": float((tr["posture"] == "consolidate").mean()) if len(tr) else 0.0,
            "frac_neutral": float((tr["posture"] == "neutral").mean()) if len(tr) else 0.0,
            "mean_abs_gap": float(tr["gap"].abs().mean()) if len(tr) else 0.0,
            "mean_lam_chase": float(tr["lambda_chase"].mean()) if len(tr) else 0.0,
            "mean_lam_cons": float(tr["lambda_cons"].mean()) if len(tr) else 0.0,
        }
        row["exp_d"] = row["exp_pts"] - row["l3_pts"]
        row["adapt_d"] = row["adapt_pts"] - row["l3_pts"]
        row["adapt_minus_exp"] = row["adapt_pts"] - row["exp_pts"]
        by_season.append(row)
    return {
        "by_season": by_season,
        "total_l3": float(sum(r["l3_pts"] for r in by_season)),
        "total_exp": float(sum(r["exp_pts"] for r in by_season)),
        "total_adapt": float(sum(r["adapt_pts"] for r in by_season)),
        "total_d_exp": float(sum(r["exp_d"] for r in by_season)),
        "total_d_adapt": float(sum(r["adapt_d"] for r in by_season)),
        "total_adapt_minus_exp": float(sum(r["adapt_minus_exp"] for r in by_season)),
        "paired": paired,
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
    defs = doc.get("defs") or {}
    paired = doc.get("paired") or {}
    chase_only = bool(defs.get("chase_only"))
    title = (
        "# Layer 4 adaptive chase-only when behind"
        if chase_only
        else "# Layer 4 adaptive posture — λ from L3−L4 cumulative gap"
    )
    schedule = (
        "`λ_chase = λ_max·tanh(max(gap,0)/scale)`; **λ_cons = 0** (neutral when ahead)."
        if chase_only
        else (
            "`λ_chase = λ_max·tanh(max(gap,0)/scale)`, "
            "`λ_cons = λ_max·tanh(max(−gap,0)/scale)`."
        )
    )
    lines = [
        title,
        "",
        "Lockstep climb. At GW **t**, adaptive L4 sees **cumulative points through t−1** "
        "only (`gap = L3_cum − L4_cum`). "
        + schedule
        + " `mu_dec = mu_post + λ_chase·z(up) − λ_cons·z(down)` from cov semi-SDs. "
        "No same-week realised leakage.",
        "",
        f"- λ_max = **{defs.get('lambda_max')}**; gap_scale = **{defs.get('gap_scale')}** pts"
        + ("; **chase_only**" if chase_only else ""),
        f"- n_scenarios = {defs.get('n_scenarios')}; γ = {defs.get('gamma')}",
        "",
        "## Paired GW Δpoints",
        "",
        "### Expanding − L3",
        "",
    ]
    lines += _fmt_iv(paired.get("exp_vs_l3"))
    lines += ["### Adaptive − L3", ""] + _fmt_iv(paired.get("adapt_vs_l3"))
    lines += ["### Adaptive − expanding", ""] + _fmt_iv(paired.get("adapt_vs_exp"))

    lines += [
        "## Season totals + posture mix",
        "",
        "| season | L3 | exp | adapt | Δadapt | chase% | cons% | neut% | mean |gap| |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['l3_pts']:.0f} | {r['exp_pts']:.0f} | {r['adapt_pts']:.0f} | "
            f"{r['adapt_d']:+.0f} | {100*r['frac_chase']:.0f}% | {100*r['frac_cons']:.0f}% | "
            f"{100*r['frac_neutral']:.0f}% | {r['mean_abs_gap']:.1f} |"
        )
    lines += [
        "",
        f"**Pooled:** L3 {doc['total_l3']:.0f} · exp {doc['total_exp']:.0f} "
        f"(Δ **{doc['total_d_exp']:+.0f}**) · adapt {doc['total_adapt']:.0f} "
        f"(Δ **{doc['total_d_adapt']:+.0f}**) · adapt−exp **{doc['total_adapt_minus_exp']:+.0f}**",
        "",
        "## Verdict",
        "",
        _verdict(paired.get("exp_vs_l3"), "Expanding vs L3"),
        "",
        _verdict(paired.get("adapt_vs_l3"), "Adaptive vs L3"),
        "",
        _verdict(paired.get("adapt_vs_exp"), "Adaptive vs expanding"),
        "",
        f"JSON: `data/processed/{'layer4_mupost_chase_behind' if chase_only else 'layer4_mupost_posture'}.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--lambda-max", type=float, default=DEFAULT_LAMBDA_MAX)
    ap.add_argument("--gap-scale", type=float, default=DEFAULT_GAP_SCALE)
    ap.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    ap.add_argument(
        "--chase-only",
        action="store_true",
        help="Chase when behind L3; neutral when ahead (no consolidate)",
    )
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
        mode = "chase-behind" if args.chase_only else "posture"
        print(f"climbing {season} lockstep ({mode}) …", flush=True)
        all_rows.extend(
            climb_season_lockstep(
                store,
                merged=merged,
                fd=fd,
                prices=prices,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
                lambda_max=args.lambda_max,
                gap_scale=args.gap_scale,
                n_scenarios=args.n_scenarios,
                chase_only=args.chase_only,
            )
        )
    doc = summarise(all_rows)
    doc["defs"] = {
        "lambda_max": args.lambda_max,
        "gap_scale": args.gap_scale,
        "chase_only": bool(args.chase_only),
        "n_scenarios": args.n_scenarios,
        "gamma": GAMMA,
        "gap_def": "L3_cum(t-1) - L4_adapt_cum(t-1)",
        "form": "mu_dec = mu_post + λc z(up) - λk z(down)",
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
    }
    stem = "layer4_mupost_chase_behind" if args.chase_only else "layer4_mupost_posture"
    out_json = PROCESSED / f"{stem}.json"
    out_md = REPORTS / f"{stem}.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(
        json.dumps(
            {
                "totals": {
                    "d_exp": doc["total_d_exp"],
                    "d_adapt": doc["total_d_adapt"],
                    "adapt_minus_exp": doc["total_adapt_minus_exp"],
                },
                "paired": {
                    k: {
                        kk: (v or {}).get(kk)
                        for kk in ("mean", "lo", "hi", "covers_zero")
                    }
                    for k, v in (doc.get("paired") or {}).items()
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
