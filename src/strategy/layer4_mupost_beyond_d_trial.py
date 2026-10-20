"""Trial: beyond-t+2 arm D (team-form FDR) vs A hard-stop under locked L4 settings.

Priced window t:t+2 unchanged. D fills t+3/t+4 with mean(priced μ)×team-form FDR
scale (expanding club pts, managers stripped). No FD beyond t+2.

Primary: D − A. Secondary: D − L3, A − L3.
γ=1, unconstrained hits — match locked priced default.

  python -m src.strategy.layer4_mupost_beyond_d_trial
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
from src.strategy.layer3_week import HeldState, held_from_frame, solve_week_transfers
from src.strategy.layer4_bookie_path import H_EXTEND, _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_beyond import (
    TEAM_FORM_FDR_AMP,
    TEAM_FORM_FDR_TAU,
    build_mupost_beyond_horizon,
)
from src.strategy.layer3_week import DEFAULT_MAX_HITS
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.shrink_climb import SeasonStore, _realised

MAX_HITS = DEFAULT_MAX_HITS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
N_BOOT = 1000
SEED = 0
TIME_LIMIT = 45.0
GAMMA = 1.0

ARM_L3 = "l3"
ARM_A = "A_hard_stop"
ARM_D = "D_team_form"
L4_ARMS: tuple[str, ...] = (ARM_A, ARM_D)


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
    h_extend: int,
    gamma: float,
    max_hits: int,
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

        if arm == ARM_L3:
            plan = solve_week_transfers(
                work, held, mu_col="mu_post", max_hits=max_hits, time_limit=time_limit
            )
            week0, H = plan, 1
        else:
            hz = build_mupost_beyond_horizon(
                store.season,
                int(gw),
                arm,  # type: ignore[arg-type]
                week0_pool=work,
                moments=moments,
                merged=merged,
                fd=fd,
                prices=prices,
                h_extend=h_extend,
            )
            pools = [_ensure_held_rows(b, held, prev_meta) for b in hz.pools]
            hplan = solve_horizon_transfers(
                pools,
                held,
                mu_col="mu_post",
                horizon=len(pools),
                gamma=float(gamma),
                terminal_mu_weight=0.0,
                terminal_value_weight=0.0,
                max_hits=max_hits,
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
    vs_l3 = {arm: _interval(df, arm, ARM_L3) for arm in L4_ARMS}
    vs_a = {ARM_D: _interval(df, ARM_D, ARM_A)}
    by_season = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "season": season,
            "n_gw": int(sub.loc[sub["arm"] == ARM_L3, "gw"].nunique()),
            "l3_pts": float(sub.loc[sub["arm"] == ARM_L3, "pts_net"].sum()),
            "l3_hits": int(sub.loc[sub["arm"] == ARM_L3, "hit_points"].sum()),
        }
        for arm in L4_ARMS:
            pts = float(sub.loc[sub["arm"] == arm, "pts_net"].sum())
            row[f"{arm}_pts"] = pts
            row[f"{arm}_d"] = pts - row["l3_pts"]
            row[f"{arm}_hits"] = int(sub.loc[sub["arm"] == arm, "hit_points"].sum())
        by_season.append(row)
    totals = {"total_l3": float(sum(r["l3_pts"] for r in by_season))}
    for arm in L4_ARMS:
        totals[f"total_{arm}"] = float(sum(r[f"{arm}_pts"] for r in by_season))
        totals[f"total_d_{arm}"] = float(sum(r[f"{arm}_d"] for r in by_season))
    return {
        "by_season": by_season,
        **totals,
        "paired_vs_l3": vs_l3,
        "paired_vs_A": vs_a,
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
    vs_l3 = doc.get("paired_vs_l3") or {}
    vs_a = doc.get("paired_vs_A") or {}
    lines = [
        "# Layer 4 priced μ_post — beyond-t+2 arm D (team-form FDR)",
        "",
        "Priced window **t:t+2** = expanding μ_post path (γ=1, unconstrained hits). "
        "Beyond t+2 never uses that week's line.",
        "",
        "| arm | beyond t+2 |",
        "| --- | --- |",
        "| `A_hard_stop` | stop (H≈3) — current default |",
        "| `D_team_form` | H=5; mean(priced μ)×tanh FDR scale; "
        "FDR = expanding club-pts μ_own−μ_opp (managers stripped) |",
        "",
        f"- H_extend = {defs.get('h_extend')}; γ = {defs.get('gamma')}; "
        f"max_hits = {defs.get('max_hits')}; terminal = 0",
        f"- FDR scale: 1 + {defs.get('fdr_amp')}·tanh(FDR/{defs.get('fdr_tau')}), "
        "clipped to [0.55, 1.35]",
        "",
        "## Primary: D − A",
        "",
    ]
    lines += _fmt_iv(vs_a.get(ARM_D))
    lines += [
        "## Vs L3",
        "",
        "### A hard stop − L3",
        "",
    ]
    lines += _fmt_iv(vs_l3.get(ARM_A))
    lines += ["### D team-form − L3", ""] + _fmt_iv(vs_l3.get(ARM_D))
    lines += [
        "## Season totals",
        "",
        "| season | L3 | A | ΔA | D | ΔD |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['l3_pts']:.0f} | {r['A_hard_stop_pts']:.0f} | "
            f"{r['A_hard_stop_d']:+.0f} | {r['D_team_form_pts']:.0f} | "
            f"{r['D_team_form_d']:+.0f} |"
        )
    lines += [
        "",
        f"**Pooled Δ vs L3:** A **{doc['total_d_A_hard_stop']:+.0f}** · "
        f"D **{doc['total_d_D_team_form']:+.0f}**",
        "",
        "## Verdict",
        "",
        _verdict(vs_a.get(ARM_D), "D vs A"),
        "",
        _verdict(vs_l3.get(ARM_A), "A vs L3"),
        "",
        _verdict(vs_l3.get(ARM_D), "D vs L3"),
        "",
        "Promote D only if D−A interval is clear of 0 above. "
        "Otherwise park long-horizon fill; keep A hard-stop.",
        "",
        "JSON: `data/processed/layer4_mupost_beyond_d.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--h-extend", type=int, default=H_EXTEND)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--max-hits", type=int, default=MAX_HITS)
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]

    print("loading …", flush=True)
    v2_all = pd.read_csv(CLOSED_V2)
    all_rows: list[dict[str, Any]] = []
    arms = (ARM_L3,) + L4_ARMS
    for season in seasons:
        merged = _load_merged_enriched(season)
        fd = _fd_index(season)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        store = SeasonStore(season, v2_all)
        for arm in arms:
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
                    h_extend=args.h_extend,
                    gamma=float(args.gamma),
                    max_hits=int(args.max_hits),
                )
            )
    doc = summarise(all_rows)
    doc["defs"] = {
        "h_extend": args.h_extend,
        "gamma": float(args.gamma),
        "terminal_mu_weight": 0.0,
        "bookie_depth": 3,
        "max_hits": int(args.max_hits),
        "fdr_tau": TEAM_FORM_FDR_TAU,
        "fdr_amp": TEAM_FORM_FDR_AMP,
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
        "default_max_hits_ref": DEFAULT_MAX_HITS,
    }
    out_json = PROCESSED / "layer4_mupost_beyond_d.json"
    out_md = REPORTS / "layer4_mupost_beyond_d.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(
        json.dumps(
            {
                "totals_d": {
                    "A": doc["total_d_A_hard_stop"],
                    "D": doc["total_d_D_team_form"],
                },
                "vs_A": {
                    k: ((doc["paired_vs_A"] or {}).get(ARM_D) or {}).get(k)
                    for k in ("mean", "lo", "hi", "covers_zero")
                },
                "vs_l3": {
                    a: {
                        k: ((doc["paired_vs_l3"] or {}).get(a) or {}).get(k)
                        for k in ("mean", "lo", "hi", "covers_zero")
                    }
                    for a in L4_ARMS
                },
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
