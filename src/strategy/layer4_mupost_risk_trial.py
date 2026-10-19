"""Trial: L4 μ_post priced path + risk-seeking mu_dec vs Layer 3.

Mean boards = priced μ_post (t:t+2). MILP maximises mu_dec =
mu_post + λ (w_se z(se_post) + w_cov z(sd_cov)). Cov draws never replace μ.

  python -m src.strategy.layer4_mupost_risk_trial
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
from src.strategy.layer4_mupost_risk import N_SCENARIOS, build_mupost_risk_horizon
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.risk_mu import DEFAULT_LAMBDA, DEFAULT_W_COV, DEFAULT_W_SE, risk_defs
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
ARM_L4 = "l4_risk"


def _prep_week0(store: SeasonStore, gw: int) -> tuple[pd.DataFrame, Any]:
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
    lambda_risk: float,
    w_se: float,
    w_cov: float,
    n_scenarios: int,
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
            hz = build_mupost_risk_horizon(
                store.season,
                int(gw),
                week0_pool=work,
                moments=moments,
                merged=merged,
                fd=fd,
                prices=prices,
                lambda_risk=lambda_risk,
                w_se=w_se,
                w_cov=w_cov,
                n_scenarios=n_scenarios,
            )
            pools = [_ensure_held_rows(b, held, prev_meta) for b in hz.pools]
            hplan = solve_horizon_transfers(
                pools,
                held,
                mu_col="mu_dec",
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


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if ARM_L4 not in block.columns or "l3" not in block.columns:
            continue
        d = (block[ARM_L4] - block["l3"]).dropna().to_numpy(float)
        if d.size:
            deltas[season] = d
    interval = None
    if set(deltas) == set(CLOSED_SEASONS):
        interval = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
        interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
        interval["by_season_mean"] = {s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS}

    by_season = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        l3 = float(sub.loc[sub["arm"] == "l3", "pts_net"].sum())
        l4 = float(sub.loc[sub["arm"] == ARM_L4, "pts_net"].sum())
        by_season.append(
            {
                "season": season,
                "n_gw": int(sub.loc[sub["arm"] == "l3", "gw"].nunique()),
                "l3_pts": l3,
                "l4_pts": l4,
                "d_pts": l4 - l3,
                "l3_hits": int(sub.loc[sub["arm"] == "l3", "hit_points"].sum()),
                "l4_hits": int(sub.loc[sub["arm"] == ARM_L4, "hit_points"].sum()),
            }
        )
    return {
        "by_season": by_season,
        "total_l3": float(sum(r["l3_pts"] for r in by_season)),
        "total_l4": float(sum(r["l4_pts"] for r in by_season)),
        "total_d": float(sum(r["d_pts"] for r in by_season)),
        "paired_gw_interval": interval,
        "weeks": df.to_dict(orient="records"),
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    iv = doc.get("paired_gw_interval") or {}
    defs = doc.get("defs") or {}
    lines = [
        "# Layer 4 μ_post + risk — GW(t:t+2) vs L3",
        "",
        "Week **t** = Layer-3 `mu_post`. Weeks **t+1, t+2** = projected `score_xp_v2` + same shrink.",
        "MILP maximises **`mu_dec = mu_post + λ · risk`** (risk-seeking). "
        "Risk = within-position z of posterior SE (`K·xp_se`) and fixture-covariance draw SD. "
        "Covariance **never** replaces `mu_post`. Hard stop (no beyond-t+2).",
        "",
        f"- λ = **{defs.get('lambda_risk')}**; w_se = {defs.get('w_se')}; "
        f"w_cov = {defs.get('w_cov')}; n_scenarios = {defs.get('n_scenarios')}",
        f"- γ = {defs.get('gamma')}; terminal = 0",
        "",
        "## Paired GW Δpoints (L4_risk − L3)",
        "",
    ]
    if iv:
        lines += [
            f"- Mean Δ/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
            f"- Covers zero: **{iv.get('covers_zero')}**",
            "",
            "| season | mean Δ/GW |",
            "| --- | ---: |",
        ]
        for s, m in (iv.get("by_season_mean") or {}).items():
            lines.append(f"| {s} | {m:+.3f} |")
    else:
        lines += ["- Interval unavailable.", ""]

    lines += [
        "",
        "## Season totals",
        "",
        "| season | n GW | L3 | L4_risk | Δ | L3 hits | L4 hits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['l3_pts']:.0f} | {r['l4_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['l3_hits']} | {r['l4_hits']} |"
        )
    lines += [
        "",
        f"**Pooled:** L3 {doc['total_l3']:.0f} · L4_risk {doc['total_l4']:.0f} · "
        f"Δ **{doc['total_d']:+.0f}**",
        "",
        "## Verdict",
        "",
    ]
    if iv:
        if iv["lo"] > 0:
            lines.append(
                "Risk-seeking L4 **beats** L3 — keep as optional chase posture on priced path."
            )
        elif iv["hi"] < 0:
            lines.append(
                "Risk-seeking L4 **loses** to L3 — park; keep priced μ_post mean-only default."
            )
        else:
            lines.append(
                "Risk-seeking L4 **inconclusive** vs L3 — park as default; optional live shadow."
            )
    else:
        lines.append("No paired interval.")
    lines += ["", "JSON: `data/processed/layer4_mupost_risk.json`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--lambda-risk", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--w-se", type=float, default=DEFAULT_W_SE)
    ap.add_argument("--w-cov", type=float, default=DEFAULT_W_COV)
    ap.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
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
        for arm in ("l3", ARM_L4):
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
                    lambda_risk=args.lambda_risk,
                    w_se=args.w_se,
                    w_cov=args.w_cov,
                    n_scenarios=args.n_scenarios,
                )
            )
    doc = summarise(all_rows)
    doc["defs"] = {
        **risk_defs(
            lambda_risk=args.lambda_risk,
            w_se=args.w_se,
            w_cov=args.w_cov,
            n_scenarios=args.n_scenarios,
        ),
        "gamma": GAMMA,
        "terminal_mu_weight": 0.0,
        "bookie_depth": 3,
        "beyond_t2": False,
        "covariance_role": "risk_sd_only",
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
    }
    out_json = PROCESSED / "layer4_mupost_risk.json"
    out_md = REPORTS / "layer4_mupost_risk.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    iv = doc.get("paired_gw_interval") or {}
    print(
        json.dumps(
            {
                "total_d": doc["total_d"],
                "paired": {
                    "mean": iv.get("mean"),
                    "lo": iv.get("lo"),
                    "hi": iv.get("hi"),
                    "covers_zero": iv.get("covers_zero"),
                }
                if iv
                else None,
            },
            indent=2,
        )
    )
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
