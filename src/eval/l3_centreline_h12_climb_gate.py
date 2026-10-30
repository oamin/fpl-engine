"""Gated climb: H=12 centreline vs H=2 priced-only L4.

Pre-registered: ``protocol.l3_centreline_horizon_h12_climb``.

Both arms use locked centreline μ (``score_team_side`` / ``mu_post≡score_xp_v2``):

* treatment ``l3_centreline_h12`` — ``build_l3_centreline_horizon(H=12)``
  (FD/priced t:t+1 + Opta blind far)
* baseline ``l3_priced_h2`` — ``build_mupost_priced_horizon`` (BOOKIE_DEPTH=2)

Shared μ_post wildcard open; paired GW realised XI+captain−hits;
``write_gated_report`` + as-of audit.

  PYTHONPATH=. python3 -m src.eval.l3_centreline_h12_climb_gate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    cluster_interval,
    comparison_key,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison
from src.rules.fpl_2026 import BUDGET_TENTHS, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.l3_centreline_horizon import build_l3_centreline_horizon
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import (
    DEFAULT_MAX_HITS,
    HeldState,
    held_from_frame,
)
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_priced import PROCESSED, build_mupost_priced_horizon
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.shrink_climb import SeasonStore, _realised

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "l3_centreline_h12_climb_gate.md"
STATE = ROOT / "REPORT_STATE.json"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

TREATMENT = "l3_centreline_h12"
BASELINE = "l3_priced_h2"
GW_START = 8
GW_END = 38
H_TREAT = 12
N_BOOT = 1000
SEED = 0
MAX_HITS = DEFAULT_MAX_HITS
GAMMA = 1.0
TIME_LIMIT = 60.0


def _now() -> float:
    return time.time()


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
                    "pts_xi_cap": float(pts),
                    "pts_net": float(pts),
                    "H": 0,
                }
            )
            prev_meta = work.copy()
            print(f"  {store.season} {arm} GW{gw:02d} WC pts={pts:.0f}", flush=True)
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(
            0.0
        )

        if arm == BASELINE:
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
        elif arm == TREATMENT:
            hz = build_l3_centreline_horizon(
                store.season,
                int(gw),
                week0_pool=work,
                moments=moments,
                merged=merged,
                fd=fd,
                prices=prices,
                horizon=H_TREAT,
            )
            pools = [_ensure_held_rows(b, held, prev_meta) for b in hz.pools]
        else:
            raise ValueError(arm)

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
        net = float(pts) - float(week0.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "phase": "transfer",
                "n_transfers": int(week0.n_transfers),
                "hits": int(week0.hits),
                "hit_points": int(week0.hit_points),
                "pts_xi_cap": float(pts),
                "pts_net": net,
                "H": int(len(pools)),
                "horizon_obj": float(hplan.objective),
                "sold": list(week0.sold),
                "bought": list(week0.bought),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} H={len(pools)} pts={net:.0f}",
            flush=True,
        )
        held = _step_held(held, week0)
        prev_meta = pd.concat([work, week0.squad], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    # Paired on transfer weeks only (exclude shared wildcard open).
    xfer = df.loc[df["phase"] == "transfer"].copy()
    piv = xfer.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
    deltas: dict[str, np.ndarray] = {}
    for season in CLOSED_SEASONS:
        if season not in piv.index.get_level_values(0):
            continue
        block = piv.loc[season]
        if TREATMENT not in block.columns or BASELINE not in block.columns:
            continue
        d = (block[TREATMENT] - block[BASELINE]).dropna().to_numpy(float)
        if d.size:
            deltas[season] = d
    if set(deltas) != set(CLOSED_SEASONS):
        raise RuntimeError(f"missing season deltas: have {sorted(deltas)}")
    interval = cluster_interval(
        deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED
    )
    interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
    interval["by_season_mean"] = {
        s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS
    }

    by_season = []
    for season in CLOSED_SEASONS:
        sub = xfer.loc[xfer["season"] == season]
        h2 = float(sub.loc[sub["arm"] == BASELINE, "pts_net"].sum())
        h12 = float(sub.loc[sub["arm"] == TREATMENT, "pts_net"].sum())
        by_season.append(
            {
                "season": season,
                "n_gw": int(sub.loc[sub["arm"] == BASELINE, "gw"].nunique()),
                "h2_pts": h2,
                "h12_pts": h12,
                "d_pts": h12 - h2,
                "h2_hits": int(sub.loc[sub["arm"] == BASELINE, "hit_points"].sum()),
                "h12_hits": int(sub.loc[sub["arm"] == TREATMENT, "hit_points"].sum()),
                "h2_transfers": int(
                    sub.loc[sub["arm"] == BASELINE, "n_transfers"].sum()
                ),
                "h12_transfers": int(
                    sub.loc[sub["arm"] == TREATMENT, "n_transfers"].sum()
                ),
                "mean_d_gw": float(np.mean(deltas[season])),
            }
        )
    return {
        "by_season": by_season,
        "total_h2": float(sum(r["h2_pts"] for r in by_season)),
        "total_h12": float(sum(r["h12_pts"] for r in by_season)),
        "total_d": float(sum(r["d_pts"] for r in by_season)),
        "paired_gw_interval": interval,
        "deltas_n_gws": {s: int(deltas[s].size) for s in CLOSED_SEASONS},
        "weeks": df.to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=GW_END)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("l3_centreline_horizon_h12_climb") or {}
    if not screen:
        raise RuntimeError("protocol.l3_centreline_horizon_h12_climb missing")

    print("[h12-gate] as-of audit…", flush=True)
    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    print("loading closed score_xp_v2 …", flush=True)
    v2_all = pd.read_csv(CLOSED_V2)
    t0 = _now()
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
        for arm in (BASELINE, TREATMENT):
            print(f"climbing {season} arm={arm} …", flush=True)
            all_rows.extend(
                climb_season(
                    store,
                    arm=arm,
                    merged=merged,
                    fd=fd,
                    prices=prices,
                    gw_start=int(args.gw_start),
                    gw_end=int(args.gw_end),
                    time_limit=float(args.time_limit),
                )
            )

    doc = summarise(all_rows)
    iv = doc["paired_gw_interval"]
    key = comparison_key(TREATMENT, BASELINE)
    comparisons = {
        **_certified_protocol_comparison(),
        key: {
            **{k: iv[k] for k in ("mean", "lo", "hi", "n_gws")},
            "incomplete_seasons": {},
            "by_season_mean": iv.get("by_season_mean"),
            "covers_zero": iv.get("covers_zero"),
        },
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "l3_centreline_horizon_h12_climb",
    }

    lo = float(iv["lo"])
    hi = float(iv["hi"])
    if lo > 0:
        verdict = "promote"
    elif hi < 0:
        verdict = "park_clear_lose"
    else:
        verdict = "inconclusive"

    elapsed = _now() - t0
    doc["defs"] = {
        "protocol": "l3_centreline_horizon_h12_climb",
        "treatment": TREATMENT,
        "baseline": BASELINE,
        "H_treat": H_TREAT,
        "gamma": GAMMA,
        "gw_start": int(args.gw_start),
        "gw_end": int(args.gw_end),
        "max_hits": MAX_HITS,
        "time_limit": float(args.time_limit),
        "elapsed_s": elapsed,
        "verdict": verdict,
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_json = PROCESSED / "l3_centreline_h12_climb_gate.json"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (PROCESSED / "l3_centreline_h12_climb_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# L3 centreline H=12 climb vs H=2 priced — gated",
        "",
        "Pre-registered: `protocol.l3_centreline_horizon_h12_climb`.",
        "",
        "Strategiser screen on **locked** centreline μ "
        "(`protocol.l3_centreline_forecast`). Not a μ promote.",
        "",
        f"- Treatment `{TREATMENT}`: `build_l3_centreline_horizon(H={H_TREAT})` "
        "(FD/priced t:t+1 + Opta blind far).",
        f"- Baseline `{BASELINE}`: `build_mupost_priced_horizon` (BOOKIE_DEPTH=2).",
        f"- γ={GAMMA}; max_hits={MAX_HITS}; shared μ_post wildcard @ GW{args.gw_start}.",
        f"- Metric: paired GW realised XI+captain−hits (transfer weeks). "
        f"Wall {elapsed:.0f}s.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Result (H=12 − H=2)",
        "",
        f"- Mean Δ pts/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
        f"- Covers 0: **{iv['covers_zero']}**",
        f"- Verdict: **{verdict}**",
        f"- Pooled transfer-week Δ: **{doc['total_d']:+.0f}** "
        f"(H12 {doc['total_h12']:.0f} − H2 {doc['total_h2']:.0f})",
        "",
        "| season | n GW | H=2 | H=12 | Δ | H2 hits | H12 hits | H2 T | H12 T | mean Δ/GW |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['h2_pts']:.0f} | {r['h12_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['h2_hits']} | {r['h12_hits']} | "
            f"{r['h2_transfers']} | {r['h12_transfers']} | {r['mean_d_gw']:+.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Positive Δ = H=12 centreline climb beats H=2 priced-only.",
        "- Opta far uses season-final place ranks (same as locked horizon dig).",
        "- Season totals are descriptive; the gate is the paired GW interval.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.l3_centreline_h12_climb_gate",
        "```",
        "",
        f"JSON: `data/processed/l3_centreline_h12_climb_gate.json`",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    STATE.write_text(
        json.dumps(
            {
                "task": "l3_centreline_horizon_h12_climb",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated H=12 centreline climb vs H=2 priced",
                "verdict": {
                    "verdict": verdict,
                    "mean_d_pts_gw": iv["mean"],
                    "lo": lo,
                    "hi": hi,
                    "covers_zero": iv["covers_zero"],
                    "total_d": doc["total_d"],
                },
                "report": str(REPORT.relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "mean": iv["mean"],
                "lo": lo,
                "hi": hi,
                "covers_zero": iv["covers_zero"],
                "total_d": doc["total_d"],
                "report": str(REPORT),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
