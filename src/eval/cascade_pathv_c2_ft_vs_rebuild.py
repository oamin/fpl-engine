"""Cascade viability C2 — gated path-V: FT path vs weekly free rebuild.

Protocol: ``cascade_pathv_c2_ft_vs_rebuild`` (locked before totals).

Treatment ``ft_path``: open with a free rebuild, then roll
``solve_week_transfers`` (FT bank + hits). Baseline ``weekly_rebuild``:
``solve_week_sigma`` free rebuild every GW (no continuity).

Boards are locked ``score_xp_v2`` μ_post — this screen is the transfer-rule
claim on path-V (PROJECT_OUTLINE), not a generative likelihood arm.

  PYTHONPATH=. python3 -m src.eval.cascade_pathv_c2_ft_vs_rebuild
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
from src.rules.fpl_2026 import BUDGET_TENTHS, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows, _step_held
from src.strategy.layer3_week import (
    DEFAULT_MAX_HITS,
    HeldState,
    held_from_frame,
    solve_week_transfers,
)
from src.strategy.shrink_climb import SeasonStore, _realised

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
INTERVALS_JSON = PROCESSED / "cascade_pathv_c2_intervals.json"
STATE_JSON = PROCESSED / "cascade_pathv_c2" / "state.json"
REPORT_MD = REPORTS / "cascade_pathv_c2_ft_vs_rebuild.md"

PROTOCOL_KEY = "cascade_pathv_c2_ft_vs_rebuild"
ARM_FT = "ft_path"
ARM_REBUILD = "weekly_rebuild"
GW_START = 8
GW_END = 38
N_BOOT = 1000
SEED = 0
MAX_HITS = DEFAULT_MAX_HITS
TIME_LIMIT = 20.0
PATHV_KEY = "ft_path_minus_weekly_rebuild"


def _prep(pool: pd.DataFrame, moments: Any) -> pd.DataFrame:
    work, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    return work


def _free_rebuild(
    work: pd.DataFrame,
    moments: Any,
    *,
    time_limit: float,
) -> tuple[Any, HeldState]:
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
    return splan.plan, held


def climb_ft_path(
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
        moments = store.moments_asof(gw)
        if moments is None and i == 0:
            continue
        pool = store.pool(gw)
        work = _prep(pool, moments)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)

        if held is None:
            plan, held = _free_rebuild(work, moments, time_limit=time_limit)
            pts = _realised(plan.xi, plan.captain_id, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": ARM_FT,
                    "phase": "wildcard",
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "pts_xi_cap": pts,
                    "pts_net": pts,
                    "bank": held.bank,
                    "ft": held.ft,
                }
            )
            held = HeldState(
                held.player_ids, held.purchase, bank=held.bank, ft=advance_ft(1, 0)
            )
            prev_meta = work.copy()
            print(
                f"  {store.season} {ARM_FT} GW{gw:02d} WC pts={pts:.0f}",
                flush=True,
            )
            continue

        work = _ensure_held_rows(work, held, prev_meta if prev_meta is not None else work)
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
                "arm": ARM_FT,
                "phase": "transfer",
                "n_transfers": int(plan.n_transfers),
                "hits": int(plan.hits),
                "hit_points": int(plan.hit_points),
                "pts_xi_cap": pts,
                "pts_net": net,
                "bank": plan.bank_after,
                "ft": plan.ft_after,
                "sold": list(plan.sold),
                "bought": list(plan.bought),
            }
        )
        print(
            f"  {store.season} {ARM_FT} GW{gw:02d} T={plan.n_transfers} "
            f"hits={plan.hits} pts={net:.0f}",
            flush=True,
        )
        held = _step_held(held, plan, work)
        prev_meta = work.copy()
    return rows


def climb_weekly_rebuild(
    store: SeasonStore,
    *,
    gw_start: int,
    gw_end: int,
    time_limit: float,
) -> list[dict[str, Any]]:
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    for i, gw in enumerate(gws):
        moments = store.moments_asof(gw)
        if moments is None and i == 0:
            continue
        pool = store.pool(gw)
        work = _prep(pool, moments)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)
        plan, _held = _free_rebuild(work, moments, time_limit=time_limit)
        pts = _realised(plan.xi, plan.captain_id, pts_map)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": ARM_REBUILD,
                "phase": "rebuild",
                "n_transfers": 0,
                "hits": 0,
                "hit_points": 0,
                "pts_xi_cap": pts,
                "pts_net": pts,
            }
        )
        print(
            f"  {store.season} {ARM_REBUILD} GW{gw:02d} rebuild pts={pts:.0f}",
            flush=True,
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
        if ARM_FT not in block.columns or ARM_REBUILD not in block.columns:
            continue
        d = (block[ARM_FT] - block[ARM_REBUILD]).dropna().to_numpy(float)
        if d.size:
            deltas[season] = d

    interval = None
    if set(deltas) == set(CLOSED_SEASONS):
        interval = cluster_interval(deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED)
        interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
        interval["clearly_loses"] = bool(interval["hi"] < 0.0)
        interval["clearly_wins"] = bool(interval["lo"] > 0.0)
        interval["by_season_mean"] = {s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS}

    by_season: list[dict[str, Any]] = []
    for season in CLOSED_SEASONS:
        sub = df.loc[df["season"] == season]
        if sub.empty:
            continue
        ft = sub.loc[sub["arm"] == ARM_FT]
        rb = sub.loc[sub["arm"] == ARM_REBUILD]
        n_gw = int(ft["gw"].nunique())
        by_season.append(
            {
                "season": season,
                "n_gw": n_gw,
                "ft_pts": float(ft["pts_net"].sum()),
                "rebuild_pts": float(rb["pts_net"].sum()),
                "d_pts": float(ft["pts_net"].sum() - rb["pts_net"].sum()),
                "ft_hits": int(ft["hit_points"].sum()),
                "ft_transfers": int(ft["n_transfers"].sum()),
                "mean_hits_per_gw": float(ft["hits"].mean()) if n_gw else 0.0,
                "mean_ft_bank": float(ft["ft"].mean()) if "ft" in ft.columns and n_gw else float("nan"),
            }
        )

    verdict = "inconclusive"
    if interval is not None:
        if interval["clearly_loses"]:
            verdict = "park_clearly_loses"
        elif interval["clearly_wins"]:
            verdict = "viable_clearly_wins"
        else:
            verdict = "viable_covers_zero"

    return {
        "by_season": by_season,
        "total_ft": float(sum(r["ft_pts"] for r in by_season)),
        "total_rebuild": float(sum(r["rebuild_pts"] for r in by_season)),
        "total_d": float(sum(r["d_pts"] for r in by_season)),
        "total_ft_hits": int(sum(r["ft_hits"] for r in by_season)),
        "paired_gw_interval": interval,
        "verdict": verdict,
        "weeks": df.to_dict(orient="records"),
    }


def _certified_protocol_comparison() -> dict[str, Any]:
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = comparison_key("score_xp", "score_exp_points")
    return {key: certified["comparisons"][key]}


def _pathv_as_comparison(interval: dict[str, Any]) -> dict[str, Any]:
    """Shape path-V interval like a gated comparison row (n_gws required)."""
    return {
        "mean": float(interval["mean"]),
        "lo": float(interval["lo"]),
        "hi": float(interval["hi"]),
        "n_gws": dict(interval["n_gws"]),
        "incomplete_seasons": {},
        "covers_zero": bool(interval.get("covers_zero")),
        "clearly_loses": bool(interval.get("clearly_loses")),
        "clearly_wins": bool(interval.get("clearly_wins")),
        "by_season_mean": dict(interval.get("by_season_mean") or {}),
        "unit": "realised_pts_net_per_gw",
    }


def write_report(
    doc: dict[str, Any],
    *,
    audit: dict[str, Any],
    elapsed_s: float,
    defs: dict[str, Any],
) -> None:
    protocol = load_protocol()
    min_gws = int(protocol["min_gws_per_season"])
    iv = doc.get("paired_gw_interval")
    if not iv:
        raise RuntimeError("refusing report: path-V interval missing for a closed season")

    pathv_row = _pathv_as_comparison(iv)
    comparisons = {
        **_certified_protocol_comparison(),
        PATHV_KEY: pathv_row,
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": min_gws,
        "comparisons": comparisons,
        "screen": PROTOCOL_KEY,
    }

    lines = [
        "# Cascade path-V C2 — FT path vs weekly free rebuild",
        "",
        f"Protocol: `{PROTOCOL_KEY}` (locked {defs.get('locked')}).",
        "",
        "## Claim",
        "",
        "- **Treatment `ft_path`:** free rebuild open, then FT/hit continuity "
        "(`solve_week_transfers` on μ_post). Maps to `xi_plus_captain_ft` path rule.",
        "- **Baseline `weekly_rebuild`:** free rebuild every GW "
        "(`solve_week_sigma`, no continuity). Maps to weekly `xi_plus_captain`.",
        "- **Boards:** locked `score_xp_v2` μ_post (`mode=none`). Transfer-rule "
        "screen — generative boards parked after B1/B2 clearly_lose.",
        "- **Metric:** paired-by-GW realised XI + captain − hits "
        f"(`{PATHV_KEY}`).",
        f"- **Window:** GW{defs['gw_start']}–{defs['gw_end']} · "
        f"seasons {', '.join(CLOSED_SEASONS)} · chips off.",
        "",
        "## Gates",
        "",
        f"- As-of audit: **{'passed' if audit.get('passed') else 'FAILED'}**.",
        "- Paired intervals cover 2022-23…2025-26. 2026-27 holdout excluded.",
        "- Protocol comparison `score_xp` − `score_exp_points` carried unchanged.",
        "",
        f"## Primary — `{PATHV_KEY}`",
        "",
        f"- Mean Δ/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}] "
        f"(cluster bootstrap B={N_BOOT}, seed={SEED}).",
        f"- Covers zero: **{iv.get('covers_zero')}** · clearly loses: "
        f"**{iv.get('clearly_loses')}** · clearly wins: **{iv.get('clearly_wins')}**.",
        f"- **Verdict: `{doc['verdict']}`**",
        "",
        "### By season (net of hits)",
        "",
        "| season | n GW | ft_path | weekly_rebuild | Δ | ft hit pts | mean hits/GW |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['ft_pts']:.0f} | {r['rebuild_pts']:.0f} | "
            f"{r['d_pts']:+.0f} | {r['ft_hits']} | {r['mean_hits_per_gw']:.2f} |"
        )
    lines += [
        "",
        f"**Pooled:** ft_path {doc['total_ft']:.0f} · rebuild {doc['total_rebuild']:.0f} · "
        f"Δ **{doc['total_d']:+.0f}** · ft hit pts **{doc['total_ft_hits']}**.",
        "",
        "| season | mean Δ/GW |",
        "| --- | ---: |",
    ]
    for s, m in (iv.get("by_season_mean") or {}).items():
        lines.append(f"| {s} | {m:+.3f} |")

    proto = comparisons[comparison_key("score_xp", "score_exp_points")]
    lines += [
        "",
        "## Carried protocol comparison — `score_xp` − `score_exp_points`",
        "",
        f"- From `procedure_intervals.json`: **{proto['mean']:+.4f}** "
        f"[{proto['lo']:+.4f}, {proto['hi']:+.4f}].",
        "- Unchanged. Not re-estimated.",
        "",
        "## Diagnostics (not win criteria)",
        "",
        f"- Wall-clock: **{elapsed_s:.1f}s**.",
        f"- Solver time_limit: {defs['time_limit']}s · max_hits={MAX_HITS}.",
        "",
        "## Artefacts",
        "",
        f"- `{INTERVALS_JSON.relative_to(ROOT)}`",
        f"- `{STATE_JSON.relative_to(ROOT)}`",
        f"- `{REPORT_MD.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.cascade_pathv_c2_ft_vs_rebuild",
        "```",
        "",
    ]

    write_gated_report(REPORT_MD, audit, intervals, lines)
    INTERVALS_JSON.write_text(json.dumps(intervals, indent=2) + "\n", encoding="utf-8")
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "task": PROTOCOL_KEY,
        "verdict": doc["verdict"],
        "elapsed_s": elapsed_s,
        "defs": defs,
        "by_season": doc["by_season"],
        "total_ft": doc["total_ft"],
        "total_rebuild": doc["total_rebuild"],
        "total_d": doc["total_d"],
        "total_ft_hits": doc["total_ft_hits"],
        "paired_gw_interval": iv,
        "report": str(REPORT_MD),
        "intervals": str(INTERVALS_JSON),
    }
    # Drop week-level rows from state (large); keep summary only.
    STATE_JSON.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")
    (PROCESSED / "cascade_pathv_c2_weeks.json").write_text(
        json.dumps({"weeks": doc["weeks"]}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=GW_END)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    args = ap.parse_args(argv)

    protocol = load_protocol()
    entry = protocol.get(PROTOCOL_KEY) or {}
    if not entry:
        raise RuntimeError(f"protocol missing {PROTOCOL_KEY!r} — lock before running")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    if set(seasons) != set(CLOSED_SEASONS):
        raise RuntimeError(
            f"gated run requires all CLOSED_SEASONS; got {seasons}"
        )

    audit = run_asof_audit()
    if not audit.get("passed"):
        raise RuntimeError("as-of audit failed: " + "; ".join(audit.get("failures") or []))

    defs = {
        "locked": entry.get("locked"),
        "gw_start": int(args.gw_start),
        "gw_end": int(args.gw_end),
        "time_limit": float(args.time_limit),
        "max_hits": MAX_HITS,
        "boards": entry.get("boards"),
        "seed": SEED,
        "n_boot": N_BOOT,
    }

    print("loading closed score_xp_v2 …", flush=True)
    v2_all = pd.read_csv(CLOSED_V2)
    t0 = time.time()
    all_rows: list[dict[str, Any]] = []
    for season in seasons:
        store = SeasonStore(season, v2_all)
        print(f"climbing {season} arm={ARM_FT} …", flush=True)
        all_rows.extend(
            climb_ft_path(
                store,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
            )
        )
        print(f"climbing {season} arm={ARM_REBUILD} …", flush=True)
        all_rows.extend(
            climb_weekly_rebuild(
                store,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
            )
        )
    elapsed = time.time() - t0
    doc = summarise(all_rows)
    write_report(doc, audit=audit, elapsed_s=elapsed, defs=defs)
    print(
        json.dumps(
            {
                "verdict": doc["verdict"],
                "paired_gw_interval": doc["paired_gw_interval"],
                "by_season": doc["by_season"],
                "total_d": doc["total_d"],
                "elapsed_s": elapsed,
                "report": str(REPORT_MD),
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
