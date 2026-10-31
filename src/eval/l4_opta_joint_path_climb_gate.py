"""Gated climb: joint path vs L4 oneshot on L4+Opta H=12 boards.

Pre-registered: ``protocol.l4_opta_joint_path_climb``.

Arms (identical L4+Opta centreline μ unless noted):

* baseline ``l4_opta_h12`` — ``solve_horizon_transfers`` (transfers at t=0 only)
* treatment A ``joint_path`` — ``pick_path_transfer_plan`` every-GW; full board;
  ``from_owned``; undecayed
* treatment B ``joint_path_decay`` — same planner; Opta far μ × ρ^(t−near+1) → 0

Shared wildcard open on ``score_xp_v2``; no captaincy (XI − hits); no chips.

  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_climb_gate
  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_climb_gate --smoke
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
from src.rules.fpl_2026 import BUDGET_TENTHS, SQUAD_SIZE, advance_ft
from src.strategy.centreline_path_plan import (
    DEFAULT_DECAY_RHO,
    plan_centreline_path,
)
from src.strategy.decision_value import attach_mu_post
from src.strategy.l3_centreline_horizon import NEAR_DEPTH, build_l3_centreline_horizon
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import HeldState, held_from_frame
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_priced import PROCESSED
from src.strategy.layer4_week import solve_horizon_transfers

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "l4_opta_joint_path_climb_gate.md"
STATE = ROOT / "REPORT_STATE.json"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

BASELINE = "l4_opta_h12"
TREAT_A = "joint_path"
TREAT_B = "joint_path_decay"
ARMS = (BASELINE, TREAT_A, TREAT_B)

GW_START = 8
GW_END = 38
H = 12
N_BOOT = 1000
SEED = 0
MAX_HITS = SQUAD_SIZE
GAMMA = 1.0
TIME_LIMIT_L4 = 60.0
TIME_LIMIT_PATH = 240.0
MIP_REL_GAP = 1e-3
RHO = DEFAULT_DECAY_RHO


def _now() -> float:
    return time.time()


def _pts_map(work: pd.DataFrame) -> pd.Series:
    tmp = work[["player_id", "total_points"]].copy()
    tmp["player_id"] = tmp["player_id"].astype(str)
    tmp["total_points"] = pd.to_numeric(tmp["total_points"], errors="coerce").fillna(
        0.0
    )
    return tmp.groupby("player_id", as_index=True)["total_points"].sum()


def _realised_xi_only(xi: pd.DataFrame, pts_map: pd.Series) -> float:
    ids = xi["player_id"].astype(str)
    return float(ids.map(pts_map).fillna(0.0).astype(float).sum())


def _prep_week0(store: Any, gw: int) -> tuple[pd.DataFrame, Any]:
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
    store: Any,
    *,
    arm: str,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    time_limit_l4: float,
    time_limit_path: float,
    rho: float,
    mip_rel_gap: float = MIP_REL_GAP,
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
                time_limit=time_limit_l4,
            )
            squad = splan.plan.squad.copy()
            squad["purchase"] = squad["value"].astype(int)
            bank = int(BUDGET_TENTHS - squad["purchase"].sum())
            held = held_from_frame(squad, bank=max(0, bank), ft=1)
            held = HeldState(
                held.player_ids, held.purchase, held.bank, advance_ft(1, 0)
            )
            pts_map = _pts_map(work)
            pts = _realised_xi_only(splan.plan.xi, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "phase": "wildcard",
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "pts_xi": float(pts),
                    "pts_net": float(pts),
                    "H": 0,
                    "n_candidates": 0,
                }
            )
            prev_meta = work.copy()
            print(
                f"  {store.season} {arm} GW{gw:02d} WC xi={pts:.0f}",
                flush=True,
            )
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = _pts_map(work)

        hz = build_l3_centreline_horizon(
            store.season,
            int(gw),
            week0_pool=work,
            moments=moments,
            merged=merged,
            fd=fd,
            prices=prices,
            horizon=H,
        )
        pools = [_ensure_held_rows(b, held, prev_meta) for b in hz.pools]

        if arm == BASELINE:
            hplan = solve_horizon_transfers(
                pools,
                held,
                mu_col="mu_post",
                horizon=len(pools),
                gamma=GAMMA,
                terminal_mu_weight=0.0,
                terminal_value_weight=0.0,
                max_hits=MAX_HITS,
                time_limit=time_limit_l4,
                value_captain=False,
            )
            week0 = hplan.week0
            horizon_obj = float(hplan.objective)
            n_cands = int(len(pools[0])) if pools else 0
        elif arm in (TREAT_A, TREAT_B):
            week0, raw = plan_centreline_path(
                pools,
                held,
                mu_col="mu_post",
                decay=(arm == TREAT_B),
                near_depth=NEAR_DEPTH,
                rho=float(rho),
                gamma=GAMMA,
                max_hits=MAX_HITS,
                ft_pe_nu=0.0,
                time_limit=time_limit_path,
                mip_rel_gap=float(mip_rel_gap),
                decision_gw=int(gw),
            )
            horizon_obj = float(raw.get("objective", float("nan")))
            n_cands = int(raw.get("n_candidates", 0))
        else:
            raise ValueError(arm)

        pts = _realised_xi_only(week0.xi, pts_map)
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
                "pts_xi": float(pts),
                "pts_net": net,
                "H": int(len(pools)),
                "horizon_obj": horizon_obj,
                "n_candidates": n_cands,
                "sold": list(week0.sold),
                "bought": list(week0.bought),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} H={len(pools)} n={n_cands} "
            f"xi={pts:.0f} net={net:.0f}",
            flush=True,
        )
        held = _step_held(held, week0)
        prev_meta = pd.concat([work, week0.squad], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    xfer = df.loc[df["phase"] == "transfer"].copy()
    pairs = (
        (TREAT_A, BASELINE),
        (TREAT_B, BASELINE),
        (TREAT_B, TREAT_A),
    )
    comparisons: dict[str, Any] = {}
    by_season_all: dict[str, list[dict[str, Any]]] = {}

    for treat, base in pairs:
        piv = xfer.pivot_table(
            index=["season", "gw"], columns="arm", values="pts_net"
        )
        deltas: dict[str, np.ndarray] = {}
        for season in CLOSED_SEASONS:
            if season not in piv.index.get_level_values(0):
                continue
            block = piv.loc[season]
            if treat not in block.columns or base not in block.columns:
                continue
            d = (block[treat] - block[base]).dropna().to_numpy(float)
            if d.size:
                deltas[season] = d
        if set(deltas) != set(CLOSED_SEASONS):
            raise RuntimeError(
                f"{treat}−{base}: missing season deltas; have {sorted(deltas)}"
            )
        for season, arr in deltas.items():
            if arr.size < 20:
                raise RuntimeError(
                    f"{treat}−{base} {season}: fewer than 20 GWs ({arr.size})"
                )
        interval = cluster_interval(
            deltas, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=SEED
        )
        interval["covers_zero"] = bool(interval["lo"] <= 0.0 <= interval["hi"])
        interval["by_season_mean"] = {
            s: float(np.mean(deltas[s])) for s in CLOSED_SEASONS
        }
        lo, hi = float(interval["lo"]), float(interval["hi"])
        if lo > 0:
            verdict = "promote"
        elif hi < 0:
            verdict = "park_clear_lose"
        else:
            verdict = "inconclusive"
        key = comparison_key(treat, base)
        comparisons[key] = {
            **{k: interval[k] for k in ("mean", "lo", "hi", "n_gws")},
            "incomplete_seasons": {},
            "covers_zero": interval["covers_zero"],
            "by_season_mean": interval["by_season_mean"],
            "verdict": verdict,
            "treatment": treat,
            "baseline": base,
        }
        season_rows = []
        for season in CLOSED_SEASONS:
            sub = xfer.loc[xfer["season"] == season]
            b_pts = float(sub.loc[sub["arm"] == base, "pts_net"].sum())
            t_pts = float(sub.loc[sub["arm"] == treat, "pts_net"].sum())
            season_rows.append(
                {
                    "season": season,
                    "n_gw": int(sub.loc[sub["arm"] == base, "gw"].nunique()),
                    "base_pts": b_pts,
                    "treat_pts": t_pts,
                    "d_pts": t_pts - b_pts,
                    "base_hits": int(
                        sub.loc[sub["arm"] == base, "hit_points"].sum()
                    ),
                    "treat_hits": int(
                        sub.loc[sub["arm"] == treat, "hit_points"].sum()
                    ),
                    "base_transfers": int(
                        sub.loc[sub["arm"] == base, "n_transfers"].sum()
                    ),
                    "treat_transfers": int(
                        sub.loc[sub["arm"] == treat, "n_transfers"].sum()
                    ),
                    "mean_d_gw": float(np.mean(deltas[season])),
                }
            )
        by_season_all[key] = season_rows

    return {
        "comparisons": comparisons,
        "by_season": by_season_all,
        "weeks": df.to_dict(orient="records"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=GW_END)
    ap.add_argument("--time-limit-l4", type=float, default=TIME_LIMIT_L4)
    ap.add_argument("--time-limit-path", type=float, default=TIME_LIMIT_PATH)
    ap.add_argument("--mip-rel-gap", type=float, default=MIP_REL_GAP)
    ap.add_argument("--rho", type=float, default=RHO)
    ap.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma list of arms to run",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="single season 2025-26, GW8–10 only; skip gated report",
    )
    ap.add_argument(
        "--rows-out",
        default="",
        help="write raw week rows JSON and exit (no gated report; for parallel seasons)",
    )
    ap.add_argument(
        "--from-rows",
        default="",
        help="comma-separated row-dump JSON paths; merge and write gated report",
    )
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("l4_opta_joint_path_climb") or {}
    if not screen:
        raise RuntimeError(
            "protocol.l4_opta_joint_path_climb missing — pre-register first"
        )

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"unknown arm {a!r}; choose from {ARMS}")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    gw_start = int(args.gw_start)
    gw_end = int(args.gw_end)
    if args.smoke:
        seasons = ["2025-26"]
        gw_start = 8
        gw_end = 10
        print("[l4-opta-path] SMOKE 2025-26 GW8–10", flush=True)

    print("[l4-opta-path] as-of audit…", flush=True)
    audit = run_asof_audit()
    v2_asof = run_score_xp_v2_asof()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    if not v2_asof["passed"]:
        raise RuntimeError(
            "score_xp_v2 as-of failed: " + "; ".join(v2_asof["failures"])
        )
    audit_out = {"passed": True, "failures": [], "score_xp_v2": v2_asof}

    t0 = _now()
    all_rows: list[dict[str, Any]] = []

    if args.from_rows:
        paths = [Path(p.strip()) for p in str(args.from_rows).split(",") if p.strip()]
        if not paths:
            raise RuntimeError("--from-rows needs at least one path")
        for path in paths:
            blob = json.loads(path.read_text(encoding="utf-8"))
            all_rows.extend(blob.get("rows") or [])
        print(
            f"[l4-opta-path] merged {len(all_rows)} rows from {len(paths)} dumps",
            flush=True,
        )
        seasons = sorted({str(r["season"]) for r in all_rows})
        arms = tuple(sorted({str(r["arm"]) for r in all_rows}))
        gw_start = int(min(int(r["gw"]) for r in all_rows))
        gw_end = int(max(int(r["gw"]) for r in all_rows))
    else:
        print("loading closed score_xp_v2 …", flush=True)
        from src.strategy.shrink_climb import SeasonStore

        v2_all = pd.read_csv(CLOSED_V2)
        for season in seasons:
            merged = _load_merged_enriched(season)
            fd = _fd_index(season)
            prices = pd.read_csv(
                PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
                usecols=[
                    "element",
                    "gw",
                    "value",
                    "team_norm",
                    "position",
                    "player_id",
                ],
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
                        gw_start=gw_start,
                        gw_end=gw_end,
                        time_limit_l4=float(args.time_limit_l4),
                        time_limit_path=float(args.time_limit_path),
                        rho=float(args.rho),
                        mip_rel_gap=float(args.mip_rel_gap),
                    )
                )

    elapsed = _now() - t0

    if (args.smoke or args.rows_out) and not args.from_rows:
        df = pd.DataFrame(all_rows)
        print(df.to_string(index=False), flush=True)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out_path = (
            Path(args.rows_out)
            if args.rows_out
            else PROCESSED / "l4_opta_joint_path_climb_smoke.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "smoke": bool(args.smoke),
                    "seasons": seasons,
                    "arms": list(arms),
                    "gw_start": gw_start,
                    "gw_end": gw_end,
                    "elapsed_s": elapsed,
                    "rows": all_rows,
                    "rho": float(args.rho),
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        STATE.write_text(
            json.dumps(
                {
                    "task": "l4_opta_joint_path_climb",
                    "status": "rows_dump" if args.rows_out else "smoke_ok",
                    "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "change": "Joint path climb rows dump / smoke",
                    "elapsed_s": elapsed,
                    "rows_out": str(out_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "smoke": bool(args.smoke),
                    "rows_out": str(out_path),
                    "elapsed_s": elapsed,
                    "n_rows": len(all_rows),
                },
                indent=2,
            )
        )
        return 0

    # Full gate needs all three arms.
    if set(arms) != set(ARMS):
        raise RuntimeError(
            f"full gate requires all arms {ARMS}; got {arms}. Use --smoke for subsets."
        )
    if set(seasons) != set(CLOSED_SEASONS):
        raise RuntimeError(
            f"full gate requires {CLOSED_SEASONS}; got {seasons}"
        )

    doc = summarise(all_rows)
    comparisons = {
        **_certified_protocol_comparison(),
        **doc["comparisons"],
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "l4_opta_joint_path_climb",
    }

    pair_keys = [
        comparison_key(TREAT_A, BASELINE),
        comparison_key(TREAT_B, BASELINE),
        comparison_key(TREAT_B, TREAT_A),
    ]
    verdicts = {k: doc["comparisons"][k]["verdict"] for k in pair_keys}
    doc["defs"] = {
        "protocol": "l4_opta_joint_path_climb",
        "arms": list(ARMS),
        "H": H,
        "gamma": GAMMA,
        "gw_start": gw_start,
        "gw_end": gw_end,
        "max_hits": MAX_HITS,
        "value_captain": False,
        "realised": "XI total_points only",
        "full_board": True,
        "from_owned": True,
        "rho": float(args.rho),
        "near_depth": NEAR_DEPTH,
        "time_limit_l4": float(args.time_limit_l4),
        "time_limit_path": float(args.time_limit_path),
        "elapsed_s": elapsed,
        "verdicts": verdicts,
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_json = PROCESSED / "l4_opta_joint_path_climb_gate.json"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (PROCESSED / "l4_opta_joint_path_climb_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# L4+Opta joint path climb — gated",
        "",
        "Pre-registered: `protocol.l4_opta_joint_path_climb`.",
        "",
        "Planner isolation on locked L4 near / Opta far centreline μ. "
        "**Not** a μ promote.",
        "",
        f"- Boards: `build_l3_centreline_horizon(H={H})` (priced t:t+1 + Opta far).",
        f"- Baseline `{BASELINE}`: `solve_horizon_transfers` (t=0 transfers only).",
        f"- `{TREAT_A}`: joint `pick_path_transfer_plan`, full board, "
        "`from_owned`, undecayed.",
        f"- `{TREAT_B}`: same planner; Opta far μ × ρ^(t−near+1) → 0 "
        f"(ρ={args.rho}, near={NEAR_DEPTH}).",
        f"- Shared wildcard open on `score_xp_v2` @ GW{gw_start}; seed held 15.",
        "- No captaincy (objective + realised XI only); no chips; "
        f"hits uncapped (max_hits={MAX_HITS}, −4/hit); γ={GAMMA}.",
        f"- Metric: paired GW realised XI−hits (transfer weeks). Wall {elapsed:.0f}s.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Results",
        "",
    ]
    labels = {
        comparison_key(TREAT_A, BASELINE): f"`{TREAT_A}` − `{BASELINE}` (planner)",
        comparison_key(TREAT_B, BASELINE): f"`{TREAT_B}` − `{BASELINE}`",
        comparison_key(TREAT_B, TREAT_A): f"`{TREAT_B}` − `{TREAT_A}` (decay)",
    }
    for key in pair_keys:
        iv = doc["comparisons"][key]
        lines += [
            f"### {labels[key]}",
            "",
            f"- Mean Δ pts/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
            f"- Covers 0: **{iv['covers_zero']}**",
            f"- Verdict: **{iv['verdict']}**",
            "",
            "| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | mean Δ/GW |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in doc["by_season"][key]:
            lines.append(
                f"| {r['season']} | {r['n_gw']} | {r['base_pts']:.0f} | "
                f"{r['treat_pts']:.0f} | {r['d_pts']:+.0f} | {r['base_hits']} | "
                f"{r['treat_hits']} | {r['base_transfers']} | {r['treat_transfers']} | "
                f"{r['mean_d_gw']:+.3f} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Positive planner Δ = joint every-GW path beats L4 oneshot on the "
        "same undecayed L4+Opta boards.",
        "- Positive decay Δ vs `joint_path` = damping Opta far xp toward 0 "
        "helps the joint planner.",
        "- Season totals are descriptive; the gate is the paired GW interval.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_climb_gate",
        "```",
        "",
        f"JSON: `{out_json.relative_to(ROOT)}`",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    # Patch protocol status with results (caller may also update).
    STATE.write_text(
        json.dumps(
            {
                "task": "l4_opta_joint_path_climb",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated joint path vs L4 on L4+Opta H=12 (+Opta far decay)",
                "verdicts": {
                    k: {
                        "verdict": doc["comparisons"][k]["verdict"],
                        "mean": doc["comparisons"][k]["mean"],
                        "lo": doc["comparisons"][k]["lo"],
                        "hi": doc["comparisons"][k]["hi"],
                    }
                    for k in pair_keys
                },
                "report": str(REPORT.relative_to(ROOT)),
                "elapsed_s": elapsed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verdicts": verdicts,
                "means": {k: doc["comparisons"][k]["mean"] for k in pair_keys},
                "report": str(REPORT),
                "elapsed_s": elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
