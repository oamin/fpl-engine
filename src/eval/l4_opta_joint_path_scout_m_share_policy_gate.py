"""Gated climb: scout-m share recompute vs freeze on joint path.

Pre-registered: ``protocol.l4_opta_joint_path_scout_m_share_policy``.

Likelihood ``score_xp_v2_scout_m_share_policy`` must not clearly lose first.

Arms (wired H=12; ρ=0.85→0; ``value_captain=false``; ``ft_pe_nu=0``; no σ):

* baseline ``share_recompute`` — scout rolling-3 m; sG/sA from scout m
* treatment ``share_freeze`` — scout rolling-3 m; sG/sA from expanding m

Realised (both): autosub XI − hits (no captain bank).

  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_scout_m_share_policy_gate
  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_scout_m_share_policy_gate --smoke
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
from src.eval.l4_opta_joint_path_captain_climb_gate import bank_autosub_captain
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
from src.strategy.layer4_mupost_priced import (
    PROCESSED,
    rebuild_week0_with_scout_share_policy,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "l4_opta_joint_path_scout_m_share_policy_gate.md"
STATE = ROOT / "REPORT_STATE.json"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

BASELINE = "share_recompute"
TREAT = "share_freeze"
ARMS = (BASELINE, TREAT)

GW_START = 8
GW_END = 38
H = 12
N_BOOT = 1000
SEED = 0
MAX_HITS = SQUAD_SIZE
GAMMA = 1.0
TIME_LIMIT_L4 = 60.0
TIME_LIMIT_PATH = 120.0
MIP_REL_GAP = 1e-3
RHO = DEFAULT_DECAY_RHO
SCOUT_WINDOW = 3


def _now() -> float:
    return time.time()


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
    mip_rel_gap: float = MIP_REL_GAP,
) -> list[dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
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
            banked = bank_autosub_captain(
                splan.plan.squad,
                splan.plan.xi,
                captain_id=str(splan.plan.captain_id),
                work=work,
            )
            pts_net = float(banked["pts_xi"])
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "phase": "wildcard",
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "pts_xi": float(banked["pts_xi"]),
                    "cap_extra": 0.0,
                    "pts_net": pts_net,
                    "H": 0,
                    "n_candidates": 0,
                    "share_policy": arm,
                    "n_autosubs": int(banked["n_autosubs"]),
                    "n_blank_intended": int(banked["n_blank_intended"]),
                }
            )
            prev_meta = work.copy()
            print(
                f"  {store.season} {arm} GW{gw:02d} WC xi={banked['pts_xi']:.0f}",
                flush=True,
            )
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)

        work_mu, wmeta = rebuild_week0_with_scout_share_policy(
            work,
            season=store.season,
            decision_gw=int(gw),
            merged=merged,
            fd=fd,
            prices=prices,
            moments=moments,
            share_policy=arm,
            minutes_window=SCOUT_WINDOW,
        )
        work_mu = _ensure_held_rows(work_mu, held, prev_meta)
        minutes_map = wmeta.get("minutes_map")
        share_map = wmeta.get("share_minutes_map")
        hz = build_l3_centreline_horizon(
            store.season,
            int(gw),
            week0_pool=work_mu,
            moments=moments,
            merged=merged,
            fd=fd,
            prices=prices,
            horizon=H,
            minutes_by_element=minutes_map,
            share_minutes_by_element=share_map,
        )
        board0 = work_mu

        pools = [_ensure_held_rows(b, held, prev_meta) for b in hz.pools]
        week0, raw = plan_centreline_path(
            pools,
            held,
            mu_col="mu_post",
            decay=True,
            near_depth=NEAR_DEPTH,
            rho=float(RHO),
            decay_target=0.0,
            gamma=GAMMA,
            max_hits=MAX_HITS,
            ft_pe_nu=0.0,
            value_captain=False,
            time_limit=time_limit_path,
            mip_rel_gap=float(mip_rel_gap),
            decision_gw=int(gw),
        )
        horizon_obj = float(raw.get("objective", float("nan")))
        n_cands = int(raw.get("n_candidates", 0))
        banked = bank_autosub_captain(
            week0.squad,
            week0.xi,
            captain_id=str(week0.captain_id),
            work=work,
        )
        net = float(banked["pts_xi"]) - float(week0.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "phase": "transfer",
                "n_transfers": int(week0.n_transfers),
                "hits": int(week0.hits),
                "hit_points": int(week0.hit_points),
                "pts_xi": float(banked["pts_xi"]),
                "cap_extra": 0.0,
                "pts_net": net,
                "H": int(len(pools)),
                "horizon_obj": horizon_obj,
                "n_candidates": n_cands,
                "sold": list(week0.sold),
                "bought": list(week0.bought),
                "share_policy": arm,
                "n_autosubs": int(banked["n_autosubs"]),
                "n_blank_intended": int(banked["n_blank_intended"]),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} xi={banked['pts_xi']:.0f} "
            f"subs={banked['n_autosubs']} net={net:.0f}",
            flush=True,
        )
        held = _step_held(held, week0)
        prev_meta = pd.concat(
            [board0, week0.squad], ignore_index=True
        ).drop_duplicates("player_id", keep="last")
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    xfer = df.loc[df["phase"] == "transfer"].copy()
    comparisons: dict[str, Any] = {}
    by_season_all: dict[str, list[dict[str, Any]]] = {}

    treat, base = TREAT, BASELINE
    piv = xfer.pivot_table(index=["season", "gw"], columns="arm", values="pts_net")
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
        b = sub.loc[sub["arm"] == base]
        t = sub.loc[sub["arm"] == treat]
        season_rows.append(
            {
                "season": season,
                "n_gw": int(b["gw"].nunique()),
                "base_pts": float(b["pts_net"].sum()),
                "treat_pts": float(t["pts_net"].sum()),
                "d_pts": float(t["pts_net"].sum() - b["pts_net"].sum()),
                "base_hits": int(b["hit_points"].sum()),
                "treat_hits": int(t["hit_points"].sum()),
                "base_transfers": int(b["n_transfers"].sum()),
                "treat_transfers": int(t["n_transfers"].sum()),
                "base_autosubs": int(b["n_autosubs"].sum()),
                "treat_autosubs": int(t["n_autosubs"].sum()),
                "base_blanks": int(b["n_blank_intended"].sum()),
                "treat_blanks": int(t["n_blank_intended"].sum()),
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
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rows-out", default="")
    ap.add_argument("--from-rows", default="")
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("l4_opta_joint_path_scout_m_share_policy") or {}
    if not screen:
        raise RuntimeError(
            "protocol.l4_opta_joint_path_scout_m_share_policy missing — "
            "pre-register first"
        )
    like = protocol.get("score_xp_v2_scout_m_share_policy") or {}
    like_status = str(like.get("status") or "")
    if like_status == "parked" or screen.get("status") == "skipped_likelihood_clear_lose":
        raise RuntimeError(
            "likelihood clear lose — climb not authorised "
            f"(score_xp_v2_scout_m_share_policy status={like_status!r})"
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
        print("[scout-m-share-climb] SMOKE 2025-26 GW8–10", flush=True)

    print("[scout-m-share-climb] as-of audit…", flush=True)
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
        for path in paths:
            blob = json.loads(path.read_text(encoding="utf-8"))
            all_rows.extend(blob.get("rows") or [])
        print(
            f"[scout-m-share-climb] merged {len(all_rows)} rows from dumps",
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
            else PROCESSED / "l4_opta_joint_path_scout_m_share_policy_smoke.json"
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
                    "task": "l4_opta_joint_path_scout_m_share_policy",
                    "status": "rows_dump" if args.rows_out else "smoke_ok",
                    "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "change": "scout_m share_policy climb rows / smoke",
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

    if set(arms) != set(ARMS):
        raise RuntimeError(
            f"full gate requires all arms {ARMS}; got {arms}. Use --smoke for subsets."
        )
    if set(seasons) != set(CLOSED_SEASONS):
        raise RuntimeError(f"full gate requires {CLOSED_SEASONS}; got {seasons}")

    doc = summarise(all_rows)
    pair_key = comparison_key(TREAT, BASELINE)
    comparisons = {
        **_certified_protocol_comparison(),
        **doc["comparisons"],
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "l4_opta_joint_path_scout_m_share_policy",
    }
    verdict = doc["comparisons"][pair_key]["verdict"]
    doc["defs"] = {
        "protocol": "l4_opta_joint_path_scout_m_share_policy",
        "arms": list(ARMS),
        "H": H,
        "gamma": GAMMA,
        "rho": RHO,
        "value_captain": False,
        "realised": "autosub XI − hits (no captain bank)",
        "ft_pe_nu": 0.0,
        "scout_window": SCOUT_WINDOW,
        "elapsed_s": elapsed,
        "verdicts": {pair_key: verdict},
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_json = PROCESSED / "l4_opta_joint_path_scout_m_share_policy_gate.json"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (PROCESSED / "l4_opta_joint_path_scout_m_share_policy_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    iv = doc["comparisons"][pair_key]
    lines = [
        "# L4+Opta joint path — scout-m share freeze vs recompute climb",
        "",
        "Pre-registered: `protocol.l4_opta_joint_path_scout_m_share_policy`.",
        "",
        "Authorised by likelihood `score_xp_v2_scout_m_share_policy` (promote).",
        "Layer-1 isolation: `value_captain=false`, no chips, no σ.",
        "",
        f"- Boards: H={H}, ρ=0.85→0, wired μ; scout m = rolling-{SCOUT_WINDOW}.",
        f"- Baseline `{BASELINE}`: sG/sA from scout m.",
        f"- Treatment `{TREAT}`: sG/sA from expanding m; appear/CS/GC from scout m.",
        f"- Realised: autosub XI − hits. Wall {elapsed:.0f}s.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Results",
        "",
        f"### `{TREAT}` − `{BASELINE}`",
        "",
        f"- Mean Δ pts/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
        f"- Covers 0: **{iv['covers_zero']}**",
        f"- Verdict: **{iv['verdict']}**",
        "",
        "| season | n GW | base | treat | Δ | base hits | treat hits | base T | treat T | "
        "base autosubs | treat autosubs | base blanks | treat blanks | mean Δ/GW |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"][pair_key]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['base_pts']:.0f} | "
            f"{r['treat_pts']:.0f} | {r['d_pts']:+.0f} | {r['base_hits']} | "
            f"{r['treat_hits']} | {r['base_transfers']} | {r['treat_transfers']} | "
            f"{r['base_autosubs']} | {r['treat_autosubs']} | "
            f"{r['base_blanks']} | {r['treat_blanks']} | "
            f"{r['mean_d_gw']:+.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Positive Δ = freeze shares beat recompute under scout rolling-3 m.",
        "- Same scout m both arms; only the share law differs.",
        "- No captain bank in realised metric (`value_captain=false`).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_scout_m_share_policy_gate",
        "```",
        "",
        f"JSON: `{out_json.relative_to(ROOT)}`",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    # Update protocol result
    protocol = load_protocol()
    climb = protocol["l4_opta_joint_path_scout_m_share_policy"]
    climb["status"] = (
        "promoted"
        if iv["verdict"] == "promote"
        else ("parked" if iv["verdict"] == "park_clear_lose" else "reported")
    )
    climb["result"] = {
        pair_key: {
            "mean": iv["mean"],
            "lo": iv["lo"],
            "hi": iv["hi"],
            "covers_zero": iv["covers_zero"],
            "verdict": iv["verdict"],
            "by_season_mean": iv["by_season_mean"],
        }
    }
    if iv["verdict"] == "promote":
        climb["note"] = (
            "Promote freeze shares under scout rolling-3 m on joint path."
        )
    elif iv["verdict"] == "park_clear_lose":
        climb["note"] = (
            "Park clear lose. Keep share_recompute under scout m on path."
        )
    else:
        climb["note"] = (
            "Inconclusive climb; likelihood still prefers freeze — do not "
            "change product without a clear path win."
        )
    protocol["l4_opta_joint_path_scout_m_share_policy"] = climb
    (ROOT / "experiments" / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    STATE.write_text(
        json.dumps(
            {
                "task": "l4_opta_joint_path_scout_m_share_policy",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated scout_m share_policy climb on joint path",
                "verdicts": {
                    pair_key: {
                        "verdict": iv["verdict"],
                        "mean": iv["mean"],
                        "lo": iv["lo"],
                        "hi": iv["hi"],
                    }
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
                "verdict": verdict,
                "mean": iv["mean"],
                "lo": iv["lo"],
                "hi": iv["hi"],
                "report": str(REPORT),
                "elapsed_s": elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
