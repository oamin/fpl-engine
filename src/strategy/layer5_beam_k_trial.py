"""Pre-registered screen: beam width K vs planned μ and realised climb.

Arms K ∈ {1, 2, 4, 8, 16}. Same open (μ_post WC), γ=1 priced horizon,
unconstrained hits. Nested check: planned Σμ should be non-decreasing in K
on a shared state trajectory.

  python -m src.strategy.layer5_beam_k_trial
  python -m src.strategy.layer5_beam_k_trial --seasons 2025-26 --ks 1,4,8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.gates import CLOSED_SEASONS
from src.rules.fpl_2026 import BUDGET_TENTHS, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import HeldState, held_from_frame
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_mupost_priced import build_mupost_priced_horizon
from src.strategy.layer5_beam import DEFAULT_BEAM_K, run_beam
from src.strategy.layer5_calendar_boards import (
    DEFAULT_CALENDAR_DEPTH,
    build_mupost_calendar_horizon,
)
from src.strategy.layer5_week import NO_CHIP, PlannerState, advance_planner_state
from src.strategy.shrink_climb import SeasonStore, _realised, _stem

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
GW_START = 8
# Screen grid only — production default is DEFAULT_BEAM_K (=8).
DEFAULT_KS = (1, 2, 4, 8, 16)
TIME_LIMIT = 10.0
assert DEFAULT_BEAM_K == 8


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


def _open_state(work: pd.DataFrame, moments: Any, time_limit: float) -> PlannerState:
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
    return PlannerState(held=held)


def _horizon_maps(
    store: SeasonStore,
    gw: int,
    work: pd.DataFrame,
    moments: Any,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    board_mode: str = "priced",
    calendar_depth: int = DEFAULT_CALENDAR_DEPTH,
) -> tuple[list[int], dict[int, pd.DataFrame], dict[int, Any]]:
    """Build as-of μ boards for the beam.

    ``priced`` — L4 t:t+2 hard-stop (default; Step 4/5 screens).
    ``calendar`` — extended depth (option 3); FD when joined else strength fill.
    """
    mode = str(board_mode).strip().lower()
    if mode == "calendar":
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
        gws = list(hz.gws)
        pools = {int(g): p for g, p in zip(gws, hz.pools, strict=True)}
    elif mode == "priced":
        hz = build_mupost_priced_horizon(
            store.season,
            int(gw),
            week0_pool=work,
            moments=moments,
            merged=merged,
            fd=fd,
            prices=prices,
        )
        gws = list(hz.priced_gws)
        pools = {int(g): p for g, p in zip(gws, hz.pools, strict=True)}
    else:
        raise ValueError(f"unknown board_mode={board_mode!r}")
    moments_by = {int(g): moments for g in gws}
    return gws, pools, moments_by


def climb_k(
    store: SeasonStore,
    *,
    K: int,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    time_limit: float,
    board_mode: str = "priced",
    calendar_depth: int = DEFAULT_CALENDAR_DEPTH,
    allowed_chips: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    state: PlannerState | None = None
    prev_meta: pd.DataFrame | None = None

    for i, gw in enumerate(gws):
        work, moments = _prep(store, gw)
        if state is None:
            state = _open_state(work, moments, time_limit)
            pts_map = work.set_index(work["player_id"].astype(str))["total_points"]
            # Score open WC week with hold XI
            frozen = HeldState(
                state.held.player_ids, state.held.purchase, state.held.bank, ft=0
            )
            from src.strategy.layer3_week import solve_week_transfers

            work2 = _ensure_held_rows(work, state.held, work)
            work2 = work2.copy()
            work2["mu"] = work2["mu_post"]
            plan0 = solve_week_transfers(
                work2, frozen, mu_col="mu", max_hits=0, time_limit=time_limit
            )
            pts = _realised(plan0.xi, plan0.captain_id, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "K": K,
                    "phase": "wildcard",
                    "chip": NO_CHIP,
                    "pts_net": pts,
                    "planned_mu": float("nan"),
                    "n_transfers": 0,
                    "hit_points": 0,
                }
            )
            prev_meta = work.copy()
            print(f"  {store.season} K={K} GW{gw:02d} WC pts={pts:.0f}", flush=True)
            continue

        assert prev_meta is not None
        work, moments = _prep(store, gw)
        work = _ensure_held_rows(work, state.held, prev_meta)

        hz_gws, pools, moments_by = _horizon_maps(
            store,
            gw,
            work,
            moments,
            merged,
            fd,
            prices,
            board_mode=board_mode,
            calendar_depth=calendar_depth,
        )
        # ensure held in every pool
        pools = {g: _ensure_held_rows(p, state.held, prev_meta) for g, p in pools.items()}

        plan = run_beam(
            state,
            horizon_gws=hz_gws,
            pools=pools,
            moments_by_gw=moments_by,
            K=K,
            time_limit=time_limit,
            allowed_chips=allowed_chips,
        )
        act = plan.best.first_action
        if act is None:
            raise RuntimeError("beam returned no first_action")

        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(0.0)
        # Realised: XI + captain (+bench if BB); TC multiplier on realised
        xi_ids = act.xi["player_id"].astype(str)
        pts = float(xi_ids.map(pts_map).astype(float).sum())
        cap_pts = float(pts_map.get(str(act.captain_id), 0.0))
        if act.chip == "triple_captain":
            pts = pts + 2.0 * cap_pts  # XI has 1x; need ×3 total
        else:
            pts = pts + cap_pts
        if act.chip == "bench_boost":
            pts += float(act.bench["player_id"].astype(str).map(pts_map).fillna(0.0).sum())
        pts -= float(act.hit_points)

        trans = advance_planner_state(
            state,
            gw=int(gw),
            chip=act.chip,
            commit_held=act.commit_held,
            n_transfers=act.n_transfers,
        )
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "K": K,
                "phase": "beam",
                "chip": act.chip,
                "pts_net": pts,
                "planned_mu": plan.planned_mu,
                "n_transfers": act.n_transfers,
                "hit_points": act.hit_points,
                "path_chips": list(plan.best.path_chips),
                "H": len(hz_gws),
                "board_mode": board_mode,
            }
        )
        print(
            f"  {store.season} K={K} GW{gw:02d} chip={act.chip} "
            f"T={act.n_transfers} planμ={plan.planned_mu:.1f} pts={pts:.0f}",
            flush=True,
        )
        state = trans.next_state
        prev_meta = pd.concat([work, act.xi], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
    return rows


def nested_planned_probe(
    store: SeasonStore,
    *,
    ks: list[int],
    merged: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    time_limit: float,
    max_probe_gws: int = 8,
) -> list[dict[str, Any]]:
    """On K=1 climb states, probe planned μ for each K (nested check)."""
    # Build K=1 trajectory states by climbing, recording state before each beam
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    state: PlannerState | None = None
    prev_meta: pd.DataFrame | None = None
    probed = 0

    for gw in gws:
        work, moments = _prep(store, gw)
        if state is None:
            state = _open_state(work, moments, time_limit)
            prev_meta = work.copy()
            continue

        work = _ensure_held_rows(work, state.held, prev_meta if prev_meta is not None else work)
        hz_gws, pools, moments_by = _horizon_maps(
            store, gw, work, moments, merged, fd, prices
        )
        pools = {
            g: _ensure_held_rows(p, state.held, prev_meta if prev_meta is not None else work)
            for g, p in pools.items()
        }

        planned: dict[int, float] = {}
        for K in ks:
            bp = run_beam(
                state,
                horizon_gws=hz_gws,
                pools=pools,
                moments_by_gw=moments_by,
                K=K,
                time_limit=time_limit,
            )
            planned[K] = bp.planned_mu
        mono = all(
            planned[ks[i]] + 1e-9 >= planned[ks[i - 1]] for i in range(1, len(ks))
        )
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "planned_by_K": planned,
                "monotone_nondec": mono,
            }
        )
        probed += 1

        # Advance with K=1 commit to next state
        bp1 = run_beam(
            state,
            horizon_gws=hz_gws,
            pools=pools,
            moments_by_gw=moments_by,
            K=1,
            time_limit=time_limit,
        )
        act = bp1.best.first_action
        assert act is not None
        trans = advance_planner_state(
            state,
            gw=int(gw),
            chip=act.chip,
            commit_held=act.commit_held,
            n_transfers=act.n_transfers,
        )
        state = trans.next_state
        prev_meta = work.copy()
        if probed >= max_probe_gws:
            break
    return rows


def summarise(realised: list[dict], probes: list[dict], ks: list[int]) -> dict[str, Any]:
    df = pd.DataFrame(realised)
    by_k = []
    for K in ks:
        sub = df.loc[df["K"] == K]
        by_k.append(
            {
                "K": K,
                "total_pts": float(sub["pts_net"].sum()) if len(sub) else float("nan"),
                "n_gw": int(sub["gw"].nunique()) if len(sub) else 0,
                "mean_planned": float(sub["planned_mu"].mean()) if len(sub) else float("nan"),
            }
        )
    # Paired Δ vs K=1 by season
    piv = df.pivot_table(index=["season", "gw"], columns="K", values="pts_net")
    deltas = {}
    for K in ks:
        if K == 1 or K not in piv.columns or 1 not in piv.columns:
            continue
        d = (piv[K] - piv[1]).dropna()
        deltas[str(K)] = {
            "mean": float(d.mean()) if len(d) else float("nan"),
            "total": float(d.sum()) if len(d) else float("nan"),
            "n": int(len(d)),
        }

    probe_df = pd.DataFrame(probes)
    mono_rate = (
        float(probe_df["monotone_nondec"].mean()) if len(probe_df) else float("nan")
    )
    return {
        "ks": ks,
        "by_k": by_k,
        "delta_vs_K1": deltas,
        "nested_probe": {
            "n": len(probes),
            "monotone_rate": mono_rate,
            "rows": probes,
        },
        "weeks": realised,
    }


def write_report(doc: dict, path: Path) -> None:
    lines = [
        "# Layer 5 beam width K screen",
        "",
        "Joint chip+transfer beam on priced μ (γ=1). K ∈ "
        + ", ".join(str(k) for k in doc["ks"])
        + ". Nested probe: planned Σμ vs K on a shared K=1 trajectory. "
        "Realised: full receding climb per K.",
        "",
        "## Nested planned μ (should be non-decreasing in K)",
        "",
        f"- Probe GWs: **{doc['nested_probe']['n']}**",
        f"- Monotone rate: **{doc['nested_probe']['monotone_rate']:.3f}**",
        "",
        "## Realised climb totals by K",
        "",
        "| K | total pts | mean planned μ |",
        "| ---: | ---: | ---: |",
    ]
    for r in doc["by_k"]:
        lines.append(
            f"| {r['K']} | {r['total_pts']:.0f} | {r['mean_planned']:.2f} |"
        )
    lines += ["", "## Realised Δ vs K=1 (pooled GW)", ""]
    for k, d in (doc.get("delta_vs_K1") or {}).items():
        lines.append(f"- K={k}: mean Δ **{d['mean']:+.3f}**/GW · total **{d['total']:+.0f}** (n={d['n']})")
    lines += [
        "",
        "## Verdict",
        "",
    ]
    totals = [r["total_pts"] for r in doc["by_k"] if np.isfinite(r["total_pts"])]
    if len(totals) >= 2 and totals[-1] > totals[0] + 1:
        lines.append(
            "Larger K **raised realised points** vs K=1 on this run — search budget matters."
        )
    elif doc["nested_probe"]["monotone_rate"] >= 0.9:
        lines.append(
            "Planned μ is largely **monotone in K**; realised gain is weak/flat — "
            "boards/horizon likely bind before beam width."
        )
    else:
        lines.append("Mixed — inspect by-season paths; do not fit K on these totals.")
    lines += ["", "JSON: `data/processed/layer5_beam_k.json`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=",".join(CLOSED_SEASONS))
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    ap.add_argument("--gw-start", type=int, default=GW_START)
    ap.add_argument("--gw-end", type=int, default=38)
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument("--probe-gws", type=int, default=8)
    ap.add_argument("--skip-realised", action="store_true")
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    ks = sorted(set(ks))

    print("loading …", flush=True)
    v2 = pd.read_csv(CLOSED_V2)
    realised: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []

    for season in seasons:
        merged = _load_merged_enriched(season)
        fd = _fd_index(season)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{_stem(season)}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        store = SeasonStore(season, v2)

        print(f"nested probe {season} …", flush=True)
        probes.extend(
            nested_planned_probe(
                store,
                ks=ks,
                merged=merged,
                fd=fd,
                prices=prices,
                gw_start=args.gw_start,
                gw_end=args.gw_end,
                time_limit=args.time_limit,
                max_probe_gws=args.probe_gws,
            )
        )

        if args.skip_realised:
            continue
        for K in ks:
            print(f"climbing {season} K={K} …", flush=True)
            realised.extend(
                climb_k(
                    store,
                    K=K,
                    merged=merged,
                    fd=fd,
                    prices=prices,
                    gw_start=args.gw_start,
                    gw_end=args.gw_end,
                    time_limit=args.time_limit,
                )
            )

    doc = summarise(realised, probes, ks)
    doc["defs"] = {
        "gamma": 1.0,
        "ks": ks,
        "gw_start": args.gw_start,
        "gw_end": args.gw_end,
        "probe_gws": args.probe_gws,
        "skip_realised": bool(args.skip_realised),
    }
    out_json = PROCESSED / "layer5_beam_k.json"
    out_md = REPORTS / "layer5_beam_k.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(doc, out_md)
    print(json.dumps({"by_k": doc["by_k"], "delta_vs_K1": doc["delta_vs_K1"], "mono": doc["nested_probe"]["monotone_rate"]}, indent=2))
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
