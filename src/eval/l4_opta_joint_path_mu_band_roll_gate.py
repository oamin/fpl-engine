"""Gated climb: μ-band strand roll on locked joint_path_opta_decay + captain.

Pre-registered: ``protocol.l4_opta_joint_path_mu_band_roll``.

Arms (identical wired L4+Opta H=12 boards; Opta far ρ=0.85→0 on μ and xp_se;
value_captain=True; ft_pe_nu=0):

* baseline ``mean`` — centreline μ (product)
* treatment ``mu_band_8`` — N=8 independent rolls ``μ̂=clip(μ+σZ, μ±2σ)``;
  week-0 commit = modal squad among strands

Realised (both arms): FPL bench autosubs + captain/VC − hits.

Secondary: mean ``xp_se`` of XI / bought (risky vs safe load).

  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_band_roll_gate
  PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_band_roll_gate --smoke
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
    DEFAULT_MU_BAND_A,
    DEFAULT_MU_BAND_B,
    DEFAULT_MU_BAND_STRANDS,
    plan_centreline_path,
    plan_centreline_path_mu_band_roll,
)
from src.strategy.decision_value import attach_mu_post
from src.strategy.l3_centreline_horizon import NEAR_DEPTH, build_l3_centreline_horizon
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import HeldState, held_from_frame
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_priced import PROCESSED

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "l4_opta_joint_path_mu_band_roll_gate.md"
STATE = ROOT / "REPORT_STATE.json"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"
ROWS_DIR = PROCESSED / "l4_opta_joint_path_mu_band_rows"

BASELINE = "mean"
TREAT = "mu_band_8"
ARMS = (BASELINE, TREAT)

GW_START = 8
GW_END = 38
H = 12
N_BOOT = 1000
SEED = 0
MAX_HITS = SQUAD_SIZE
GAMMA = 1.0
TIME_LIMIT_L4 = 60.0
TIME_LIMIT_PATH = 180.0
MIP_REL_GAP = 1e-3
RHO = DEFAULT_DECAY_RHO
N_STRANDS = DEFAULT_MU_BAND_STRANDS
BAND_A = DEFAULT_MU_BAND_A
BAND_B = DEFAULT_MU_BAND_B


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


def _mean_se(frame: pd.DataFrame, ids: list[str] | tuple[str, ...]) -> float:
    if not ids or "xp_se" not in frame.columns:
        return float("nan")
    work = frame.copy()
    work["player_id"] = work["player_id"].astype(str)
    by = work.drop_duplicates("player_id", keep="last").set_index("player_id")
    vals = []
    for pid in ids:
        if pid in by.index:
            vals.append(
                float(pd.to_numeric(by.at[pid, "xp_se"], errors="coerce") or 0.0)
            )
    return float(np.mean(vals)) if vals else float("nan")


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
    n_strands: int = N_STRANDS,
    band_a: float = BAND_A,
    band_b: float = BAND_B,
    seed: int = SEED,
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
            xi_ids = [str(p) for p in splan.plan.xi["player_id"].astype(str).tolist()]
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
                    "cap_extra": float(banked["cap_extra"]),
                    "pts_net": float(banked["pts_gross"]),
                    "H": 0,
                    "n_candidates": 0,
                    "n_autosubs": int(banked["n_autosubs"]),
                    "n_blank_intended": int(banked["n_blank_intended"]),
                    "captain_id": banked["captain_id"],
                    "vice_id": banked["vice_id"],
                    "cap_source": banked["cap_source"],
                    "xi_mean_xp_se": _mean_se(work, xi_ids),
                    "bought_mean_xp_se": float("nan"),
                    "modal_squad_count": None,
                    "n_distinct_squads": None,
                }
            )
            prev_meta = work.copy()
            print(
                f"  {store.season} {arm} GW{gw:02d} WC "
                f"xi={banked['pts_xi']:.0f} cap={banked['cap_extra']:.0f}",
                flush=True,
            )
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)

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
                value_captain=True,
                time_limit=time_limit_path,
                mip_rel_gap=float(mip_rel_gap),
                decision_gw=int(gw),
            )
            modal_count = None
            n_distinct = None
            horizon_obj = float(raw.get("objective", float("nan")))
        else:
            # Strand seed mixes season + gw so arms/seasons stay independent.
            season_key = sum(ord(c) for c in str(store.season))
            strand_seed = int(seed) + 100_003 * int(gw) + 1_000_003 * int(season_key)
            week0, raw = plan_centreline_path_mu_band_roll(
                pools,
                held,
                mu_col="mu_post",
                se_col="xp_se",
                n_strands=int(n_strands),
                band_a=float(band_a),
                band_b=float(band_b),
                seed=strand_seed,
                decay=True,
                near_depth=NEAR_DEPTH,
                rho=float(RHO),
                decay_target=0.0,
                gamma=GAMMA,
                max_hits=MAX_HITS,
                ft_pe_nu=0.0,
                value_captain=True,
                time_limit=time_limit_path,
                mip_rel_gap=float(mip_rel_gap),
                decision_gw=int(gw),
            )
            mb = (raw.get("meta") or {}).get("mu_band") or {}
            modal_count = mb.get("modal_squad_count")
            n_distinct = mb.get("n_distinct_squads")
            horizon_obj = float((raw.get("meta") or {}).get("objective", float("nan")))

        banked = bank_autosub_captain(
            week0.squad,
            week0.xi,
            captain_id=str(week0.captain_id),
            work=work,
        )
        net = float(banked["pts_gross"]) - float(week0.hit_points)
        xi_ids = [str(p) for p in week0.xi["player_id"].astype(str).tolist()]
        bought = list(week0.bought)
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
                "cap_extra": float(banked["cap_extra"]),
                "pts_net": net,
                "H": int(len(pools)),
                "horizon_obj": horizon_obj,
                "n_candidates": int(
                    (raw.get("meta") or raw).get("n_candidates", 0)
                    if isinstance(raw, dict)
                    else 0
                ),
                "sold": list(week0.sold),
                "bought": bought,
                "n_autosubs": int(banked["n_autosubs"]),
                "n_blank_intended": int(banked["n_blank_intended"]),
                "captain_id": banked["captain_id"],
                "vice_id": banked["vice_id"],
                "cap_source": banked["cap_source"],
                "xi_mean_xp_se": _mean_se(work, xi_ids),
                "bought_mean_xp_se": _mean_se(work, bought),
                "modal_squad_count": modal_count,
                "n_distinct_squads": n_distinct,
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} xi={banked['pts_xi']:.0f} "
            f"cap={banked['cap_extra']:.0f} net={net:.0f} "
            f"xi_se={rows[-1]['xi_mean_xp_se']:.3f}"
            + (
                f" modal={modal_count}/{n_strands}"
                if modal_count is not None
                else ""
            ),
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
    pairs = ((TREAT, BASELINE),)
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
                    "base_xi_se": float(b["xi_mean_xp_se"].mean()),
                    "treat_xi_se": float(t["xi_mean_xp_se"].mean()),
                    "base_bought_se": float(
                        b["bought_mean_xp_se"].dropna().mean()
                        if b["bought_mean_xp_se"].notna().any()
                        else float("nan")
                    ),
                    "treat_bought_se": float(
                        t["bought_mean_xp_se"].dropna().mean()
                        if t["bought_mean_xp_se"].notna().any()
                        else float("nan")
                    ),
                    "treat_modal_mean": float(
                        pd.to_numeric(t["modal_squad_count"], errors="coerce").mean()
                    ),
                    "treat_distinct_mean": float(
                        pd.to_numeric(t["n_distinct_squads"], errors="coerce").mean()
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
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--n-strands", type=int, default=N_STRANDS)
    ap.add_argument("--band-a", type=float, default=BAND_A)
    ap.add_argument("--band-b", type=float, default=BAND_B)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--rows-out", default="")
    ap.add_argument("--from-rows", default="")
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("l4_opta_joint_path_mu_band_roll") or {}
    if not screen:
        raise RuntimeError(
            "protocol.l4_opta_joint_path_mu_band_roll missing — pre-register first"
        )

    arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
    for a in arms:
        if a not in ARMS:
            raise ValueError(f"unknown arm {a!r}; choose from {ARMS}")

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    gw_start = int(args.gw_start)
    gw_end = int(args.gw_end)
    n_strands = int(args.n_strands)
    if args.smoke:
        seasons = ["2025-26"]
        gw_start = 8
        gw_end = 10
        n_strands = min(n_strands, 2)
        print(
            f"[mu-band-roll] SMOKE 2025-26 GW8–10 n_strands={n_strands}",
            flush=True,
        )

    print("[mu-band-roll] as-of audit…", flush=True)
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
        print(f"[mu-band-roll] merged {len(all_rows)} rows from dumps", flush=True)
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
                        n_strands=n_strands,
                        band_a=float(args.band_a),
                        band_b=float(args.band_b),
                        seed=int(args.seed),
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
            else PROCESSED / "l4_opta_joint_path_mu_band_roll_smoke.json"
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
                    "n_strands": n_strands,
                    "band_a": float(args.band_a),
                    "band_b": float(args.band_b),
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
                    "task": "l4_opta_joint_path_mu_band_roll",
                    "status": "rows_dump" if args.rows_out else "smoke_ok",
                    "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "change": "μ-band roll climb rows / smoke",
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
        "screen": "l4_opta_joint_path_mu_band_roll",
    }

    verdict = doc["comparisons"][pair_key]["verdict"]
    doc["defs"] = {
        "protocol": "l4_opta_joint_path_mu_band_roll",
        "arms": list(ARMS),
        "H": H,
        "gamma": GAMMA,
        "rho": RHO,
        "n_strands": N_STRANDS,
        "band_a": BAND_A,
        "band_b": BAND_B,
        "gw_start": gw_start,
        "gw_end": gw_end,
        "value_captain": True,
        "realised": "XI after FPL autosubs + captain/VC − hits",
        "ft_pe_nu": 0.0,
        "full_board": True,
        "from_owned": True,
        "near_depth": NEAR_DEPTH,
        "elapsed_s": elapsed,
        "verdicts": {pair_key: verdict},
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_json = PROCESSED / "l4_opta_joint_path_mu_band_roll_gate.json"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (PROCESSED / "l4_opta_joint_path_mu_band_roll_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    iv = doc["comparisons"][pair_key]
    lines = [
        "# L4+Opta joint path — μ-band strand roll climb",
        "",
        "Pre-registered: `protocol.l4_opta_joint_path_mu_band_roll`.",
        "",
        "Roll each player's μ inside uncertainty bands (σ=`xp_se`, decayed with μ),",
        "solve N independent joint paths, commit modal week-0 squad.",
        "",
        f"- Boards: `build_l3_centreline_horizon(H={H})` (wired μ, ρ=0.85→0 on μ+σ).",
        f"- Baseline `{BASELINE}`: centreline μ (product).",
        f"- Treatment `{TREAT}`: N={N_STRANDS} strands, band ±{BAND_A}σ, modal week-0.",
        "- Shared: joint path, captain, full board, `from_owned`, `ft_pe_nu=0`, no chips.",
        f"- Metric: paired GW realised XI(autosub)+cap/VC−hits. Wall {elapsed:.0f}s.",
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
        "base XI σ | treat XI σ | base buy σ | treat buy σ | modal/N | distinct | mean Δ/GW |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in doc["by_season"][pair_key]:
        lines.append(
            f"| {r['season']} | {r['n_gw']} | {r['base_pts']:.0f} | "
            f"{r['treat_pts']:.0f} | {r['d_pts']:+.0f} | {r['base_hits']} | "
            f"{r['treat_hits']} | {r['base_transfers']} | {r['treat_transfers']} | "
            f"{r['base_xi_se']:.3f} | {r['treat_xi_se']:.3f} | "
            f"{r['base_bought_se']:.3f} | {r['treat_bought_se']:.3f} | "
            f"{r['treat_modal_mean']:.2f}/{N_STRANDS} | {r['treat_distinct_mean']:.2f} | "
            f"{r['mean_d_gw']:+.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Positive Δ = strand μ-band modal commit beats product mean boards.",
        "- Secondary XI/buy σ columns diagnose whether the path loads high-σ (risky) names.",
        "- Not cascade generative dice; not single-board μ±SE tilt.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.l4_opta_joint_path_mu_band_roll_gate",
        "```",
        "",
        f"JSON: `{out_json.relative_to(ROOT)}`",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    STATE.write_text(
        json.dumps(
            {
                "task": "l4_opta_joint_path_mu_band_roll",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated μ-band strand roll climb on joint path + captain",
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
