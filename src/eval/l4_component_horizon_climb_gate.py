"""Gated climb: L4+Opta H=12 vs L4 H=2 with μ = one xP component.

Pre-registered: ``protocol.l4_component_horizon_climb``.

For each component C ∈ {xp_goals, xp_assists, xp_cs, xp_gc}:

* treatment ``h12_<C>`` — ``build_l3_centreline_horizon(H=12)``
  (L4/FD priced t:t+1 + Opta blind far); ``mu_post := C``
* baseline ``h2_<C>`` — ``build_mupost_priced_horizon`` (BOOKIE_DEPTH=2);
  ``mu_post := C``

Shared wildcard open on full ``score_xp_v2`` (L4 default t:t+1 scorer).
Transfers / XI / bench / hits allowed; **no captaincy** in objective or
realised points; **no chips**.

  PYTHONPATH=. python3 -m src.eval.l4_component_horizon_climb_gate
  PYTHONPATH=. python3 -m src.eval.l4_component_horizon_climb_gate --components xp_goals
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
from src.markets.numagent_xp_pilot import _asof_alphas
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison
from src.rules.fpl_2026 import BUDGET_TENTHS, SQUAD_SIZE, advance_ft
from src.strategy.decision_value import attach_mu_post
from src.strategy.l3_centreline_horizon import build_l3_centreline_horizon
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import HeldState, held_from_frame
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_priced import (
    COMPONENT_MU_COLS,
    PROCESSED,
    build_mupost_priced_horizon,
    project_week_score_xp_v2,
)
from src.strategy.layer4_week import solve_horizon_transfers

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "l4_component_horizon_climb_gate.md"
STATE = ROOT / "REPORT_STATE.json"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

DEFAULT_COMPONENTS = ("xp_goals", "xp_assists", "xp_cs", "xp_gc")
GW_START = 8
GW_END = 38
H_TREAT = 12
N_BOOT = 1000
SEED = 0
MAX_HITS = SQUAD_SIZE  # uncapped; −4/hit
GAMMA = 1.0
TIME_LIMIT = 60.0


def _now() -> float:
    return time.time()


def _arm_h2(comp: str) -> str:
    return f"h2_{comp}"


def _arm_h12(comp: str) -> str:
    return f"h12_{comp}"


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


def _ensure_component_cols(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    for c in COMPONENT_MU_COLS:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    for c in ("score_xp_v2", "xp_se", "mu_post"):
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def _attach_decision_gw_components(
    work: pd.DataFrame,
    *,
    season: str,
    gw: int,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay as-of FD centreline components for the decision GW onto L4 pool."""
    out = _ensure_component_cols(work)
    hist = merged.loc[merged["GW"] < int(gw)]
    if hist.empty:
        return out
    alpha_g, alpha_a = _asof_alphas(hist)
    raw, _n = project_week_score_xp_v2(
        hist,
        merged,
        fd,
        decision_gw=int(gw),
        target_gw=int(gw),
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
    )
    if raw.empty:
        return out
    raw = raw.copy()
    raw["element"] = pd.to_numeric(raw["element"], errors="coerce")
    keep = ["element"] + [c for c in COMPONENT_MU_COLS if c in raw.columns]
    if "xp" in raw.columns:
        keep.append("xp")
    slim = raw[keep].drop_duplicates("element", keep="last")
    out["element"] = pd.to_numeric(out["element"], errors="coerce")
    out = out.drop(columns=[c for c in COMPONENT_MU_COLS if c in out.columns], errors="ignore")
    out = out.merge(slim, on="element", how="left")
    out = _ensure_component_cols(out)
    return out


def _apply_component_mu(pools: list[pd.DataFrame], component: str) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    for p in pools:
        q = _ensure_component_cols(p)
        if component not in q.columns:
            raise RuntimeError(f"missing component {component!r}")
        q["mu_post"] = pd.to_numeric(q[component], errors="coerce").fillna(0.0)
        out.append(q)
    return out


def _prep_week0(
    store: Any,
    gw: int,
    *,
    merged: pd.DataFrame,
    fd: pd.DataFrame,
) -> tuple[pd.DataFrame, Any]:
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
    work = _attach_decision_gw_components(
        work, season=store.season, gw=int(gw), merged=merged, fd=fd
    )
    return work, moments


def climb_season(
    store: Any,
    *,
    arm: str,
    component: str,
    horizon_mode: str,
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
        work, moments = _prep_week0(store, gw, merged=merged, fd=fd)

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
            pts_map = _pts_map(work)
            pts = _realised_xi_only(splan.plan.xi, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "component": component,
                    "phase": "wildcard",
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "pts_xi": float(pts),
                    "pts_net": float(pts),
                    "H": 0,
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
        work = _ensure_component_cols(work)
        pts_map = _pts_map(work)

        if horizon_mode == "h2":
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
        elif horizon_mode == "h12":
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
            raise ValueError(horizon_mode)
        pools = _apply_component_mu(pools, component)

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
            value_captain=False,
        )
        week0 = hplan.week0
        pts = _realised_xi_only(week0.xi, pts_map)
        net = float(pts) - float(week0.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "component": component,
                "phase": "transfer",
                "n_transfers": int(week0.n_transfers),
                "hits": int(week0.hits),
                "hit_points": int(week0.hit_points),
                "pts_xi": float(pts),
                "pts_net": net,
                "H": int(len(pools)),
                "horizon_obj": float(hplan.objective),
                "sold": list(week0.sold),
                "bought": list(week0.bought),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={week0.n_transfers} "
            f"hits={week0.hits} H={len(pools)} xi={pts:.0f} net={net:.0f}",
            flush=True,
        )
        held = _step_held(held, week0)
        prev_meta = pd.concat([work, week0.squad], ignore_index=True).drop_duplicates(
            "player_id", keep="last"
        )
    return rows


def summarise(
    rows: list[dict[str, Any]], components: tuple[str, ...]
) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    xfer = df.loc[df["phase"] == "transfer"].copy()
    comparisons: dict[str, Any] = {}
    by_season_all: dict[str, list[dict[str, Any]]] = {}

    for comp in components:
        treat = _arm_h12(comp)
        base = _arm_h2(comp)
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
                f"{comp}: missing season deltas; have {sorted(deltas)}"
            )
        for season, arr in deltas.items():
            if arr.size < 20:
                raise RuntimeError(f"{comp} {season}: fewer than 20 GWs ({arr.size})")
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
            "component": comp,
        }
        season_rows = []
        for season in CLOSED_SEASONS:
            sub = xfer.loc[xfer["season"] == season]
            h2 = float(sub.loc[sub["arm"] == base, "pts_net"].sum())
            h12 = float(sub.loc[sub["arm"] == treat, "pts_net"].sum())
            season_rows.append(
                {
                    "season": season,
                    "n_gw": int(sub.loc[sub["arm"] == base, "gw"].nunique()),
                    "h2_pts": h2,
                    "h12_pts": h12,
                    "d_pts": h12 - h2,
                    "h2_hits": int(sub.loc[sub["arm"] == base, "hit_points"].sum()),
                    "h12_hits": int(sub.loc[sub["arm"] == treat, "hit_points"].sum()),
                    "h2_transfers": int(
                        sub.loc[sub["arm"] == base, "n_transfers"].sum()
                    ),
                    "h12_transfers": int(
                        sub.loc[sub["arm"] == treat, "n_transfers"].sum()
                    ),
                    "mean_d_gw": float(np.mean(deltas[season])),
                }
            )
        by_season_all[comp] = season_rows

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
    ap.add_argument("--time-limit", type=float, default=TIME_LIMIT)
    ap.add_argument(
        "--components",
        default=",".join(DEFAULT_COMPONENTS),
        help="comma list of xp_* columns; each runs H=12 vs H=2",
    )
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("l4_component_horizon_climb") or {}
    if not screen:
        raise RuntimeError(
            "protocol.l4_component_horizon_climb missing — pre-register first"
        )

    components = tuple(
        c.strip() for c in str(args.components).split(",") if c.strip()
    )
    for c in components:
        if c not in DEFAULT_COMPONENTS:
            raise ValueError(f"unknown component {c!r}; choose from {DEFAULT_COMPONENTS}")

    print("[l4-comp-h] as-of audit…", flush=True)
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
    from src.strategy.shrink_climb import SeasonStore

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
        for comp in components:
            for horizon_mode, arm_fn in (
                ("h2", _arm_h2),
                ("h12", _arm_h12),
            ):
                arm = arm_fn(comp)
                print(
                    f"climbing {season} arm={arm} μ={comp} mode={horizon_mode} …",
                    flush=True,
                )
                all_rows.extend(
                    climb_season(
                        store,
                        arm=arm,
                        component=comp,
                        horizon_mode=horizon_mode,
                        merged=merged,
                        fd=fd,
                        prices=prices,
                        gw_start=int(args.gw_start),
                        gw_end=int(args.gw_end),
                        time_limit=float(args.time_limit),
                    )
                )

    doc = summarise(all_rows, components)
    comparisons = {
        **_certified_protocol_comparison(),
        **doc["comparisons"],
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "screen": "l4_component_horizon_climb",
    }

    elapsed = _now() - t0
    verdicts = {
        c: doc["comparisons"][comparison_key(_arm_h12(c), _arm_h2(c))]["verdict"]
        for c in components
    }
    doc["defs"] = {
        "protocol": "l4_component_horizon_climb",
        "components": list(components),
        "H_treat": H_TREAT,
        "gamma": GAMMA,
        "gw_start": int(args.gw_start),
        "gw_end": int(args.gw_end),
        "max_hits": MAX_HITS,
        "value_captain": False,
        "realised": "XI total_points only",
        "time_limit": float(args.time_limit),
        "elapsed_s": elapsed,
        "verdicts": verdicts,
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_json = PROCESSED / "l4_component_horizon_climb_gate.json"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (PROCESSED / "l4_component_horizon_climb_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# L4 component-μ H=12 vs H=2 climb — gated",
        "",
        "Pre-registered: `protocol.l4_component_horizon_climb`.",
        "",
        "Per-component μ screen on the locked L4 near / Opta far centreline "
        "recipes. Isolates which xP pieces help or hurt long-horizon transfer "
        "decisions. **Not** a μ promote.",
        "",
        f"- Treatment: `build_l3_centreline_horizon(H={H_TREAT})` with "
        "`mu_post :=` component.",
        "- Baseline: `build_mupost_priced_horizon` (BOOKIE_DEPTH=2) with "
        "`mu_post :=` same component.",
        "- Near scorer: **L4** (`score_xp_v2` / priced FD); far: Opta blind λ → "
        "`score_team_side`.",
        f"- Shared wildcard open on full `score_xp_v2` @ GW{args.gw_start}.",
        "- No captaincy (objective + realised XI only); no chips; "
        f"hits uncapped (max_hits={MAX_HITS}, −4/hit); γ={GAMMA}.",
        f"- Metric: paired GW realised XI−hits (transfer weeks). Wall {elapsed:.0f}s.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
        "## Results (H=12 − H=2, same component μ)",
        "",
    ]
    for comp in components:
        key = comparison_key(_arm_h12(comp), _arm_h2(comp))
        iv = doc["comparisons"][key]
        lines += [
            f"### `{comp}`",
            "",
            f"- Mean Δ pts/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
            f"- Covers 0: **{iv['covers_zero']}**",
            f"- Verdict: **{iv['verdict']}**",
            "",
            "| season | n GW | H=2 | H=12 | Δ | H2 hits | H12 hits | H2 T | H12 T | mean Δ/GW |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in doc["by_season"][comp]:
            lines.append(
                f"| {r['season']} | {r['n_gw']} | {r['h2_pts']:.0f} | {r['h12_pts']:.0f} | "
                f"{r['d_pts']:+.0f} | {r['h2_hits']} | {r['h12_hits']} | "
                f"{r['h2_transfers']} | {r['h12_transfers']} | {r['mean_d_gw']:+.3f} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Positive Δ = H=12 (Opta far) beats H=2 priced-only when both climb "
        "on that component alone.",
        "- Prior `l3_component_climb_gate` (component vs full, L3 one-week) "
        "answered a different question and stays parked.",
        "- Season totals are descriptive; the gate is the paired GW interval.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.l4_component_horizon_climb_gate",
        "```",
        "",
        f"JSON: `data/processed/l4_component_horizon_climb_gate.json`",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    STATE.write_text(
        json.dumps(
            {
                "task": "l4_component_horizon_climb",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated L4 H=12 vs H=2 climbs with component-only μ",
                "verdicts": {
                    c: {
                        "verdict": doc["comparisons"][
                            comparison_key(_arm_h12(c), _arm_h2(c))
                        ]["verdict"],
                        "mean": doc["comparisons"][
                            comparison_key(_arm_h12(c), _arm_h2(c))
                        ]["mean"],
                        "lo": doc["comparisons"][
                            comparison_key(_arm_h12(c), _arm_h2(c))
                        ]["lo"],
                        "hi": doc["comparisons"][
                            comparison_key(_arm_h12(c), _arm_h2(c))
                        ]["hi"],
                    }
                    for c in components
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
                "verdicts": verdicts,
                "means": {
                    c: doc["comparisons"][comparison_key(_arm_h12(c), _arm_h2(c))][
                        "mean"
                    ]
                    for c in components
                },
                "report": str(REPORT),
                "elapsed_s": elapsed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
