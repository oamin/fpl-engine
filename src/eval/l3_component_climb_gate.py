"""Gated climb: single centreline xP components vs full score_xp_v2.

Pre-registered: ``protocol.l3_component_climb_gate``.

Stripped L3 transfer climb (no horizon). Each arm chooses transfers on one
μ column from the **same** as-of FD centreline board:

* baseline ``full`` — ``score_xp_v2`` / ``xp``
* treatments — ``xp_goals``, ``xp_assists``, ``xp_cs``, ``xp_gc``

Rules for this screen:

* **No captaincy** in the objective (``value_captain=False``) or in realised
  points (XI ``total_points`` only).
* Hits **uncapped** (solver bound ``SQUAD_SIZE``; still −4/hit).
* Shared μ_post-style wildcard open on full ``score_xp_v2`` (identical open).

  PYTHONPATH=. python3 -m src.eval.l3_component_climb_gate
  PYTHONPATH=. python3 -m src.eval.l3_component_climb_gate --components xp_goals
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
from src.markets import numagent_xp
from src.markets.numagent_xp_pilot import _asof_alphas, load_merged
from src.markets.score_xp_v2_asof import run_score_xp_v2_asof
from src.markets.score_xp_v2_gate import _certified_protocol_comparison
from src.models.short_horizon_cascade import (
    build_frozen_side_table,
    centreline_fixture_fits,
    fixtures_for_gw,
)
from src.rules.fpl_2026 import BUDGET_TENTHS, SQUAD_SIZE, advance_ft, normalize_position
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import HeldState, held_from_frame, solve_week_transfers
from src.strategy.layer4_bookie_path import _fd_index, _load_merged_enriched
from src.strategy.layer4_climb import _step_held
from src.strategy.layer4_mupost_priced import PROCESSED
from src.strategy.shrink_climb import SeasonStore
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
REPORT = REPORTS / "l3_component_climb_gate.md"
STATE = ROOT / "REPORT_STATE.json"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

BASELINE = "full"
DEFAULT_COMPONENTS = ("xp_goals", "xp_assists", "xp_cs", "xp_gc")
GW_START = 8
GW_END = 38
N_BOOT = 1000
SEED = 0
MAX_HITS = SQUAD_SIZE  # uncapped (full rebuild bound)
TIME_LIMIT = 45.0


def _now() -> float:
    return time.time()


def _realised_xi_only(xi: pd.DataFrame, pts_map: pd.Series) -> float:
    ids = xi["player_id"].astype(str)
    return float(ids.map(pts_map).astype(float).sum())


def _score_component_board(
    side: pd.DataFrame,
    fits: list[dict[str, Any]],
    *,
    alpha_g: float,
    alpha_a: float,
) -> pd.DataFrame:
    """Aggregate centreline components to element (DGW-safe sum)."""
    if side.empty or not fits:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for fit in fits:
        for team, team_n, prefix in (
            (str(fit["home"]), str(fit["home_n"]), "home"),
            (str(fit["away"]), str(fit["away_n"]), "away"),
        ):
            club = side.loc[side["team"].astype(str) == team].copy()
            if club.empty and "team_n" in side.columns:
                club = side.loc[side["team_n"] == norm_team(team_n)].copy()
            if club.empty:
                club = side.loc[
                    side["team"].map(lambda t: norm_team(str(t)) == norm_team(team_n))
                ].copy()
            if club.empty:
                continue
            scored = numagent_xp.score_team_side(
                club,
                xg=float(fit[f"xg_{prefix}"]),
                xcs=float(fit[f"xcs_{prefix}"]),
                xgc=float(fit[f"xgc_{prefix}"]),
                alpha_g=float(alpha_g),
                alpha_a=float(alpha_a),
            ).copy()
            scored["element"] = club["element"].to_numpy()
            scored["team"] = team
            scored["club"] = team
            parts.append(scored)
    if not parts:
        return pd.DataFrame()
    all_p = pd.concat(parts, ignore_index=True)
    sum_cols = [
        c
        for c in (
            "xp",
            "xp_goals",
            "xp_assists",
            "xp_cs",
            "xp_gc",
            "xp_appear",
            "xp_defcon",
            "xp_bonus",
            "xp_cards",
            "xp_saves",
            "xp_se",
        )
        if c in all_p.columns
    ]
    agg: dict[str, Any] = {c: "sum" for c in sum_cols}
    agg.update(
        {
            "name": "first",
            "position": "first",
            "team": "first",
            "club": "first",
        }
    )
    out = all_p.groupby("element", as_index=False).agg(agg)
    out["score_xp_v2"] = out["xp"]
    out["position"] = out["position"].map(normalize_position)
    return out


def build_gw_component_pool(
    *,
    season: str,
    gw: int,
    merged: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    store: SeasonStore,
) -> pd.DataFrame | None:
    """As-of FD centreline board + prices + realised total_points."""
    hist = numagent_xp.asof_share_history(merged, before_gw=int(gw), season=season)
    if hist.empty:
        return None
    alpha_g, alpha_a = _asof_alphas(hist)
    side, alpha_g, alpha_a = build_frozen_side_table(
        hist,
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        season=season,
        historical_rate_baseline=True,
    )
    if side.empty:
        return None
    fx = fixtures_for_gw(enriched, int(gw))
    fits = centreline_fixture_fits(fx, fd)
    if len(fits) < 5:
        return None
    board = _score_component_board(
        side, fits, alpha_g=float(alpha_g), alpha_a=float(alpha_a)
    )
    if board.empty:
        return None

    pr = prices.loc[prices["gw"] == int(gw)].copy()
    pr["element"] = pd.to_numeric(pr["element"], errors="coerce")
    out = board.merge(
        pr[["element", "value", "team_norm", "player_id"]],
        on="element",
        how="inner",
    )
    if out.empty:
        return None
    out["player_id"] = out["player_id"].astype(str)
    out["club"] = out["team_norm"].fillna(out["club"]).astype(str)
    # Realised points from closed store (oracle minutes season sheet).
    actual = store.pool(int(gw))[["element", "total_points"]].copy()
    actual["element"] = pd.to_numeric(actual["element"], errors="coerce")
    out = out.merge(actual, on="element", how="left")
    out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(
        0.0
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(45).astype(int)
    for c in DEFAULT_COMPONENTS:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["score_xp_v2"] = pd.to_numeric(out["score_xp_v2"], errors="coerce").fillna(0.0)
    return out.reset_index(drop=True)


def climb_season(
    store: SeasonStore,
    *,
    arm: str,
    mu_col: str,
    merged: pd.DataFrame,
    enriched: pd.DataFrame,
    fd: pd.DataFrame,
    prices: pd.DataFrame,
    gw_start: int,
    gw_end: int,
    time_limit: float,
    board_cache: dict[int, pd.DataFrame],
) -> list[dict[str, Any]]:
    gws = store.gws(gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    held: HeldState | None = None
    prev_meta: pd.DataFrame | None = None

    for gw in gws:
        if int(gw) not in board_cache:
            pool = build_gw_component_pool(
                season=store.season,
                gw=int(gw),
                merged=merged,
                enriched=enriched,
                fd=fd,
                prices=prices,
                store=store,
            )
            if pool is None:
                continue
            board_cache[int(gw)] = pool
        work = board_cache[int(gw)].copy()
        if mu_col not in work.columns:
            raise RuntimeError(f"missing μ column {mu_col!r} at GW{gw}")
        if "xp_se" not in work.columns:
            work["xp_se"] = 0.0

        if held is None:
            # Shared open: always on full score_xp_v2 (identical across arms).
            splan = solve_week_sigma(
                work,
                mu_col="score_xp_v2",
                se_col="xp_se",
                mode="none",
                moments_by_pos=None,
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
            pts = _realised_xi_only(splan.plan.xi, pts_map)
            rows.append(
                {
                    "season": store.season,
                    "gw": int(gw),
                    "arm": arm,
                    "phase": "wildcard",
                    "mu_col": "score_xp_v2",
                    "n_transfers": 0,
                    "hits": 0,
                    "hit_points": 0,
                    "pts_xi": float(pts),
                    "pts_net": float(pts),
                }
            )
            prev_meta = work.copy()
            print(f"  {store.season} {arm} GW{gw:02d} WC xi={pts:.0f}", flush=True)
            continue

        assert held is not None and prev_meta is not None
        work = _ensure_held_rows(work, held, prev_meta)
        pts_map = work.set_index(work["player_id"].astype(str))["total_points"].fillna(
            0.0
        )
        plan = solve_week_transfers(
            work,
            held,
            mu_col=mu_col,
            max_hits=MAX_HITS,
            time_limit=time_limit,
            value_captain=False,
        )
        pts = _realised_xi_only(plan.xi, pts_map)
        net = float(pts) - float(plan.hit_points)
        rows.append(
            {
                "season": store.season,
                "gw": int(gw),
                "arm": arm,
                "phase": "transfer",
                "mu_col": mu_col,
                "n_transfers": int(plan.n_transfers),
                "hits": int(plan.hits),
                "hit_points": int(plan.hit_points),
                "pts_xi": float(pts),
                "pts_net": net,
                "sold": list(plan.sold),
                "bought": list(plan.bought),
            }
        )
        print(
            f"  {store.season} {arm} GW{gw:02d} T={plan.n_transfers} "
            f"hits={plan.hits} xi={pts:.0f} net={net:.0f}",
            flush=True,
        )
        held = _step_held(held, plan)
        prev_meta = pd.concat([work, plan.squad], ignore_index=True).drop_duplicates(
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
        arm = comp
        piv = xfer.pivot_table(
            index=["season", "gw"], columns="arm", values="pts_net"
        )
        deltas: dict[str, np.ndarray] = {}
        for season in CLOSED_SEASONS:
            if season not in piv.index.get_level_values(0):
                continue
            block = piv.loc[season]
            if arm not in block.columns or BASELINE not in block.columns:
                continue
            d = (block[arm] - block[BASELINE]).dropna().to_numpy(float)
            if d.size:
                deltas[season] = d
        if set(deltas) != set(CLOSED_SEASONS):
            raise RuntimeError(
                f"{arm}: missing season deltas; have {sorted(deltas)}"
            )
        for season, arr in deltas.items():
            if arr.size < 20:
                raise RuntimeError(f"{arm} {season}: fewer than 20 GWs ({arr.size})")
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
        key = comparison_key(arm, BASELINE)
        comparisons[key] = {
            **{k: interval[k] for k in ("mean", "lo", "hi", "n_gws")},
            "incomplete_seasons": {},
            "covers_zero": interval["covers_zero"],
            "by_season_mean": interval["by_season_mean"],
            "verdict": verdict,
            "component": arm,
        }
        season_rows = []
        for season in CLOSED_SEASONS:
            sub = xfer.loc[xfer["season"] == season]
            base = float(sub.loc[sub["arm"] == BASELINE, "pts_net"].sum())
            tr = float(sub.loc[sub["arm"] == arm, "pts_net"].sum())
            season_rows.append(
                {
                    "season": season,
                    "n_gw": int(sub.loc[sub["arm"] == BASELINE, "gw"].nunique()),
                    "full_pts": base,
                    "comp_pts": tr,
                    "d_pts": tr - base,
                    "full_hits": int(
                        sub.loc[sub["arm"] == BASELINE, "hit_points"].sum()
                    ),
                    "comp_hits": int(sub.loc[sub["arm"] == arm, "hit_points"].sum()),
                    "mean_d_gw": float(np.mean(deltas[season])),
                }
            )
        by_season_all[arm] = season_rows

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
        help="comma list of xp_* columns to climb vs full",
    )
    args = ap.parse_args(argv)

    protocol = load_protocol()
    screen = protocol.get("l3_component_climb_gate") or {}
    if not screen:
        raise RuntimeError("protocol.l3_component_climb_gate missing — pre-register first")

    components = tuple(
        c.strip() for c in str(args.components).split(",") if c.strip()
    )
    if not components:
        raise RuntimeError("need at least one --components value")

    print("[component-climb] as-of audit…", flush=True)
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
        merged = load_merged(season)
        enriched = _load_merged_enriched(season)
        fd = _fd_index(season)
        prices = pd.read_csv(
            PROCESSED / f"score_xp_{season.replace('-', '_')}.csv",
            usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
        )
        prices["gw"] = pd.to_numeric(prices["gw"], errors="coerce").astype(int)
        store = SeasonStore(season, v2_all)
        board_cache: dict[int, pd.DataFrame] = {}
        arms = [(BASELINE, "score_xp_v2")] + [(c, c) for c in components]
        for arm, mu_col in arms:
            print(f"climbing {season} arm={arm} mu={mu_col} …", flush=True)
            # Ensure xp_se for wildcard open
            # board_cache filled on first arm
            all_rows.extend(
                climb_season(
                    store,
                    arm=arm,
                    mu_col=mu_col,
                    merged=merged,
                    enriched=enriched,
                    fd=fd,
                    prices=prices,
                    gw_start=int(args.gw_start),
                    gw_end=int(args.gw_end),
                    time_limit=float(args.time_limit),
                    board_cache=board_cache,
                )
            )

    doc = summarise(all_rows, components)
    elapsed = _now() - t0

    comps = {
        **_certified_protocol_comparison(),
        **{
            k: {
                "mean": v["mean"],
                "lo": v["lo"],
                "hi": v["hi"],
                "n_gws": v["n_gws"],
                "incomplete_seasons": {},
            }
            for k, v in doc["comparisons"].items()
        },
    }
    intervals = {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comps,
        "screen": "l3_component_climb_gate",
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_json = PROCESSED / "l3_component_climb_gate.json"
    payload = {
        **doc,
        "defs": {
            "protocol": "l3_component_climb_gate",
            "baseline": BASELINE,
            "components": list(components),
            "value_captain": False,
            "max_hits": MAX_HITS,
            "gw_start": int(args.gw_start),
            "gw_end": int(args.gw_end),
            "elapsed_s": elapsed,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (PROCESSED / "l3_component_climb_intervals.json").write_text(
        json.dumps(intervals, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# L3 component climb — gated (no captain, uncapped hits)",
        "",
        "Pre-registered: `protocol.l3_component_climb_gate`.",
        "",
        "Same as-of FD centreline board each GW. Arms differ only by μ column.",
        f"Baseline `{BASELINE}` = `score_xp_v2`. Treatments: "
        + ", ".join(f"`{c}`" for c in components)
        + ".",
        "",
        "- **No captaincy** in objective or realised score (XI points only).",
        f"- Hits uncapped (solver bound max_hits={MAX_HITS}; still −4/hit).",
        f"- Shared wildcard open on full `score_xp_v2` @ GW{args.gw_start}.",
        f"- Wall {elapsed:.0f}s.",
        "",
        "## Gates",
        "",
        "- Engine as-of audit: **passed**.",
        "- `score_xp_v2` as-of: **passed**.",
        "",
    ]
    for comp in components:
        key = comparison_key(comp, BASELINE)
        iv = doc["comparisons"][key]
        lines += [
            f"## {comp} − full",
            "",
            f"- Mean Δ pts/GW: **{iv['mean']:+.3f}** [{iv['lo']:+.3f}, {iv['hi']:+.3f}]",
            f"- Covers 0: **{iv['covers_zero']}**",
            f"- Verdict: **{iv['verdict']}**",
            "",
            "| season | n GW | full | component | Δ | full hits | comp hits | mean Δ/GW |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in doc["by_season"][comp]:
            lines.append(
                f"| {r['season']} | {r['n_gw']} | {r['full_pts']:.0f} | "
                f"{r['comp_pts']:.0f} | {r['d_pts']:+.0f} | {r['full_hits']} | "
                f"{r['comp_hits']} | {r['mean_d_gw']:+.3f} |"
            )
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Positive Δ = component-μ climb beats full `score_xp_v2` climb.",
        "- Diagnostic isolation of which centreline piece drives transfer value.",
        "- Season totals descriptive; gate = paired GW interval per component.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.l3_component_climb_gate",
        "```",
        "",
        f"JSON: `data/processed/l3_component_climb_gate.json`",
        "",
    ]
    write_gated_report(REPORT, audit_out, intervals, lines)

    verdicts = {
        c: doc["comparisons"][comparison_key(c, BASELINE)]["verdict"]
        for c in components
    }
    STATE.write_text(
        json.dumps(
            {
                "task": "l3_component_climb_gate",
                "status": "done",
                "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "change": "Gated L3 component climbs vs full xp (no captain, uncapped hits)",
                "verdicts": {
                    c: {
                        "verdict": doc["comparisons"][comparison_key(c, BASELINE)][
                            "verdict"
                        ],
                        "mean": doc["comparisons"][comparison_key(c, BASELINE)][
                            "mean"
                        ],
                        "lo": doc["comparisons"][comparison_key(c, BASELINE)]["lo"],
                        "hi": doc["comparisons"][comparison_key(c, BASELINE)]["hi"],
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
    print(json.dumps({"verdicts": verdicts, "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
