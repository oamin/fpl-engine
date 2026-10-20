"""Audit Layer 3 (transfers + hits) on closed weeks.

Opens each case with a Layer-2 μ_post wildcard 15, then solves one transfer
deadline (ft=1) on the same week's pool — legality / hit identity / dominate-hold.

  python -m src.strategy.layer3_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, HIT_COST, advance_ft, hit_cost
from src.strategy.layer1_audit import DEFAULT_CASES, load_pool
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_week import HeldState, held_from_frame, solve_week_transfers
from src.strategy.shrinkage import estimate_cohort_moments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"


def _pool_with_se_and_post(season: str, gw: int) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    pool = load_pool(season, gw)
    v2 = pd.read_csv(CLOSED_V2)
    cur = v2.loc[(v2["season"] == season) & (v2["gw"] == int(gw))].copy()
    cur["element"] = pd.to_numeric(cur["element"], errors="coerce")
    cur["xp_se"] = pd.to_numeric(cur["xp_se"], errors="coerce").fillna(0.0)
    se = (
        cur.groupby("element", as_index=False)
        .agg(xp_se=("xp_se", lambda s: float((s.astype(float) ** 2).sum() ** 0.5)))
    )
    pool = pool.merge(se, on="element", how="left")
    pool["xp_se"] = pool["xp_se"].fillna(0.0)

    hist = v2.loc[(v2["season"] == season) & (v2["gw"] < int(gw))].copy()
    moments = None
    if not hist.empty:
        hist["mu"] = pd.to_numeric(hist["score_xp_v2"], errors="coerce")
        hist["se"] = pd.to_numeric(hist["xp_se"], errors="coerce").fillna(0.0)
        hist["y"] = pd.to_numeric(hist["total_points"], errors="coerce")
        moments = {}
        for p, g in hist.groupby("position"):
            moments[str(p)] = estimate_cohort_moments(
                g["mu"].to_numpy(float),
                g["se"].to_numpy(float),
                g["y"].to_numpy(float),
            )
    return pool, moments


def audit_gw(season: str, gw: int) -> dict[str, Any]:
    pool, moments = _pool_with_se_and_post(season, gw)
    # Opening 15 on μ_post (wildcard)
    open_plan = solve_week_sigma(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        budget=BUDGET_TENTHS,
        time_limit=60.0,
    )
    squad = open_plan.plan.squad.copy()
    work = pool.copy()
    from src.strategy.decision_value import attach_mu_post

    work, _ = attach_mu_post(
        work,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    squad["purchase"] = squad["value"].astype(int)
    bank = int(BUDGET_TENTHS - squad["purchase"].sum())
    held = held_from_frame(squad, bank=max(0, bank), ft=1)

    xfer = solve_week_transfers(
        work, held, mu_col="mu_post", max_hits=2, time_limit=60.0
    )
    hold = solve_week_transfers(
        work, held, mu_col="mu_post", max_hits=0, time_limit=60.0
    )
    # With ft=1, max_hits=0 still allows 1 FT — force true hold
    held0 = HeldState(held.player_ids, held.purchase, bank=held.bank, ft=0)
    true_hold = solve_week_transfers(
        work, held0, mu_col="mu_post", max_hits=0, time_limit=60.0
    )

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    add("squad_size", len(xfer.squad) == 15, f"n={len(xfer.squad)}")
    add("xi_size", len(xfer.xi) == 11, f"n={len(xfer.xi)}")
    add("captain_in_xi", xfer.captain_id in set(xfer.xi["player_id"].astype(str)), xfer.captain_id)
    add(
        "hit_identity",
        hit_cost(held.ft, xfer.n_transfers) == xfer.hit_points,
        f"hits={xfer.hits} pts={xfer.hit_points}",
    )
    add(
        "ft_advance",
        xfer.ft_after == advance_ft(held.ft, xfer.n_transfers),
        f"ft_after={xfer.ft_after}",
    )
    add("bank_nonneg", xfer.bank_after >= 0, f"bank={xfer.bank_after}")
    add(
        "objective_identity",
        abs(xfer.objective - (xfer.xi_mu + xfer.captain_mu - xfer.hit_points)) < 1e-6,
        f"obj={xfer.objective:.4f}",
    )
    add(
        "dominates_true_hold",
        xfer.objective + 1e-6 >= true_hold.objective,
        f"xfer={xfer.objective:.4f} hold={true_hold.objective:.4f}",
    )
    add(
        "hit_cost_constant",
        HIT_COST == 4,
        f"HIT_COST={HIT_COST}",
    )
    # If took a hit, μ gain vs true hold must cover it (same board comparison on obj)
    if xfer.hits > 0:
        add(
            "hit_justified_vs_hold_obj",
            xfer.objective + 1e-6 >= true_hold.objective,
            f"Δobj={xfer.objective - true_hold.objective:+.4f}",
        )

    return {
        "season": season,
        "gw": int(gw),
        "n_pool": int(len(work)),
        "opening_obj_post": float(open_plan.post_objective),
        "xfer": xfer.summary(),
        "true_hold_obj": float(true_hold.objective),
        "ft1_maxhits0": hold.summary(),
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def write_report(results: list[dict[str, Any]], path: Path) -> None:
    n_pass = sum(1 for r in results if r["pass"])
    lines = [
        "# Strategy Layer 3 — audit (transfers + hits)",
        "",
        "Single-deadline transfers from a Layer-2 μ_post opening 15.",
        "Objective: `Σ μ_post·start + Σ μ_post·captain − 4·hits`. Chips/horizon out.",
        "",
        f"Cases: **{n_pass}/{len(results)} passed**.",
        "",
        "| season | gw | transfers | hits | obj | vs hold | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        x = r["xfer"]
        lines.append(
            f"| {r['season']} | {r['gw']} | {x['n_transfers']} | {x['hits']} | "
            f"{x['objective']:.2f} | {x['objective'] - r['true_hold_obj']:+.2f} | "
            f"{'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Protocol: `experiments/protocol.json` → `strategy_layer3_transfers`.",
        f"JSON: `{(PROCESSED / 'strategy_layer3_audit.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    results = []
    for season, gw in DEFAULT_CASES:
        print(f"auditing L3 {season} GW{gw} …", flush=True)
        row = audit_gw(season, gw)
        results.append(row)
        x = row["xfer"]
        print(
            f"  {'PASS' if row['pass'] else 'FAIL'}  "
            f"T={x['n_transfers']} hits={x['hits']} obj={x['objective']:.2f} "
            f"Δhold={x['objective'] - row['true_hold_obj']:+.2f}",
            flush=True,
        )
        if not row["pass"]:
            for c in row["checks"]:
                if not c["pass"]:
                    print(f"    FAIL {c['name']}: {c['detail']}", flush=True)
    write_report(results, REPORTS / "strategy_layer3_audit.md")
    (PROCESSED / "strategy_layer3_audit.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {REPORTS / 'strategy_layer3_audit.md'}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
