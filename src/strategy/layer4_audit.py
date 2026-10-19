"""Audit Layer 4 (horizon) on closed weeks.

Checks:
  1. H=1 + zero terminal recovers Layer-3 transfers / XI / captain / objective
  2. H=4 solves legally on the same opening board
  3. Hit / FT / bank identities on the commit week

  python -m src.strategy.layer4_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.season_climb_ft import GAMMA
from src.rules.fpl_2026 import BUDGET_TENTHS, advance_ft, hit_cost
from src.strategy.layer1_audit import DEFAULT_CASES
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_week import held_from_frame, solve_week_transfers
from src.strategy.layer4_week import (
    DEFAULT_HORIZON,
    DEFAULT_TERMINAL_MU_WEIGHT,
    DEFAULT_TERMINAL_VALUE_WEIGHT,
    solve_horizon_transfers,
)
from src.strategy.shrinkage import estimate_cohort_moments

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"


def _pool_with_se_and_post(season: str, gw: int):
    from src.strategy.layer1_audit import load_pool
    from src.strategy.decision_value import attach_mu_post

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
    work, _ = attach_mu_post(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        out_col="mu_post",
    )
    return work, moments


def _future_pools(season: str, gw: int, H: int, moments) -> list[pd.DataFrame]:
    """As-of μ_post boards for gw .. gw+H-1 (missing weeks → copy last)."""
    from src.strategy.layer1_audit import load_pool
    from src.strategy.decision_value import attach_mu_post

    v2 = pd.read_csv(CLOSED_V2)
    out: list[pd.DataFrame] = []
    last = None
    for t in range(H):
        g = int(gw) + t
        try:
            pool = load_pool(season, g)
        except Exception:
            if last is None:
                raise
            out.append(last.copy())
            continue
        cur = v2.loc[(v2["season"] == season) & (v2["gw"] == g)].copy()
        if cur.empty and last is not None:
            out.append(last.copy())
            continue
        cur["element"] = pd.to_numeric(cur["element"], errors="coerce")
        cur["xp_se"] = pd.to_numeric(cur["xp_se"], errors="coerce").fillna(0.0)
        se = (
            cur.groupby("element", as_index=False)
            .agg(xp_se=("xp_se", lambda s: float((s.astype(float) ** 2).sum() ** 0.5)))
        )
        pool = pool.merge(se, on="element", how="left")
        pool["xp_se"] = pool["xp_se"].fillna(0.0)
        # Moments stay as-of decision week (no future shrink fit)
        work, _ = attach_mu_post(
            pool,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            out_col="mu_post",
        )
        last = work
        out.append(work)
    return out


def audit_gw(season: str, gw: int, *, horizon: int = DEFAULT_HORIZON) -> dict[str, Any]:
    work0, moments = _pool_with_se_and_post(season, gw)
    open_plan = solve_week_sigma(
        work0,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode="none",
        moments_by_pos=moments,
        budget=BUDGET_TENTHS,
        time_limit=60.0,
    )
    squad = open_plan.plan.squad.copy()
    squad["purchase"] = squad["value"].astype(int)
    bank = int(BUDGET_TENTHS - squad["purchase"].sum())
    held = held_from_frame(squad, bank=max(0, bank), ft=1)

    l3 = solve_week_transfers(work0, held, mu_col="mu_post", max_hits=2, time_limit=60.0)
    h1 = solve_horizon_transfers(
        [work0],
        held,
        mu_col="mu_post",
        horizon=1,
        terminal_mu_weight=0.0,
        terminal_value_weight=0.0,
        max_hits=2,
        time_limit=60.0,
    )
    pools = _future_pools(season, gw, horizon, moments)
    hN = solve_horizon_transfers(
        pools,
        held,
        mu_col="mu_post",
        horizon=horizon,
        gamma=GAMMA,
        terminal_mu_weight=DEFAULT_TERMINAL_MU_WEIGHT,
        terminal_value_weight=DEFAULT_TERMINAL_VALUE_WEIGHT,
        max_hits=2,
        time_limit=120.0,
    )

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    add(
        "h1_bought",
        set(h1.week0.bought) == set(l3.bought),
        f"h1={h1.week0.bought} l3={l3.bought}",
    )
    add(
        "h1_sold",
        set(h1.week0.sold) == set(l3.sold),
        f"h1={h1.week0.sold} l3={l3.sold}",
    )
    add(
        "h1_captain",
        h1.week0.captain_id == l3.captain_id,
        f"h1={h1.week0.captain_id} l3={l3.captain_id}",
    )
    add(
        "h1_xi",
        set(h1.week0.xi["player_id"].astype(str)) == set(l3.xi["player_id"].astype(str)),
        "xi match",
    )
    add(
        "h1_objective",
        abs(h1.week0.objective - l3.objective) < 1e-5,
        f"h1={h1.week0.objective:.4f} l3={l3.objective:.4f}",
    )
    add("hN_squad", len(hN.week0.squad) == 15, f"n={len(hN.week0.squad)}")
    add("hN_xi", len(hN.week0.xi) == 11, f"n={len(hN.week0.xi)}")
    add(
        "hN_hit_identity",
        hit_cost(held.ft, hN.week0.n_transfers) == hN.week0.hit_points,
        f"hits={hN.week0.hits}",
    )
    add(
        "hN_ft_advance",
        hN.week0.ft_after == advance_ft(held.ft, hN.week0.n_transfers),
        f"ft_after={hN.week0.ft_after}",
    )
    add("hN_bank", hN.week0.bank_after >= 0, f"bank={hN.week0.bank_after}")
    add(
        "hN_horizon_len",
        len(hN.week_xi_mu) == horizon,
        f"len={len(hN.week_xi_mu)}",
    )

    return {
        "season": season,
        "gw": int(gw),
        "horizon": horizon,
        "passed": all(c["pass"] for c in checks),
        "checks": checks,
        "l3": l3.summary(),
        "h1": h1.summary(),
        "hN": hN.summary(),
        "n_transfer_diff_vs_l3": int(
            len(set(hN.week0.bought).symmetric_difference(set(l3.bought)))
        ),
    }


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    n_ok = sum(1 for r in rows if r["passed"])
    lines = [
        "# Strategy Layer 4 — horizon audit",
        "",
        "Transfers only at the decision week; XI+captain each week in H on that 15.",
        f"γ = {GAMMA}, default H = {DEFAULT_HORIZON}, "
        f"terminal μ weight = {DEFAULT_TERMINAL_MU_WEIGHT}.",
        "",
        f"Cases passed: **{n_ok}/{len(rows)}**",
        "",
        "| season | gw | pass | H=1≡L3 | H transfers≠L3 | H obj | L3 obj |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for r in rows:
        h1_ok = all(
            c["pass"]
            for c in r["checks"]
            if c["name"].startswith("h1_")
        )
        lines.append(
            f"| {r['season']} | {r['gw']} | {'PASS' if r['passed'] else 'FAIL'} | "
            f"{'yes' if h1_ok else 'no'} | {r['n_transfer_diff_vs_l3']} | "
            f"{r['hN']['horizon_objective']:.2f} | {r['l3']['objective']:.2f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- **H=1≡L3:** identity required (gates the wrapper).",
        "- **H transfers≠L3:** count of symmetric-diff buys; look-ahead can differ.",
        "- Climb sanity (realised points) is separate: `python -m src.strategy.layer4_climb`.",
        "",
        f"JSON: `data/processed/strategy_layer4_audit.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = p.parse_args(argv)
    rows = []
    for season, gw in DEFAULT_CASES:
        print(f"audit {season} GW{gw}...", flush=True)
        rows.append(audit_gw(season, gw, horizon=args.horizon))
    doc = {
        "defs": {
            "horizon": args.horizon,
            "gamma": GAMMA,
            "terminal_mu_weight": DEFAULT_TERMINAL_MU_WEIGHT,
            "terminal_value_weight": DEFAULT_TERMINAL_VALUE_WEIGHT,
        },
        "n_pass": sum(1 for r in rows if r["passed"]),
        "n_cases": len(rows),
        "cases": rows,
    }
    out_json = PROCESSED / "strategy_layer4_audit.json"
    out_md = REPORTS / "strategy_layer4_audit.md"
    out_json.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    write_report(rows, out_md)
    print(json.dumps({"n_pass": doc["n_pass"], "n_cases": doc["n_cases"]}, indent=2))
    print(f"wrote {out_md}")
    return 0 if doc["n_pass"] == doc["n_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
