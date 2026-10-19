"""Independent audit of Layer-1 week MILP (no ownership / chips / horizon).

Builds a per-GW pool from closed ``score_xp_v2`` + prices from ``score_xp_*``,
solves ``solve_week``, and checks legality + dominance vs sequential
``pick_squad`` → ``pick_xi`` (captain = max μ in XI).

  python -m src.strategy.layer1_audit
  python -m src.strategy.layer1_audit --season 2025-26 --gws 10,20,30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.season_climb import pick_xi
from src.models.season_climb_budget import pick_squad
from src.rules.fpl_2026 import (
    BUDGET_TENTHS,
    MAX_PER_CLUB,
    SQUAD_QUOTA,
    normalize_position,
    squad_legal,
    xi_legal,
)
from src.strategy.layer1_week import WeekPlan, solve_week

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"

DEFAULT_CASES: list[tuple[str, int]] = [
    ("2022-23", 10),
    ("2023-24", 15),
    ("2024-25", 20),
    ("2025-26", 10),
    ("2025-26", 25),
    ("2025-26", 30),
]


def _season_stem(season: str) -> str:
    return season.replace("-", "_")


def load_pool(season: str, gw: int) -> pd.DataFrame:
    """One row per player: μ = Σ score_xp_v2 over fixtures that GW; value from score_xp."""
    v2 = pd.read_csv(CLOSED_V2)
    v2 = v2.loc[(v2["season"] == season) & (v2["gw"] == int(gw))].copy()
    if v2.empty:
        raise RuntimeError(f"no score_xp_v2 rows for {season} GW{gw}")
    v2["element"] = pd.to_numeric(v2["element"], errors="coerce")
    v2["score_xp_v2"] = pd.to_numeric(v2["score_xp_v2"], errors="coerce")
    agg = (
        v2.groupby("element", as_index=False)
        .agg(
            score_xp_v2=("score_xp_v2", "sum"),
            name=("name", "first"),
            position=("position", "first"),
            team=("team", "first"),
            n_fixtures=("fixture_id", "nunique"),
        )
    )
    prices = pd.read_csv(
        PROCESSED / f"score_xp_{_season_stem(season)}.csv",
        usecols=["element", "gw", "value", "team_norm", "position", "player_id"],
    )
    prices = prices.loc[prices["gw"] == int(gw)].drop_duplicates("element", keep="first")
    prices["element"] = pd.to_numeric(prices["element"], errors="coerce")
    prices["value"] = pd.to_numeric(prices["value"], errors="coerce")
    pool = agg.merge(prices[["element", "value", "team_norm", "player_id"]], on="element", how="inner")
    pool = pool.loc[np.isfinite(pool["score_xp_v2"]) & np.isfinite(pool["value"])].copy()
    pool["player_id"] = pool["player_id"].fillna(pool["element"]).astype(str)
    pool["club"] = pool["team_norm"].fillna(pool["team"]).astype(str)
    pool["position"] = pool["position"].map(normalize_position)
    pool["mu"] = pool["score_xp_v2"]
    return pool.reset_index(drop=True)


def sequential_objective(pool: pd.DataFrame, *, budget: int = BUDGET_TENTHS) -> dict[str, Any]:
    """Old path: MILP squad on μ, then greedy formation XI, captain = max μ in XI."""
    work = pool.copy()
    work["team_norm"] = work["club"].astype(str)
    squad = pick_squad(work, "score_xp_v2", budget=budget)
    xi, form = pick_xi(squad, "score_xp_v2")
    cap_row = xi.loc[xi["score_xp_v2"].idxmax()]
    xi_mu = float(xi["score_xp_v2"].sum())
    cap_mu = float(cap_row["score_xp_v2"])
    return {
        "objective": xi_mu + cap_mu,
        "xi_mu": xi_mu,
        "captain_mu": cap_mu,
        "captain_id": str(cap_row["player_id"]),
        "formation": form,
        "cost": int(pd.to_numeric(squad["value"], errors="coerce").sum()),
        "squad_ids": set(squad["player_id"].astype(str)),
        "xi_ids": set(xi["player_id"].astype(str)),
    }


def check_plan(plan: WeekPlan, pool: pd.DataFrame, *, budget: int) -> list[dict[str, Any]]:
    """Return a list of {name, pass, detail} checks."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    squad = plan.squad
    xi = plan.xi
    add("squad_size", len(squad) == 15, f"n={len(squad)}")
    add("xi_size", len(xi) == 11, f"n={len(xi)}")
    add(
        "squad_legal",
        squad_legal(
            squad["position"].astype(str).tolist(),
            squad["club"].astype(str).tolist(),
            [int(v) for v in squad["value"].tolist()],
            budget=budget,
        ),
        f"cost={plan.cost}",
    )
    add("xi_legal", xi_legal(xi["position"].astype(str).tolist()), f"form={plan.formation}")
    add("budget", plan.cost <= budget, f"{plan.cost} ≤ {budget}")
    club_ok = True
    for cl, n in squad["club"].value_counts().items():
        if int(n) > MAX_PER_CLUB:
            club_ok = False
            break
    add("club_cap", club_ok, str(squad["club"].value_counts().to_dict()))
    quota_ok = squad["position"].value_counts().to_dict() == dict(SQUAD_QUOTA)
    add("quota", quota_ok, str(squad["position"].value_counts().to_dict()))
    xi_ids = set(xi["player_id"].astype(str))
    squad_ids = set(squad["player_id"].astype(str))
    add("xi_subset_squad", xi_ids <= squad_ids, f"extra={xi_ids - squad_ids}")
    add("captain_in_xi", plan.captain_id in xi_ids, plan.captain_id)
    top = xi.sort_values("mu", ascending=False).iloc[0]
    add(
        "captain_is_max_mu_in_xi",
        str(top["player_id"]) == plan.captain_id,
        f"cap={plan.captain_id} top={top['player_id']} μ={float(top['mu']):.4f}",
    )
    add(
        "objective_identity",
        abs(plan.objective - (plan.xi_mu + plan.captain_mu)) < 1e-8,
        f"obj={plan.objective:.6f} xi+cap={plan.xi_mu + plan.captain_mu:.6f}",
    )
    # Recompute objective from pool μ
    mu_map = pool.set_index(pool["player_id"].astype(str))["mu"]
    rebuilt = float(mu_map.loc[list(xi_ids)].sum()) + float(mu_map.loc[plan.captain_id])
    add(
        "objective_matches_pool_mu",
        abs(rebuilt - plan.objective) < 1e-6,
        f"rebuilt={rebuilt:.6f} plan={plan.objective:.6f}",
    )
    add("no_duplicate_squad", squad["player_id"].nunique() == 15, "")
    return checks


def audit_gw(
    season: str,
    gw: int,
    *,
    budget: int = BUDGET_TENTHS,
    time_limit: float = 60.0,
) -> dict[str, Any]:
    pool = load_pool(season, gw)
    plan = solve_week(
        pool,
        mu_col="score_xp_v2",
        budget=budget,
        time_limit=time_limit,
    )
    checks = check_plan(plan, pool, budget=budget)
    seq = sequential_objective(pool, budget=budget)
    gap = float(plan.objective) - float(seq["objective"])
    # Joint must weakly dominate sequential on the same μ board
    checks.append(
        {
            "name": "dominates_sequential",
            "pass": gap >= -1e-6,
            "detail": f"joint={plan.objective:.4f} seq={seq['objective']:.4f} gap={gap:+.4f}",
        }
    )
    all_pass = all(c["pass"] for c in checks)
    return {
        "season": season,
        "gw": int(gw),
        "n_pool": int(len(pool)),
        "n_dgw_players": int((pool.get("n_fixtures", 1) > 1).sum()) if "n_fixtures" in pool.columns else 0,
        "plan": plan.summary(),
        "captain_name": (
            str(plan.xi.loc[plan.xi["player_id"] == plan.captain_id, "name"].iloc[0])
            if "name" in plan.xi.columns
            else plan.captain_id
        ),
        "sequential": {
            "objective": seq["objective"],
            "captain_id": seq["captain_id"],
            "formation": f"1-{seq['formation'][0]}-{seq['formation'][1]}-{seq['formation'][2]}",
            "cost": seq["cost"],
        },
        "gap_joint_minus_seq": gap,
        "checks": checks,
        "pass": all_pass,
        "xi_names": plan.xi.sort_values("mu", ascending=False)[
            [c for c in ("name", "position", "club", "mu", "is_captain") if c in plan.xi.columns]
        ].to_dict(orient="records"),
    }


def write_report(results: list[dict[str, Any]], path: Path) -> None:
    n = len(results)
    n_pass = sum(1 for r in results if r["pass"])
    lines = [
        "# Strategy Layer 1 — independent audit",
        "",
        "Scope: single-week joint MILP on absolute `score_xp_v2` μ only.",
        "No ownership, SE, transfers, chips, or horizon.",
        "",
        f"Cases: **{n_pass}/{n} passed** all checks.",
        "",
        "## Checks per GW",
        "",
        "| season | gw | pool | joint obj | seq obj | gap | cost | form | captain | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in results:
        p = r["plan"]
        lines.append(
            f"| {r['season']} | {r['gw']} | {r['n_pool']} | {p['objective']:.2f} | "
            f"{r['sequential']['objective']:.2f} | {r['gap_joint_minus_seq']:+.3f} | "
            f"{p['cost']} | {p['formation']} | {r['captain_name']} | "
            f"{'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += ["", "## Failed checks (if any)", ""]
    any_fail = False
    for r in results:
        fails = [c for c in r["checks"] if not c["pass"]]
        if not fails:
            continue
        any_fail = True
        lines.append(f"### {r['season']} GW{r['gw']}")
        for c in fails:
            lines.append(f"- **{c['name']}**: {c['detail']}")
        lines.append("")
    if not any_fail:
        lines.append("None.")
        lines.append("")
    lines += [
        "## Method",
        "",
        "- Pool: `score_xp_v2_closed.csv` (μ summed over DGW fixtures) joined to `score_xp_*` values.",
        "- Joint: `src.strategy.layer1_week.solve_week`.",
        "- Sequential baseline: `pick_squad` then `pick_xi`; captain = max μ in XI.",
        "- Expectation: joint objective ≥ sequential (same μ, joint formation+captain).",
        "",
        f"JSON: `{(PROCESSED / 'strategy_layer1_audit.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=str, default=None)
    ap.add_argument("--gws", type=str, default=None, help="comma-separated GWs")
    ap.add_argument("--budget", type=int, default=BUDGET_TENTHS)
    args = ap.parse_args(argv)

    if args.season and args.gws:
        cases = [(args.season, int(g)) for g in args.gws.split(",") if g.strip()]
    else:
        cases = list(DEFAULT_CASES)

    results: list[dict[str, Any]] = []
    for season, gw in cases:
        print(f"auditing {season} GW{gw} …", flush=True)
        results.append(audit_gw(season, gw, budget=args.budget))
        print(
            f"  {'PASS' if results[-1]['pass'] else 'FAIL'}  "
            f"obj={results[-1]['plan']['objective']:.2f}  "
            f"gap_vs_seq={results[-1]['gap_joint_minus_seq']:+.3f}",
            flush=True,
        )

    out_md = REPORTS / "strategy_layer1_audit.md"
    out_json = PROCESSED / "strategy_layer1_audit.json"
    write_report(results, out_md)
    out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
