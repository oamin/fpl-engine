"""Audit Layer 2 (μ_post shrinkage) vs Layer 1 on closed weeks.

Solver **legality / identity smoke** on closed μ boards — same class as Layer 1.
Moments for shrink are as-of (GW < t). This does **not** claim a forecast win;
that is `se_credibility` (paired on realised points).

  python -m src.strategy.layer2_audit
  python -m src.strategy.layer2_audit --mode constant
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.rules.fpl_2026 import BUDGET_TENTHS
from src.strategy.layer1_audit import DEFAULT_CASES, check_plan, load_pool
from src.strategy.layer1_week import solve_week
from src.strategy.layer2_week import solve_week_sigma

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLOSED_V2 = PROCESSED / "score_xp_v2_closed.csv"


def _load_pool_with_se(season: str, gw: int):
    """Pool + xp_se (RSS over DGW), using hist moments only inside solve via cohort on pool."""
    import pandas as pd

    pool = load_pool(season, gw)
    v2 = pd.read_csv(CLOSED_V2)
    v2 = v2.loc[(v2["season"] == season) & (v2["gw"] == int(gw))].copy()
    v2["element"] = pd.to_numeric(v2["element"], errors="coerce")
    v2["xp_se"] = pd.to_numeric(v2["xp_se"], errors="coerce").fillna(0.0)
    se = (
        v2.groupby("element", as_index=False)
        .agg(xp_se=("xp_se", lambda s: float((s.astype(float) ** 2).sum() ** 0.5)))
    )
    out = pool.merge(se, on="element", how="left")
    out["xp_se"] = out["xp_se"].fillna(0.0)
    return out


def audit_gw(season: str, gw: int, *, mode: str = "se") -> dict[str, Any]:
    pool = _load_pool_with_se(season, gw)
    # As-of moments: use same-GW pool only for audit simplicity would leak τ² from
    # the decision week μ distribution — acceptable for solver smoke; credibility
    # screen is the honest walk-forward. Prefer prior weeks when available.
    import pandas as pd

    v2 = pd.read_csv(CLOSED_V2)
    hist = v2.loc[(v2["season"] == season) & (v2["gw"] < int(gw))].copy()
    moments = None
    if not hist.empty:
        from src.strategy.shrinkage import estimate_cohort_moments

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

    l1 = solve_week(pool, mu_col="score_xp_v2", budget=BUDGET_TENTHS, time_limit=60.0)
    l2 = solve_week_sigma(
        pool,
        mu_col="score_xp_v2",
        se_col="xp_se",
        mode=mode,
        moments_by_pos=moments,
        budget=BUDGET_TENTHS,
        time_limit=60.0,
    )
    checks = check_plan(l2.plan, pool, budget=BUDGET_TENTHS)
    checks = [c for c in checks if c["name"] != "objective_matches_pool_mu"]
    checks.append(
        {
            "name": "post_objective_identity",
            "pass": abs(l2.plan.objective - (l2.plan.xi_mu + l2.plan.captain_mu)) < 1e-8,
            "detail": f"obj={l2.plan.objective:.6f}",
        }
    )
    overlap = len(
        set(l1.xi["player_id"].astype(str)) & set(l2.plan.xi["player_id"].astype(str))
    )
    mu_delta = float(l2.abs_objective - l1.objective)
    return {
        "season": season,
        "gw": int(gw),
        "mode": mode,
        "n_pool": int(len(pool)),
        "l1_objective_mu": float(l1.objective),
        "l2": l2.summary(),
        "xi_overlap": overlap,
        "abs_mu_delta": mu_delta,
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def write_report(results: list[dict[str, Any]], path: Path, *, mode: str) -> None:
    n_pass = sum(1 for r in results if r["pass"])
    lines = [
        "# Strategy Layer 2 — audit (μ_post shrinkage)",
        "",
        "Ownership and η·SE **penalties are scrapped**. Layer 2 shrinks μ → μ_post,",
        "then runs the same joint MILP on μ_post.",
        "",
        "**Scope:** solver legality / objective identity on closed μ boards (as-of",
        "moments, GW < t). Not a forecast proof — that is `reports/se_credibility.md`",
        "(paired GW log-score vs realised points).",
        "",
        f"Mode: `{mode}`. Cases: **{n_pass}/{len(results)} passed**.",
        "",
        "| season | gw | L1 μ | L2 abs-μ | Δμ | L2 post-obj | overlap | pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['season']} | {r['gw']} | {r['l1_objective_mu']:.2f} | "
            f"{r['l2']['abs_objective']:.2f} | {r['abs_mu_delta']:+.2f} | "
            f"{r['l2']['post_objective']:.2f} | {r['xi_overlap']}/11 | "
            f"{'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Δμ is absolute-μ of the L2 XI vs L1 (shrink can change the XI). Overlap is XI",
        "player-id intersection. Forecast size/uncertainty: `reports/se_credibility.md`.",
        "",
        f"JSON: `{(PROCESSED / 'strategy_layer2_audit.json').relative_to(ROOT)}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommended_mode() -> str:
    path = PROCESSED / "se_credibility.json"
    if not path.exists():
        return "none"
    doc = json.loads(path.read_text(encoding="utf-8"))
    return str(doc.get("verdict", {}).get("recommended_mode", "none"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["none", "constant", "se", "auto"],
        default="auto",
        help="auto = se_credibility recommended_mode (default)",
    )
    args = ap.parse_args(argv)
    mode = _recommended_mode() if args.mode == "auto" else args.mode
    print(f"Layer-2 audit mode={mode}" + (" (from se_credibility)" if args.mode == "auto" else ""), flush=True)
    results = []
    for season, gw in DEFAULT_CASES:
        print(f"auditing L2-shrink {season} GW{gw} mode={mode} …", flush=True)
        row = audit_gw(season, gw, mode=mode)
        results.append(row)
        print(
            f"  {'PASS' if row['pass'] else 'FAIL'}  "
            f"Δμ={row['abs_mu_delta']:+.2f}  overlap={row['xi_overlap']}/11",
            flush=True,
        )
    write_report(results, REPORTS / "strategy_layer2_audit.md", mode=mode)
    (PROCESSED / "strategy_layer2_audit.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {REPORTS / 'strategy_layer2_audit.md'}")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
