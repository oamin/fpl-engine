"""Team-form FDR as live/diagnostic colour — never fed into the MILP.

As-of expanding club Σ pts (managers stripped). For fixtures in
t+LEAD … t+LEAD+DEPTH−1, report FDR = μ_own − μ_opp. Hard-stop L4 boards
stay priced t:t+1 only (`strategy_layer4_mupost_priced`, BOOKIE_DEPTH=2).

  python -m src.strategy.team_form_diagnostic --season 2025-26 --gw 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.strategy.layer4_bookie_path import BOOKIE_DEPTH, _load_merged_enriched, _schedule
from src.strategy.team_form_fdr import (
    DEFAULT_DEPTH,
    DEFAULT_LEAD,
    _form_asof,
    _team_gw_actuals,
)
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def snapshot_asof(
    merged: pd.DataFrame,
    decision_gw: int,
    *,
    lead: int = DEFAULT_LEAD,
    depth: int = DEFAULT_DEPTH,
) -> dict[str, Any]:
    """Form table + future FDR rows; as-of GW < decision_gw only."""
    panel = _team_gw_actuals(merged)
    form = _form_asof(panel, int(decision_gw))
    sched = _schedule(merged)
    mu_map = (
        dict(zip(form["team"].astype(str), form["team_mu"].astype(float), strict=True))
        if not form.empty
        else {}
    )
    fixtures: list[dict[str, Any]] = []
    for k in range(int(lead), int(lead) + int(depth)):
        tgt = int(decision_gw) + k
        week = sched.loc[sched["GW"] == tgt]
        for r in week.itertuples(index=False):
            home = norm_team(str(r.home_team))
            away = norm_team(str(r.away_team))
            mu_h = float(mu_map.get(home, float("nan")))
            mu_a = float(mu_map.get(away, float("nan")))
            fixtures.append(
                {
                    "target_gw": tgt,
                    "lead": k,
                    "home": home,
                    "away": away,
                    "fdr_home": mu_h - mu_a,
                    "fdr_away": mu_a - mu_h,
                    "team_mu_home": mu_h,
                    "team_mu_away": mu_a,
                    "priced_window": bool(k < BOOKIE_DEPTH),
                }
            )
    return {
        "decision_gw": int(decision_gw),
        "bookie_depth": BOOKIE_DEPTH,
        "lead": int(lead),
        "depth": int(depth),
        "n_manager_rows_dropped": int(panel.attrs.get("n_manager_rows_dropped", 0)),
        "form": form.to_dict(orient="records") if not form.empty else [],
        "fixtures": fixtures,
        "note": "diagnostic only — not a MILP board or terminal",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--gw", type=int, default=20)
    ap.add_argument("--lead", type=int, default=DEFAULT_LEAD)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--write", action="store_true", help="write JSON under data/processed/")
    args = ap.parse_args(argv)
    merged = _load_merged_enriched(args.season)
    doc = snapshot_asof(
        merged, args.gw, lead=args.lead, depth=args.depth
    )
    doc["season"] = args.season
    print(json.dumps({k: doc[k] for k in doc if k != "form"}, indent=2, default=str))
    print(f"form_teams={len(doc['form'])} fixtures={len(doc['fixtures'])}")
    if args.write:
        stem = args.season.replace("-", "_")
        path = PROCESSED / f"team_form_diagnostic_{stem}_gw{args.gw:02d}.json"
        path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        # flat fixture CSV for colour
        fx = pd.DataFrame(doc["fixtures"])
        csv = PROCESSED / f"team_form_diagnostic_{stem}_gw{args.gw:02d}.csv"
        fx.to_csv(csv, index=False)
        print(f"wrote {path}")
        print(f"wrote {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
