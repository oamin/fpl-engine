"""Run front-end stages 0–5."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features.baseline import run_baseline
from src.features.shares import run_shares
from src.ingest.fpl_odds import run_ingest
from src.models.player_mu import run_player_mu
from src.models.team_market import run_team_market

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def write_stage0() -> None:
    text = """# Stage 0 — Clean slate

## Removed
- Old analyse / λ / DefCon / match_intensity / parse_vaastav scripts
- `odds_data.py`, `build_match_odds_points.py` (replaced by `src.ingest`)
- Derived CSVs, plots, regimes, obsolete matrix Cursor rule

## Kept
- `.env.example`, `.gitignore`, `.venv`, `uv.lock`
- `.cursor/rules/odds-api-quota.mdc` (The Odds API is not called)
- `data/cache/odds_snapshot.json` (historical snapshot only; no bookmaker re-fetch)

## New layout
- `src/ingest/` — FPL + football-data odds
- `src/features/` — shares, baseline
- `src/models/` — team market eval, player μ
- `src/markets/shin.py` — Shin de-vig
- `reports/stage_*.md` — mini-reports
- `data/processed/` — pipeline outputs

## Next
`uv run python -m src.run_frontend`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stage_0_clean.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FPL front-end stages 0–5")
    parser.add_argument(
        "--from-stage",
        type=int,
        default=0,
        help="Start from stage N (0–5). Default 0 writes clean report then runs 1–5.",
    )
    args = parser.parse_args()

    if args.from_stage <= 0:
        write_stage0()
        print("Stage 0: wrote reports/stage_0_clean.md")

    if args.from_stage <= 1:
        s1 = run_ingest()
        print(
            f"Stage 1: fixtures={s1['n_fixtures_odds']} "
            f"joined={s1['n_player_joined']} rate={s1['join_rate']:.3f}"
        )

    if args.from_stage <= 2:
        s2 = run_shares()
        print(
            f"Stage 2: pots={s2['n_team_pos_pots']} "
            f"mean_max_share={s2['mean_max_points_share']:.3f}"
        )

    if args.from_stage <= 3:
        s3 = run_team_market()
        go = s3["go_positions"]
        print("Stage 3: go/no-go " + ", ".join(f"{k}={'Y' if v else 'N'}" for k, v in go.items()))

    if args.from_stage <= 4:
        s4 = run_baseline()
        print(
            f"Stage 4: n={s4.get('n')} corr(base,pts)={s4.get('corr_base_points', float('nan')):.3f} "
            f"lift={s4.get('lift_points', float('nan')):.2f}"
        )

    if args.from_stage <= 5:
        s5 = run_player_mu()
        for r in s5["by_position"]:
            print(
                f"Stage 5: {r['position']} R²_resid={r.get('r2_resid', float('nan')):.3f} "
                f"corr_μ={r.get('corr_mu_points', float('nan')):.3f} "
                f"lift={r.get('top_bottom_lift_mu', float('nan')):.2f}"
            )

    print("Done. Mini-reports in reports/stage_*.md")


if __name__ == "__main__":
    main()
