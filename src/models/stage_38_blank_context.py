"""Stage 38 — fixture tags on the 2023/24 free-transfer climb.

Gameweek 29 has four matches. The published climb left 8 blanks there.
This run ranks a club with no fixture at 0 and writes the hold and the move.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.ridge_multiseason import build_one_season
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
GWS = list(range(5, 39))


def run() -> pd.DataFrame:
    feat = build_one_season("2023-24", "2324")
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    roster = load_vaastav_roster("2023-24")
    decisions: list[dict] = []
    weekly = run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        horizon=HORIZON,
        decisions=decisions,
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "stage_38_blank_context.csv", index=False)
    flat = []
    for row in decisions:
        for week in row["weeks"]:
            flat.append(
                {
                    "decision_gw": row["gw"],
                    "role": row["role"],
                    "n_transfers": row["n_transfers"],
                    "margin": row["margin"],
                    "value": row["value"],
                    "horizon_gw": week["gw"],
                    "xi_sum": week["xi_sum"],
                    "n_no_fixture": week["n_no_fixture"],
                }
            )
    pd.DataFrame(flat).to_csv(PROCESSED / "stage_38_blank_decisions.csv", index=False)
    _write(weekly, pd.DataFrame(flat))
    return weekly


def _write(weekly: pd.DataFrame, decisions: pd.DataFrame) -> None:
    xp = weekly.loc[weekly["method"] == "xp_ft"]
    gw29 = xp.loc[xp["gw"] == 29]
    blanks_29 = float(gw29["n_blank_final"].sum()) if len(gw29) else float("nan")
    total = float(xp["xi_points_cap"].sum()) if len(xp) else float("nan")
    hits = float(xp["hits"].sum()) if len(xp) else float("nan")
    looked = decisions.loc[
        (decisions["role"] == "hold") & (decisions["horizon_gw"] == 29)
    ]
    named = int((looked["n_no_fixture"] > 0).sum()) if len(looked) else 0
    lines = [
        "# Stage 38 — fixture context",
        "",
        "A club with no row in the week is ranked at 0. A score from an earlier "
        "week is not used. A week with no clubs is skipped. The points model and "
        "the transfer penalty are unchanged. The hold and the chosen move each "
        "record the XI sum and the number of no-fixture starters in every week "
        "of the three-week hold.",
        "",
        f"2023/24 Gameweeks 5–38: **{total:.0f}** points, {hits:.0f} hits, "
        f"{int(xp['n_blank_final'].sum()) if len(xp) else 0} blanks after substitutes.",
        "",
        f"Gameweek 29 blanks after substitutes: **{blanks_29:.0f}**. "
        "The previous climb on this scorer left 8.",
        "",
        f"Hold records that still name a no-fixture player inside the Gameweek 29 "
        f"XI, when that week is inside the three-week window: **{named}**.",
        "",
    ]
    (REPORTS / "stage_38_blank_context.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    weekly = run()
    xp = weekly.loc[weekly["method"] == "xp_ft"]
    gw29 = xp.loc[xp["gw"] == 29, "n_blank_final"]
    print(
        f"points {xp['xi_points_cap'].sum():.0f}  "
        f"blanks {xp['n_blank_final'].sum():.0f}  "
        f"gw29 {float(gw29.sum()) if len(gw29) else float('nan'):.0f}"
    )


if __name__ == "__main__":
    main()
