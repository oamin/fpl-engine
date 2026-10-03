"""Stage 41 — later weeks use that fixture's opening price.

The current week keeps ``score_xp``. A later week keeps the decision-week
share and minutes and takes that fixture's opening 1X2. A missing line
uses the club's earlier rate. A blank week is 0 for that week only.
A double is still one fixture. γ, the hold margin, the penalty, and
``score_xp`` stay put.

Screen: 2023/24 free-transfer climb, Gameweeks 5–38. Park unless the
season total is higher and the weeks where all twenty clubs play are higher.
A pass does not replace the published value.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.blank_context import clubs_by_gw
from src.models.open_horizon import (
    deadline_rate_history,
    make_horizon_scores,
    opening_pots_for_sheet,
    single_fixture_calendar,
)
from src.models.ridge_multiseason import build_one_season
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
GWS = list(range(5, 39))
SEASON = "2023-24"


def _points(weekly: pd.DataFrame, gws: list[int] | None = None) -> float:
    frame = weekly.loc[weekly["method"] == "xp_ft"]
    if gws is not None:
        frame = frame.loc[frame["gw"].isin(gws)]
    if frame.empty:
        return float("nan")
    return float(frame["xi_points_cap"].sum())


def run_season(season: str, code: str) -> dict[str, float]:
    """One season. Does not replace the published value."""
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    roster = load_vaastav_roster(season)
    slug = season.replace("-", "_")
    sheet = pd.read_csv(
        CACHE / f"merged_gw_{slug}.csv",
        usecols=["team", "kickoff_time", "was_home", "GW", "fixture"],
    )
    odds = pd.read_csv(CACHE / f"E0_{code}.csv")
    pots = opening_pots_for_sheet(odds, sheet)
    calendar = single_fixture_calendar(roster)
    covered = sum(1 for key in calendar if pots.get(key))
    cover = covered / len(calendar) if calendar else float("nan")
    if cover < 0.98:
        raise RuntimeError(f"{season} opening-price coverage is {cover:.3f}")
    history = deadline_rate_history(feat)
    callback = make_horizon_scores(pots, history, calendar, horizon=HORIZON)
    clubs = clubs_by_gw(roster)
    full = [gw for gw, names in clubs.items() if len(names) >= 20 and gw in GWS]
    base = run_ft_season(feat, {"xp": "score_xp"}, GWS, roster=roster, horizon=HORIZON)
    arm = run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        horizon=HORIZON,
        horizon_scores=callback,
    )
    base_total = _points(base)
    arm_total = _points(arm)
    base_full = _points(base, full)
    arm_full = _points(arm, full)
    summary = {
        "season": season,
        "base_total": base_total,
        "arm_total": arm_total,
        "base_full": base_full,
        "arm_full": arm_full,
        "passed": float(arm_total > base_total and arm_full > base_full),
        "pot_cover": cover,
        "n_full": float(len(full)),
    }
    tag = slug
    base.assign(arm="published").to_csv(
        PROCESSED / f"stage_41_open_horizon_base_{tag}.csv", index=False
    )
    arm.assign(arm="opening").to_csv(
        PROCESSED / f"stage_41_open_horizon_{tag}.csv", index=False
    )
    return summary


def run() -> dict[str, float]:
    feat = build_one_season(SEASON, "2324")
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    roster = load_vaastav_roster(SEASON)
    sheet = pd.read_csv(
        CACHE / "merged_gw_2023_24.csv",
        usecols=["team", "kickoff_time", "was_home", "GW", "fixture"],
    )
    odds = pd.read_csv(CACHE / "E0_2324.csv")
    pots = opening_pots_for_sheet(odds, sheet)
    calendar = single_fixture_calendar(roster)
    covered = sum(1 for key in calendar if pots.get(key))
    history = deadline_rate_history(feat)
    callback = make_horizon_scores(pots, history, calendar, horizon=HORIZON)
    clubs = clubs_by_gw(roster)
    full = [gw for gw, names in clubs.items() if len(names) >= 20 and gw in GWS]

    base = run_ft_season(
        feat, {"xp": "score_xp"}, GWS, roster=roster, horizon=HORIZON
    )
    arm = run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        horizon=HORIZON,
        horizon_scores=callback,
    )
    base_total = _points(base)
    arm_total = _points(arm)
    base_full = _points(base, full)
    arm_full = _points(arm, full)
    passed = arm_total > base_total and arm_full > base_full
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    base.assign(arm="published").to_csv(PROCESSED / "stage_41_open_horizon_base.csv", index=False)
    arm.assign(arm="opening").to_csv(PROCESSED / "stage_41_open_horizon.csv", index=False)
    summary = {
        "base_total": base_total,
        "arm_total": arm_total,
        "base_full": base_full,
        "arm_full": arm_full,
        "passed": float(passed),
        "pot_cover": covered / len(calendar) if calendar else float("nan"),
        "n_full": float(len(full)),
    }
    _write(summary, full)
    return summary


def _write(summary: dict[str, float], full: list[int]) -> None:
    gap = summary["arm_total"] - summary["base_total"]
    full_gap = summary["arm_full"] - summary["base_full"]
    verdict = "PASS" if summary["passed"] else "PARK"
    lines = [
        "# Stage 41 — opening-price horizon",
        "",
        "The current week keeps `score_xp`. A later week in the three-week "
        "hold keeps the decision-week share and minutes and uses that fixture's "
        "opening 1X2 and over/under. A missing line uses the club's earlier "
        "rate, not zero. A week with no fixture is 0 for that week only. "
        "A double is still scored once. The discount, the hold margin, the "
        "penalty, and `score_xp` are unchanged.",
        "",
        f"2023/24 Gameweeks 5–38. Published **{summary['base_total']:.0f}**. "
        f"Opening horizon **{summary['arm_total']:.0f}**. Gap **{gap:+.0f}**.",
        "",
        f"Weeks where all twenty clubs play ({len(full)} weeks): published "
        f"**{summary['base_full']:.0f}**, opening horizon **{summary['arm_full']:.0f}**, "
        f"gap **{full_gap:+.0f}**. The other weeks are **{gap - full_gap:+.0f}**.",
        "",
        f"Opening prices cover **{summary['pot_cover']:.0%}** of club-weeks that "
        "have a sheet row. The rest use that club's earlier rate.",
        "",
        f"**{verdict}.** A pass on this season does not replace the published value. "
        "The other seasons are the next gate, under the same rule. A gain that "
        "exists only on the blank weeks is not a pass.",
        "",
    ]
    (REPORTS / "stage_41_open_horizon.md").write_text("\n".join(lines), encoding="utf-8")


FOLLOW = [("2022-23", "2223"), ("2024-25", "2425"), ("2025-26", "2526")]


def run_follow() -> list[dict[str, float]]:
    """The other seasons, under the same rule. Does not replace the published value."""
    rows = [run_season(season, code) for season, code in FOLLOW]
    lines = [
        "# Stage 41 — the other seasons",
        "",
        "Same rule as 2023/24. A season passes when its total is higher and the "
        "weeks where all twenty clubs play are higher. Coverage below 0.98 stops "
        "the season. The published climb is not switched by this script.",
        "",
        "| season | published | opening | gap | full weeks published | full weeks opening | full gap | cover | result |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        gap = row["arm_total"] - row["base_total"]
        full_gap = row["arm_full"] - row["base_full"]
        mark = "pass" if row["passed"] else "park"
        lines.append(
            f"| {row['season']} | {row['base_total']:.0f} | {row['arm_total']:.0f} | "
            f"{gap:+.0f} | {row['base_full']:.0f} | {row['arm_full']:.0f} | "
            f"{full_gap:+.0f} | {row['pot_cover']:.0%} | {mark} |"
        )
    lines.append("")
    (REPORTS / "stage_41_open_horizon_follow.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "follow":
        print(run_follow())
    else:
        print(run())
