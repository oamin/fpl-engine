"""Stage 43 — a double is the two fixtures added.

The baseline is the opening horizon with a double scored once. This arm
adds the two opening pots, each with the deadline share and the per-match
minutes. The current week still uses ``score_xp``.

Screen: 2023/24, Gameweeks 5–38. Pass when the season is higher, the weeks
that are not full slates are higher, and the full slates are not more than
5 points behind. A pass does not replace the single-fixture horizon until
the other seasons are run the same way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.blank_context import clubs_by_gw
from src.models.open_horizon import attach_opening_horizon
from src.models.ridge_multiseason import build_one_season
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
GWS = list(range(5, 39))
SEASON = "2023-24"
FULL_SLACK = 5.0


def _points(weekly: pd.DataFrame, gws: list[int] | None = None) -> float:
    frame = weekly.loc[weekly["method"] == "xp_ft"]
    if gws is not None:
        frame = frame.loc[frame["gw"].isin(gws)]
    if frame.empty:
        return float("nan")
    return float(frame["xi_points_cap"].sum())


def run() -> dict[str, float]:
    feat = build_one_season(SEASON, "2324")
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    roster = load_vaastav_roster(SEASON)
    single = attach_opening_horizon(feat, doubles=False)
    both = attach_opening_horizon(feat, doubles=True)
    if single is None or both is None:
        raise RuntimeError(f"{SEASON} has no opening horizon")
    clubs = clubs_by_gw(roster)
    full = [gw for gw, names in clubs.items() if len(names) >= 20 and gw in GWS]
    other = [gw for gw in GWS if gw not in full and clubs.get(gw)]
    base = run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        horizon=HORIZON,
        horizon_scores=single,
        use_early_scores=False,
    )
    arm = run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        horizon=HORIZON,
        horizon_scores=both,
        use_early_scores=False,
    )
    base_total = _points(base)
    arm_total = _points(arm)
    base_full = _points(base, full)
    arm_full = _points(arm, full)
    base_other = _points(base, other)
    arm_other = _points(arm, other)
    passed = (
        arm_total > base_total
        and arm_other > base_other
        and arm_full >= base_full - FULL_SLACK
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    base.assign(arm="single").to_csv(PROCESSED / "stage_43_dgw_base.csv", index=False)
    arm.assign(arm="both").to_csv(PROCESSED / "stage_43_dgw.csv", index=False)
    summary = {
        "base_total": base_total,
        "arm_total": arm_total,
        "base_full": base_full,
        "arm_full": arm_full,
        "base_other": base_other,
        "arm_other": arm_other,
        "passed": float(passed),
        "n_full": float(len(full)),
        "n_other": float(len(other)),
    }
    _write(summary)
    return summary


def _write(summary: dict[str, float]) -> None:
    gap = summary["arm_total"] - summary["base_total"]
    full_gap = summary["arm_full"] - summary["base_full"]
    other_gap = summary["arm_other"] - summary["base_other"]
    verdict = "PASS" if summary["passed"] else "PARK"
    lines = [
        "# Stage 43 — both fixtures of a double",
        "",
        "The baseline scores a double as one fixture. This arm adds the two "
        "opening pots. Each uses the decision-week share and the per-match "
        "minutes. The current week keeps `score_xp`. A blank week is 0 for "
        "that week only.",
        "",
        f"2023/24 Gameweeks 5–38. One fixture **{summary['base_total']:.0f}**. "
        f"Both fixtures **{summary['arm_total']:.0f}**. Gap **{gap:+.0f}**.",
        "",
        f"Weeks where all twenty clubs play ({summary['n_full']:.0f}): "
        f"**{full_gap:+.0f}**. The other weeks ({summary['n_other']:.0f}): "
        f"**{other_gap:+.0f}**.",
        "",
        f"**{verdict}.** The season must be ahead, the other weeks must be ahead, "
        "and the full slates may be at most 5 points behind. A pass on this "
        "season does not replace the single-fixture horizon.",
        "",
    ]
    (REPORTS / "stage_43_dgw.md").write_text("\n".join(lines), encoding="utf-8")


FOLLOW = [("2022-23", "2223"), ("2024-25", "2425"), ("2025-26", "2526")]


def run_follow() -> list[dict[str, float]]:
    """The other seasons, under the same rule. Does not replace the horizon."""
    rows = []
    for season, code in FOLLOW:
        feat = build_one_season(season, code)
        feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
        roster = load_vaastav_roster(season)
        single = attach_opening_horizon(feat, doubles=False)
        both = attach_opening_horizon(feat, doubles=True)
        if single is None or both is None:
            raise RuntimeError(f"{season} has no opening horizon")
        clubs = clubs_by_gw(roster)
        full = [gw for gw, names in clubs.items() if len(names) >= 20 and gw in GWS]
        other = [gw for gw in GWS if gw not in full and clubs.get(gw)]
        base = run_ft_season(
            feat,
            {"xp": "score_xp"},
            GWS,
            roster=roster,
            horizon=HORIZON,
            horizon_scores=single,
            use_early_scores=False,
        )
        arm = run_ft_season(
            feat,
            {"xp": "score_xp"},
            GWS,
            roster=roster,
            horizon=HORIZON,
            horizon_scores=both,
            use_early_scores=False,
        )
        base_total = _points(base)
        arm_total = _points(arm)
        base_full = _points(base, full)
        arm_full = _points(arm, full)
        base_other = _points(base, other)
        arm_other = _points(arm, other)
        passed = (
            arm_total > base_total
            and arm_other > base_other
            and arm_full >= base_full - FULL_SLACK
        )
        tag = season.replace("-", "_")
        base.assign(arm="single").to_csv(PROCESSED / f"stage_43_dgw_base_{tag}.csv", index=False)
        arm.assign(arm="both").to_csv(PROCESSED / f"stage_43_dgw_{tag}.csv", index=False)
        rows.append(
            {
                "season": season,
                "base_total": base_total,
                "arm_total": arm_total,
                "base_full": base_full,
                "arm_full": arm_full,
                "base_other": base_other,
                "arm_other": arm_other,
                "passed": float(passed),
            }
        )
        print(rows[-1], flush=True)
    lines = [
        "# Stage 43 — the other seasons",
        "",
        "Same rule as 2023/24. The season must be ahead, the weeks that are not "
        "full slates must be ahead, and the full slates may be at most 5 points behind.",
        "",
        "| season | one fixture | both | gap | full gap | other gap | result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        mark = "pass" if row["passed"] else "park"
        lines.append(
            f"| {row['season']} | {row['base_total']:.0f} | {row['arm_total']:.0f} | "
            f"{row['arm_total'] - row['base_total']:+.0f} | "
            f"{row['arm_full'] - row['base_full']:+.0f} | "
            f"{row['arm_other'] - row['base_other']:+.0f} | {mark} |"
        )
    lines.append("")
    (REPORTS / "stage_43_dgw_follow.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "follow":
        print(run_follow())
    else:
        print(run())
