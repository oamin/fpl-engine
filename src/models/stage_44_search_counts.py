"""Counts that do not change the published list or the search.

5-2-3: how often that shape is the best XI on the eligible pool, the mean
score advantage when it is, and the actual points. The published formation
list is not edited.

Cross-position: on one season, the value the beam kept against the best
cross-position pair it did not try. The gap is expected value, not points.
A mean near a tenth of a point is not worth a search rewrite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.season_climb import FORMATIONS, pick_xi
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.rules.fpl_2026 import OFFICIAL_FORMATIONS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
GWS = list(range(5, 39))
WITH_523 = list(FORMATIONS) + [(5, 2, 3)]


def formation_count(season: str, code: str) -> dict[str, float]:
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    wins = 0
    gaps = []
    actual_gaps = []
    weeks = 0
    for gw in GWS:
        pool = feat.loc[(feat["gw"] == gw) & feat["eligible"]].copy()
        if pool["position"].nunique() < 4 or len(pool) < 15:
            continue
        published, _shape = pick_xi(pool, "score_xp", formations=list(FORMATIONS))
        official, shape = pick_xi(pool, "score_xp", formations=WITH_523)
        weeks += 1
        pub_score = float(pd.to_numeric(published["score_xp"], errors="coerce").sum())
        off_score = float(pd.to_numeric(official["score_xp"], errors="coerce").sum())
        pub_pts = float(pd.to_numeric(published["total_points"], errors="coerce").fillna(0).sum())
        off_pts = float(pd.to_numeric(official["total_points"], errors="coerce").fillna(0).sum())
        if shape == (5, 2, 3) and off_score > pub_score + 1e-9:
            wins += 1
            gaps.append(off_score - pub_score)
            actual_gaps.append(off_pts - pub_pts)
    return {
        "season": season,
        "weeks": float(weeks),
        "wins": float(wins),
        "mean_xp": float(sum(gaps) / len(gaps)) if gaps else 0.0,
        "mean_actual": float(sum(actual_gaps) / len(actual_gaps)) if actual_gaps else 0.0,
        "actual_sum": float(sum(actual_gaps)) if actual_gaps else 0.0,
    }


def structural_gap(season: str = "2025-26", code: str = "2526") -> dict[str, float]:
    """Mean value left by ignoring the cross-position pairs. One season."""
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    roster = load_vaastav_roster(season)
    decisions: list[dict] = []
    run_ft_season(
        feat,
        {"xp": "score_xp"},
        GWS,
        roster=roster,
        horizon=HORIZON,
        decisions=decisions,
        shadow_structural=True,
    )
    gaps = [
        float(row["margin"])
        for row in decisions
        if row.get("role") == "structural_gap"
    ]
    mean = float(sum(gaps) / len(gaps)) if gaps else float("nan")
    return {
        "season": season,
        "weeks": float(len(gaps)),
        "mean_gap": mean,
        "weeks_over_1": float(sum(gap >= 1.0 for gap in gaps)),
        "max_gap": float(max(gaps)) if gaps else float("nan"),
    }


def run_shapes() -> pd.DataFrame:
    rows = [formation_count(season, code) for season, code in SEASONS]
    frame = pd.DataFrame(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "stage_44_formation.csv", index=False)
    return frame


def run() -> dict[str, object]:
    shapes = run_shapes()
    gap = structural_gap()
    pd.DataFrame([gap]).to_csv(PROCESSED / "stage_44_structural.csv", index=False)
    _write(shapes, gap)
    return {"shapes": shapes, "structural": gap}


def _write(shapes: pd.DataFrame, gap: dict[str, float]) -> None:
    lines = [
        "# Stage 44 — shapes and the cross-position gap",
        "",
        "5-2-3 is counted on the eligible pool, not inside a 15-man squad. "
        "The published formation list is unchanged. Official formations are "
        f"{len(OFFICIAL_FORMATIONS)}. The published list omits 5-2-3 only.",
        "",
        "| season | weeks | 5-2-3 wins | mean xP when it wins | mean actual | actual sum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in shapes.itertuples(index=False):
        lines.append(
            f"| {row.season} | {row.weeks:.0f} | {row.wins:.0f} | {row.mean_xp:.2f} | "
            f"{row.mean_actual:.2f} | {row.actual_sum:.1f} |"
        )
    lines += [
        "",
        f"On {gap['season']}, the cross-position pairs the beam does not try "
        f"are worth **{gap['mean_gap']:.2f}** points of three-week value per "
        f"week, across {gap['weeks']:.0f} weeks. {gap['weeks_over_1']:.0f} weeks "
        f"are at least 1. The largest week is {gap['max_gap']:.2f}. "
        "This is the value of those pairs, not the points they would have scored. "
        "A mean near a tenth of a point is not a search rewrite.",
        "",
    ]
    (REPORTS / "stage_44_search_counts.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(run())
