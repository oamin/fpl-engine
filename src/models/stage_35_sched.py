"""Stage 35 — scheduled-minutes score, fast XI first.

``score_xp`` is unchanged. ``score_xp_sched`` rebuilds the minutes terms
from the last three scheduled club weeks, blanks included. The screen is
2025/26, GW5–38. A capped total 100 or more behind ``score_xp`` stops
there. Inside that gap, the free-transfer climb is the test, because the
fast XI never sees a player who did not play.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.ridge_multiseason import EVAL_SEASON, build_one_season
from src.models.sched_minutes import attach_scheduled_score, per_fixture_minutes
from src.models.season_climb import summarize
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.stage_29_batch import KILL_GAP, fast_xi

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
GWS = list(range(5, 39))


def _prepare() -> pd.DataFrame:
    feat = build_one_season(EVAL_SEASON, "2526")
    raw = pd.read_csv(CACHE / "merged_gw_2025_26.csv")
    minutes = per_fixture_minutes(raw, EVAL_SEASON)
    feat = attach_scheduled_score(feat, minutes)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    return feat


def _total(weekly: pd.DataFrame, method: str) -> float:
    rows = weekly.loc[weekly["method"] == method]
    return float(rows["xi_points_cap"].sum()) if len(rows) else float("nan")


def run() -> dict[str, float]:
    feat = _prepare()
    screen = fast_xi(feat, GWS, {"xp": "score_xp", "sched": "score_xp_sched"})
    xp_total = _total(screen, "xp")
    sched_total = _total(screen, "sched")
    gap = sched_total - xp_total
    result = {
        "screen_xp": xp_total,
        "screen_sched": sched_total,
        "screen_gap": gap,
        "ft_xp": float("nan"),
        "ft_sched": float("nan"),
        "ft_gap": float("nan"),
    }
    climbed = gap > -KILL_GAP
    if climbed:
        roster = load_vaastav_roster(EVAL_SEASON)
        xp_ft = run_ft_season(
            feat, {"xp": "score_xp"}, GWS, roster=roster, horizon=HORIZON
        )
        sched_ft = run_ft_season(
            feat,
            {"sched": "score_xp_sched"},
            GWS,
            roster=roster,
            horizon=HORIZON,
        )
        weekly = pd.concat([xp_ft, sched_ft], ignore_index=True)
        result["ft_xp"] = _total(weekly, "xp_ft")
        result["ft_sched"] = _total(weekly, "sched_ft")
        result["ft_gap"] = result["ft_sched"] - result["ft_xp"]
        result["ft_hits_xp"] = float(xp_ft["hit_cost"].sum())
        result["ft_hits_sched"] = float(sched_ft["hit_cost"].sum())
    else:
        weekly = screen
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(PROCESSED / "stage_35_sched.csv", index=False)
    _write(result, climbed, feat)
    return result


def _write(result: dict[str, float], climbed: bool, feat: pd.DataFrame) -> None:
    both = feat["xmi_sched"].notna() & feat["score_xp"].notna()
    agree = both & (feat["xmi_sched"] >= 85) & ((feat["score_xp"] - feat["score_xp_sched"]).abs() < 0.05)
    lines = [
        "# Stage 35 — scheduled minutes",
        "",
        "OpenFPL and Dastan both find that weeks with no minutes are where "
        "public models separate. This score keeps those weeks in the minutes "
        "prior only. Share and the published `score_xp` stay as they are. "
        "A regular whose last three scheduled weeks were full matches is "
        "unchanged, because the chance of playing is 1.",
        "",
        f"Fast XI, 2025/26 GW5–38: `score_xp` **{result['screen_xp']:.0f}**, "
        f"`score_xp_sched` **{result['screen_sched']:.0f}**, "
        f"gap **{result['screen_gap']:+.0f}**. "
        f"The kill line is {KILL_GAP:.0f} behind.",
        "",
    ]
    if climbed:
        lines += [
            f"Free-transfer climb: `score_xp` **{result['ft_xp']:.0f}** "
            f"(hits {result['ft_hits_xp']:.0f}), "
            f"`score_xp_sched` **{result['ft_sched']:.0f}** "
            f"(hits {result['ft_hits_sched']:.0f}), "
            f"gap **{result['ft_gap']:+.0f}**.",
            "",
            "The fast XI only ranks players who played. The transfer climb is "
            "where a blank week can keep someone out of the side.",
            "",
        ]
    else:
        lines += [
            "The screen is 100 or more behind, so the free-transfer climb was not run.",
            "",
        ]
    lines += [
        f"Rows with a scheduled prior: {int(both.sum())} of {len(feat)}. "
        f"Full-match priors within 0.05 of `score_xp`: {int(agree.sum())}.",
        "",
    ]
    (REPORTS / "stage_35_sched.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = run()
    print(
        f"screen {result['screen_gap']:+.1f}  "
        f"ft {result['ft_gap']:+.1f}"
    )


if __name__ == "__main__":
    main()
