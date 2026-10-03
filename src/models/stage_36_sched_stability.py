"""Stage 36 — is the scheduled-minutes score stable across seasons?

``pass_margin`` stays 34. That bar is for a claim that a new score should
replace ``score_xp``. This batch asks whether ``score_xp_sched`` keeps its
sign. The rule was locked before these seasons were run.

A season is killed when its fast XI is 100 or more behind, and that kill
makes the column unstable. A climbed season must not lose by 34 or more.
At least three climbed seasons must be level or ahead, at least three
seasons must be climbed, and the sum of the climbed gaps must be positive.
A stable result does not replace ``score_xp``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.sched_minutes import attach_scheduled_score, per_fixture_minutes
from src.models.season_climb_ft import HORIZON, load_vaastav_roster, run_ft_season
from src.models.stage_29_batch import KILL_GAP, PASS_MARGIN, fast_xi

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
GWS = list(range(5, 39))
# 2025/26 was already climbed. It is an input, not a rerun.
KNOWN_SEASON = "2025-26"
KNOWN_FAST_GAP = -4.0
KNOWN_FT_GAP = 9.0


def judge(records: list[dict[str, Any]]) -> str:
    """Locked rule. ``ft_gap`` is sched minus xp. A missing file is not a zero."""
    present = [row for row in records if not row.get("missing")]
    climbed = [row for row in present if row.get("climbed")]
    if len(climbed) < 3:
        return "INCONCLUSIVE"
    if any(row.get("killed") for row in present):
        return "UNSTABLE"
    gaps = [float(row["ft_gap"]) for row in climbed]
    if min(gaps) <= -PASS_MARGIN:
        return "UNSTABLE"
    if sum(gap >= 0.0 for gap in gaps) < 3:
        return "UNSTABLE"
    if sum(gaps) <= 0.0:
        return "UNSTABLE"
    return "STABLE"


def _cap(weekly: pd.DataFrame, method: str) -> float:
    rows = weekly.loc[weekly["method"] == method, "xi_points_cap"]
    return float(rows.sum()) if len(rows) else float("nan")


def weekly_gap(weekly: pd.DataFrame, base: str, other: str) -> pd.Series:
    left = weekly.loc[weekly["method"] == base].set_index("gw")["xi_points_cap"]
    right = weekly.loc[weekly["method"] == other].set_index("gw")["xi_points_cap"]
    return (right - left).dropna()


def gap_stats(delta: pd.Series) -> dict[str, float]:
    if delta.empty:
        return {"win_rate": float("nan"), "worst_gw": float("nan"), "std_gw": float("nan")}
    return {
        "win_rate": float((delta >= 0).mean()),
        "worst_gw": float(delta.min()),
        "std_gw": float(delta.std(ddof=0)),
    }


def perturbation(feat: pd.DataFrame) -> float:
    """Share of eligible GW5–38 rows where the two scores differ by more than 0.5."""
    sl = feat.loc[feat["gw"].isin(GWS) & feat["eligible"].astype(bool)]
    if sl.empty:
        return float("nan")
    gap = (
        pd.to_numeric(sl["score_xp_sched"], errors="coerce")
        - pd.to_numeric(sl["score_xp"], errors="coerce")
    ).abs()
    return float((gap > 0.5).mean())


def _prepare(season: str, code: str) -> pd.DataFrame:
    path = CACHE / f"merged_gw_{season.replace('-', '_')}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    feat = build_one_season(season, code)
    raw = pd.read_csv(path)
    feat = attach_scheduled_score(feat, per_fixture_minutes(raw, season))
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    return feat


def _known_weekly() -> pd.DataFrame:
    path = PROCESSED / "stage_35_sched.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run() -> tuple[pd.DataFrame, str]:
    records: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []

    known_weekly = _known_weekly()
    known_delta = weekly_gap(known_weekly, "xp_ft", "sched_ft")
    known_stats = gap_stats(known_delta)
    records.append(
        {
            "season": KNOWN_SEASON,
            "missing": False,
            "killed": False,
            "climbed": True,
            "fast_gap": KNOWN_FAST_GAP,
            "ft_gap": KNOWN_FT_GAP,
            "perturbation": float("nan"),
            **known_stats,
        }
    )

    for season, code in SEASONS:
        if season == KNOWN_SEASON:
            feat = _prepare(season, code)
            known = next(row for row in records if row["season"] == KNOWN_SEASON)
            known["perturbation"] = perturbation(feat)
            continue
        path = CACHE / f"merged_gw_{season.replace('-', '_')}.csv"
        if not path.exists():
            records.append(
                {
                    "season": season,
                    "missing": True,
                    "killed": False,
                    "climbed": False,
                    "fast_gap": float("nan"),
                    "ft_gap": float("nan"),
                    "perturbation": float("nan"),
                    "win_rate": float("nan"),
                    "worst_gw": float("nan"),
                    "std_gw": float("nan"),
                }
            )
            continue
        feat = _prepare(season, code)
        screen = fast_xi(feat, GWS, {"xp": "score_xp", "sched": "score_xp_sched"})
        screen = screen.copy()
        screen["season"] = season
        fast_gap = _cap(screen, "sched") - _cap(screen, "xp")
        killed = bool(fast_gap <= -KILL_GAP)
        row: dict[str, Any] = {
            "season": season,
            "missing": False,
            "killed": killed,
            "climbed": not killed,
            "fast_gap": fast_gap,
            "ft_gap": float("nan"),
            "perturbation": perturbation(feat),
            "win_rate": float("nan"),
            "worst_gw": float("nan"),
            "std_gw": float("nan"),
        }
        if not killed:
            roster = load_vaastav_roster(season)
            xp_ft = run_ft_season(
                feat, {"xp": "score_xp"}, GWS, roster=roster, horizon=HORIZON
            )
            sched_ft = run_ft_season(
                feat, {"sched": "score_xp_sched"}, GWS, roster=roster, horizon=HORIZON
            )
            climbed = pd.concat([xp_ft, sched_ft], ignore_index=True)
            climbed["season"] = season
            weekly_parts.append(climbed)
            delta = weekly_gap(climbed, "xp_ft", "sched_ft")
            row["ft_gap"] = float(delta.sum()) if len(delta) else float("nan")
            row.update(gap_stats(delta))
        else:
            weekly_parts.append(screen)
        records.append(row)
        print(
            f"{season} fast {row['fast_gap']:+.1f} "
            f"ft {row['ft_gap']:+.1f} killed {killed}",
            flush=True,
        )

    table = pd.DataFrame(records)
    verdict = judge(records)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "stage_36_sched_stability.csv", index=False)
    if weekly_parts:
        pd.concat(weekly_parts, ignore_index=True).to_csv(
            PROCESSED / "stage_36_sched_weekly.csv", index=False
        )
    _write(table, verdict)
    return table, verdict


def _fmt(value: float, digits: int = 0) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    if digits == 0:
        return f"{value:+.0f}" if abs(value) >= 1 or value == 0 else f"{value:+.2f}"
    return f"{value:.{digits}f}"


def _write(table: pd.DataFrame, verdict: str) -> None:
    lines = [
        "# Stage 36 — scheduled minutes, across seasons",
        "",
        "The +34 bar stays the bar for replacing `score_xp`. It is about one "
        "point a week, and a single season can clear it and then give the "
        "points back the next year. This test asks whether the scheduled-minutes "
        "score keeps its sign. The rule was locked before these seasons were read. "
        "2025/26 is the run already on file: fast XI −4, free-transfer climb +9.",
        "",
        "A season whose fast XI is 100 or more behind is not climbed, and that "
        "kill makes the column unstable. Stable means the worst climbed season "
        "loses by less than 34, at least three climbed seasons are level or "
        "ahead, at least three seasons were climbed, and the gaps sum to more "
        "than zero. Stable does not replace `score_xp`.",
        "",
        f"**{verdict}**",
        "",
        "| season | fast gap | transfer gap | weeks ahead or level | worst week | weekly sd | scores moved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        if row.missing:
            lines.append(f"| {row.season} | missing file | | | | | |")
            continue
        moved = "—" if not np.isfinite(row.perturbation) else f"{100 * row.perturbation:.0f}%"
        rate = "—" if not np.isfinite(row.win_rate) else f"{100 * row.win_rate:.0f}%"
        lines.append(
            f"| {row.season} | {_fmt(row.fast_gap)} | {_fmt(row.ft_gap)} | "
            f"{rate} | {_fmt(row.worst_gw)} | {_fmt(row.std_gw, 1)} | {moved} |"
        )
    lines += [
        "",
        "Weeks ahead or level, the worst week, and the weekly standard deviation "
        "are the free-transfer climb. Scores moved is the share of eligible "
        "rows in Gameweeks 5–38 where the two scores differ by more than 0.5. "
        "Those three columns are diagnostics. They are not a second gate.",
        "",
    ]
    (REPORTS / "stage_36_sched_stability.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    _table, verdict = run()
    print(verdict)


if __name__ == "__main__":
    main()
