"""Cross-position value the beam does not try, on the three earlier seasons.

The search is unchanged. ``structural_gap`` is the same shadow as stage 44.
A rewrite is rejected when every season averages under 0.25 points of
three-week value a week and has fewer than three weeks at or above 1.
A season above either bar is inconclusive. The search stays either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.stage_44_search_counts import structural_gap

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

EARLIER: list[tuple[str, str]] = [
    ("2022-23", "2223"),
    ("2023-24", "2324"),
    ("2024-25", "2425"),
]
MEAN_MAX = 0.25
WEEKS_AT_LEAST_1 = 3


def season_quiet(mean_gap: float, weeks_over_1: float) -> bool:
    """True when this season is too small to justify a search rewrite."""
    if pd.isna(mean_gap) or pd.isna(weeks_over_1):
        return False
    return float(mean_gap) < MEAN_MAX and float(weeks_over_1) < WEEKS_AT_LEAST_1


def rewrite_rejected(rows: list[dict[str, float]]) -> bool:
    """True only when all three locked seasons are quiet."""
    if len(rows) != len(EARLIER):
        return False
    return all(
        season_quiet(float(row["mean_gap"]), float(row["weeks_over_1"])) for row in rows
    )


def _write(frame: pd.DataFrame, rejected: bool) -> None:
    lines = [
        "# Search shadow",
        "",
        "The published search is unchanged. Each row is the three-week value "
        "of the best cross-position pair the beam does not try, minus the "
        "move it did choose. The same shadow on 2025/26 was 0.06 a week, "
        "with no week at 1. A rewrite is rejected when every season here "
        f"averages under {MEAN_MAX:.2f} and has fewer than {WEEKS_AT_LEAST_1} "
        "weeks at or above 1. A season above either bar is inconclusive. "
        "The search stays in both cases.",
        "",
        "| season | weeks | mean value | weeks at or above 1 | largest week | quiet |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        quiet = season_quiet(float(row.mean_gap), float(row.weeks_over_1))
        lines.append(
            f"| {row.season} | {row.weeks:.0f} | {row.mean_gap:.3f} | "
            f"{row.weeks_over_1:.0f} | {row.max_gap:.2f} | "
            f"{'yes' if quiet else 'no'} |"
        )
    if rejected:
        call = (
            "Every season is under both bars. The rewrite is rejected. "
            "The search stays as it is."
        )
    else:
        call = (
            "At least one season is above a bar, or a season is missing. "
            "That is inconclusive. The search stays as it is."
        )
    lines += ["", call, ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "search_shadow.md").write_text("\n".join(lines), encoding="utf-8")


def run_season(season: str, code: str) -> dict[str, float]:
    row = structural_gap(season, code)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(PROCESSED / f"search_shadow_{code}.csv", index=False)
    return row


def run() -> dict[str, object]:
    rows = [run_season(season, code) for season, code in EARLIER]
    frame = pd.DataFrame(rows)
    rejected = rewrite_rejected(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED / "search_shadow.csv", index=False)
    _write(frame, rejected)
    return {"rejected": rejected, "seasons": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="")
    args = parser.parse_args()
    if args.season:
        matched = [pair for pair in EARLIER if pair[0] == args.season]
        if len(matched) != 1:
            raise SystemExit(f"unknown season {args.season}")
        print(run_season(*matched[0]), flush=True)
        return
    print(run(), flush=True)


if __name__ == "__main__":
    main()
