"""Stage 37 — switch penalty 2.0 on the seasons that were not climbed.

2025/26 is already known: penalty 2.0 beat the default 1.0 by 46, with one
hit and three blank XI slots after substitutes, against three hits and eight
blanks. That season is an input. It does not count toward the out-of-sample
sum.

The arm replaces the default only if every climbed season loses by less than
34, at least two of the three earlier seasons are level or ahead, those three
sum to more than zero, and blank XI slots after substitutes do not rise on
those three seasons. Otherwise the default stays 1.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.models.ridge_multiseason import SEASONS
from src.models.season_climb_ft import load_vaastav_roster
from src.models.stage_29_batch import PASS_MARGIN
from src.models.stage_31_churn import _climb, _gws, _prepare, arm_stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OOS = ("2022-23", "2023-24", "2024-25")
KNOWN = {
    "season": "2025-26",
    "oos": False,
    "missing": False,
    "climbed": True,
    "delta": 46.0,
    "points_pen": 1914.0,
    "points_base": 1868.0,
    "hits_pen": 1.0,
    "hits_base": 3.0,
    "transfers_pen": 0.94,
    "transfers_base": 1.03,
    "blanks_pen": 3.0,
    "blanks_base": 8.0,
    "n_gw": 34.0,
}


def judge(records: list[dict[str, Any]]) -> str:
    """Locked rule. ``delta`` is penalty 2.0 minus the default 1.0."""
    climbed = [row for row in records if row.get("climbed") and not row.get("missing")]
    oos = [row for row in climbed if row.get("oos")]
    if len(climbed) < 3 or len(oos) < 3:
        return "INCONCLUSIVE"
    if min(float(row["delta"]) for row in climbed) <= -PASS_MARGIN:
        return "PARKED"
    if sum(float(row["delta"]) >= 0.0 for row in oos) < 2:
        return "PARKED"
    if sum(float(row["delta"]) for row in oos) <= 0.0:
        return "PARKED"
    if sum(float(row["blanks_pen"]) for row in oos) > sum(float(row["blanks_base"]) for row in oos):
        return "PARKED"
    return "REPLACE"


def _one_season(season: str) -> dict[str, Any]:
    if season not in {name for name, _code in SEASONS}:
        raise KeyError(season)
    path = ROOT / "data" / "cache" / f"merged_gw_{season.replace('-', '_')}.csv"
    if not path.exists():
        return {
            "season": season,
            "oos": True,
            "missing": True,
            "climbed": False,
            "delta": float("nan"),
            "points_pen": float("nan"),
            "points_base": float("nan"),
            "hits_pen": float("nan"),
            "hits_base": float("nan"),
            "transfers_pen": float("nan"),
            "transfers_base": float("nan"),
            "blanks_pen": float("nan"),
            "blanks_base": float("nan"),
            "n_gw": 0.0,
        }
    feat = _prepare(season)
    gws = _gws(feat)
    roster = load_vaastav_roster(season)
    base_w = _climb(feat, gws, roster, None, "xp")
    pen_w = _climb(feat, gws, roster, 2.0, "pen")
    base = arm_stats(base_w, "xp_ft")
    pen = arm_stats(pen_w, "pen_ft")
    return {
        "season": season,
        "oos": True,
        "missing": False,
        "climbed": True,
        "delta": pen["xi_points"] - base["xi_points"],
        "points_pen": pen["xi_points"],
        "points_base": base["xi_points"],
        "hits_pen": pen["hits"],
        "hits_base": base["hits"],
        "transfers_pen": pen["mean_transfers"],
        "transfers_base": base["mean_transfers"],
        "blanks_pen": pen["blank_final"],
        "blanks_base": base["blank_final"],
        "n_gw": pen["n_gw"],
    }


def run() -> tuple[pd.DataFrame, str]:
    records = [dict(KNOWN)]
    for season in OOS:
        print(f"{season}…", flush=True)
        row = _one_season(season)
        records.append(row)
        print(
            f"  gap {row['delta']:+.1f}  blanks {row['blanks_pen']:.0f} vs {row['blanks_base']:.0f}",
            flush=True,
        )
    table = pd.DataFrame(records)
    verdict = judge(records)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "stage_37_penalty2.csv", index=False)
    _write(table, verdict)
    return table, verdict


def _write(table: pd.DataFrame, verdict: str) -> None:
    lines = [
        "# Stage 37 — penalty 2.0",
        "",
        "XI and captain stay on expected points. Only the per-transfer penalty "
        "changes, from 1.0 to 2.0. The 2025/26 climb is the one already on file "
        "(+46, one hit, three blank starters after substitutes). It is not in "
        "the out-of-sample sum.",
        "",
        "The default becomes 2.0 only if no season loses by 34 or more, at least "
        "two of the three earlier seasons are level or ahead, those three sum "
        "to more than zero, and blank XI slots after substitutes do not rise "
        "across those three. Otherwise the default stays 1.0.",
        "",
        f"**{verdict}**",
        "",
        "| season | penalty 2.0 | default 1.0 | gap | hits | transfers/GW | blanks after subs | weeks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        if row.missing:
            lines.append(f"| {row.season} | missing file | | | | | | |")
            continue
        lines.append(
            f"| {row.season} | {row.points_pen:.0f} | {row.points_base:.0f} | "
            f"{row.delta:+.0f} | {row.hits_pen:.0f} vs {row.hits_base:.0f} | "
            f"{row.transfers_pen:.2f} vs {row.transfers_base:.2f} | "
            f"{row.blanks_pen:.0f} vs {row.blanks_base:.0f} | {row.n_gw:.0f} |"
        )
    lines.append("")
    (REPORTS / "stage_37_penalty2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _table, verdict = run()
    print(verdict)


if __name__ == "__main__":
    main()
