"""Rebuild per-season score_xp frames for closed seasons (local cache only).

Writes ``data/processed/score_xp_<season>.csv`` and a stacked
``data/processed/score_xp_closed.csv``. Does not overwrite the 2025/26-only
``xp_engine.csv`` / ``player_matches.csv``.

  python -m src.markets.build_score_xp_seasons
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.ridge_multiseason import SEASONS, build_one_season

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

KEEP = [
    "season",
    "element",
    "player_id",
    "player_name",
    "team_norm",
    "position",
    "gw",
    "date",
    "fixture_id",
    "minutes",
    "total_points",
    "score_xp",
    "score_exp_points",
    "n_prior",
    "xmi",
    "value",
]


def export_season(season: str, fd_code: str) -> dict[str, Any]:
    feat = build_one_season(season, fd_code)
    # element was stashed before season-prefixing player_id
    if "element" not in feat.columns:
        feat = feat.copy()
        feat["element"] = feat["player_id"].astype(str).str.split(":", n=1).str[-1]
    cols = [c for c in KEEP if c in feat.columns]
    out = feat[cols].copy()
    stem = f"score_xp_{season.replace('-', '_')}"
    path = PROCESSED / f"{stem}.csv"
    out.to_csv(path, index=False)
    return {
        "season": season,
        "n": int(len(out)),
        "gw_min": int(out["gw"].min()),
        "gw_max": int(out["gw"].max()),
        "path": str(path.relative_to(ROOT)),
    }


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for season, code in SEASONS:
        print(f"building {season} …")
        info = export_season(season, code)
        summaries.append(info)
        frames.append(pd.read_csv(PROCESSED / Path(info["path"]).name))
        print(f"  → {info['n']} rows  gw {info['gw_min']}-{info['gw_max']}")

    stacked = pd.concat(frames, ignore_index=True)
    stack_path = PROCESSED / "score_xp_closed.csv"
    stacked.to_csv(stack_path, index=False)
    summary = {
        "seasons": summaries,
        "n_total": int(len(stacked)),
        "stacked": str(stack_path.relative_to(ROOT)),
    }
    (PROCESSED / "score_xp_closed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"stacked {len(stacked)} → {stack_path}")


if __name__ == "__main__":
    main()
