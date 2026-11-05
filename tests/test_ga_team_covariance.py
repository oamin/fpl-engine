"""Smoke tests for team G/A covariance dig."""

from __future__ import annotations

import pandas as pd

from src.eval.ga_team_covariance import analyse_club, run_season


def _toy_club() -> pd.DataFrame:
    # Two MIDs: A scores when B assists (positive cross); C is noise.
    rows = []
    for fx in range(1, 13):
        for eid, name, g, a, m in (
            (1, "Alpha", 1 if fx % 3 == 0 else 0, 0, 90),
            (2, "Beta", 0, 1 if fx % 3 == 0 else 0, 90),
            (3, "Gamma", 1 if fx == 1 else 0, 0, 90),
        ):
            rows.append(
                {
                    "element": eid,
                    "name": name,
                    "team": "Toy",
                    "position": "MID",
                    "fixture": fx,
                    "GW": fx,
                    "minutes": m,
                    "goals_scored": g,
                    "assists": a,
                }
            )
    return pd.DataFrame(rows)


def test_analyse_club_cross_detects_link() -> None:
    doc = analyse_club(_toy_club(), "Toy")
    assert not doc["skipped"]
    cross = doc["blocks"]["attack_cross_ga"]
    assert len(cross) > 0
    ab = cross.loc[
        (cross["name_a"] == "Alpha") & (cross["name_b"] == "Beta")
    ]
    assert len(ab) == 1
    assert float(ab.iloc[0]["corr"]) > 0.9


def test_run_season_pooled_keys() -> None:
    df = _toy_club()
    # Need a second club so groupby is realistic
    other = df.copy()
    other["team"] = "Toy2"
    other["element"] = other["element"] + 10
    other["name"] = other["name"] + "2"
    doc = run_season(pd.concat([df, other], ignore_index=True), season="toy")
    assert doc["pooled"]["attack_cross_ga"]["n_pairs"] >= 2
    assert "attack_active_cross_ga" in doc["pooled"]
