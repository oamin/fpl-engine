"""Fixture-level minutes covariance (DGW = two obs)."""

from __future__ import annotations

import unittest

import pandas as pd

from src.markets.minutes_covariance import club_minutes_panel


class MinutesCovariance(unittest.TestCase):
    def test_dgw_two_fixtures_are_two_rows(self) -> None:
        club = pd.DataFrame(
            {
                "element": [1, 2, 1, 2, 1, 2],
                "name": ["A", "B", "A", "B", "A", "B"],
                "position": ["MID", "MID", "MID", "MID", "MID", "MID"],
                "team": ["X"] * 6,
                "fixture": [10, 10, 11, 11, 12, 12],
                "GW": [1, 1, 1, 1, 2, 2],
                "minutes": [90, 0, 0, 90, 90, 30],
            }
        )
        panel, _ = club_minutes_panel(club, min_fixtures_with_minutes=1)
        self.assertEqual(list(panel.index), [10, 11, 12])
        self.assertEqual(panel.shape[0], 3)
        self.assertEqual(float(panel.loc[10, 1]), 90.0)
        self.assertEqual(float(panel.loc[11, 2]), 90.0)

    def test_substitutes_negative_corr(self) -> None:
        rows = []
        for fx in range(6):
            rows.append(
                {
                    "element": 1,
                    "name": "A",
                    "position": "FWD",
                    "fixture": fx,
                    "minutes": 90 if fx % 2 == 0 else 0,
                }
            )
            rows.append(
                {
                    "element": 2,
                    "name": "B",
                    "position": "FWD",
                    "fixture": fx,
                    "minutes": 0 if fx % 2 == 0 else 90,
                }
            )
        club = pd.DataFrame(rows)
        club["team"] = "X"
        club["GW"] = club["fixture"]
        panel, _ = club_minutes_panel(club, min_fixtures_with_minutes=2)
        c = float(panel.corr().loc[1, 2])
        self.assertLess(c, -0.9)


if __name__ == "__main__":
    unittest.main()
