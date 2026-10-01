"""Points brought in by automatic substitutes."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.season_climb import points_from_subs


class PointsFromSubsTest(unittest.TestCase):
    def test_sums_only_players_who_came_in(self) -> None:
        final = pd.DataFrame(
            {
                "player_id": ["a", "b", "c"],
                "total_points": [2.0, 6.0, 5.0],
            }
        )
        self.assertEqual(points_from_subs(final, {"a", "b"}), 5.0)
        self.assertEqual(points_from_subs(final, {"a", "b", "c"}), 0.0)
