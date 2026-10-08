"""Goal-rate cap for anytime prices. No network."""

from __future__ import annotations

import math
import unittest

from src.live.player_goals import (
    fold_name,
    index_players,
    match_player,
    poisson_mean,
    team_goal_rates,
)


class PlayerGoalsTest(unittest.TestCase):
    def test_a_shared_web_name_is_not_matched(self) -> None:
        elements = [
            {"id": 1, "first_name": "A", "second_name": "Gabriel", "web_name": "Gabriel"},
            {"id": 2, "first_name": "B", "second_name": "Gabriel", "web_name": "Gabriel"},
        ]
        index = index_players(elements)
        self.assertIsNone(match_player("Gabriel", index))
        self.assertEqual(fold_name("Erling Haaland"), "erlinghaaland")

    def test_the_poisson_mean_clips_the_price(self) -> None:
        probability, mean = poisson_mean([2.0, 2.0])
        self.assertAlmostEqual(probability, 0.5)
        self.assertAlmostEqual(mean, -math.log(0.5))

    def test_priced_rates_cannot_exceed_the_match_line(self) -> None:
        rates = team_goal_rates(
            [
                {"id": 1, "share": 0.5, "xmi": 90.0, "mu_raw": 1.5},
                {"id": 2, "share": 0.5, "xmi": 90.0, "mu_raw": None},
            ],
            2.0,
        )
        self.assertAlmostEqual(rates[1], 1.0)
        self.assertNotIn(2, rates)

    def test_minutes_scale_a_doubtful_player(self) -> None:
        rates = team_goal_rates(
            [{"id": 9, "share": 0.0, "xmi": 45.0, "mu_raw": 1.0}],
            2.0,
        )
        self.assertAlmostEqual(rates[9], 0.5)


if __name__ == "__main__":
    unittest.main()
