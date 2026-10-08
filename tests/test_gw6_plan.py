"""Availability tags for the Gameweek 6 plan. No network."""

from __future__ import annotations

import unittest

from src.live.gw6_plan import news_minutes


class NewsMinutesTest(unittest.TestCase):
    def test_a_firm_starter_keeps_the_file(self) -> None:
        minutes, tag = news_minutes("a", None, 90.0, 10.0)
        self.assertEqual(minutes, 90.0)
        self.assertEqual(tag, "minutes file")

    def test_a_ruled_out_player_is_zero(self) -> None:
        minutes, tag = news_minutes("i", 0, 90.0, 90.0)
        self.assertEqual(minutes, 0.0)
        self.assertEqual(tag, "ruled out")

    def test_a_doubtful_player_is_scaled(self) -> None:
        minutes, tag = news_minutes("d", 75, 80.0, 90.0)
        self.assertEqual(minutes, 60.0)
        self.assertEqual(tag, "doubtful")

    def test_a_missing_file_uses_the_last_appearance(self) -> None:
        minutes, tag = news_minutes("a", 100, None, 70.0)
        self.assertEqual(minutes, 70.0)
        self.assertEqual(tag, "last observed")


if __name__ == "__main__":
    unittest.main()
