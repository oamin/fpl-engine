"""The clean-fill bar, locked before the three seasons are opened."""

from __future__ import annotations

import unittest

from src.models.clean_fill import hypothesis_kept, link_id, season_kept


class CleanFillBarTest(unittest.TestCase):
    def test_link_uses_the_opta_code(self) -> None:
        codes = {111: "28"}
        self.assertEqual(link_id("7", 111, codes, "2024-25", "2023-24"), "2024-25:28")
        self.assertEqual(link_id("8", 999, codes, "2024-25", "2023-24"), "2023-24:8")
        self.assertEqual(link_id("9", None, codes, "2024-25", "2023-24"), "2023-24:9")

    def test_both_bars_keep_a_season(self) -> None:
        self.assertTrue(season_kept(2.0, 1.0))
        self.assertFalse(season_kept(1.99, 1.0))
        self.assertFalse(season_kept(2.0, 0.99))

    def test_a_missing_window_does_not_keep(self) -> None:
        self.assertFalse(season_kept(float("nan"), 3.0))

    def test_two_seasons_keep_the_hypothesis(self) -> None:
        self.assertTrue(hypothesis_kept([True, True, False]))
        self.assertFalse(hypothesis_kept([True, False, False]))
        self.assertFalse(hypothesis_kept([]))
