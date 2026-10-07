"""The search-shadow bar, locked before the three seasons are opened."""

from __future__ import annotations

import unittest

from src.models.search_shadow import rewrite_rejected, season_quiet


def _row(mean_gap: float, weeks_over_1: float) -> dict[str, float]:
    return {"mean_gap": mean_gap, "weeks_over_1": weeks_over_1}


class SearchShadowBarTest(unittest.TestCase):
    def test_three_quiet_seasons_reject_a_rewrite(self) -> None:
        rows = [_row(0.06, 0), _row(0.24, 2), _row(0.0, 0)]
        self.assertTrue(all(season_quiet(row["mean_gap"], row["weeks_over_1"]) for row in rows))
        self.assertTrue(rewrite_rejected(rows))

    def test_one_large_mean_is_not_a_rewrite(self) -> None:
        rows = [_row(0.06, 0), _row(0.25, 0), _row(0.10, 1)]
        self.assertFalse(season_quiet(0.25, 0))
        self.assertFalse(rewrite_rejected(rows))

    def test_three_weeks_at_one_is_not_quiet(self) -> None:
        self.assertFalse(season_quiet(0.10, 3))
        self.assertFalse(rewrite_rejected([_row(0.10, 3), _row(0.10, 0), _row(0.10, 0)]))

    def test_a_missing_season_does_not_reject(self) -> None:
        self.assertFalse(rewrite_rejected([_row(0.01, 0), _row(0.01, 0)]))

    def test_a_missing_mean_is_not_quiet(self) -> None:
        self.assertFalse(season_quiet(float("nan"), 0))
