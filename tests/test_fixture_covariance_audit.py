"""As-of fixture covariance audit helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.markets.fixture_covariance_audit import (
    asof_groups_for_team,
    assert_asof_pairs_ignore_future,
)
from src.markets.minutes_covariance import load_merged


class AsOfPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_merged(season="2025-26")

    def test_poison_future_does_not_change_pairs(self) -> None:
        team = "Liverpool"
        self.assertTrue(assert_asof_pairs_ignore_future(self.df, team, 20))

    def test_later_gw_can_add_pairs(self) -> None:
        early = asof_groups_for_team(self.df, "Liverpool", 10)
        late = asof_groups_for_team(self.df, "Liverpool", 30)
        # Not required that late > early, but both must be lists
        self.assertIsInstance(early, list)
        self.assertIsInstance(late, list)


if __name__ == "__main__":
    unittest.main()
