"""Eligibility is a filter on stored weeks. It does not repair the score."""

from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src.eval.eligibility import (
    FORBIDDEN,
    REQUIRED,
    filter_weeks,
    prior_season_available,
    week_eligible,
)
import src.eval.eligibility as eligibility


class EligibilityRuleTest(unittest.TestCase):
    def test_both_conditions_are_required(self) -> None:
        self.assertFalse(prior_season_available("2022-23"))
        self.assertTrue(prior_season_available("2023-24"))
        self.assertFalse(week_eligible(xg_populated=True, prior_season=False))
        self.assertFalse(week_eligible(xg_populated=False, prior_season=True))
        self.assertTrue(week_eligible(xg_populated=True, prior_season=True))

    def test_the_filter_drops_only_ineligible_weeks(self) -> None:
        rows = pd.DataFrame(
            {
                "season": ["2022-23", "2022-23", "2023-24", "2024-25"],
                "gw": [5, 16, 5, 10],
                "gap": [1.0, 2.0, 3.0, 4.0],
            }
        )
        flags = pd.DataFrame(
            {
                "season": ["2022-23", "2022-23", "2023-24", "2024-25"],
                "gw": [5, 16, 5, 10],
                "eligible": [False, False, True, True],
            }
        )
        kept = filter_weeks(rows, flags)
        self.assertEqual(kept["gap"].tolist(), [3.0, 4.0])

    def test_the_module_does_not_repair_the_score(self) -> None:
        source = inspect.getsource(eligibility)
        self.assertNotIn("fill_from", source)
        self.assertNotIn("compute_xp(", source)
        text = "\n".join(REQUIRED)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)
        self.assertIn("inconclusive", text)
