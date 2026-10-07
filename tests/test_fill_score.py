"""A missing score is zero. The price is the budget column."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.season_climb_ft import _fill_score


class FillScoreTests(unittest.TestCase):
    def test_a_missing_score_does_not_become_the_price(self) -> None:
        frame = pd.DataFrame(
            {
                "score_xp": [np.nan, 4.2],
                "value": [56, 80],
            }
        )
        filled = _fill_score(frame, "score_xp")
        self.assertEqual(float(filled.iloc[0]), 0.0)
        self.assertAlmostEqual(float(filled.iloc[1]), 4.2)

    def test_another_points_column_still_fills_a_hole(self) -> None:
        frame = pd.DataFrame(
            {
                "score_xp": [np.nan],
                "score_exp_points": [3.2],
                "value": [56],
            }
        )
        filled = _fill_score(frame, "score_xp")
        self.assertAlmostEqual(float(filled.iloc[0]), 3.2)

    def test_a_real_score_is_left_alone(self) -> None:
        frame = pd.DataFrame(
            {
                "score_xp": [1.5],
                "score_exp_points": [9.0],
                "value": [100],
            }
        )
        filled = _fill_score(frame, "score_xp")
        self.assertAlmostEqual(float(filled.iloc[0]), 1.5)


if __name__ == "__main__":
    unittest.main()
