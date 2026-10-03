"""Metric helpers for the score calibration. No season file is read."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.stage_39_calibration import ablation, deciles, fit_row


class FitRowTests(unittest.TestCase):
    def test_perfect_rank_has_no_bias(self) -> None:
        pred = pd.Series([1.0, 2.0, 3.0, 4.0])
        actual = pd.Series([1.0, 2.0, 3.0, 4.0])
        row = fit_row(pred, actual)
        self.assertEqual(row["n"], 4)
        self.assertAlmostEqual(row["bias"], 0.0)
        self.assertAlmostEqual(row["mae"], 0.0)
        self.assertAlmostEqual(row["spearman"], 1.0)

    def test_cancelling_bias_still_has_absolute_error(self) -> None:
        pred = pd.Series([0.0, 4.0])
        actual = pd.Series([2.0, 2.0])
        row = fit_row(pred, actual)
        self.assertAlmostEqual(row["bias"], 0.0)
        self.assertAlmostEqual(row["mae"], 2.0)

    def test_removing_the_signal_drops_rank_agreement(self) -> None:
        frame = pd.DataFrame(
            {
                "xp": [1.0, 2.0, 3.0, 4.0],
                "noise": [0.2, 0.1, 0.0, 0.3],
                "total_points": [1.0, 2.0, 3.0, 4.0],
            }
        )
        drop = ablation(frame, "xp", "xp", "total_points")
        self.assertGreater(drop, 0.5)

    def test_deciles_cover_the_rows(self) -> None:
        pred = pd.Series(range(20), dtype=float)
        actual = pred * 2
        bins = deciles(pred, actual)
        self.assertEqual(int(bins["n"].sum()), 20)
        self.assertGreater(float(bins["mean_actual"].iloc[-1]), float(bins["mean_actual"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
