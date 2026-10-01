"""Holdout selector only. The season climb is not run here."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.stage_31_churn import choose_holdout_arm


class Stage31ChurnTest(unittest.TestCase):
    def test_none_pass(self) -> None:
        screen = pd.DataFrame(
            {
                "penalty": [0.0, 2.0, 3.0],
                "xi_points": [1800.0, 1890.0, 1880.0],
                "delta_vs_xp": [10.0, 33.0, -5.0],
                "mean_transfers": [1.2, 0.7, 0.4],
            }
        )
        self.assertIsNone(choose_holdout_arm(screen))

    def test_highest_total_then_fewer_transfers(self) -> None:
        screen = pd.DataFrame(
            {
                "penalty": [0.0, 2.0, 3.0],
                "xi_points": [1920.0, 1920.0, 1900.0],
                "delta_vs_xp": [40.0, 40.0, 20.0],
                "mean_transfers": [1.1, 0.6, 0.3],
            }
        )
        chosen = choose_holdout_arm(screen)
        assert chosen is not None
        self.assertEqual(float(chosen["penalty"]), 2.0)
