"""The 20-week power check uses centred gameweek residuals."""

from __future__ import annotations

import unittest

import numpy as np

from src.eval.power import minimum_detectable, power_at


class PowerTest(unittest.TestCase):
    def test_a_zero_effect_on_a_flat_series_is_not_detected(self) -> None:
        self.assertEqual(
            power_at(np.zeros(8), 0.0, n_weeks=20, n_sims=12, n_boot=30, seed=0),
            0.0,
        )
        self.assertEqual(
            power_at(np.zeros(8), 0.05, n_weeks=20, n_sims=12, n_boot=30, seed=0),
            1.0,
        )

    def test_the_smallest_detectable_effect_moves_with_the_noise(self) -> None:
        flat = minimum_detectable(
            np.ones(12), n_sims=12, n_boot=30, step=0.01, grid_max=0.05, n_weeks=20
        )
        self.assertEqual(flat["mde"], 0.01)
        self.assertEqual(flat["median_null_half_width"], 0.0)
        noisy = minimum_detectable(
            np.random.default_rng(0).normal(0.0, 1.0, size=60),
            n_sims=24,
            n_boot=40,
            step=0.25,
            grid_max=3.0,
            n_weeks=20,
            seed=0,
        )
        self.assertIsNotNone(noisy["mde"])
        self.assertGreater(float(noisy["mde"]), 0.25)
        self.assertGreater(noisy["median_null_half_width"], 0.0)


if __name__ == "__main__":
    unittest.main()
