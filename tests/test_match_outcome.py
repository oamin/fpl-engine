"""match_outcome_probs / match_outcome_roll — bookie M and categorical rolls."""

from __future__ import annotations

import random
import unittest

import numpy as np

from src.markets.match_outcome import (
    match_outcome_probs,
    match_outcome_probs_from_decimals,
    match_outcome_probs_from_lambdas,
    match_outcome_roll,
    match_outcome_roll_many,
    probs_close_to_uniform_margin,
)
from src.markets.scoreline import GRID_MAX, build_matrix


class MatchOutcomeProbsTest(unittest.TestCase):
    def test_ah_ou_without_btts(self) -> None:
        probs = match_outcome_probs(
            p_ah=0.55,
            ah_line=-0.5,
            p_over_25=0.52,
            p_btts=None,
        )
        self.assertEqual(probs.markets, ("AH", "OU2.5"))
        self.assertEqual(probs.weights[2], 0.0)
        self.assertAlmostEqual(float(probs.matrix.sum()), 1.0, places=9)
        self.assertIn("xg_home", probs.moments)

    def test_with_btts(self) -> None:
        probs = match_outcome_probs(
            p_ah=0.48,
            ah_line=0.0,
            p_over_25=0.50,
            p_btts=0.55,
        )
        self.assertIn("BTTS", probs.markets)
        self.assertEqual(probs.weights[2], 1.0)

    def test_from_decimals(self) -> None:
        probs = match_outcome_probs_from_decimals(
            ah_home_dec=1.90,
            ah_away_dec=1.95,
            ah_line=-0.25,
            over_dec=1.85,
            under_dec=2.05,
        )
        self.assertGreater(probs.lam_h, 0.0)
        self.assertGreater(probs.lam_a, 0.0)

    def test_from_lambdas(self) -> None:
        probs = match_outcome_probs_from_lambdas(1.4, 1.1, rho=-0.05)
        mat = build_matrix(1.4, 1.1, -0.05)
        np.testing.assert_allclose(probs.matrix, mat)


class MatchOutcomeRollTest(unittest.TestCase):
    def test_roll_stays_in_support(self) -> None:
        probs = match_outcome_probs_from_lambdas(1.5, 1.2, 0.0)
        rng = random.Random(0)
        for _ in range(200):
            roll = match_outcome_roll(probs, rng)
            self.assertGreaterEqual(roll.goals_home, 0)
            self.assertGreaterEqual(roll.goals_away, 0)
            self.assertLessEqual(roll.goals_home, GRID_MAX)
            self.assertLessEqual(roll.goals_away, GRID_MAX)
            self.assertGreater(roll.p, 0.0)

    def test_empirical_mass_matches_m(self) -> None:
        """High-prob cells should appear more often than rare ones."""
        probs = match_outcome_probs_from_lambdas(1.6, 0.9, 0.0)
        rng = random.Random(1)
        rolls = match_outcome_roll_many(probs, 4000, rng)
        # Count (1,0) vs (0,5) if both in support
        c_10 = sum(1 for r in rolls if r.goals_home == 1 and r.goals_away == 0)
        c_05 = sum(1 for r in rolls if r.goals_home == 0 and r.goals_away == 5)
        p_10 = float(probs.matrix[1, 0])
        p_05 = float(probs.matrix[0, 5])
        if p_10 > 0 and p_05 > 0:
            self.assertGreater(c_10 / max(c_05, 1), (p_10 / p_05) * 0.25)

    def test_mean_near_moments(self) -> None:
        probs = match_outcome_probs_from_lambdas(1.3, 1.1, 0.0)
        rolls = match_outcome_roll_many(probs, 8000, random.Random(2))
        diag = probs_close_to_uniform_margin(probs, rolls)
        self.assertLess(diag["abs_err_h"], 0.08)
        self.assertLess(diag["abs_err_a"], 0.08)

    def test_numpy_generator(self) -> None:
        probs = match_outcome_probs_from_lambdas(1.0, 1.0, 0.0)
        roll = match_outcome_roll(probs, np.random.default_rng(0))
        self.assertIsInstance(roll.goals_home, int)


if __name__ == "__main__":
    unittest.main()
