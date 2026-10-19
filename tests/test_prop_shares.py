"""Prop → goal-share rescale (lift-only anytime overlay)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.markets.prop_shares import anytime_p_to_lambda, rescale_goal_shares


class PropSharesTest(unittest.TestCase):
    def test_anytime_to_lambda(self) -> None:
        self.assertAlmostEqual(anytime_p_to_lambda(0.0), 0.0)
        self.assertAlmostEqual(anytime_p_to_lambda(1 - math.e**-0.5), 0.5, places=5)

    def test_props_lift_depressed_rolling_share(self) -> None:
        """Returner with tiny baseline but strong prop gets mass (Havertz case)."""
        s0 = np.array([0.35, 0.30, 0.25, 0.10])
        G = 2.50  # room so lifts do not force a global cap
        prop_p = {3: 0.40}  # only the depressed returner
        s1, diag = rescale_goal_shares(s0, G_team=G, prop_p=prop_p)
        self.assertAlmostEqual(float(s1.sum()), 1.0, places=6)
        eg0 = s0 * G
        eg1 = s1 * G
        self.assertGreater(eg1[3], eg0[3])
        self.assertAlmostEqual(eg1[3], anytime_p_to_lambda(0.40), places=5)
        self.assertFalse(diag["capped"])
        self.assertEqual(diag["n_lifted"], 1)
        self.assertEqual(diag["mode"], "lift_only")

    def test_baseline_above_lambda_not_cut(self) -> None:
        """Star with E[G]>1 from share keeps it when λ(p) is lower (to-score ≠ E[G])."""
        s0 = np.array([0.55, 0.25, 0.20])
        G = 2.0
        # baseline E[G]_0 = 1.10; p=0.55 → λ≈0.80 < 1.10 → no cut
        prop_p = {0: 0.55, 1: 0.30}
        s1, diag = rescale_goal_shares(s0, G_team=G, prop_p=prop_p)
        eg0 = s0 * G
        eg1 = s1 * G
        self.assertAlmostEqual(float(eg1.sum()), G, places=6)
        self.assertAlmostEqual(eg1[0], eg0[0], places=5)
        self.assertEqual(diag["n_lifted"], 0)  # neither lift if λ ≤ baseline
        # index 1: λ(0.30)≈0.357 vs baseline 0.50 — no lift
        self.assertAlmostEqual(eg1[1], eg0[1], places=5)

    def test_props_exhausting_pot_scale_down(self) -> None:
        """Many strong lifts that would exceed G_team are scaled back."""
        s0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        G = 0.8
        # each λ(0.55)≈0.80; five lifts → sum ≫ G
        prop_p = {i: 0.55 for i in range(5)}
        s1, diag = rescale_goal_shares(s0, G_team=G, prop_p=prop_p)
        eg = s1 * G
        self.assertAlmostEqual(float(eg.sum()), G, places=6)
        self.assertTrue(diag["capped"])
        self.assertLess(diag["prop_scale"], 1.0)
        self.assertEqual(diag["n_lifted"], 5)


if __name__ == "__main__":
    unittest.main()
