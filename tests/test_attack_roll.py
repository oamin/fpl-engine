"""Generative attack dice on a rolled scoreline."""

from __future__ import annotations

import unittest

import numpy as np

from src.markets.attack_roll import EMPTY_A, EMPTY_G, build_side_dice, roll_side_attack


class AttackRollTest(unittest.TestCase):
    def test_scorer_excluded_from_assist(self) -> None:
        dice = build_side_dice(
            ["A", "B"],
            sG=[0.6, 0.4],
            sA=[0.5, 0.5],
            alpha_g=1.0,
            alpha_a=1.0,
            G_team=2.0,
        )
        # Force scorer A every goal: pi concentrates on A
        dice.pi = np.array([1.0, 0.0])
        dice.rho = np.array([0.5, 0.5])
        rng = np.random.default_rng(0)
        roll = roll_side_attack(dice, goals_for=20, rng=rng, team="T")
        for ev in roll.events:
            self.assertEqual(ev.scorer, "A")
            self.assertNotEqual(ev.assister, "A")
            self.assertIn(ev.assister, ("B", EMPTY_A))

    def test_alpha_g_empty_mass(self) -> None:
        dice = build_side_dice(
            ["A"],
            sG=[1.0],
            sA=[1.0],
            alpha_g=0.0,
            alpha_a=1.0,
            G_team=1.0,
        )
        rng = np.random.default_rng(1)
        roll = roll_side_attack(dice, goals_for=10, rng=rng)
        self.assertEqual(roll.scorers, ())
        self.assertTrue(all(e.scorer == EMPTY_G for e in roll.events))

    def test_props_shape_pi(self) -> None:
        dice = build_side_dice(
            ["Star", "Other"],
            sG=[0.2, 0.8],
            sA=[0.5, 0.5],
            alpha_g=0.96,
            alpha_a=0.90,
            G_team=2.0,
            prop_p={0: 0.55},
        )
        self.assertGreater(float(dice.pi[0]), 0.2)
        self.assertAlmostEqual(float(dice.pi.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
