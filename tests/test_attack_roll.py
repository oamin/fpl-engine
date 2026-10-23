"""Generative attack dice on a rolled scoreline."""

from __future__ import annotations

import unittest

import numpy as np

from src.markets.attack_roll import (
    EMPTY_A,
    EMPTY_G,
    build_side_dice,
    expected_side_ga,
    roll_side_attack,
)


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

    def test_expected_ga_matches_mc_mean(self) -> None:
        dice = build_side_dice(
            ["A", "B", "C"],
            sG=[0.5, 0.3, 0.2],
            sA=[0.4, 0.4, 0.2],
            alpha_g=0.9,
            alpha_a=0.85,
            G_team=2.0,
        )
        k = 3
        eg, ea = expected_side_ga(dice, k)
        rng = np.random.default_rng(42)
        n_mc = 8000
        g_sum = {n: 0.0 for n in dice.names}
        a_sum = {n: 0.0 for n in dice.names}
        for _ in range(n_mc):
            roll = roll_side_attack(dice, goals_for=k, rng=rng)
            for s in roll.scorers:
                g_sum[s] += 1.0
            for a in roll.assisters:
                a_sum[a] += 1.0
        for n in dice.names:
            self.assertAlmostEqual(g_sum[n] / n_mc, eg[n], delta=0.08)
            self.assertAlmostEqual(a_sum[n] / n_mc, ea[n], delta=0.08)


if __name__ == "__main__":
    unittest.main()
