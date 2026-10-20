"""Props shape goal π; M sets the team goal level (row 1)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.markets.player_attack import player_attack_from_m, player_attack_on_roll
from src.markets.prop_shares import (
    allocate_goals_with_pi,
    anytime_p_to_lambda,
    prop_goal_pi,
    rescale_goal_shares,
)


class PropSharesTest(unittest.TestCase):
    def test_anytime_to_lambda(self) -> None:
        self.assertAlmostEqual(anytime_p_to_lambda(0.0), 0.0)
        self.assertAlmostEqual(anytime_p_to_lambda(1 - math.e**-0.5), 0.5, places=5)

    def test_props_shape_relative_not_absolute_pin(self) -> None:
        """Covered λ is a weight; E[G] = π · G_team (level from M)."""
        s0 = np.array([0.35, 0.30, 0.25, 0.10])
        G = 2.0
        prop_p = {3: 0.40}  # depressed returner, λ≈0.511
        pi, diag = prop_goal_pi(s0, G_team=G, prop_p=prop_p)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=6)
        self.assertEqual(diag["mode"], "props_shape_m_level")
        lam = anytime_p_to_lambda(0.40)
        # Uncovered keep baseline E[G] as weight; covered uses λ.
        w_unc = float((s0[:3] * G).sum())
        want_pi3 = lam / (w_unc + lam)
        self.assertAlmostEqual(float(pi[3]), want_pi3, places=5)
        eg = pi * G
        self.assertAlmostEqual(float(eg.sum()), G, places=6)
        # Absolute λ is NOT pinned when Σw ≠ G.
        self.assertNotAlmostEqual(float(eg[3]), lam, places=2)

    def test_covered_ratio_preserved(self) -> None:
        """Two covered props keep λ ratio after renormalise onto G_team."""
        s0 = np.array([0.5, 0.5])
        G = 2.0
        # λ(0.55)/λ(0.30) ≈ 0.799/0.357
        prop_p = {0: 0.55, 1: 0.30}
        pi, diag = rescale_goal_shares(s0, G_team=G, prop_p=prop_p)
        lam0 = anytime_p_to_lambda(0.55)
        lam1 = anytime_p_to_lambda(0.30)
        self.assertAlmostEqual(float(pi[0] / pi[1]), lam0 / lam1, places=5)
        self.assertAlmostEqual(float((pi * G).sum()), G, places=6)
        self.assertEqual(diag["mode"], "props_shape_m_level")

    def test_star_baseline_can_be_reshaped_down(self) -> None:
        """Row 1: prop below baseline share weight cuts the star (lift-only retired)."""
        s0 = np.array([0.55, 0.25, 0.20])
        G = 2.0
        # baseline E[G]_0 = 1.10; λ(0.55)≈0.80 — relative reshape cuts him
        prop_p = {0: 0.55, 1: 0.30}
        pi, _ = prop_goal_pi(s0, G_team=G, prop_p=prop_p)
        eg = pi * G
        self.assertLess(float(eg[0]), float(s0[0] * G))
        self.assertAlmostEqual(float(eg.sum()), G, places=6)

    def test_all_covered_renormalises_to_pot(self) -> None:
        s0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        G = 0.8
        prop_p = {i: 0.55 for i in range(5)}
        pi, diag = prop_goal_pi(s0, G_team=G, prop_p=prop_p)
        eg = pi * G
        self.assertAlmostEqual(float(eg.sum()), G, places=6)
        self.assertTrue(diag["capped"])  # raw Σλ > G
        # Equal props → equal π
        self.assertTrue(np.allclose(pi, 0.2))

    def test_allocate_goals_with_pi_on_roll(self) -> None:
        pi = np.array([0.4, 0.35, 0.25])
        eg2 = allocate_goals_with_pi(2, pi)
        eg3 = allocate_goals_with_pi(3, pi)
        self.assertAlmostEqual(float(eg2.sum()), 2.0, places=6)
        self.assertAlmostEqual(float(eg3.sum()), 3.0, places=6)
        self.assertAlmostEqual(float(eg2[0]), 0.8, places=6)
        self.assertAlmostEqual(float(eg3[0]), 1.2, places=6)

    def test_player_attack_on_roll_assists_untouched_by_pi(self) -> None:
        pi = np.array([0.5, 0.5])
        sA = np.array([0.6, 0.4])
        eG, eA = player_attack_on_roll(pi, sA, goals_for=2, a_team=1.5)
        self.assertAlmostEqual(float(eG.sum()), 2.0)
        self.assertAlmostEqual(float(eA[0]), 0.9)
        self.assertAlmostEqual(float(eA[1]), 0.6)

    def test_player_attack_from_m_props_hook(self) -> None:
        pots, exp, diag = player_attack_from_m(
            [0.7, 0.3],
            [0.5, 0.5],
            xg=2.0,
            alpha_g=1.0,
            alpha_a=1.0,
            prop_p={0: 0.40},
            positions=["MID", "MID"],
        )
        self.assertIsNotNone(diag)
        assert diag is not None
        self.assertEqual(diag["mode"], "props_shape_m_level")
        self.assertAlmostEqual(float(exp.E_G.sum()), pots.G_team, places=6)
        # Assist channel unchanged by props (equal shares → equal E[A]).
        self.assertAlmostEqual(float(exp.E_A[0]), float(exp.E_A[1]), places=6)


if __name__ == "__main__":
    unittest.main()
