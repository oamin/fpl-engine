"""Player attack from M moments × shares — unit tests (no network)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.markets.player_attack import (
    attack_xp_from_expectations,
    player_attack_expectations,
    player_attack_from_m,
    team_attack_pots,
)


class TeamAttackPots(unittest.TestCase):
    def test_pots_scale_xg(self) -> None:
        pots = team_attack_pots(2.0, alpha_g=0.96, alpha_a=0.90)
        self.assertAlmostEqual(pots.G_team, 1.92)
        self.assertAlmostEqual(pots.A_team, 1.80)
        self.assertAlmostEqual(pots.rel_g, 0.0)
        self.assertAlmostEqual(pots.rel_a, 0.0)

    def test_relative_se_from_alpha(self) -> None:
        pots = team_attack_pots(2.0, alpha_g=1.0, alpha_a=1.0, d_alpha_g=0.1, d_alpha_a=0.2)
        self.assertAlmostEqual(pots.rel_g, 0.1)
        self.assertAlmostEqual(pots.rel_a, 0.2)
        self.assertAlmostEqual(pots.d_G_team, 0.2)
        self.assertAlmostEqual(pots.d_A_team, 0.4)

    def test_relative_se_combines_xg(self) -> None:
        pots = team_attack_pots(2.0, alpha_g=1.0, alpha_a=1.0, d_alpha_g=0.0, d_xg=0.2)
        self.assertAlmostEqual(pots.rel_g, 0.1)
        self.assertAlmostEqual(pots.rel_a, 0.1)


class PlayerAttackExpectations(unittest.TestCase):
    def test_expectations_partition_pot(self) -> None:
        pots = team_attack_pots(1.5, alpha_g=0.96, alpha_a=0.90)
        sG = np.array([0.4, 0.6])
        sA = np.array([0.25, 0.75])
        exp = player_attack_expectations(sG, sA, pots)
        self.assertAlmostEqual(float(exp.E_G.sum()), pots.G_team)
        self.assertAlmostEqual(float(exp.E_A.sum()), pots.A_team)
        self.assertAlmostEqual(float(exp.E_G[0]), 0.4 * pots.G_team)

    def test_saka_shaped_mid_points(self) -> None:
        # Arsenal away Bournemouth GW20 2025/26 FD moment (approx).
        xg = 2.078965854134929
        alpha_g, alpha_a = 0.9601357082273113, 0.9050042408821035
        sG, sA = 0.13088690920638033, 0.15853528405420997
        pots, exp, prop_diag = player_attack_from_m(
            [sG],
            [sA],
            xg=xg,
            alpha_g=alpha_g,
            alpha_a=alpha_a,
            positions=["MID"],
        )
        self.assertIsNone(prop_diag)
        self.assertAlmostEqual(pots.G_team, alpha_g * xg, places=10)
        self.assertAlmostEqual(float(exp.E_G[0]), sG * pots.G_team, places=10)
        self.assertAlmostEqual(float(exp.E_A[0]), sA * pots.A_team, places=10)
        assert exp.xp_goals is not None and exp.xp_assists is not None
        self.assertAlmostEqual(float(exp.xp_goals[0]), float(exp.E_G[0]) * 5.0, places=10)
        self.assertAlmostEqual(float(exp.xp_assists[0]), float(exp.E_A[0]) * 3.0, places=10)

    def test_uncertainty_quadrature(self) -> None:
        pots = team_attack_pots(2.0, alpha_g=1.0, alpha_a=1.0, d_alpha_g=0.0)
        # Δs/s = 0.2, rel pot = 0 → ΔE/E = 0.2
        exp = player_attack_expectations(
            np.array([0.5]),
            np.array([0.5]),
            pots,
            dsG=np.array([0.1]),
            dsA=np.array([0.1]),
        )
        self.assertAlmostEqual(float(exp.dE_G[0]) / float(exp.E_G[0]), 0.2)
        # Add α SE: rel = 0.1 → ΔE/E = sqrt(0.2² + 0.1²)
        pots2 = team_attack_pots(2.0, alpha_g=1.0, alpha_a=1.0, d_alpha_g=0.1)
        exp2 = player_attack_expectations(
            np.array([0.5]),
            np.array([0.5]),
            pots2,
            dsG=np.array([0.1]),
            dsA=np.array([0.1]),
        )
        want = math.sqrt(0.2**2 + 0.1**2)
        self.assertAlmostEqual(float(exp2.dE_G[0]) / float(exp2.E_G[0]), want)


class AttackXp(unittest.TestCase):
    def test_fwd_goal_points(self) -> None:
        pots = team_attack_pots(1.0, alpha_g=1.0, alpha_a=1.0)
        exp = player_attack_expectations(np.array([1.0]), np.array([1.0]), pots)
        scored = attack_xp_from_expectations(exp, ["FWD"])
        assert scored.xp_goals is not None
        self.assertAlmostEqual(float(scored.xp_goals[0]), 4.0)


if __name__ == "__main__":
    unittest.main()
