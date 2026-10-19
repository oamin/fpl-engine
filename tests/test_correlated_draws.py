"""Correlated scoreline → player-xP draws."""

from __future__ import annotations

import unittest

import numpy as np

from src.markets.correlated_draws import (
    _toy_side,
    dirichlet_shares,
    draw_fixture,
    sample_lambdas,
)
from src.markets import scoreline as sl


class SampleLambdas(unittest.TestCase):
    def test_mean_near_centre(self) -> None:
        rng = np.random.default_rng(0)
        lh, la, rh = sample_lambdas(1.6, 1.1, -0.05, 2000, sigma_log_lam=0.1, rng=rng)
        self.assertAlmostEqual(float(lh.mean()), 1.6, delta=0.08)
        self.assertAlmostEqual(float(la.mean()), 1.1, delta=0.08)
        self.assertAlmostEqual(float(rh.mean()), -0.05, delta=0.02)


class DirichletShares(unittest.TestCase):
    def test_sums_to_one(self) -> None:
        rng = np.random.default_rng(1)
        w = np.array([0.2, 0.5, 0.3, 0.0])
        s = dirichlet_shares(w, strength=30.0, rng=rng)
        self.assertAlmostEqual(float(s.sum()), 1.0, places=6)
        self.assertEqual(float(s[3]), 0.0)

    def test_negative_corr_across_draws(self) -> None:
        rng = np.random.default_rng(2)
        w = np.array([0.4, 0.35, 0.25])
        draws = np.vstack([dirichlet_shares(w, strength=20.0, rng=rng) for _ in range(800)])
        c = np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]
        self.assertLess(c, -0.1)


class FixtureDraws(unittest.TestCase):
    def setUp(self) -> None:
        self.home = _toy_side("H", "H")
        self.away = _toy_side("A", "A")

    def test_mean_near_point_match_only(self) -> None:
        d = draw_fixture(
            1.7,
            1.1,
            -0.05,
            self.home,
            self.away,
            n_scenarios=500,
            seed=0,
            dirichlet_shares_flag=False,
            sigma_log_lam=0.10,
            sigma_idio=0.0,
        )
        err = np.max(np.abs(d.mean_xp() - d.mu_point))
        self.assertLess(err, 0.25)

    def test_zero_shock_matches_point(self) -> None:
        d = draw_fixture(
            1.5,
            1.2,
            0.0,
            self.home,
            self.away,
            n_scenarios=5,
            seed=0,
            dirichlet_shares_flag=False,
            sigma_log_lam=0.0,
            sigma_idio=0.0,
        )
        np.testing.assert_allclose(d.xp[0], d.mu_point, atol=1e-9)

    def test_teammate_goal_xp_positive_corr_match_shock(self) -> None:
        # Fixed shares: shared xG moves both mids' goal xP together.
        d = draw_fixture(
            1.8,
            1.0,
            -0.05,
            self.home,
            self.away,
            n_scenarios=600,
            seed=1,
            dirichlet_shares_flag=False,
            sigma_log_lam=0.18,
            sigma_idio=0.0,
        )
        ids = list(d.player_ids)
        i1, i2 = ids.index("H_M1"), ids.index("H_M2")
        c = float(np.corrcoef(d.xp_goals[:, i1], d.xp_goals[:, i2])[0, 1])
        self.assertGreater(c, 0.85)

    def test_dirichlet_only_negative_goal_corr(self) -> None:
        # Fixed pots: compositional shares push goal xP in opposite directions.
        d = draw_fixture(
            1.6,
            1.2,
            0.0,
            self.home,
            self.away,
            n_scenarios=800,
            seed=2,
            dirichlet_shares_flag=True,
            sigma_log_lam=0.0,
            sigma_idio=0.0,
            dirichlet_strength=15.0,
        )
        ids = list(d.player_ids)
        i1, i2 = ids.index("H_M1"), ids.index("H_M2")
        c = float(np.corrcoef(d.xp_goals[:, i1], d.xp_goals[:, i2])[0, 1])
        self.assertLess(c, -0.15)

    def test_home_attack_vs_away_gk_coupled(self) -> None:
        d = draw_fixture(
            1.9,
            0.9,
            -0.05,
            self.home,
            self.away,
            n_scenarios=600,
            seed=3,
            dirichlet_shares_flag=False,
            sigma_log_lam=0.20,
            sigma_idio=0.0,
        )
        ids = list(d.player_ids)
        i_m = ids.index("H_M1")
        i_gk = ids.index("A_GK")
        # Higher home λ ⇒ more home attack xP and worse away CS / more GC for away GK
        # so total away GK xP often falls → negative corr expected
        c = float(np.corrcoef(d.xp[:, i_m], d.xp[:, i_gk])[0, 1])
        self.assertLess(c, 0.0)

    def test_idio_dilutes_mid_mid_corr(self) -> None:
        base = draw_fixture(
            1.7,
            1.1,
            -0.05,
            self.home,
            self.away,
            n_scenarios=500,
            seed=4,
            dirichlet_shares_flag=False,
            sigma_log_lam=0.12,
            sigma_idio=0.0,
        )
        noisy = draw_fixture(
            1.7,
            1.1,
            -0.05,
            self.home,
            self.away,
            n_scenarios=500,
            seed=4,
            dirichlet_shares_flag=False,
            sigma_log_lam=0.12,
            sigma_idio=3.0,
        )
        ids = list(base.player_ids)
        i1, i2 = ids.index("H_M1"), ids.index("H_M2")
        c0 = float(np.corrcoef(base.xp[:, i1], base.xp[:, i2])[0, 1])
        c1 = float(np.corrcoef(noisy.xp[:, i1], noisy.xp[:, i2])[0, 1])
        self.assertGreater(c0, 0.9)
        self.assertLess(c1, c0 - 0.3)

    def test_sigma_out_column_on_frames(self) -> None:
        home = self.home.copy()
        away = self.away.copy()
        home["sigma_out"] = [2.0, 2.0, 4.0, 1.0]
        away["sigma_out"] = [2.0, 2.0, 2.0, 2.0]
        d = draw_fixture(
            1.7,
            1.1,
            -0.05,
            home,
            away,
            n_scenarios=600,
            seed=5,
            sigma_log_lam=0.05,
            dirichlet_shares_flag=False,
        )
        ids = list(d.player_ids)
        i1, i2 = ids.index("H_M1"), ids.index("H_M2")
        self.assertGreater(float(d.xp[:, i1].std(ddof=1)), float(d.xp[:, i2].std(ddof=1)))

    def test_matrix_moments_finite(self) -> None:
        mat = sl.build_matrix(1.4, 1.1, 0.0)
        mom = sl.team_moments(mat)
        self.assertTrue(all(np.isfinite(v) for v in mom.values()))


if __name__ == "__main__":
    unittest.main()
