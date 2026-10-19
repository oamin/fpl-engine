"""Short-horizon advisor uses Numagent 2.0 score_xp_v2 mechanics."""

from __future__ import annotations

import random
import unittest

import numpy as np

from src.markets import numagent_xp
from src.markets.scoreline import build_matrix, team_moments
from src.models import short_horizon_cascade as sh


class ScorerIdentity(unittest.TestCase):
    def test_module_uses_score_xp_v2(self) -> None:
        self.assertEqual(sh.SCORER, "score_xp_v2")
        self.assertEqual(sh.SCORER, numagent_xp.SCORE_NAME)


class SampleMoments(unittest.TestCase):
    def test_sample_rebuilds_dixon_coles_moments(self) -> None:
        fit = {
            "fixture": 1,
            "lam_h": 1.6,
            "lam_a": 1.1,
            "rho": -0.05,
            "home_n": "a",
            "away_n": "b",
            "xg_home": 1.6,
            "xcs_home": 0.3,
            "xgc_home": 0.4,
            "xg_away": 1.1,
            "xcs_away": 0.2,
            "xgc_away": 0.5,
        }
        rng = random.Random(0)
        ov = sh.sample_fixture_moments(fit, rng, band=0.15)
        for key in (
            "xg_home",
            "xg_away",
            "xcs_home",
            "xcs_away",
            "xgc_home",
            "xgc_away",
        ):
            self.assertTrue(math_isfinite(ov[key]))
        # Moments must match rebuild from the sampled λ.
        mom = team_moments(build_matrix(ov["lam_h"], ov["lam_a"], fit["rho"]))
        self.assertAlmostEqual(ov["xg_home"], mom["xg_home"], places=9)
        self.assertAlmostEqual(ov["xcs_away"], mom["xcs_away"], places=9)
        # Band respects ±15% on λ.
        self.assertGreaterEqual(ov["lam_h"], fit["lam_h"] * 0.85 - 1e-9)
        self.assertLessEqual(ov["lam_h"], fit["lam_h"] * 1.15 + 1e-9)


class CentrelineMatchesBuild(unittest.TestCase):
    def test_moments_from_lams_consistent(self) -> None:
        mom = sh.moments_from_lams(1.4, 1.2, 0.0)
        again = team_moments(build_matrix(1.4, 1.2, 0.0))
        for k in again:
            self.assertAlmostEqual(mom[k], again[k], places=12)


def math_isfinite(x: float) -> bool:
    return bool(np.isfinite(x))


if __name__ == "__main__":
    unittest.main()
