"""Cascade long-horizon scorer uses Numagent 2.0 score_xp_v2."""

from __future__ import annotations

import unittest

import pandas as pd

from src.markets import numagent_xp
from src.markets.cascade_score_v2 import (
    SCORER,
    score_club_side,
    side_moments_from_lams,
)


class CascadeScoreV2Test(unittest.TestCase):
    def test_scorer_name(self) -> None:
        self.assertEqual(SCORER, "score_xp_v2")
        self.assertEqual(SCORER, numagent_xp.SCORE_NAME)

    def test_side_moments_positive(self) -> None:
        mom = side_moments_from_lams(1.6, 1.1)
        self.assertGreater(mom["xg"], 0.5)
        self.assertGreater(mom["xcs"], 0.0)
        self.assertGreaterEqual(mom["xgc"], 0.0)

    def test_score_club_side_finite(self) -> None:
        club = pd.DataFrame(
            {
                "player_id": ["1", "2"],
                "position": ["MID", "FWD"],
                "rG": [0.02, 0.04],
                "drG": [0.01, 0.01],
                "rA": [0.01, 0.01],
                "drA": [0.005, 0.005],
                "m": [90.0, 75.0],
                "sigma_m": [5.0, 10.0],
                "bonus_mu": [0.3, 0.4],
                "bonus_se": [0.1, 0.1],
                "card_mu": [-0.1, -0.1],
                "card_se": [0.05, 0.05],
                "defcon_mu": [6.0, 2.0],
                "defcon_sig": [2.0, 2.0],
                "defcon_n": [5, 5],
            }
        )
        out = score_club_side(
            club, lam_for=1.5, lam_against=1.0, alpha_g=0.96, alpha_a=0.90
        )
        self.assertTrue((out["xp"] > 0).all())
        self.assertAlmostEqual(float(out["sG"].sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
