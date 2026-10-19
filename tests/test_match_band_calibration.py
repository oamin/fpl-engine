"""Match-band calibration helpers."""

from __future__ import annotations

import random
import unittest

from src.models.multistage_mc_milp import sample_scoreline
from src.models.opta_gf_ga_horizon import gf_ga_from_rank, match_prediction


class ZBandParamTest(unittest.TestCase):
    def test_wider_z_widens_season_band(self) -> None:
        narrow = gf_ga_from_rank(5, z_band=0.5)
        wide = gf_ga_from_rank(5, z_band=2.0)
        self.assertAlmostEqual(narrow["gf"], wide["gf"])
        self.assertLess(wide["gf_lo"], narrow["gf_lo"])
        self.assertGreater(wide["gf_hi"], narrow["gf_hi"])

    def test_wider_z_widens_match_lambda_band(self) -> None:
        n = match_prediction("Arsenal", "Fulham", z_band=0.5)
        w = match_prediction("Arsenal", "Fulham", z_band=2.0)
        self.assertAlmostEqual(n["lam_home"], w["lam_home"])
        self.assertLessEqual(w["lam_home_lo"], n["lam_home_lo"])
        self.assertGreaterEqual(w["lam_home_hi"], n["lam_home_hi"])


class CentreBlendTest(unittest.TestCase):
    def test_blend_one_is_centreline(self) -> None:
        rng = random.Random(0)
        # Many draws; with blend=1 λ sample equals centreline before Poisson.
        _, _, sh, sa = sample_scoreline(
            1.5,
            0.8,
            lam_h_lo=1.0,
            lam_h_hi=2.0,
            lam_a_lo=0.4,
            lam_a_hi=1.2,
            rng=rng,
            centre_blend=1.0,
        )
        self.assertAlmostEqual(sh, 1.5)
        self.assertAlmostEqual(sa, 0.8)


if __name__ == "__main__":
    unittest.main()
