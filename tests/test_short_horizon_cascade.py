"""Short-horizon scenario helpers."""

from __future__ import annotations

import random
import unittest

from src.models.short_horizon_cascade import (
    centreline_club_pots,
    sample_week_club_pots,
)


class SamplePotsTest(unittest.TestCase):
    def test_sampled_clubs_cover_fixtures(self) -> None:
        centre = {
            "arsenal": {"lam_scored": 2.0, "lam_assist": 1.5, "e_total": 3.0, "p_cs_mkt": 0.3},
            "chelsea": {"lam_scored": 1.2, "lam_assist": 0.9, "e_total": 2.5, "p_cs_mkt": 0.25},
            "fulham": {"lam_scored": 1.0, "lam_assist": 0.75, "e_total": 2.2, "p_cs_mkt": 0.2},
        }
        fx = [("arsenal", "chelsea")]
        out = sample_week_club_pots(fx, centre, random.Random(0), band=0.15)
        self.assertIn("arsenal", out)
        self.assertIn("chelsea", out)
        self.assertIn("fulham", out)  # bye keeps centreline
        self.assertAlmostEqual(out["fulham"]["lam_scored"], 1.0)


class CentrelineTest(unittest.TestCase):
    def test_picks_gw(self) -> None:
        pots = {
            (10, "arsenal"): [{"lam_scored": 1.8, "lam_assist": 1.0, "e_total": 3.0, "p_cs_mkt": 0.3}],
            (11, "arsenal"): [{"lam_scored": 2.2, "lam_assist": 1.0, "e_total": 3.2, "p_cs_mkt": 0.3}],
        }
        c = centreline_club_pots(pots, 10)
        self.assertAlmostEqual(c["arsenal"]["lam_scored"], 1.8)


if __name__ == "__main__":
    unittest.main()
