"""α_G / α_A: prior season + evolving live counts."""

from __future__ import annotations

import unittest

from src.markets import fpl_score_rates


class LiveAlphaPoolTest(unittest.TestCase):
    def test_pool_adds_live_to_prior(self) -> None:
        prior = {"match_goals": 100.0, "player_goals": 96.0, "assists": 90.0}
        live = {"match_goals": 50.0, "player_goals": 48.0, "assists": 45.0}
        pooled = fpl_score_rates.pool_counts(prior, live)
        self.assertAlmostEqual(pooled["player_g_per_match_g"], 144.0 / 150.0)
        self.assertAlmostEqual(pooled["a_per_match_g"], 135.0 / 150.0)
        self.assertTrue(pooled["includes_live"])

    def test_refresh_reads_live_log(self) -> None:
        if not fpl_score_rates.LIVE_PLAYER_GW.is_file():
            self.skipTest("live player_gw missing")
        doc = fpl_score_rates.refresh_pooled(write=False)
        self.assertIn("prior", doc)
        self.assertEqual(doc["prior"]["season"], "2025-26")
        live = doc.get("live")
        self.assertIsNotNone(live)
        assert live is not None
        self.assertGreater(float(live["match_goals"]), 0.0)
        pooled = doc["pooled"]
        self.assertTrue(pooled["includes_live"])
        # Pooled should sit near prior when live is small relative to 1038 MG
        self.assertGreater(float(pooled["player_g_per_match_g"]), 0.9)
        self.assertLess(float(pooled["player_g_per_match_g"]), 1.02)

    def test_asof_before_gw1_has_no_live(self) -> None:
        if not fpl_score_rates.LIVE_PLAYER_GW.is_file():
            self.skipTest("live player_gw missing")
        live = fpl_score_rates.live_counts(before_gw=1)
        self.assertIsNone(live)


if __name__ == "__main__":
    unittest.main()
