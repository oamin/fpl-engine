"""Bookie horizon path — as-of freeze + arm shapes."""

from __future__ import annotations

import unittest

from src.strategy.layer4_bookie_path import (
    build_horizon,
    crude_fdr_scale,
)


class BookiePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hz_a = build_horizon(
            "2025-26", 12, "A_hard_stop", n_scenarios=16, seed=0
        )
        cls.hz_b = build_horizon(
            "2025-26", 12, "B_crude_fdr", n_scenarios=16, seed=0
        )
        cls.hz_c = build_horizon(
            "2025-26", 12, "C_decay", n_scenarios=16, seed=0
        )

    def test_hard_stop_h_is_priced_only(self) -> None:
        self.assertEqual(len(self.hz_a.pools), len(self.hz_a.priced_gws))
        self.assertEqual(self.hz_a.filler_gws, ())

    def test_extend_arms_longer(self) -> None:
        self.assertGreater(len(self.hz_b.pools), len(self.hz_a.pools))
        self.assertGreater(len(self.hz_c.pools), len(self.hz_a.pools))
        self.assertTrue(self.hz_b.filler_gws)
        self.assertTrue(self.hz_c.filler_gws)

    def test_mu_finite(self) -> None:
        for hz in (self.hz_a, self.hz_b, self.hz_c):
            for pool in hz.pools:
                self.assertTrue(pool["mu_post"].notna().all())
                self.assertGreater(len(pool), 50)

    def test_fdr_scale_bounds(self) -> None:
        strengths = {
            "A": {"att": 2.0, "defn": 0.8, "n": 10},
            "B": {"att": 0.8, "defn": 2.0, "n": 10},
        }
        s = crude_fdr_scale("A", "B", "MID", strengths)
        self.assertGreaterEqual(s, 0.55)
        self.assertLessEqual(s, 1.35)


if __name__ == "__main__":
    unittest.main()
