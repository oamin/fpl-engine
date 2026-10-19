"""Early-season rate M0 + SE floor."""

from __future__ import annotations

import unittest

import pandas as pd

from src.markets import numagent_xp as nx


class RateM0(unittest.TestCase):
    def test_pseudo_minutes_pulls_early_scorer(self) -> None:
        hist = pd.DataFrame(
            {
                "element": [1, 1],
                "GW": [1, 2],
                "season": ["2025-26", "2025-26"],
                "minutes": [90.0, 90.0],
                "goals_scored": [1, 0],
                "assists": [0, 0],
            }
        )
        raw = nx.player_rates_from_history(
            hist, id_col="element", m0=0.0, rate_se_floor=0.0
        )
        m0 = nx.player_rates_from_history(hist, id_col="element", m0=270.0)
        self.assertAlmostEqual(float(raw.iloc[0]["rG"]), 1.0 / 180.0, places=8)
        self.assertAlmostEqual(float(m0.iloc[0]["rG"]), 1.0 / 450.0, places=8)
        self.assertLess(float(m0.iloc[0]["rG"]), float(raw.iloc[0]["rG"]))

    def test_zero_event_se_floor(self) -> None:
        hist = pd.DataFrame(
            {
                "element": [1],
                "GW": [1],
                "season": ["2025-26"],
                "minutes": [90.0],
                "goals_scored": [0],
                "assists": [0],
            }
        )
        raw = nx.player_rates_from_history(
            hist, id_col="element", m0=0.0, rate_se_floor=0.0
        )
        flo = nx.player_rates_from_history(
            hist, id_col="element", m0=0.0, rate_se_floor=0.25
        )
        self.assertEqual(float(raw.iloc[0]["drG"]), 0.0)
        self.assertAlmostEqual(float(flo.iloc[0]["drG"]), 0.25 / 90.0, places=8)

    def test_legacy_mle_when_m0_zero(self) -> None:
        hist = pd.DataFrame(
            {
                "element": [1],
                "GW": [1],
                "season": ["2025-26"],
                "minutes": [90.0],
                "goals_scored": [1],
                "assists": [2],
            }
        )
        rates = nx.player_rates_from_history(
            hist, id_col="element", m0=0.0, rate_se_floor=0.0
        )
        self.assertAlmostEqual(float(rates.iloc[0]["rG"]), 1.0 / 90.0, places=8)
        self.assertAlmostEqual(float(rates.iloc[0]["drG"]), 1.0 / 90.0, places=8)
        self.assertAlmostEqual(float(rates.iloc[0]["rA"]), 2.0 / 90.0, places=8)
        self.assertAlmostEqual(float(rates.iloc[0]["drA"]), (2.0**0.5) / 90.0, places=8)

    def test_default_is_promoted_m0(self) -> None:
        hist = pd.DataFrame(
            {
                "element": [1],
                "GW": [1],
                "season": ["2025-26"],
                "minutes": [90.0],
                "goals_scored": [1],
                "assists": [0],
            }
        )
        rates = nx.player_rates_from_history(hist, id_col="element")
        self.assertAlmostEqual(float(rates.iloc[0]["rate_m0"]), nx.RATE_M0)
        self.assertAlmostEqual(
            float(rates.iloc[0]["rG"]), 1.0 / (90.0 + nx.RATE_M0), places=8
        )
        self.assertAlmostEqual(
            float(rates.iloc[0]["drG"]),
            max(1.0, nx.RATE_SE_FLOOR) / (90.0 + nx.RATE_M0),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
