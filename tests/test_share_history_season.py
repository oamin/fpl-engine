"""Player shares / rates: same-season only (no prior-season fill)."""

from __future__ import annotations

import unittest

import pandas as pd

from src.markets import numagent_xp


class ShareHistorySeasonTest(unittest.TestCase):
    def test_asof_filters_gw_and_season(self) -> None:
        hist = pd.DataFrame(
            {
                "GW": [1, 2, 3, 1, 2],
                "season": ["2026-27"] * 3 + ["2025-26"] * 2,
                "element": [1, 1, 1, 1, 1],
                "minutes": [90, 90, 90, 90, 90],
                "goals_scored": [1, 0, 0, 5, 5],
                "assists": [0, 0, 0, 0, 0],
            }
        )
        cut = numagent_xp.asof_share_history(
            hist, before_gw=3, season="2026-27"
        )
        self.assertEqual(set(cut["GW"].tolist()), {1, 2})
        self.assertEqual(set(cut["season"].tolist()), {"2026-27"})
        rates = numagent_xp.player_rates_from_history(
            cut, id_col="element", m0=0.0, rate_se_floor=0.0
        )
        # Goals from this season only (1), not 10 from 2025/26
        self.assertAlmostEqual(float(rates.iloc[0]["rate_goals"]), 1.0)
        self.assertAlmostEqual(float(rates.iloc[0]["rG"]), 1.0 / 180.0)

    def test_mixed_season_raises(self) -> None:
        hist = pd.DataFrame(
            {
                "season": ["2025-26", "2026-27"],
                "minutes": [90.0, 90.0],
                "goals_scored": [1.0, 1.0],
                "assists": [0.0, 0.0],
                "element": [1, 1],
            }
        )
        with self.assertRaises(ValueError):
            numagent_xp.player_rates_from_history(hist, id_col="element")


if __name__ == "__main__":
    unittest.main()
