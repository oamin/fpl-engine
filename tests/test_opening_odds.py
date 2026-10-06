"""Opening prices only. A closing number must not fill an opening cell."""

from __future__ import annotations

import unittest

import pandas as pd

from src.ingest.fpl_odds import load_football_data, opening_prices


class OpeningOddsTest(unittest.TestCase):
    def test_avg_h_beats_avg_ch(self) -> None:
        prices = opening_prices(
            {
                "AvgCH": "1.2",
                "AvgCD": "6.0",
                "AvgCA": "12.0",
                "AvgH": "3.0",
                "AvgD": "3.4",
                "AvgA": "2.4",
                "AvgC>2.5": "1.4",
                "AvgC<2.5": "2.8",
                "Avg>2.5": "1.9",
                "Avg<2.5": "1.9",
                "AHCh": "-1.5",
                "AHh": "-0.25",
                "AvgCAHH": "2.2",
                "AvgCAHA": "1.7",
                "AvgAHH": "1.95",
                "AvgAHA": "1.95",
            }
        )
        self.assertIsNotNone(prices)
        assert prices is not None
        self.assertAlmostEqual(float(prices["home_odds"]), 3.0)
        self.assertAlmostEqual(float(prices["over25_odds"]), 1.9)
        self.assertAlmostEqual(float(prices["ah_line"] or 0), -0.25)
        self.assertAlmostEqual(float(prices["ah_home_odds"] or 0), 1.95)

    def test_closing_only_fixture_is_dropped(self) -> None:
        self.assertIsNone(
            opening_prices(
                {
                    "AvgCH": "1.2",
                    "AvgCD": "6.0",
                    "AvgCA": "12.0",
                    "AvgC>2.5": "1.8",
                    "AvgC<2.5": "2.0",
                }
            )
        )

    def test_b365_fills_only_when_avg_group_is_absent(self) -> None:
        prices = opening_prices(
            {
                "B365H": "2.5",
                "B365D": "3.3",
                "B365A": "2.8",
                "B365>2.5": "2.0",
                "B365<2.5": "1.8",
            }
        )
        self.assertIsNotNone(prices)
        assert prices is not None
        self.assertAlmostEqual(float(prices["home_odds"]), 2.5)

    def test_cached_file_uses_opening_home_price(self) -> None:
        raw = pd.read_csv("data/cache/E0_2526.csv")
        differ = raw.loc[raw["AvgH"].notna() & raw["AvgCH"].notna() & (raw["AvgH"] != raw["AvgCH"])]
        self.assertFalse(differ.empty)
        row = differ.iloc[0]
        prices = opening_prices(row.to_dict())
        self.assertIsNotNone(prices)
        assert prices is not None
        self.assertAlmostEqual(float(prices["home_odds"]), float(row["AvgH"]))
        loaded = load_football_data(code="2526")
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])
        match = loaded.loc[(loaded["home"] == home) & (loaded["away"] == away)]
        self.assertFalse(match.empty)
        self.assertAlmostEqual(float(match["home_odds"].iloc[0]), float(row["AvgH"]))


if __name__ == "__main__":
    unittest.main()
