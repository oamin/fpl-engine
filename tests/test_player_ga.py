"""Player finishing scales."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.player_ga import add_finish_scales


class FinishScaleTest(unittest.TestCase):
    def test_overperformer_above_one(self) -> None:
        df = pd.DataFrame(
            [
                {"player_id": "1", "gw": 1, "date": "2024-08-01", "goals": 2.0, "assists": 0.0, "xG": 0.5, "xA": 0.2},
                {"player_id": "1", "gw": 2, "date": "2024-08-08", "goals": 0.0, "assists": 0.0, "xG": 0.4, "xA": 0.1},
            ]
        )
        out = add_finish_scales(df)
        # GW1: no history → prior only → 1.0
        self.assertAlmostEqual(float(out.loc[out["gw"] == 1, "finish_g"].iloc[0]), 1.0)
        # GW2: (2+2)/(0.5+2) = 4/2.5 = 1.6 → clipped to 1.40
        self.assertAlmostEqual(float(out.loc[out["gw"] == 2, "finish_g"].iloc[0]), 1.40)


if __name__ == "__main__":
    unittest.main()
