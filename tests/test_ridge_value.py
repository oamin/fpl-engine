"""The sheet join must keep a value column the player log already has."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.models import ridge_multiseason as ridge


class ValueJoinTests(unittest.TestCase):
    def _sheet(self, directory: Path, rows: list[dict]) -> None:
        frame = pd.DataFrame(rows)
        frame.to_csv(directory / "merged_gw_2025_26.csv", index=False)

    def test_a_log_price_is_kept_when_the_sheet_also_has_one(self) -> None:
        players = pd.DataFrame(
            {
                "player_id": ["1"],
                "gw": [1],
                "position": ["MID"],
                "value": [55.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._sheet(
                root,
                [
                    {
                        "element": 1,
                        "GW": 1,
                        "value": 99,
                        "transfers_balance": 3,
                        "defensive_contribution": 4,
                    }
                ],
            )
            with patch.object(ridge, "CACHE", root):
                out = ridge._attach_value_defcon(players, "2025-26")
        self.assertIn("value", out.columns)
        self.assertNotIn("value_sheet", out.columns)
        self.assertEqual(float(out.iloc[0]["value"]), 55.0)
        self.assertEqual(float(out.iloc[0]["defcon_raw"]), 4.0)

    def test_a_missing_log_price_takes_the_sheet(self) -> None:
        players = pd.DataFrame(
            {
                "player_id": ["1", "2"],
                "gw": [1, 1],
                "position": ["MID", "MID"],
                "value": [float("nan"), 80.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._sheet(
                root,
                [
                    {"element": 1, "GW": 1, "value": 70, "transfers_balance": 0},
                    {"element": 2, "GW": 1, "value": 80, "transfers_balance": 0},
                ],
            )
            with patch.object(ridge, "CACHE", root):
                out = ridge._attach_value_defcon(players, "2025-26")
        filled = float(out.loc[out["player_id"] == "1", "value"].iloc[0])
        kept = float(out.loc[out["player_id"] == "2", "value"].iloc[0])
        self.assertEqual(filled, 70.0)
        self.assertEqual(kept, 80.0)


if __name__ == "__main__":
    unittest.main()
