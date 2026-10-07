"""The 90-minute table chooses players from ownership, not from points."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.reversion_players import (
    FORBIDDEN,
    REQUIRED,
    _nearest,
    _tail_sentence,
    manager_percent,
    qualifying_rows,
)


class PlayerTableTest(unittest.TestCase):
    def test_percentiles_ignore_points_and_break_ties_by_id(self) -> None:
        frame = pd.DataFrame(
            {
                "player_id": ["b", "a", "c", "d", "e"],
                "own_median": [10.0, 10.0, 30.0, 50.0, 90.0],
                "name": ["B", "A", "C", "D", "E"],
                "weeks": [8, 8, 8, 8, 8],
            }
        )
        picked = _nearest(frame, (10, 90))
        self.assertEqual(picked["player_id"].tolist()[0], "a")
        self.assertEqual(picked["player_id"].tolist()[-1], "e")
        text = "\n".join(REQUIRED)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)

    def test_manager_percent_scales_by_fifteen(self) -> None:
        selected = pd.Series([10.0, 30.0])
        gw = pd.Series([1, 1])
        percent = manager_percent(selected, gw)
        self.assertAlmostEqual(float(percent.iloc[1]), 100.0 * 15.0 * 30.0 / 40.0)

    def test_three_week_cell_stays_blank_unless_every_week_is_ninety(self) -> None:
        panel = pd.DataFrame(
            [
                {"season": "2024-25", "gw": 5, "player_id": "a", "n_fix": 1, "actual": 2.0, "minutes": 90.0, "exp": 4.0, "xp": 3.0, "lam": 1.0, "xmi": 80.0},
                {"season": "2024-25", "gw": 6, "player_id": "a", "n_fix": 1, "actual": 8.0, "minutes": 45.0, "exp": 4.0, "xp": 3.0, "lam": 1.0, "xmi": 80.0},
                {"season": "2024-25", "gw": 7, "player_id": "a", "n_fix": 1, "actual": 6.0, "minutes": 90.0, "exp": 5.0, "xp": 4.0, "lam": 1.0, "xmi": 80.0},
                {"season": "2024-25", "gw": 8, "player_id": "a", "n_fix": 1, "actual": 1.0, "minutes": 12.0, "exp": 5.5, "xp": 4.5, "lam": 1.0, "xmi": 80.0},
            ]
        )
        rows = qualifying_rows(panel)
        week_8 = rows.loc[rows["gw"] == 8].iloc[0]
        self.assertAlmostEqual(float(week_8["exp_minus_actual"]), 5.5 - 6.0)
        self.assertTrue(pd.isna(week_8["exp_minus_mean3"]))
        week_7 = rows.loc[rows["gw"] == 7]
        self.assertTrue(week_7.empty)

    def test_tail_sentence_names_the_pool_maximum(self) -> None:
        summary = pd.DataFrame(
            {
                "player_id": ["1", "2", "3"],
                "own_median": [1.0, 10.0, 66.7],
                "name": ["Low", "Mid", "Mohamed Salah"],
                "weeks": [8, 8, 8],
            }
        )
        high = pd.Series({"name": "Kai Havertz", "own_median": 13.5})
        text = _tail_sentence(summary, high)
        self.assertIn("Mohamed Salah at 66.7%", text)
        self.assertIn("does not sample the extreme ownership tail.", text)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)
