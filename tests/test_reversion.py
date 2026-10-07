"""Residual bins and the oracle blend do not use same-week minutes as form."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.reversion import (
    BIN_LABELS,
    FORBIDDEN,
    REQUIRED,
    bin_label,
    blend_xmi,
    reversion_rows,
    sheet_predecessor,
)


class ReversionRuleTest(unittest.TestCase):
    def test_exact_edges_fall_in_the_upper_bin(self) -> None:
        self.assertEqual(bin_label(-5.0), "[-5, -3)")
        self.assertEqual(bin_label(-5.01), "< -5")
        self.assertEqual(bin_label(0.0), "[0, +1)")
        self.assertEqual(bin_label(5.0), "> +5")
        self.assertEqual(bin_label(4.9), "[+3, +5)")
        self.assertEqual(len(BIN_LABELS), 10)

    def test_a_missing_sheet_week_is_the_predecessor(self) -> None:
        gameweeks = [1, 2, 3, 4, 5, 6, 8]
        self.assertIsNone(sheet_predecessor(gameweeks, 1, 1))
        self.assertEqual(sheet_predecessor(gameweeks, 8, 1), 6)
        self.assertEqual(sheet_predecessor(gameweeks, 8, 3), 4)
        self.assertIsNone(sheet_predecessor(gameweeks, 8, 8))

    def test_a_double_or_a_gap_is_excluded(self) -> None:
        panel = pd.DataFrame(
            [
                {"season": "2023-24", "gw": 5, "player_id": "a", "n_fix": 1, "actual": 2.0, "minutes": 90.0, "exp": 6.0, "xp": 5.0, "lam": 1.4, "xmi": 80.0},
                {"season": "2023-24", "gw": 6, "player_id": "a", "n_fix": 2, "actual": 4.0, "minutes": 180.0, "exp": 6.0, "xp": 5.0, "lam": 1.2, "xmi": 80.0},
                {"season": "2023-24", "gw": 7, "player_id": "a", "n_fix": 1, "actual": 8.0, "minutes": 90.0, "exp": 6.0, "xp": 5.0, "lam": 1.5, "xmi": 70.0},
                {"season": "2023-24", "gw": 5, "player_id": "b", "n_fix": 1, "actual": 1.0, "minutes": 90.0, "exp": 4.0, "xp": 3.0, "lam": 1.0, "xmi": 50.0},
                {"season": "2023-24", "gw": 7, "player_id": "b", "n_fix": 1, "actual": 3.0, "minutes": 90.0, "exp": 4.0, "xp": 3.0, "lam": 1.1, "xmi": 50.0},
            ]
        )
        self.assertEqual(reversion_rows(panel), [])

    def test_the_pair_uses_actual_minus_the_current_baseline(self) -> None:
        panel = pd.DataFrame(
            [
                {"season": "2023-24", "gw": 5, "player_id": "a", "n_fix": 1, "actual": 2.0, "minutes": 90.0, "exp": 5.0, "xp": 4.0, "lam": 1.4, "xmi": 80.0},
                {"season": "2023-24", "gw": 6, "player_id": "a", "n_fix": 1, "actual": 8.0, "minutes": 35.0, "exp": 6.0, "xp": 5.5, "lam": 1.6, "xmi": 75.0},
            ]
        )
        row = reversion_rows(panel)[0]
        self.assertEqual(row["gw"], 6)
        self.assertAlmostEqual(row["previous_exp"], 2.0 - 6.0)
        self.assertAlmostEqual(row["next_exp"], 8.0 - 6.0)
        self.assertAlmostEqual(row["minutes_prev"], 90.0)
        self.assertTrue(pd.isna(row["previous3_exp"]))

    def test_the_blend_keeps_the_locked_weights(self) -> None:
        self.assertAlmostEqual(blend_xmi(90.0, 30.0, 0.8), 78.0)
        self.assertAlmostEqual(blend_xmi(0.0, 60.0, 0.6), 24.0)
        with self.assertRaises(ValueError):
            blend_xmi(90.0, 30.0, 0.7)
        text = "\n".join(REQUIRED)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)
        self.assertIn("score_xp` is unchanged", text)
