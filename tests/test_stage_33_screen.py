"""Formula checks. The season climb is not run here."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.stage_33_screen import FORMULAS, apply_formula, screen_table


class Stage33ScreenTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "gw": [1, 1, 1, 1],
                "position": ["GKP", "DEF", "MID", "FWD"],
                "score_xp": [4.0, 4.0, 6.0, 8.0],
                "value": [45.0, 50.0, 80.0, 100.0],
                "xmi": [90.0, 45.0, 0.0, 90.0],
                "sigma_xp": [1.0, 2.0, 0.0, 4.0],
                "xp_goals": [0.0, 1.0, 2.0, 3.0],
                "xp_assists": [0.0, 0.5, 1.0, 1.0],
                "xp_deductions": [0.2, 0.4, 0.0, 0.5],
                "score_exp_points": [5.0, 3.0, 6.0, 7.0],
                "xp_cs": [1.0, 2.0, 0.0, 0.0],
                "xp_saves": [1.5, 0.0, 0.0, 0.0],
                "xp_appear": [2.0, 2.0, 1.0, 2.0],
            }
        )

    def test_every_formula_returns_a_row(self) -> None:
        for formula in FORMULAS:
            got = apply_formula(self.df, formula)
            self.assertEqual(len(got), 4, formula)

    def test_locked_points(self) -> None:
        self.assertAlmostEqual(float(apply_formula(self.df, "per_million").iloc[0]), 4.0 / 4.5)
        self.assertAlmostEqual(float(apply_formula(self.df, "minutes").iloc[1]), 2.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "upside").iloc[3]), 9.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "attack").iloc[2]), 3.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "no_deduction").iloc[0]), 4.2)
        self.assertAlmostEqual(float(apply_formula(self.df, "agree_min").iloc[0]), 4.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "agree_max").iloc[0]), 5.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "premium").iloc[2]), 36.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "starter").iloc[1]), 0.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "split").iloc[0]), 4.5)
        self.assertAlmostEqual(float(apply_formula(self.df, "split").iloc[2]), 3.0)
        self.assertAlmostEqual(float(apply_formula(self.df, "goals_tilt").iloc[3]), 11.0)

    def test_within_pos_is_zero_when_a_position_has_one_player(self) -> None:
        got = apply_formula(self.df, "within_pos")
        self.assertTrue((got == 0.0).all())

    def test_kill_bar_is_not_strictly_inside(self) -> None:
        weekly = pd.DataFrame(
            {
                "method": ["xp", "alive", "dead"],
                "xi_points_cap": [1000.0, 901.0, 900.0],
            }
        )
        table = screen_table(weekly).set_index("candidate")
        self.assertFalse(bool(table.loc["alive", "killed"]))
        self.assertTrue(bool(table.loc["dead", "killed"]))
