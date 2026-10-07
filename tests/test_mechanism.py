"""The disagreement labels do not read realised points."""

from __future__ import annotations

import unittest

from src.eval.mechanism import (
    FLAGS,
    FORBIDDEN,
    REQUIRED,
    SLICES,
    _require_two_seasons,
    flag_row,
    slice_name,
)


class MechanismFlagTest(unittest.TestCase):
    def test_a_tie_or_a_missing_value_does_not_fire(self) -> None:
        tied = flag_row(
            {
                "xp_xmi": 80.0,
                "exp_xmi": 80.0,
                "xp_attack": 0.5,
                "exp_attack": 0.5,
                "xp_price": 100.0,
                "exp_price": 100.0,
                "xp_cs_defcon": 1.0,
                "exp_cs_defcon": 1.0,
                "xp_goals_assists": 2.0,
                "exp_goals_assists": 2.0,
                "xp_form": 4.0,
                "exp_form": 4.0,
                "xp_lam": 1.4,
                "exp_lam": 1.4,
            }
        )
        self.assertTrue(tied["other"])
        self.assertEqual(tied["slice"], "neither")
        missing = flag_row({"xp_lam": 1.2, "exp_lam": float("nan"), "exp_xmi": 90.0, "xp_xmi": 70.0})
        self.assertFalse(missing["fixture_to_xp"])
        self.assertTrue(missing["minutes_to_exp"])
        self.assertEqual(missing["slice"], "minutes_only")

    def test_the_slices_partition_and_ignore_realised_points(self) -> None:
        base = {
            "xp_xmi": 70.0,
            "exp_xmi": 90.0,
            "xp_attack": 0.8,
            "exp_attack": 0.4,
            "xp_price": 80.0,
            "exp_price": 120.0,
            "r1_xp_minus_r1_exp": -10.0,
            "r3_xp_minus_r3_exp": -30.0,
        }
        first = flag_row(base)
        flipped = flag_row({**base, "r1_xp_minus_r1_exp": 10.0, "r3_xp_minus_r3_exp": 30.0})
        self.assertEqual(first, flipped)
        self.assertEqual(first["slice"], "both")
        self.assertTrue(first["cheaper_xp"])
        self.assertEqual(slice_name(False, True), "attack_only")
        self.assertEqual(set(SLICES), {"both", "minutes_only", "attack_only", "neither"})
        self.assertEqual(len(FLAGS), 7)
        text = "\n".join(REQUIRED)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)
        self.assertIn("score_xp` is unchanged", text)

    def test_one_season_is_undefined(self) -> None:
        blank = _require_two_seasons(
            {"mean": -3.0, "lo": -6.0, "hi": 0.2, "n_disagree": {"2022-23": 7}}
        )
        self.assertIsNone(blank["mean"])
        kept = _require_two_seasons(
            {"mean": 1.0, "lo": -1.0, "hi": 2.0, "n_disagree": {"2023-24": 5, "2024-25": 7}}
        )
        self.assertEqual(kept["mean"], 1.0)
