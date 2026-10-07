"""Locked stage-45 scores. No season is loaded."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.stage_45_score_repair import repair_scores


def _row(**overrides) -> pd.DataFrame:
    base = {
        "score_xp": 5.0,
        "xp_defcon": 0.4,
        "xp_cs": 1.0,
        "xp_appear": 2.0,
        "xmi": 90.0,
        "xp_goals": 1.0,
        "xp_gc_loss": 0.5,
        "position": "MID",
    }
    base.update(overrides)
    return pd.DataFrame([base])


class RepairScoresTest(unittest.TestCase):
    def test_a_full_match_appearance_stays_put(self) -> None:
        scores = repair_scores(_row())
        self.assertAlmostEqual(float(scores["appear_linear"].iloc[0]), 5.0)

    def test_the_buy_gate_appearance_drops_half_a_point(self) -> None:
        scores = repair_scores(_row(xp_appear=1.5, xmi=45.0))
        self.assertAlmostEqual(float(scores["appear_linear"].iloc[0]), 4.5)

    def test_goalkeeper_cut_does_not_touch_a_midfielder(self) -> None:
        scores = repair_scores(_row(position="MID", xp_goals=8.0))
        self.assertAlmostEqual(float(scores["gk6"].iloc[0]), 5.0)

    def test_goalkeeper_goal_uses_the_official_six(self) -> None:
        scores = repair_scores(_row(position="GKP", xp_goals=10.0))
        self.assertAlmostEqual(float(scores["gk6"].iloc[0]), 5.0 - 4.72)

    def test_clean_sheet_removal_includes_the_bonus_share(self) -> None:
        scores = repair_scores(_row(xp_cs=2.0))
        self.assertAlmostEqual(float(scores["no_cs"].iloc[0]), 5.0 - 2.16)
        self.assertAlmostEqual(float(scores["cs_half"].iloc[0]), 5.0 - 1.08)

    def test_half_the_goals_conceded_deduction_comes_back(self) -> None:
        scores = repair_scores(_row(xp_gc_loss=0.8))
        self.assertAlmostEqual(float(scores["gc_half"].iloc[0]), 5.4)

    def test_position_shift_is_defenders_and_forwards_only(self) -> None:
        frame = pd.concat(
            [
                _row(position="DEF"),
                _row(position="FWD"),
                _row(position="GKP"),
            ],
            ignore_index=True,
        )
        scores = repair_scores(frame)
        self.assertAlmostEqual(float(scores["pos_shift"].iloc[0]), 4.75)
        self.assertAlmostEqual(float(scores["pos_shift"].iloc[1]), 5.25)
        self.assertAlmostEqual(float(scores["pos_shift"].iloc[2]), 5.0)

    def test_defcon_is_subtracted_when_it_is_positive(self) -> None:
        scores = repair_scores(_row(xp_defcon=0.7))
        self.assertAlmostEqual(float(scores["no_defcon"].iloc[0]), 4.3)

    def test_a_missing_component_inside_a_present_column_is_zero(self) -> None:
        scores = repair_scores(_row(xp_cs=float("nan"), xp_defcon=float("nan")))
        self.assertAlmostEqual(float(scores["no_cs"].iloc[0]), 5.0)
        self.assertAlmostEqual(float(scores["no_defcon"].iloc[0]), 5.0)

    def test_a_missing_column_is_unusable(self) -> None:
        frame = _row().drop(columns=["xp_defcon"])
        with self.assertRaises(RuntimeError):
            repair_scores(frame)


if __name__ == "__main__":
    unittest.main()
