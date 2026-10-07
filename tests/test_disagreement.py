"""Disagreement and opening-portfolio rules, locked before the new totals."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.disagreement import (
    cross_margin,
    disagreement_rows,
    driver_name,
    freeze_portfolios,
    hierarchy_frame,
    realised_call,
    signed_contributions,
)
from tests.test_decision import _core, _hand_squad, _row


class DisagreementRuleTest(unittest.TestCase):
    def test_a_zero_gap_is_a_tie(self) -> None:
        self.assertEqual(realised_call(3.0, 1.0), "xp")
        self.assertEqual(realised_call(1.0, 3.0), "exp")
        self.assertEqual(realised_call(2.0, 2.0), "equal")
        self.assertNotEqual(realised_call(2.0, 2.1), "equal")

    def test_the_cross_margin_stays_in_one_score(self) -> None:
        scores = {"a": 5.0, "b": 1.0, "c": 4.0, "d": 2.0}
        margin = cross_margin(scores, "a", "b", "c", "d")
        self.assertAlmostEqual(margin, 2.0)

    def test_the_driver_is_the_largest_signed_piece(self) -> None:
        signed = signed_contributions(
            {"xp_goals": 3.0, "xp_bps": 0.2, "xp_deductions": 1.0},
            {"xp_goals": 0.0, "xp_bps": 0.0, "xp_deductions": 0.0},
        )
        self.assertAlmostEqual(signed["xp_goals"], 3.0)
        self.assertAlmostEqual(signed["xp_deductions"], -1.0)
        self.assertEqual(driver_name(signed), "xp_goals")
        tied = {"xp_appear": 1.0, "xp_goals": 1.0}
        self.assertEqual(driver_name(tied), "xp_appear")

    def test_agreement_weeks_stay_out_of_the_disagreement_frame(self) -> None:
        frame = pd.DataFrame(
            {
                "season": ["2022-23", "2022-23"],
                "agree_xp_exp": [1, 0],
                "r1_exp": [1.0, 2.0],
                "r1_shuffled": [1.0, 0.0],
                "gw": [6, 7],
            }
        )
        differ = disagreement_rows(frame)
        self.assertEqual(differ["gw"].tolist(), [7])
        hierarchy = hierarchy_frame(frame)
        self.assertAlmostEqual(float(hierarchy["r1_exp_minus_r1_shuffled"].mean()), 1.0)

    def test_a_later_arrival_does_not_enter_the_frozen_fifteen(self) -> None:
        rows = _core(5, score=2.0) + _core(6, score=0.0)
        rows.append(_row("99", "MID", "h", 45, 50.0, 0.0, 6))
        for row in rows:
            row["roll3_points"] = 0.0
        start, squads = freeze_portfolios(
            pd.DataFrame(rows), 0, gw_start=5, gw_end=6
        )
        self.assertEqual(start, 5)
        for squad in squads.values():
            self.assertNotIn("99", squad.purchase)
            self.assertEqual(len(squad.purchase), 15)

    def test_deployment_does_not_mutate_or_read_realised_points(self) -> None:
        from src.eval.disagreement import deployed_points

        squad = _hand_squad()
        rows = [
            _row(pid, pos, squad.club[pid], 40, 1.0, 2.0, 6)
            for pid, pos in squad.position.items()
        ]
        for row in rows:
            row["roll3_points"] = 0.0
        week = pd.DataFrame(rows)
        before = dict(squad.purchase)
        points = deployed_points({"xp": squad, "exp": squad, "shuffled": squad, "neutral": squad}, week, "score_xp")
        self.assertEqual(squad.purchase, before)
        self.assertEqual(len(points), 4)
        with self.assertRaises(RuntimeError):
            deployed_points({"xp": squad}, week, "total_points")


if __name__ == "__main__":
    unittest.main()
