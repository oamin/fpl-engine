"""Common-state rules, locked before any transfer total is read."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.eval.common_state import pairwise_concordance
from src.eval.decision import greedy_step, legal_moves, neutral_squad
from src.models.xp_engine import _roll_mean
from tests.test_decision import _core, _hand_squad, _row


class CommonStateRuleTest(unittest.TestCase):
    def test_the_neutral_squad_ignores_score(self) -> None:
        rows = _core(5)
        for row in rows:
            if row["player_id"] == "15":
                row["score_xp"] = float("nan")
                row["score_exp_points"] = float("nan")
        rows.append(_row("16", "FWD", "h", 55, 999.0, 0.0, 5))
        week = pd.DataFrame(rows)
        squad = neutral_squad(week)
        self.assertIn("15", squad.purchase)
        self.assertNotIn("16", squad.purchase)
        flipped = week.copy()
        flipped.loc[flipped["player_id"].astype(str) == "15", "score_xp"] = 999.0
        flipped.loc[flipped["player_id"].astype(str) == "16", "score_xp"] = 0.0
        again = neutral_squad(flipped)
        self.assertEqual(again.purchase, squad.purchase)
        self.assertEqual(len(squad.purchase), 15)

    def test_the_second_score_sees_the_unmutated_squad(self) -> None:
        squad = _hand_squad()
        rows = [
            _row(pid, pos, squad.club[pid], 40, 1.0, 2.0, 6, exp=1.0)
            for pid, pos in squad.position.items()
        ]
        rows.append(_row("99", "FWD", "z", 40, 10.0, 1.0, 6, exp=0.0))
        rows.append(_row("98", "FWD", "y", 40, 0.0, 9.0, 6, exp=10.0))
        week = pd.DataFrame(rows)
        before = dict(squad.purchase)
        _xp_squad, xp_move = greedy_step(squad, week, "score_xp")
        _exp_squad, exp_move = greedy_step(squad, week, "score_exp_points")
        self.assertEqual(xp_move["player_in"], "99")
        self.assertEqual(exp_move["player_in"], "98")
        self.assertEqual(squad.purchase, before)
        self.assertIn(xp_move["player_out"], squad.purchase)

    def test_legal_moves_exclude_the_wrong_position_and_an_unaffordable_buy(self) -> None:
        squad = _hand_squad()
        squad.bank = 0
        rows = [
            _row(pid, pos, squad.club[pid], 40, 1.0, 2.0, 6)
            for pid, pos in squad.position.items()
        ]
        rows.append(_row("90", "MID", "z", 40, 5.0, 0.0, 6))
        rows.append(_row("91", "MID", "y", 500, 9.0, 0.0, 6))
        rows.append(_row("92", "FWD", "x", 40, 9.0, 0.0, 6, eligible=False))
        moves = legal_moves(squad, pd.DataFrame(rows), "score_xp")
        incoming = {move["player_in"] for move in moves}
        self.assertIn("90", incoming)
        self.assertNotIn("91", incoming)
        self.assertNotIn("92", incoming)
        for move in moves:
            self.assertEqual(move["position"], squad.position[move["player_out"]])

    def test_concordance_is_one_zero_or_undefined(self) -> None:
        perfect = pairwise_concordance(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        reversed_points = pairwise_concordance(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
        tied = pairwise_concordance(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]))
        self.assertAlmostEqual(perfect, 1.0)
        self.assertAlmostEqual(reversed_points, 0.0)
        self.assertIsNone(tied)

    def test_roll3_uses_only_earlier_weeks(self) -> None:
        rolled = _roll_mean(pd.Series([10.0, 0.0, 99.0]), 3)
        self.assertTrue(pd.isna(rolled.iloc[0]))
        self.assertAlmostEqual(float(rolled.iloc[2]), 5.0)


if __name__ == "__main__":
    unittest.main()
