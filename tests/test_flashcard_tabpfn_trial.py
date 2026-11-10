"""The numeric card hides the same events the picture hides."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_squad_trial import MASK_ID
from src.models.flashcard_tabpfn_trial import (
    CLUB_CODE,
    align_probs,
    club_code,
    table_at,
    take_cards,
)
from tests.test_flashcard_squad_trial import _card, _league


class NumericCardTests(unittest.TestCase):
    def test_a_hidden_goal_does_not_change_the_card(self) -> None:
        league = _league()
        left = table_at([_card()], league, {}, [19])
        other = _card()
        other.goals = other.goals.copy()
        other.goals[0, 25] = 0
        right = table_at([other], league, {}, [19])
        compared = [name for name in left.columns if name not in {"y_goal", "y_assist"}]
        self.assertTrue(left[compared].equals(right[compared]))
        hidden = left["fixture"] == 25
        self.assertEqual(int(left.loc[hidden, "goal_visible"].iloc[0]), MASK_ID)
        self.assertEqual(int(left.loc[hidden, "y_goal"].iloc[0]), 3)
        self.assertEqual(int(right.loc[hidden, "y_goal"].iloc[0]), 0)

    def test_a_revealed_goal_is_the_token_on_that_cell(self) -> None:
        league = _league()
        card = _card()
        full = table_at([card], league, {}, [19])
        card.goals = card.goals.copy()
        card.goals[0, 5] = 0
        short = table_at([card], league, {}, [19])
        revealed = full["fixture"] == 5
        self.assertEqual(int(full.loc[revealed, "goal_visible"].iloc[0]), 3)
        self.assertEqual(int(short.loc[revealed, "goal_visible"].iloc[0]), 0)
        hidden = full["fixture"] == 25
        self.assertTrue((full.loc[hidden, "goal_visible"] == MASK_ID).all())
        self.assertTrue((short.loc[hidden, "goal_visible"] == MASK_ID).all())

    def test_the_fixture_column_is_the_squad_on_that_week(self) -> None:
        frame = table_at([_card()], _league(), {}, [19])
        cell = frame[frame["fixture"] == 25].iloc[0]
        self.assertEqual(float(cell["mate_minutes_00"]), 90.0)
        self.assertEqual(float(cell["mate_minutes_01"]), 0.0)
        self.assertEqual(int(cell["mate_position_00"]), 2)
        self.assertEqual(int(cell["opp_minutes_00"]), 0)
        self.assertNotIn("goals_so_far", frame.columns)
        self.assertNotIn("goals_per_90", frame.columns)

    def test_club_codes_stay_put_across_seasons(self) -> None:
        self.assertEqual(club_code("Arsenal"), 1)
        self.assertEqual(club_code("Liverpool"), 14)
        self.assertEqual(club_code("Wolves"), 25)
        self.assertEqual(len(CLUB_CODE), len(set(CLUB_CODE.values())))
        self.assertEqual(club_code("Home"), 0)

    def test_probabilities_land_on_the_count_they_came_from(self) -> None:
        raw = np.array([[0.2, 0.7, 0.1]])
        aligned = align_probs(raw, np.array([0, 2, 3]))
        self.assertEqual(aligned.shape, (1, 4))
        self.assertAlmostEqual(float(aligned[0, 0]), 0.2)
        self.assertAlmostEqual(float(aligned[0, 1]), 0.0)
        self.assertAlmostEqual(float(aligned[0, 2]), 0.7)
        self.assertAlmostEqual(float(aligned.sum()), 1.0)

    def test_whole_cards_fill_the_context(self) -> None:
        import pandas as pd

        blocks = [
            pd.DataFrame({"card_index": index, "y_goal": [0, 1]})
            for index in range(10)
        ]
        sample = take_cards(blocks, np.random.default_rng(0), limit=5)
        self.assertLessEqual(len(sample), 6)
        self.assertGreaterEqual(sample["card_index"].nunique(), 1)
        counts = sample.groupby("card_index").size()
        self.assertTrue((counts == 2).all())
