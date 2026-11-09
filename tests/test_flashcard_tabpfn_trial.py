"""Hidden events stay out of the tabular row. Revealed events stay in."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_squad_trial import fit_clubs
from src.models.flashcard_tabpfn_trial import CONTEXT, align_probs, stratified_sample, table_at
from tests.test_flashcard_squad_trial import _card, _league


class TabularRowTests(unittest.TestCase):
    def test_a_hidden_goal_does_not_change_the_row(self) -> None:
        league = _league()
        clubs = fit_clubs([_card()])
        left = table_at([_card()], league, clubs, {}, [19])
        other = _card()
        other.goals = other.goals.copy()
        other.goals[0, 25] = 0
        right = table_at([other], league, clubs, {}, [19])
        compared = [name for name in left.columns if name != "y_goal"]
        self.assertTrue(left[compared].equals(right[compared]))
        self.assertEqual(int(left.loc[left["fixture"] == 25, "y_goal"].iloc[0]), 3)
        self.assertEqual(int(right.loc[right["fixture"] == 25, "y_goal"].iloc[0]), 0)

    def test_a_revealed_goal_is_in_the_prefix(self) -> None:
        league = _league()
        card = _card()
        clubs = fit_clubs([card])
        full = table_at([card], league, clubs, {}, [19])
        card.goals = card.goals.copy()
        card.goals[0, 5] = 0
        short = table_at([card], league, clubs, {}, [19])
        self.assertGreater(float(full["goals_so_far"].iloc[0]), float(short["goals_so_far"].iloc[0]))
        self.assertEqual(int(full.loc[full["fixture"] == 25, "y_goal"].iloc[0]), 3)

    def test_probabilities_land_on_the_count_they_came_from(self) -> None:
        raw = np.array([[0.2, 0.7, 0.1]])
        aligned = align_probs(raw, np.array([0, 2, 3]))
        self.assertEqual(aligned.shape, (1, 4))
        self.assertAlmostEqual(float(aligned[0, 0]), 0.2)
        self.assertAlmostEqual(float(aligned[0, 1]), 0.0)
        self.assertAlmostEqual(float(aligned[0, 2]), 0.7)
        self.assertAlmostEqual(float(aligned.sum()), 1.0)

    def test_the_context_keeps_each_count(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"y_goal": np.array([0] * 9000 + [1] * 800 + [2] * 150 + [3] * 50)})
        sample = stratified_sample(frame, "y_goal", np.random.default_rng(0))
        self.assertEqual(len(sample), CONTEXT)
        self.assertEqual(set(sample["y_goal"]), {0, 1, 2, 3})
        self.assertGreater(int((sample["y_goal"] == 0).sum()), int((sample["y_goal"] == 1).sum()))
