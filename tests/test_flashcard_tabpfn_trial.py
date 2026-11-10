"""The tabular card hides the same events the picture hides, and shows Shin odds."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.flashcard_squad_trial import COLS, MASK_ID
from src.models.flashcard_tabpfn_trial import (
    MISSING,
    align_probs,
    card_rows,
    features,
    position_name,
    take_cards,
)
from tests.test_flashcard_squad_trial import _card, _league


def _probs() -> np.ndarray:
    probs = np.full((COLS, 3), np.nan, dtype=np.float64)
    probs[:30] = (0.55, 0.25, 0.20)
    return probs


class TabularCardTests(unittest.TestCase):
    def test_a_hidden_goal_does_not_change_the_card(self) -> None:
        league = _league()
        left = card_rows(_card(), league, {}, _probs(), "shot", cut=19)
        other = _card()
        other.goals = other.goals.copy()
        other.goals[0, 25] = 0
        right = card_rows(other, league, {}, _probs(), "shot", cut=19)
        compared = [name for name in left.columns if name not in {"y_goal", "y_assist"}]
        self.assertTrue(left[compared].equals(right[compared]))
        hidden = left["fixture"] == 25
        self.assertEqual(left.loc[hidden, "goal_25"].iloc[0], str(MASK_ID))
        self.assertEqual(int(left.loc[hidden, "y_goal"].iloc[0]), 3)
        self.assertEqual(int(right.loc[hidden, "y_goal"].iloc[0]), 0)
        self.assertNotIn("y_goal", features(left).columns)

    def test_a_revealed_goal_is_the_token_on_that_column(self) -> None:
        league = _league()
        card = _card()
        full = card_rows(card, league, {}, _probs(), "shot", cut=19)
        card.goals = card.goals.copy()
        card.goals[0, 5] = 0
        short = card_rows(card, league, {}, _probs(), "shot", cut=19)
        self.assertEqual(full["goal_05"].iloc[0], "3")
        self.assertEqual(short["goal_05"].iloc[0], "0")
        self.assertTrue((full["goal_25"] == str(MASK_ID)).all())
        self.assertTrue((short["goal_25"] == str(MASK_ID)).all())

    def test_hidden_weeks_keep_the_header_and_the_odds(self) -> None:
        frame = card_rows(_card(), _league(), {}, _probs(), "shot", cut=19)
        cell = frame[frame["fixture"] == 25].iloc[0]
        self.assertEqual(cell["team"], "Home")
        self.assertEqual(cell["position"], "MID")
        self.assertEqual(cell["opp_25"], "Opp")
        self.assertEqual(float(cell["home_25"]), 0.0)
        self.assertAlmostEqual(float(cell["odds_h_25"]), 0.55)
        self.assertAlmostEqual(float(cell["odds_d_25"]), 0.25)
        self.assertAlmostEqual(float(cell["odds_a_25"]), 0.20)
        self.assertEqual(float(cell["min_25"]), 90.0)
        self.assertEqual(float(cell["mate_min_00"]), 90.0)
        self.assertEqual(cell["mate_pos_00"], "MID")
        self.assertTrue(np.isnan(cell["mate_min_01"]))
        self.assertEqual(cell["mate_pos_01"], MISSING)
        self.assertEqual(position_name(0), "GK")

    def test_a_missing_price_stays_empty(self) -> None:
        probs = _probs()
        probs[25] = np.nan
        frame = card_rows(_card(), _league(), {}, probs, "shot", cut=19)
        cell = frame[frame["fixture"] == 25].iloc[0]
        self.assertTrue(np.isnan(cell["odds_h_25"]))
        self.assertEqual(cell["opp_25"], "Opp")

    def test_the_stepped_card_drops_later_columns(self) -> None:
        frame = card_rows(_card(), _league(), {}, _probs(), "step", target=20)
        self.assertTrue((frame["fixture"] == 19).all())
        cell = frame.iloc[0]
        self.assertEqual(float(cell["min_19"]), 90.0)
        self.assertEqual(cell["goal_19"], str(MASK_ID))
        self.assertEqual(cell["opp_19"], MISSING)
        self.assertEqual(float(cell["home_19"]), 0.0)
        self.assertAlmostEqual(float(cell["odds_h_19"]), 0.55)
        self.assertEqual(cell["goal_18"], "3")
        self.assertEqual(cell["opp_18"], "Opp")
        self.assertTrue(np.isnan(cell["min_20"]))
        self.assertEqual(cell["goal_20"], MISSING)
        self.assertTrue(np.isnan(cell["odds_h_20"]))
        self.assertEqual(cell["opp_20"], MISSING)

    def test_probabilities_land_on_the_count_they_came_from(self) -> None:
        raw = np.array([[0.2, 0.7, 0.1]])
        aligned = align_probs(raw, np.array([0, 2, 3]))
        self.assertEqual(aligned.shape, (1, 4))
        self.assertAlmostEqual(float(aligned[0, 0]), 0.2)
        self.assertAlmostEqual(float(aligned[0, 1]), 0.0)
        self.assertAlmostEqual(float(aligned[0, 2]), 0.7)
        self.assertAlmostEqual(float(aligned.sum()), 1.0)

    def test_whole_cards_fill_the_context(self) -> None:
        blocks = [pd.DataFrame({"card_index": index, "y_goal": [0, 1]}) for index in range(10)]
        sample = take_cards(blocks, np.random.default_rng(0), limit=5)
        self.assertLessEqual(len(sample), 6)
        self.assertGreaterEqual(sample["card_index"].nunique(), 1)
        counts = sample.groupby("card_index").size()
        self.assertTrue((counts == 2).all())


if __name__ == "__main__":
    unittest.main()
