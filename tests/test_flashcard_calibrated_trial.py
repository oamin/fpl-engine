"""A matched bar count, ranked inside the fixture, scored without the true zeros."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_calibrated_trial import event_rates, integer_budgets, place_expected


class CalibrationTests(unittest.TestCase):
    def test_budgets_add_up_to_the_rounded_total(self) -> None:
        budgets = integer_budgets(np.array([0.2, 1.2, 0.2]))
        self.assertEqual(int(budgets.sum()), 2)
        self.assertEqual(int(budgets[1]), 1)

    def test_placement_skips_zero_minutes_and_keeps_the_count(self) -> None:
        probs = np.zeros((3, 2, 4))
        probs[..., 0] = 1.0
        probs[0, 0, 0] = 0.1
        probs[0, 0, 1] = 0.9
        probs[1, 0, 0] = 0.1
        probs[1, 0, 1] = 0.9
        probs[2, 0, 0] = 0.3
        probs[2, 0, 1] = 0.7
        minutes = np.zeros((3, 2))
        minutes[1, 0] = 90
        minutes[2, 0] = 90
        mask = np.zeros((3, 2), dtype=bool)
        mask[:, 0] = True
        drawn = place_expected(probs, minutes, mask)
        self.assertEqual(int(drawn[0].sum()), 0)
        self.assertEqual(int(drawn.sum()), 2)
        self.assertEqual(int(drawn[1, 0]), 1)

    def test_a_blank_card_scores_zero_and_extra_bars_lower_the_mean(self) -> None:
        actual = np.array([1, 0, 0, 0])
        blank = event_rates(actual, np.zeros(4))
        exact = event_rates(actual, np.array([1, 0, 0, 0]))
        extra = event_rates(actual, np.array([1, 1, 1, 0]))
        self.assertEqual(blank["f1"], 0.0)
        self.assertGreater(exact["f1"], extra["f1"])
        self.assertEqual(exact["recall"], 1.0)
        self.assertEqual(exact["precision"], 1.0)
