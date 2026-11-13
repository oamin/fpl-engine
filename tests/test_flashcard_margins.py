"""The column count and the margin fit."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_card_score import column_count, fit_margins


class MarginTests(unittest.TestCase):
    def test_a_zero_row_stays_empty_when_the_week_is_fat(self) -> None:
        horizon = np.zeros((2, 3, 4))
        horizon[0, :, 0] = (0.2, 0.8, 0.95)
        horizon[1, :, 0] = 1.0
        column = np.zeros((2, 3, 4))
        column[:, :, 0] = 1.0
        column[:, 2, 0] = 0.0
        mask = np.ones((2, 3), dtype=bool)
        painted = fit_margins(horizon, column, mask)
        self.assertEqual(int(painted[1].sum()), 0)
        self.assertEqual(int((painted[0] > 0).sum()), 1)
        self.assertGreater(painted[0, 2], 0)

    def test_the_row_count_is_the_horizon_count(self) -> None:
        horizon = np.zeros((1, 4, 4))
        horizon[0, :, 0] = (0.2, 0.2, 0.9, 0.9)
        column = np.full((1, 4, 4), 0.5)
        mask = np.ones((1, 4), dtype=bool)
        painted = fit_margins(horizon, column, mask)
        self.assertEqual(int((painted[0] > 0).sum()), 2)

    def test_column_count_scores_the_squad_sum(self) -> None:
        probs = torch.zeros(1, 2, 2, 4)
        probs[..., 0] = 1.0
        probs[0, :, 0, 0] = 0.0
        probs[0, :, 0, 1] = 1.0
        labels = torch.zeros(1, 2, 2)
        labels[0, :, 0] = 1
        mask = torch.ones(1, 2, 2)
        self.assertEqual(float(column_count(probs, labels, mask)), 0.0)


if __name__ == "__main__":
    unittest.main()
