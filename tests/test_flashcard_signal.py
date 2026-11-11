"""Band blanks hit the row they name and leave the others."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_signal import (
    GOAL_BAND,
    MINUTE_BAND,
    blank_player_band,
    blank_rows,
    place_ranked,
)
from src.models.flashcard_square_trial import HEADER_H, PLAYER_H
from src.models.flashcard_squad_trial import COLS, ROWS


class SignalTests(unittest.TestCase):
    def test_goal_blank_spares_the_minute_row(self) -> None:
        images = torch.zeros(1, 3, HEADER_H + ROWS * PLAYER_H, COLS * 4)
        blank_player_band(images, GOAL_BAND)
        self.assertEqual(float(images[0, 0, HEADER_H + MINUTE_BAND[0], 0]), 0.0)
        self.assertEqual(float(images[0, 0, HEADER_H + GOAL_BAND[0], 0]), 1.0)
        self.assertEqual(float(images[0, 0, HEADER_H + GOAL_BAND[1] - 1, 0]), 1.0)
        blank_rows(images, 0, HEADER_H)
        self.assertEqual(float(images[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(images[0, 0, HEADER_H + MINUTE_BAND[0], 0]), 0.0)

    def test_the_budget_goes_to_the_largest_key(self) -> None:
        key = np.array([[1.0, 0.0], [3.0, 5.0], [2.0, 4.0]])
        budgets = np.array([2, 1])
        mask = np.ones_like(key, dtype=bool)
        minutes = np.array([[10.0, 0.0], [20.0, 30.0], [40.0, 50.0]])
        drawn = place_ranked(key, budgets, mask, minutes)
        self.assertEqual(drawn[:, 0].tolist(), [0, 1, 1])
        self.assertEqual(drawn[:, 1].tolist(), [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
