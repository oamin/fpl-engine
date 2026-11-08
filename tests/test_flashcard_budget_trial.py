"""Flat event colors, a hidden concede band, and a budget that skips zero minutes."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_budget_trial import COUNT_C, place_budget, render_mark
from src.models.flashcard_squad_trial import (
    BAND_H,
    CELL_W,
    COLS,
    CONCEDE_C,
    HEADER_H,
    MASK_C,
    ROWS,
    apply_cut,
)
from tests.test_flashcard_squad_trial import _card, _league


class MarkTests(unittest.TestCase):
    def test_a_goal_fills_its_band_and_a_hidden_week_stays_gray(self) -> None:
        view = apply_cut(_card(), _league(), 19)
        image = render_mark(view)
        y = HEADER_H + 2 * BAND_H
        shown = image[y : y + BAND_H, 0:CELL_W]
        hidden = image[y : y + BAND_H, 19 * CELL_W : 20 * CELL_W]
        self.assertTrue(np.all(shown == COUNT_C[3]))
        self.assertTrue(np.all(hidden == MASK_C))
        concede = image[BAND_H : BAND_H * 2, 19 * CELL_W : 20 * CELL_W]
        self.assertFalse(np.any(np.all(concede == CONCEDE_C, axis=-1)))

    def test_the_budget_skips_a_zero_minute_row(self) -> None:
        probs = np.zeros((ROWS, COLS, 4))
        probs[:, :, 0] = 1.0
        probs[0, 0] = (0.0, 0.0, 0.0, 1.0)
        probs[1, 0] = (0.0, 0.2, 0.8, 0.0)
        probs[2, 0] = (0.7, 0.3, 0.0, 0.0)
        match = np.zeros((COLS, 9))
        match[0, 3] = 1.0
        minutes = np.zeros((ROWS, COLS))
        minutes[1, 0] = 90
        minutes[2, 0] = 90
        drawn = place_budget(probs, match, minutes, np.random.default_rng(0))
        self.assertEqual(int(drawn[0, 0]), 0)
        self.assertEqual(int(drawn[1, 0]), 2)
        self.assertEqual(int(drawn[2, 0]), 1)
        self.assertEqual(int(drawn[:, 0].sum()), 3)
