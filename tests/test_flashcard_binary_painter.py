"""Binary cards use a checker for a hidden event and black for a bar."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_binary_painter_trial import binary_ink, checker_patch, hide_checker, render_binary
from src.models.flashcard_image_trial import EVENT_ROW, EVENT_SLOTS
from src.models.flashcard_remainder_trial import CELL_ROWS
from src.models.flashcard_squad_trial import COLS, ROWS
from src.models.flashcard_square_trial import SQ


def _card() -> object:
    card = type("Card", (), {})()
    card.gw = np.arange(1, COLS + 1)
    card.n_fixtures = 1
    card.opponent = ("",)
    card.home = np.ones(COLS, dtype=np.int64)
    card.kickoff = np.zeros(COLS, dtype=np.int64)
    card.team = "Arsenal"
    card.season = "2024-25"
    card.present = np.zeros((ROWS, COLS), dtype=bool)
    card.present[0, 0] = 1
    card.minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    card.minutes[0, 0] = 90
    card.goals = np.zeros((ROWS, COLS), dtype=np.int64)
    card.goals[0, 0] = 1
    card.assists = np.zeros((ROWS, COLS), dtype=np.int64)
    card.position = np.full(ROWS, -1, dtype=np.int64)
    card.position[0] = 3
    return card


class BinaryPainterTests(unittest.TestCase):
    def test_one_goal_is_the_bottom_black_square(self) -> None:
        cells = render_binary(_card(), {})
        present = np.zeros((1, ROWS, COLS), dtype=bool)
        present[0, 0, 0] = True
        ink = binary_ink(cells, present[0])
        self.assertTrue(bool(ink[0, 2, 0]))
        self.assertFalse(bool(ink[0, 0, 0]))
        self.assertFalse(bool(ink[0, 1, 0]))
        self.assertEqual(int(ink.sum()), 1)

    def test_a_checker_is_not_a_bar(self) -> None:
        cells = np.full((1, CELL_ROWS, COLS, SQ, SQ, 3), 255, dtype=np.uint8)
        where = np.zeros((1, ROWS, EVENT_SLOTS, COLS), dtype=bool)
        where[0, 0, 2, 0] = True
        hide_checker(cells, where)
        present = np.ones((ROWS, COLS), dtype=bool)
        ink = binary_ink(cells[0], present)
        self.assertFalse(bool(ink[0, 2, 0]))
        self.assertTrue(np.array_equal(cells[0, EVENT_ROW[0, 2], 0], checker_patch()))

    def test_black_counts_and_white_does_not(self) -> None:
        cells = np.full((CELL_ROWS, COLS, SQ, SQ, 3), 255, dtype=np.uint8)
        cells[EVENT_ROW[0, 2], 0] = 0
        present = np.ones((ROWS, COLS), dtype=bool)
        ink = binary_ink(cells, present)
        self.assertTrue(bool(ink[0, 2, 0]))
        self.assertEqual(int(ink.sum()), 1)
