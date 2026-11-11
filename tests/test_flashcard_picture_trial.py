"""The picture painter writes colors, and the bars are read from the picture."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_image_trial import EVENT_ROW, EVENT_SLOTS, apply_blank, hidden_events, ink_of
from src.models.flashcard_picture_trial import PicturePainter, event_targets, picture_loss, stamp_colors
from src.models.flashcard_remainder_trial import CELL_ROWS
from src.models.flashcard_squad_trial import COLS, GOALS_C, ROWS
from src.models.flashcard_square_trial import SQ


def _card() -> object:
    card = type("Card", (), {})()
    card.gw = np.arange(1, COLS + 1)
    card.present = np.zeros((ROWS, COLS), dtype=bool)
    card.present[:, 5] = True
    card.n_fixtures = COLS
    return card


class PicturePainterTests(unittest.TestCase):
    def test_one_pass_paints_a_color_for_every_event_square(self) -> None:
        torch.manual_seed(0)
        model = PicturePainter()
        model.eval()
        patches = torch.rand(2, SQ * SQ * 3, CELL_ROWS, COLS)
        first = model(patches)
        second = model(patches)
        self.assertEqual(first.shape, (2, ROWS, EVENT_SLOTS, COLS, 3))
        self.assertTrue(bool(torch.allclose(first, second)))
        self.assertTrue(bool(torch.all((first > 0.9) & (first < 1.0))))

    def test_an_exact_picture_has_no_color_error(self) -> None:
        color = torch.tensor([0.2, 0.4, 0.6])
        colors = color.view(1, 1, 1, 1, 3).expand(1, 2, EVENT_SLOTS, 3, 3).clone()
        target = color.view(1, 1, 1, 1, 1, 1, 3).expand(1, 2, EVENT_SLOTS, 3, SQ, SQ, 3).clone()
        hidden = torch.ones(1, 2, EVENT_SLOTS, 3, dtype=torch.bool)
        self.assertEqual(float(picture_loss(colors, target, hidden)), 0.0)

    def test_a_missing_bar_and_an_extra_bar_cost_the_same(self) -> None:
        white = torch.ones(1, 1, 1, 1, 3)
        goal = torch.tensor(GOALS_C, dtype=torch.float32).view(1, 1, 1, 1, 3) / 255.0
        patch = torch.ones(1, 1, 1, 1, SQ, SQ, 3)
        goal_patch = goal.view(1, 1, 1, 1, 1, 1, 3).expand(1, 1, 1, 1, SQ, SQ, 3).clone()
        hidden = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        missing = picture_loss(white, goal_patch, hidden)
        extra = picture_loss(goal, patch, hidden)
        self.assertAlmostEqual(float(missing), float(extra), places=6)

    def test_only_the_gray_squares_take_the_new_color(self) -> None:
        cells = np.full((1, CELL_ROWS, COLS, SQ, SQ, 3), 255, dtype=np.uint8)
        cells[:, EVENT_ROW[:, 0], 4] = (1, 2, 3)
        where = np.zeros((1, ROWS, EVENT_SLOTS, COLS), dtype=bool)
        where[0, 0, 0, 5] = True
        colors = np.zeros((1, ROWS, EVENT_SLOTS, COLS, 3), dtype=np.float32)
        colors[0, 0, 0, 5] = np.array(GOALS_C) / 255.0
        stamped = stamp_colors(cells, colors, where)
        self.assertTrue(np.all(stamped[0, EVENT_ROW[0, 0], 4, 0, 0] == (1, 2, 3)))
        self.assertTrue(np.all(stamped[0, EVENT_ROW[0, 0], 5, 0, 0] == GOALS_C))

    def test_a_goal_color_is_read_as_a_bar(self) -> None:
        cells = np.full((1, CELL_ROWS, COLS, SQ, SQ, 3), 255, dtype=np.uint8)
        card = _card()
        hidden = hidden_events([card], [3])
        apply_blank(cells, hidden)
        colors = np.ones((1, ROWS, EVENT_SLOTS, COLS, 3), dtype=np.float32)
        colors[0, 1, 2, 5] = np.array(GOALS_C) / 255.0
        stamped = stamp_colors(cells, colors, hidden)
        ink = ink_of(stamped, hidden[:, :, 0, :])
        self.assertTrue(bool(ink[0, 1, 2, 5]))
        self.assertEqual(int(ink.sum()), 1)

    def test_event_targets_keep_the_square_pixels(self) -> None:
        cells = np.full((1, CELL_ROWS, COLS, SQ, SQ, 3), 255, dtype=np.uint8)
        cells[0, EVENT_ROW[1, 0], 2, 0, 0] = GOALS_C
        target = event_targets(cells)
        self.assertEqual(target.shape, (1, ROWS, EVENT_SLOTS, COLS, SQ, SQ, 3))
        self.assertAlmostEqual(float(target[0, 1, 0, 2, 0, 0, 0]), GOALS_C[0] / 255.0)
