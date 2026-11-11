"""The image painters read squares and paint squares."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_image_trial import (
    CELL_ROWS,
    EVENT_ROW,
    HEADER_CELLS,
    MINUTE_CELLS,
    CellPainter,
    apply_blank,
    band_picture,
    cells_of,
    hidden_events,
    image_of,
    ink_of,
    mask_from_bottom,
    pool_max,
    repeat_blocks,
)
from src.models.flashcard_squad_trial import ASSISTS_C, COLS, GOALS_C, MASK_C, MINUTES_C, ROWS
from src.models.flashcard_square_trial import SQ, paint_square


def _view() -> dict[str, np.ndarray]:
    view = {
        "column_on": np.ones(COLS, dtype=np.float32),
        "home": np.ones(COLS, dtype=np.float32),
        "position": np.full(ROWS, -1, dtype=np.int64),
        "present": np.zeros((ROWS, COLS), dtype=np.float32),
        "minutes": np.zeros((ROWS, COLS), dtype=np.float32),
        "goal_token": np.zeros((ROWS, COLS), dtype=np.int64),
        "assist_token": np.zeros((ROWS, COLS), dtype=np.int64),
    }
    view["position"][0] = 3
    view["present"][0, 0] = 1
    view["minutes"][0, 0] = 0
    view["present"][0, 1] = 1
    view["minutes"][0, 1] = 90
    view["goal_token"][0, 1] = 2
    view["assist_token"][0, 1] = 1
    return view


def _picture(view: dict[str, np.ndarray]) -> np.ndarray:
    probs = np.full((COLS, 3), np.nan, dtype=np.float32)
    visible = np.zeros(COLS, dtype=bool)
    return paint_square(view, probs, visible, opponents=("",) * COLS)


class ImagePainterTests(unittest.TestCase):
    def test_the_picture_splits_into_the_squares_it_was_painted_with(self) -> None:
        image = _picture(_view())
        cells = cells_of(image)
        self.assertEqual(cells.shape, (CELL_ROWS, COLS, SQ, SQ, 3))
        self.assertTrue(np.array_equal(image_of(cells), image))
        blank = cells[HEADER_CELLS : HEADER_CELLS + MINUTE_CELLS, 0].reshape(-1, 3)
        full = cells[HEADER_CELLS : HEADER_CELLS + MINUTE_CELLS, 1].reshape(-1, 3)
        self.assertFalse(np.any(np.all(blank == MINUTES_C, axis=1)))
        self.assertTrue(np.all(np.all(full == MINUTES_C, axis=1)))
        goals = cells[EVENT_ROW[0, :3], 1]
        self.assertFalse(np.all(goals[0] == GOALS_C))
        self.assertTrue(np.all(goals[1] == GOALS_C))
        self.assertTrue(np.all(goals[2] == GOALS_C))
        assists = cells[EVENT_ROW[0, 3:], 1]
        self.assertFalse(np.all(assists[0] == ASSISTS_C))
        self.assertFalse(np.all(assists[1] == ASSISTS_C))
        self.assertTrue(np.all(assists[2] == ASSISTS_C))

    def test_a_gray_event_square_leaves_the_minute_stack(self) -> None:
        cells = cells_of(_picture(_view()))
        images = cells[None]
        where = np.zeros((1, ROWS, 6, COLS), dtype=bool)
        where[0, 0, :, 1] = True
        minutes = images[0, HEADER_CELLS : HEADER_CELLS + MINUTE_CELLS, 1].copy()
        apply_blank(images, where)
        self.assertTrue(np.array_equal(images[0, HEADER_CELLS : HEADER_CELLS + MINUTE_CELLS, 1], minutes))
        self.assertTrue(np.all(images[0, EVENT_ROW[0], 1] == 176))
        present = np.zeros((ROWS, COLS), dtype=bool)
        present[0, 1] = True
        ink = ink_of(images[0], present)
        self.assertEqual(int(ink[0, :, 1].sum()), 0)

    def test_a_filled_square_reads_back_as_ink(self) -> None:
        cells = cells_of(_picture(_view()))
        present = np.zeros((ROWS, COLS), dtype=bool)
        present[0, 1] = True
        ink = ink_of(cells, present)
        self.assertEqual(int(ink[0, :3, 1].sum()), 2)
        self.assertEqual(int(ink[0, 3:, 1].sum()), 1)
        self.assertEqual(int(ink[0, :, 0].sum()), 0)

    def test_a_coarse_block_keeps_a_filled_square(self) -> None:
        bits = np.zeros((1, ROWS, 6, COLS), dtype=bool)
        bits[0, 3, 0, 7] = True
        pooled = pool_max(bits, 20)
        self.assertEqual(pooled.shape, (1, 20, 6, 20))
        self.assertTrue(bool(pooled[0, 1, 0, 3]))
        self.assertEqual(int(pooled.sum()), 1)
        restored = repeat_blocks(pooled, 20)
        self.assertTrue(bool(restored[0, 3, 0, 7]))
        self.assertTrue(bool(restored[0, 2, 0, 6]))

    def test_future_squares_of_a_present_player_are_the_ones_to_paint(self) -> None:
        card = type("Card", (), {})()
        card.gw = np.arange(1, COLS + 1)
        card.present = np.zeros((ROWS, COLS), dtype=bool)
        card.present[2, 0] = True
        card.present[2, 10] = True
        hidden = hidden_events([card], [5])
        self.assertTrue(bool(hidden[0, 2, :, 10].all()))
        self.assertFalse(bool(hidden[0, 2, :, 0].any()))
        self.assertFalse(bool(hidden[0, 3].any()))

    def test_a_coarse_band_keeps_only_the_bottom_square(self) -> None:
        bits = np.zeros((1, ROWS, 6, COLS), dtype=bool)
        bits[0, 3, :, 7] = True
        picture = band_picture(bits, 20)
        self.assertTrue(bool(picture[0, 3, 2, 7]))
        self.assertTrue(bool(picture[0, 3, 5, 7]))
        self.assertFalse(bool(picture[0, :, 0].any()))
        self.assertFalse(bool(picture[0, :, 1].any()))
        self.assertFalse(bool(picture[0, :, 3].any()))
        self.assertFalse(bool(picture[0, :, 4].any()))

    def test_a_higher_square_is_not_a_target_while_the_lower_one_is_hidden(self) -> None:
        hidden = np.ones((1, 2, 6, 3), dtype=bool)
        _blank, drop = mask_from_bottom(hidden, np.random.default_rng(0))
        self.assertFalse(bool((drop[:, :, 1] & _blank[:, :, 2]).any()))
        self.assertFalse(bool((drop[:, :, 0] & (_blank[:, :, 1] | _blank[:, :, 2])).any()))
        self.assertFalse(bool((drop[:, :, 4] & _blank[:, :, 5]).any()))
        self.assertFalse(bool((drop[:, :, 3] & (_blank[:, :, 4] | _blank[:, :, 5])).any()))
        self.assertGreater(int(drop.sum()), 0)

    def test_three_gray_squares_keep_three_places(self) -> None:
        torch.manual_seed(0)
        model = CellPainter(causal=False)
        model.eval()
        patches = torch.zeros(1, SQ * SQ * 3, CELL_ROWS, COLS)
        blue = torch.tensor(MINUTES_C, dtype=torch.float32).div(255).repeat(SQ * SQ)
        gray = torch.tensor(MASK_C, dtype=torch.float32).div(255).repeat(SQ * SQ)
        for slot in range(MINUTE_CELLS):
            patches[0, :, HEADER_CELLS + slot, :] = blue[:, None]
        for slot in range(6):
            patches[0, :, HEADER_CELLS + MINUTE_CELLS + slot, :] = gray[:, None]
        logits = model(patches)[0, 0, :3, 0].detach()
        self.assertGreater(float((logits[0] - logits[1]).abs()), 1e-4)
        self.assertGreater(float((logits[1] - logits[2]).abs()), 1e-4)

    def test_a_column_cannot_read_a_later_column(self) -> None:
        torch.manual_seed(0)
        model = CellPainter(causal=True)
        model.eval()
        patches = torch.rand(2, SQ * SQ * 3, CELL_ROWS, COLS)
        first = model(patches)
        patches[:, :, :, 3:] = 1
        second = model(patches)
        self.assertEqual(first.shape, (2, ROWS, 6, COLS))
        self.assertTrue(torch.allclose(first[:, :, :, :3], second[:, :, :, :3]))

    def test_the_whole_card_can_read_a_later_column(self) -> None:
        torch.manual_seed(0)
        model = CellPainter(causal=False)
        model.eval()
        patches = torch.rand(1, SQ * SQ * 3, CELL_ROWS, COLS)
        first = model(patches)
        patches[:, :, :, 3:] = 0
        second = model(patches)
        self.assertFalse(torch.allclose(first[:, :, :, :3], second[:, :, :, :3]))
