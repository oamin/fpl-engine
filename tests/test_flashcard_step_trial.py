"""The next gameweek keeps minutes. Every later column is off."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_squad_trial import (
    BAND_H,
    CELL_W,
    COLS,
    HEADER_H,
    MASK_ID,
    ROWS,
    grey_except_minutes,
)
from src.models.flashcard_step_trial import chop_future


def _view() -> dict[str, np.ndarray]:
    view = {
        "column_on": np.ones(COLS, dtype=np.float32),
        "minutes": np.full((ROWS, COLS), 90, dtype=np.float32),
        "present": np.ones((ROWS, COLS), dtype=np.float32),
        "goal_token": np.ones((ROWS, COLS), dtype=np.int64),
        "assist_token": np.ones((ROWS, COLS), dtype=np.int64),
        "home": np.ones(COLS, dtype=np.float32),
        "opponent": np.full(COLS, 3, dtype=np.int64),
        "matchup": np.ones((COLS, 8), dtype=np.float32),
        "opp_minutes": np.full((ROWS, COLS), 80, dtype=np.float32),
        "opp_present": np.ones((ROWS, COLS), dtype=np.float32),
        "opp_position": np.zeros((ROWS, COLS), dtype=np.int64),
        "opp_goal_token": np.ones((ROWS, COLS), dtype=np.int64),
        "opp_assist_token": np.ones((ROWS, COLS), dtype=np.int64),
        "opp_pad": np.zeros((ROWS, COLS), dtype=bool),
    }
    for key in ("concede", "regulars", "own_hist", "opp_hist", "own_form"):
        view[key] = np.ones(COLS, dtype=np.float32)
        view[key + "_missing"] = np.zeros(COLS, dtype=np.float32)
    return view


class StepTests(unittest.TestCase):
    def test_future_columns_are_cleared_and_the_target_keeps_minutes(self) -> None:
        gameweeks = np.zeros(COLS, dtype=np.int64)
        gameweeks[:4] = np.array([23, 24, 25, 26])
        view = _view()
        view["goal_token"][:, 0] = 2
        chop_future(view, gameweeks, 25)
        self.assertEqual(float(view["minutes"][0, 2]), 90.0)
        self.assertEqual(float(view["minutes"][0, 3]), 0.0)
        self.assertEqual(float(view["column_on"][3]), 0.0)
        self.assertEqual(int(view["goal_token"][0, 2]), MASK_ID)
        self.assertEqual(int(view["goal_token"][0, 0]), 2)
        self.assertEqual(int(view["opponent"][3]), 0)
        self.assertEqual(float(view["matchup"][2, 0]), 0.0)
        self.assertEqual(float(view["opp_minutes"][0, 3]), 0.0)
        self.assertTrue(bool(view["opp_pad"][0, 2]))

    def test_grey_leaves_the_minute_band(self) -> None:
        image = np.zeros((HEADER_H + ROWS * 5 * BAND_H, COLS * CELL_W, 3), dtype=np.uint8)
        minute_y = HEADER_H + 1
        goal_y = HEADER_H + 2 * BAND_H + 1
        image[minute_y, 0, :] = (0, 114, 178)
        image[goal_y, 0, :] = (213, 94, 0)
        grey_except_minutes(image, np.array([0]))
        self.assertEqual(tuple(int(v) for v in image[minute_y, 0]), (0, 114, 178))
        self.assertNotEqual(tuple(int(v) for v in image[goal_y, 0]), (213, 94, 0))
