"""The stripped card keeps venue and drops the other four header rows."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_squad_trial import (
    BAND_H,
    CELL_W,
    COLS,
    CONCEDE_C,
    GOALS_C,
    HEADER_H,
    HOME_C,
    HATCH_FG,
    IMG_W,
    MASK_ID,
    MINUTES_C,
    PLAYER_H,
    ROWS,
    render_grid,
)
from src.models.flashcard_strip_trial import StripVision, compose_stripped, strip_features


def _view() -> dict[str, np.ndarray]:
    view = {
        "column_on": np.ones(COLS, dtype=np.float32),
        "home": np.ones(COLS, dtype=np.float32),
        "minutes": np.zeros((ROWS, COLS), dtype=np.float32),
        "present": np.ones((ROWS, COLS), dtype=np.float32),
        "goal_token": np.zeros((ROWS, COLS), dtype=np.int64),
        "assist_token": np.zeros((ROWS, COLS), dtype=np.int64),
        "position": np.zeros(ROWS, dtype=np.int64),
        "matchup": np.ones((COLS, 8), dtype=np.float32),
    }
    view["matchup"][:, 4] = 0.4
    view["matchup"][:, 5] = 0.0
    for key in ("concede", "regulars", "own_hist", "opp_hist", "own_form"):
        view[key] = np.ones(COLS, dtype=np.float32)
        view[key + "_missing"] = np.zeros(COLS, dtype=np.float32)
    view["concede"][:] = 3.0
    return view


class StripTests(unittest.TestCase):
    def test_the_four_rows_leave_the_picture_and_the_matchup(self) -> None:
        view = _view()
        full = render_grid(view)
        self.assertEqual(tuple(int(v) for v in full[BAND_H, 0]), CONCEDE_C)
        stripped = compose_stripped(strip_features(view))
        self.assertEqual(stripped.shape[0], BAND_H + ROWS * PLAYER_H)
        self.assertEqual(tuple(int(v) for v in stripped[0, 0]), HOME_C)
        self.assertNotEqual(tuple(int(v) for v in stripped[BAND_H, 0]), CONCEDE_C)
        self.assertEqual(float(view["matchup"][:, list((0, 1, 2, 3, 6, 7))].sum()), 0.0)
        self.assertAlmostEqual(float(view["matchup"][0, 4]), 0.4)
        self.assertEqual(full.shape[0], HEADER_H + ROWS * PLAYER_H)
        self.assertEqual(stripped.shape[1], IMG_W)

    def test_the_step_keeps_venue_and_minutes_on_the_short_card(self) -> None:
        view = strip_features(_view())
        view["goal_token"][0, 0] = 2
        view["minutes"][0, 2] = 90
        view["goal_token"][:, 2] = MASK_ID
        view["column_on"][3] = 0
        image = compose_stripped(view, np.array([2]))
        self.assertEqual(image.shape[0], BAND_H + ROWS * PLAYER_H)
        self.assertEqual(tuple(int(v) for v in image[0, 2 * CELL_W]), HOME_C)
        goal_y = BAND_H + 3 * BAND_H - 1
        self.assertEqual(tuple(int(v) for v in image[goal_y, 0]), GOALS_C)
        self.assertNotEqual(tuple(int(v) for v in image[goal_y, 2 * CELL_W]), GOALS_C)
        minute_y = BAND_H + 2 * BAND_H - 1
        self.assertEqual(tuple(int(v) for v in image[minute_y, 2 * CELL_W]), MINUTES_C)
        self.assertEqual(tuple(int(v) for v in image[0, 3 * CELL_W]), HATCH_FG)

    def test_the_short_header_runs_through_the_reader(self) -> None:
        model = StripVision(BAND_H, n_clubs=2)
        image = torch.zeros(1, 3, BAND_H + ROWS * PLAYER_H, COLS * CELL_W)
        row_pad = torch.zeros(1, ROWS, dtype=torch.bool)
        out, _assists = model(
            image,
            row_pad,
            torch.ones(1, COLS),
            torch.zeros(1, ROWS, dtype=torch.long),
            torch.zeros(1, ROWS, COLS),
            torch.zeros(1, COLS, 8),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, COLS, dtype=torch.long),
            None,
        )
        self.assertEqual(tuple(out.shape), (1, ROWS, COLS, 4))
