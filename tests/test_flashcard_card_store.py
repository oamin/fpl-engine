"""The current card and the 798 card stay stored as different pictures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_card_store import load_reader, save_reader
from src.models.flashcard_legacy_card import HEADER_H as LEGACY_HEADER
from src.models.flashcard_legacy_card import ODDS_H, paint_legacy
from src.models.flashcard_odds_trial import HOME_ODDS
from src.models.flashcard_squad_trial import CELL_W, COLS, HOME_C, ROWS
from src.models.flashcard_square_trial import HEADER_H, PLAYER_H, paint_square


def _view() -> dict[str, np.ndarray]:
    view = {
        "column_on": np.ones(COLS, dtype=np.float32),
        "home": np.ones(COLS, dtype=np.float32),
        "minutes": np.zeros((ROWS, COLS), dtype=np.float32),
        "present": np.ones((ROWS, COLS), dtype=np.float32),
        "goal_token": np.zeros((ROWS, COLS), dtype=np.int64),
        "assist_token": np.zeros((ROWS, COLS), dtype=np.int64),
        "position": np.zeros(ROWS, dtype=np.int64),
    }
    view["position"][1:] = -1
    view["minutes"][0, 0] = 90
    return view


class CardStoreTests(unittest.TestCase):
    def test_the_legacy_header_is_odds_then_venue_then_opponent(self) -> None:
        probs = np.full((COLS, 3), np.nan)
        probs[0] = (0.844, 0.110, 0.046)
        visible = np.zeros(COLS, dtype=bool)
        visible[0] = True
        image = paint_legacy(_view(), probs, visible, opponents=("Wolves",) + ("",) * (COLS - 1))
        self.assertEqual(LEGACY_HEADER, 72)
        self.assertEqual(image.shape[0], LEGACY_HEADER + ROWS * PLAYER_H)
        self.assertEqual(tuple(int(v) for v in image[0, 0]), HOME_ODDS)
        self.assertEqual(tuple(int(v) for v in image[ODDS_H, 0]), HOME_C)
        self.assertNotEqual(tuple(int(v) for v in image[ODDS_H + CELL_W, 0]), HOME_ODDS)

    def test_the_current_card_is_taller_than_the_legacy_card(self) -> None:
        self.assertEqual(HEADER_H, 76)
        self.assertGreater(HEADER_H, LEGACY_HEADER)
        probs = np.full((COLS, 3), np.nan)
        visible = np.zeros(COLS, dtype=bool)
        current = paint_square(_view(), probs, visible, club="Arsenal")
        legacy = paint_legacy(_view(), probs, visible)
        self.assertEqual(current.shape[0] - legacy.shape[0], CELL_W)

    def test_a_saved_reader_loads_the_same_weights(self) -> None:
        from src.models.flashcard_intensity_trial import IntensityReader

        torch.manual_seed(0)
        model = IntensityReader(LEGACY_HEADER, mode="fixture", dropout=0.1, player_h=PLAYER_H)
        with torch.no_grad():
            model.header.weight.add_(1.0)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            save_reader(
                directory,
                "horizon",
                model,
                {"role": "horizon", "dropout": 0.1, "arch": "fixture_intensity", "mode": "fixture"},
            )
            loaded = load_reader(directory, "horizon")
        self.assertTrue(torch.equal(loaded.header.weight, model.header.weight))
        self.assertEqual(loaded.header.in_features, 3 * LEGACY_HEADER * CELL_W)
        self.assertEqual(type(loaded).__name__, "IntensityReader")
