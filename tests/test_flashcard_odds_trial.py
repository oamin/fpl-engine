"""Odds stay inside the horizon, and the bar is green, grey, red from the top."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_odds_trial import (
    AWAY_ODDS,
    DRAW_ODDS,
    HOME_ODDS,
    ODDS_H,
    OddsVision,
    paint_odds_bar,
    paint_odds_strip,
    split_heights,
    visible_odds,
)
from src.models.flashcard_squad_trial import BAND_H, CELL_W, COLS, IMG_W, PLAYER_H, ROWS


class OddsTests(unittest.TestCase):
    def test_half_fifth_and_three_tenths_fill_the_bar(self) -> None:
        heights = split_heights(np.array([0.5, 0.2, 0.3]), 10)
        self.assertEqual(heights.tolist(), [5, 2, 3])
        image = np.zeros((BAND_H, CELL_W, 3), dtype=np.uint8)
        paint_odds_bar(image, 0, np.array([0.5, 0.2, 0.3]))
        counts = split_heights(np.array([0.5, 0.2, 0.3]), BAND_H)
        self.assertEqual(tuple(int(v) for v in image[0, 0]), HOME_ODDS)
        self.assertEqual(tuple(int(v) for v in image[int(counts[0]), 0]), DRAW_ODDS)
        self.assertEqual(tuple(int(v) for v in image[int(counts[0] + counts[1]), 0]), AWAY_ODDS)

    def test_the_painter_shows_only_the_next_two_gameweeks(self) -> None:
        gameweeks = np.array([10, 11, 12, 13, 14, 0])
        show = visible_odds(gameweeks, "painter", cut=11)
        self.assertEqual(show.tolist(), [False, False, True, True, False, False])

    def test_the_step_shows_the_past_and_the_next_week_only(self) -> None:
        gameweeks = np.array([10, 11, 12, 13])
        show = visible_odds(gameweeks, "step", target=12)
        self.assertEqual(show.tolist(), [True, True, True, False])

    def test_a_hidden_price_is_hatched(self) -> None:
        column_on = np.ones(COLS, dtype=np.float32)
        probs = np.full((COLS, 3), np.nan)
        probs[1] = (0.5, 0.2, 0.3)
        visible = np.zeros(COLS, dtype=bool)
        visible[1] = True
        strip = paint_odds_strip(column_on, probs, visible)
        self.assertEqual(tuple(int(v) for v in strip[0, 0]), (168, 168, 168))
        self.assertEqual(tuple(int(v) for v in strip[0, CELL_W]), HOME_ODDS)

    def test_the_taller_header_runs_through_the_reader(self) -> None:
        model = OddsVision(n_clubs=2)
        height = ODDS_H + ROWS * PLAYER_H
        image = torch.zeros(1, 3, height, IMG_W)
        row_pad = torch.zeros(1, ROWS, dtype=torch.bool)
        column_on = torch.ones(1, COLS)
        position = torch.zeros(1, ROWS, dtype=torch.long)
        minutes = torch.zeros(1, ROWS, COLS)
        matchup = torch.zeros(1, COLS, 8)
        club = torch.zeros(1, dtype=torch.long)
        opponent = torch.zeros(1, COLS, dtype=torch.long)
        goals, assists = model(image, row_pad, column_on, position, minutes, matchup, club, opponent, None)
        self.assertEqual(tuple(goals.shape), (1, ROWS, COLS, 4))
        self.assertEqual(tuple(assists.shape), (1, ROWS, COLS, 4))
