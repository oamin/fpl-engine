"""The square card uses one column-width square as its unit."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.models.flashcard_squad_trial import (
    ASSISTS_C,
    AWAY_C,
    CELL_W,
    COLS,
    GOALS_C,
    HATCH_BG,
    HATCH_FG,
    HOME_C,
    MASK_C,
    MASK_ID,
    MINUTES_C,
    POS_C,
    ROWS,
)
from src.models.flashcard_square_trial import (
    CLUB_H,
    CLUB_MARKS,
    EVENT_H,
    GIVING,
    HEADER_H,
    MINUTE_H,
    ODDS_Y,
    OPP_H,
    PLAYER_H,
    RECEIVING,
    SQ,
    compose_binary,
    compose_square,
    minute_squares,
    paint_square,
)
from src.models.flashcard_strip_trial import StripVision


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
    return view


def _card():
    gw = np.zeros(COLS, dtype=np.int64)
    gw[:3] = (1, 2, 3)
    kick = np.zeros(COLS, dtype=np.int64)
    kick[0] = 1_700_000_000
    kick[2] = 1_700_086_400
    home = np.zeros(COLS, dtype=np.float32)
    home[0] = 1
    return SimpleNamespace(
        season="2024-25",
        team="Arsenal",
        n_fixtures=3,
        gw=gw,
        opponent=("Wolves", "Spurs", "Chelsea") + ("",) * (COLS - 3),
        kickoff=kick,
        home=home,
    )


def _prices(card) -> dict:
    import pandas as pd

    from src.teams import norm_team

    table = {}
    for col in (0, 2):
        date = pd.to_datetime(int(card.kickoff[col]), utc=True).strftime("%Y-%m-%d")
        team = norm_team(card.team)
        opponent = norm_team(card.opponent[col])
        if int(card.home[col]) == 1:
            key = (card.season, date, team, opponent)
        else:
            key = (card.season, date, opponent, team)
        table[key] = (-2.0, 0.5, 0.5, 0.75, 0.25)
    return table


def _pixel(image: np.ndarray, y: int, x: int) -> tuple[int, int, int]:
    return tuple(int(v) for v in image[y, x])


def _filled(image: np.ndarray, y0: int, height: int, x0: int, color: tuple[int, int, int]) -> int:
    return sum(
        _pixel(image, y, x0 + 1) == color
        for y in range(y0 + 1, y0 + height, SQ)
    )


class SquareCardTests(unittest.TestCase):
    def test_minute_bands_are_fifteen_minutes(self) -> None:
        expected = {
            0: 0,
            1: 1,
            15: 1,
            16: 2,
            30: 2,
            31: 3,
            45: 3,
            46: 4,
            60: 4,
            61: 5,
            75: 5,
            76: 6,
            90: 6,
            94: 6,
        }
        for minutes, squares in expected.items():
            self.assertEqual(minute_squares(minutes), squares)

    def test_the_rows_are_whole_squares(self) -> None:
        view = _view()
        view["minutes"][0, 0] = 15
        view["minutes"][0, 1] = 16
        view["minutes"][0, 2] = 90
        view["goal_token"][0, 0] = 1
        view["goal_token"][0, 1] = 3
        view["goal_token"][0, 2] = MASK_ID
        view["assist_token"][0, 0] = 1
        view["home"][1] = 0
        market = np.full((COLS, 3), np.nan)
        market[0] = (-2.0, 0.75, 0.25)
        market[1] = (0.75, 0.5, 0.5)
        visible = np.zeros(COLS, dtype=bool)
        visible[:2] = True
        image = paint_square(view, market, visible, club="Arsenal")
        self.assertEqual(image.shape[0], HEADER_H + ROWS * PLAYER_H)
        self.assertEqual(_pixel(image, ODDS_Y + 1, 1), GIVING)
        self.assertEqual(_pixel(image, ODDS_Y + 13 * SQ + 1, 1), GIVING)
        self.assertEqual(_pixel(image, ODDS_Y + 14 * SQ + 1, 1), RECEIVING)
        self.assertEqual(_pixel(image, ODDS_Y + 15 * SQ + 1, 1), RECEIVING)
        self.assertEqual(_pixel(image, ODDS_Y + 5 * SQ + 1, CELL_W + 1), GIVING)
        self.assertEqual(_pixel(image, ODDS_Y + 6 * SQ + 1, CELL_W + 1), RECEIVING)
        self.assertEqual(_pixel(image, CLUB_H + OPP_H + 1, 1), HOME_C)
        self.assertEqual(_pixel(image, CLUB_H + OPP_H + 1, CELL_W + 1), AWAY_C)
        y = HEADER_H
        self.assertEqual(_filled(image, y, MINUTE_H, 0, MINUTES_C), 1)
        self.assertEqual(_filled(image, y, MINUTE_H, CELL_W, MINUTES_C), 2)
        self.assertEqual(_filled(image, y, MINUTE_H, 2 * CELL_W, MINUTES_C), 6)
        goal = y + MINUTE_H
        self.assertEqual(_filled(image, goal, EVENT_H, 0, GOALS_C), 1)
        self.assertEqual(_filled(image, goal, EVENT_H, CELL_W, GOALS_C), 3)
        self.assertEqual(_pixel(image, goal + 1, 2 * CELL_W + 1), MASK_C)
        self.assertEqual(_filled(image, goal + EVENT_H, EVENT_H, 0, ASSISTS_C), 1)
        self.assertEqual(_pixel(image, y + PLAYER_H - SQ + 1, 1), POS_C[0])

    def test_a_step_keeps_odds_venue_and_minutes(self) -> None:
        view = _view()
        view["minutes"][0, 0] = 90
        view["goal_token"][0, 0] = 2
        card = _card()
        image = compose_square(view, card, _prices(card), np.array([0]), target=1)
        self.assertEqual(_pixel(image, ODDS_Y + 1, 1), GIVING)
        self.assertEqual(_pixel(image, CLUB_H + OPP_H + 1, 1), HOME_C)
        self.assertEqual(_pixel(image, HEADER_H + MINUTE_H - SQ + 1, 1), MINUTES_C)
        goal = _pixel(image, HEADER_H + MINUTE_H + 1, 1)
        position = _pixel(image, HEADER_H + PLAYER_H - SQ + 1, 1)
        self.assertIn(goal, (HATCH_FG, HATCH_BG))
        self.assertIn(position, (HATCH_FG, HATCH_BG))
        self.assertNotEqual(goal, GOALS_C)
        self.assertNotEqual(position, POS_C[0])
        self.assertEqual(_pixel(image, ODDS_Y + 1, 2 * CELL_W + 1), HATCH_FG)

    def test_the_square_picture_runs_through_the_reader(self) -> None:
        model = StripVision(HEADER_H, n_clubs=2, player_h=PLAYER_H)
        image = torch.zeros(1, 3, HEADER_H + ROWS * PLAYER_H, COLS * CELL_W)
        out, _assists = model(image, torch.zeros(1, ROWS, dtype=torch.bool), torch.ones(1, COLS))
        self.assertEqual(tuple(out.shape), (1, ROWS, COLS, 4))

    def test_the_opponent_square_is_the_club_symbol(self) -> None:
        card = _card()
        image = compose_square(_view(), card, _prices(card), np.array([0]), target=1)

        def seen(bit: str) -> tuple[int, int, int]:
            return (0, 0, 0) if bit == "1" else (255, 255, 255)

        for yy, row in enumerate(CLUB_MARKS["Arsenal"]):
            for xx, bit in enumerate(row):
                self.assertEqual(_pixel(image, yy, xx), seen(bit))
                self.assertEqual(_pixel(image, yy, CELL_W + xx), seen(bit))
        y = CLUB_H
        for yy, row in enumerate(CLUB_MARKS["Wolves"]):
            for xx, bit in enumerate(row):
                self.assertEqual(_pixel(image, y + yy, xx), seen(bit))
        for yy, row in enumerate(CLUB_MARKS["Spurs"]):
            for xx, bit in enumerate(row):
                self.assertEqual(_pixel(image, y + yy, CELL_W + xx), seen(bit))

    def test_the_binary_card_uses_black_white_and_a_rule(self) -> None:
        view = _view()
        view["minutes"][0, 0] = 15
        view["goal_token"][0, 0] = 1
        view["home"][1] = 0
        view["column_on"][1] = 1
        card = _card()
        image = compose_binary(view, card, _prices(card))
        self.assertEqual(_pixel(image, ODDS_Y, 0), (0, 0, 0))
        self.assertEqual(_pixel(image, ODDS_Y + 13 * SQ, 0), (0, 0, 0))
        self.assertEqual(_pixel(image, ODDS_Y + 14 * SQ, 0), (255, 255, 255))
        self.assertEqual(_pixel(image, ODDS_Y + 15 * SQ, 0), (255, 255, 255))
        self.assertEqual(_pixel(image, CLUB_H + OPP_H, 0), (0, 0, 0))
        self.assertEqual(_pixel(image, CLUB_H + OPP_H, CELL_W), (255, 255, 255))
        y = HEADER_H + MINUTE_H + 2 * EVENT_H
        self.assertEqual(tuple(int(v) for v in image[y, 0]), (255, 255, 255))
        self.assertEqual(tuple(int(v) for v in image[y + 1, 0]), (255, 255, 255))
        self.assertEqual(tuple(int(v) for v in image[y + 2, 0]), (0, 0, 0))
        self.assertEqual(tuple(int(v) for v in image[y + 3, 8 * CELL_W]), (0, 0, 0))

    def test_the_line_has_no_ticks_or_grid(self) -> None:
        view = _view()
        market = np.full((COLS, 3), np.nan)
        market[0] = (0.0, 0.5, 0.5)
        visible = np.zeros(COLS, dtype=bool)
        visible[0] = True
        image = paint_square(view, market, visible, club="Arsenal")
        self.assertEqual(_pixel(image, ODDS_Y + 7 * SQ + 1, 1), GIVING)
        self.assertEqual(_pixel(image, ODDS_Y + 8 * SQ + 1, 1), RECEIVING)


if __name__ == "__main__":
    unittest.main()
