"""The stripped card keeps venue and drops the other four header rows."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_squad_trial import (
    BAND_H,
    CELL_W,
    COLS,
    ASSISTS_C,
    CONCEDE_C,
    EVENT_H,
    GOALS_C,
    HEADER_H,
    HOME_C,
    HATCH_FG,
    IMG_W,
    MASK_C,
    MASK_ID,
    MINUTE_H,
    MINUTES_C,
    PLAYER_H,
    ROWS,
    event_pixels,
    render_grid,
)
from src.models.flashcard_odds_trial import AWAY_ODDS, DRAW_ODDS, HOME_ODDS
from src.models.flashcard_square_trial import EVENT_H as SQUARE_EVENT
from src.models.flashcard_square_trial import HEADER_H as SQUARE_HEADER
from src.models.flashcard_square_trial import MINUTE_H as SQUARE_MINUTE
from src.models.flashcard_square_trial import ODDS_H, PLAYER_H as SQUARE_PLAYER
from src.models.flashcard_strip_trial import StripVision, compose_card, compose_stripped, strip_features


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


def _card():
    from types import SimpleNamespace

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
        table[key] = (0.844, 0.110, 0.046)
    return table


class StripTests(unittest.TestCase):
    def test_the_four_rows_leave_the_picture_and_the_matchup(self) -> None:
        view = _view()
        full = render_grid(view)
        self.assertEqual(tuple(int(v) for v in full[BAND_H, 0]), CONCEDE_C)
        stripped = compose_stripped(strip_features(view))
        self.assertEqual(stripped.shape[0], BAND_H + ROWS * PLAYER_H)
        self.assertEqual(tuple(int(v) for v in stripped[0, 0]), HOME_C)
        self.assertNotEqual(tuple(int(v) for v in stripped[BAND_H, 0]), CONCEDE_C)
        self.assertEqual(float(view["matchup"].sum()), 0.0)
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
        goal_y = BAND_H + MINUTE_H + EVENT_H - 1
        self.assertEqual(tuple(int(v) for v in image[goal_y, 0]), GOALS_C)
        self.assertNotEqual(tuple(int(v) for v in image[goal_y, 2 * CELL_W]), GOALS_C)
        minute_y = BAND_H + MINUTE_H - 1
        self.assertEqual(tuple(int(v) for v in image[minute_y, 2 * CELL_W]), MINUTES_C)
        self.assertEqual(tuple(int(v) for v in image[0, 3 * CELL_W]), HATCH_FG)

    def test_the_short_header_runs_through_the_reader(self) -> None:
        model = StripVision(BAND_H, n_clubs=2)
        image = torch.zeros(1, 3, BAND_H + ROWS * PLAYER_H, COLS * CELL_W)
        row_pad = torch.zeros(1, ROWS, dtype=torch.bool)
        out, _assists = model(image, row_pad, torch.ones(1, COLS))
        self.assertEqual(tuple(out.shape), (1, ROWS, COLS, 4))

    def test_one_goal_is_several_pixels_and_three_goals_fill_the_band(self) -> None:
        view = strip_features(_view())
        view["goal_token"][0, 0] = 1
        view["goal_token"][0, 1] = 3
        view["goal_token"][0, 2] = MASK_ID
        view["assist_token"][0, 0] = 1
        image = compose_stripped(view)
        y0 = BAND_H + MINUTE_H
        one = np.all(image[y0 : y0 + EVENT_H, 0:CELL_W] == GOALS_C, axis=-1).all(axis=1)
        self.assertEqual(int(one.sum()), event_pixels(1))
        self.assertEqual(event_pixels(1), 5)
        self.assertTrue(bool(one[-1]))
        self.assertFalse(bool(one[0]))
        three = image[y0 : y0 + EVENT_H, CELL_W : 2 * CELL_W]
        self.assertTrue(np.all(three == GOALS_C))
        hidden = image[y0 : y0 + EVENT_H, 2 * CELL_W : 3 * CELL_W]
        self.assertTrue(np.all(hidden == MASK_C))
        assist_y = y0 + EVENT_H
        assist = np.all(image[assist_y : assist_y + EVENT_H, 0:CELL_W] == ASSISTS_C, axis=-1).all(axis=1)
        self.assertEqual(int(assist.sum()), event_pixels(1))

    def test_the_default_card_is_the_square_card(self) -> None:
        view = strip_features(_view())
        view["minutes"][0, 0] = 15
        view["goal_token"][0, 0] = 1
        view["column_on"][1] = 0
        card = _card()
        image = compose_card(view, card, _prices(card))
        self.assertEqual(image.shape[0], SQUARE_HEADER + ROWS * SQUARE_PLAYER)
        self.assertEqual(tuple(int(v) for v in image[0, 0]), HOME_ODDS)
        self.assertEqual(tuple(int(v) for v in image[51, 0]), HOME_ODDS)
        self.assertEqual(tuple(int(v) for v in image[52, 0]), DRAW_ODDS)
        self.assertEqual(tuple(int(v) for v in image[60, 0]), AWAY_ODDS)
        self.assertEqual(tuple(int(v) for v in image[ODDS_H, 0]), HOME_C)
        minute_y = SQUARE_HEADER
        self.assertEqual(tuple(int(v) for v in image[minute_y + SQUARE_MINUTE - 1, 0]), MINUTES_C)
        self.assertNotEqual(tuple(int(v) for v in image[minute_y + SQUARE_MINUTE - CELL_W - 1, 0]), MINUTES_C)
        self.assertEqual(tuple(int(v) for v in image[minute_y + SQUARE_MINUTE + SQUARE_EVENT - 1, 0]), GOALS_C)
        self.assertEqual(tuple(int(v) for v in image[0, CELL_W]), HATCH_FG)
        model = StripVision(SQUARE_HEADER, n_clubs=2, player_h=SQUARE_PLAYER)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        out, _assists = model(tensor, torch.zeros(1, ROWS, dtype=torch.bool), torch.ones(1, COLS))
        self.assertEqual(tuple(out.shape), (1, ROWS, COLS, 4))
        fresh = strip_features(_view())
        stepped = compose_card(fresh, card, _prices(card), target=1)
        self.assertEqual(tuple(int(v) for v in stepped[0, 0]), HOME_ODDS)
        self.assertEqual(tuple(int(v) for v in stepped[0, 2 * CELL_W]), HATCH_FG)

    def test_the_locked_reader_has_no_numeric_sidecar(self) -> None:
        model = StripVision(BAND_H, n_clubs=2)
        names = [name for name, _param in model.named_parameters()]
        banned = ("pos_goal", "club_vec", "opp_vec", "form_goal", "club_context", "club_goal", "cross")
        self.assertFalse(any(any(token in name for token in banned) for name in names))
        image = torch.zeros(1, 3, BAND_H + ROWS * PLAYER_H, COLS * CELL_W)
        out, _assists = model(image, torch.zeros(1, ROWS, dtype=torch.bool), torch.ones(1, COLS))
        self.assertEqual(tuple(out.shape), (1, ROWS, COLS, 4))
