"""The remainder painter draws one code and paints the gray squares."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_image_trial import (
    HEADER_CELLS,
    MINUTE_CELLS,
    apply_blank,
    hidden_events,
    ink_of,
    patches_of,
)
from src.models.flashcard_remainder_trial import (
    CELL_ROWS,
    CODES,
    PLAYER_CELLS,
    RANK,
    RemainderPainter,
    assist_partners,
    draw_strings,
    form_runs,
    paint_remainder,
    string_density,
)
from src.models.flashcard_squad_trial import ASSISTS_C, COLS, D, GOALS_C, MASK_C, MINUTES_C, ROWS
from src.models.flashcard_square_trial import SQ


def _card() -> object:
    card = type("Card", (), {})()
    card.gw = np.arange(1, COLS + 1)
    card.present = np.zeros((ROWS, COLS), dtype=bool)
    card.minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    card.n_players = ROWS
    card.n_fixtures = COLS
    card.names = tuple(f"p{i}" for i in range(ROWS))
    card.position = np.zeros(ROWS, dtype=np.int64)
    card.team = "Arsenal"
    return card


def _cells() -> np.ndarray:
    return np.full((CELL_ROWS, COLS, SQ, SQ, 3), 255, dtype=np.uint8)


def _fill(cells: np.ndarray, player: int, column: int, slot: int, color: tuple[int, int, int]) -> None:
    row = HEADER_CELLS + player * PLAYER_CELLS + slot
    cells[row, column] = color


class RemainderPainterTests(unittest.TestCase):
    def test_one_code_is_a_single_pattern(self) -> None:
        torch.manual_seed(0)
        model = RemainderPainter()
        model.eval()
        logits = torch.zeros(2, CODES)
        generator = torch.Generator().manual_seed(1)
        code = model.pick(logits, generator)
        distance = (model.book.weight[None] - code[:, None]).abs().sum(dim=-1)
        self.assertEqual(code.shape, (2, D))
        self.assertTrue(bool(torch.all(distance.min(dim=-1).values < 1e-5)))
        self.assertTrue(bool(torch.all((distance < 1e-5).sum(dim=-1) == 1)))

    def test_one_code_changes_the_paint_in_every_column(self) -> None:
        torch.manual_seed(0)
        model = RemainderPainter()
        model.eval()
        patches = torch.rand(1, SQ * SQ * 3, CELL_ROWS, COLS)
        tokens = model.tokens_of(patches)
        quiet = model.decode(tokens, torch.zeros(1, D))
        busy = model.decode(tokens, torch.ones(1, D) * 3)
        self.assertEqual(quiet.shape, (1, ROWS, 6, COLS))
        self.assertFalse(torch.allclose(quiet[:, :, :, 0], busy[:, :, :, 0]))
        self.assertFalse(torch.allclose(quiet[:, :, :, -1], busy[:, :, :, -1]))

    def test_one_code_can_light_one_week_and_leave_the_next_dark(self) -> None:
        torch.manual_seed(0)
        model = RemainderPainter()
        model.eval()
        with torch.no_grad():
            model.paint.weight.zero_()
            model.week.weight.zero_()
            model.week.bias.zero_()
            model.player_goal.weight.zero_()
            model.player_goal.bias.zero_()
            model.week.bias.view(COLS, RANK)[3, 0] = 2
            model.week.bias.view(COLS, RANK)[4, 0] = -2
            model.player_goal.bias.view(ROWS, RANK)[0, 0] = 2
        tokens = torch.zeros(1, ROWS, COLS, D)
        paint = model.decode(tokens, torch.zeros(1, D)).detach()
        self.assertGreater(float(paint[0, 0, 2, 3]), float(paint[0, 0, 2, 4]))
        self.assertGreater(float(paint[0, 0, 2, 3]), 0.0)
        self.assertLess(float(paint[0, 0, 2, 4]), 0.0)

    def test_a_string_keeps_the_players_own_count(self) -> None:
        generator = torch.Generator().manual_seed(0)
        logits = torch.full((1, 1, 6, 8), -20.0)
        logits[0, 0, 2, 1] = 8
        hidden = torch.zeros(1, 1, 6, 8, dtype=torch.bool)
        hidden[0, 0, :, :] = True
        drawn = [draw_strings(logits, hidden, generator) for _ in range(40)]
        counts = torch.stack([item[0, 0, 2].sum() for item in drawn]).float()
        hot = torch.stack([item[0, 0, 2, 1] for item in drawn]).float()
        self.assertLess(float(counts.mean()), 3.0)
        self.assertGreater(float(hot.mean()), 0.5)
        self.assertEqual(int(torch.stack([item[0, 0, 0].sum() + item[0, 0, 1].sum() for item in drawn]).sum()), 0)

    def test_a_full_string_is_the_busiest_player(self) -> None:
        goals = np.zeros((1, ROWS, COLS), dtype=np.int64)
        minutes = np.zeros((1, ROWS, COLS), dtype=np.float32)
        goals[0, 0, 2:8] = 1
        goals[0, 1, 3] = 1
        minutes[0, :, :] = 90
        density = string_density(goals, minutes, [_card()], [0])
        self.assertEqual(density["peak"], 6)
        self.assertGreater(density["share"], 0)

    def test_a_later_column_reaches_an_earlier_one(self) -> None:
        torch.manual_seed(0)
        model = RemainderPainter()
        model.eval()
        patches = torch.rand(1, SQ * SQ * 3, CELL_ROWS, COLS)
        code = torch.zeros(1, D)
        first = model.decode(model.tokens_of(patches), code)
        patches = patches.clone()
        patches[:, :, :, 20:] = 0
        second = model.decode(model.tokens_of(patches), code)
        self.assertFalse(torch.allclose(first[:, :, :, :5], second[:, :, :, :5]))

    def test_one_player_reaches_a_teammate_in_the_same_week(self) -> None:
        torch.manual_seed(0)
        model = RemainderPainter()
        model.eval()
        patches = torch.rand(1, SQ * SQ * 3, CELL_ROWS, COLS)
        code = torch.zeros(1, D)
        first = model.decode(model.tokens_of(patches), code)
        patches = patches.clone()
        start = HEADER_CELLS
        patches[:, :, start : start + PLAYER_CELLS, 4] = 0
        second = model.decode(model.tokens_of(patches), code)
        self.assertFalse(torch.allclose(first[:, 10, :, 4], second[:, 10, :, 4]))

    def test_one_pass_commits_every_gray_square_and_leaves_the_minutes(self) -> None:
        torch.manual_seed(0)
        model = RemainderPainter()
        card = _card()
        card.present[:, 8:] = True
        card.minutes[:, 8:] = 90
        cells = _cells()
        for player in range(ROWS):
            for slot in range(MINUTE_CELLS):
                _fill(cells, player, 10, slot, MINUTES_C)
        minutes = cells[HEADER_CELLS : HEADER_CELLS + MINUTE_CELLS, 10].copy()
        generator = torch.Generator().manual_seed(0)
        _goals, _assists, kept, _ink, _fault = paint_remainder(
            model, [card], cells[None], [4], torch.device("cpu"), generator, keep=0
        )
        self.assertIsNotNone(kept)
        hidden = hidden_events([card], [4])
        picked = kept[EVENT_INDEX()]
        gray = np.all(picked == MASK_C, axis=(-1, -2, -3))
        self.assertEqual(gray.shape, hidden[0].shape)
        self.assertFalse(bool((gray & hidden[0]).any()))
        self.assertTrue(np.array_equal(kept[HEADER_CELLS : HEADER_CELLS + MINUTE_CELLS, 10], minutes))

    def test_a_neighboring_goal_counts_as_a_run(self) -> None:
        goals = np.zeros((1, ROWS, COLS), dtype=np.int64)
        goals[0, 0, 2] = 1
        goals[0, 0, 3] = 1
        goals[0, 1, 5] = 1
        beside, total = form_runs(goals, [_card()], [0])
        self.assertEqual(total, 3)
        self.assertEqual(beside, 2)

    def test_an_assist_on_another_player_pairs_the_goal(self) -> None:
        goals = np.zeros((1, ROWS, COLS), dtype=np.int64)
        assists = np.zeros((1, ROWS, COLS), dtype=np.int64)
        minutes = np.zeros((1, ROWS, COLS), dtype=np.float32)
        goals[0, 0, 4] = 1
        assists[0, 2, 4] = 1
        minutes[0, :, 4] = 90
        card = _card()
        paired, total = assist_partners(goals, assists, minutes, [card], [0])
        self.assertEqual((paired, total), (1, 1))
        assists[0, 2, 4] = 0
        assists[0, 0, 4] = 1
        paired, total = assist_partners(goals, assists, minutes, [card], [0])
        self.assertEqual((paired, total), (0, 1))

    def test_one_code_rebuilds_a_goal_and_a_teammate_assist(self) -> None:
        torch.manual_seed(0)
        model = RemainderPainter()
        cells = _cells()
        for player in (0, 1):
            for slot in range(MINUTE_CELLS):
                _fill(cells, player, 5, slot, MINUTES_C)
        _fill(cells, 0, 5, MINUTE_CELLS + 2, GOALS_C)
        _fill(cells, 1, 5, MINUTE_CELLS + 5, ASSISTS_C)
        card = _card()
        card.present[0, 5] = True
        card.present[1, 5] = True
        card.minutes[0, 5] = 90
        card.minutes[1, 5] = 90
        present = card.present
        ink = ink_of(cells, present)
        self.assertTrue(bool(ink[0, 2, 5]))
        self.assertTrue(bool(ink[1, 5, 5]))
        self.assertFalse(bool(ink[0, 0, 5]))
        hidden = hidden_events([card], [1])
        blank = cells[None].copy()
        apply_blank(blank, hidden)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        full = patches_of(cells[None])
        seen = patches_of(blank)
        target = torch.from_numpy(ink[None]).float()
        mask = torch.from_numpy(hidden).float()
        model.train()
        for _ in range(200):
            opt.zero_grad()
            logits, _, _ = model(full, seen)
            per = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = (per * mask).sum() / mask.sum()
            loss.backward()
            opt.step()
        model.eval()
        tokens_full = model.tokens_of(full)
        tokens_seen = model.tokens_of(seen)
        index = int(model.full_logits(tokens_full).argmax(dim=-1))
        paint = model.decode(tokens_seen, model.book.weight[index][None]).detach()
        self.assertGreater(float(paint[0, 0, 2, 5]), 0.0)
        self.assertGreater(float(paint[0, 1, 5, 5]), 0.0)
        self.assertLess(float(paint[0, 0, 0, 5]), 0.0)


def EVENT_INDEX() -> np.ndarray:
    rows = [
        HEADER_CELLS + player * PLAYER_CELLS + MINUTE_CELLS + slot
        for player in range(ROWS)
        for slot in range(6)
    ]
    return np.array(rows, dtype=np.int64).reshape(ROWS, 6)
