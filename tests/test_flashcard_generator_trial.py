"""The three card painters keep their order and their square cap."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_generator_trial import (
    MASKGIT_STEPS,
    TokenGrid,
    any_pool,
    context_tokens,
    cosine_ratio,
    event_bits,
    max_pool,
    still_masked,
)
from src.models.flashcard_squad_trial import COLS, MASK_ID, ROWS


class GeneratorTests(unittest.TestCase):
    def test_the_mask_share_shrinks_across_the_passes(self) -> None:
        self.assertAlmostEqual(cosine_ratio(0.0), 1.0)
        self.assertAlmostEqual(cosine_ratio(1.0), 0.0)
        self.assertGreater(cosine_ratio(0.25), cosine_ratio(0.75))

    def test_a_hidden_square_is_masked_and_a_revealed_square_keeps_its_count(self) -> None:
        labels = np.zeros((2, 2), dtype=np.int64)
        labels[0, 0] = 2
        labels[1, 1] = 1
        hidden = np.array([[True, False], [False, True]])
        context = context_tokens(labels, hidden)
        self.assertEqual(int(context[0, 0]), MASK_ID)
        self.assertEqual(int(context[1, 1]), MASK_ID)
        self.assertEqual(int(context[0, 1]), 0)
        self.assertEqual(int(context[1, 0]), 0)

    def test_the_sure_squares_are_kept(self) -> None:
        confidence = np.array([0.1, 0.9, 0.4, 0.8])
        unknown = np.array([True, True, True, True])
        early = still_masked(confidence, unknown, 0, MASKGIT_STEPS, 4)
        self.assertGreater(int(early.sum()), 0)
        self.assertFalse(bool(early[1]))
        done = still_masked(confidence, unknown, MASKGIT_STEPS - 1, MASKGIT_STEPS, 4)
        self.assertEqual(int(done.sum()), 0)

    def test_a_coarse_block_keeps_the_highest_count(self) -> None:
        values = np.zeros((1, ROWS, COLS), dtype=np.int64)
        values[0, 3, 7] = 2
        pooled = max_pool(values, 20)
        self.assertEqual(pooled.shape, (1, 20, 20))
        self.assertEqual(int(pooled[0, 1, 3]), 2)
        self.assertEqual(int(pooled.sum()), 2)
        mask = np.zeros((1, ROWS, COLS), dtype=bool)
        mask[0, 3, 7] = True
        self.assertTrue(bool(any_pool(mask, 20)[0, 1, 3]))
        self.assertEqual(int(any_pool(mask, 20).sum()), 1)

    def test_a_column_cannot_read_a_later_column(self) -> None:
        torch.manual_seed(0)
        model = TokenGrid(n_clubs=4, causal=True)
        model.eval()
        batch, rows, cols = 2, 4, 6
        goal = torch.randint(0, 5, (batch, rows, cols))
        assist = torch.randint(0, 5, (batch, rows, cols))
        parent = torch.full_like(goal, MASK_ID)
        position = torch.zeros(batch, rows, dtype=torch.long)
        minutes = torch.zeros(batch, rows, cols)
        opponent = torch.zeros(batch, cols, dtype=torch.long)
        features = torch.zeros(batch, cols, 6)
        row_pad = torch.zeros(batch, rows, dtype=torch.bool)
        col_pad = torch.zeros(batch, cols, dtype=torch.bool)
        first, _ = model(goal, assist, parent, parent, position, minutes, opponent, features, row_pad, col_pad)
        goal[:, :, 3:] = 3
        second, _ = model(goal, assist, parent, parent, position, minutes, opponent, features, row_pad, col_pad)
        self.assertTrue(torch.equal(first[:, :, :3], second[:, :, :3]))

    def test_a_count_is_three_squares_and_a_mask_bit(self) -> None:
        bits = event_bits(np.array([0, 2, 3, MASK_ID]))
        self.assertTrue(np.array_equal(bits[0], [0, 0, 0, 0]))
        self.assertTrue(np.array_equal(bits[1], [1, 1, 0, 0]))
        self.assertTrue(np.array_equal(bits[2], [1, 1, 1, 0]))
        self.assertTrue(np.array_equal(bits[3], [0, 0, 0, 1]))

    def test_the_binary_head_has_one_logit_per_square(self) -> None:
        torch.manual_seed(0)
        model = TokenGrid(n_clubs=4, causal=False, binary=True)
        model.eval()
        batch, rows, cols = 1, 4, 4
        goal = torch.tensor([0, 1, 2, MASK_ID]).reshape(1, 1, 4).expand(batch, rows, cols).contiguous()
        assist = torch.zeros_like(goal)
        parent = torch.full_like(goal, MASK_ID)
        position = torch.zeros(batch, rows, dtype=torch.long)
        minutes = torch.zeros(batch, rows, cols)
        opponent = torch.zeros(batch, cols, dtype=torch.long)
        features = torch.zeros(batch, cols, 6)
        row_pad = torch.zeros(batch, rows, dtype=torch.bool)
        col_pad = torch.zeros(batch, cols, dtype=torch.bool)
        out, _ = model(goal, assist, parent, parent, position, minutes, opponent, features, row_pad, col_pad)
        self.assertEqual(tuple(out.shape), (batch, rows, cols, 3))

    def test_a_draw_stays_inside_the_row(self) -> None:
        generator = torch.Generator().manual_seed(0)
        logits = torch.zeros(2, 3, 4)
        logits[..., 2] = 4.0
        probs = torch.softmax(logits, dim=-1)
        choice = torch.multinomial(probs.reshape(-1, 4), 1, generator=generator).reshape(2, 3)
        self.assertTrue(int(choice.min()) >= 0)
        self.assertTrue(int(choice.max()) <= 3)
        self.assertTrue(torch.all(choice == 2))
