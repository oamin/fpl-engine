"""The denoiser restores event colors, and the bars are read from that picture."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_diffusion_trial import Denoiser, overwrite_events, q_sample, restore
from src.models.flashcard_image_trial import EVENT_ROW, EVENT_SLOTS, patches_of
from src.models.flashcard_remainder_trial import CELL_ROWS
from src.models.flashcard_squad_trial import COLS, ROWS
from src.models.flashcard_square_trial import SQ


class DiffusionPainterTests(unittest.TestCase):
    def test_the_schedule_starts_clean_and_ends_in_noise(self) -> None:
        model = Denoiser(steps=32)
        self.assertGreater(float(model.bar[0]), 0.99)
        self.assertLess(float(model.bar[-1]), 0.1)

    def test_a_noised_square_can_be_written_without_touching_its_neighbor(self) -> None:
        cells = np.full((1, CELL_ROWS, COLS, SQ, SQ, 3), 255, dtype=np.uint8)
        patches = patches_of(cells)
        colors = torch.zeros(1, ROWS, EVENT_SLOTS, COLS, 3)
        colors[0, 0, 0, 1] = torch.tensor([0.25, 0.5, 0.75])
        where = torch.zeros(1, ROWS, EVENT_SLOTS, COLS, dtype=torch.bool)
        where[0, 0, 0, 1] = True
        written = overwrite_events(patches, colors, where)
        row = int(EVENT_ROW[0, 0])
        self.assertTrue(bool(torch.allclose(written[0, 0:3, row, 1], torch.tensor([0.25, 0.5, 0.75]))))
        self.assertTrue(bool(torch.allclose(written[0, :, row, 2], patches[0, :, row, 2])))

    def test_the_first_noise_level_keeps_the_picture(self) -> None:
        model = Denoiser(steps=8)
        x0 = torch.rand(2, ROWS, EVENT_SLOTS, COLS, 3)
        eps = torch.randn_like(x0)
        noised = q_sample(model, x0, torch.zeros(2, dtype=torch.long), eps)
        self.assertLess(float((noised - x0).abs().mean()), 0.02)

    def test_a_restore_returns_one_color_per_event_square(self) -> None:
        torch.manual_seed(0)
        model = Denoiser(steps=2)
        model.eval()
        patches = torch.rand(1, SQ * SQ * 3, CELL_ROWS, COLS)
        hidden = torch.zeros(1, ROWS, EVENT_SLOTS, COLS, dtype=torch.bool)
        hidden[:, :, :, 4:] = True
        generator = torch.Generator().manual_seed(1)
        colors = restore(model, patches, hidden, generator)
        self.assertEqual(colors.shape, (1, ROWS, EVENT_SLOTS, COLS, 3))
        self.assertTrue(bool(torch.all((colors >= 0) & (colors <= 1))))
