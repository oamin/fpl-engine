"""The frozen reader keeps the grid and does not train the ImageNet weights."""

from __future__ import annotations

import unittest

import torch

from src.models.flashcard_imagenet_trial import CHUNK, ImageNetReader, ResNet18Front
from src.models.flashcard_squad_trial import BAND_H, CELL_W, COLS, PLAYER_H, ROWS


class ImageNetReaderTests(unittest.TestCase):
    def test_the_stem_stays_frozen_and_the_grid_comes_back(self) -> None:
        model = ImageNetReader(n_clubs=2, weights=None)
        self.assertTrue(all(not param.requires_grad for param in model.stem.parameters()))
        image = torch.zeros(1, 3, BAND_H + ROWS * PLAYER_H, COLS * CELL_W)
        image[:, :, :BAND_H, :CELL_W] = 1
        header, body = model.extract(image)
        self.assertEqual(tuple(header.shape), (1, COLS, 64))
        self.assertEqual(tuple(body.shape), (1, ROWS, COLS, 64))
        goals, assists = model(image, torch.zeros(1, ROWS, dtype=torch.bool), torch.ones(1, COLS))
        self.assertEqual(tuple(goals.shape), (1, ROWS, COLS, 4))
        self.assertEqual(tuple(assists.shape), (1, ROWS, COLS, 4))
        model.train()
        self.assertFalse(model.stem.training)
        self.assertGreater(CHUNK, 1)
        self.assertEqual(ResNet18Front().conv1.weight.shape, (64, 3, 7, 7))
