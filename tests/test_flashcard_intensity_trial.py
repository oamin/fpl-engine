"""Owned rates sum to the card volume before paint."""

from __future__ import annotations

import unittest

import torch

from src.models.flashcard_intensity_trial import IntensityReader, intensity_goal_loss
from src.models.flashcard_legacy_card import HEADER_H
from src.models.flashcard_squad_trial import COLS, ROWS
from src.models.flashcard_square_trial import PLAYER_H


class IntensityTrialTests(unittest.TestCase):
    def test_row_mass_equals_the_owned_rate(self) -> None:
        model = IntensityReader(HEADER_H, mode="row", player_h=PLAYER_H)
        images = torch.zeros(2, 3, HEADER_H + ROWS * PLAYER_H, COLS * 4)
        row_pad = torch.zeros(2, ROWS, dtype=torch.bool)
        column_on = torch.ones(2, COLS)
        loss = torch.zeros(2, ROWS, COLS)
        loss[:, :3, :4] = 1.0
        pack = model(images, row_pad, column_on, loss)
        owned = pack["row_mu"][:, :3]
        mass = pack["presence"][:, :3, :4].sum(dim=-1)
        self.assertTrue(torch.allclose(owned, mass, atol=1e-5))

    def test_a_cold_owned_rate_costs_more_than_a_matched_rate(self) -> None:
        labels = torch.zeros(1, 1, 4)
        labels[0, 0, :2] = 1
        mask = torch.ones(1, 1, 4)
        matched = {
            "presence": torch.full((1, 1, 4), 0.5),
            "row_mu": torch.tensor([[2.0]]),
            "height_logits": torch.zeros(1, 1, 4, 3),
        }
        cold = {
            "presence": torch.full((1, 1, 4), 0.1),
            "row_mu": torch.tensor([[0.4]]),
            "height_logits": torch.zeros(1, 1, 4, 3),
        }
        matched["height_logits"][..., 0] = 10.0
        cold["height_logits"][..., 0] = 10.0
        warm_loss, _ = intensity_goal_loss(matched, labels, mask, ste=False)
        cold_loss, _ = intensity_goal_loss(cold, labels, mask, ste=False)
        self.assertGreater(float(cold_loss), float(warm_loss))


if __name__ == "__main__":
    unittest.main()
