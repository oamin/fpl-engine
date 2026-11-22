"""Slice × horizon mix helpers."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card import TEST_SEASON, TRAIN_SEASONS, build_tensor_cards
from src.models.tensor_card_slice_horizon_mix import SliceLogistic, _score_pairs, screen_blends


def test_slice_predict_and_blend():
    cards = build_tensor_cards((TRAIN_SEASONS[0], TEST_SEASON))
    train = [c for c in cards if c.season == TRAIN_SEASONS[0]]
    test = [c for c in cards if c.season == TEST_SEASON]
    model = SliceLogistic.fit(train[:8])
    soft = model.predict_card(test[0])
    assert soft.shape[0] == 40
    assert soft.max() <= 1.0 + 1e-6
    ph = np.full(100, 0.1)
    ps = np.full(100, 0.2)
    y = np.zeros(100)
    y[:5] = 1
    grid = screen_blends(ph, ps, y)
    assert grid["best_alpha"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert _score_pairs(ps, y)["n"] == 100
