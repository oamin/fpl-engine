"""Zero-gated SliceLogistic helpers."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card import build_tensor_cards, iter_slices
from src.models.tensor_card_zero_gate import (
    P_FLOOR,
    _fit_logistic,
    _predict_gated,
    _predict_ungated,
    assert_mates_off,
)


def test_zero_gate_floors_blank_minutes():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=(n, 5))
    # Active rows only for fit
    y = (rng.random(n) < 0.1).astype(int)
    model = _fit_logistic(x, y)
    minutes = np.array([0.0, 0.0, 90.0, 45.0])
    x_te = rng.normal(size=(4, 5))
    p = _predict_gated(model, x_te, minutes)
    assert p[0] == P_FLOOR and p[1] == P_FLOOR
    assert p[2] > P_FLOOR and p[3] > P_FLOOR


def test_ungated_can_leak_on_zero_minutes():
    rng = np.random.default_rng(1)
    n = 300
    minutes = rng.choice([0.0, 90.0], size=n, p=[0.4, 0.6])
    # Feature correlated with minutes so ungated learns a minutes slope.
    x = np.column_stack([minutes, rng.normal(size=n)])
    y = ((minutes > 0) & (rng.random(n) < 0.12)).astype(int)
    model = _fit_logistic(x, y)
    x0 = np.array([[0.0, 0.0]])
    p0 = _predict_ungated(model, x0)[0]
    # Ungated typically assigns small but non-floor mass at m=0.
    assert p0 >= P_FLOOR


def test_mates_off_on_slices():
    cards = build_tensor_cards(("2025-26",))
    rows = []
    for card in cards[:2]:
        rows.extend(
            list(
                iter_slices(
                    card,
                    min_history=1,
                    minutes_mode="actual",
                    include_mates=False,
                )
            )
        )
    assert rows
    assert_mates_off(rows)
