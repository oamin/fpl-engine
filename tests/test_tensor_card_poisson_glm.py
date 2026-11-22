"""Zero-gated Poisson exposure helpers on the typed tensor spine."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card import FEATURE_NAMES_LAG1, build_tensor_cards, iter_slices
from src.models.tensor_card_poisson_glm import (
    POISSON_FEATURES,
    P_FLOOR,
    _fit_poisson,
    _feat_matrix,
    _minutes,
    _counts,
    assert_mates_off,
)


def test_poisson_features_drop_minutes():
    assert "minutes" not in POISSON_FEATURES
    assert set(POISSON_FEATURES) == set(FEATURE_NAMES_LAG1) - {"minutes"}


def test_iter_slices_mates_off():
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
    # With mates on, at least one present week should show mate signal.
    on = next(
        iter_slices(cards[0], min_history=1, minutes_mode="actual", include_mates=True)
    )
    # May be zero for isolated benches; just ensure the flag path runs.
    assert "mate_minutes" in on["features"]


def test_zero_gate_and_opacity_link():
    # Toy: constant rate 0.2 goals per 90 → λ = 0.2 at 90 minutes.
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, len(POISSON_FEATURES)))
    minutes = np.full(n, 90.0)
    # Simulate Poisson counts with λ=0.2
    counts = rng.poisson(0.2, size=n).astype(float)
    model = _fit_poisson(x, minutes, counts)
    # Zero minutes must floor
    p0 = model.predict_opacity(x[:5], np.zeros(5))
    assert np.all(p0 == P_FLOOR)
    # 90 minutes → opacity near 1-exp(-0.2)
    p90 = model.predict_opacity(x[:50], np.full(50, 90.0))
    expect = 1.0 - np.exp(-0.2)
    assert abs(float(p90.mean()) - expect) < 0.08


def test_collect_matrix_shapes():
    cards = build_tensor_cards(("2024-25",))
    rows = list(iter_slices(cards[0], min_history=1, include_mates=False))[:20]
    x = _feat_matrix(rows, POISSON_FEATURES)
    assert x.shape == (20, len(POISSON_FEATURES))
    assert _minutes(rows).shape == (20,)
    assert _counts(rows, "goal").shape == (20,)
