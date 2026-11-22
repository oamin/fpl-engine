"""Pre-match minutes head helpers."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card import build_tensor_cards, lag1_minutes, role_minutes_prior
from src.models.tensor_card_minutes_head import (
    MINUTES_FEATURES,
    event_features_for_minutes,
    minutes_asof_features,
    rich_extras_for_minutes,
)


def test_minutes_asof_features_are_finite_and_complete():
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    prior = role_minutes_prior(cards)
    feats = minutes_asof_features(card, 0, 10, prior)
    assert set(feats) == set(MINUTES_FEATURES)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["lag1_minutes"] == lag1_minutes(card, 0, 10, role_prior=prior)


def test_event_features_respect_substituted_minutes():
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    prior = role_minutes_prior(cards)
    extras = rich_extras_for_minutes(card, 0, 10, over25=0.55, minutes=0.0)
    assert extras["log1p_minutes"] == 0.0
    assert extras["minutes_ge60"] == 0.0
    feats = event_features_for_minutes(card, 0, 10, 0.55, minutes=75.0, role_prior=prior)
    assert feats["minutes"] == 75.0
    assert feats["minutes_ge60"] == 1.0
    assert abs(feats["log1p_minutes"] - float(np.log1p(75.0))) < 1e-12
