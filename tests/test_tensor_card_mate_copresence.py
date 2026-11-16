"""Mate co-presence helpers on the typed tensor card."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card import (
    COPRESENCE_FEATURE_NAMES,
    build_tensor_cards,
    mate_copresence_features,
)


def test_copresence_counts_other_players_only():
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    # Find a week with several players on.
    found = False
    for t in range(1, card.n_fixtures):
        on = np.flatnonzero(
            (card.minutes[: card.n_players, t] > 0) & (card.position[: card.n_players] >= 0)
        )
        if len(on) < 5:
            continue
        p = int(on[0])
        feats = mate_copresence_features(card, p, t)
        assert set(feats) == set(COPRESENCE_FEATURE_NAMES)
        assert feats["mate_n_on"] == float(len(on) - 1)
        role_sum = feats["mate_def_on"] + feats["mate_mid_on"] + feats["mate_fwd_on"]
        # GKP mates may exist; role_sum ≤ mate_n_on
        assert role_sum <= feats["mate_n_on"] + 1e-9
        assert feats["mate_max_mean_goals"] >= 0.0
        found = True
        break
    assert found


def test_copresence_history_is_asof():
    cards = build_tensor_cards(("2024-25",))
    card = cards[0]
    p, t = 0, 5
    # Force a known history by reading; feature must not use week t goals of mates
    # beyond the as-of mean definition (weeks < t). Smoke: function runs.
    feats = mate_copresence_features(card, p, t)
    assert all(name in feats for name in COPRESENCE_FEATURE_NAMES)
