"""Lag-1 minutes helpers on the typed tensor card."""

from __future__ import annotations

from src.models.tensor_card import (
    TEST_SEASON,
    TRAIN_SEASONS,
    asof_inputs,
    build_tensor_cards,
    lag1_minutes,
    role_minutes_prior,
)


def test_lag1_uses_previous_week_not_scored_week():
    cards = build_tensor_cards((TEST_SEASON,))
    card = next(c for c in cards if c.team == "Arsenal")
    prior = role_minutes_prior(cards)
    # Find player with two consecutive present weeks and different minutes if possible.
    found = False
    for p in range(card.n_players):
        for t in range(1, card.n_fixtures):
            if not (card.present[p, t] and card.present[p, t - 1]):
                continue
            pred = lag1_minutes(card, p, t, role_prior=prior)
            assert pred == float(card.minutes[p, t - 1])
            feats = asof_inputs(card, p, t, minutes_mode="lag1", role_prior=prior)
            assert feats["minutes"] == pred
            assert feats["mate_minutes"] == 0.0
            assert feats["mate_n"] == 0.0
            actual = asof_inputs(card, p, t, minutes_mode="actual")
            assert actual["minutes"] == float(card.minutes[p, t])
            found = True
            break
        if found:
            break
    assert found


def test_role_prior_from_train():
    cards = build_tensor_cards((TRAIN_SEASONS[0],))
    prior = role_minutes_prior(cards)
    assert set(prior) == {0, 1, 2, 3}
    assert all(v >= 0 for v in prior.values())
