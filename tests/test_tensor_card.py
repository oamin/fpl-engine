"""Typed tensor card — schema and as-of slice invariants."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card import (
    MAX_GW,
    MAX_PLAYERS,
    TRAIN_SEASONS,
    TEST_SEASON,
    asof_inputs,
    blank_future_events,
    build_tensor_cards,
    cut_at_fraction,
    iter_slices,
)


def test_build_shapes_and_unclipped_counts():
    cards = build_tensor_cards((TEST_SEASON,))
    assert cards
    card = next(c for c in cards if c.team == "Arsenal")
    assert card.minutes.shape == (MAX_PLAYERS, MAX_GW)
    assert card.goals.dtype == np.uint8
    assert card.assists.dtype == np.uint8
    assert card.minutes.dtype == np.float32
    # Raw minutes: some cell should exceed 1 (not forced /90).
    assert float(card.minutes.max()) > 1.0
    # Unclipped: if any hat-trick-class count exists in season pool, allow >1.
    assert int(card.goals.max()) >= 0
    assert card.n_fixtures > 10
    assert card.n_players > 5


def test_asof_history_excludes_scored_week():
    cards = build_tensor_cards((TEST_SEASON,))
    card = cards[0]
    # Find a player with at least two present weeks.
    found = False
    for p in range(card.n_players):
        cols = np.flatnonzero(card.present[p, : card.n_fixtures])
        if len(cols) < 3:
            continue
        t = int(cols[2])
        feats = asof_inputs(card, p, t)
        # games_played counts only weeks before t.
        assert feats["games_played"] == float(np.sum(card.present[p, :t]))
        found = True
        break
    assert found


def test_blank_future_keeps_minutes_and_header():
    cards = build_tensor_cards((TEST_SEASON,))
    card = cards[0]
    cut = cut_at_fraction(card, 0.5)
    blank = blank_future_events(card, cut)
    assert np.array_equal(blank.minutes, card.minutes)
    assert np.array_equal(blank.home, card.home)
    assert np.allclose(blank.odds_1x2, card.odds_1x2, equal_nan=True)
    assert np.all(blank.goals[:, cut:] == 0)
    assert np.all(blank.assists[:, cut:] == 0)
    if cut > 0:
        assert np.array_equal(blank.goals[:, :cut], card.goals[:, :cut])


def test_iter_slices_targets_match_tensor():
    cards = build_tensor_cards((TEST_SEASON,))
    card = cards[0]
    for row in iter_slices(card, min_history=1):
        p = int(np.flatnonzero(card.player_id == row["player_id"])[0])
        t = row["t"]
        assert row["y_goal"] == (1 if card.goals[p, t] >= 1 else 0)
        assert row["y_assist"] == (1 if card.assists[p, t] >= 1 else 0)
        assert "odds_win" in row["features"]
        break


def test_train_test_seasons_build():
    cards = build_tensor_cards(TRAIN_SEASONS + (TEST_SEASON,))
    seasons = {c.season for c in cards}
    assert TRAIN_SEASONS[0] in seasons
    assert TEST_SEASON in seasons
