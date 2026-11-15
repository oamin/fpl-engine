"""Horizon FOV on the typed tensor card."""

from __future__ import annotations

import numpy as np
import torch

from src.models.tensor_card import TEST_SEASON, build_tensor_cards, cut_at_fraction
from src.models.tensor_card_horizon import assert_horizon_fov, horizon_loss, horizon_view
from src.models.tensor_card_stepped import (
    TypedCardMixer,
    _event_ink,
    _pack_card,
    _stack_batch,
    fit_player_codes,
    n_clubs,
)


def test_horizon_fov_blanks_remainder_input():
    cards = build_tensor_cards((TEST_SEASON,))
    card = next(c for c in cards if c.team == "Arsenal")
    assert_horizon_fov(card)
    cut = cut_at_fraction(card, 0.4)
    view = horizon_view(card, cut)
    assert float(view["events_in"][:, cut:, :].sum()) == 0.0
    assert float(view["target"][:, :cut, :].sum()) == 0.0
    assert np.array_equal(view["target"][:, cut:, :], _event_ink(card)[:, cut:, :])


def test_horizon_forward_loss():
    cards = build_tensor_cards((TEST_SEASON,))
    codes = fit_player_codes(cards)
    model = TypedCardMixer(n_club=n_clubs(cards), n_player=max(codes.values()))
    card = cards[0]
    cut = cut_at_fraction(card, 0.5)
    batch = _stack_batch([_pack_card(card, horizon_view(card, cut), codes, torch.device("cpu"))])
    pred = model(
        batch["club_id"],
        batch["opp_id"],
        batch["home"],
        batch["odds"],
        batch["player_code"],
        batch["role"],
        batch["minutes"],
        batch["events_in"],
        batch["row_on"],
        batch["player_pad"],
    )
    loss = horizon_loss(pred, batch["target"], batch["present"], batch["player_pad"])
    assert torch.isfinite(loss)
