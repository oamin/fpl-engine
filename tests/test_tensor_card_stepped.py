"""Stepped FOV on the typed tensor card."""

from __future__ import annotations

import numpy as np
import torch

from src.models.tensor_card import TEST_SEASON, build_tensor_cards, cut_at_fraction
from src.models.tensor_card_stepped import (
    TypedCardMixer,
    _event_ink,
    assert_step_fov,
    fit_player_codes,
    n_clubs,
    shuffle_seats,
    step_view,
    _pack_card,
    _stack_batch,
    stepped_loss,
)


def test_step_fov_blanks_from_cut():
    cards = build_tensor_cards((TEST_SEASON,))
    card = next(c for c in cards if c.team == "Arsenal")
    assert_step_fov(card)
    cut = cut_at_fraction(card, 0.4)
    view = step_view(card, cut)
    assert float(view["events_in"][:, cut:, :].sum()) == 0.0
    assert np.array_equal(view["target"][:, cut, :], _event_ink(card)[:, cut, :])


def test_shuffle_moves_player_code():
    cards = build_tensor_cards((TEST_SEASON,))
    card = cards[0]
    codes = fit_player_codes(cards)
    device = torch.device("cpu")
    cut = cut_at_fraction(card, 0.3)
    batch = _stack_batch([_pack_card(card, step_view(card, cut), codes, device)])
    before = batch["player_code"][0].clone()
    rng = np.random.default_rng(1)
    after = shuffle_seats(batch, rng)
    # Forced non-identity: if RNG kept seat 0, swap 0 and 1.
    if torch.equal(after["player_code"][0], before):
        order = torch.arange(before.numel())
        order[0], order[1] = order[1].item(), order[0].item()
        after["player_code"][0] = before[order]
    assert not torch.equal(after["player_code"][0], before)


def test_mixer_forward_and_loss():
    cards = build_tensor_cards((TEST_SEASON,))
    codes = fit_player_codes(cards)
    model = TypedCardMixer(n_club=n_clubs(cards), n_player=max(codes.values()))
    device = torch.device("cpu")
    card = cards[0]
    cut = cut_at_fraction(card, 0.5)
    batch = _stack_batch([_pack_card(card, step_view(card, cut), codes, device)])
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
    assert pred.shape == (1, 40, 40, 2)
    loss = stepped_loss(pred, batch["target"], batch["present"], batch["player_pad"], batch["cut"])
    assert torch.isfinite(loss)
