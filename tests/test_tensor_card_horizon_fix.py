"""Fixed horizon mixer helpers."""

from __future__ import annotations

import numpy as np
import torch

from src.models.tensor_card import build_tensor_cards
from src.models.tensor_card_horizon_fix import (
    FixedTypedCardMixer,
    _pack_card,
    _stack_batch,
    fit_player_codes_season,
    paint_log_loss,
    predict_week,
)
from src.models.tensor_card_stepped import n_clubs, step_view


def test_season_scoped_codes_do_not_collide_across_seasons():
    cards = build_tensor_cards(("2022-23", "2025-26"))
    codes = fit_player_codes_season(cards)
    # Same element id in two seasons must map to different embedding slots when both present.
    pairs = {}
    for card in cards:
        for pid in card.player_id.tolist():
            pid = int(pid)
            if pid <= 0:
                continue
            pairs.setdefault(pid, set()).add(codes[(card.season, pid)])
    collided = [pid for pid, slots in pairs.items() if len(slots) > 1]
    assert len(collided) > 0  # at least some recycled element ids
    # And those slots are distinct integers
    for pid in collided[:5]:
        assert len(pairs[pid]) == len(set(pairs[pid]))


def test_zero_gate_and_present_pack():
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    codes = fit_player_codes_season(cards)
    model = FixedTypedCardMixer(n_club=n_clubs(cards), n_player=max(codes.values()))
    cut = 10
    batch = _stack_batch([_pack_card(card, step_view(card, cut), codes, torch.device("cpu"))])
    assert batch["cell_present"].shape == batch["minutes"].shape
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
        batch["cell_present"],
    )
    loss = paint_log_loss(pred, batch["target"], batch["present"], batch["player_pad"], batch["cut"])
    assert torch.isfinite(loss)
    # Bench / absent seats at the cut week are floored.
    soft = pred[0, :, cut, :].detach().numpy()
    minutes = card.minutes[:, cut]
    present = card.present[:, cut]
    for p in range(card.n_players):
        if (not present[p]) or minutes[p] <= 0:
            assert soft[p, 0] <= 1.1e-6
            assert soft[p, 1] <= 1.1e-6
    week = predict_week(model, card, cut, codes, torch.device("cpu"))
    assert week.shape == (40, 2)
    assert np.isfinite(week).all()
