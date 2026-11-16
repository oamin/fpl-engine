"""Appearance-clean history helpers."""

from __future__ import annotations

from src.models.tensor_card import asof_inputs, build_tensor_cards


def test_appearance_history_excludes_zero_minute_weeks():
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    found = False
    for p in range(card.n_players):
        for t in range(5, card.n_fixtures):
            if not card.present[p, t]:
                continue
            present_hist = card.present[p, :t].sum()
            appear_hist = ((card.present[p, :t]) & (card.minutes[p, :t] > 0)).sum()
            if present_hist <= appear_hist:
                continue
            a = asof_inputs(card, p, t, include_mates=False, history_mode="appearance")
            b = asof_inputs(card, p, t, include_mates=False, history_mode="present")
            assert a["games_played"] == float(appear_hist)
            assert b["games_played"] == float(present_hist)
            assert a["games_played"] < b["games_played"]
            # Appearance means should not be diluted by zero-minute blanks.
            # mean_minutes on appearance history is over played weeks only.
            if a["games_played"] > 0:
                assert a["mean_minutes"] > 0.0
            found = True
            break
        if found:
            break
    assert found
