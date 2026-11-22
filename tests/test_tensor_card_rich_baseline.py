"""Rich zero-gated baseline helpers."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card import build_tensor_cards
from src.models.tensor_card_rich_baseline import (
    RICH_EXTRA,
    _m0_rates,
    _team_asof_rates,
    load_over25_table,
    rich_extras,
)


def test_over25_table_loads():
    table = load_over25_table(("2025-26",))
    assert len(table) > 100
    vals = list(table.values())
    assert all(0.0 < v < 1.0 for v in vals[:20])


def test_team_asof_and_m0_are_nonnegative():
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    g, a = _team_asof_rates(card, 10)
    assert g >= 0.0 and a >= 0.0
    rg, ra = _m0_rates(card, 0, 10)
    assert rg >= 0.0 and ra >= 0.0
    extras = rich_extras(card, 0, 10, over25=0.55)
    assert set(extras) == set(RICH_EXTRA)
    assert extras["p_over25"] == 0.55
    assert abs(extras["log1p_minutes"] - float(np.log1p(float(card.minutes[0, 10])))) < 1e-9
