"""Market pot / role× helpers on the rich zero-gated spine."""

from __future__ import annotations

from src.models.tensor_card import build_tensor_cards
from src.models.tensor_card_market_role import (
    MARKET_ROLE_EXTRA,
    ROLE_X,
    _fixture_market,
    load_market_pot_table,
    market_role_extras,
)
from src.models.tensor_card_rich_baseline import rich_extras
from src.teams import norm_team


def test_market_pot_table_loads_and_covers_arsenal():
    table = load_market_pot_table(("2025-26",))
    assert len(table) > 300
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    hit = 0
    for t in range(card.n_fixtures):
        m = _fixture_market(card, t, table)
        assert "ah_line_own" in m and "xg_own" in m
        assert m["xg_own"] > 0.0
        key = (card.season, norm_team(card.team), norm_team(card.opponent[t]))
        if key in table:
            hit += 1
            assert abs(m["xg_own"] - table[key]["xg_own"]) < 1e-12
    assert hit >= 30


def test_market_role_extras_shape_and_role_mask():
    cards = build_tensor_cards(("2025-26",))
    card = next(c for c in cards if c.team == "Arsenal")
    # Find a MID with minutes in an early GW.
    p = None
    t = 5
    for i in range(card.n_players):
        if int(card.position[i]) == 2 and float(card.minutes[i, t]) > 0:
            p = i
            break
    assert p is not None
    feats = {
        "role_gkp": 0.0,
        "role_def": 0.0,
        "role_mid": 1.0,
        "role_fwd": 0.0,
    }
    feats.update(rich_extras(card, p, t, over25=0.55))
    market = {"ah_line_own": -0.75, "xg_own": 1.8}
    extras = market_role_extras(feats, market)
    assert set(extras) == set(MARKET_ROLE_EXTRA)
    assert extras["ah_line_own"] == -0.75
    assert extras["xg_own"] == 1.8
    assert abs(extras["role_mid_x_xg_own"] - 1.8) < 1e-12
    assert extras["role_gkp_x_xg_own"] == 0.0
    assert extras["role_def_x_m0_rG"] == 0.0
    assert len(ROLE_X) == 16
