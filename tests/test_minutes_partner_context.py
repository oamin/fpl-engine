"""Partner anti-corr blurbs — context only, no minutes rating."""

from __future__ import annotations

import pandas as pd

from src.live.minutes_partner_context import (
    attach_partner_blurbs_to_overlays,
    partner_blurbs_as_of,
)
from src.live.scout_minutes import ScoutOverlay


def _toy_hist() -> pd.DataFrame:
    # Two MIDs on same team: perfect anti-corr across 6 GWs.
    rows = []
    for gw in range(1, 7):
        rows.append(
            {
                "gw": gw,
                "player_id": 1,
                "player_name": "Alpha",
                "team": "Toy FC",
                "position": "MID",
                "minutes": 90.0 if gw % 2 == 1 else 0.0,
            }
        )
        rows.append(
            {
                "gw": gw,
                "player_id": 2,
                "player_name": "Beta",
                "team": "Toy FC",
                "position": "MID",
                "minutes": 0.0 if gw % 2 == 1 else 90.0,
            }
        )
    return pd.DataFrame(rows)


def test_partner_blurbs_detect_anticorr() -> None:
    partners = partner_blurbs_as_of(_toy_hist(), decision_gw=7, min_prior_gws=5)
    assert 1 in partners and 2 in partners
    assert partners[1].anti_name == "Beta"
    assert partners[1].anti_corr is not None and partners[1].anti_corr <= -0.35
    assert "anti-correlated with Beta" in partners[1].blurb


def test_attach_does_not_change_m0() -> None:
    ov = ScoutOverlay(
        element=1,
        name="Alpha",
        club="Toy FC",
        tag="injured",
        m0=0.0,
        reason="test",
        context_label="injured, unknown return",
    )
    overlays = {1: ov}
    diag = attach_partner_blurbs_to_overlays(
        overlays, _toy_hist(), decision_gw=7, min_prior_gws=5
    )
    assert ov.m0 == 0.0
    assert ov.force_steps == {}
    assert diag["applies_minutes_rating"] is False
    assert ov.anti_name == "Beta"
    assert "Beta" in ov.context_label
