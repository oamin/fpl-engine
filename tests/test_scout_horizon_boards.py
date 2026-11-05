"""Scout minutes overlay on L4+Opta horizon boards."""

from __future__ import annotations

import pandas as pd

from src.live.scout_horizon_boards import overlay_minutes
from src.live.scout_minutes import ScoutMinutesPacket, ScoutOverlay


def test_overlay_minutes_uses_step_map() -> None:
    side = pd.DataFrame(
        {
            "element": [1, 2],
            "m": [80.0, 70.0],
            "team": ["A", "B"],
        }
    )
    out = overlay_minutes(side, {1: 0.0, 2: 15.0})
    assert float(out.loc[out["element"] == 1, "m"].iloc[0]) == 0.0
    assert float(out.loc[out["element"] == 2, "m"].iloc[0]) == 15.0


def test_packet_covers_centreline_horizon() -> None:
    ov = ScoutOverlay(
        element=200,
        name="Lacroix",
        club="Chelsea",
        tag="injured",
        m0=0.0,
        reason="out",
        context="injured_expected_back",
        return_gws=2,
        force_steps={0: 0.0, 1: 0.0},
    )
    p = ScoutMinutesPacket(
        season="2026-27",
        decision_gw=9,
        horizon=12,
        source="test",
        fetched_at="t",
        baseline_m={200: 79.0},
        overlays={200: ov},
    )
    path = [p.m_at(200, h) for h in range(12)]
    assert path[:2] == [0.0, 0.0]
    assert path[2:] == [79.0] * 10
