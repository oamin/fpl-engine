"""Smoke tests for next-event sequence dig helpers."""

from __future__ import annotations

import pandas as pd

from src.eval.next_event_sequence_dig import HISTORY_CHANNELS, build_eligible_rows


def test_no_padding_requires_full_k_history() -> None:
    # One player, GWs 1..4 only — with K=3, only GW4 is eligible.
    rows = []
    for gw in range(1, 5):
        rec = {
            "season": "toy",
            "element": 1,
            "GW": gw,
            "name": "P",
            "position": "MID",
            "team": "A",
            "opponent": "B",
            "was_home": True,
            "goals_scored": 0,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 0,
            "bonus": 0,
            "defensive_contribution": 0,
            "y_goal": 0,
            "y_assist": 0,
            "y_appear60": 1,
            "y_defcon": 0,
            "y_bonus": 0,
            "y_cs": 0,
        }
        for ch in HISTORY_CHANNELS:
            rec[ch] = float(gw if ch == "minutes" else 0)
        rows.append(rec)
    panel = pd.DataFrame(rows)
    elig = build_eligible_rows(panel, k=3)
    assert list(elig["GW"]) == [4]
    assert len(elig.iloc[0]["hist"]) == 3 * len(HISTORY_CHANNELS)
