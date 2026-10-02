"""Chip squad rebuild. No season climb."""

from __future__ import annotations

import unittest

from src.models.season_climb_ft import SquadState, rebuild_squad


def _row(pid: str, position: str, club: str, value: int, score: float) -> dict:
    return {
        "player_id": pid,
        "position": position,
        "team_norm": club,
        "value": value,
        "score": score,
        "eligible": True,
        "minutes": 90,
        "total_points": score,
    }


def _fifteen() -> list[dict]:
    rows = []
    positions = (
        ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    )
    clubs = [
        "ars", "che", "liv", "mci", "tot", "new", "avl",
        "bha", "ful", "cry", "wol", "eve", "bou", "bre", "nfo",
    ]
    for i, (pos, club) in enumerate(zip(positions, clubs)):
        rows.append(_row(f"p{i}", pos, club, 50, 1.0))
    return rows


class RebuildSquadTest(unittest.TestCase):
    def test_keeps_purchase_price_and_cannot_spend_a_fresh_hundred(self) -> None:
        rows = _fifteen()
        rows.append(_row("star", "FWD", "whu", 200, 100.0))
        import pandas as pd

        pool = pd.DataFrame(rows)
        purchase = {f"p{i}": 50 for i in range(15)}
        state = SquadState(purchase=purchase, bank=0, ft=2)
        rebuilt = rebuild_squad(state, pool, "score")
        self.assertNotIn("star", rebuilt.ids())
        self.assertEqual(rebuilt.purchase["p12"], 50)
        self.assertEqual(rebuilt.bank, 0)
        self.assertEqual(state.purchase["p12"], 50)
        self.assertEqual(state.ft, 2)

    def test_a_price_rise_does_not_rewrite_the_purchase_price(self) -> None:
        import pandas as pd

        pool = pd.DataFrame(_fifteen())
        pool.loc[pool["player_id"] == "p12", "value"] = 70
        state = SquadState(purchase={f"p{i}": 50 for i in range(15)}, bank=5, ft=3)
        rebuilt = rebuild_squad(state, pool, "score")
        self.assertEqual(rebuilt.purchase["p12"], 50)
        self.assertEqual(rebuilt.bank, 5)
        self.assertEqual(set(rebuilt.ids()), set(state.ids()))
