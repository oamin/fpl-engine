"""Shared FFScout player_pool for cascade_short and cascade_long."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models import cascade_long, cascade_short
from src.models.player_pool import filter_board_to_pool, load_player_pool_ids
from src.models.short_horizon_cascade import attach_prices_and_cap


class PlayerPoolLoad(unittest.TestCase):
    def test_snapshot_loads(self) -> None:
        ids = load_player_pool_ids(missing_ok=False)
        self.assertGreater(len(ids), 50)
        # Bare element ids
        sample = next(iter(ids))
        self.assertTrue(sample.isdigit() or ":" not in sample or sample.split(":")[-1].isdigit())


class FilterBoard(unittest.TestCase):
    def test_keeps_pool_and_force(self) -> None:
        board = pd.DataFrame(
            {
                "element": [1, 2, 3],
                "player_id": ["2026-27:1", "2026-27:2", "2026-27:3"],
                "score_xp_v2": [1.0, 2.0, 3.0],
                "mu_post": [1.0, 2.0, 3.0],
                "position": ["MID", "MID", "FWD"],
                "club": ["a", "b", "c"],
                "value": [50, 55, 60],
            }
        )
        out = filter_board_to_pool(board, {"1", "9"}, force_ids=["2026-27:3"])
        kept = set(out["element"].astype(int))
        self.assertEqual(kept, {1, 3})


class CascadeBranchSurfaces(unittest.TestCase):
    def test_short_long_share_scorer_and_pool(self) -> None:
        self.assertEqual(cascade_short.SCORER, cascade_long.SCORER)
        self.assertEqual(cascade_short.BRANCH, "cascade_short")
        self.assertEqual(cascade_long.BRANCH, "cascade_long")
        self.assertEqual(cascade_short.POOL_JSON, cascade_long.POOL_JSON)


class AttachUsesPool(unittest.TestCase):
    def test_attach_filters_to_pool_when_ids_given(self) -> None:
        board = pd.DataFrame(
            {
                "element": [1, 2],
                "name": ["a", "b"],
                "position": ["MID", "FWD"],
                "team": ["Arsenal", "Chelsea"],
                "score_xp_v2": [4.0, 5.0],
                "xp_se": [0.1, 0.2],
            }
        )
        prices = pd.DataFrame(
            {
                "element": [1, 2],
                "gw": [10, 10],
                "value": [50, 60],
                "team_norm": ["arsenal", "chelsea"],
                "position": ["MID", "FWD"],
                "player_id": ["2026-27:1", "2026-27:2"],
            }
        )
        # Avoid private attach path needing full L4 helpers — only pool filter path.
        # attach_prices_and_cap merges prices; restrict to element 1.
        out = attach_prices_and_cap(
            board,
            prices,
            season="2026-27",
            decision_gw=10,
            selectable_ids={"1"},
            apply_player_pool=True,
        )
        self.assertEqual(set(out["element"].astype(int)), {1})


if __name__ == "__main__":
    unittest.main()
