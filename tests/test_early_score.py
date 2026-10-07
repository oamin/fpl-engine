"""An owned player with a short history keeps a past-only score, capped."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.season_climb_ft import EARLY_SCORE_CAP, _gw_pool, early_score_table


class EarlyScoreTests(unittest.TestCase):
    def test_a_one_match_share_cannot_clear_the_cap(self) -> None:
        scored = pd.DataFrame(
            {
                "player_id": ["2024-25:1"],
                "gw": [2],
                "n_prior": [1],
                "xp": [11.5],
            }
        )
        early = early_score_table(scored)
        self.assertEqual(float(early["score_xp"].iloc[0]), EARLY_SCORE_CAP)

    def test_the_buy_pool_does_not_gain_him(self) -> None:
        feat = pd.DataFrame(
            {
                "player_id": ["2024-25:2"],
                "gw": [2],
                "eligible": [True],
                "position": ["MID"],
                "team": ["a"],
                "team_norm": ["a"],
                "value": [50],
                "score_xp": [4.0],
            }
        )
        roster = pd.DataFrame(
            {
                "player_id": ["2024-25:9", "2024-25:8"],
                "gw": [2, 2],
                "player_name": ["Owned", "Other"],
                "position": ["MID", "MID"],
                "team": ["b", "c"],
                "team_norm": ["b", "c"],
                "value": [55, 40],
                "total_points": [0, 0],
                "minutes": [0, 0],
            }
        )
        early = pd.DataFrame(
            {
                "player_id": ["2024-25:9", "2024-25:8"],
                "gw": [2, 2],
                "score_xp": [11.0, 5.0],
            }
        )
        pool = _gw_pool(feat, roster, 2, {"2024-25:9"}, early)
        ids = set(pool["player_id"].astype(str))
        self.assertIn("2024-25:9", ids)
        self.assertNotIn("2024-25:8", ids)
        owned = pool.loc[pool["player_id"] == "2024-25:9"].iloc[0]
        self.assertFalse(bool(owned["eligible"]))
        self.assertAlmostEqual(float(owned["score_xp"]), 11.0)


if __name__ == "__main__":
    unittest.main()
