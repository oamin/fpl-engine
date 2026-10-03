"""Scheduled minutes keep a bench week, and a stub cannot see the future."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.sched_minutes import per_fixture_minutes, score_xp_sched, xmi_sched_for
from src.models.season_climb_ft import _gw_pool


class SchedMinutesTests(unittest.TestCase):
    def test_a_blank_week_lowers_the_minutes_prior(self) -> None:
        raw = pd.DataFrame(
            [
                {"element": 1, "GW": 1, "minutes": 90, "team": "Chelsea", "fixture": 1},
                {"element": 1, "GW": 2, "minutes": 90, "team": "Chelsea", "fixture": 2},
                {"element": 1, "GW": 3, "minutes": 0, "team": "Chelsea", "fixture": 3},
                {"element": 1, "GW": 4, "minutes": 90, "team": "Chelsea", "fixture": 4},
            ]
        )
        minutes = per_fixture_minutes(raw, "2025-26")
        feat = pd.DataFrame(
            [{"player_id": "2025-26:1", "gw": 4, "team": "Chelsea"}]
        )
        prior = xmi_sched_for(feat, minutes)
        self.assertAlmostEqual(float(prior.iloc[0]), 60.0)

    def test_a_double_is_minutes_per_fixture(self) -> None:
        raw = pd.DataFrame(
            [
                {"element": 1, "GW": 1, "minutes": 90, "team": "Chelsea", "fixture": 1},
                {"element": 1, "GW": 1, "minutes": 0, "team": "Chelsea", "fixture": 2},
                {"element": 1, "GW": 2, "minutes": 90, "team": "Chelsea", "fixture": 3},
            ]
        )
        minutes = per_fixture_minutes(raw, "2025-26")
        feat = pd.DataFrame([{"player_id": "2025-26:1", "gw": 2, "team": "Chelsea"}])
        prior = float(xmi_sched_for(feat, minutes).iloc[0])
        self.assertAlmostEqual(prior, 45.0)

    def test_no_later_week_fills_the_first(self) -> None:
        raw = pd.DataFrame(
            [
                {"element": 1, "GW": 2, "minutes": 90, "team": "Chelsea", "fixture": 2},
            ]
        )
        minutes = per_fixture_minutes(raw, "2025-26")
        feat = pd.DataFrame([{"player_id": "2025-26:1", "gw": 1, "team": "Chelsea"}])
        self.assertTrue(pd.isna(xmi_sched_for(feat, minutes).iloc[0]))

    def test_a_regular_matches_the_undamped_goal_term(self) -> None:
        feat = pd.DataFrame(
            [
                {
                    "position": "FWD",
                    "share_xG": 0.25,
                    "share_xA": 0.0,
                    "lam_scored": 2.0,
                    "lam_assist": 1.5,
                    "p_cs_mkt": 0.2,
                    "exp_defcon_hit": 0.0,
                    "fwd_goal_scale": 1.0,
                    "lam_conceded": 1.0,
                }
            ]
        )
        score = float(score_xp_sched(feat, pd.Series([90.0])).iloc[0])
        # p_play 1, p60 1, appear 2, goals 0.25*2*4 = 2, bps 0.18*2, card 0.15
        self.assertAlmostEqual(score, 2 + 2 + 0.36 - 0.15, places=6)

    def test_stub_does_not_read_a_later_week(self) -> None:
        feat = pd.DataFrame(
            [
                {
                    "gw": 1,
                    "player_id": "a",
                    "player_name": "a",
                    "position": "MID",
                    "team": "a",
                    "team_norm": "a",
                    "value": 50,
                    "eligible": True,
                    "score_xp": 4.0,
                    "total_points": 2.0,
                    "minutes": 90.0,
                },
                {
                    "gw": 4,
                    "player_id": "b",
                    "player_name": "b",
                    "position": "MID",
                    "team": "b",
                    "team_norm": "b",
                    "value": 50,
                    "eligible": True,
                    "score_xp": 9.0,
                    "total_points": 2.0,
                    "minutes": 90.0,
                },
            ]
        )
        roster = pd.DataFrame(
            [
                {
                    "gw": 1,
                    "player_id": "b",
                    "player_name": "b",
                    "position": "MID",
                    "team": "b",
                    "team_norm": "b",
                    "value": 50,
                    "total_points": 0.0,
                    "minutes": 0.0,
                }
            ]
        )
        pool = _gw_pool(feat, roster, 1, {"b"})
        row = pool.loc[pool["player_id"] == "b"].iloc[0]
        self.assertTrue(pd.isna(row["score_xp"]))


if __name__ == "__main__":
    unittest.main()
