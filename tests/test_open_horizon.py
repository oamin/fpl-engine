"""Opening-price horizon: future weeks are not a copy of this week's score."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.open_horizon import (
    opening_pots_by_team_gw,
    opening_pots_for_sheet,
    prior_pots,
    project_player,
    side_pot,
    single_fixture_calendar,
    xp_on_pot,
)
from src.models.season_climb_ft import transfer_value


class OpenHorizonTests(unittest.TestCase):
    def test_opening_file_joins_a_fixture_on_kickoff_date(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "Date": "12/09/2026",
                    "HomeTeam": "Chelsea",
                    "AwayTeam": "Hull",
                    "AvgH": 1.21,
                    "AvgD": 6.50,
                    "AvgA": 11.92,
                    "Avg>2.5": 1.36,
                    "Avg<2.5": 3.10,
                }
            ]
        )
        fixtures = [
            {
                "event": 4,
                "kickoff_time": "2026-09-12T14:00:00Z",
                "team_h": 6,
                "team_a": 11,
            }
        ]
        pots = opening_pots_by_team_gw(odds, fixtures, {6: "Chelsea", 11: "Hull City"})
        self.assertGreater(pots[(4, "chelsea")][0]["lam_scored"], 2.2)
        self.assertLess(pots[(4, "hull")][0]["lam_scored"], 1.0)

    def test_hull_open_is_a_different_rate_from_arsenal(self) -> None:
        arsenal = side_pot(1.69, 3.80, 4.82, 1.73, 2.10, is_home=False)
        hull = side_pot(1.21, 6.50, 11.92, 1.36, 3.10, is_home=True)
        self.assertLess(arsenal["lam_scored"], 1.0)
        self.assertGreater(hull["lam_scored"], 2.2)

    def test_frozen_share_on_hull_is_a_start_and_arsenal_is_not(self) -> None:
        arsenal = side_pot(1.69, 3.80, 4.82, 1.73, 2.10, is_home=False)
        hull = side_pot(1.21, 6.50, 11.92, 1.36, 3.10, is_home=True)
        common = dict(
            position="FWD",
            xmi=90.0,
            share_xg=0.251,
            share_xa=0.05,
            exp_defcon_hit=0.0,
            fwd_goal_scale=1.0,
        )
        hard = xp_on_pot(**common, pot=arsenal)
        easy = xp_on_pot(**common, pot=hull)
        self.assertLess(hard, 3.5)
        self.assertGreater(easy, 4.5)

    def test_missing_price_uses_the_earlier_rate_not_zero(self) -> None:
        history = pd.DataFrame(
            {
                "gw": [1, 2],
                "team_norm": ["chelsea", "chelsea"],
                "lam_scored": [1.5, 1.7],
                "lam_assist": [1.1, 1.3],
                "e_total": [2.6, 2.8],
                "p_cs_mkt": [0.3, 0.28],
            }
        )
        priors = prior_pots(history, before_gw=3)
        row = {
            "team_norm": "chelsea",
            "position": "FWD",
            "xmi": 90.0,
            "share_xG": 0.25,
            "share_xA": 0.05,
            "exp_defcon_hit": 0.0,
            "fwd_goal_scale": 1.0,
            "score_xp": 2.9,
        }
        scored = project_player(
            row,
            4,
            pots={},
            priors=priors,
            calendar={(4, "chelsea"): 1},
        )
        self.assertGreater(scored, 3.5)
        blank = project_player(row, 4, pots={}, priors=priors, calendar={})
        self.assertEqual(blank, 0.0)

    def test_a_blank_this_week_does_not_zero_the_next_fixture(self) -> None:
        history = pd.DataFrame(
            {
                "gw": [1, 2],
                "team_norm": ["chelsea", "chelsea"],
                "lam_scored": [1.5, 1.7],
                "lam_assist": [1.1, 1.3],
                "e_total": [2.6, 2.8],
                "p_cs_mkt": [0.3, 0.28],
            }
        )
        priors = prior_pots(history, before_gw=29)
        row = {
            "team_norm": "chelsea",
            "position": "FWD",
            "xmi": 90.0,
            "share_xG": 0.25,
            "share_xA": 0.05,
            "exp_defcon_hit": 0.0,
            "fwd_goal_scale": 1.0,
            "score_xp": 0.0,
            "fixture_tag": "no_fixture",
        }
        nxt = project_player(
            row, 30, pots={}, priors=priors, calendar={(30, "chelsea"): 1}
        )
        self.assertGreater(nxt, 3.0)
        still_blank = project_player(row, 30, pots={}, priors=priors, calendar={})
        self.assertEqual(still_blank, 0.0)

    def test_a_double_on_the_sheet_is_still_one_fixture(self) -> None:
        roster = pd.DataFrame(
            {
                "gw": [29, 29],
                "team": ["Arsenal", "Arsenal"],
                "player_id": ["a", "b"],
            }
        )
        calendar = single_fixture_calendar(roster)
        self.assertEqual(calendar[(29, "arsenal")], 1)

    def test_opening_prices_join_the_sheet_kickoff(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "Date": "12/09/2026",
                    "HomeTeam": "Chelsea",
                    "AwayTeam": "Hull",
                    "AvgH": 1.21,
                    "AvgD": 6.50,
                    "AvgA": 11.92,
                    "Avg>2.5": 1.36,
                    "Avg<2.5": 3.10,
                }
            ]
        )
        sheet = pd.DataFrame(
            [
                {
                    "gw": 4,
                    "kickoff_time": "2026-09-12T14:00:00Z",
                    "team": "Chelsea",
                    "was_home": True,
                },
                {
                    "gw": 4,
                    "kickoff_time": "2026-09-12T14:00:00Z",
                    "team": "Hull City",
                    "was_home": False,
                },
            ]
        )
        pots = opening_pots_for_sheet(odds, sheet)
        self.assertGreater(pots[(4, "chelsea")][0]["lam_scored"], 2.2)
        self.assertLess(pots[(4, "hull")][0]["lam_scored"], 1.0)

    def test_two_matches_at_one_kickoff_both_keep_a_price(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "Date": "12/08/2023",
                    "HomeTeam": "Chelsea",
                    "AwayTeam": "Luton",
                    "AvgH": 1.30,
                    "AvgD": 5.5,
                    "AvgA": 9.0,
                    "Avg>2.5": 1.5,
                    "Avg<2.5": 2.5,
                },
                {
                    "Date": "12/08/2023",
                    "HomeTeam": "Brighton",
                    "AwayTeam": "Burnley",
                    "AvgH": 1.60,
                    "AvgD": 4.0,
                    "AvgA": 5.5,
                    "Avg>2.5": 1.7,
                    "Avg<2.5": 2.1,
                },
            ]
        )
        kick = "2023-08-12T14:00:00Z"
        sheet = pd.DataFrame(
            [
                {"fixture": 1, "gw": 1, "kickoff_time": kick, "team": "Chelsea", "was_home": True},
                {"fixture": 1, "gw": 1, "kickoff_time": kick, "team": "Luton", "was_home": False},
                {"fixture": 2, "gw": 1, "kickoff_time": kick, "team": "Brighton", "was_home": True},
                {"fixture": 2, "gw": 1, "kickoff_time": kick, "team": "Burnley", "was_home": False},
            ]
        )
        pots = opening_pots_for_sheet(odds, sheet)
        self.assertIn((1, "chelsea"), pots)
        self.assertIn((1, "brighton"), pots)
        self.assertIn((1, "luton"), pots)
        self.assertIn((1, "burnley"), pots)

    def test_step_scores_replace_the_frozen_week(self) -> None:
        spec = (
            [("g1", "GKP"), ("g2", "GKP")]
            + [(f"d{i}", "DEF") for i in range(5)]
            + [(f"m{i}", "MID") for i in range(5)]
            + [("f1", "FWD"), ("f2", "FWD"), ("f3", "FWD")]
        )
        ids = {pid for pid, _pos in spec}
        meta = {pid: {"position": pos, "team_norm": "a"} for pid, pos in spec}
        now = {pid: 2.0 for pid in ids}
        later = {pid: 2.0 for pid in ids}
        later["f1"] = 9.0
        frozen = transfer_value(
            ids, now, meta, 0, 3, [3, 4], {3: ids, 4: ids}, "score_xp", horizon=2
        )
        opened = transfer_value(
            ids,
            now,
            meta,
            0,
            3,
            [3, 4],
            {3: ids, 4: ids},
            "score_xp",
            horizon=2,
            score_by_gw={3: now, 4: later},
        )
        self.assertGreater(opened, frozen)


if __name__ == "__main__":
    unittest.main()
