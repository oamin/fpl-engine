"""A club with no fixture ranks at 0. An empty week is not played."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.blank_context import (
    apply_fixture_tags,
    clubs_by_gw,
    horizon_sheet,
    playable_gws,
)
from src.models.season_climb import pick_xi


def _player(pid: str, pos: str, team: str, score: float, priority: float) -> dict:
    return {
        "player_id": pid,
        "position": pos,
        "team": team,
        "team_norm": team,
        "score_xp": score,
        "value": 50,
        "xi_priority": priority,
        "minutes": 90,
        "total_points": 2,
        "eligible": True,
    }


class BlankContextTests(unittest.TestCase):
    def test_a_missing_club_is_no_fixture_and_the_price_stays(self) -> None:
        roster = pd.DataFrame(
            [
                {"gw": 29, "team": "Fulham", "player_id": "a"},
                {"gw": 29, "team": "Brentford", "player_id": "b"},
            ]
        )
        clubs = clubs_by_gw(roster)
        pool = pd.DataFrame(
            [
                {
                    "player_id": "ars",
                    "team": "Arsenal",
                    "position": "FWD",
                    "score_xp": 6.0,
                    "exp_points": 5.5,
                    "value": 140,
                },
                {
                    "player_id": "ful",
                    "team": "Fulham",
                    "position": "MID",
                    "score_xp": 4.0,
                    "exp_points": 4.0,
                    "value": 55,
                },
            ]
        )
        tagged = apply_fixture_tags(pool, 29, clubs)
        arsenal = tagged.loc[tagged["player_id"] == "ars"].iloc[0]
        fulham = tagged.loc[tagged["player_id"] == "ful"].iloc[0]
        self.assertEqual(arsenal["fixture_tag"], "no_fixture")
        self.assertEqual(float(arsenal["score_xp"]), 0.0)
        self.assertEqual(float(arsenal["exp_points"]), 0.0)
        self.assertEqual(int(arsenal["value"]), 140)
        self.assertEqual(fulham["fixture_tag"], "fixture")
        self.assertEqual(float(fulham["score_xp"]), 4.0)

    def test_a_benched_player_on_a_playing_club_keeps_his_score(self) -> None:
        roster = pd.DataFrame([{"gw": 10, "team": "Arsenal", "player_id": "a"}])
        pool = pd.DataFrame(
            [
                {
                    "player_id": "a",
                    "team": "Arsenal",
                    "position": "MID",
                    "score_xp": 6.0,
                    "value": 80,
                    "minutes": 0,
                }
            ]
        )
        tagged = apply_fixture_tags(pool, 10, clubs_by_gw(roster))
        self.assertEqual(tagged.iloc[0]["fixture_tag"], "fixture")
        self.assertEqual(float(tagged.iloc[0]["score_xp"]), 6.0)

    def test_an_empty_week_is_not_played(self) -> None:
        clubs = {6: {"arsenal"}, 8: {"arsenal"}}
        self.assertEqual(playable_gws([6, 7, 8], clubs), [6, 8])

    def test_a_fixture_player_wins_a_tie_with_a_blank_club(self) -> None:
        rows = [
            _player("gk_blank", "GKP", "Arsenal", 0.0, 0.0),
            _player("gk_play", "GKP", "Fulham", 0.0, 1.0),
        ]
        for i in range(5):
            rows.append(_player(f"d{i}", "DEF", "Fulham", 3.0, 1.0))
        for i in range(5):
            rows.append(_player(f"m{i}", "MID", "Fulham", 3.0, 1.0))
        for i in range(3):
            rows.append(_player(f"f{i}", "FWD", "Fulham", 3.0, 1.0))
        xi, _form = pick_xi(pd.DataFrame(rows), "score_xp", priority_col="xi_priority")
        self.assertIn("gk_play", set(xi["player_id"]))
        self.assertNotIn("gk_blank", set(xi["player_id"]))

    def test_the_hold_record_names_the_blank_week(self) -> None:
        clubs = {
            27: {"arsenal", "fulham"},
            28: {"arsenal", "fulham"},
            29: {"fulham"},
        }
        ids = {f"p{i}" for i in range(15)}
        # 2 GKP, 5 DEF, 5 MID, 3 FWD. First arsenal forward blanks in 29.
        positions = (
            ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        )
        meta = {}
        scores = {}
        for i, pid in enumerate(sorted(ids)):
            team = "arsenal" if i == 14 else "fulham"
            meta[pid] = {"position": positions[i], "team_norm": team}
            scores[pid] = 6.0 if team == "arsenal" else 4.0
        sheet = horizon_sheet(ids, scores, meta, 27, [28, 29], clubs)
        by_gw = {row["gw"]: row for row in sheet}
        self.assertEqual(by_gw[29]["n_no_fixture"], 0)
        self.assertGreater(by_gw[27]["xi_sum"], 0)
        arsenal_id = next(pid for pid, info in meta.items() if info["team_norm"] == "arsenal")
        self.assertEqual(scores[arsenal_id], 6.0)
        self.assertLess(by_gw[29]["xi_sum"], by_gw[27]["xi_sum"])

    def test_an_all_zero_squad_still_forms_an_xi(self) -> None:
        rows = []
        rows += [_player(f"g{i}", "GKP", "Arsenal", 0.0, 0.0) for i in range(2)]
        rows += [_player(f"d{i}", "DEF", "Arsenal", 0.0, 0.0) for i in range(5)]
        rows += [_player(f"m{i}", "MID", "Arsenal", 0.0, 0.0) for i in range(5)]
        rows += [_player(f"f{i}", "FWD", "Arsenal", 0.0, 0.0) for i in range(3)]
        xi, _form = pick_xi(pd.DataFrame(rows), "score_xp", priority_col="xi_priority")
        self.assertEqual(len(xi), 11)


if __name__ == "__main__":
    unittest.main()
