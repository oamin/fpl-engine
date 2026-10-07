"""The early-row attachment. A 0-minute player stays out. The cap is 6."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.early_row import BROOKES_GW5, pool_keys
from src.models.squad_frontier import attach_early_players, blank_tag


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["2026-27:1"],
            "position": ["GKP"],
            "team_norm": ["arsenal"],
            "team": ["Arsenal"],
            "value": [45],
            "score_xp": [3.0],
            "eligible": [True],
        }
    )


class AttachTest(unittest.TestCase):
    def test_zero_minutes_stay_out(self) -> None:
        roster = pd.DataFrame(
            {
                "player_id": ["2026-27:2"],
                "gw": [3],
                "position": ["DEF"],
                "team_norm": ["chelsea"],
                "team": ["Chelsea"],
                "value": [40],
                "minutes": [0],
            }
        )
        early = pd.DataFrame({"player_id": ["2026-27:2"], "gw": [3], "score_xp": [5.0]})
        out, attached = attach_early_players(_pool(), roster, early, 3, ["2026-27:2"])
        self.assertEqual(attached, [])
        self.assertNotIn("2026-27:2", set(out["player_id"]))

    def test_a_played_player_is_capped_and_not_buyable(self) -> None:
        roster = pd.DataFrame(
            {
                "player_id": ["2026-27:2"],
                "gw": [3],
                "position": ["GK"],
                "team_norm": ["chelsea"],
                "team": ["Chelsea"],
                "value": [40],
                "minutes": [90],
            }
        )
        early = pd.DataFrame({"player_id": ["2026-27:2"], "gw": [3], "score_xp": [9.2]})
        out, attached = attach_early_players(_pool(), roster, early, 3, ["2026-27:2"])
        self.assertEqual(attached, ["2026-27:2"])
        row = out.loc[out["player_id"] == "2026-27:2"].iloc[0]
        self.assertEqual(float(row["score_xp"]), 6.0)
        self.assertEqual(str(row["position"]), "GKP")
        self.assertFalse(bool(row["eligible"]))

    def test_a_published_score_is_left_alone(self) -> None:
        pool = _pool()
        roster = pd.DataFrame(
            {
                "player_id": ["2026-27:1"],
                "gw": [3],
                "position": ["GKP"],
                "team_norm": ["arsenal"],
                "value": [45],
                "minutes": [90],
            }
        )
        early = pd.DataFrame({"player_id": ["2026-27:1"], "gw": [3], "score_xp": [5.5]})
        out, attached = attach_early_players(pool, roster, early, 3, ["2026-27:1"])
        self.assertEqual(attached, [])
        self.assertEqual(float(out.iloc[0]["score_xp"]), 3.0)
        self.assertTrue(bool(out.iloc[0]["eligible"]))

    def test_a_played_player_with_no_early_score_stays_out(self) -> None:
        roster = pd.DataFrame(
            {
                "player_id": ["2026-27:9"],
                "gw": [3],
                "position": ["MID"],
                "team_norm": ["brighton"],
                "value": [55],
                "minutes": [70],
            }
        )
        early = pd.DataFrame({"player_id": ["2026-27:2"], "gw": [3], "score_xp": [4.0]})
        out, attached = attach_early_players(_pool(), roster, early, 3, ["2026-27:9"])
        self.assertEqual(attached, [])
        self.assertNotIn("2026-27:9", set(out["player_id"]))

    def test_a_missing_score_on_an_existing_row_is_filled(self) -> None:
        pool = _pool()
        extra = pd.DataFrame(
            {
                "player_id": ["2026-27:4"],
                "position": ["DEF"],
                "team_norm": ["everton"],
                "team": ["Everton"],
                "value": [40],
                "score_xp": [float("nan")],
                "eligible": [True],
            }
        )
        pool = pd.concat([pool, extra], ignore_index=True)
        roster = pd.DataFrame(
            {
                "player_id": ["2026-27:4"],
                "gw": [3],
                "position": ["DEF"],
                "team_norm": ["everton"],
                "value": [40],
                "minutes": [90],
            }
        )
        early = pd.DataFrame({"player_id": ["2026-27:4"], "gw": [3], "score_xp": [3.49]})
        out, attached = attach_early_players(pool, roster, early, 3, ["2026-27:4"])
        self.assertEqual(attached, ["2026-27:4"])
        row = out.loc[out["player_id"] == "2026-27:4"].iloc[0]
        self.assertAlmostEqual(float(row["score_xp"]), 3.49)
        self.assertFalse(bool(row["eligible"]))


class BlankTagTest(unittest.TestCase):
    def _sheet(self, minutes: list[int], positions: list[str] | None = None) -> pd.DataFrame:
        positions = positions or ["DEF", "DEF"]
        return pd.DataFrame(
            {
                "player_id": ["2026-27:1", "2026-27:2"],
                "gw": [5, 5],
                "position": positions,
                "team_norm": ["chelsea", "chelsea"],
                "minutes": minutes,
            }
        )

    def test_the_three_sheet_tags(self) -> None:
        clubs = {5: {"chelsea"}}
        self.assertEqual(blank_tag(self._sheet([0, 90]), clubs, 5, "2026-27:1"), "replaced")
        self.assertEqual(blank_tag(self._sheet([0, 0]), clubs, 5, "2026-27:1"), "benched")
        self.assertEqual(blank_tag(self._sheet([0, 90]), {5: {"arsenal"}}, 5, "2026-27:1"), "no_fixture")
        self.assertEqual(blank_tag(self._sheet([90, 0]), clubs, 5, "2026-27:1"), "")

    def test_a_short_appearance_is_unresolved(self) -> None:
        clubs = {5: {"chelsea"}}
        self.assertEqual(blank_tag(self._sheet([0, 45]), clubs, 5, "2026-27:1"), "unresolved")

    def test_a_goalkeeper_label_matches_the_squad_label(self) -> None:
        roster = self._sheet([0, 60], ["GK", "GKP"])
        self.assertEqual(blank_tag(roster, {5: {"chelsea"}}, 5, "2026-27:1"), "replaced")


class PopulationTest(unittest.TestCase):
    def test_nine_pool_weeks_exclude_the_money_week(self) -> None:
        rows = [
            {"entry_id": index, "gw": 3, "chip": "wildcard", "status": "pool"}
            for index in range(1, 10)
        ]
        frame = pd.DataFrame(rows)
        keys = pool_keys(frame)
        self.assertEqual(len(keys), 9)
        self.assertNotIn(BROOKES_GW5, keys)
        money = pd.DataFrame(
            [{"entry_id": 616, "gw": 5, "chip": "wildcard", "status": "unreachable_money"}]
        )
        self.assertEqual(pool_keys(pd.concat([frame, money], ignore_index=True)), keys)
        swapped = frame.copy()
        swapped.loc[0, ["entry_id", "gw", "chip"]] = [616, 5, "wildcard"]
        with self.assertRaises(RuntimeError):
            pool_keys(swapped)
