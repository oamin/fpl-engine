"""Backfill of a public FPL entry from week 0."""

from __future__ import annotations

import unittest

from src.live.entry import (
    build_entry,
    chip_name,
    chips_remaining,
    roll_free_transfers,
    stamp_matchday_teams,
)


def _pick(element: int, position: int, captain: bool = False, vice: bool = False) -> dict:
    return {
        "element": element,
        "position": position,
        "is_captain": captain,
        "is_vice_captain": vice,
        "multiplier": 2 if captain else 1,
    }


class RollTest(unittest.TestCase):
    def test_gw1_awards_one_transfer_and_this_season_ends_on_one(self) -> None:
        weeks = [
            {"event": 1, "transfers": 0, "chip": "triple_captain"},
            {"event": 2, "transfers": 1, "chip": None},
            {"event": 3, "transfers": 0, "chip": None},
            {"event": 4, "transfers": 1, "chip": None},
            {"event": 5, "transfers": 2, "chip": None},
        ]
        available, nxt = roll_free_transfers(weeks)
        self.assertEqual(available, {1: 0, 2: 1, 3: 1, 4: 2, 5: 2})
        self.assertEqual(nxt, 1)

    def test_triple_captain_in_gw1_leaves_the_other_three(self) -> None:
        left = chips_remaining({1: "triple_captain"}, 6)
        self.assertEqual(left, ("wildcard", "free_hit", "bench_boost"))

    def test_chip_code(self) -> None:
        self.assertEqual(chip_name("3xc"), "triple_captain")
        self.assertIsNone(chip_name(None))


class BuildTest(unittest.TestCase):
    def test_opening_squad_is_the_gw1_fifteen(self) -> None:
        names = {
            1: {"web_name": "Keeper", "position": "GKP", "team": "CHE"},
            2: {"web_name": "Captain", "position": "FWD", "team": "MCI"},
            3: {"web_name": "Vice", "position": "MID", "team": "MUN"},
        }
        picks = {
            1: {
                "active_chip": "3xc",
                "picks": [
                    _pick(1, 1),
                    _pick(3, 2, vice=True),
                    _pick(2, 3, captain=True),
                ],
            }
        }
        payload = build_entry(
            {"id": 5, "name": "Test", "summary_overall_points": 62, "summary_overall_rank": 1},
            {
                "current": [
                    {
                        "event": 1,
                        "points": 62,
                        "total_points": 62,
                        "event_transfers": 0,
                        "event_transfers_cost": 0,
                        "points_on_bench": 0,
                        "bank": 0,
                        "value": 1000,
                    }
                ],
                "chips": [{"name": "3xc", "event": 1}],
            },
            [],
            picks,
            names,
        )
        self.assertEqual(payload["opening_squad"][0]["name"], "Keeper")
        self.assertEqual(payload["gameweeks"][0]["captain"], "Captain")
        self.assertEqual(payload["ft_for_next"], 1)
        self.assertEqual(payload["chips_played"], [{"gw": 1, "chip": "triple_captain"}])

    def test_matchday_club_replaces_the_current_club(self) -> None:
        payload = {
            "gameweeks": [
                {
                    "gw": 1,
                    "xi": [{"id": 28, "name": "Martinez", "position": "GKP", "team": "CHE", "slot": 1}],
                    "bench": [],
                },
                {
                    "gw": 2,
                    "xi": [{"id": 28, "name": "Martinez", "position": "GKP", "team": "CHE", "slot": 1}],
                    "bench": [],
                },
            ],
            "opening_squad": [
                {"id": 28, "name": "Martinez", "position": "GKP", "team": "CHE", "slot": 1}
            ],
        }
        stamped = stamp_matchday_teams(payload, {(28, 1): "AVL"})
        self.assertEqual(stamped["gameweeks"][0]["xi"][0]["team"], "AVL")
        self.assertEqual(stamped["gameweeks"][1]["xi"][0]["team"], "CHE")
        self.assertEqual(stamped["opening_squad"][0]["team"], "AVL")
        self.assertEqual(payload["gameweeks"][0]["xi"][0]["team"], "CHE")


if __name__ == "__main__":
    unittest.main()
