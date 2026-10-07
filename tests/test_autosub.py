"""Automatic substitutes when a forward plays zero minutes."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.season_climb import bank_squad_gw


def _player(
    pid: str,
    position: str,
    score: float,
    minutes: float,
    points: float,
) -> dict[str, object]:
    return {
        "player_id": pid,
        "position": position,
        "score_xp": score,
        "minutes": minutes,
        "total_points": points,
    }


def _squad(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class AutosubForwardTest(unittest.TestCase):
    def test_three_forwards_brings_in_the_next_midfielder(self) -> None:
        """A 3-4-3 with one forward on zero minutes becomes 3-5-2.

        The bench is ordered by score. The highest outfield substitute
        played, and two forwards remain, so a midfielder is legal.
        """
        rows = [
            _player("gk", "GKP", 5.0, 90, 2),
            _player("gk2", "GKP", 1.0, 0, 0),
        ]
        rows += [_player(f"d{i}", "DEF", 5.0, 90, 6) for i in range(3)]
        rows += [
            _player("d3", "DEF", 2.0, 90, 1),
            _player("d4", "DEF", 1.0, 90, 1),
        ]
        rows += [_player(f"m{i}", "MID", 5.0, 90, 5) for i in range(4)]
        rows.append(_player("m4", "MID", 3.0, 90, 7))
        rows += [
            _player("jp", "FWD", 5.3, 0, 0),
            _player("f2", "FWD", 5.0, 90, 8),
            _player("f3", "FWD", 5.0, 90, 4),
        ]
        banked = bank_squad_gw(_squad(rows), "score_xp")
        self.assertEqual(banked["form_intended"], (3, 4, 3))
        self.assertEqual(banked["n_autosubs"], 1)
        self.assertEqual(banked["form"], (3, 5, 2))
        self.assertIn("m4", set(banked["xi"]["player_id"].astype(str)))
        self.assertNotIn("jp", set(banked["xi"]["player_id"].astype(str)))
        self.assertEqual(banked["sub_points"], 7.0)

    def test_a_substitute_who_did_not_play_is_skipped(self) -> None:
        rows = [
            _player("gk", "GKP", 5.0, 90, 2),
            _player("gk2", "GKP", 1.0, 0, 0),
        ]
        rows += [_player(f"d{i}", "DEF", 5.0, 90, 6) for i in range(3)]
        rows += [_player("d3", "DEF", 2.0, 90, 4), _player("d4", "DEF", 1.0, 0, 0)]
        rows += [_player(f"m{i}", "MID", 5.0, 90, 5) for i in range(4)]
        rows.append(_player("m4", "MID", 3.0, 0, 0))
        rows += [
            _player("jp", "FWD", 5.3, 0, 0),
            _player("f2", "FWD", 5.0, 90, 8),
            _player("f3", "FWD", 5.0, 90, 4),
        ]
        banked = bank_squad_gw(_squad(rows), "score_xp")
        self.assertEqual(banked["form_intended"], (3, 4, 3))
        self.assertEqual(banked["n_autosubs"], 1)
        self.assertEqual(banked["form"], (4, 4, 2))
        self.assertIn("d3", set(banked["xi"]["player_id"].astype(str)))

    def test_the_only_forward_is_not_replaced_by_a_midfielder(self) -> None:
        rows = [
            _player("gk", "GKP", 5.0, 90, 2),
            _player("gk2", "GKP", 1.0, 0, 0),
        ]
        rows += [_player(f"d{i}", "DEF", 5.0, 90, 6) for i in range(3)]
        rows += [_player("d3", "DEF", 4.0, 90, 2), _player("d4", "DEF", 4.0, 90, 2)]
        rows += [_player(f"m{i}", "MID", 6.0, 90, 5) for i in range(5)]
        rows += [
            _player("jp", "FWD", 5.0, 0, 0),
            _player("f2", "FWD", 0.5, 0, 0),
            _player("f3", "FWD", 0.4, 0, 0),
        ]
        banked = bank_squad_gw(_squad(rows), "score_xp")
        self.assertEqual(banked["form_intended"][2], 1)
        self.assertEqual(banked["n_autosubs"], 0)
        self.assertIn("jp", set(banked["xi"]["player_id"].astype(str)))
        self.assertEqual(banked["n_blank_final"], 1)
