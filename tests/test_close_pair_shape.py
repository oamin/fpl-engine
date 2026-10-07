"""The close-pair shape reading, locked before the seasons are opened."""

from __future__ import annotations

import unittest

import pandas as pd

from fractions import Fraction

from src.models.close_pair_shape import (
    build_pair,
    choose_pair,
    prior_points,
    shape_rates,
    summarise,
    week_points,
)


def _player(pid: str, score: float, points: float = 0.0) -> dict:
    return {
        "player_id": pid,
        "element": pid,
        "score_xp": score,
        "points": points,
    }


def _history(points: list[float]) -> dict[int, float]:
    return {index + 1: float(value) for index, value in enumerate(points)}


class TieTest(unittest.TestCase):
    def test_an_equal_score_forms_no_pair(self) -> None:
        chosen = choose_pair([_player("b", 5.0), _player("a", 5.0)])
        self.assertIsNone(chosen)
        self.assertEqual(
            build_pair([_player("b", 5.0), _player("a", 5.0)], {}, 6),
            None,
        )


class HistoryTest(unittest.TestCase):
    def test_this_week_does_not_enter_the_rate(self) -> None:
        history = _history([1, 1, 1, 1, 1])
        history[6] = 0
        earlier = prior_points(history, 6)
        self.assertEqual(earlier, [1, 1, 1, 1, 1])
        blank, haul = shape_rates(earlier)
        self.assertEqual(blank, 1)
        self.assertEqual(haul, 0)
        self.assertNotIn(6, [week for week in history if week < 6])

    def test_a_zero_minute_week_counts_as_a_blank(self) -> None:
        frame = pd.DataFrame(
            {
                "element": ["1", "1", "1", "1", "1", "1"],
                "gw": [1, 2, 3, 4, 5, 5],
                "total_points": [6, 2, 0, 9, 1, 0],
                "minutes": [90, 90, 0, 90, 90, 0],
            }
        )
        history = week_points(frame)
        earlier = prior_points(history["1"], 6)
        self.assertEqual(earlier, [6, 2, 0, 9, 1])
        blank, haul = shape_rates(earlier)
        self.assertEqual(blank, Fraction(3, 5))
        self.assertEqual(haul, Fraction(1, 5))

    def test_five_earlier_weeks_are_required(self) -> None:
        players = [_player("a", 5.0, 4), _player("b", 4.6, 2)]
        history = {"a": _history([1, 1, 1, 1]), "b": _history([1, 1, 1, 1, 1])}
        self.assertIsNone(build_pair(players, history, 6))


class PairTest(unittest.TestCase):
    def test_half_a_point_is_kept_and_the_next_player_is_not_the_pair(self) -> None:
        players = [
            _player("a", 6.0, 3),
            _player("b", 5.5, 8),
            _player("c", 4.0, 12),
        ]
        history = {
            "a": _history([2, 2, 2, 2, 2]),
            "b": _history([8, 8, 8, 8, 8]),
            "c": _history([0, 0, 0, 0, 0]),
        }
        chosen = build_pair(players, history, 6)
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(chosen["leader"]["player_id"], "a")
        self.assertEqual(chosen["other"]["player_id"], "b")
        self.assertEqual(chosen["gap"], 0.5)
        self.assertEqual(chosen["mean"], "loss")
        self.assertEqual(chosen["blank"], "win")
        self.assertEqual(chosen["haul"], "win")

    def test_a_gap_above_half_a_point_is_dropped(self) -> None:
        self.assertIsNone(choose_pair([_player("a", 6.0), _player("b", 5.4)]))


class ReadingTest(unittest.TestCase):
    def _rows(self, season: str, mean: str, blank: str, haul: str, n: int) -> list[dict]:
        return [
            {
                "season": season,
                "position": "MID",
                "mean": mean,
                "blank": blank,
                "haul": haul,
            }
            for _ in range(n)
        ]

    def test_a_shape_needs_every_season(self) -> None:
        rows: list[dict] = []
        for season in ("2022-23", "2023-24", "2024-25", "2025-26"):
            rows.extend(self._rows(season, "loss", "win", "loss", 80))
        result = summarise(rows, ["2022-23", "2023-24", "2024-25", "2025-26"])
        self.assertEqual(result["blank_call"], "ahead")
        self.assertEqual(result["haul_call"], "parked")
        self.assertFalse(result["mean_neutral"])

    def test_seventy_nine_pairs_is_thin(self) -> None:
        rows: list[dict] = []
        for season in ("2022-23", "2023-24", "2024-25"):
            rows.extend(self._rows(season, "loss", "win", "win", 80))
        rows.extend(self._rows("2025-26", "loss", "win", "win", 79))
        result = summarise(rows, ["2022-23", "2023-24", "2024-25", "2025-26"])
        self.assertEqual(result["blank_call"], "parked")
        self.assertEqual(result["haul_call"], "parked")
