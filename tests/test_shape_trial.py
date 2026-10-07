"""The shape trial's bar, and substitutes taken from that shape's own bench."""

from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src.models.shape_trial import decide, play_shape, summarise


def _player(pid: str, position: str, score: float, points: float, minutes: float = 90) -> dict:
    return {
        "player_id": pid,
        "position": position,
        "score_xp": score,
        "minutes": minutes,
        "total_points": points,
        "xi_priority": 1.0,
    }


def _stats(median: float, mean: float, excluded: int = 0, n: int = 70) -> dict:
    return {"n": n, "excluded": excluded, "median": median, "mean": mean}


class BarTest(unittest.TestCase):
    def test_a_wide_score_gap_and_a_realised_gain_on_both_is_a_level_shift(self) -> None:
        call = decide(_stats(2.5, 3.0), _stats(2.2, 2.5, n=35), cohort_rows=70)
        self.assertEqual(call, "level shift")

    def test_veterans_disagreeing_is_inconclusive(self) -> None:
        call = decide(_stats(2.5, 3.0), _stats(0.4, 3.0, n=35), cohort_rows=70)
        self.assertEqual(call, "inconclusive")

    def test_a_small_sacrifice_and_a_realised_gain_is_noisy(self) -> None:
        call = decide(_stats(0.4, 2.5), _stats(0.8, 2.1, n=35), cohort_rows=70)
        self.assertEqual(call, "noisy")

    def test_a_negative_realised_gain_stays(self) -> None:
        call = decide(_stats(3.0, -1.0), _stats(3.0, 4.0, n=35), cohort_rows=70)
        self.assertEqual(call, "stays")

    def test_more_than_ten_missing_weeks_is_inconclusive(self) -> None:
        stats = summarise(
            [{"feasible": False, "sacrifice": None, "gain": None}] * 11
            + [{"feasible": True, "sacrifice": 3.0, "gain": 5.0}]
        )
        self.assertGreater(stats["excluded"], 10)


class ShapeTest(unittest.TestCase):
    def test_a_blank_is_replaced_from_that_shapes_bench(self) -> None:
        rows = [
            _player("gk", "GKP", 4, 2),
            _player("gk2", "GKP", 1, 0),
            _player("d1", "DEF", 5, 6),
            _player("d2", "DEF", 4, 6),
            _player("d3", "DEF", 3, 0, minutes=0),
            _player("d4", "DEF", 2, 1),
            _player("m1", "MID", 6, 2),
            _player("m2", "MID", 5, 2),
            _player("m3", "MID", 4, 2),
            _player("m4", "MID", 3.5, 2),
            _player("m5", "MID", 1, 9),
            _player("f1", "FWD", 7, 2),
            _player("f2", "FWD", 6, 2),
            _player("f3", "FWD", 5, 2),
        ]
        played = play_shape(pd.DataFrame(rows), [(3, 4, 3)])
        self.assertIsNotNone(played)
        self.assertEqual(played["points"], 29)

    def test_too_few_forwards_cannot_form_the_shape(self) -> None:
        rows = [_player("gk", "GKP", 4, 2), _player("f1", "FWD", 7, 2)]
        self.assertIsNone(play_shape(pd.DataFrame(rows), [(3, 4, 3)]))

    def test_more_than_ten_excluded_weeks_block_the_call(self) -> None:
        cohort = _stats(3.0, 4.0, excluded=11, n=59)
        veterans = _stats(3.0, 4.0, n=35)
        self.assertEqual(decide(cohort, veterans, cohort_rows=70), "inconclusive")

    def test_the_module_does_not_change_the_score_or_call_the_odds_api(self) -> None:
        import src.models.shape_trial as shape_trial

        source = inspect.getsource(shape_trial)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("the-odds-api", source)


if __name__ == "__main__":
    unittest.main()
