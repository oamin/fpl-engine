"""Lineup tags sum to the gap and do not pair across positions."""

from __future__ import annotations

import inspect
import unittest

from src.models.lineup_cause import tag_week


def _player(
    pid: str,
    position: str,
    score: float,
    points: float,
    *,
    intended: bool = True,
    minutes: float = 90,
) -> dict:
    return {
        "id": pid,
        "position": position,
        "score_xp": score,
        "minutes": minutes,
        "points": points,
        "intended": intended,
    }


class TagTest(unittest.TestCase):
    def test_a_higher_score_that_scores_fewer_points_is_ranked_ahead(self) -> None:
        records = tag_week(
            [_player("m", "MID", 6.0, 2)],
            [_player("h", "MID", 4.0, 10)],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["tag"], "ranked_ahead")
        self.assertTrue(records[0]["inversion"])
        self.assertEqual(records[0]["points"], -8)

    def test_a_lower_score_is_ranked_behind(self) -> None:
        records = tag_week(
            [_player("m", "DEF", 3.0, 1)],
            [_player("h", "DEF", 5.0, 8)],
        )
        self.assertEqual(records[0]["tag"], "ranked_behind")
        self.assertFalse(records[0]["inversion"])
        self.assertEqual(records[0]["points"], -7)

    def test_an_autosub_is_not_paired(self) -> None:
        records = tag_week(
            [_player("m", "MID", 6.0, 6, intended=False)],
            [_player("h", "MID", 4.0, 2)],
        )
        tags = {row["tag"] for row in records}
        self.assertEqual(tags, {"autosub", "shape"})
        self.assertAlmostEqual(sum(row["points"] for row in records), 4)

    def test_a_different_position_count_is_shape(self) -> None:
        records = tag_week(
            [_player("m", "MID", 5.0, 4)],
            [_player("h", "DEF", 5.0, 9)],
        )
        self.assertEqual({row["tag"] for row in records}, {"shape"})
        self.assertAlmostEqual(sum(row["points"] for row in records), -5)

    def test_a_blank_is_not_paired_with_a_starter(self) -> None:
        records = tag_week(
            [_player("m", "MID", 7.0, 0, minutes=0)],
            [_player("h", "MID", 4.0, 6)],
        )
        self.assertEqual({row["tag"] for row in records}, {"blank", "shape"})
        self.assertAlmostEqual(sum(row["points"] for row in records), -6)


class SourceTest(unittest.TestCase):
    def test_the_module_does_not_change_the_score_or_call_the_odds_api(self) -> None:
        import src.models.lineup_cause as lineup_cause

        source = inspect.getsource(lineup_cause)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("HOLD_EPS", source)


if __name__ == "__main__":
    unittest.main()
