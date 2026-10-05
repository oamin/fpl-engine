"""Formation is the extra place. Displacement is what the extra place does not cover."""

from __future__ import annotations

import inspect
import unittest

from src.models.shape_split import label_shape


def _counts(defenders: int, midfielders: int, forwards: int) -> dict[str, int]:
    return {"GKP": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}


def _row(pid: str, side: str, position: str, score: float, points: float) -> dict:
    sign = 1.0 if side == "model" else -1.0
    return {
        "id": pid,
        "side": side,
        "position": position,
        "score_xp": score,
        "points": sign * points,
        "tag": "shape",
    }


class SplitTest(unittest.TestCase):
    def test_the_same_shape_is_all_displacement(self) -> None:
        rows = label_shape(
            [_row("a", "human", "MID", 6.0, 10), _row("b", "human", "MID", 4.0, 8)],
            _counts(4, 4, 2),
            _counts(4, 4, 2),
        )
        self.assertEqual({row["reason"] for row in rows}, {"displacement"})
        self.assertAlmostEqual(sum(row["points"] for row in rows), -18)

    def test_one_extra_midfield_place_labels_the_higher_score(self) -> None:
        rows = label_shape(
            [
                _row("low", "human", "MID", 3.0, 12),
                _row("high", "human", "MID", 7.0, 2),
            ],
            _counts(4, 4, 2),
            _counts(3, 5, 2),
        )
        by_id = {row["id"]: row["reason"] for row in rows}
        self.assertEqual(by_id["high"], "formation")
        self.assertEqual(by_id["low"], "displacement")

    def test_more_extra_places_than_players_are_all_formation(self) -> None:
        rows = label_shape(
            [_row("a", "model", "DEF", 5.0, 6)],
            _counts(5, 4, 1),
            _counts(3, 5, 2),
        )
        self.assertEqual(rows[0]["reason"], "formation")
        self.assertEqual(rows[0]["points"], 6)


class SourceTest(unittest.TestCase):
    def test_the_module_does_not_change_the_score_or_call_the_odds_api(self) -> None:
        import src.models.shape_split as shape_split

        source = inspect.getsource(shape_split)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("score_xp =", source)


if __name__ == "__main__":
    unittest.main()
