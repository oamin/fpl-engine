"""Chip buys, lower scores, and the rest of the transfer gap."""

from __future__ import annotations

import inspect
import unittest

from src.models.squad_gap import call_gap, label_transfers


def _player(pid: str, position: str, score: float, points: float) -> dict:
    return {"id": pid, "position": position, "score_xp": score, "points": points}


class LabelTest(unittest.TestCase):
    def test_a_wildcard_buy_is_a_chip_squad_player(self) -> None:
        rows = label_transfers(
            [],
            [_player("palmer", "MID", 8.0, 12)],
            human_pre={"other"},
            chip="wildcard",
        )
        self.assertEqual(rows[0]["tag"], "chip_squad")
        self.assertEqual(rows[0]["points"], -12)

    def test_a_higher_human_score_is_ranked_lower(self) -> None:
        rows = label_transfers(
            [_player("kept", "MID", 4.0, 2)],
            [_player("bought", "MID", 7.0, 10)],
            human_pre=set(),
            chip=None,
        )
        self.assertEqual(rows[0]["tag"], "ranked_lower")
        self.assertEqual(rows[0]["points"], -8)

    def test_a_higher_model_score_stays_unmatched(self) -> None:
        rows = label_transfers(
            [_player("kept", "MID", 8.0, 2)],
            [_player("bought", "MID", 4.0, 10)],
            human_pre=set(),
            chip=None,
        )
        self.assertEqual(rows[0]["tag"], "unmatched_squad")
        self.assertEqual(rows[0]["points"], -8)

    def test_the_tags_sum_to_the_signed_points(self) -> None:
        model = [_player("a", "DEF", 5.0, 6)]
        human = [_player("b", "MID", 7.0, 9), _player("c", "FWD", 3.0, 4)]
        rows = label_transfers(model, human, human_pre={"c"}, chip="free_hit")
        signed = 6 - 9 - 4
        self.assertAlmostEqual(sum(row["points"] for row in rows), signed)
        self.assertEqual({row["tag"] for row in rows}, {"chip_squad", "unmatched_squad"})

    def test_half_the_transfer_gap_isolates_the_chip_reset(self) -> None:
        self.assertEqual(call_gap(-91, -10), "chip reset")
        self.assertEqual(call_gap(-10, -91), "ranked lower")
        self.assertEqual(call_gap(-90, -90), "no single lever")


class SourceTest(unittest.TestCase):
    def test_the_module_does_not_change_the_score_or_call_the_odds_api(self) -> None:
        import src.models.squad_gap as squad_gap

        source = inspect.getsource(squad_gap)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("the-odds-api", source)


if __name__ == "__main__":
    unittest.main()
