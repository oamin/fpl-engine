"""Carry from a Gameweek 1 fifteen. No weekly reset and no network."""

from __future__ import annotations

import inspect
import unittest

from src.models.friend_start import assert_carried, carried_state
from src.models.season_climb_ft import SquadState


def _state(ids: list[str], ft: int, bank: int = 0) -> SquadState:
    return SquadState(purchase={pid: 50 for pid in ids}, bank=bank, ft=ft)


class CarryTest(unittest.TestCase):
    def test_a_free_hit_returns_to_the_previous_fifteen(self) -> None:
        before = _state(["a", "b"], ft=2)
        after = _state(["a", "c"], ft=2, bank=4)
        nxt = carried_state(before, after, "free_hit", n_tx=1)
        self.assertEqual(set(nxt.ids()), {"a", "b"})
        self.assertEqual(nxt.ft, 2)

    def test_a_wildcard_keeps_the_rebuild_and_the_free_transfers(self) -> None:
        before = _state(["a", "b"], ft=2)
        after = _state(["a", "c"], ft=2, bank=1)
        nxt = carried_state(before, after, "wildcard", n_tx=1)
        self.assertEqual(set(nxt.ids()), {"a", "c"})
        self.assertEqual(nxt.ft, 2)
        self.assertEqual(nxt.bank, 1)

    def test_a_normal_week_rolls_one_free_transfer(self) -> None:
        before = _state(["a"], ft=1)
        after = _state(["b"], ft=1, bank=3)
        nxt = carried_state(before, after, None, n_tx=1)
        self.assertEqual(set(nxt.ids()), {"b"})
        self.assertEqual(nxt.ft, 1)

    def test_two_carried_weeks_fail_if_the_squad_resets(self) -> None:
        rows = [
            {"gw": 1, "chip": None, "pre_ids": ["a"], "model_ids": ["b"]},
            {"gw": 2, "chip": None, "pre_ids": ["a"], "model_ids": ["a"]},
        ]
        with self.assertRaises(RuntimeError):
            assert_carried(rows)

    def test_a_carried_squad_is_accepted(self) -> None:
        rows = [
            {"gw": 1, "chip": None, "pre_ids": ["a"], "model_ids": ["b"]},
            {"gw": 2, "chip": "free_hit", "pre_ids": ["b"], "model_ids": ["c"]},
            {"gw": 3, "chip": None, "pre_ids": ["b"], "model_ids": ["d"]},
        ]
        assert_carried(rows)


class SourceTest(unittest.TestCase):
    def test_the_module_does_not_copy_the_half_or_call_the_odds_api(self) -> None:
        import src.models.friend_start as friend_start

        source = inspect.getsource(friend_start)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("reset_gap_gw15", source)
        self.assertNotIn("reset_chips_gw15", source)
        self.assertNotIn("run_ft_season", source)


if __name__ == "__main__":
    unittest.main()
