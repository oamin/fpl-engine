"""Wildcard judgement. The decision number is the discounted XI plus captain."""

from __future__ import annotations

import inspect
import unittest

from src.live.wildcard_plan import (
    _squad_sum,
    best_free_transfer,
    build_plan,
    horizon_weights,
    later_week_score,
    minutes_ok,
    solve_wildcard,
)
from src.models.season_climb_ft import GAMMA


def _player(
    pid: int,
    position: str,
    club: str,
    weeks: list[float],
    *,
    owned: bool = False,
    buyable: bool = True,
    now: int = 40,
    sell: int = 40,
    name: str = "",
) -> dict:
    return {
        "pid": pid,
        "name": name or f"p{pid}",
        "position": position,
        "club": club,
        "now_cost": now,
        "sell": sell,
        "owned": owned,
        "buyable": buyable,
        "weeks": weeks,
    }


def _owned_fifteen(score: float = 1.0) -> list[dict]:
    roles = (
        [(1, "GKP"), (2, "GKP")]
        + [(pid, "DEF") for pid in range(3, 8)]
        + [(pid, "MID") for pid in range(8, 13)]
        + [(pid, "FWD") for pid in range(13, 16)]
    )
    return [
        _player(pid, pos, f"c{pid}", [score, 0.0, 0.0, 0.0], owned=True, name=f"o{pid}")
        for pid, pos in roles
    ]


class FormulaTest(unittest.TestCase):
    def test_weights_follow_gamma_and_do_not_drop_a_missing_week(self) -> None:
        weights = horizon_weights()
        self.assertEqual(weights, [GAMMA**h for h in range(4)])
        row = _player(1, "MID", "a", [10.0, 10.0, 0.0, 0.0])
        self.assertAlmostEqual(_squad_sum([row], weights), 10.0 + 9.0)

    def test_a_missing_pot_is_zero(self) -> None:
        self.assertAlmostEqual(later_week_score(10.0, 2.0, 1.0), 5.0)
        self.assertEqual(later_week_score(10.0, None, 1.0), 0.0)
        self.assertEqual(later_week_score(10.0, 0.0, 1.0), 0.0)

    def test_the_minutes_filter_needs_two_starts_and_no_doubt(self) -> None:
        self.assertTrue(minutes_ok(2, "a", None))
        self.assertTrue(minutes_ok(2, "a", 100))
        self.assertFalse(minutes_ok(1, "a", None))
        self.assertFalse(minutes_ok(2, "d", None))
        self.assertFalse(minutes_ok(2, "i", 100))
        self.assertFalse(minutes_ok(2, "a", 75))

    def test_the_builder_does_not_read_score_xp(self) -> None:
        source = inspect.getsource(build_plan)
        self.assertNotIn("score_xp", source)
        self.assertIn("sell_price", source)
        self.assertNotIn("fetch_odds_api", source)


class PlanTest(unittest.TestCase):
    def test_one_free_transfer_takes_the_higher_score(self) -> None:
        players = _owned_fifteen()
        players.append(_player(20, "MID", "new", [20.0, 0.0, 0.0, 0.0], buyable=True, now=40, sell=40, name="in"))
        moved = best_free_transfer(players, horizon_weights(), bank=0)
        self.assertEqual(moved["in"], "in")
        self.assertAlmostEqual(moved["week0_xi_captain"], 50.0)
        quiet = [row for row in players if row["pid"] != 20]
        quiet.append(_player(21, "MID", "new", [0.0, 0.0, 0.0, 0.0], buyable=True, name="low"))
        held = best_free_transfer(quiet, horizon_weights(), bank=0)
        self.assertEqual(held["move"], "hold")

    def test_the_wildcard_keeps_the_club_cap_and_the_sell_budget(self) -> None:
        players = _owned_fifteen()
        for pid in range(4):
            players.append(
                _player(30 + pid, "DEF", "same", [30.0, 0.0, 0.0, 0.0], buyable=True, now=10, name=f"d{pid}")
            )
        players.append(_player(40, "FWD", "rich", [80.0, 0.0, 0.0, 0.0], buyable=True, now=200, name="dear"))
        budget = sum(int(row["sell"]) for row in players if row["owned"])
        wild = solve_wildcard(players, horizon_weights(), budget)
        self.assertTrue(wild["feasible"])
        self.assertLessEqual(sum(1 for name in wild["players"] if name.startswith("d")), 3)
        self.assertNotIn("dear", wild["players"])

    def test_a_short_position_is_infeasible(self) -> None:
        players = [_player(1, "GKP", "a", [1.0, 0.0, 0.0, 0.0], owned=True, buyable=True)]
        wild = solve_wildcard(players, horizon_weights(), 1000)
        self.assertFalse(wild["feasible"])
        self.assertIn("GKP", wild["reason"])


if __name__ == "__main__":
    unittest.main()
