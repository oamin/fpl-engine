"""Half-season chip schedule. No squad is picked."""

from __future__ import annotations

import inspect
import unittest

from src.live.half_plan import (
    HalfPlanError,
    SquadOutlook,
    WeekInputs,
    plan_half,
)
from src.live.policy import FH_MARGIN, WC_MARGIN


def _squad(xi: float, bench: float = 0.0, cap: float = 0.0) -> SquadOutlook:
    return SquadOutlook(xi_xp=xi, bench_xp=bench, cap_xp=cap)


def _weeks(
    start: int,
    end: int,
    *,
    held: SquadOutlook | list[SquadOutlook],
    rebuilt: SquadOutlook | list[SquadOutlook] | None = None,
    fh: dict[int, float] | None = None,
) -> list[WeekInputs]:
    rows: list[WeekInputs] = []
    n = end - start + 1
    held_rows = held if isinstance(held, list) else [held] * n
    if rebuilt is None:
        rebuilt_rows: list[SquadOutlook | None] = [None] * n
    elif isinstance(rebuilt, list):
        rebuilt_rows = list(rebuilt)
    else:
        rebuilt_rows = [rebuilt] * n
    for offset, gw in enumerate(range(start, end + 1)):
        rows.append(
            WeekInputs(
                gw=gw,
                held=held_rows[offset],
                rebuilt=rebuilt_rows[offset],
                fh_xi=None if fh is None else fh.get(gw),
            )
        )
    return rows


class WindowTests(unittest.TestCase):
    def test_a_gap_in_the_half_is_rejected(self) -> None:
        with self.assertRaises(HalfPlanError):
            plan_half(17, _weeks(17, 18, held=_squad(1.0)))

    def test_gameweek_1_cannot_play_wildcard_or_free_hit(self) -> None:
        rows = _weeks(
            1,
            19,
            held=_squad(1.0, bench=1.0, cap=1.0),
            rebuilt=_squad(5.0, bench=1.0, cap=1.0),
            fh={1: 100.0},
        )
        plan = plan_half(1, rows)
        self.assertNotIn(plan.chip, {"wildcard", "free_hit"})
        self.assertNotEqual(plan.schedule["wildcard"], 1)
        self.assertNotEqual(plan.schedule["free_hit"], 1)


class ExpiryTests(unittest.TestCase):
    def test_a_played_free_hit_blocks_the_next_week(self) -> None:
        rows = _weeks(
            20,
            38,
            held=_squad(0.0),
            fh={20: 100.0, 21: 40.0},
        )
        plan = plan_half(20, rows, played={19: "free_hit"})
        self.assertIsNone(plan.chip)
        self.assertEqual(plan.schedule["free_hit"], 21)

    def test_a_used_bench_boost_cannot_be_played_again(self) -> None:
        rows = _weeks(18, 19, held=_squad(1.0, bench=20.0, cap=1.0))
        plan = plan_half(18, rows, played={17: "bench_boost"})
        self.assertNotEqual(plan.chip, "bench_boost")
        self.assertIsNone(plan.schedule["bench_boost"])


class BenchTests(unittest.TestCase):
    def test_bench_boost_waits_for_the_better_ordinary_week(self) -> None:
        rows = _weeks(
            17,
            19,
            held=[
                _squad(10.0, bench=1.0),
                _squad(10.0, bench=1.0),
                _squad(10.0, bench=6.0),
            ],
        )
        plan = plan_half(17, rows)
        self.assertIsNone(plan.chip)
        self.assertEqual(plan.schedule["bench_boost"], 19)

    def test_bench_boost_plays_on_the_best_week(self) -> None:
        rows = _weeks(
            17,
            19,
            held=[
                _squad(10.0, bench=6.0),
                _squad(10.0, bench=1.0),
                _squad(10.0, bench=1.0),
            ],
        )
        plan = plan_half(17, rows)
        self.assertEqual(plan.chip, "bench_boost")
        self.assertEqual(plan.schedule["bench_boost"], 17)
        self.assertAlmostEqual(plan.value, 30.0 + 6.0)

    def test_an_equal_later_week_does_not_spend_the_chip(self) -> None:
        rows = _weeks(
            17,
            19,
            held=[
                _squad(10.0, bench=6.0),
                _squad(10.0, bench=6.0),
                _squad(10.0, bench=1.0),
            ],
        )
        plan = plan_half(17, rows)
        self.assertIsNone(plan.chip)


class ChipValueTests(unittest.TestCase):
    def test_triple_captain_adds_one_copy(self) -> None:
        rows = _weeks(19, 19, held=_squad(10.0, bench=1.0, cap=7.0))
        plan = plan_half(19, rows)
        self.assertEqual(plan.chip, "triple_captain")
        self.assertAlmostEqual(plan.value, 17.0)

    def test_free_hit_does_not_add_the_bench(self) -> None:
        rows = _weeks(
            18,
            19,
            held=[_squad(10.0, bench=50.0), _squad(10.0, bench=9.0)],
            fh={18: 80.0},
        )
        plan = plan_half(18, rows)
        self.assertEqual(plan.chip, "free_hit")
        self.assertEqual(plan.schedule["bench_boost"], 19)
        self.assertAlmostEqual(plan.value, 80.0 + 10.0 + 9.0)

    def test_free_hit_needs_the_week_margin(self) -> None:
        rows = _weeks(19, 19, held=_squad(10.0), fh={19: 10.0 + FH_MARGIN - 1.0})
        plan = plan_half(19, rows)
        self.assertIsNone(plan.chip)
        self.assertIsNone(plan.schedule["free_hit"])


class WildcardTests(unittest.TestCase):
    def test_wildcard_plays_when_the_half_clears_the_margin(self) -> None:
        rows = _weeks(
            17,
            19,
            held=[_squad(10.0), _squad(10.0), _squad(10.0)],
            rebuilt=[_squad(16.0), _squad(15.0), _squad(15.0)],
        )
        self.assertEqual(6.0 + 5.0 + 5.0, WC_MARGIN)
        plan = plan_half(17, rows)
        self.assertEqual(plan.chip, "wildcard")
        self.assertEqual(plan.schedule["wildcard"], 17)

    def test_wildcard_waits_when_the_half_is_short_of_the_margin(self) -> None:
        rows = _weeks(
            17,
            19,
            held=[_squad(10.0), _squad(10.0), _squad(10.0)],
            rebuilt=[_squad(15.0), _squad(15.0), _squad(15.0)],
        )
        plan = plan_half(17, rows)
        self.assertIsNone(plan.chip)
        self.assertEqual(plan.schedule["wildcard"], 18)

    def test_wildcard_waits_for_a_later_week(self) -> None:
        rows = _weeks(
            17,
            19,
            held=[_squad(40.0), _squad(10.0), _squad(10.0)],
            rebuilt=[_squad(20.0), _squad(20.0), _squad(20.0)],
        )
        plan = plan_half(17, rows)
        self.assertIsNone(plan.chip)
        self.assertEqual(plan.schedule["wildcard"], 18)


class TieTests(unittest.TestCase):
    def test_two_chips_with_the_same_value_play_nothing(self) -> None:
        rows = _weeks(19, 19, held=_squad(10.0, bench=5.0, cap=5.0))
        plan = plan_half(19, rows)
        self.assertIsNone(plan.chip)
        self.assertAlmostEqual(plan.value, 10.0)


class IsolationTests(unittest.TestCase):
    def test_the_module_does_not_climb_or_restate_the_chip_law(self) -> None:
        import src.live.half_plan as mod

        source = inspect.getsource(mod)
        self.assertNotIn("recommend_chip", source)
        self.assertNotIn("pick_squad", source)
        self.assertNotIn("run_ft_season", source)
        self.assertNotIn("benchmark", source)
        self.assertIn("ChipWallet", source)


if __name__ == "__main__":
    unittest.main()
