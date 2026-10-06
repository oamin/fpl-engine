"""The forced chip calendar. No squad is scored."""

from __future__ import annotations

import unittest

from src.models.forced_chips import (
    CHIP_VALUE_FLOOR,
    FORCED_GAP_FLOOR,
    hypothesis_killed,
    legal_calendar,
)


class CalendarTests(unittest.TestCase):
    def test_a_gameweek_1_wildcard_is_dropped_and_a_later_one_is_kept(self) -> None:
        kept, dropped = legal_calendar({1: "wildcard", 3: "wildcard", 4: "bench_boost"})
        self.assertEqual(kept, {3: "wildcard", 4: "bench_boost"})
        self.assertEqual(dropped, ((1, "wildcard"),))

    def test_a_gameweek_1_free_hit_is_dropped(self) -> None:
        kept, dropped = legal_calendar({1: "free_hit", 2: "bench_boost"})
        self.assertEqual(kept, {2: "bench_boost"})
        self.assertEqual(dropped, ((1, "free_hit"),))

    def test_bench_boost_in_gameweek_1_is_kept(self) -> None:
        kept, dropped = legal_calendar({1: "bench_boost", 3: "triple_captain"})
        self.assertEqual(kept, {1: "bench_boost", 3: "triple_captain"})
        self.assertEqual(dropped, ())

    def test_a_repeated_wildcard_in_the_half_is_dropped(self) -> None:
        kept, dropped = legal_calendar({3: "wildcard", 5: "wildcard"})
        self.assertEqual(kept, {3: "wildcard"})
        self.assertEqual(dropped, ((5, "wildcard"),))

    def test_back_to_back_free_hits_drop_the_second(self) -> None:
        kept, dropped = legal_calendar({3: "free_hit", 4: "free_hit"})
        self.assertEqual(kept, {3: "free_hit"})
        self.assertEqual(dropped, ((4, "free_hit"),))


class BarTests(unittest.TestCase):
    def test_the_floors_are_the_locked_ones(self) -> None:
        self.assertEqual(CHIP_VALUE_FLOOR, 0.0)
        self.assertEqual(FORCED_GAP_FLOOR, -20.0)

    def test_a_non_positive_chip_value_kills_the_claim(self) -> None:
        self.assertTrue(hypothesis_killed(0.0, -10.0))
        self.assertTrue(hypothesis_killed(-1.0, -10.0))

    def test_a_forced_gap_at_or_below_the_floor_kills_the_claim(self) -> None:
        self.assertTrue(hypothesis_killed(7.0, -20.0))
        self.assertTrue(hypothesis_killed(7.0, -21.0))

    def test_a_positive_chip_value_above_the_floor_keeps_the_claim_alive(self) -> None:
        self.assertFalse(hypothesis_killed(7.5, -19.9))


if __name__ == "__main__":
    unittest.main()
