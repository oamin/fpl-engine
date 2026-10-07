"""The discounted wildcard sum. The hurdle stays 16."""

from __future__ import annotations

import unittest

from src.models.chip_horizon import live_row, summarise
from src.models.reset_chips import StepOutlook, choose_chip, discounted_gap
from src.models.season_climb_ft import GAMMA


def _step(gw: int, gap: float, *, cap: float = 1.0, bench: float = 1.0) -> StepOutlook:
    return StepOutlook(
        gw=gw,
        held_xi=0.0,
        bench_xp=bench,
        cap_xp=cap,
        rebuilt_xi=gap,
        fh_xi=gap,
    )


class DiscountTest(unittest.TestCase):
    def test_the_third_step_uses_081(self) -> None:
        self.assertAlmostEqual(discounted_gap([16.0]), 16.0)
        self.assertAlmostEqual(discounted_gap([0.0, 0.0, 16.0]), 12.96)
        self.assertAlmostEqual(discounted_gap([10.0, 10.0]), 19.0)
        self.assertAlmostEqual(discounted_gap([10.0, -4.0]), 6.4)
        self.assertEqual(GAMMA, 0.9)

    def test_the_stored_live_leads_add_up(self) -> None:
        row = live_row()
        self.assertAlmostEqual(row["raw"], 17.67)
        self.assertAlmostEqual(row["discounted"], 12.56 + 0.9 * 5.11)
        self.assertTrue(row["clears"])

    def test_a_front_loaded_miss_can_clear(self) -> None:
        steps = [_step(2, 30.0), _step(3, -8.0), _step(4, -8.0)]
        self.assertIsNone(choose_chip(steps, ("wildcard",))[0])
        chip, gain = choose_chip(steps, ("wildcard",), gamma=GAMMA)
        self.assertEqual(chip, "wildcard")
        self.assertAlmostEqual(gain, 30.0 - 0.9 * 8.0 - 0.81 * 8.0)

    def test_a_late_sixteen_misses_once_it_is_discounted(self) -> None:
        steps = [_step(2, 0.0), _step(3, 0.0), _step(4, 16.0)]
        self.assertEqual(choose_chip(steps, ("wildcard",))[0], "wildcard")
        self.assertIsNone(choose_chip(steps, ("wildcard",), gamma=GAMMA)[0])

    def test_weight_one_matches_the_raw_sum(self) -> None:
        steps = [_step(2, 10.0), _step(3, 6.0), _step(4, 4.0)]
        raw = choose_chip(steps, ("wildcard",))
        same = choose_chip(steps, ("wildcard",), gamma=1.0)
        self.assertEqual(raw, same)

    def test_gameweek_1_still_refuses_the_chip(self) -> None:
        steps = [_step(1, 40.0), _step(2, 40.0), _step(3, 40.0)]
        chip, _gain = choose_chip(steps, ("wildcard", "free_hit"), gamma=GAMMA)
        self.assertIsNone(chip)

    def test_the_free_hit_hurdle_stays_12(self) -> None:
        shy = [_step(3, 11.0)]
        self.assertIsNone(choose_chip(shy, ("free_hit",), gamma=GAMMA)[0])
        clear = [_step(3, 12.0)]
        self.assertEqual(choose_chip(clear, ("free_hit",), gamma=GAMMA)[0], "free_hit")

    def test_the_reference_manager_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            summarise([{"group": "reference", "flip": False, "raw_clears": False, "disc_clears": False}])
