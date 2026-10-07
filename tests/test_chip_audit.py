"""Chip-audit splits. No season climb."""

from __future__ import annotations

import unittest

import pandas as pd

from src.live.half_plan import SquadOutlook, WeekInputs
from src.models.chip_audit import (
    CHURN_CSV,
    LIFT_CSV,
    WILDCARD_CSV,
    bench_award,
    churn_windows,
    partition_lift,
    wildcard_margins,
)
from src.models.half_plan_scores import SCORE_CSV as STORED_CSV
from src.models.season_climb_ft import run_ft_season


def _outlook(xi: float) -> SquadOutlook:
    return SquadOutlook(xi_xp=xi, bench_xp=0.0, cap_xp=0.0)


def _weeks() -> list[WeekInputs]:
    rows = []
    for gw, gap in ((4, 6.0), (5, 6.0), (6, 6.0), (7, 6.0), (8, 0.0)):
        rows.append(WeekInputs(gw, _outlook(10.0), _outlook(10.0 + gap), 10.0 + gap))
    return rows


class WildcardMarginTest(unittest.TestCase):
    def test_priced_steps_exclude_the_copied_tail(self) -> None:
        split = wildcard_margins(4, _weeks(), [4, 5, 6])
        self.assertAlmostEqual(float(split["gap_priced"]), 18.0)
        self.assertAlmostEqual(float(split["gap_tail"]), 6.0)
        self.assertAlmostEqual(float(split["gap_all"]), 24.0)
        self.assertTrue(split["clears_priced"])
        self.assertEqual(int(split["n_priced"]), 3)
        self.assertEqual(int(split["n_tail"]), 2)

    def test_a_tail_can_clear_a_priced_gap_that_misses_16(self) -> None:
        rows = [
            WeekInputs(4, _outlook(10.0), _outlook(15.0), 15.0),
            WeekInputs(5, _outlook(10.0), _outlook(15.0), 15.0),
        ]
        rows.extend(
            WeekInputs(gw, _outlook(10.0), _outlook(16.0), 16.0)
            for gw in range(6, 20)
        )
        split = wildcard_margins(4, rows, [4, 5])
        self.assertAlmostEqual(float(split["gap_priced"]), 10.0)
        self.assertFalse(split["clears_priced"])
        self.assertGreater(float(split["gap_all"]), 16.0)
        self.assertGreater(float(split["tail_share"]), 0.75)


class PartitionTest(unittest.TestCase):
    def test_the_three_pieces_add_to_the_lift(self) -> None:
        chip = pd.DataFrame(
            {
                "gw": [4, 5, 6],
                "xi_points_cap": [40.0, 30.0, 20.0],
                "chip": ["wildcard", "bench_boost", None],
            }
        )
        empty = pd.DataFrame({"gw": [4, 5, 6], "xi_points_cap": [25.0, 22.0, 21.0]})
        split = partition_lift(chip, empty, {4: 12.0, 5: 4.0})
        self.assertAlmostEqual(split["active"], 16.0)
        self.assertAlmostEqual(split["path_gap"], (15.0 - 12.0) + (8.0 - 4.0))
        self.assertAlmostEqual(split["residual"], -1.0)
        self.assertAlmostEqual(split["lift"], 40 + 30 + 20 - (25 + 22 + 21))

    def test_a_chip_week_without_a_mechanic_fails(self) -> None:
        chip = pd.DataFrame({"gw": [4], "xi_points_cap": [10.0], "chip": ["wildcard"]})
        empty = pd.DataFrame({"gw": [4], "xi_points_cap": [8.0]})
        with self.assertRaises(Exception):
            partition_lift(chip, empty, {})


class ChurnWindowTest(unittest.TestCase):
    def test_the_wildcard_week_itself_is_outside_the_window(self) -> None:
        chip = pd.DataFrame(
            {
                "gw": [4, 5, 6, 20],
                "n_transfers": [8, 1, 2, 7],
                "hits": [0, 1, 0, 0],
                "bank": [3, 4, 5, 6],
            }
        )
        empty = pd.DataFrame(
            {
                "gw": [4, 5, 6, 20],
                "n_transfers": [1, 0, 1, 1],
                "hits": [0, 0, 1, 0],
                "bank": [10, 9, 8, 7],
            }
        )
        rows = churn_windows(chip, empty, [4, 20])
        self.assertEqual(len(rows), 2)
        first = rows[0]
        self.assertEqual(first["chip_transfers"], 3)
        self.assertEqual(first["empty_transfers"], 1)
        self.assertEqual(first["chip_hits"], 1)
        self.assertEqual(first["empty_hits"], 1)
        self.assertEqual(first["chip_bank"], 5)
        self.assertEqual(int(first["until_gw"]), 20)
        self.assertEqual(rows[1]["chip_transfers"], 0)


class BenchAwardTest(unittest.TestCase):
    def test_the_toy_bench_boost_is_the_eight_point_bench(self) -> None:
        from tests.test_half_plan_scores import _toy_season

        frame, opening = _toy_season()

        def policy(gw, state, pool, gws):
            del state, pool, gws
            if int(gw) == 1:
                return "bench_boost", 1
            return None, None

        trace: list[dict] = []
        boosted = run_ft_season(
            frame,
            {"xp": "score_xp"},
            [1],
            roster=frame,
            opening=opening,
            chip_policy=policy,
            trace=trace,
        )
        self.assertEqual(boosted.iloc[0]["chip"], "bench_boost")
        self.assertAlmostEqual(bench_award(trace[0]["squad"], trace[0]["final_xi"]), 8.0)


class StoredFileTest(unittest.TestCase):
    def test_the_audit_does_not_name_the_stored_chip_file_as_its_output(self) -> None:
        self.assertEqual(STORED_CSV.name, "half_plan_scores.csv")
        for path in (WILDCARD_CSV, LIFT_CSV, CHURN_CSV):
            self.assertNotEqual(path.name, STORED_CSV.name)
