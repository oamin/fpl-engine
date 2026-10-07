"""Chips left after Gameweek 5. The points gap is not rewritten."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.chip_stock import cohort_mean, count_diff, inventory, remaining


class ChipStockTest(unittest.TestCase):
    def test_a_spent_chip_is_not_still_held(self) -> None:
        left = remaining({"bench_boost", "triple_captain"})
        self.assertEqual(left, frozenset({"wildcard", "free_hit"}))

    def test_the_model_can_hold_fewer_chips(self) -> None:
        model = {"bench_boost", "triple_captain"}
        human = {"triple_captain"}
        self.assertEqual(count_diff(model, human), -1)

    def test_the_reference_row_stays_out_of_the_mean(self) -> None:
        frame = inventory(
            pd.DataFrame(
                [
                    _week("a", "rank", "bench_boost", "wildcard", gap=-10),
                    _week("a", "rank", "triple_captain", "free_hit", gap=-4),
                    _week("a", "rank", "", "bench_boost", gap=0),
                    _week("ref", "reference", "bench_boost", "", gap=20),
                    _week("ref", "reference", "triple_captain", "", gap=2),
                ]
            )
        )
        summary = cohort_mean(frame)
        self.assertEqual(summary["n"], 1.0)
        self.assertEqual(summary["mean_gap"], -14.0)
        self.assertEqual(summary["mean_count_diff"], 1.0)


def _week(entry: str, group: str, chip: str, his: str, gap: float) -> dict[str, object]:
    return {
        "entry_id": entry,
        "label": entry,
        "group": group,
        "chip": chip,
        "his_chip": his,
        "gap": gap,
    }
