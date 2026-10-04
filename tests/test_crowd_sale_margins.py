"""The sale-margin calls stay inside the lock."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.crowd_sale_margins import (
    LEADERS,
    diagnose,
    half_stats,
    sale_rows,
    summarise,
    verdict,
)


def _stats(n: int, median: float, share_under: float) -> dict[str, float]:
    return {"n": n, "median": median, "share_under": share_under, "share_over": 0.0}


class CallTests(unittest.TestCase):
    def test_a_low_median_and_a_fatter_tail_is_a_loose_bar(self) -> None:
        call = diagnose(_stats(16, 2.0, 0.60), _stats(16, 4.0, 0.30))
        self.assertEqual(call, "bar loose")

    def test_a_high_median_without_a_fatter_tail_is_a_sure_score(self) -> None:
        call = diagnose(_stats(16, 6.0, 0.40), _stats(16, 6.0, 0.30))
        self.assertEqual(call, "score sure and wrong")

    def test_a_middle_median_is_inconclusive(self) -> None:
        call = diagnose(_stats(16, 3.0, 0.70), _stats(16, 6.0, 0.20))
        self.assertEqual(call, "inconclusive")

    def test_a_high_median_with_a_fatter_tail_is_inconclusive(self) -> None:
        call = diagnose(_stats(16, 6.0, 0.55), _stats(16, 6.0, 0.30))
        self.assertEqual(call, "inconclusive")

    def test_fewer_than_eight_sales_cannot_fire(self) -> None:
        call = diagnose(_stats(7, 1.0, 0.90), _stats(16, 6.0, 0.10))
        self.assertEqual(call, "insufficient")

    def test_the_leaders_are_the_four_named_climbs(self) -> None:
        self.assertEqual(
            LEADERS,
            (
                ("2022-23", "template"),
                ("2023-24", "template"),
                ("2024-25", "template"),
                ("2025-26", "next"),
            ),
        )


class SaleTests(unittest.TestCase):
    def test_an_illegal_move_is_dropped(self) -> None:
        decisions = [
            {"role": "move", "gw": 20, "n_transfers": 1, "margin": 1.4, "hits": 0, "hold_legal": True},
            {"role": "move", "gw": 21, "n_transfers": 1, "margin": 9.0, "hits": 0, "hold_legal": False},
            {"role": "move", "gw": 22, "n_transfers": 0, "margin": 0.0, "hits": 0, "hold_legal": True},
            {"role": "hold", "gw": 22, "n_transfers": 0, "margin": 0.0, "hits": 0, "hold_legal": True},
        ]
        kept, dropped = sale_rows(decisions)
        self.assertEqual(list(kept["gw"]), [20])
        self.assertEqual(list(dropped["gw"]), [21])
        self.assertEqual(kept.iloc[0]["half"], "H2")

    def test_the_call_uses_every_sale_in_the_second_half(self) -> None:
        sales = pd.DataFrame(
            [
                {"season": "2022-23", "squad": "template", "half": "H2", "margin": 6.0, "hits": 0},
                {"season": "2022-23", "squad": "template", "half": "H2", "margin": 6.0, "hits": 1},
                {"season": "2022-23", "squad": "template", "half": "H1", "margin": 1.3, "hits": 0},
            ]
            + [
                {"season": "2022-23", "squad": "template", "half": "H2", "margin": 6.0, "hits": 0}
                for _ in range(6)
            ]
            + [
                {"season": "2023-24", "squad": "template", "half": "H2", "margin": 6.0, "hits": 0}
                for _ in range(8)
            ]
        )
        summary = summarise(sales)
        self.assertEqual(verdict(summary), "score sure and wrong")
        h2 = summary.loc[
            (summary["season"] == "2022-23") & (summary["half"] == "H2") & (summary["kind"] == "all")
        ].iloc[0]
        self.assertEqual(int(h2["n"]), 8)
        stats = half_stats(pd.Series([1.0, 3.0, 5.0]))
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["median"], 3.0)
        self.assertAlmostEqual(stats["share_under"], 1 / 3)
        self.assertAlmostEqual(stats["share_over"], 1 / 3)
