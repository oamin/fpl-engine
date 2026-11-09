"""A one-week miss is a one-week miss, and a blank mass has no transport cost to invent."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_temporal_trial import (
    EXACT,
    ONE,
    SAME_GW,
    categorize,
    earth_mover,
    mass_near,
    rank_histogram,
)


class TemporalTests(unittest.TestCase):
    def test_transport_matches_the_gameweek_gap(self) -> None:
        self.assertEqual(earth_mover(np.array([24, 25]), np.array([1.0, 0.0]), np.array([0.0, 1.0])), 1.0)
        self.assertEqual(earth_mover(np.array([24, 26]), np.array([1.0, 0.0]), np.array([0.0, 1.0])), 2.0)
        self.assertEqual(earth_mover(np.array([24, 25]), np.array([1.0, 1.0]), np.array([1.0, 1.0])), 0.0)
        self.assertTrue(np.isnan(earth_mover(np.array([24]), np.array([1.0]), np.array([0.0]))))

    def test_a_neighboring_week_is_not_an_exact_hit(self) -> None:
        gameweeks = np.array([24, 25, 26])
        buckets = categorize(gameweeks, np.array([1]), np.array([True, False, False]))
        self.assertEqual(int(buckets[0]), ONE)
        exact = categorize(gameweeks, np.array([0]), np.array([True, False, False]))
        self.assertEqual(int(exact[0]), EXACT)
        other = categorize(np.array([24, 24]), np.array([1]), np.array([True, False]))
        self.assertEqual(int(other[0]), SAME_GW)

    def test_mass_on_the_next_week_counts_inside_one_gameweek(self) -> None:
        gameweeks = np.array([24, 25])
        actual = np.array([1.0, 0.0])
        shifted = np.array([0.0, 1.0])
        self.assertEqual(mass_near(gameweeks, shifted, actual, 0), 0.0)
        self.assertEqual(mass_near(gameweeks, shifted, actual, 1), 1.0)

    def test_top_slice_exact_rate_is_the_highlighted_share(self) -> None:
        row = {
            "gw": np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
            "actual": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64),
            "chance": np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0], dtype=np.float64),
        }
        summary = rank_histogram([row], "chance", 0.1, seeds=1)
        self.assertEqual(summary["exact"], 1.0)
        row["chance"] = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        shifted = rank_histogram([row], "chance", 0.1, seeds=1)
        self.assertEqual(shifted["exact"], 0.0)
        self.assertEqual(shifted["rest"], 1.0)
        self.assertEqual(shifted["miss_within1"], 0.0)
        row["chance"] = np.array([0, 0, 0, 5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        adjacent = rank_histogram([row], "chance", 0.1, seeds=1)
        self.assertEqual(adjacent["exact"], 0.0)
        self.assertEqual(adjacent["one"], 1.0)
        self.assertEqual(adjacent["miss_within1"], 1.0)
