"""Deadline-causal minutes helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.deadline_minutes import (
    deadline_minute_views,
    minutes_vector_for_pool,
    oracle_minutes_by_element,
    pool_with_minutes,
)


class DeadlineMinutesTest(unittest.TestCase):
    def test_oracle_and_asof_split(self) -> None:
        merged = pd.DataFrame(
            [
                {"GW": 5, "element": 1, "minutes": 90.0, "season": "2025-26"},
                {"GW": 5, "element": 2, "minutes": 60.0, "season": "2025-26"},
                {"GW": 6, "element": 1, "minutes": 0.0, "season": "2025-26"},
                {"GW": 6, "element": 2, "minutes": 90.0, "season": "2025-26"},
            ]
        )
        pool = pd.DataFrame(
            {
                "element": [1, 2],
                "player_id": ["1", "2"],
                "m": [45.0, 45.0],
            }
        )
        views = deadline_minute_views(
            pool, merged, deadline_gw=6, season="2025-26"
        )
        # This GW uses oracle (0 / 90); as-of is from GW<6 (90 / 60).
        self.assertEqual(list(views["this_gw"]), [0.0, 90.0])
        self.assertEqual(list(views["asof"]), [90.0, 60.0])

    def test_pool_with_minutes(self) -> None:
        pool = pd.DataFrame({"element": [1], "m": [10.0], "xmi": [10.0]})
        out = pool_with_minutes(pool, np.array([77.0]))
        self.assertEqual(float(out.iloc[0]["m"]), 77.0)
        self.assertEqual(float(out.iloc[0]["xmi"]), 77.0)
        self.assertEqual(float(out.iloc[0]["sigma_m"]), 0.0)

    def test_minutes_vector_fallback(self) -> None:
        pool = pd.DataFrame({"element": [1, 9], "m": [1.0, 2.0]})
        vec = minutes_vector_for_pool(pool, {1: 90.0})
        self.assertEqual(list(vec), [90.0, 2.0])
        self.assertEqual(oracle_minutes_by_element(pd.DataFrame(), 1), {})


if __name__ == "__main__":
    unittest.main()
