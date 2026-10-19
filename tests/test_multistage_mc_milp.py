"""Multistage MC strand helpers (no full-season MILP in CI)."""

from __future__ import annotations

import random
import unittest

from src.models.multistage_mc_milp import ClubPath, aggregate_frequencies, sample_scoreline
from src.models.multistage_mc_milp import StrandResult
import pandas as pd


class SampleScorelineTest(unittest.TestCase):
    def test_samples_near_centre(self) -> None:
        rng = random.Random(0)
        totals_h = totals_a = 0
        for _ in range(200):
            gh, ga, _, _ = sample_scoreline(
                1.5,
                1.0,
                lam_h_lo=1.0,
                lam_h_hi=2.0,
                lam_a_lo=0.6,
                lam_a_hi=1.4,
                rng=rng,
            )
            totals_h += gh
            totals_a += ga
            self.assertGreaterEqual(gh, 0)
            self.assertGreaterEqual(ga, 0)
        self.assertGreater(totals_h / 200, 0.8)
        self.assertLess(totals_h / 200, 2.5)


class ClubPathTest(unittest.TestCase):
    def test_observe_updates_expanding_mean(self) -> None:
        path = ClubPath(att=1.5, dfn=1.0, n=0.0)
        path.observe(3, 0)
        self.assertEqual(path.n, 1.0)
        self.assertAlmostEqual(path.att, 3.0)
        self.assertAlmostEqual(path.dfn, 0.0)
        path.observe(1, 2)
        self.assertAlmostEqual(path.att, 2.0)
        self.assertAlmostEqual(path.dfn, 1.0)


class AggregateTest(unittest.TestCase):
    def test_frequency(self) -> None:
        pool = pd.DataFrame(
            {
                "player_id": ["1", "2", "3"],
                "name": ["A", "B", "C"],
                "team_norm": ["x", "y", "z"],
                "position": ["MID", "MID", "FWD"],
                "value": [70, 60, 80],
            }
        )
        strands = [
            StrandResult(
                0,
                squads=[
                    {"gw": 8, "player_ids": ["1", "2"], "names": ["A", "B"], "score_sum": 1.0},
                    {"gw": 10, "player_ids": ["1", "3"], "names": ["A", "C"], "score_sum": 1.0},
                ],
            ),
            StrandResult(
                1,
                squads=[
                    {"gw": 8, "player_ids": ["1", "2"], "names": ["A", "B"], "score_sum": 1.0},
                ],
            ),
        ]
        freq = aggregate_frequencies(strands, pool)
        by_id = {r["player_id"]: r["freq"] for r in freq.to_dict("records")}
        self.assertAlmostEqual(by_id["1"], 1.0)
        self.assertAlmostEqual(by_id["2"], 2.0 / 3.0)
        self.assertAlmostEqual(by_id["3"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
