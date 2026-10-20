"""Strand hand-off plumbing smoke."""

from __future__ import annotations

import unittest

from src.models.strand_handoff import apply_priced_observations, handoff_paths
from src.models.multistage_mc_milp import init_paths_from_opta
import random


class StrandHandoffTest(unittest.TestCase):
    def test_priced_observe_moves_path_n(self) -> None:
        paths = init_paths_from_opta()
        # Pick two clubs that exist in Opta init.
        clubs = list(paths.keys())[:2]
        self.assertGreaterEqual(len(clubs), 2)
        home, away = clubs[0], clubs[1]
        before_n = paths[home].n
        recs = apply_priced_observations(
            paths,
            fixtures=[
                {"home": home, "away": away, "lam_h": 1.8, "lam_a": 0.9, "rho": 0.0}
            ],
            rng=random.Random(0),
        )
        self.assertEqual(len(recs), 1)
        self.assertGreater(paths[home].n, before_n)
        self.assertGreater(paths[away].n, 0.0)

    def test_handoff_paths_returns_records(self) -> None:
        paths = init_paths_from_opta()
        clubs = list(paths.keys())[:2]
        out_paths, recs = handoff_paths(
            priced_fixtures=[
                {
                    "home": clubs[0],
                    "away": clubs[1],
                    "lam_h": 1.5,
                    "lam_a": 1.1,
                }
            ],
            seed=3,
        )
        self.assertEqual(len(recs), 1)
        self.assertIn(clubs[0], out_paths)


if __name__ == "__main__":
    unittest.main()
