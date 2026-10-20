"""Planning-glue captain/stack helpers (as-of + policy selection)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.fixture_covariance import (
    build_club_side,
    draw_fixture_covariant,
)
from src.markets.fixture_covariance_audit import assert_asof_pairs_ignore_future
from src.markets.fixture_planning_glue import (
    attacker_candidates,
    policy_captains,
    policy_stacks,
    _draw_index_by_element,
)
from src.markets.minutes_covariance import load_merged


class PlanningGluePolicies(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_merged(season="2025-26")
        hist = cls.df.loc[cls.df["GW"] < 20]
        cls.home, _ = build_club_side(hist, "Liverpool")
        cls.away, _ = build_club_side(hist, "Man City")
        cls.draws = draw_fixture_covariant(
            1.8,
            1.3,
            -0.05,
            cls.home,
            cls.away,
            n_scenarios=40,
            seed=0,
        )
        cls.el_to_i = _draw_index_by_element(cls.draws, cls.home, cls.away)
        ids = list(cls.draws.player_ids)
        cls.mu = {}
        for side in (cls.home, cls.away):
            for r in side.itertuples(index=False):
                if str(r.name) in ids:
                    cls.mu[int(r.element)] = float(
                        cls.draws.mu_point[ids.index(str(r.name))]
                    )

    def test_asof_poison(self) -> None:
        self.assertTrue(assert_asof_pairs_ignore_future(self.df, "Liverpool", 20))

    def test_captain_policies_return_candidates(self) -> None:
        cands = pd.concat(
            [attacker_candidates(self.home), attacker_candidates(self.away)],
            ignore_index=True,
        )
        caps = policy_captains(self.draws, self.el_to_i, cands, self.mu)
        self.assertIsNotNone(caps["mu"])
        self.assertIn(caps["mu"], self.el_to_i)
        self.assertIn(caps["haul"], self.el_to_i)
        self.assertIn(caps["cvar"], self.el_to_i)

    def test_stack_policies_two_players(self) -> None:
        stacks = policy_stacks(self.home, self.away, self.mu)
        self.assertIsNotNone(stacks["diversify"])
        a, b = stacks["diversify"]
        self.assertNotEqual(a, b)
        if stacks["stack_same"] is not None:
            c, d = stacks["stack_same"]
            self.assertNotEqual(c, d)

    def test_mu_captain_is_argmax_mu(self) -> None:
        cands = pd.concat(
            [attacker_candidates(self.home), attacker_candidates(self.away)],
            ignore_index=True,
        )
        caps = policy_captains(self.draws, self.el_to_i, cands, self.mu)
        els = [int(e) for e in cands["element"] if int(e) in self.el_to_i]
        best = max(els, key=lambda e: self.mu[e])
        self.assertEqual(caps["mu"], best)


if __name__ == "__main__":
    unittest.main()
