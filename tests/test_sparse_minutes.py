"""Sparse minutes coupler + fixture covariance path."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.fixture_covariance import (
    _toy_fb_group,
    _toy_side,
    draw_fixture_covariant,
    sample_scoreline,
)
from src.markets.scoreline import build_matrix
from src.markets.sparse_minutes import (
    CompetitionGroup,
    competition_groups_from_pairs,
    draw_club_minutes,
    draw_competition_minutes,
)


class SparseMinutes(unittest.TestCase):
    def test_competition_draw_negative_corr(self) -> None:
        rng = np.random.default_rng(0)
        draws = np.vstack(
            [
                draw_competition_minutes(
                    np.array([70.0, 35.0]),
                    positions=["DEF", "DEF"],
                    strength=2.0,
                    rng=rng,
                )
                for _ in range(400)
            ]
        )
        c = float(np.corrcoef(draws[:, 0], draws[:, 1])[0, 1])
        self.assertLess(c, -0.2)

    def test_groups_from_pairs_greedy_no_chain(self) -> None:
        # Chain 1-2 and 2-3: greedy takes strongest pair only (1-2)
        pairs = pd.DataFrame(
            {
                "element_a": [1, 2],
                "element_b": [2, 3],
                "corr": [-0.9, -0.8],
            }
        )
        meta = pd.DataFrame(
            {
                "element": [1, 2, 3],
                "name": ["A", "B", "C"],
                "position": ["DEF", "DEF", "DEF"],
                "mean_minutes": [60.0, 50.0, 40.0],
            }
        )
        groups = competition_groups_from_pairs(pairs, meta, "X", corr_max=-0.7)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0].elements), {1, 2})

    def test_draw_club_minutes_marks_source(self) -> None:
        players = _toy_side("H")
        g = _toy_fb_group("H")
        out = draw_club_minutes(players, [g], rng=np.random.default_rng(1))
        self.assertIn("competition", set(out["m_source"]))
        self.assertTrue((out["m"] <= 90).all())


class FixtureCovariance(unittest.TestCase):
    def test_sample_scoreline_finite(self) -> None:
        mat = build_matrix(1.5, 1.2, -0.05)
        rng = np.random.default_rng(0)
        hs, as_ = zip(*[sample_scoreline(mat, rng=rng) for _ in range(200)])
        self.assertTrue(min(hs) >= 0 and max(hs) <= 6)
        self.assertTrue(min(as_) >= 0 and max(as_) <= 6)

    def test_fb_minutes_negatively_coupled(self) -> None:
        home = _toy_side("H")
        away = _toy_side("A")
        d = draw_fixture_covariant(
            1.6,
            1.2,
            0.0,
            home,
            away,
            home_groups=[_toy_fb_group("H")],
            away_groups=[],
            n_scenarios=400,
            seed=2,
        )
        ids = list(d.player_ids)
        i1, i2 = ids.index("H_D1"), ids.index("H_D2")
        c = float(np.corrcoef(d.minutes[:, i1], d.minutes[:, i2])[0, 1])
        self.assertLess(c, -0.15)

    def test_attack_vs_opp_gk_negative_xp(self) -> None:
        home = _toy_side("H")
        away = _toy_side("A")
        d = draw_fixture_covariant(
            2.0,
            0.9,
            -0.05,
            home,
            away,
            home_groups=[_toy_fb_group("H")],
            away_groups=[_toy_fb_group("A")],
            n_scenarios=500,
            seed=3,
        )
        ids = list(d.player_ids)
        c = float(
            np.corrcoef(d.xp[:, ids.index("H_M1")], d.xp[:, ids.index("A_GK")])[0, 1]
        )
        self.assertLess(c, 0.05)


if __name__ == "__main__":
    unittest.main()
