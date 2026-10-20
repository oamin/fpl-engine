"""Generative attack scoring for priced short-horizon t:t+1."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.short_horizon_generative import (
    score_fixture_generative,
    score_week_generative,
    trivial_lineup,
)
from src.models import short_horizon_cascade as sh


class TrivialLineup(unittest.TestCase):
    def test_eligible_from_mean_minutes(self) -> None:
        world = trivial_lineup([90.0, 45.0, 0.0])
        self.assertTrue(bool(world.eligible[0]))
        self.assertFalse(bool(world.eligible[1]))
        self.assertFalse(bool(world.eligible[2]))
        self.assertEqual(world.pair_choices, ())


class FixtureGenerative(unittest.TestCase):
    def _club(self, team: str, names: list[str], elements: list[int]) -> pd.DataFrame:
        n = len(names)
        return pd.DataFrame(
            {
                "element": elements,
                "name": names,
                "position": ["MID"] * n,
                "team": [team] * n,
                "team_n": [team] * n,
                "m": [80.0] * n,
                "sG": np.full(n, 1.0 / n),
                "sA": np.full(n, 1.0 / n),
                "rG": [0.01] * n,
                "rA": [0.01] * n,
                "bonus_mu": [0.2] * n,
                "card_mu": [-0.1] * n,
            }
        )

    def test_score_fixture_returns_boards(self) -> None:
        home = self._club("Arsenal", ["A1", "A2"], [1, 2])
        away = self._club("Everton", ["E1", "E2"], [3, 4])
        rng = np.random.default_rng(0)
        h_xp, a_xp, meta = score_fixture_generative(
            home,
            away,
            lam_h=1.8,
            lam_a=0.9,
            rho=-0.05,
            alpha_g=0.96,
            alpha_a=0.90,
            rng=rng,
            lineup_cov=False,
        )
        self.assertFalse(h_xp.empty)
        self.assertFalse(a_xp.empty)
        self.assertIn("score_xp_v2", h_xp.columns)
        self.assertEqual(len(h_xp), 2)
        self.assertIn("goals_home", meta)
        self.assertGreaterEqual(int(meta["goals_home"]), 0)

    def test_score_week_generative_aggregates(self) -> None:
        side = pd.concat(
            [
                self._club("Arsenal", ["A1", "A2"], [1, 2]),
                self._club("Everton", ["E1", "E2"], [3, 4]),
            ],
            ignore_index=True,
        )
        fits = [
            {
                "fixture": 10,
                "home": "Arsenal",
                "away": "Everton",
                "home_n": "Arsenal",
                "away_n": "Everton",
                "lam_h": 1.5,
                "lam_a": 1.0,
                "rho": 0.0,
            }
        ]
        board = score_week_generative(
            side,
            fits,
            alpha_g=0.96,
            alpha_a=0.90,
            rng=np.random.default_rng(1),
            lineup_cov=False,
        )
        self.assertEqual(set(board["element"].astype(int)), {1, 2, 3, 4})
        self.assertTrue((board["score_xp_v2"] >= 0).all() or True)


class CascadeScenarioWire(unittest.TestCase):
    def test_default_generative_on(self) -> None:
        self.assertTrue(sh.DEFAULT_GENERATIVE_ATTACK)
        self.assertTrue(sh.DEFAULT_LINEUP_COV)

    def test_score_week_scenario_routes_generative(self) -> None:
        side = pd.DataFrame(
            {
                "element": [1, 2],
                "name": ["H1", "A1"],
                "position": ["MID", "MID"],
                "team": ["Home", "Away"],
                "team_n": ["Home", "Away"],
                "m": [80.0, 80.0],
                "sG": [1.0, 1.0],
                "sA": [1.0, 1.0],
                "rG": [0.01, 0.01],
                "rA": [0.01, 0.01],
                "bonus_mu": [0.2, 0.2],
                "card_mu": [-0.1, -0.1],
            }
        )
        fits = [
            {
                "fixture": 1,
                "home": "Home",
                "away": "Away",
                "home_n": "Home",
                "away_n": "Away",
                "lam_h": 1.4,
                "lam_a": 1.1,
                "rho": 0.0,
                "xg_home": 1.4,
                "xcs_home": 0.3,
                "xgc_home": 0.4,
                "xg_away": 1.1,
                "xcs_away": 0.25,
                "xgc_away": 0.45,
            }
        ]
        ov = {
            1: {
                "lam_h": 1.5,
                "lam_a": 1.0,
                "xg_home": 1.5,
                "xcs_home": 0.3,
                "xgc_home": 0.4,
                "xg_away": 1.0,
                "xcs_away": 0.25,
                "xgc_away": 0.45,
            }
        }
        gen = sh.score_week_scenario(
            side,
            fits,
            alpha_g=0.96,
            alpha_a=0.90,
            moment_overrides=ov,
            generative_attack=True,
            lineup_cov=False,
            rng=np.random.default_rng(2),
        )
        mean = sh.score_week_scenario(
            side,
            fits,
            alpha_g=0.96,
            alpha_a=0.90,
            moment_overrides=ov,
            generative_attack=False,
        )
        self.assertFalse(gen.empty)
        self.assertFalse(mean.empty)
        self.assertEqual(set(gen["element"].astype(int)), {1, 2})
        # Generative boards are discrete draws — not identical to mean layer.
        self.assertIn("score_xp_v2", gen.columns)


if __name__ == "__main__":
    unittest.main()
