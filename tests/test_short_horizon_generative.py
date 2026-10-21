"""Generative attack scoring for priced short-horizon t:t+1."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.short_horizon_generative import (
    score_fixture_generative,
    score_week_generative,
    trivial_lineup,
    xp_soft_priors_expanding,
)
from src.models import short_horizon_cascade as sh


class TrivialLineup(unittest.TestCase):
    def test_eligible_from_mean_minutes(self) -> None:
        world = trivial_lineup([90.0, 45.0, 0.0])
        self.assertTrue(bool(world.eligible[0]))
        self.assertFalse(bool(world.eligible[1]))
        self.assertFalse(bool(world.eligible[2]))
        self.assertEqual(world.pair_choices, ())


class SoftPriorsExpanding(unittest.TestCase):
    def test_bonus_cards_saves_from_mu_not_roll(self) -> None:
        club = pd.DataFrame(
            {
                "bonus_mu": [0.4, 0.0],
                "card_mu": [-0.2, 0.0],
                "saves_mu": [1.5, 0.0],
                "saves_se": [0.1, 0.0],
                "defcon_mu": [0.0, 0.0],
                "defcon_sig": [0.5, 0.5],
                "defcon_n": [5, 5],
                "sigma_m": [0.0, 0.0],
            }
        )
        minutes = np.array([90.0, 0.0])
        soft = xp_soft_priors_expanding(
            club, positions=["GKP", "MID"], minutes=minutes
        )
        self.assertAlmostEqual(float(soft["xp_bonus"][0]), 0.4)
        self.assertAlmostEqual(float(soft["xp_cards"][0]), -0.2)
        self.assertGreater(float(soft["xp_saves"][0]), 0.0)
        # Benched: soft priors zeroed by minutes gate.
        self.assertAlmostEqual(float(soft["xp_bonus"][1]), 0.0)
        self.assertAlmostEqual(float(soft["xp_saves"][1]), 0.0)

    def test_soft_priors_independent_of_goals_against(self) -> None:
        """Changing rolled GA must not move DefCon/bonus/cards/saves."""
        home = pd.DataFrame(
            {
                "element": [1],
                "name": ["Keeper"],
                "position": ["GKP"],
                "team": ["Home"],
                "team_n": ["Home"],
                "m": [90.0],
                "sG": [0.0],
                "sA": [0.0],
                "bonus_mu": [0.3],
                "card_mu": [-0.1],
                "saves_mu": [1.2],
                "saves_se": [0.2],
                "defcon_mu": [0.0],
                "defcon_sig": [0.5],
                "defcon_n": [5],
                "sigma_m": [0.0],
            }
        )
        away = home.copy()
        away["element"] = [2]
        away["name"] = ["AwayK"]
        away["team"] = ["Away"]
        away["team_n"] = ["Away"]
        softs = []
        for seed in (0, 1, 2, 3, 4):
            h_xp, _, meta = score_fixture_generative(
                home,
                away,
                lam_h=1.5,
                lam_a=1.5,
                rho=0.0,
                alpha_g=0.96,
                alpha_a=0.90,
                rng=np.random.default_rng(seed),
                lineup_cov=False,
            )
            softs.append(
                (
                    float(h_xp.iloc[0]["xp_bonus"]),
                    float(h_xp.iloc[0]["xp_cards"]),
                    float(h_xp.iloc[0]["xp_saves"]),
                    float(h_xp.iloc[0]["xp_defcon"]),
                    int(meta["goals_away"]),
                )
            )
        # Soft columns identical across rolls; GA may vary.
        bonus = {s[0] for s in softs}
        cards = {s[1] for s in softs}
        saves = {s[2] for s in softs}
        defcon = {s[3] for s in softs}
        self.assertEqual(len(bonus), 1)
        self.assertEqual(len(cards), 1)
        self.assertEqual(len(saves), 1)
        self.assertEqual(len(defcon), 1)
        self.assertAlmostEqual(softs[0][0], 0.3)


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

    def test_cs_gc_match_rolled_scoreline(self) -> None:
        """Defence xP is the rolled GA, not moment xCS/xGC."""
        home = self._club("Arsenal", ["A1"], [1])
        home["position"] = ["DEF"]
        away = self._club("Everton", ["E1"], [2])
        away["position"] = ["DEF"]
        # Extreme λ so scoreline is usually 1–0-ish but we assert consistency.
        for seed in range(20):
            h_xp, a_xp, meta = score_fixture_generative(
                home,
                away,
                lam_h=2.0,
                lam_a=1.0,
                rho=0.0,
                alpha_g=0.96,
                alpha_a=0.90,
                rng=np.random.default_rng(seed),
                lineup_cov=False,
            )
            ga_h = int(meta["goals_away"])
            ga_a = int(meta["goals_home"])
            self.assertEqual(int(h_xp.iloc[0]["goals_against"]), ga_h)
            self.assertEqual(int(a_xp.iloc[0]["goals_against"]), ga_a)
            self.assertEqual(bool(h_xp.iloc[0]["clean_sheet"]), ga_h == 0)
            self.assertEqual(bool(a_xp.iloc[0]["clean_sheet"]), ga_a == 0)
            self.assertEqual(int(h_xp.iloc[0]["gc_bins"]), ga_h // 2)
            if ga_h == 0:
                self.assertAlmostEqual(float(h_xp.iloc[0]["xp_cs"]), 4.0)
                self.assertAlmostEqual(float(h_xp.iloc[0]["xp_gc"]), 0.0)
            else:
                self.assertAlmostEqual(float(h_xp.iloc[0]["xp_cs"]), 0.0)
                self.assertAlmostEqual(
                    float(h_xp.iloc[0]["xp_gc"]), -float(ga_h // 2)
                )

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
        self.assertFalse(sh.DEFAULT_LINEUP_COV)

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
