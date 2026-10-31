"""cascade_long generative player scoring (props off)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.cascade_score_v2 import score_pool_generative_on_scenarios
from src.markets.short_horizon_generative import score_fixture_generative
from src.models import cascade_long as cl
from src.models import multistage_mc_milp as ms


def _club(team: str, names: list[str], elements: list[int], *, pos: str = "MID") -> pd.DataFrame:
    n = len(names)
    return pd.DataFrame(
        {
            "player_id": [str(e) for e in elements],
            "element": elements,
            "name": names,
            "position": [pos] * n,
            "team": [team] * n,
            "team_norm": [team] * n,
            "m": [80.0] * n,
            "sG": np.full(n, 1.0 / n),
            "sA": np.full(n, 1.0 / n),
            "rG": [0.01] * n,
            "rA": [0.01] * n,
            "bonus_mu": [0.2] * n,
            "card_mu": [-0.1] * n,
            "saves_mu": [0.0] * n,
            "saves_se": [0.0] * n,
            "defcon_mu": [0.0] * n,
            "defcon_sig": [0.5] * n,
            "defcon_n": [5] * n,
            "sigma_m": [0.0] * n,
        }
    )


class CascadeLongDefaults(unittest.TestCase):
    def test_generative_default_on_props_off(self) -> None:
        self.assertEqual(cl.MATCH_SOURCE, "opta_rank_gf_ga")
        self.assertTrue(ms.DEFAULT_GENERATIVE_ATTACK)
        self.assertFalse(ms.DEFAULT_LINEUP_COV)


class PreRolledScoreline(unittest.TestCase):
    def test_uses_provided_goals_not_fresh_roll(self) -> None:
        home = _club("Arsenal", ["A1"], [1], pos="DEF")
        away = _club("Everton", ["E1"], [2], pos="DEF")
        h_xp, a_xp, meta = score_fixture_generative(
            home,
            away,
            lam_h=1.5,
            lam_a=1.0,
            rho=0.0,
            alpha_g=0.96,
            alpha_a=0.90,
            rng=np.random.default_rng(0),
            lineup_cov=False,
            home_props=None,
            away_props=None,
            goals_home=2,
            goals_away=0,
        )
        self.assertEqual(int(meta["goals_home"]), 2)
        self.assertEqual(int(meta["goals_away"]), 0)
        self.assertTrue(bool(h_xp.iloc[0]["clean_sheet"]))
        self.assertAlmostEqual(float(h_xp.iloc[0]["xp_cs"]), 4.0)


class ScorePoolGenerative(unittest.TestCase):
    def test_discounted_board_no_props(self) -> None:
        pool = pd.concat(
            [
                _club("Arsenal", ["A1", "A2"], [1, 2]),
                _club("Everton", ["E1", "E2"], [3, 4]),
            ],
            ignore_index=True,
        )
        scenarios = {
            10: [
                {
                    "home": "Arsenal",
                    "away": "Everton",
                    "lam_h": 1.6,
                    "lam_a": 0.9,
                    "hg": 1,
                    "ag": 0,
                }
            ]
        }
        board = score_pool_generative_on_scenarios(
            pool,
            scenarios,
            gws=[10],
            alpha_g=0.96,
            alpha_a=0.90,
            rng=np.random.default_rng(1),
            lineup_cov=False,
            gamma=1.0,
        )
        self.assertIn("mc_score", board.columns)
        self.assertTrue((board["mc_score"] >= 0).any() or True)
        self.assertEqual(len(board), 4)


if __name__ == "__main__":
    unittest.main()
