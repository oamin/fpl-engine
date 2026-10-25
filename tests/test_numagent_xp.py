"""Numagent 2.0 xP assembler — unit tests (no network)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.markets import numagent_xp as nx
from src.markets import scoreline as sl


class WeightsShares(unittest.TestCase):
    def test_weight_delta_includes_minutes_sigma(self) -> None:
        w, dw = nx.weight_and_delta(r=0.02, dr=0.0, m=90.0, sigma_m=10.0)
        self.assertAlmostEqual(w, 0.02)
        self.assertAlmostEqual(dw, 0.02 * 10.0 / 90.0)

    def test_shares_sum_to_one(self) -> None:
        w = np.array([0.1, 0.3, 0.6])
        dw = np.array([0.01, 0.02, 0.03])
        s, ds = nx.shares_and_deltas(w, dw)
        self.assertAlmostEqual(float(s.sum()), 1.0, places=6)
        self.assertTrue(np.all(ds >= 0))


class AppearanceDefconBonus(unittest.TestCase):
    def test_appearance_points(self) -> None:
        self.assertEqual(nx.appearance_xp(0), 0)
        self.assertEqual(nx.appearance_xp(45), 1)
        self.assertEqual(nx.appearance_xp(60), 2)

    def test_defcon_high_mean_near_two(self) -> None:
        xp, se, p = nx.defcon_xp(mu_count=14.0, sig_count=2.0, n_prior=20, tau=10.0, m=90.0)
        self.assertGreater(p, 0.9)
        self.assertGreater(xp, 1.8)
        self.assertGreater(se, 0.0)

    def test_defcon_gated_by_minutes(self) -> None:
        xp, se, p = nx.defcon_xp(mu_count=14.0, sig_count=2.0, n_prior=20, tau=10.0, m=45.0)
        self.assertEqual(xp, 0.0)
        self.assertEqual(p, 0.0)

    def test_prior_mean_se(self) -> None:
        mu, se, n = nx.prior_mean_se(np.array([0.0, 2.0, 0.0, 2.0]))
        self.assertEqual(n, 4)
        self.assertAlmostEqual(mu, 1.0)
        self.assertGreater(se, 0.0)

    def test_history_priors_saves_use_stepped_points(self) -> None:
        # Raw saves 2,3,5,6 → ⌊/3⌋ = 0,1,1,2 → mean 1.0
        hist = pd.DataFrame(
            {
                "name": ["GK"] * 4,
                "position": ["GKP"] * 4,
                "minutes": [90.0] * 4,
                "bonus": [0.0] * 4,
                "defensive_contribution": [0.0] * 4,
                "yellow_cards": [0] * 4,
                "red_cards": [0] * 4,
                "saves": [2, 3, 5, 6],
            }
        )
        pri = nx.history_priors_for_xp(hist)
        self.assertAlmostEqual(float(pri.loc[0, "saves_mu"]), 1.0, places=12)
        self.assertEqual(int(pri.loc[0, "saves_n"]), 4)


class FullClubShareProjection(unittest.TestCase):
    def test_thin_selection_keeps_full_club_mass(self) -> None:
        """Impact sub dilutes star π; selection board must not re-normalise."""
        full = pd.DataFrame(
            {
                "player_id": ["1", "2", "3"],
                "name": ["Star", "Mate", "Sub"],
                "position": ["FWD", "MID", "MID"],
                "rG": [0.05, 0.02, 0.02],
                "drG": [0.01, 0.005, 0.005],
                "rA": [0.01, 0.01, 0.01],
                "drA": [0.003, 0.003, 0.003],
                "m": [90.0, 90.0, 20.0],
                "sigma_m": [0.0, 0.0, 0.0],
                "bonus_mu": [0.0, 0.0, 0.0],
                "bonus_se": [0.0, 0.0, 0.0],
                "card_mu": [0.0, 0.0, 0.0],
                "card_se": [0.0, 0.0, 0.0],
                "defcon_mu": [0.0, 0.0, 0.0],
                "defcon_sig": [nx.SIG_FLOOR, nx.SIG_FLOOR, nx.SIG_FLOOR],
                "defcon_n": [0, 0, 0],
            }
        )
        thin = full.loc[full["m"] >= 60].copy()
        thin_only = nx.score_team_side(
            thin, xg=2.0, xcs=0.3, xgc=0.8, alpha_g=0.96, alpha_a=0.90,
            share_se_shrink=False,
        )
        with_full = nx.score_team_side(
            thin,
            xg=2.0,
            xcs=0.3,
            xgc=0.8,
            alpha_g=0.96,
            alpha_a=0.90,
            share_players=full,
            share_se_shrink=False,
        )
        self.assertAlmostEqual(float(thin_only["sG"].sum()), 1.0, places=6)
        self.assertLess(float(with_full["sG"].sum()), 0.95)
        self.assertLess(
            float(with_full.loc[with_full["name"] == "Star", "sG"].iloc[0]),
            float(thin_only.loc[thin_only["name"] == "Star", "sG"].iloc[0]),
        )
        self.assertLess(
            float(with_full.loc[with_full["name"] == "Star", "xp_goals"].iloc[0]),
            float(thin_only.loc[thin_only["name"] == "Star", "xp_goals"].iloc[0]),
        )


class ScoreTeamSide(unittest.TestCase):
    def test_rotation_renormalises_attack(self) -> None:
        # Two MIDs share attack; starter gets most weight.
        players = pd.DataFrame(
            {
                "name": ["A", "B"],
                "position": ["MID", "MID"],
                "rG": [0.02, 0.02],
                "drG": [0.005, 0.005],
                "rA": [0.01, 0.01],
                "drA": [0.003, 0.003],
                "m": [90.0, 30.0],
                "sigma_m": [5.0, 15.0],
                "defcon_mu": [8.0, 8.0],
                "defcon_sig": [3.0, 3.0],
                "defcon_n": [10, 10],
                "bonus_mu": [0.5, 0.2],
                "bonus_se": [0.1, 0.1],
                "card_mu": [-0.1, -0.1],
                "card_se": [0.05, 0.05],
            }
        )
        out = nx.score_team_side(
            players,
            xg=1.5,
            xcs=0.3,
            xgc=0.8,
            alpha_g=0.96,
            alpha_a=0.90,
        )
        self.assertAlmostEqual(float(out["sG"].sum()), 1.0, places=6)
        self.assertGreater(float(out.loc[0, "sG"]), float(out.loc[1, "sG"]))
        self.assertGreater(float(out.loc[0, "xp"]), float(out.loc[1, "xp"]))
        # Appearance 2 vs 1
        self.assertEqual(float(out.loc[0, "xp_appear"]), 2.0)
        self.assertEqual(float(out.loc[1, "xp_appear"]), 1.0)
        # MID CS = 1 * xcs when m>=60
        self.assertAlmostEqual(float(out.loc[0, "xp_cs"]), 0.3, places=6)
        self.assertEqual(float(out.loc[1, "xp_cs"]), 0.0)
        # MID no GC
        self.assertEqual(float(out.loc[0, "xp_gc"]), 0.0)

    def test_def_gets_gc_and_cs(self) -> None:
        players = pd.DataFrame(
            {
                "name": ["D"],
                "position": ["DEF"],
                "rG": [0.0],
                "drG": [0.0],
                "rA": [0.0],
                "drA": [0.0],
                "m": [90.0],
                "sigma_m": [0.0],
                "bonus_mu": [0.0],
                "bonus_se": [0.0],
                "card_mu": [0.0],
                "card_se": [0.0],
                "defcon_mu": [11.0],
                "defcon_sig": [2.0],
                "defcon_n": [15],
            }
        )
        out = nx.score_team_side(
            players, xg=1.0, xcs=0.4, xgc=1.2, alpha_g=0.96, alpha_a=0.90
        )
        self.assertAlmostEqual(float(out.loc[0, "xp_cs"]), 0.4 * 4, places=6)
        self.assertAlmostEqual(float(out.loc[0, "xp_gc"]), -1.2, places=6)
        self.assertGreater(float(out.loc[0, "xp_defcon"]), 1.0)

    def test_out_player_zero(self) -> None:
        players = pd.DataFrame(
            {
                "name": ["X"],
                "position": ["FWD"],
                "rG": [0.05],
                "drG": [0.01],
                "rA": [0.02],
                "drA": [0.01],
                "m": [0.0],
                "sigma_m": [0.0],
                "bonus_mu": [1.0],
                "bonus_se": [0.2],
                "card_mu": [-0.2],
                "card_se": [0.1],
            }
        )
        out = nx.score_team_side(
            players, xg=2.0, xcs=0.2, xgc=0.5, alpha_g=0.96, alpha_a=0.90
        )
        self.assertEqual(float(out.loc[0, "xp"]), 0.0)


class ScoreFixture(unittest.TestCase):
    def test_fixture_uses_home_away_moments(self) -> None:
        mat = sl.build_matrix(1.8, 1.0, rho=0.0)
        mom = sl.team_moments(mat)
        home = pd.DataFrame(
            {
                "name": ["H"],
                "position": ["FWD"],
                "rG": [0.03],
                "drG": [0.01],
                "rA": [0.01],
                "drA": [0.005],
                "m": [90.0],
                "sigma_m": [5.0],
            }
        )
        away = pd.DataFrame(
            {
                "name": ["A"],
                "position": ["FWD"],
                "rG": [0.03],
                "drG": [0.01],
                "rA": [0.01],
                "drA": [0.005],
                "m": [90.0],
                "sigma_m": [5.0],
            }
        )
        out = nx.score_fixture(mat, home, away, alpha_g=0.96, alpha_a=0.90)
        self.assertEqual(len(out), 2)
        h = out[out["side"] == "home"].iloc[0]
        a = out[out["side"] == "away"].iloc[0]
        self.assertAlmostEqual(float(h["xg"]), mom["xg_home"], places=6)
        self.assertAlmostEqual(float(a["xg"]), mom["xg_away"], places=6)
        self.assertGreater(float(h["xp_goals"]), float(a["xp_goals"]))


if __name__ == "__main__":
    unittest.main()
