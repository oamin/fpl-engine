"""Share-SE shrink toward minutes prior."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets import numagent_xp as nx


class ShareSeShrink(unittest.TestCase):
    def test_high_se_share_moves_toward_minutes(self) -> None:
        # Fragile inflated share (Gvardiol-style) vs equal minutes prior (1/3).
        s = np.array([0.10, 0.70, 0.20])
        ds = np.array([0.02, 0.25, 0.05])  # middle row very uncertain
        m = np.array([90.0, 90.0, 90.0])
        s1, diag = nx.shrink_shares_by_se(s, ds, m, tau2=0.01)
        self.assertAlmostEqual(float(s1.sum()), 1.0, places=6)
        self.assertLess(s1[1], s[1])  # uncertain mass cut toward ~1/3
        self.assertGreater(diag["max_abs_delta"], 0.0)

    def test_zero_se_leaves_share(self) -> None:
        s = np.array([0.2, 0.5, 0.3])
        ds = np.zeros(3)
        m = np.array([90.0, 90.0, 90.0])
        s1, _ = nx.shrink_shares_by_se(s, ds, m, tau2=0.01)
        np.testing.assert_allclose(s1, s, atol=1e-9)

    def test_positive_share_prior_keeps_zero_rate_def_and_gk_at_zero(self) -> None:
        # One scorer + 90′ DEF/GK with s=0 must not absorb minutes prior.
        s = np.array([0.0, 0.0, 1.0])  # GKP, DEF, FWD
        ds = np.array([0.07, 0.07, 0.20])
        m = np.array([90.0, 90.0, 90.0])
        s1, _ = nx.shrink_shares_by_se(
            s, ds, m, tau2=0.01, prior_support="positive_share"
        )
        self.assertAlmostEqual(float(s1[0]), 0.0, places=9)
        self.assertAlmostEqual(float(s1[1]), 0.0, places=9)
        self.assertAlmostEqual(float(s1[2]), 1.0, places=6)

    def test_all_minutes_prior_leaks_onto_zero_share(self) -> None:
        s = np.array([0.0, 0.0, 1.0])
        ds = np.array([0.07, 0.07, 0.20])
        m = np.array([90.0, 90.0, 90.0])
        s1, _ = nx.shrink_shares_by_se(
            s, ds, m, tau2=0.01, prior_support="all_minutes"
        )
        self.assertGreater(float(s1[0]), 0.0)
        self.assertGreater(float(s1[1]), 0.0)
        self.assertLess(float(s1[2]), 1.0)

    def test_score_team_side_flag_changes_assist_mass(self) -> None:
        players = pd.DataFrame(
            {
                "player_id": ["a", "b", "c"],
                "position": ["DEF", "MID", "FWD"],
                "rG": [0.002, 0.01, 0.02],
                "drG": [0.002, 0.005, 0.005],
                "rA": [0.02, 0.005, 0.005],  # DEF monopolises assists on rate×m
                "drA": [0.02, 0.002, 0.002],
                "m": [90.0, 90.0, 90.0],
                "sigma_m": [0.0, 0.0, 0.0],
            }
        )
        raw = nx.score_team_side(
            players,
            xg=2.0,
            xcs=0.4,
            xgc=0.2,
            alpha_g=0.96,
            alpha_a=0.90,
            share_se_shrink=False,
        )
        shr = nx.score_team_side(
            players,
            xg=2.0,
            xcs=0.4,
            xgc=0.2,
            alpha_g=0.96,
            alpha_a=0.90,
            share_se_shrink=True,
            share_se_tau2=0.01,
            share_se_prior="all_minutes",
        )
        self.assertTrue(bool(shr["share_se_shrink"].iloc[0]))
        # Uncertain DEF assist share should not stay as extreme
        self.assertNotAlmostEqual(float(raw.loc[0, "sA"]), float(shr.loc[0, "sA"]), places=3)

    def test_position_minutes_prior_pulls_cb_goal_mass_down(self) -> None:
        # Sparse Spurs-like: one CB goal + FWD + short MID → raw CB share huge.
        s = np.array([0.37, 0.45, 0.18])  # DEF, FWD, MID
        ds = np.array([0.20, 0.15, 0.12])
        m = np.array([90.0, 72.0, 26.0])
        pos = ["DEF", "FWD", "MID"]
        s1, _ = nx.shrink_shares_by_se(
            s,
            ds,
            m,
            tau2=0.01,
            prior_support="position_minutes",
            positions=pos,
            channel="G",
        )
        self.assertAlmostEqual(float(s1.sum()), 1.0, places=6)
        self.assertLess(float(s1[0]), float(s[0]))  # CB pulled down
        self.assertGreater(float(s1[1]), float(s1[0]))  # FWD above CB after shrink

    def test_score_team_side_scorer_prior_zeros_def_goals(self) -> None:
        players = pd.DataFrame(
            {
                "player_id": ["gk", "def", "fwd"],
                "position": ["GKP", "DEF", "FWD"],
                "rG": [0.0, 0.0, 0.02],
                "drG": [0.001, 0.001, 0.005],
                "rA": [0.0, 0.0, 0.01],
                "drA": [0.001, 0.001, 0.002],
                "m": [90.0, 90.0, 90.0],
                "sigma_m": [0.0, 0.0, 0.0],
            }
        )
        out = nx.score_team_side(
            players,
            xg=2.0,
            xcs=0.4,
            xgc=0.2,
            alpha_g=0.96,
            alpha_a=0.90,
            share_se_shrink=True,
            share_se_tau2=0.01,
            share_se_prior="positive_share",
        )
        self.assertAlmostEqual(float(out.loc[0, "sG"]), 0.0, places=9)
        self.assertAlmostEqual(float(out.loc[1, "sG"]), 0.0, places=9)
        self.assertAlmostEqual(float(out.loc[0, "xp_goals"]), 0.0, places=9)
        self.assertAlmostEqual(float(out.loc[1, "xp_goals"]), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
