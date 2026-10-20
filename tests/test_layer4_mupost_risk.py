"""μ_post + risk L4 — λ=0 recovers mean path; λ>0 tilts mu_dec."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_week import held_from_frame, solve_week_transfers
from src.strategy.layer4_mupost_risk import build_mupost_risk_horizon
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.risk_mu import attach_risk_mu, posterior_se, z_by_position
from src.strategy.shrink_climb import SeasonStore
from src.strategy.shrinkage import CohortMoments


class RiskMuUnit(unittest.TestCase):
    def test_z_by_position_zero_mean(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        pos = np.array(["MID", "MID", "MID", "FWD", "FWD", "FWD"])
        z = z_by_position(v, pos)
        self.assertAlmostEqual(float(z[:3].mean()), 0.0, places=9)
        self.assertAlmostEqual(float(z[3:].mean()), 0.0, places=9)

    def test_lambda_zero_recovers_mu_post(self) -> None:
        df = pd.DataFrame(
            {
                "mu_post": [4.0, 5.0, 6.0],
                "xp_se": [1.0, 2.0, 0.5],
                "sd_cov": [3.0, 1.0, 2.0],
                "position": ["MID", "MID", "MID"],
            }
        )
        mom = {"MID": CohortMoments(prior=5.0, tau2=1.0, sigma_model2=0.0, n=3, mean_se2=1.0)}
        out = attach_risk_mu(df, lambda_risk=0.0, moments_by_pos=mom)
        self.assertTrue(np.allclose(out["mu_dec"], out["mu_post"]))

    def test_posterior_se_scales_by_k(self) -> None:
        mom = {"MID": CohortMoments(prior=5.0, tau2=1.0, sigma_model2=0.0, n=10, mean_se2=1.0)}
        # K = 1/(1+1) = 0.5
        se = posterior_se(np.array([2.0, 4.0]), np.array(["MID", "MID"]), mom)
        self.assertTrue(np.allclose(se, [1.0, 2.0]))


class MupostRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        v2 = pd.read_csv("data/processed/score_xp_v2_closed.csv")
        cls.store = SeasonStore("2025-26", v2)
        cls.gw = 15
        pool = cls.store.pool(cls.gw)
        cls.moments = cls.store.moments_asof(cls.gw)
        cls.work, _ = attach_mu_post(
            pool,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=cls.moments,
            out_col="mu_post",
        )

    def test_lambda0_h1_recovers_l3(self) -> None:
        hz = build_mupost_risk_horizon(
            "2025-26",
            self.gw,
            week0_pool=self.work,
            moments=self.moments,
            lambda_risk=0.0,
            n_scenarios=8,
        )
        splan = solve_week_sigma(
            self.work,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=self.moments,
            budget=BUDGET_TENTHS,
            time_limit=30.0,
        )
        squad = splan.plan.squad.copy()
        squad["purchase"] = squad["value"].astype(int)
        bank = int(BUDGET_TENTHS - squad["purchase"].sum())
        held = held_from_frame(squad, bank=max(0, bank), ft=1)
        l3 = solve_week_transfers(self.work, held, mu_col="mu_post", max_hits=2)
        h1 = solve_horizon_transfers(
            [hz.pools[0]],
            held,
            mu_col="mu_dec",
            horizon=1,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=2,
        )
        self.assertEqual(set(h1.week0.bought), set(l3.bought))
        self.assertEqual(h1.week0.captain_id, l3.captain_id)

    def test_mu_post_intact_and_risk_tilts(self) -> None:
        hz = build_mupost_risk_horizon(
            "2025-26",
            self.gw,
            week0_pool=self.work,
            moments=self.moments,
            lambda_risk=0.35,
            n_scenarios=8,
        )
        w0 = hz.pools[0]
        self.assertIn("mu_dec", w0.columns)
        self.assertIn("mu_post", w0.columns)
        # week0 mu_post matches L3 board
        m0 = self.work.set_index("player_id")["mu_post"]
        m0b = w0.set_index("player_id")["mu_post"]
        common = m0.index.intersection(m0b.index)
        self.assertGreater(len(common), 50)
        self.assertLess(float((m0.loc[common] - m0b.loc[common]).abs().max()), 1e-9)
        # risk tilt moves at least some players
        delta = (w0["mu_dec"] - w0["mu_post"]).abs()
        self.assertGreater(float(delta.max()), 1e-6)
        self.assertGreater(int((w0["sd_cov"] > 0).sum()), 10)


if __name__ == "__main__":
    unittest.main()
