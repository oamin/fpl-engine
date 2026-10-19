"""Layer-2 μ_post shrinkage + week MILP — unit tests."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, SQUAD_QUOTA
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer1_week import solve_week
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.shrinkage import CohortMoments, shrink_mu
from tests.test_strategy_layer1 import _toy_pool


class Shrinkage(unittest.TestCase):
    def test_mode_none_is_identity(self) -> None:
        mu = np.array([1.0, 2.0, 3.0])
        se = np.array([0.5, 0.1, 2.0])
        out = shrink_mu(mu, se, prior=0.0, tau2=1.0, mode="none")
        np.testing.assert_allclose(out, mu)

    def test_se_k_formula(self) -> None:
        # prior=0, τ²=1, SE=1, σ_model=0 → K=0.5 → μ_post=0.5*μ
        out = shrink_mu(
            np.array([4.0]),
            np.array([1.0]),
            prior=0.0,
            tau2=1.0,
            sigma_model2=0.0,
            mode="se",
        )
        self.assertAlmostEqual(float(out[0]), 2.0, places=12)

    def test_constant_k(self) -> None:
        out = shrink_mu(
            np.array([10.0, 0.0]),
            np.array([1.0, 1.0]),
            prior=4.0,
            tau2=1.0,
            mode="constant",
            k_constant=0.5,
        )
        np.testing.assert_allclose(out, np.array([7.0, 2.0]))

    def test_tau2_zero_collapses_to_prior(self) -> None:
        out = shrink_mu(
            np.array([9.0, 1.0]),
            np.array([0.1, 0.1]),
            prior=3.0,
            tau2=0.0,
            mode="se",
        )
        np.testing.assert_allclose(out, np.array([3.0, 3.0]))

    def test_high_se_shrinks_more(self) -> None:
        mom = CohortMoments(prior=2.0, tau2=1.0, sigma_model2=0.0, n=10, mean_se2=0.25)
        low = shrink_mu(
            np.array([4.0]), np.array([0.1]), prior=mom.prior, tau2=mom.tau2, mode="se"
        )
        high = shrink_mu(
            np.array([4.0]), np.array([2.0]), prior=mom.prior, tau2=mom.tau2, mode="se"
        )
        self.assertGreater(float(low[0]), float(high[0]))


class Layer2ShrinkWeek(unittest.TestCase):
    def _pool(self) -> pd.DataFrame:
        pool = _toy_pool()
        pool["xp_se"] = (pool["score_xp_v2"] / 4.0).clip(lower=0.05)
        return pool

    def test_solves_legal_on_mu_post(self) -> None:
        splan = solve_week_sigma(self._pool(), mode="none")
        plan = splan.plan
        self.assertEqual(len(plan.squad), 15)
        self.assertEqual(len(plan.xi), 11)
        self.assertEqual(plan.squad["position"].value_counts().to_dict(), dict(SQUAD_QUOTA))
        self.assertLessEqual(plan.cost, BUDGET_TENTHS)
        top = plan.xi.sort_values("mu", ascending=False).iloc[0]
        self.assertEqual(str(top["player_id"]), plan.captain_id)

    def test_mode_none_matches_layer1(self) -> None:
        pool = self._pool()
        abs_plan = solve_week(pool, mu_col="score_xp_v2")
        splan = solve_week_sigma(pool, mode="none")
        self.assertEqual(
            set(abs_plan.xi["player_id"].astype(str)),
            set(splan.plan.xi["player_id"].astype(str)),
        )
        self.assertEqual(abs_plan.captain_id, splan.plan.captain_id)

    def test_attach_mu_post_column(self) -> None:
        pool = self._pool()
        out, moments = attach_mu_post(pool, mode="se")
        self.assertIn("mu_post", out.columns)
        self.assertTrue(moments)
        self.assertTrue(np.isfinite(out["mu_post"]).all())


if __name__ == "__main__":
    unittest.main()
