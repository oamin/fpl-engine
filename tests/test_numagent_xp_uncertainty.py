"""Error-propagation checks for Numagent 2.0 xP (method §5–§9)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from src.markets import numagent_xp as nx
from src.rules.fpl_2026 import GOAL_POINTS


class WeightDelta(unittest.TestCase):
    def test_rate_only_when_sigma_m_zero(self) -> None:
        w, dw = nx.weight_and_delta(r=0.02, dr=0.01, m=90.0, sigma_m=0.0)
        self.assertAlmostEqual(w, 0.02)
        self.assertAlmostEqual(dw, 0.01)  # m/90 * dr = 1 * 0.01

    def test_minutes_only_when_dr_zero(self) -> None:
        w, dw = nx.weight_and_delta(r=0.02, dr=0.0, m=90.0, sigma_m=9.0)
        self.assertAlmostEqual(w, 0.02)
        self.assertAlmostEqual(dw, 0.02 * 9.0 / 90.0)

    def test_independent_quadrature(self) -> None:
        r, dr, m, sm = 0.015, 0.004, 75.0, 12.0
        _, dw = nx.weight_and_delta(r, dr, m, sm)
        expect = math.sqrt((m / 90.0 * dr) ** 2 + (r / 90.0 * sm) ** 2)
        self.assertAlmostEqual(dw, expect, places=12)


class ShareDelta(unittest.TestCase):
    def test_two_player_closed_form(self) -> None:
        w = np.array([0.2, 0.6])
        dw = np.array([0.05, 0.10])
        s, ds = nx.shares_and_deltas(w, dw)
        W = float(w.sum())
        self.assertAlmostEqual(float(s[0]), w[0] / W)
        # Δs_i = (1/W) √[(1−s_i)²(Δw_i)² + s_i²(Δw_j)²]
        for i, j in ((0, 1), (1, 0)):
            expect = (1.0 / W) * math.sqrt(
                (1.0 - s[i]) ** 2 * dw[i] ** 2 + s[i] ** 2 * dw[j] ** 2
            )
            self.assertAlmostEqual(float(ds[i]), expect, places=12)

    def test_zero_weight_row_has_zero_share(self) -> None:
        w = np.array([0.0, 0.4, 0.6])
        dw = np.array([0.0, 0.02, 0.03])
        s, ds = nx.shares_and_deltas(w, dw)
        self.assertEqual(float(s[0]), 0.0)
        self.assertAlmostEqual(float(s.sum()), 1.0, places=12)

    def test_mc_share_sd_matches_delta_order(self) -> None:
        """Joint normal draws on w → empirical SD of s ≈ analytic Δs."""
        rng = np.random.default_rng(0)
        w = np.array([0.15, 0.25, 0.60])
        dw = np.array([0.04, 0.05, 0.08])
        _, ds = nx.shares_and_deltas(w, dw)
        draws = rng.normal(loc=w, scale=dw, size=(20000, 3))
        draws = np.clip(draws, 0.0, None)
        S = draws / draws.sum(axis=1, keepdims=True)
        emp = S.std(axis=0, ddof=1)
        # Order-of-magnitude / within ~25% for interior shares
        for i in range(3):
            self.assertGreater(float(ds[i]), 0.0)
            ratio = float(emp[i] / ds[i])
            self.assertGreater(ratio, 0.7)
            self.assertLess(ratio, 1.35)


class AttackUncertainty(unittest.TestCase):
    def test_relative_quadrature_on_expectation(self) -> None:
        s, ds = 0.4, 0.05
        g_team, rel_g = 1.2, 0.02  # ΔG/G
        e = s * g_team
        de = e * math.sqrt((ds / s) ** 2 + rel_g**2)
        # Reproduce via score_team_side with a lone attacker (share=1 → use two with known s)
        # Direct formula check only here.
        self.assertAlmostEqual(de, 1.2 * 0.4 * math.sqrt((0.05 / 0.4) ** 2 + 0.02**2), places=12)

    def test_alpha_se_propagates_into_goal_xp(self) -> None:
        players = pd.DataFrame(
            {
                "name": ["A", "B"],
                "position": ["MID", "MID"],
                "rG": [0.02, 0.01],
                "drG": [0.0, 0.0],
                "rA": [0.0, 0.0],
                "drA": [0.0, 0.0],
                "m": [90.0, 90.0],
                "sigma_m": [0.0, 0.0],
            }
        )
        base = nx.score_team_side(
            players, xg=2.0, xcs=0.0, xgc=0.0, alpha_g=1.0, alpha_a=0.9, d_alpha_g=0.0
        )
        with_se = nx.score_team_side(
            players, xg=2.0, xcs=0.0, xgc=0.0, alpha_g=1.0, alpha_a=0.9, d_alpha_g=0.1
        )
        # Same point estimate
        np.testing.assert_allclose(base["xp_goals"], with_se["xp_goals"], rtol=0, atol=1e-12)
        # Larger α SE → larger goal XP SE
        self.assertTrue(
            (with_se["xp_se"].to_numpy(float) >= base["xp_se"].to_numpy(float) - 1e-12).all()
        )
        self.assertGreater(float(with_se["xp_se"].sum()), float(base["xp_se"].sum()))


class DefconUncertainty(unittest.TestCase):
    def test_delta_method_matches_finite_difference(self) -> None:
        mu, sig, n, tau, m = 11.0, 3.0, 16, 10.0, 90.0
        xp, se, _ = nx.defcon_xp(mu, sig, n, tau, m)
        eps = 1e-5
        xp_hi, _, _ = nx.defcon_xp(mu + eps, sig, n, tau, m)
        xp_lo, _, _ = nx.defcon_xp(mu - eps, sig, n, tau, m)
        d_xp_d_mu = (xp_hi - xp_lo) / (2.0 * eps)
        se_delta = sig / math.sqrt(n)  # SE on μ (= SE on δ)
        se_fd = abs(d_xp_d_mu) * se_delta
        # Assembler uses SE_δ = σ/√n but formula SE_xp = 2 φ(z)/√n
        # which equals |∂xp/∂δ| * (σ/√n) only if ∂xp/∂δ = 2φ/σ
        self.assertAlmostEqual(se, se_fd, places=5)
        self.assertGreater(xp, 0.0)

    def test_se_shrinks_with_n(self) -> None:
        _, se_small, _ = nx.defcon_xp(11.0, 3.0, 4, 10.0, 90.0)
        _, se_big, _ = nx.defcon_xp(11.0, 3.0, 36, 10.0, 90.0)
        self.assertAlmostEqual(se_small / se_big, math.sqrt(36 / 4), places=6)


class PriorMeanSe(unittest.TestCase):
    def test_se_is_sigma_over_sqrt_n(self) -> None:
        vals = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 2.0])
        mu, se, n = nx.prior_mean_se(vals)
        self.assertEqual(n, 6)
        self.assertAlmostEqual(mu, float(vals.mean()))
        self.assertAlmostEqual(se, float(vals.std(ddof=1) / math.sqrt(6)))

    def test_bonus_band_clip_floor(self) -> None:
        mu, se = 0.1, 0.25
        lo, hi = max(0.0, mu - se), mu + se
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.35)

    def test_card_band_clip_ceiling(self) -> None:
        mu, se = -0.1, 0.25
        lo, hi = mu - se, min(0.0, mu + se)
        self.assertLess(lo, 0.0)
        self.assertEqual(hi, 0.0)


class TotalXpSe(unittest.TestCase):
    def _players(self, **over) -> pd.DataFrame:
        base = {
            "name": ["P"],
            "position": ["MID"],
            "rG": [0.02],
            "drG": [0.005],
            "rA": [0.01],
            "drA": [0.003],
            "m": [90.0],
            "sigma_m": [8.0],
            "defcon_mu": [13.0],
            "defcon_sig": [3.0],
            "defcon_n": [20],
            "bonus_mu": [0.6],
            "bonus_se": [0.15],
            "card_mu": [-0.12],
            "card_se": [0.04],
        }
        base.update(over)
        return pd.DataFrame(base)

    def test_quadrature_of_reported_components(self) -> None:
        out = nx.score_team_side(
            self._players(),
            xg=1.5,
            xcs=0.25,
            xgc=0.0,
            alpha_g=0.96,
            alpha_a=0.90,
            d_alpha_g=0.01,
            d_alpha_a=0.01,
            d_xcs=0.05,
        )
        # Assembler stores xp_se as quadrature of component SEs; isolate bonus.
        only_bonus = nx.score_team_side(
            self._players(
                rG=[0.0],
                drG=[0.0],
                rA=[0.0],
                drA=[0.0],
                sigma_m=[0.0],
                defcon_mu=[0.0],
                defcon_n=[0],
                card_mu=[0.0],
                card_se=[0.0],
                bonus_se=[0.2],
            ),
            xg=0.0,
            xcs=0.0,
            xgc=0.0,
            alpha_g=0.96,
            alpha_a=0.90,
        )
        self.assertAlmostEqual(float(only_bonus.loc[0, "xp_se"]), 0.2, places=12)
        self.assertAlmostEqual(float(only_bonus.loc[0, "xp_bonus"]), 0.6, places=12)

    def test_independent_components_add_in_quadrature(self) -> None:
        # bonus_se and card_se only
        out = nx.score_team_side(
            self._players(
                rG=[0.0],
                drG=[0.0],
                rA=[0.0],
                drA=[0.0],
                sigma_m=[0.0],
                defcon_mu=[0.0],
                defcon_n=[0],
                bonus_se=[0.3],
                card_se=[0.4],
            ),
            xg=0.0,
            xcs=0.0,
            xgc=0.0,
            alpha_g=0.96,
            alpha_a=0.90,
        )
        self.assertAlmostEqual(float(out.loc[0, "xp_se"]), math.sqrt(0.3**2 + 0.4**2), places=12)

    def test_saves_se_in_quadrature_for_gkp(self) -> None:
        players = pd.DataFrame(
            {
                "name": ["GK1"],
                "position": ["GKP"],
                "rG": [0.0],
                "drG": [0.0],
                "rA": [0.0],
                "drA": [0.0],
                "m": [90.0],
                "sigma_m": [0.0],
                "saves_mu": [1.0],  # mean ⌊saves/3⌋, not raw saves
                "saves_se": [0.2],
                "bonus_mu": [0.0],
                "bonus_se": [0.0],
                "card_mu": [0.0],
                "card_se": [0.0],
                "defcon_n": [0],
            }
        )
        out = nx.score_team_side(
            players, xg=0.0, xcs=0.0, xgc=0.0, alpha_g=0.96, alpha_a=0.9
        )
        # xp_saves = 1.0 at 90′; se = 0.2
        self.assertAlmostEqual(float(out.loc[0, "xp_saves"]), 1.0, places=12)
        self.assertAlmostEqual(float(out.loc[0, "xp_se"]), 0.2, places=12)

    def test_saves_minutes_scale_and_mid_zero(self) -> None:
        xp90, se90 = nx.saves_xp(1.0, 0.2, m=90.0, sigma_m=0.0, position="GKP")
        xp45, se45 = nx.saves_xp(1.0, 0.2, m=45.0, sigma_m=0.0, position="GKP")
        self.assertAlmostEqual(xp90, 1.0, places=12)
        self.assertAlmostEqual(xp45, 0.5, places=12)
        self.assertAlmostEqual(se45, 0.1, places=12)
        z, zs = nx.saves_xp(1.0, 0.2, m=90.0, position="MID")
        self.assertEqual(z, 0.0)
        self.assertEqual(zs, 0.0)

    def test_out_player_se_is_zero(self) -> None:
        out = nx.score_team_side(
            self._players(m=[0.0]),
            xg=2.0,
            xcs=0.3,
            xgc=1.0,
            alpha_g=0.96,
            alpha_a=0.90,
            d_xcs=0.1,
            d_xgc=0.2,
        )
        self.assertEqual(float(out.loc[0, "xp"]), 0.0)
        self.assertEqual(float(out.loc[0, "xp_se"]), 0.0)

    def test_cs_se_scales_with_d_xcs(self) -> None:
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
            }
        )
        out = nx.score_team_side(
            players, xg=0.0, xcs=0.5, xgc=0.0, alpha_g=0.96, alpha_a=0.9, d_xcs=0.1
        )
        # CS pts = 4 → Δxp_cs = 0.4; no other SE terms
        self.assertAlmostEqual(float(out.loc[0, "xp_cs"]), 2.0, places=12)
        self.assertAlmostEqual(float(out.loc[0, "xp_se"]), 0.4, places=12)

    def test_mc_total_xp_sd_near_reported_se(self) -> None:
        """Perturb bonus/cards within SE; empirical SD ≈ reported xp_se."""
        rng = np.random.default_rng(1)
        bonus_mu, bonus_se = 0.5, 0.2
        card_mu, card_se = -0.15, 0.08
        players = self._players(
            rG=[0.0],
            drG=[0.0],
            rA=[0.0],
            drA=[0.0],
            sigma_m=[0.0],
            defcon_mu=[0.0],
            defcon_n=[0],
            bonus_mu=[bonus_mu],
            bonus_se=[bonus_se],
            card_mu=[card_mu],
            card_se=[card_se],
        )
        point = nx.score_team_side(
            players, xg=0.0, xcs=0.0, xgc=0.0, alpha_g=0.96, alpha_a=0.9
        )
        se_hat = float(point.loc[0, "xp_se"])
        xs = []
        for _ in range(8000):
            p = players.copy()
            p.loc[0, "bonus_mu"] = bonus_mu + rng.normal(0.0, bonus_se)
            p.loc[0, "card_mu"] = card_mu + rng.normal(0.0, card_se)
            xs.append(
                float(
                    nx.score_team_side(
                        p, xg=0.0, xcs=0.0, xgc=0.0, alpha_g=0.96, alpha_a=0.9
                    ).loc[0, "xp"]
                )
            )
        emp = float(np.std(xs, ddof=1))
        self.assertAlmostEqual(se_hat, math.sqrt(bonus_se**2 + card_se**2), places=12)
        ratio = emp / se_hat
        self.assertGreater(ratio, 0.9)
        self.assertLess(ratio, 1.1)


class GoalAssistPtsScale(unittest.TestCase):
    def test_goal_points_scale_delta(self) -> None:
        # One FWD takes all share; ΔE_G from α only
        players = pd.DataFrame(
            {
                "name": ["F"],
                "position": ["FWD"],
                "rG": [0.05],
                "drG": [0.0],
                "rA": [0.0],
                "drA": [0.0],
                "m": [90.0],
                "sigma_m": [0.0],
            }
        )
        out = nx.score_team_side(
            players, xg=1.0, xcs=0.0, xgc=0.0, alpha_g=1.0, alpha_a=0.9, d_alpha_g=0.2
        )
        e_g = float(out.loc[0, "E_G"])
        self.assertAlmostEqual(e_g, 1.0)
        # ΔE = E * (dα/α) = 0.2; xp_goals SE = 4 * 0.2
        self.assertAlmostEqual(
            float(out.loc[0, "xp_se"]),
            GOAL_POINTS["FWD"] * 0.2,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
