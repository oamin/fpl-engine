"""Dixon–Coles scoreline fit to AH / OU / BTTS (no network)."""

from __future__ import annotations

import unittest

import numpy as np

from src.markets import scoreline as sl


class MatrixBasics(unittest.TestCase):
    def test_matrix_sums_to_one(self) -> None:
        mat = sl.build_matrix(1.4, 1.1, rho=0.05)
        self.assertAlmostEqual(float(mat.sum()), 1.0, places=6)

    def test_independent_btts_matches_closed_form(self) -> None:
        lam_h, lam_a = 1.5, 1.2
        mat = sl.build_matrix(lam_h, lam_a, rho=0.0)
        # Truncation means only approximate; allow a small gap.
        closed = (1.0 - math_exp_neg(lam_h)) * (1.0 - math_exp_neg(lam_a))
        self.assertAlmostEqual(sl.p_btts(mat), closed, delta=0.03)


def math_exp_neg(lam: float) -> float:
    return float(np.exp(-lam))


class AsianCover(unittest.TestCase):
    def test_minus_half_is_home_win(self) -> None:
        mat = sl.build_matrix(2.0, 0.8, rho=0.0)
        # Strong home favourite: AH -0.5 cover ≈ P(home win).
        p_win = sum(
            mat[i, j]
            for i in range(mat.shape[0])
            for j in range(mat.shape[1])
            if (i if i < sl.GRID_MAX else sl.GRID_MAX)
            > (j if j < sl.GRID_MAX else sl.GRID_MAX)
        )
        self.assertAlmostEqual(sl.p_ah_cover(mat, -0.5), p_win, delta=0.02)

    def test_quarter_line_is_average(self) -> None:
        mat = sl.build_matrix(1.5, 1.2, rho=0.0)
        mid = sl.p_ah_cover(mat, -0.25)
        avg = 0.5 * (sl.p_ah_cover(mat, 0.0) + sl.p_ah_cover(mat, -0.5))
        self.assertAlmostEqual(mid, avg, places=6)


class FitSynthetic(unittest.TestCase):
    def test_recovers_near_generating_markets(self) -> None:
        # Generate targets from a known DC matrix, then refit.
        truth = sl.build_matrix(1.65, 1.05, rho=-0.08)
        line = -0.5
        targets = {
            "p_ah": sl.p_ah_cover(truth, line),
            "p_over_25": sl.p_over_25(truth),
            "p_btts": sl.p_btts(truth),
        }
        fit = sl.fit_scoreline(
            p_ah=targets["p_ah"],
            ah_line=line,
            p_over_25=targets["p_over_25"],
            p_btts=targets["p_btts"],
        )
        self.assertLess(abs(fit.residual["p_ah"]), 0.015)
        self.assertLess(abs(fit.residual["p_over_25"]), 0.015)
        self.assertLess(abs(fit.residual["p_btts"]), 0.015)
        self.assertLess(fit.sse, 5e-4)

    def test_even_match_battery(self) -> None:
        fit = sl.fit_scoreline(
            p_ah=0.50,
            ah_line=0.0,
            p_over_25=0.48,
            p_btts=0.52,
        )
        self.assertLess(fit.sse, 0.002)
        self.assertAlmostEqual(fit.lam_h, fit.lam_a, delta=0.25)

    def test_ah_ou_only_when_btts_unweighted(self) -> None:
        fit = sl.fit_scoreline(
            p_ah=0.52,
            ah_line=-0.75,
            p_over_25=0.58,
            p_btts=0.5,
            weights=(1.0, 1.0, 0.0),
        )
        self.assertLess(abs(fit.residual["p_ah"]), 0.02)
        self.assertLess(abs(fit.residual["p_over_25"]), 0.02)
        self.assertLess(fit.sse, 0.002)
        mom = sl.team_moments(fit.matrix)
        self.assertGreater(mom["xg_home"], mom["xg_away"])

    def test_home_favourite_battery(self) -> None:
        fit = sl.fit_scoreline(
            p_ah=0.62,
            ah_line=-0.5,
            p_over_25=0.55,
            p_btts=0.50,
        )
        self.assertLess(abs(fit.residual["p_ah"]), 0.02)
        self.assertLess(abs(fit.residual["p_over_25"]), 0.02)
        self.assertLess(abs(fit.residual["p_btts"]), 0.02)
        self.assertGreater(fit.lam_h, fit.lam_a)


if __name__ == "__main__":
    unittest.main()
