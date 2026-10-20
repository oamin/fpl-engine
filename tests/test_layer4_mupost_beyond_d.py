"""Arm D team-form FDR scale + as-of adaptive fit."""

from __future__ import annotations

import unittest

import numpy as np

from src.strategy.layer4_mupost_beyond import (
    _ols_amp,
    team_form_fdr_scale,
)


class TeamFormFdrScale(unittest.TestCase):
    def test_favourable_fdr_raises_scale(self) -> None:
        form = {"arsenal": 55.0, "burnley": 35.0, "leagueavg": 45.0}
        hi = team_form_fdr_scale("Arsenal", "Burnley", form)
        lo = team_form_fdr_scale("Burnley", "Arsenal", form)
        self.assertGreater(hi, 1.0)
        self.assertLess(lo, 1.0)
        self.assertGreaterEqual(hi, 0.55)
        self.assertLessEqual(hi, 1.35)

    def test_decay_shrinks_effect(self) -> None:
        form = {"arsenal": 55.0, "burnley": 35.0}
        k0 = team_form_fdr_scale("Arsenal", "Burnley", form, amp=0.4, rho=0.7, fill_k=0)
        k1 = team_form_fdr_scale("Arsenal", "Burnley", form, amp=0.4, rho=0.7, fill_k=1)
        self.assertGreater(k0, k1)
        self.assertGreater(k1, 1.0)

    def test_empty_form_is_neutral(self) -> None:
        self.assertEqual(team_form_fdr_scale("Arsenal", "Burnley", {}), 1.0)

    def test_ols_amp_recovers_slope(self) -> None:
        z = np.array([0.2, 0.4, 0.6, 0.8])
        y = 0.35 * z
        amp, sse = _ols_amp(z, y)
        self.assertAlmostEqual(amp, 0.35, places=6)
        self.assertAlmostEqual(sse, 0.0, places=10)

    def test_terminal_by_player_accepted(self) -> None:
        """Smoke: horizon solver accepts terminal_by_player kwarg."""
        from src.strategy.layer4_week import solve_horizon_transfers

        self.assertTrue(callable(solve_horizon_transfers))
        import inspect

        sig = inspect.signature(solve_horizon_transfers)
        self.assertIn("terminal_by_player", sig.parameters)


if __name__ == "__main__":
    unittest.main()
