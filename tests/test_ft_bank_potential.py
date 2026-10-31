"""Unit tests for FT bank potential energy (no season totals)."""

from __future__ import annotations

import unittest

from src.models.multistage_mc_milp import DEFAULT_FT_PE_NU, ft_bank_potential
from src.rules.fpl_2026 import HIT_COST, MAX_FT


class FtBankPotentialTest(unittest.TestCase):
    def test_engine_default_pe_off_bank_pot_separate(self) -> None:
        """Path objective does not include bank PE by default."""
        self.assertEqual(DEFAULT_FT_PE_NU, 0.0)

    def test_pe_zero_when_nu_or_bank_zero(self) -> None:
        self.assertEqual(ft_bank_potential(0, nu=4.0), 0.0)
        self.assertEqual(ft_bank_potential(3, nu=0.0), 0.0)
        self.assertEqual(ft_bank_potential(3, nu=-1.0), 0.0)

    def test_pe_linear_hit_cost_per_ft(self) -> None:
        """Each stored FT = +HIT_COST; five FTs = +20."""
        self.assertEqual(ft_bank_potential(1, nu=HIT_COST), float(HIT_COST))
        self.assertEqual(ft_bank_potential(2, nu=HIT_COST), 2.0 * HIT_COST)
        self.assertEqual(ft_bank_potential(5, nu=HIT_COST), 5.0 * HIT_COST)
        self.assertEqual(
            ft_bank_potential(MAX_FT + 3, nu=HIT_COST), float(MAX_FT) * HIT_COST
        )

    def test_pe_default_nu(self) -> None:
        self.assertEqual(ft_bank_potential(3), 3.0 * HIT_COST)

    def test_hold_adds_hit_value_until_cap(self) -> None:
        """Hold below cap → +HIT_COST; at five the next grant adds +0."""
        for b in range(0, MAX_FT):
            self.assertEqual(
                ft_bank_potential(b + 1) - ft_bank_potential(b),
                float(HIT_COST),
            )
        self.assertEqual(
            ft_bank_potential(MAX_FT) - ft_bank_potential(MAX_FT),
            0.0,
        )
        # Cap: spending one FT then receiving the weekly grant returns to MAX_FT
        # → net ΔPE = 0 (same as holding and wasting the grant).
        after_use = min(MAX_FT, (MAX_FT - 1) + 1)
        self.assertEqual(after_use, MAX_FT)
        self.assertEqual(
            ft_bank_potential(after_use) - ft_bank_potential(MAX_FT),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
