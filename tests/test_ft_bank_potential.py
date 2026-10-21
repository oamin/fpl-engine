"""Unit tests for FT bank potential energy (no season totals)."""

from __future__ import annotations

from src.models.multistage_mc_milp import ft_bank_potential
from src.rules.fpl_2026 import MAX_FT


def test_pe_zero_when_nu_or_bank_zero():
    assert ft_bank_potential(0, nu=2.0) == 0.0
    assert ft_bank_potential(3, nu=0.0) == 0.0
    assert ft_bank_potential(3, nu=-1.0) == 0.0


def test_pe_increases_concave():
    nu = 2.0
    vals = [ft_bank_potential(b, nu=nu) for b in range(0, MAX_FT + 1)]
    assert vals[0] == 0.0
    deltas = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    assert all(d > 0 for d in deltas)
    # Marginal value non-increasing (concave φ).
    assert all(deltas[i] >= deltas[i + 1] - 1e-12 for i in range(len(deltas) - 1))


def test_pe_scales_with_nu():
    assert abs(ft_bank_potential(2, nu=4.0) - 2 * ft_bank_potential(2, nu=2.0)) < 1e-9
