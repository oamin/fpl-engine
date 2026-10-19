"""Geometric minutes decay helper."""

from __future__ import annotations

import unittest

from src.live.news_packets import compile_xmi_from_numeric, rolling_game_minutes


class MinutesDecay(unittest.TestCase):
    def test_rho_one_is_flat(self) -> None:
        h = [90.0, 0.0, 90.0]
        self.assertAlmostEqual(rolling_game_minutes(h, rho=1.0), 60.0)

    def test_rho_favours_recent(self) -> None:
        h = [0.0, 0.0, 90.0]
        flat = rolling_game_minutes(h, rho=1.0)
        dec = rolling_game_minutes(h, rho=0.7)
        assert flat is not None and dec is not None
        self.assertGreater(dec, flat)

    def test_numeric_compile_mixture(self) -> None:
        out = compile_xmi_from_numeric(
            player_id=1,
            name="Test",
            position="MID",
            prior=70.0,
            status="a",
            chance=100.0,
            p_start=0.5,
            m_if_start=90.0,
            m_if_bench=0.0,
        )
        self.assertAlmostEqual(float(out["xmi_compiled"]), 45.0)
        self.assertGreater(float(out["xmi_sigma"]), 0.0)
        self.assertEqual(out["xmi_source"], "llm_numeric")

    def test_numeric_hard_flag_zeros(self) -> None:
        out = compile_xmi_from_numeric(
            player_id=1,
            name="Test",
            position="MID",
            prior=70.0,
            status="i",
            chance=0.0,
            p_start=1.0,
            m_if_start=90.0,
        )
        self.assertEqual(float(out["xmi_compiled"]), 0.0)
        self.assertEqual(out["xmi_source"], "override:hard_flag")


if __name__ == "__main__":
    unittest.main()
