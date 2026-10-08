"""Unit tests for Betfair mid-price and expected-rank helpers (no network)."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.live import betfair as bf


class MidImplied(unittest.TestCase):
    def test_mid_of_even_market(self) -> None:
        # back 2.0 / lay 2.02 → about 0.4975
        mid = bf.mid_implied(2.0, 2.02)
        assert mid is not None
        self.assertAlmostEqual(mid, 0.5 * (0.5 + 1 / 2.02), places=6)

    def test_rejects_one_sided(self) -> None:
        self.assertIsNone(bf.mid_implied(2.0, None))
        self.assertIsNone(bf.mid_implied(None, 2.1))

    def test_rejects_crossed_or_wide(self) -> None:
        self.assertIsNone(bf.mid_implied(2.2, 2.0))
        self.assertIsNone(bf.mid_implied(2.0, 3.0))  # 50% relative spread


class Simplex(unittest.TestCase):
    def test_match_odds_sum_to_one(self) -> None:
        out = bf.simplex({"Home": 0.45, "Draw": 0.28, "Away": 0.22}, mass=1.0)
        self.assertAlmostEqual(sum(out.values()), 1.0, places=9)

    def test_top6_mass(self) -> None:
        out = bf.simplex({"A": 0.5, "B": 0.5, "C": 0.5}, mass=6.0)
        self.assertAlmostEqual(sum(out.values()), 6.0, places=9)


class LiquidityTiers(unittest.TestCase):
    def test_tier1_keeps_probs(self) -> None:
        h, d, a, tier = bf.shrink_1x2(0.5, 0.25, 0.25, matched=50_000)
        self.assertEqual(tier, "tier1")
        self.assertAlmostEqual(h, 0.5)

    def test_tier3_is_neutral(self) -> None:
        h, d, a, tier = bf.shrink_1x2(0.9, 0.05, 0.05, matched=100)
        self.assertEqual(tier, "tier3_neutral")
        self.assertAlmostEqual(h, bf.NEUTRAL_1X2[0])

    def test_fair_decimal_round_trip(self) -> None:
        self.assertAlmostEqual(1.0 / bf.fair_decimal(0.4), 0.4, places=6)


class ExpectedRank(unittest.TestCase):
    def test_identity_over_twenty_equal_clubs(self) -> None:
        # Uniform: each club win 1/20, top6 6/20, rel 3/20
        rank = bf.expected_rank(1 / 20, 6 / 20, 3 / 20)
        self.assertAlmostEqual(20 * rank, 210.0, places=6)

    def test_favourite_ranks_above_relegation_candidate(self) -> None:
        fav = bf.expected_rank(0.35, 0.85, 0.01)
        dog = bf.expected_rank(0.001, 0.02, 0.40)
        self.assertLess(fav, dog)

    def test_strength_sign(self) -> None:
        self.assertGreater(bf.strength_from_rank(3.0), 0.0)
        self.assertLess(bf.strength_from_rank(18.0), 0.0)


class Classify(unittest.TestCase):
    def test_names(self) -> None:
        self.assertEqual(bf.classify_outright("Premier League Winner"), "winner")
        self.assertEqual(bf.classify_outright("To Finish in Top 6"), "top6")
        self.assertEqual(bf.classify_outright("Relegation"), "relegation")
        self.assertIsNone(bf.classify_outright("Match Odds"))


class Redact(unittest.TestCase):
    def test_strips_app_key(self) -> None:
        secret = "not-a-real-key"
        out = bf.redact(f"X-Application: {secret} failed", secret)
        self.assertNotIn(secret, out)
        self.assertIn("[redacted]", out)


class SecretsNotInSource(unittest.TestCase):
    def test_modules_load_key_from_env_helper(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "src/live/betfair.py").read_text(encoding="utf-8")
        self.assertIn('load_secret("BETFAIR_APP_KEY")', text)
        # No hardcoded 16-char app-key assignment in source.
        self.assertNotRegex(text, r'BETFAIR_APP_KEY\s*=\s*["\'][A-Za-z0-9]{16}["\']')


if __name__ == "__main__":
    unittest.main()
