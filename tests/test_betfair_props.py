"""Unit tests for Betfair prop helpers (no network)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.live import betfair_props as bp
from src.models import forecast_xp as fx


class GoalRates(unittest.TestCase):
    def test_minutes_scale_and_cap(self) -> None:
        rates = {"2026-27:1": 1.0}
        shares = {"2026-27:1": 0.5, "2026-27:2": 0.5}
        out = bp.team_goal_rates(rates, shares, lam=2.0, minutes={"2026-27:1": 45.0})
        # raw 1.0 * 45/90 = 0.5; budget = 2*(1-0.5)=1.0 → no further cut
        self.assertAlmostEqual(out["2026-27:1"], 0.5)

    def test_cap_scales_down_not_up(self) -> None:
        rates = {"2026-27:1": 1.5}
        shares = {"2026-27:1": 0.5}
        out = bp.team_goal_rates(rates, shares, lam=2.0, minutes={"2026-27:1": 90.0})
        # budget = 2*(1-0)=2, priced 1.5 → stays 1.5 (never scale up)
        self.assertAlmostEqual(out["2026-27:1"], 1.5)
        # if priced exceeds budget
        rates = {"2026-27:1": 3.0}
        out = bp.team_goal_rates(rates, shares, lam=2.0, minutes={"2026-27:1": 90.0})
        self.assertAlmostEqual(out["2026-27:1"], 2.0)


class ForecastRename(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertIs(fx.project_player, fx.compute_player_forecast)
        self.assertIs(fx.make_horizon_scores, fx.make_forecast_steps)
        self.assertIs(fx.attach_opening_horizon, fx.attach_forecast_xp)

    def test_strength_pots_shrink(self) -> None:
        home, away = fx.strength_match_pots(0.8, -0.5, shrink=0.5)
        self.assertIn("lam_scored", home)
        self.assertGreater(home["lam_scored"], away["lam_scored"])

    def test_missing_outright_club_is_bottom_tier(self) -> None:
        self.assertAlmostEqual(bp.MISSING_OUTRIGHT_STRENGTH, (10.5 - 18.5) / 9.5)
        self.assertAlmostEqual(bp.club_strength({"arsenal": 0.8}, "arsenal"), 0.8)
        self.assertAlmostEqual(
            bp.club_strength({"arsenal": 0.8}, "burnley"),
            bp.MISSING_OUTRIGHT_STRENGTH,
        )

    def test_forecast_pots_skip_priced_and_use_missing_default(self) -> None:
        fixtures = [
            {"event": 8, "team_h": 1, "team_a": 2},
            {"event": 7, "team_h": 1, "team_a": 2},
        ]
        names = {1: "Arsenal", 2: "Burnley"}
        pots = bp.forecast_pots_from_outrights(
            fixtures,
            names,
            {"arsenal": 0.9},
            start=6,
            end=9,
            priced_weeks={7},
        )
        self.assertNotIn((7, "arsenal"), pots)
        self.assertIn((8, "arsenal"), pots)
        self.assertIn((8, "burnley"), pots)
        # Strong home vs missing (bottom) away → home λ above away λ.
        self.assertGreater(
            pots[(8, "arsenal")][0]["lam_scored"],
            pots[(8, "burnley")][0]["lam_scored"],
        )


class Poisson(unittest.TestCase):
    def test_even_money(self) -> None:
        p, mu = bp.poisson_mean([2.0, 2.0])
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(mu, -__import__("math").log(0.5))


class Discover(unittest.TestCase):
    def test_env_dir_wins(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "betfair_to_score.json").write_text("[]", encoding="utf-8")
            with mock.patch.dict(os.environ, {"BETFAIR_ARTIFACTS_DIR": str(root)}):
                found = bp.discover_betfair_artifacts(6)
            self.assertEqual(found, root)


if __name__ == "__main__":
    unittest.main()
