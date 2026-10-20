"""Unit tests for Betfair prop helpers (no network)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.live import betfair_props as bp
from src.models import forecast_xp as fx


class ToScoreRows(unittest.TestCase):
    def _market(self, matched: float, runners: list[dict]) -> tuple[list[dict], list[dict]]:
        catalogue = [
            {
                "marketId": "1.1",
                "marketName": "Player To Score",
                "marketStartTime": "2026-10-10T11:30:00.000Z",
                "event": {"name": "Arsenal v Leeds"},
                "runners": [
                    {"selectionId": row["selectionId"], "runnerName": row["runnerName"]}
                    for row in runners
                ],
            }
        ]
        book = [{"marketId": "1.1", "totalMatched": matched, "runners": runners}]
        return catalogue, book

    def test_delayed_book_uses_market_matched(self) -> None:
        catalogue, books = self._market(
            1715.0,
            [
                {
                    "selectionId": 1,
                    "runnerName": "Bukayo Saka",
                    "totalMatched": 0,
                    "ex": {
                        "availableToBack": [{"price": 2.98, "size": 20}],
                        "availableToLay": [{"price": 3.1, "size": 20}],
                    },
                },
                {
                    "selectionId": 2,
                    "runnerName": "Wide Runner",
                    "totalMatched": 0,
                    "ex": {
                        "availableToBack": [{"price": 4.0, "size": 10}],
                        "availableToLay": [{"price": 8.0, "size": 10}],
                    },
                },
            ],
        )
        rows = bp.to_score_rows(catalogue, books)
        self.assertEqual([row["runner"] for row in rows], ["Bukayo Saka"])
        self.assertEqual(rows[0]["matched_scope"], "market")
        self.assertEqual(rows[0]["matched"], 1715.0)

    def test_thin_market_stays_empty(self) -> None:
        catalogue, books = self._market(
            100.0,
            [
                {
                    "selectionId": 1,
                    "runnerName": "Bukayo Saka",
                    "totalMatched": 0,
                    "ex": {
                        "availableToBack": [{"price": 2.98, "size": 20}],
                        "availableToLay": [{"price": 3.1, "size": 20}],
                    },
                }
            ],
        )
        self.assertEqual(bp.to_score_rows(catalogue, books), [])

    def test_runner_volume_keeps_its_own_gate(self) -> None:
        catalogue, books = self._market(
            5000.0,
            [
                {
                    "selectionId": 1,
                    "runnerName": "Bukayo Saka",
                    "totalMatched": 400,
                    "ex": {
                        "availableToBack": [{"price": 2.98, "size": 20}],
                        "availableToLay": [{"price": 3.1, "size": 20}],
                    },
                },
                {
                    "selectionId": 2,
                    "runnerName": "Thin Runner",
                    "totalMatched": 10,
                    "ex": {
                        "availableToBack": [{"price": 2.5, "size": 20}],
                        "availableToLay": [{"price": 2.6, "size": 20}],
                    },
                },
            ],
        )
        rows = bp.to_score_rows(catalogue, books)
        self.assertEqual([row["runner"] for row in rows], ["Bukayo Saka"])
        self.assertEqual(rows[0]["matched_scope"], "runner")
        self.assertEqual(rows[0]["matched"], 400)


class NameMatch(unittest.TestCase):
    def test_folds_accents_and_short_first_names(self) -> None:
        players = [
            {"id": 25, "web_name": "Gyökeres", "first_name": "Viktor", "second_name": "Gyökeres"},
            {"id": 15, "web_name": "Ødegaard", "first_name": "Martin", "second_name": "Ødegaard"},
            {"id": 4, "web_name": "Gabriel", "first_name": "Gabriel", "second_name": "dos Santos Magalhães"},
            {"id": 10, "web_name": "White", "first_name": "Benjamin", "second_name": "White"},
            {"id": 480, "web_name": "Gibbs-White", "first_name": "Morgan", "second_name": "Gibbs-White"},
            {"id": 5, "web_name": "J.Timber", "first_name": "Jurriën", "second_name": "Timber"},
            {"id": 644, "web_name": "Timber", "first_name": "Quinten", "second_name": "Timber"},
        ]
        rows = [
            {"runner": "Viktor Gyokeres", "mu_raw": 0.4, "matched": 1000},
            {"runner": "Martin Odegaard", "mu_raw": 0.2, "matched": 1000},
            {"runner": "Gabriel Magalhaes", "mu_raw": 0.1, "matched": 1000},
            {"runner": "Ben White", "mu_raw": 0.05, "matched": 1000},
            {"runner": "Jurrien Timber", "mu_raw": 0.07, "matched": 1000},
        ]
        out = bp.match_to_score_runners(rows, players)
        self.assertEqual(out["2026-27:25"], 0.4)
        self.assertEqual(out["2026-27:15"], 0.2)
        self.assertEqual(out["2026-27:4"], 0.1)
        self.assertEqual(out["2026-27:10"], 0.05)
        self.assertEqual(out["2026-27:5"], 0.07)
        self.assertNotIn("2026-27:644", out)


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
