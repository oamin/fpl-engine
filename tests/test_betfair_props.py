"""Unit tests for Betfair prop helpers (no network)."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.live import betfair_props as bp
from src.models import forecast_xp as fx


class GoalRates(unittest.TestCase):
    def test_minutes_scale_ignores_other_players(self) -> None:
        rates = {"2026-27:1": 1.0}
        out = bp.team_goal_rates(rates, lam=2.0, minutes={"2026-27:1": 45.0})
        # 1.0 * 45/90 = 0.5, under λ, so the rate is kept in full.
        self.assertAlmostEqual(out["2026-27:1"], 0.5)

    def test_cap_scales_down_not_up(self) -> None:
        out = bp.team_goal_rates(
            {"2026-27:1": 1.5}, lam=2.0, minutes={"2026-27:1": 90.0}
        )
        self.assertAlmostEqual(out["2026-27:1"], 1.5)
        out = bp.team_goal_rates(
            {"2026-27:1": 3.0}, lam=2.0, minutes={"2026-27:1": 90.0}
        )
        self.assertAlmostEqual(out["2026-27:1"], 2.0)

    def test_two_players_share_only_the_team_ceiling(self) -> None:
        under = bp.team_goal_rates(
            {"2026-27:1": 0.4, "2026-27:2": 0.8},
            lam=2.0,
            minutes={"2026-27:1": 90.0, "2026-27:2": 90.0},
        )
        self.assertAlmostEqual(under["2026-27:1"], 0.4)
        self.assertAlmostEqual(under["2026-27:2"], 0.8)
        over = bp.team_goal_rates(
            {"2026-27:1": 1.5, "2026-27:2": 1.5},
            lam=2.0,
            minutes={"2026-27:1": 90.0, "2026-27:2": 90.0},
        )
        self.assertAlmostEqual(over["2026-27:1"], 1.0)
        self.assertAlmostEqual(over["2026-27:2"], 1.0)

    def test_zero_minutes_writes_zero_and_empty_rates_stay_empty(self) -> None:
        out = bp.team_goal_rates(
            {"2026-27:1": 1.0}, lam=2.0, minutes={"2026-27:1": 0.0}
        )
        self.assertAlmostEqual(out["2026-27:1"], 0.0)
        self.assertEqual(
            bp.team_goal_rates({}, lam=2.0, minutes={}),
            {},
        )
        self.assertEqual(
            bp.team_goal_rates({"2026-27:1": 1.0}, lam=0.0, minutes={"2026-27:1": 90.0}),
            {},
        )


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


class T1Book(unittest.TestCase):
    def test_every_populated_price_enters_the_score(self) -> None:
        root = Path("data/predictions/2026-27/gw06/betfair_t1")
        boot = json.loads((root / "bootstrap.json").read_text(encoding="utf-8"))
        fixtures = json.loads((root / "fixtures.json").read_text(encoding="utf-8"))
        names = {int(team["id"]): str(team["name"]) for team in boot["teams"]}
        odds = pd.read_csv(root / "gw_lines.csv")
        pots = fx.opening_pots_by_team_gw(odds, fixtures, names)
        from src.teams import norm_team

        for row in odds.to_dict("records"):
            home = norm_team(row["HomeTeam"])
            away = norm_team(row["AwayTeam"])
            gw = int(row["gw"])
            self.assertIn((gw, home), pots, row["HomeTeam"])
            self.assertIn((gw, away), pots, row["AwayTeam"])
            home_pot = pots[(gw, home)][0]
            over = row.get("Avg>2.5")
            under = row.get("Avg<2.5")
            populated = (
                isinstance(over, float)
                and isinstance(under, float)
                and math.isfinite(over)
                and math.isfinite(under)
            )
            if populated:
                self.assertNotAlmostEqual(home_pot["e_total"], fx.NEUTRAL_TOTAL)
            else:
                bare = fx.side_pot(
                    float(row["AvgH"]),
                    float(row["AvgD"]),
                    float(row["AvgA"]),
                    None,
                    None,
                    is_home=True,
                )
                self.assertAlmostEqual(home_pot["e_total"], bare["e_total"])
                self.assertAlmostEqual(home_pot["lam_scored"], bare["lam_scored"])

        rows = json.loads((root / "betfair_to_score.json").read_text(encoding="utf-8"))
        mapped = bp.match_to_score_runners(rows, boot["elements"])
        self.assertEqual(len(mapped), len(rows))
        by_mu = {float(row["mu_raw"]) for row in rows}
        self.assertEqual(set(mapped.values()), by_mu)
        for row in rows:
            p_mid = float(row["p_mid"])
            self.assertAlmostEqual(float(row["mu_raw"]), -math.log(1.0 - p_mid), places=9)

        table = json.loads((root / "outrights_ranks.json").read_text(encoding="utf-8"))
        strengths = bp.strength_index(table)
        self.assertEqual(len(strengths), 19)
        self.assertNotIn("nottm forest", strengths)
        forecast = bp.forecast_pots_from_outrights(
            fixtures,
            names,
            strengths,
            start=6,
            end=8,
            priced_weeks={6, 7},
        )
        self.assertIn((8, "nottm forest"), forecast)
        self.assertEqual(len({club for gw, club in forecast if gw == 8}), 20)


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
