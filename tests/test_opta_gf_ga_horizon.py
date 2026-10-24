"""Opta GF/GA long-horizon pots."""

from __future__ import annotations

import random
import unittest

from src.models.opta_gf_ga_horizon import (
    DEFAULT_RANK_METHOD,
    LIVE_E0,
    assessed_league_gf,
    gf_ga_from_rank,
    lambdas_from_gf_ga,
    load_opta_ranks,
    match_prediction,
    pots_from_season_gf_ga,
    rank_gf_ga_profile,
    sample_match_from_opta_ranks,
    sample_season_gf_ga_for_rank,
)


class OptaGfGaHorizonTest(unittest.TestCase):
    def test_opta_ranks_cover_twenty(self) -> None:
        ranks = load_opta_ranks()
        self.assertEqual(len(ranks), 20)
        self.assertEqual({r["rank"] for r in ranks}, set(range(1, 21)))

    def test_default_is_empirical_rankmean(self) -> None:
        self.assertEqual(DEFAULT_RANK_METHOD, "rankmean")
        prof = rank_gf_ga_profile()
        top = gf_ga_from_rank(1)
        self.assertEqual(top["method"], "rankmean")
        self.assertAlmostEqual(top["gf"], prof[1]["mean_gf"])
        self.assertAlmostEqual(top["ga"], prof[1]["mean_ga"])
        self.assertAlmostEqual(top["sd_gf"], prof[1]["sd_gf"])
        self.assertAlmostEqual(top["sd_ga"], prof[1]["sd_ga"])

    def test_better_rank_higher_gf_lower_ga(self) -> None:
        top = gf_ga_from_rank(1)
        bot = gf_ga_from_rank(20)
        self.assertGreater(top["gf"], bot["gf"])
        self.assertLess(top["ga"], bot["ga"])
        self.assertLess(top["gf_lo"], top["gf_hi"])
        self.assertLess(top["ga_lo"], top["ga_hi"])

    def test_favourite_scores_more_at_home(self) -> None:
        pot_h, pot_a = pots_from_season_gf_ga(80, 35, 35, 70)
        self.assertGreater(pot_h["lam_scored"], pot_a["lam_scored"])
        self.assertGreater(pot_h["p_cs_mkt"], pot_a["p_cs_mkt"])

    def test_arsenal_vs_fulham_centreline(self) -> None:
        pred = match_prediction("Arsenal", "Fulham")
        self.assertEqual(pred["home"], "arsenal")
        self.assertEqual(pred["away"], "fulham")
        self.assertGreater(pred["lam_home"], pred["lam_away"])
        self.assertLessEqual(pred["lam_home_lo"], pred["lam_home"])
        self.assertGreaterEqual(pred["lam_home_hi"], pred["lam_home"])

    def test_rank1_vs_rank20_baseline_rates(self) -> None:
        rng = random.Random(0)
        top = sample_season_gf_ga_for_rank(1, rng, centre_blend=1.0)
        bot = sample_season_gf_ga_for_rank(20, rng, centre_blend=1.0)
        self.assertAlmostEqual(top["gf"], gf_ga_from_rank(1)["gf"])
        self.assertAlmostEqual(bot["gf"], gf_ga_from_rank(20)["gf"])
        self.assertGreater(top["gf"], bot["gf"])
        self.assertLess(top["ga"], bot["ga"])

    def test_sample_match_keeps_rank_ordering(self) -> None:
        # Arsenal (top) home vs bottom side should tend to higher home λ.
        ranks = {r["club_norm"]: float(r["rank"]) for r in load_opta_ranks()}
        top = min(ranks, key=ranks.get)
        bot = max(ranks, key=ranks.get)
        rng = random.Random(1)
        draw = sample_match_from_opta_ranks(top, bot, rng, ranks=ranks, centre_blend=1.0)
        self.assertEqual(draw["home_rank"], ranks[top])
        self.assertEqual(draw["away_rank"], ranks[bot])
        self.assertGreater(draw["lam_home"], draw["lam_away"])

    def test_assessed_league_gf_expanding_live(self) -> None:
        prior = assessed_league_gf(None)
        live = assessed_league_gf(LIVE_E0)
        self.assertEqual(prior["source"], "closed_prior")
        self.assertEqual(live["source"], "expanding_assessed")
        self.assertGreater(live["n_matches"], 0)
        self.assertGreater(live["g_pm"], 0.5)
        self.assertLess(live["g_pm"], 2.5)
        self.assertGreater(live["se_league_gf"], 0.0)
        self.assertLess(live["se_league_gf"], live["sd_league_gf"])

    def test_lambda_uncertainty_propagates(self) -> None:
        prop = lambdas_from_gf_ga(
            80.0,
            35.0,
            40.0,
            60.0,
            league_gf=56.0,
            sd_home_gf=10.0,
            sd_home_ga=5.0,
            sd_away_gf=8.0,
            sd_away_ga=6.0,
            sd_league_gf=2.0,
        )
        self.assertGreater(prop["sd_lam_home"], 0.0)
        self.assertGreater(prop["sd_lam_away"], 0.0)
        self.assertLess(prop["lam_home_lo"], prop["lam_home"])
        self.assertGreater(prop["lam_home_hi"], prop["lam_home"])
        # Leaky away defence raises λ_H vs tight defence.
        tight = lambdas_from_gf_ga(
            80.0, 35.0, 40.0, 40.0, league_gf=56.0, sd_away_ga=6.0
        )
        self.assertGreater(prop["lam_home"], tight["lam_home"])

    def test_match_prediction_exposes_sd(self) -> None:
        pred = match_prediction("Arsenal", "Fulham", e0_fname=LIVE_E0)
        self.assertGreater(pred["sd_lam_home"], 0.0)
        self.assertGreater(pred["sd_lam_away"], 0.0)
        self.assertEqual(pred["league"]["source"], "expanding_assessed")


if __name__ == "__main__":
    unittest.main()
