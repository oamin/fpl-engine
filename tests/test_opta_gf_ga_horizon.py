"""Opta GF/GA long-horizon pots."""

from __future__ import annotations

import random
import unittest

from src.models.opta_gf_ga_horizon import (
    gf_ga_from_rank,
    load_opta_ranks,
    match_prediction,
    pots_from_season_gf_ga,
    sample_match_from_opta_ranks,
    sample_season_gf_ga_for_rank,
)


class OptaGfGaHorizonTest(unittest.TestCase):
    def test_opta_ranks_cover_twenty(self) -> None:
        ranks = load_opta_ranks()
        self.assertEqual(len(ranks), 20)
        self.assertEqual({r["rank"] for r in ranks}, set(range(1, 21)))

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


if __name__ == "__main__":
    unittest.main()
