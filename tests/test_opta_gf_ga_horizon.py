"""Opta GF/GA long-horizon pots."""

from __future__ import annotations

import unittest

from src.models.opta_gf_ga_horizon import (
    gf_ga_from_rank,
    load_opta_ranks,
    match_prediction,
    pots_from_season_gf_ga,
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


if __name__ == "__main__":
    unittest.main()
