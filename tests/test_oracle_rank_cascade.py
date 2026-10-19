"""Oracle-rank cascade helpers."""

from __future__ import annotations

import unittest

from src.models.opta_gf_ga_horizon import (
    fit_rank_ols_excluding,
    gf_ga_from_rank,
    rank_gf_ga_profile_excluding,
    season_final_ranks,
)
from src.eval.oracle_rank_cascade import flat_ranks, prior_season_ranks


class OracleRankCascadeTest(unittest.TestCase):
    def test_prior_ols_excludes_target(self) -> None:
        ols = fit_rank_ols_excluding("E0_2526.csv")
        self.assertEqual(ols["exclude"], "E0_2526.csv")
        self.assertNotIn("E0_2526.csv", ols["files"])
        self.assertEqual(len(ols["files"]), 3)
        self.assertEqual(ols["n_rows"], 60)

    def test_oracle_ranks_cover_twenty(self) -> None:
        ranks = season_final_ranks("E0_2526.csv")
        self.assertEqual(len(ranks), 20)
        self.assertEqual(set(ranks.values()), set(float(i) for i in range(1, 21)))

    def test_prior_assigns_promoted(self) -> None:
        oracle = set(season_final_ranks("E0_2526.csv"))
        prior = prior_season_ranks(oracle)
        self.assertEqual(set(prior), oracle)
        promoted = {"sunderland", "leeds", "burnley"}
        self.assertTrue(promoted <= set(prior))
        self.assertEqual(sorted(prior[c] for c in promoted), [18.0, 19.0, 20.0])

    def test_flat_and_better_rank_from_prior_ols(self) -> None:
        ols = fit_rank_ols_excluding("E0_2526.csv")
        prof = rank_gf_ga_profile_excluding("E0_2526.csv")
        top = gf_ga_from_rank(
            1,
            profile=prof,
            ols_gf=tuple(ols["ols_gf"]),
            ols_ga=tuple(ols["ols_ga"]),
            mae_gf=ols["mae_gf"],
            mae_ga=ols["mae_ga"],
        )
        bot = gf_ga_from_rank(
            20,
            profile=prof,
            ols_gf=tuple(ols["ols_gf"]),
            ols_ga=tuple(ols["ols_ga"]),
            mae_gf=ols["mae_gf"],
            mae_ga=ols["mae_ga"],
        )
        self.assertGreater(top["gf"], bot["gf"])
        self.assertLess(top["ga"], bot["ga"])
        self.assertEqual(len(flat_ranks({"a", "b"})), 2)


if __name__ == "__main__":
    unittest.main()
