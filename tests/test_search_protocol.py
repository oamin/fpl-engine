"""Selector and formulas. The season climb is not run here."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.search_protocol import (
    apply_score,
    iter_grid,
    score_diagnostics,
    select_architectures,
)


class SearchProtocolTest(unittest.TestCase):
    def test_formulas(self) -> None:
        df = pd.DataFrame(
            {
                "score_xp": [4.0, 8.0],
                "sigma_xp": [1.0, 4.0],
                "score_exp_points": [2.0, 6.0],
                "ow": [0.2, 0.0],
                "score_xmi": [1.0, 2.0],
            }
        )
        self.assertAlmostEqual(float(apply_score(df, "linear_risk", {"lambda": 0.25}).iloc[0]), 3.75)
        self.assertAlmostEqual(float(apply_score(df, "sharpe", {"eps": 1.0}).iloc[0]), 2.0)
        self.assertAlmostEqual(float(apply_score(df, "ownership", {"weight": 0.5}).iloc[0]), 3.6)
        self.assertAlmostEqual(float(apply_score(df, "blend", {"alpha": 0.5}).iloc[1]), 7.0)
        # sigma 4 is 1 above the floor of 3, so the penalty is lambda * 1.
        self.assertAlmostEqual(float(apply_score(df, "tail_risk", {"lambda": 1.0}).iloc[1]), 7.0)

    def test_grid_list(self) -> None:
        grid = iter_grid({"id": "linear_risk", "params_grid": [{"lambda": 0.1}, {"lambda": 0.5}]})
        self.assertEqual(grid, [{"lambda": 0.1}, {"lambda": 0.5}])

    def test_top_two_skip_baseline_and_clear_losses(self) -> None:
        tier1 = pd.DataFrame(
            [
                {"candidate": "xp", "family": "baseline", "delta_vs_xp": 0.0, "params": "a"},
                {"candidate": "blend", "family": "player_score", "delta_vs_xp": -10.0, "params": "a"},
                {"candidate": "blend", "family": "player_score", "delta_vs_xp": -40.0, "params": "b"},
                {"candidate": "exp_points", "family": "player_score", "delta_vs_xp": -20.0, "params": "a"},
                {"candidate": "sharpe", "family": "player_score", "delta_vs_xp": -120.0, "params": "a"},
                {"candidate": "risk_inside_value", "family": "transfer_value", "delta_vs_xp": 90.0, "params": "a"},
                {"candidate": "price", "family": "player_score", "delta_vs_xp": -5.0, "params": "a"},
            ]
        )
        chosen = select_architectures(tier1, advance_n=2, kill_gap=100)
        self.assertEqual(chosen["candidate"].tolist(), ["price", "blend"])

    def test_diagnostics_do_not_require_a_climb(self) -> None:
        feat = pd.DataFrame(
            {
                "gw": [5, 5, 6, 6],
                "eligible": [True, True, True, True],
                "score": [1.0, 3.0, 2.0, 4.0],
                "total_points": [1.0, 2.0, 2.0, 5.0],
            }
        )
        diag = score_diagnostics(feat, [5, 6], "score")
        self.assertGreater(diag["spearman"], 0.9)
        self.assertAlmostEqual(diag["bias"], -0.0)
        self.assertEqual(diag["n"], 4.0)


if __name__ == "__main__":
    unittest.main()
