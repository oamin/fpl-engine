"""Crowd-context regression. The score is not rewritten."""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd

from src.models.stage_47_crowd_context import (
    CrowdJoinError,
    aggregate_player_weeks,
    attach_market,
    center_ownership,
    dedupe_market,
    eligible_mask,
    log_transfer_flow,
    loso,
    ols,
    ownership_share,
    qualifies,
    within_week_z,
)


class FlowTests(unittest.TestCase):
    def test_flow_is_the_difference_of_logs(self) -> None:
        flow = log_transfer_flow(np.array([100.0, 1.0]), np.array([1.0, 100.0]))
        self.assertAlmostEqual(flow[0], float(np.log1p(100) - np.log1p(1)))
        self.assertLess(flow[1], 0.0)
        self.assertTrue(np.isfinite(log_transfer_flow(np.array([10.0]), np.array([60.0]))[0]))

    def test_negative_counts_abort(self) -> None:
        with self.assertRaises(CrowdJoinError):
            log_transfer_flow(np.array([-1.0]), np.array([1.0]))


class OwnershipTests(unittest.TestCase):
    def test_share_matches_managers(self) -> None:
        # 10 managers, 15 slots, 150 selected-counts. Eight owners is 80%.
        selected = pd.Series([8.0, 142.0])
        gw = pd.Series([1, 1])
        share = ownership_share(selected, gw)
        self.assertAlmostEqual(float(share.iloc[0]), 0.8)


class DoubleTests(unittest.TestCase):
    def test_first_fixture_anchors_the_gate_and_points_sum(self) -> None:
        feat = pd.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "element": ["7", "7"],
                "gw": [10, 10],
                "date": ["2024-11-02", "2024-11-03"],
                "total_points": [2.0, 6.0],
                "score_xp": [3.0, 4.0],
                "xmi": [40.0, 90.0],
                "n_prior": [5, 6],
            }
        )
        week = aggregate_player_weeks(feat)
        self.assertEqual(len(week), 1)
        self.assertEqual(int(week["n_fixtures"].iloc[0]), 2)
        self.assertAlmostEqual(float(week["total_points"].iloc[0]), 8.0)
        self.assertAlmostEqual(float(week["score_xp"].iloc[0]), 7.0)
        self.assertAlmostEqual(float(week["xmi"].iloc[0]), 40.0)
        self.assertAlmostEqual(float(week["n_prior"].iloc[0]), 5.0)
        self.assertFalse(bool(eligible_mask(week).iloc[0]))


class MergeTests(unittest.TestCase):
    def test_a_short_join_aborts(self) -> None:
        eligible = pd.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "element": ["1", "2"],
                "gw": [6, 6],
                "total_points": [2.0, 3.0],
                "score_xp": [2.0, 3.0],
                "xmi": [90.0, 90.0],
                "n_prior": [4, 4],
                "n_fixtures": [1, 1],
                "date": ["2024-09-01", "2024-09-01"],
            }
        )
        market = _market(["1"], gw=6)
        with self.assertRaises(CrowdJoinError):
            attach_market(eligible, market)

    def test_a_complete_join_keeps_raw_counts(self) -> None:
        eligible = pd.DataFrame(
            {
                "season": ["2024-25"],
                "element": ["1"],
                "gw": [6],
                "total_points": [2.0],
                "score_xp": [2.0],
                "xmi": [90.0],
                "n_prior": [4],
                "n_fixtures": [1],
                "date": ["2024-09-01"],
            }
        )
        sample, rates = attach_market(eligible, _market(["1"], gw=6))
        self.assertAlmostEqual(rates["2024-25"], 1.0)
        self.assertIn("transfers_in", sample.columns)
        self.assertIn("transfers_out", sample.columns)
        self.assertIn("log_transfer_flow", sample.columns)


class ScaleTests(unittest.TestCase):
    def test_within_week_z_is_centered(self) -> None:
        flow = pd.Series([1.0, 2.0, 3.0])
        season = pd.Series(["2024-25"] * 3)
        gw = pd.Series([6, 6, 6])
        z = within_week_z(flow, season, gw)
        self.assertAlmostEqual(float(z.mean()), 0.0, places=8)
        self.assertAlmostEqual(float(z.std(ddof=0)), 1.0, places=8)

    def test_holdout_uses_the_training_mean(self) -> None:
        train = pd.DataFrame({"own_10": [1.0, 3.0], "volume_z": [0.0, 1.0]})
        test = pd.DataFrame({"own_10": [5.0], "volume_z": [0.5]})
        _, held, mu = center_ownership(train, test)
        self.assertAlmostEqual(mu, 2.0)
        self.assertAlmostEqual(float(held["own_10_c"].iloc[0]), 3.0)


class ReadingTests(unittest.TestCase):
    def test_one_season_without_a_gain_fails(self) -> None:
        self.assertFalse(qualifies([0.05, 0.05, 0.05, 0.0]))

    def test_a_small_mean_fails(self) -> None:
        self.assertFalse(qualifies([0.01, 0.01, 0.01, 0.01]))

    def test_four_gains_at_the_bar_pass(self) -> None:
        self.assertTrue(qualifies([0.02, 0.03, 0.02, 0.02]))


class FitTests(unittest.TestCase):
    def test_loso_recovers_a_flow_that_moves_points(self) -> None:
        rows = []
        rng = np.random.default_rng(7)
        for season_i, season in enumerate(("2022-23", "2023-24", "2024-25", "2025-26")):
            for i in range(80):
                volume = float(rng.normal())
                own = 2.0 + float(rng.normal())
                score = 4.0 + 0.2 * i / 80.0
                points = score + 0.8 * volume + float(rng.normal(scale=0.05))
                rows.append(
                    {
                        "season": season,
                        "gw": 6 + (i % 10),
                        "score_xp": score,
                        "volume_z": volume,
                        "own_10": own,
                        "total_points": points,
                        "season_i": season_i,
                    }
                )
        result = loso(pd.DataFrame(rows))
        self.assertTrue(result["additive_qualifies"])
        for fold in result["folds"]:
            self.assertGreater(fold["c"], 0.4)
            self.assertGreater(fold["delta_additive"], 0.02)

    def test_ols_standard_error_is_finite(self) -> None:
        x = np.column_stack([np.ones(20), np.arange(20, dtype=float)])
        y = 1.0 + 2.0 * x[:, 1]
        beta, se = ols(x, y)
        self.assertAlmostEqual(float(beta[1]), 2.0)
        self.assertTrue(np.isfinite(se[1]))


class IsolationTests(unittest.TestCase):
    def test_the_module_does_not_call_the_benchmark_writer(self) -> None:
        import src.models.stage_47_crowd_context as mod

        source = inspect.getsource(mod)
        self.assertNotIn("benchmark", source)
        self.assertNotIn("live_benchmark", source)


def _market(elements: list[str], gw: int) -> pd.DataFrame:
    # Two selected counts that sum with the rest of a fake slate.
    rows = []
    for i, element in enumerate(elements):
        rows.append(
            {
                "season": "2024-25",
                "element": element,
                "gw": gw,
                "transfers_in": 1000.0 + i,
                "transfers_out": 100.0,
                "selected": 80_000.0,
                "ow": 0.2,
            }
        )
    return pd.DataFrame(rows)


class MarketDedupeTests(unittest.TestCase):
    def test_a_double_keeps_the_earlier_kickoff(self) -> None:
        raw = pd.DataFrame(
            {
                "element": [7, 7, 8],
                "GW": [12, 12, 12],
                "kickoff_time": ["2024-12-04T20:00:00Z", "2024-12-03T19:00:00Z", "2024-12-03T19:00:00Z"],
                "transfers_in": [50, 10, 3],
                "transfers_out": [5, 1, 1],
                "selected": [100, 40, 10],
            }
        )
        market = dedupe_market(raw, "2024-25")
        kept = market.loc[market["element"] == "7"].iloc[0]
        self.assertAlmostEqual(float(kept["transfers_in"]), 10.0)
        self.assertAlmostEqual(float(kept["selected"]), 40.0)


if __name__ == "__main__":
    unittest.main()
