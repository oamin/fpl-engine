"""A result is written only with a passed audit and four-season paired intervals."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.gates import (
    CLOSED_SEASONS,
    COMPARISONS,
    assert_reportable,
    cluster_interval,
    comparison_key,
    write_gated_report,
)
from src.models.search_protocol import load_matrix, run_holdout, write_report


def _intervals(*, seasons: list[str] | None = None, gws: int = 34) -> dict:
    comps = {}
    for left, right in COMPARISONS:
        comps[comparison_key(left, right)] = {
            "mean": 0.01,
            "lo": -0.02,
            "hi": 0.04,
            "n_gws": {season: gws for season in CLOSED_SEASONS},
            "n_rows": 1000,
        }
    return {
        "seasons": list(seasons or CLOSED_SEASONS),
        "min_gws": 20,
        "comparisons": comps,
    }


class GateTest(unittest.TestCase):
    def test_refuses_a_failed_audit(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_reportable({"passed": False, "failures": ["leak"]}, _intervals())

    def test_refuses_a_missing_season(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_reportable({"passed": True, "failures": []}, _intervals(seasons=["2025-26"]))

    def test_refuses_the_live_holdout_inside_the_interval(self) -> None:
        seasons = list(CLOSED_SEASONS) + ["2026-27"]
        with self.assertRaises(RuntimeError):
            assert_reportable({"passed": True, "failures": []}, _intervals(seasons=seasons))

    def test_refuses_a_short_season(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_reportable({"passed": True, "failures": []}, _intervals(gws=19))

    def test_writes_only_when_the_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            write_gated_report(
                path,
                {"passed": True, "failures": []},
                _intervals(),
                ["# Note", "", "A paired interval is attached."],
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("as-of audit passed", text)
        self.assertIn("2022-23", text)
        self.assertIn("2025-26", text)

    def test_bootstrap_is_reproducible_and_covers_the_mean(self) -> None:
        rng = np.random.default_rng(1)
        by_season = {season: rng.normal(0.0, 1.0, size=30) for season in CLOSED_SEASONS}
        first = cluster_interval(by_season, n_boot=200, seed=0)
        second = cluster_interval(by_season, n_boot=200, seed=0)
        self.assertEqual(first, second)
        self.assertLess(first["lo"], first["mean"])
        self.assertGreater(first["hi"], first["mean"])
        self.assertEqual(set(first["n_gws"]), set(CLOSED_SEASONS))

    def test_matrix_has_no_season_total_bar(self) -> None:
        spec = load_matrix()
        self.assertNotIn("pass_margin", spec)
        self.assertEqual(spec["holdout_season"], "2026-27")
        self.assertTrue(spec["holdout_frozen"])

    def test_frozen_holdout_is_not_climbed(self) -> None:
        with self.assertRaises(RuntimeError):
            run_holdout({"holdout_season": "2026-27"}, pd.Series(dtype=float), [])

    def test_search_report_does_not_declare_a_winner(self) -> None:
        spec = {
            "screen_season": "2025-26",
            "holdout_season": "2023-24",
            "gw_start": 5,
            "gw_end": 38,
            "kill_gap": 100,
            "advance_n": 2,
        }
        tier = pd.DataFrame(
            columns=[
                "candidate",
                "params",
                "xi_points",
                "delta_vs_xp",
                "spearman",
                "mae",
                "bias",
                "mean_transfers",
                "hits",
            ]
        )
        holdout = pd.DataFrame(
            {
                "season": ["2023-24", "2023-24"],
                "method": ["xp_ft", "winner_ft"],
                "gw": [5, 5],
                "xi_points": [40.0, 80.0],
                "xi_points_cap": [50.0, 90.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "search.md"
            write_report(path, spec, tier, tier, tier, holdout)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn("WINNER", text)
        self.assertIn("not a pass", text)


if __name__ == "__main__":
    unittest.main()
