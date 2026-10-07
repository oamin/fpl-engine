"""Bootstrap unit, unfilled official xP, and the encompassing coefficient."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.encompassing import engine_coefficient
from src.eval.gates import CLOSED_SEASONS, COMPARISONS, assert_reportable, cluster_interval, comparison_key
from src.eval.honest_pool import logscore_rows
from src.eval.official_xp import sheet_fill, unfilled_gameweeks
from src.eval.predictions import write_prediction


def _intervals(*, gws: int = 34, incomplete: dict | None = None) -> dict:
    comps = {}
    for left, right in COMPARISONS:
        counts = {season: gws for season in CLOSED_SEASONS}
        left_out: dict[str, int] = {}
        if incomplete:
            for season, count in incomplete.items():
                counts.pop(season, None)
                left_out[season] = count
        comps[comparison_key(left, right)] = {
            "mean": 0.01,
            "lo": -0.02,
            "hi": 0.04,
            "n_gws": counts,
            "n_rows": 1000,
            "incomplete_seasons": left_out,
        }
    return {
        "seasons": list(CLOSED_SEASONS),
        "min_gws": 20,
        "comparisons": comps,
    }


class ProcedureTest(unittest.TestCase):
    def test_incomplete_season_cannot_also_be_pooled(self) -> None:
        payload = _intervals(incomplete={"2025-26": 7})
        key = comparison_key("score_xp", "score_exp_points")
        payload["comparisons"][key]["n_gws"]["2025-26"] = 7
        with self.assertRaises(RuntimeError):
            assert_reportable({"passed": True, "failures": []}, payload)

    def test_incomplete_count_at_the_floor_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            assert_reportable(
                {"passed": True, "failures": []},
                _intervals(incomplete={"2025-26": 20}),
            )

    def test_short_official_season_is_accepted_when_it_is_not_pooled(self) -> None:
        assert_reportable(
            {"passed": True, "failures": []},
            _intervals(incomplete={"2025-26": 7}),
        )

    def test_bootstrap_resamples_gameweeks(self) -> None:
        shocks = np.random.default_rng(0).normal(0.0, 2.0, size=20)
        result = cluster_interval({"2024-25": shocks}, seasons=("2024-25",), n_boot=1000, seed=0)
        self.assertEqual(result["n_gws"], {"2024-25": 20})
        self.assertGreater(result["hi"] - result["lo"], 0.8)
        self.assertIn(
            "rng.integers(0, arr.size, size=arr.size)",
            inspect.getsource(cluster_interval),
        )

    def test_log_score_emits_one_row_per_gameweek(self) -> None:
        rows = []
        for gw in (5, 6):
            for idx in range(10):
                rows.append(
                    {
                        "season": "2024-25",
                        "gw": gw,
                        "position": "MID",
                        "total_points": float(idx),
                        "score_xp": 1.0 + idx / 10,
                        "score_exp_points": 1.0,
                        "score_official_xp": 2.0,
                    }
                )
        protocol = {
            "gw_start": 5,
            "gw_end": 38,
            "sigma": 3.0,
        }
        out = logscore_rows(pd.DataFrame(rows), protocol)
        self.assertEqual(len(out), 2 * len(COMPARISONS))
        self.assertEqual({row["gw"] for row in out}, {5, 6})

    def test_2025_26_gameweek_7_is_an_unfilled_scrape(self) -> None:
        missing = sheet_fill("2025-26")["unfilled"]
        self.assertIn(7, missing)
        self.assertNotIn(5, missing)
        frame = pd.DataFrame({"GW": [5, 5, 7], "xP": [3.0, 0.0, 0.0]})
        self.assertEqual(unfilled_gameweeks(frame), [7])

    def test_engine_coefficient_recovers_a_known_slope(self) -> None:
        official = np.array([1.0, 2.0, 3.0, 4.0])
        engine = np.array([0.0, 1.0, 0.0, 1.0])
        y = 1.0 + 2.0 * official + 3.0 * engine
        self.assertAlmostEqual(engine_coefficient(y, official, engine), 3.0)

    def test_prediction_file_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "20261007T000000Z.csv"
            frame = pd.DataFrame({"player_id": ["2026-27:1"], "gw": [6], "score": [1.0]})
            write_prediction(path, frame)
            with self.assertRaises(FileExistsError):
                write_prediction(path, frame)


if __name__ == "__main__":
    unittest.main()
