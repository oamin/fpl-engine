"""The per-90 z-score uses five prior active weeks and does not fill a blank."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.eval.reversion_signal import (
    FORBIDDEN,
    LOOKBACK,
    MIN_MINUTES,
    REQUIRED,
    Z_BAR,
    _readings,
    bucket_interval,
    hurst_rs,
    signal_frame,
    spearman_interval,
)


def _row(
    season: str,
    gw: int,
    player_id: str,
    actual: float,
    *,
    minutes: float = 90.0,
    n_fix: int = 1,
    exp: float = 0.0,
    xp: float = 0.0,
) -> dict[str, object]:
    return {
        "season": season,
        "gw": gw,
        "player_id": player_id,
        "actual": actual,
        "minutes": minutes,
        "n_fix": n_fix,
        "exp": exp,
        "xp": xp,
    }


class SignalTest(unittest.TestCase):
    def test_locks_are_the_pre_registered_ones(self) -> None:
        self.assertEqual(LOOKBACK, 5)
        self.assertEqual(MIN_MINUTES, 30.0)
        self.assertEqual(Z_BAR, 1.5)
        text = "\n".join(REQUIRED)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)

    def test_history_skips_blanks_doubles_and_the_current_week(self) -> None:
        rows = [
            _row("2024-25", 1, "a", 100),
            _row("2024-25", 2, "a", 1),
            _row("2024-25", 3, "a", 2),
            _row("2024-25", 4, "a", 99, minutes=0),
            _row("2024-25", 5, "a", 3),
            _row("2024-25", 6, "a", 4),
            _row("2024-25", 7, "a", 5),
            _row("2024-25", 8, "a", 10),
            _row("2024-25", 9, "a", 50, n_fix=2, minutes=180),
            _row("2024-25", 1, "b", 4),
            _row("2024-25", 2, "b", 4),
            _row("2024-25", 3, "b", 4),
            _row("2024-25", 4, "b", 4),
            _row("2024-25", 5, "b", 4),
            _row("2024-25", 6, "b", 9),
            _row("2023-24", 1, "a", 100),
            _row("2023-24", 2, "a", 100),
            _row("2023-24", 3, "a", 100),
            _row("2023-24", 4, "a", 100),
            _row("2023-24", 5, "a", 100),
            _row("2023-24", 6, "a", 100),
            _row("2024-25", 2, "c", 20, minutes=20),
            _row("2024-25", 3, "c", 2, minutes=30),
        ]
        frame = signal_frame(pd.DataFrame(rows))
        last = frame.loc[(frame["player_id"] == "a") & (frame["season"] == "2024-25") & (frame["gw"] == 8)].iloc[0]
        self.assertAlmostEqual(float(last["mu_xp"]), 3.0)
        self.assertAlmostEqual(float(last["sigma_xp"]), float(np.sqrt(2.5)))
        self.assertAlmostEqual(float(last["z_xp"]), 7.0 / float(np.sqrt(2.5)))
        self.assertTrue(pd.isna(last["fwd_xp_1"]))
        self.assertNotIn(9, set(frame.loc[frame["player_id"] == "a", "gw"]))
        blank = frame.loc[
            (frame["season"] == "2024-25") & (frame["player_id"] == "a") & (frame["gw"] == 4)
        ]
        self.assertTrue(blank.empty)
        short = frame.loc[frame["player_id"] == "c"]
        self.assertEqual(short["gw"].tolist(), [3])
        self.assertAlmostEqual(float(short.iloc[0]["res_xp"]), 6.0)
        flat = frame.loc[(frame["player_id"] == "b") & (frame["gw"] == 6)].iloc[0]
        self.assertTrue(pd.isna(flat["z_xp"]))
        self.assertFalse(np.isfinite(flat["z_xp"]))

    def test_forward_horizon_does_not_skip_an_inactive_week(self) -> None:
        rows = [
            _row("2024-25", 1, "a", 1),
            _row("2024-25", 2, "a", 50, minutes=10),
            _row("2024-25", 3, "a", 7),
        ]
        frame = signal_frame(pd.DataFrame(rows))
        first = frame.loc[frame["gw"] == 1].iloc[0]
        self.assertTrue(pd.isna(first["fwd_xp_1"]))
        self.assertAlmostEqual(float(first["fwd_xp_2"]), 7.0)

    def test_equal_threshold_is_neither_bucket(self) -> None:
        rows = []
        for season, gw in (("2023-24", 1), ("2024-25", 1)):
            rows.extend(
                [
                    _row(season, gw, "low", 0),
                    _row(season, gw, "edge_low", 0),
                    _row(season, gw, "edge_high", 0),
                    _row(season, gw, "high", 0),
                ]
            )
        frame = signal_frame(pd.DataFrame(rows))
        # The history is empty, so replace the z-score with the boundary values.
        frame["z_xp"] = np.tile([-2.0, -1.5, 1.5, 2.0], 2)
        frame["fwd_xp_1"] = np.tile([3.0, 9.0, 9.0, 1.0], 2)
        result = bucket_interval(
            frame, "z_xp", "fwd_xp_1", n_boot=30, seed=0, floor=1
        )
        self.assertAlmostEqual(float(result["mean_oversold"]), 3.0)
        self.assertAlmostEqual(float(result["mean_overbought"]), 1.0)
        self.assertEqual(int(result["n_oversold"]), 2)
        self.assertEqual(int(result["n_overbought"]), 2)

    def test_one_season_leaves_the_interval_undefined(self) -> None:
        rows = []
        for gw in range(1, 21):
            rows.append(_row("2024-25", gw, "a", float(gw)))
            rows.append(_row("2024-25", gw, "b", float(gw + 1)))
        frame = signal_frame(pd.DataFrame(rows))
        frame["z_xp"] = 1.0
        frame["fwd_xp_1"] = 2.0
        result = spearman_interval(frame, "z_xp", "fwd_xp_1", n_boot=10, seed=0, floor=20)
        self.assertIsNone(result["mean"])

    def test_hurst_needs_two_dyadic_lags(self) -> None:
        self.assertTrue(np.isnan(hurst_rs(np.arange(16, dtype=float))))
        self.assertTrue(np.isnan(hurst_rs(np.ones(40))))
        trend = hurst_rs(np.arange(64, dtype=float))
        wave = hurst_rs(np.resize([10.0, -10.0], 64))
        self.assertGreater(float(trend), 0.5)
        self.assertLess(float(wave), 0.5)
        self.assertGreater(float(trend), float(wave))

    def test_readings_follow_the_intervals(self) -> None:
        def row(mean: float, lo: float, hi: float) -> dict[str, float]:
            return {"mean": mean, "lo": lo, "hi": hi, "point": mean}

        summary = {
            "spearman": {f"{name}:{horizon}": row(0.01, -0.01, 0.02) for name in ("exp", "xp") for horizon in (1, 2, 3)},
            "bucket": {f"{name}:{horizon}": {**row(0.1, -0.2, 0.3), "per_season": {}} for name in ("exp", "xp") for horizon in (1, 2, 3)},
            "hurst": {"exp": row(0.64, 0.61, 0.68), "xp": row(0.65, 0.63, 0.68)},
        }
        summary["spearman"]["xp:1"] = row(0.034, 0.018, 0.053)
        summary["bucket"]["xp:2"]["per_season"] = {"2025-26": {"point": 0.62}}
        text = "\n".join(_readings(summary))
        self.assertIn("No bar is cleared.", text)
        self.assertIn("The one-week score_xp correlation stays above zero.", text)
        self.assertIn("Both Hurst intervals stay above 0.5.", text)
        self.assertIn("The 2025-26 horizon-2 point is not the pool.", text)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)
