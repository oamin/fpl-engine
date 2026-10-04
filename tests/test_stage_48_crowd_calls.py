"""Close-call crowd count. The published fit is not re-estimated."""

from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src.models.stage_48_crowd_calls import (
    LOCKED_FITS,
    CrowdCallError,
    locked_fits_from_csv,
    resolve_band,
    stays_open,
)


def _row(element: str, score: float, volume: float, own: float, points: float) -> dict:
    return {
        "season": "2024-25",
        "gw": 10,
        "position": "MID",
        "element": element,
        "score_xp": score,
        "volume_z": volume,
        "own_10": own,
        "total_points": points,
    }


class BandTests(unittest.TestCase):
    def test_an_equal_crowd_adjustment_is_not_a_flip(self) -> None:
        group = pd.DataFrame(
            [
                _row("2", 5.00, 0.0, 1.0, 2.0),
                _row("9", 4.80, 0.0, 1.0, 8.0),
            ]
        )
        call = resolve_band(group, c=1.0, d=1.0, own_mean=0.0)
        self.assertTrue(call["close"])
        self.assertFalse(call["flip"])
        self.assertEqual(call["chosen"], "2")

    def test_a_strict_preference_inside_the_band_flips(self) -> None:
        group = pd.DataFrame(
            [
                _row("2", 5.00, 0.0, 1.0, 2.0),
                _row("9", 4.80, 2.0, 1.0, 8.0),
                _row("3", 4.70, 5.0, 1.0, 1.0),
            ]
        )
        call = resolve_band(group, c=1.0, d=0.0, own_mean=0.0)
        self.assertEqual(call["chosen"], "9")
        self.assertTrue(call["flip"])
        self.assertTrue(call["decisive"])
        self.assertTrue(call["crowd_win"])
        self.assertAlmostEqual(call["point_gap"], 6.0)

    def test_a_player_outside_the_band_cannot_flip_the_leader(self) -> None:
        group = pd.DataFrame(
            [
                _row("2", 5.00, 0.0, 1.0, 4.0),
                _row("9", 4.70, 5.0, 1.0, 12.0),
            ]
        )
        call = resolve_band(group, c=1.0, d=0.0, own_mean=0.0)
        self.assertFalse(call["close"])
        self.assertFalse(call["flip"])

    def test_a_points_tie_is_not_decisive(self) -> None:
        group = pd.DataFrame(
            [
                _row("2", 5.00, 0.0, 1.0, 6.0),
                _row("9", 4.90, 1.0, 1.0, 6.0),
            ]
        )
        call = resolve_band(group, c=1.0, d=0.0, own_mean=0.0)
        self.assertTrue(call["flip"])
        self.assertFalse(call["decisive"])
        self.assertAlmostEqual(call["point_gap"], 0.0)

    def test_the_lower_element_id_is_the_leader_on_a_score_tie(self) -> None:
        group = pd.DataFrame(
            [
                _row("9", 5.00, 1.0, 1.0, 1.0),
                _row("2", 5.00, 0.0, 1.0, 4.0),
            ]
        )
        call = resolve_band(group, c=1.0, d=0.0, own_mean=0.0)
        self.assertEqual(call["leader"], "2")
        self.assertEqual(call["chosen"], "9")


class ReadingTests(unittest.TestCase):
    def test_one_short_season_closes_the_question(self) -> None:
        row = {
            "n_decisive": 30,
            "win_rate": 0.70,
            "mean_flip_gap": 0.40,
            "slot_gain": 0.05,
        }
        seasons = [dict(row) for _ in range(4)]
        seasons[2]["win_rate"] = 0.59
        self.assertFalse(stays_open(seasons))

    def test_the_four_bars_together_leave_it_open(self) -> None:
        row = {
            "n_decisive": 30,
            "win_rate": 0.60,
            "mean_flip_gap": 0.20,
            "slot_gain": 0.030,
        }
        self.assertTrue(stays_open([dict(row) for _ in range(4)]))


class LockTests(unittest.TestCase):
    def test_the_published_fit_is_the_lock(self) -> None:
        fits = locked_fits_from_csv(
            __import__("pathlib").Path("data/processed/stage_47_crowd_context.csv")
        )
        self.assertEqual(fits["2024-25"]["c"], LOCKED_FITS["2024-25"]["c"])

    def test_the_module_does_not_refit(self) -> None:
        import src.models.stage_48_crowd_calls as mod

        source = inspect.getsource(mod)
        self.assertNotIn("loso", source)
        self.assertNotIn("lstsq", source)
        self.assertNotIn("benchmark", source)


class MismatchTests(unittest.TestCase):
    def test_a_changed_coefficient_file_aborts(self) -> None:
        import tempfile
        from pathlib import Path

        text = Path("data/processed/stage_47_crowd_context.csv").read_text(encoding="utf-8")
        text = text.replace("0.2123620032248541", "0.99", 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stage_47_crowd_context.csv"
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(CrowdCallError):
                locked_fits_from_csv(path)


if __name__ == "__main__":
    unittest.main()
