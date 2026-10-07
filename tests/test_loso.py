"""Leave-one-season-out rules, locked before the stored contrasts are re-read."""

from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src.eval.decision import greedy_step
from src.eval.decision_spec import LOSO_BOOTSTRAP, LOSO_CONDITIONAL_FLOOR, LOSO_FLOOR, LOSO_SEED
from src.eval.loso import (
    CONTRASTS,
    FORBIDDEN,
    REQUIRED_SENTENCES,
    contrast_keys,
    leave_one_out,
    refuse_replacement,
    sign_differs,
)


def _rows(spec: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"season": season, "gw": gw, "gap": value} for season, count, value in spec for gw in range(count)],
    )


class LeaveOneSeasonOutTest(unittest.TestCase):
    def test_holding_out_a_season_removes_only_that_season(self) -> None:
        frame = _rows([("2022-23", 20, 1.0), ("2023-24", 20, 2.0), ("2024-25", 20, 4.0)])
        before = frame.copy()
        published = {"mean": -1.6667}
        fold = leave_one_out(frame, "gap", holdout="2022-23", n_boot=20, seed=0)
        self.assertAlmostEqual(float(fold["mean"]), 3.0)
        self.assertNotIn("2022-23", fold["n_gws"])
        self.assertEqual(fold["n_gws"]["2023-24"], 20)
        self.assertEqual(fold["n_gws"]["2024-25"], 20)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(published["mean"], -1.6667)

    def test_the_pool_is_the_concatenated_weeks(self) -> None:
        frame = _rows([("2023-24", 20, 0.0), ("2024-25", 40, 3.0)])
        fold = leave_one_out(frame, "gap", holdout="2022-23", n_boot=20, seed=0)
        self.assertAlmostEqual(float(fold["mean"]), 2.0)
        self.assertNotAlmostEqual(float(fold["mean"]), 1.5)

    def test_a_short_season_is_not_pooled(self) -> None:
        frame = _rows([("2023-24", 20, 1.0), ("2024-25", 20, 3.0), ("2025-26", 19, 100.0)])
        fold = leave_one_out(frame, "gap", holdout="2023-24", minimum=20, n_boot=20, seed=0)
        self.assertTrue(fold["undefined"])
        self.assertIsNone(fold["mean"])
        self.assertEqual(fold["incomplete_seasons"]["2025-26"], 19)
        kept = leave_one_out(frame, "gap", holdout="2025-26", minimum=20, n_boot=20, seed=0)
        self.assertAlmostEqual(float(kept["mean"]), 2.0)
        self.assertNotIn("2025-26", kept["n_gws"])

    def test_one_remaining_season_is_not_identified(self) -> None:
        frame = _rows([("2024-25", 20, 1.0)])
        fold = leave_one_out(frame, "gap", holdout="2022-23", n_boot=20, seed=0)
        self.assertTrue(fold["undefined"])
        self.assertIsNone(fold["mean"])

    def test_the_conditional_floor_keeps_a_season_under_twenty_weeks(self) -> None:
        frame = _rows([("2023-24", 5, 1.0), ("2024-25", 19, 1.0), ("2025-26", 4, 100.0)])
        waived = leave_one_out(
            frame, "gap", holdout="2022-23", minimum=LOSO_CONDITIONAL_FLOOR, n_boot=20, seed=0
        )
        self.assertFalse(waived["undefined"])
        self.assertAlmostEqual(float(waived["mean"]), 1.0)
        self.assertIn("2024-25", waived["n_gws"])
        self.assertEqual(waived["incomplete_seasons"]["2025-26"], 4)
        strict = leave_one_out(frame, "gap", holdout="2022-23", minimum=LOSO_FLOOR, n_boot=20, seed=0)
        self.assertTrue(strict["undefined"])

    def test_the_draw_is_locked_and_the_seed_moves_the_interval(self) -> None:
        self.assertEqual(LOSO_BOOTSTRAP, 1000)
        self.assertEqual(LOSO_SEED, 0)
        self.assertEqual(LOSO_FLOOR, 20)
        frame = _rows([("2023-24", 20, 0.0), ("2024-25", 20, 1.0)])
        frame.loc[frame.index % 2 == 0, "gap"] = -2.0
        left = leave_one_out(frame, "gap", holdout="2022-23", n_boot=40, seed=0)
        right = leave_one_out(frame, "gap", holdout="2022-23", n_boot=40, seed=1)
        self.assertAlmostEqual(float(left["mean"]), float(right["mean"]))
        self.assertNotEqual(round(float(left["lo"]), 6), round(float(right["lo"]), 6))

    def test_replacement_and_a_sign_change_are_labelled(self) -> None:
        with self.assertRaises(RuntimeError):
            refuse_replacement(True)
        refuse_replacement(False)
        self.assertTrue(sign_differs(2.0, -1.0))
        self.assertFalse(sign_differs(-0.2, -1.0))
        self.assertFalse(sign_differs(None, -1.0))

    def test_greedy_still_has_no_hurdle(self) -> None:
        source = inspect.getsource(greedy_step)
        self.assertNotIn("loso_thresholds", source)
        self.assertNotIn("hit_threshold", source)

    def test_the_catalogue_does_not_rebuild_a_season_or_concordance(self) -> None:
        import src.eval.loso as loso

        source = inspect.getsource(loso)
        self.assertNotIn("build_season", source)
        self.assertNotIn("honest_pool", source)
        self.assertNotIn("c_xp_minus_c_exp", source)
        self.assertEqual(list(contrast_keys()), [str(item["key"]) for item in CONTRASTS])
        straw = [str(item["key"]) for item in CONTRASTS if item["strawman"]]
        self.assertEqual(straw, ["placebo:greedy_xp_minus_hold_xp"])
        self.assertTrue(all("c_xp" not in key for key in contrast_keys()))

    def test_the_required_sentences_avoid_a_winner(self) -> None:
        text = "\n".join(REQUIRED_SENTENCES)
        for banned in FORBIDDEN:
            self.assertNotIn(banned, text)
        self.assertIn("do not replace the four-season results", text)
        self.assertIn("`ep_next`", text)
