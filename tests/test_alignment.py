"""score_xp against scraped xP on one gameweek is a join check, not a benchmark."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.alignment import alignment_report


def _frame(official: list[float], score: list[float] | None = None) -> pd.DataFrame:
    score = score if score is not None else official
    rows = []
    for index, (xp, scraped) in enumerate(zip(score, official, strict=True), start=1):
        rows.append(
            {
                "player_id": str(index),
                "fixture_id": f"f{index}",
                "gw": 10,
                "position": "MID",
                "score_xp": xp,
                "official_xp": scraped,
                "total_points": scraped,
            }
        )
    return pd.DataFrame(rows)


class AlignmentTest(unittest.TestCase):
    def test_matching_forecasts_line_up(self) -> None:
        report = alignment_report(_frame([1.0, 2.0, 4.0, 8.0]), 10)
        report.pop("_paired")
        self.assertEqual(report["duplicate_player_fixture"], 0)
        self.assertTrue(report["same_row"])
        self.assertTrue(report["positions_ok"])
        self.assertTrue(report["gw_ok"])
        self.assertAlmostEqual(report["spearman_score_vs_official"], 1.0)
        self.assertFalse(report["sign_flip_fits_better"])
        self.assertGreater(report["spearman_score_vs_official"], 0.0)

    def test_a_sign_flip_is_visible(self) -> None:
        report = alignment_report(_frame([1.0, 2.0, 4.0, 8.0], [8.0, 4.0, 2.0, 1.0]), 10)
        self.assertLess(report["spearman_score_vs_official"], 0.0)
        self.assertTrue(report["sign_flip_fits_better"])
        self.assertAlmostEqual(report["spearman_negated_score_vs_official"], 1.0)

    def test_a_duplicated_fixture_key_is_flagged(self) -> None:
        frame = _frame([1.0, 2.0, 3.0, 4.0])
        extra = frame.iloc[[0]].copy()
        report = alignment_report(pd.concat([frame, extra], ignore_index=True), 10)
        self.assertGreater(report["duplicate_player_fixture"], 0)
        self.assertFalse(report["same_row"])

    def test_a_double_gameweek_stays_one_player(self) -> None:
        frame = _frame([1.0, 2.0, 3.0, 4.0])
        second = frame.iloc[[0]].copy()
        second["fixture_id"] = "f1b"
        second["official_xp"] = 3.0
        second["score_xp"] = 5.0
        second["total_points"] = 6.0
        report = alignment_report(pd.concat([frame, second], ignore_index=True), 10)
        self.assertEqual(report["duplicate_player_fixture"], 0)
        self.assertEqual(report["max_rows_per_player"], 2)
        self.assertEqual(report["n_players"], 4)
        paired = report["_paired"]
        row = paired.loc[paired["player_id"].astype(str) == "1"].iloc[0]
        self.assertAlmostEqual(float(row["official_xp"]), 4.0)
        self.assertAlmostEqual(float(row["score_xp"]), 6.0)


if __name__ == "__main__":
    unittest.main()
