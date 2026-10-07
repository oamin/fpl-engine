"""Official xP without a pre-deadline capture time is refused."""

from __future__ import annotations

import unittest

import pandas as pd

from src.eval.encompassing import coefficient_rows
from src.eval.gates import assert_reportable
from src.eval.provenance import assert_predeadline_xp, aware_utc, commit_is_before


DEADLINES = {"6": "2026-10-10T10:00:00Z"}


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "player_id": "2026-27:1",
        "gw": 6,
        "official_xp": 4.0,
        "source_field": "ep_next",
        "event_role": "next",
        "captured_at": "2026-10-07T07:00:00Z",
        "deadline": "2026-10-10T10:00:00Z",
    }
    base.update(overrides)
    return base


class ProvenanceTest(unittest.TestCase):
    def test_a_pre_deadline_capture_passes(self) -> None:
        assert_predeadline_xp(pd.DataFrame([_row()]), DEADLINES)

    def test_a_missing_capture_time_is_refused(self) -> None:
        frame = pd.DataFrame([_row()]).drop(columns=["captured_at"])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_a_naive_timestamp_is_refused(self) -> None:
        frame = pd.DataFrame([_row(captured_at="2026-10-07 07:00:00")])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_the_deadline_instant_is_refused(self) -> None:
        frame = pd.DataFrame([_row(captured_at="2026-10-10T10:00:00Z")])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_a_post_deadline_capture_is_refused(self) -> None:
        frame = pd.DataFrame([_row(captured_at="2026-10-11T10:00:00Z")])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_a_forged_deadline_is_refused(self) -> None:
        frame = pd.DataFrame([_row(deadline="2026-10-12T10:00:00Z", captured_at="2026-10-11T10:00:00Z")])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_ep_this_on_the_next_event_is_refused(self) -> None:
        frame = pd.DataFrame([_row(source_field="ep_this", event_role="next")])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_one_bad_row_fails_the_frame(self) -> None:
        frame = pd.DataFrame([_row(), _row(player_id="2026-27:2", captured_at="2026-10-11T00:00:00Z")])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_a_historical_sheet_cannot_be_the_regression(self) -> None:
        frame = pd.DataFrame(
            {
                "gw": [5],
                "total_points": [2.0],
                "score_official_xp": [3.0],
                "score_xp": [1.0],
                "xP": [3.0],
            }
        )
        with self.assertRaises(RuntimeError):
            coefficient_rows(frame, DEADLINES)

    def test_a_renamed_column_is_not_loaded_as_official_xp(self) -> None:
        frame = pd.DataFrame([{"gw": 6, "fpl_xp": 4.0, "captured_at": "2026-10-07T07:00:00Z"}])
        with self.assertRaises(RuntimeError):
            assert_predeadline_xp(frame, DEADLINES)

    def test_a_certified_report_cannot_carry_scraped_xp(self) -> None:
        from src.eval.gates import CLOSED_SEASONS, comparison_key

        key = comparison_key("score_xp", "score_exp_points")
        payload = {
            "seasons": list(CLOSED_SEASONS),
            "min_gws": 20,
            "comparisons": {
                key: {
                    "mean": 0.01,
                    "lo": 0.0,
                    "hi": 0.02,
                    "n_gws": {season: 30 for season in CLOSED_SEASONS},
                },
                "score_xp_minus_score_official_xp": {
                    "mean": -0.05,
                    "lo": -0.06,
                    "hi": -0.04,
                    "n_gws": {season: 30 for season in CLOSED_SEASONS},
                },
            },
        }
        with self.assertRaises(RuntimeError):
            assert_reportable({"passed": True, "failures": []}, payload)

    def test_a_commit_after_the_deadline_does_not_count(self) -> None:
        self.assertFalse(commit_is_before("2026-10-10T10:00:00Z", "2026-10-10T10:00:00Z"))
        self.assertTrue(commit_is_before("2026-10-07T07:06:17+00:00", "2026-10-10T10:00:00Z"))
        with self.assertRaises(ValueError):
            aware_utc("2026-10-07 07:00:00")


if __name__ == "__main__":
    unittest.main()
