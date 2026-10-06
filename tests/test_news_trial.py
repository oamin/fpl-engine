"""The loose tag trial does not zero a week the note does not cover."""

from __future__ import annotations

import unittest

import pandas as pd

from src.live.news_trial import apply_tag_scores, load_writes


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("2026-27:140", 2, 90.0, 3.61),
            _row("2026-27:140", 3, 90.0, 3.61, goals=0.2),
            _row("2026-27:165", 5, 90.0, 5.30, position="FWD", goals=2.4),
            _row("2026-27:28", 3, 90.0, 4.10),
            _row("2026-27:9", 1, 80.0, 4.50, position="DEF"),
        ]
    )


def _row(
    player_id: str,
    gw: int,
    xmi: float,
    score: float,
    *,
    position: str = "GKP",
    goals: float = 0.1,
) -> dict:
    return {
        "player_id": player_id,
        "gw": gw,
        "xmi": xmi,
        "score_xp": score,
        "eligible": True,
        "position": position,
        "xp_goals": goals,
        "xp_assists": 0.0,
        "p_cs_mkt": 0.3,
        "exp_defcon_hit": 0.2,
        "expected_saves": 2.0,
        "lam_conceded": 1.0,
        "share_xG": 0.05,
        "share_xA": 0.02,
        "fwd_level_add": 0.0,
    }


class OverlayTest(unittest.TestCase):
    def test_a_later_zero_does_not_touch_the_earlier_week(self) -> None:
        tagged, counts = apply_tag_scores(_frame(), {(3, 140): 0.0})
        earlier = tagged.loc[tagged["gw"] == 2].iloc[0]
        later = tagged.loc[tagged["gw"] == 3].iloc[0]
        self.assertEqual(float(earlier["score_xp"]), 3.61)
        self.assertEqual(float(earlier["xmi"]), 90.0)
        self.assertEqual(float(later["score_xp"]), 0.0)
        self.assertEqual(float(later["share_xG"]), 0.0)
        self.assertEqual(float(later["share_xA"]), 0.0)
        self.assertFalse(bool(later["eligible"]))
        self.assertEqual(counts["zeroed"], 1)

    def test_a_doubtful_flag_is_not_zeroed(self) -> None:
        tagged, _counts = apply_tag_scores(_frame(), {(5, 165): 67.5})
        row = tagged.loc[tagged["player_id"] == "2026-27:165"].iloc[0]
        self.assertGreater(float(row["score_xp"]), 0.0)
        self.assertEqual(float(row["share_xG"]), 0.05)
        self.assertTrue(bool(row["eligible"]))

    def test_a_firm_starter_at_the_same_minutes_is_left_alone(self) -> None:
        tagged, counts = apply_tag_scores(_frame(), {(3, 28): 90.0})
        row = tagged.loc[tagged["player_id"] == "2026-27:28"].iloc[0]
        self.assertEqual(float(row["score_xp"]), 4.10)
        self.assertEqual(counts["changed"], 0)

    def test_an_untagged_row_does_not_move(self) -> None:
        original = _frame()
        tagged, _counts = apply_tag_scores(original, {(3, 140): 0.0, (5, 165): 67.5})
        untouched = tagged["player_id"] == "2026-27:9"
        self.assertEqual(
            float(tagged.loc[untouched, "score_xp"].iloc[0]),
            float(original.loc[untouched, "score_xp"].iloc[0]),
        )

    def test_the_sheet_leaves_gameweek_2_open(self) -> None:
        writes = load_writes()
        self.assertNotIn((2, 140), writes)
        self.assertEqual(writes[(3, 140)], 0.0)
        self.assertGreater(writes[(5, 165)], 0.0)


if __name__ == "__main__":
    unittest.main()
