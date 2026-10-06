"""The early-buy gate. Season totals stay out of this file."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.early_buy import (
    BASELINES,
    EARLY_SCORE_CAP,
    buy_eligible,
    cap_decision_score,
    clip_horizon_steps,
    decision_kept,
    pool_bar,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": ["debut", "cameo", "none", "regular", "tie_a", "tie_b"],
            "n_prior": [1, 1, 0, 4, 2, 2],
            "xmi": [50.0, 44.0, 90.0, 90.0, 80.0, 80.0],
            "score_xp": [12.0, 4.0, 9.0, 9.0, 3.0, 3.0],
        }
    )


class EarlyBuyRuleTest(unittest.TestCase):
    def test_the_cap_is_six(self) -> None:
        self.assertEqual(EARLY_SCORE_CAP, 6.0)

    def test_one_and_two_appearances_are_capped(self) -> None:
        capped = cap_decision_score(_frame())
        by_id = dict(zip(capped["player_id"], capped["score_xp"], strict=True))
        self.assertEqual(by_id["debut"], 6.0)
        self.assertEqual(by_id["regular"], 9.0)
        self.assertEqual(by_id["tie_a"], by_id["tie_b"])

    def test_fifty_minutes_can_be_bought_and_a_blank_history_cannot(self) -> None:
        mask = buy_eligible(_frame())
        allowed = set(_frame().loc[mask, "player_id"])
        self.assertEqual(allowed, {"debut", "regular", "tie_a", "tie_b"})

    def test_a_later_week_above_the_cap_is_clipped_for_an_early_player(self) -> None:
        steps = {2: {"debut": 9.0, "regular": 9.0}, 3: {"debut": 8.0, "regular": 7.0}}
        clipped = clip_horizon_steps(steps, {"debut"})
        self.assertEqual(clipped[3]["debut"], 6.0)
        self.assertEqual(clipped[3]["regular"], 7.0)

    def test_the_bar_needs_the_pool_three_seasons_and_no_collapse(self) -> None:
        self.assertEqual(pool_bar(), 7948.0)
        passing = {season: base + 10.0 for season, base in BASELINES.items()}
        self.assertTrue(decision_kept(passing)["kept"])
        short = dict(passing)
        short["2022-23"] = BASELINES["2022-23"] - 1.0
        short["2023-24"] = BASELINES["2023-24"] - 1.0
        self.assertFalse(decision_kept(short)["kept"])
        collapsed = dict(passing)
        collapsed["2024-25"] = BASELINES["2024-25"] - 31.0
        collapsed["2025-26"] = BASELINES["2025-26"] + 100.0
        self.assertFalse(decision_kept(collapsed)["kept"])
        self.assertFalse(decision_kept({"2022-23": 1800.0})["kept"])
