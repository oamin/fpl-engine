"""Fresh-open helper: crowd template → held 15, open week holds."""

from __future__ import annotations

import unittest

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, SQUAD_SIZE
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer3_week import solve_week_transfers
from src.strategy.layer4_mupost_fresh_open import (
    held_from_template,
    inject_template_into_pool,
    load_template_opening,
)
from src.strategy.shrink_climb import SeasonStore


class FreshOpen(unittest.TestCase):
    def test_template_loads_and_holds(self) -> None:
        season = "2025-26"
        opening = load_template_opening(season)
        self.assertEqual(len(opening), SQUAD_SIZE)
        held = held_from_template(opening, ft=0)
        self.assertEqual(len(held.player_ids), SQUAD_SIZE)
        self.assertEqual(held.ft, 0)
        self.assertLessEqual(sum(held.purchase.values()) + held.bank, BUDGET_TENTHS)

        v2 = pd.read_csv("data/processed/score_xp_v2_closed.csv")
        store = SeasonStore(season, v2)
        gw = 8
        pool = store.pool(gw)
        moments = store.moments_asof(gw)
        work, _ = attach_mu_post(
            pool,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            out_col="mu_post",
        )
        work = inject_template_into_pool(
            work, opening, season=season, gw=gw, prices=store.prices
        )
        plan = solve_week_transfers(work, held, mu_col="mu_post", max_hits=0, time_limit=30.0)
        self.assertEqual(plan.n_transfers, 0)
        self.assertEqual(set(plan.squad["player_id"].astype(str)), set(held.player_ids))


if __name__ == "__main__":
    unittest.main()
