"""μ_post priced L4 path — H=1 identity + projection shape."""

from __future__ import annotations

import unittest

import pandas as pd

from src.strategy.decision_value import attach_mu_post
from src.strategy.layer3_week import held_from_frame, solve_week_transfers
from src.strategy.layer4_mupost_priced import build_mupost_priced_horizon
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.shrink_climb import SeasonStore
from src.rules.fpl_2026 import BUDGET_TENTHS
from src.strategy.layer2_week import solve_week_sigma


class MupostPriced(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        v2 = pd.read_csv("data/processed/score_xp_v2_closed.csv")
        cls.store = SeasonStore("2025-26", v2)
        cls.gw = 15
        cls.work, cls.moments = cls._week0()

    @classmethod
    def _week0(cls):
        pool = cls.store.pool(cls.gw)
        moments = cls.store.moments_asof(cls.gw)
        work, _ = attach_mu_post(
            pool,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=moments,
            out_col="mu_post",
        )
        return work, moments

    def test_horizon_has_up_to_two_weeks(self) -> None:
        hz = build_mupost_priced_horizon(
            "2025-26",
            self.gw,
            week0_pool=self.work,
            moments=self.moments,
        )
        self.assertGreaterEqual(len(hz.pools), 1)
        self.assertLessEqual(len(hz.pools), 2)
        # week0 mu unchanged
        m0 = self.work.set_index("player_id")["mu_post"]
        m0b = hz.pools[0].set_index("player_id")["mu_post"]
        common = m0.index.intersection(m0b.index)
        self.assertGreater(len(common), 50)
        delta = (m0.loc[common] - m0b.loc[common]).abs().max()
        self.assertLess(float(delta), 1e-9)

    def test_h1_recovers_l3(self) -> None:
        splan = solve_week_sigma(
            self.work,
            mu_col="score_xp_v2",
            se_col="xp_se",
            mode="none",
            moments_by_pos=self.moments,
            budget=BUDGET_TENTHS,
            time_limit=30.0,
        )
        squad = splan.plan.squad.copy()
        squad["purchase"] = squad["value"].astype(int)
        bank = int(BUDGET_TENTHS - squad["purchase"].sum())
        held = held_from_frame(squad, bank=max(0, bank), ft=1)
        l3 = solve_week_transfers(self.work, held, mu_col="mu_post", max_hits=2)
        h1 = solve_horizon_transfers(
            [self.work],
            held,
            mu_col="mu_post",
            horizon=1,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=2,
        )
        self.assertEqual(set(h1.week0.bought), set(l3.bought))
        self.assertEqual(h1.week0.captain_id, l3.captain_id)


if __name__ == "__main__":
    unittest.main()
