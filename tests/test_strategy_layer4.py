"""Layer-4 horizon MILP — unit tests."""

from __future__ import annotations

import unittest

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS
from src.strategy.layer3_week import HeldState, solve_week_transfers
from src.strategy.layer4_week import solve_horizon_transfers
from tests.test_strategy_layer3 import _held_from_l1


class Layer4Horizon(unittest.TestCase):
    def test_h1_recovers_layer3(self) -> None:
        pool, held = _held_from_l1()
        l3 = solve_week_transfers(pool, held, mu_col="mu_post", max_hits=2)
        h1 = solve_horizon_transfers(
            [pool],
            held,
            mu_col="mu_post",
            horizon=1,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=2,
        )
        self.assertEqual(h1.horizon, 1)
        self.assertEqual(set(h1.week0.bought), set(l3.bought))
        self.assertEqual(set(h1.week0.sold), set(l3.sold))
        self.assertEqual(h1.week0.captain_id, l3.captain_id)
        self.assertEqual(set(h1.week0.xi["player_id"].astype(str)), set(l3.xi["player_id"].astype(str)))
        self.assertAlmostEqual(h1.week0.objective, l3.objective, places=6)
        self.assertAlmostEqual(h1.objective, l3.objective, places=6)

    def test_horizon_looks_ahead(self) -> None:
        """A player weak now but strong later should be bought under H>1."""
        pool, held = _held_from_l1()
        owned_mids = [
            pid
            for pid in held.player_ids
            if str(pool.loc[pool["player_id"] == pid, "position"].iloc[0]) == "MID"
        ]
        weak = owned_mids[0]
        pool0 = pool.copy()
        pool0.loc[pool0["player_id"] == weak, "mu_post"] = 0.05
        star = {
            "player_id": "9001",
            "position": "MID",
            "club": "F",
            "value": 50,
            "score_xp_v2": 2.0,
            "mu_post": 2.0,
            "name": "FUTURE_STAR",
        }
        pool0 = pd.concat([pool0, pd.DataFrame([star])], ignore_index=True)
        # Week 0: star mediocre; weeks 1-3: star dominates
        pools = [pool0]
        for t in range(1, 4):
            p = pool0.copy()
            p.loc[p["player_id"] == "9001", "mu_post"] = 15.0
            # Cap others so the future XI wants the star
            pools.append(p)

        h1 = solve_horizon_transfers(
            pools[:1],
            held,
            mu_col="mu_post",
            horizon=1,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=0,
            gamma=0.9,
        )
        h4 = solve_horizon_transfers(
            pools,
            held,
            mu_col="mu_post",
            horizon=4,
            terminal_mu_weight=0.25,
            terminal_value_weight=0.0,
            max_hits=0,
            gamma=0.9,
            time_limit=120.0,
        )
        # H=1 may or may not buy (mu=2 vs weak 0.05 — likely buys). Force contrast:
        # make star worse than held mid on week 0 only.
        pool0b = pool0.copy()
        pool0b.loc[pool0b["player_id"] == "9001", "mu_post"] = 0.01
        pool0b.loc[pool0b["player_id"] == weak, "mu_post"] = 3.0
        pools_b = [pool0b]
        for _ in range(1, 4):
            p = pool0b.copy()
            p.loc[p["player_id"] == "9001", "mu_post"] = 20.0
            p.loc[p["player_id"] == weak, "mu_post"] = 0.5
            pools_b.append(p)

        h1b = solve_horizon_transfers(
            pools_b[:1],
            held,
            mu_col="mu_post",
            horizon=1,
            terminal_mu_weight=0.0,
            terminal_value_weight=0.0,
            max_hits=0,
        )
        h4b = solve_horizon_transfers(
            pools_b,
            held,
            mu_col="mu_post",
            horizon=4,
            terminal_mu_weight=0.25,
            terminal_value_weight=0.0,
            max_hits=0,
            time_limit=120.0,
        )
        self.assertNotIn("9001", h1b.week0.bought)
        self.assertIn("9001", h4b.week0.bought)
        self.assertEqual(len(h4b.week_xi_mu), 4)
        self.assertGreater(h4b.objective, h1b.objective)

    def test_legal_squad_after_horizon(self) -> None:
        pool, held = _held_from_l1()
        pools = [pool.copy() for _ in range(4)]
        plan = solve_horizon_transfers(
            pools,
            held,
            mu_col="mu_post",
            horizon=4,
            max_hits=2,
            time_limit=60.0,
        )
        self.assertEqual(len(plan.week0.squad), 15)
        self.assertEqual(len(plan.week0.xi), 11)
        self.assertGreaterEqual(plan.week0.bank_after, 0)


if __name__ == "__main__":
    unittest.main()
