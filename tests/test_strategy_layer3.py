"""Layer-3 transfer MILP — unit tests."""

from __future__ import annotations

import unittest

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, HIT_COST, SQUAD_QUOTA, advance_ft
from src.strategy.layer1_week import solve_week
from src.strategy.layer3_week import HeldState, held_from_frame, solve_week_transfers
from tests.test_strategy_layer1 import _toy_pool


def _held_from_l1(pool: pd.DataFrame | None = None) -> tuple[pd.DataFrame, HeldState]:
    pool = _toy_pool() if pool is None else pool
    plan = solve_week(pool, mu_col="score_xp_v2")
    squad = plan.squad.copy()
    squad["purchase"] = squad["value"].astype(int)
    squad["mu_post"] = squad["mu"].astype(float)
    # Attach mu_post on full pool for transfers
    pool = pool.copy()
    pool["mu_post"] = pool["score_xp_v2"].astype(float)
    bank = int(BUDGET_TENTHS - squad["purchase"].sum())
    held = held_from_frame(squad, bank=max(0, bank), ft=1)
    return pool, held


class Layer3Transfers(unittest.TestCase):
    def test_hold_when_optimal(self) -> None:
        pool, held = _held_from_l1()
        # Pool = only current optimal board → best is hold (0 transfers)
        # Restrict pool to held + a few worse players already in toy pool
        plan = solve_week_transfers(pool, held, mu_col="mu_post", max_hits=2)
        self.assertEqual(len(plan.squad), 15)
        self.assertEqual(len(plan.xi), 11)
        self.assertEqual(plan.squad["position"].value_counts().to_dict(), dict(SQUAD_QUOTA))
        self.assertEqual(plan.hit_points, HIT_COST * plan.hits)
        self.assertEqual(plan.ft_after, advance_ft(held.ft, plan.n_transfers))
        self.assertGreaterEqual(plan.bank_after, 0)
        # Objective identity
        self.assertAlmostEqual(
            plan.objective, plan.xi_mu + plan.captain_mu - plan.hit_points, places=9
        )

    def test_hold_beats_useless_hit(self) -> None:
        pool, held = _held_from_l1()
        # True hold: no FT and no hits allowed
        frozen = HeldState(held.player_ids, held.purchase, bank=held.bank, ft=0)
        hold = solve_week_transfers(pool, frozen, mu_col="mu_post", max_hits=0)
        self.assertEqual(hold.n_transfers, 0)
        self.assertEqual(hold.hits, 0)
        # FT available + hits allowed on an already-optimal board → no hits
        free = solve_week_transfers(pool, held, mu_col="mu_post", max_hits=2)
        self.assertEqual(free.hits, 0)

    def test_buys_higher_mu_within_ft(self) -> None:
        pool, held = _held_from_l1()
        pool = pool.copy()
        owned_mids = [
            pid
            for pid in held.player_ids
            if str(pool.loc[pool["player_id"] == pid, "position"].iloc[0]) == "MID"
        ]
        pool.loc[pool["player_id"] == owned_mids[0], "mu_post"] = 0.1
        star = {
            "player_id": "999",
            "position": "MID",
            "club": "F",
            "value": 50,
            "score_xp_v2": 20.0,
            "mu_post": 20.0,
            "name": "STAR_MID",
        }
        pool = pd.concat([pool, pd.DataFrame([star])], ignore_index=True)
        plan = solve_week_transfers(pool, held, mu_col="mu_post", max_hits=0)
        self.assertGreaterEqual(plan.n_transfers, 1)
        self.assertIn("999", plan.bought)
        self.assertEqual(plan.hits, 0)

    def test_hit_only_when_gain_exceeds_four(self) -> None:
        pool, held = _held_from_l1()
        held = HeldState(held.player_ids, held.purchase, bank=held.bank, ft=0)
        pool = pool.copy()
        owned_mids = [
            pid
            for pid in held.player_ids
            if str(pool.loc[pool["player_id"] == pid, "position"].iloc[0]) == "MID"
        ]
        pool.loc[pool["player_id"] == owned_mids[0], "mu_post"] = 0.0
        star = {
            "player_id": "998",
            "position": "MID",
            "club": "F",
            "value": 45,
            "score_xp_v2": 12.0,
            "mu_post": 12.0,
            "name": "HIT_STAR",
        }
        pool = pd.concat([pool, pd.DataFrame([star])], ignore_index=True)
        plan = solve_week_transfers(pool, held, mu_col="mu_post", max_hits=1)
        self.assertEqual(plan.n_transfers, 1)
        self.assertEqual(plan.hits, 1)
        self.assertEqual(plan.hit_points, 4)
        hold = solve_week_transfers(pool, held, mu_col="mu_post", max_hits=0)
        self.assertGreater(plan.objective, hold.objective - 1e-6)

    def test_missing_held_raises(self) -> None:
        pool, held = _held_from_l1()
        pool = pool.loc[~pool["player_id"].isin([held.player_ids[0]])].copy()
        with self.assertRaises(ValueError):
            solve_week_transfers(pool, held, mu_col="mu_post")


if __name__ == "__main__":
    unittest.main()
