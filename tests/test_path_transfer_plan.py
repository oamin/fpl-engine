"""Joint multi-GW path transfer plan MILP."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.path_transfer_plan import (
    pick_path_transfer_plan,
    score_matrix_for_candidates,
    select_plan_candidates,
)
from src.rules.fpl_2026 import HIT_COST, MAX_FT, SQUAD_SIZE


def _candidates(n_extra: int = 20) -> pd.DataFrame:
    """Legal-enough pool: quotas + room to swap."""
    rows: list[dict] = []
    pid = 1
    for pos, n, base in (
        ("GKP", 4, 40),
        ("DEF", 10, 45),
        ("MID", 12, 55),
        ("FWD", 8, 70),
    ):
        for i in range(n):
            rows.append(
                {
                    "player_id": str(pid),
                    "name": f"{pos}{i}",
                    "team_norm": f"club{(pid % 12) + 1}",
                    "position": pos,
                    "value": base + (i % 5),
                }
            )
            pid += 1
    for i in range(n_extra):
        rows.append(
            {
                "player_id": str(pid),
                "name": f"EX{i}",
                "team_norm": f"club{(pid % 12) + 1}",
                "position": "MID",
                "value": 60,
            }
        )
        pid += 1
    return pd.DataFrame(rows)


class SelectCandidatesTest(unittest.TestCase):
    def test_top_n_and_buy_filter(self) -> None:
        pool = _candidates()
        H = 4
        gws = list(range(8, 8 + H))
        # Higher ids slightly better so selection is stable.
        scores = {
            g: np.linspace(1.0, 3.0, len(pool)) for g in gws
        }
        buy = set(pool["player_id"].astype(str).tolist()[:40])
        cands = select_plan_candidates(
            pool, scores, gws=gws, buy_ids=buy, top_n=36
        )
        self.assertLessEqual(len(cands), 36)
        self.assertGreaterEqual(len(cands), SQUAD_SIZE)
        self.assertTrue(set(cands["player_id"]).issubset(buy))
        self.assertEqual(cands["position"].nunique(), 4)


class JointPlanTest(unittest.TestCase):
    def test_schedules_hold_then_transfer(self) -> None:
        """Joint plan looks ahead: bench/hold a dead asset, XI them when they spike."""
        pool = _candidates()
        gws = [8, 9, 10, 11]
        n = len(pool)
        S_pool = {g: np.full(n, 2.0) for g in gws}
        star_i = int(pool.index[pool["position"] == "MID"][-1])
        for g in (8, 9):
            S_pool[g][star_i] = 0.05
        S_pool[10][star_i] = 25.0
        S_pool[11][star_i] = 25.0

        cands = select_plan_candidates(
            pool, S_pool, gws=gws, buy_ids=None, top_n=48
        )
        star_id = str(pool.iloc[star_i]["player_id"])
        self.assertIn(star_id, set(cands["player_id"].astype(str)))
        S = score_matrix_for_candidates(cands, pool, S_pool, gws)
        plan = pick_path_transfer_plan(
            cands, S, gws, gamma=1.0, ft_pe_nu=0.0, time_limit=60.0
        )
        weeks = {int(w["gw"]): w for w in plan["weeks"]}
        self.assertEqual(sorted(weeks), gws)
        self.assertEqual(weeks[8]["n_transfers"], 0)
        self.assertEqual(weeks[8]["ft_after"], 1)
        self.assertEqual(len(weeks[8]["player_ids"]), 15)
        self.assertEqual(len(weeks[8]["xi_ids"]), 11)
        self.assertIn(weeks[8]["captain_id"], weeks[8]["xi_ids"])

        for w in weeks.values():
            self.assertGreaterEqual(int(w["ft_after"]), 1)
            self.assertLessEqual(int(w["ft_after"]), MAX_FT)
            paid = max(0, int(w["n_transfers"]) - int(w["ft_before"]))
            self.assertEqual(int(w["hits"]), paid * HIT_COST)

        # Dead weeks: not in XI. Spike weeks: in XI (captain preferred).
        self.assertNotIn(star_id, weeks[8]["xi_ids"])
        self.assertNotIn(star_id, weeks[9]["xi_ids"])
        self.assertIn(star_id, weeks[10]["xi_ids"])
        self.assertIn(star_id, weeks[11]["xi_ids"])
        self.assertEqual(weeks[10]["captain_id"], star_id)

        # Either pre-owned on the bench or bought with FT — not an early hit.
        self.assertEqual(weeks[8]["hits"] + weeks[9]["hits"], 0)
        # Holding banks FT when no transfers are needed.
        if weeks[9]["n_transfers"] == 0:
            self.assertEqual(weeks[9]["ft_after"], 2)

    def test_forced_transfer_schedule(self) -> None:
        """Two expensive exclusives → open with A, transfer to B mid-horizon."""
        pool = _candidates().copy()
        gws = [8, 9, 10]
        n = len(pool)
        S_pool = {g: np.full(n, 2.0) for g in gws}
        fwd_idx = pool.index[pool["position"] == "FWD"].tolist()
        a_i, b_i = int(fwd_idx[-2]), int(fwd_idx[-1])
        pool.loc[a_i, "value"] = 200
        pool.loc[b_i, "value"] = 200
        # Mutually exclusive spikes (both too dear to hold together).
        S_pool[8][a_i] = 30.0
        S_pool[9][a_i] = 30.0
        S_pool[10][a_i] = 0.01
        S_pool[8][b_i] = 0.01
        S_pool[9][b_i] = 0.01
        S_pool[10][b_i] = 30.0

        cands = select_plan_candidates(
            pool, S_pool, gws=gws, buy_ids=None, top_n=48
        )
        a_id = str(pool.iloc[a_i]["player_id"])
        b_id = str(pool.iloc[b_i]["player_id"])
        S = score_matrix_for_candidates(cands, pool, S_pool, gws)
        plan = pick_path_transfer_plan(
            cands, S, gws, gamma=1.0, ft_pe_nu=0.0, time_limit=60.0
        )
        weeks = {int(w["gw"]): w for w in plan["weeks"]}
        self.assertIn(a_id, weeks[8]["player_ids"])
        self.assertNotIn(b_id, weeks[8]["player_ids"])
        self.assertIn(b_id, weeks[10]["player_ids"])
        self.assertGreaterEqual(
            sum(int(w["n_transfers"]) for w in plan["weeks"][1:]), 1
        )

    def test_must_spend_ft_before_hits(self) -> None:
        """Banked FT cannot be skipped to take hits (hard constraint, not soft)."""
        pool = _candidates()
        gws = [8, 9, 10, 11]
        n = len(pool)
        S_pool = {g: np.full(n, 1.0) for g in gws}
        # Rotating exclusive stars force multi-transfers after the bank grows.
        fwd = pool.index[pool["position"] == "FWD"].tolist()
        mid = pool.index[pool["position"] == "MID"].tolist()
        stars = [int(fwd[-1]), int(fwd[-2]), int(mid[-1]), int(mid[-2])]
        for i in stars:
            pool.loc[i, "value"] = 180
        for t, g in enumerate(gws):
            S_pool[g][stars[t]] = 40.0
        cands = select_plan_candidates(
            pool, S_pool, gws=gws, buy_ids=None, top_n=48
        )
        S = score_matrix_for_candidates(cands, pool, S_pool, gws)
        plan = pick_path_transfer_plan(
            cands, S, gws, gamma=1.0, ft_pe_nu=0.0, time_limit=60.0
        )
        saw_hit = False
        for w in plan["weeks"][1:]:
            nt = int(w["n_transfers"])
            fb = int(w["ft_before"])
            paid = max(0, nt - fb)
            self.assertEqual(int(w["hits"]), paid * HIT_COST)
            # Bank advance must match must-spend: u = min(k, f).
            u = min(nt, fb)
            self.assertEqual(
                int(w["ft_after"]),
                min(MAX_FT, fb - u + 1),
            )
            if paid > 0:
                saw_hit = True
        self.assertTrue(saw_hit, "fixture should force at least one paid transfer")

    def test_pe_banks_ft_when_scores_flat(self) -> None:
        """With U=4·b and flat scores, hold to grow bank toward the cap."""
        from src.rules.fpl_2026 import HIT_COST as _HC

        pool = _candidates()
        gws = [8, 9, 10, 11, 12, 13]
        n = len(pool)
        S_pool = {g: np.full(n, 1.0) for g in gws}
        cands = select_plan_candidates(
            pool, S_pool, gws=gws, buy_ids=None, top_n=40
        )
        S = score_matrix_for_candidates(cands, pool, S_pool, gws)
        plan = pick_path_transfer_plan(
            cands,
            S,
            gws,
            gamma=1.0,
            ft_pe_nu=float(_HC),
            time_limit=60.0,
        )
        # Open leaves ft=1; PE banks toward MAX_FT (free-FT churn OK if
        # terminal bank still reaches the cap — no buy-side ε).
        self.assertEqual(plan["weeks"][0]["ft_after"], 1)
        self.assertEqual(plan["weeks"][-1]["ft_after"], MAX_FT)
        self.assertEqual(plan["terminal_pe"], float(MAX_FT) * _HC)
        for w in plan["weeks"][1:]:
            self.assertEqual(w["hits"], 0)

    def test_hit_cost_in_objective(self) -> None:
        pool = _candidates()
        gws = [8, 9]
        n = len(pool)
        S_pool = {g: np.full(n, 1.0) for g in gws}
        cands = select_plan_candidates(
            pool, S_pool, gws=gws, buy_ids=None, top_n=40
        )
        S = score_matrix_for_candidates(cands, pool, S_pool, gws)
        plan = pick_path_transfer_plan(
            cands, S, gws, gamma=1.0, ft_pe_nu=0.0, time_limit=45.0
        )
        # Flat scores → never pay hits (any churn is free-FT only).
        self.assertEqual(plan["weeks"][1]["hits"], 0)
        paid = max(
            0,
            int(plan["weeks"][1]["n_transfers"]) - int(plan["weeks"][1]["ft_before"]),
        )
        self.assertEqual(paid, 0)


if __name__ == "__main__":
    unittest.main()
