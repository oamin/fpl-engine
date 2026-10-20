"""Layer-1 single-week MILP — unit tests (no network)."""

from __future__ import annotations

import unittest

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, SQUAD_QUOTA
from src.strategy.layer1_week import solve_week


def _toy_pool() -> pd.DataFrame:
    """Cheap legal pool: ≥5 clubs (15 / max-3) and excess candidates."""
    rows: list[dict] = []
    pid = 1
    # Clubs A–F so a 15 with ≤3/club is feasible.
    specs = [
        ("GKP", ["A", "B", "C", "D", "E"], [40, 45, 50, 55, 42], [3.0, 4.0, 5.0, 2.0, 3.5]),
        (
            "DEF",
            ["A", "B", "C", "D", "E", "F", "A", "B"],
            [40] * 8,
            [2, 3, 4, 5, 3, 6, 2.5, 4.5],
        ),
        (
            "MID",
            ["A", "B", "C", "D", "E", "F", "C", "D"],
            [50] * 8,
            [3, 5, 4, 6, 2, 7, 3.5, 4.5],
        ),
        ("FWD", ["A", "B", "C", "D", "E", "F"], [60] * 6, [4, 8, 5, 3, 6, 2]),
    ]
    for pos, clubs, values, mus in specs:
        for club, value, mu in zip(clubs, values, mus, strict=True):
            rows.append(
                {
                    "player_id": str(pid),
                    "position": pos,
                    "club": club,
                    "value": value,
                    "score_xp_v2": float(mu),
                    "name": f"{pos}_{club}_{pid}",
                }
            )
            pid += 1
    return pd.DataFrame(rows)


class Layer1Week(unittest.TestCase):
    def test_solves_legal_squad_xi_captain(self) -> None:
        plan = solve_week(_toy_pool(), mu_col="score_xp_v2", budget=BUDGET_TENTHS)
        self.assertEqual(len(plan.squad), 15)
        self.assertEqual(len(plan.xi), 11)
        self.assertIn(plan.captain_id, set(plan.xi["player_id"].astype(str)))
        counts = plan.squad["position"].value_counts().to_dict()
        self.assertEqual(counts, dict(SQUAD_QUOTA))
        self.assertLessEqual(plan.cost, BUDGET_TENTHS)
        self.assertAlmostEqual(plan.objective, plan.xi_mu + plan.captain_mu, places=9)
        # Captain should be the highest-μ starter (absolute μ layer).
        top = plan.xi.sort_values("mu", ascending=False).iloc[0]
        self.assertEqual(str(top["player_id"]), plan.captain_id)

    def test_respects_tight_budget(self) -> None:
        pool = _toy_pool()
        # Min feasible toy cost is 715; keep a thin surplus so the cap binds.
        plan = solve_week(pool, budget=750)
        self.assertLessEqual(plan.cost, 750)
        self.assertEqual(len(plan.squad), 15)

    def test_captain_gets_double_in_objective(self) -> None:
        plan = solve_week(_toy_pool())
        # objective = sum(XI μ) + captain μ ≡ each starter once, captain twice
        xi_ids = set(plan.xi["player_id"].astype(str))
        rebuilt = float(plan.xi["mu"].sum()) + float(
            plan.xi.loc[plan.xi["player_id"] == plan.captain_id, "mu"].iloc[0]
        )
        self.assertAlmostEqual(plan.objective, rebuilt, places=9)
        self.assertTrue(plan.captain_id in xi_ids)

    def test_club_cap_respected(self) -> None:
        plan = solve_week(_toy_pool())
        counts = plan.squad["club"].value_counts()
        self.assertTrue((counts <= 3).all(), counts.to_dict())

    def test_infeasible_too_few_clubs_raises(self) -> None:
        pool = _toy_pool()
        pool = pool.loc[pool["club"].isin(["A", "B", "C", "D"])].copy()
        with self.assertRaises(RuntimeError):
            solve_week(pool, budget=BUDGET_TENTHS)

    def test_bench_not_captain(self) -> None:
        plan = solve_week(_toy_pool())
        bench = plan.squad.loc[~plan.squad["player_id"].isin(plan.xi["player_id"])]
        self.assertFalse((bench["player_id"] == plan.captain_id).any())


if __name__ == "__main__":
    unittest.main()

