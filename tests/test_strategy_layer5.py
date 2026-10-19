"""Layer 5 Step 1 — chip wallet helpers."""

from __future__ import annotations

import unittest

from src.rules.fpl_2026 import ChipWallet
from src.strategy.layer3_week import HeldState
from src.strategy.layer5_week import (
    NO_CHIP,
    ChipDecisionState,
    PlannerState,
    advance_planner_state,
    audit_legality_cases,
    bench_from_squad,
    captain_factor,
    ft_after_deadline,
    held_copy,
    hit_points_for,
    play_chip,
    score_commit_week,
    summarise_audit,
    wallet_from_played,
)


class Layer5Step1(unittest.TestCase):
    def test_audit_all_pass(self) -> None:
        doc = summarise_audit(audit_legality_cases())
        failed = [c["name"] for c in doc["checks"] if not c["pass"]]
        self.assertTrue(doc["passed"], msg=f"failed={failed}")

    def test_actions_and_play_copy(self) -> None:
        held = HeldState(
            tuple(f"p{i}" for i in range(15)),
            {f"p{i}": 40 for i in range(15)},
            bank=200,
            ft=2,
        )
        st = ChipDecisionState(held=held)
        acts = st.actions(8)
        self.assertEqual(acts[0], NO_CHIP)
        self.assertIn("wildcard", acts)

        w2 = play_chip(st.wallet, 8, "wildcard")
        self.assertNotIn("wildcard", w2.available(9))
        # original unchanged
        self.assertIn("wildcard", st.wallet.available(9))

    def test_wallet_from_played(self) -> None:
        w = wallet_from_played({6: "triple_captain", 22: "bench_boost"})
        self.assertNotIn("triple_captain", w.available(10))
        self.assertIn("triple_captain", w.available(20))
        self.assertNotIn("bench_boost", w.available(25))

    def test_hit_ft_helpers(self) -> None:
        self.assertEqual(hit_points_for(1, 3, chip=NO_CHIP), 8)
        self.assertEqual(hit_points_for(1, 10, chip="wildcard"), 0)
        self.assertEqual(ft_after_deadline(3, 2, chip="free_hit"), 3)
        self.assertEqual(captain_factor("triple_captain"), 3)
        self.assertEqual(captain_factor(NO_CHIP), 2)

    def test_score_commit_tc_bb(self) -> None:
        import pandas as pd

        xi = pd.DataFrame(
            {"player_id": [f"x{i}" for i in range(11)], "mu": [1.0] * 10 + [4.0]}
        )
        bench = pd.DataFrame(
            {"player_id": [f"b{i}" for i in range(4)], "mu": [0.5, 1.0, 1.5, 2.0]}
        )
        none = score_commit_week(xi=xi, captain_id="x10", chip=NO_CHIP)
        # xi=14, +cap 4 → 18
        self.assertAlmostEqual(none.objective, 18.0)
        tc = score_commit_week(xi=xi, captain_id="x10", chip="triple_captain")
        self.assertAlmostEqual(tc.objective, 22.0)
        self.assertGreaterEqual(tc.objective, none.objective)
        bb = score_commit_week(
            xi=xi, captain_id="x10", bench=bench, chip="bench_boost"
        )
        self.assertAlmostEqual(bb.bench_mu, 5.0)
        self.assertAlmostEqual(bb.objective, 23.0)
        with self.assertRaises(ValueError):
            score_commit_week(xi=xi, captain_id="x10", chip="wildcard")

    def test_bench_from_squad(self) -> None:
        import pandas as pd

        squad = pd.DataFrame(
            {
                "player_id": [f"p{i}" for i in range(15)],
                "mu": [float(i) for i in range(15)],
            }
        )
        xi = squad.iloc[:11]
        bench = bench_from_squad(squad, xi)
        self.assertEqual(len(bench), 4)
        self.assertEqual(set(bench["player_id"]), {f"p{i}" for i in range(11, 15)})

    def test_fh_reverts_wc_keeps(self) -> None:
        before = HeldState(
            tuple(f"a{i}" for i in range(15)),
            {f"a{i}": 45 for i in range(15)},
            bank=80,
            ft=3,
        )
        rebuilt = HeldState(
            tuple(f"b{i}" for i in range(15)),
            {f"b{i}": 50 for i in range(15)},
            bank=20,
            ft=3,
        )
        fh = advance_planner_state(
            PlannerState(held=before),
            gw=10,
            chip="free_hit",
            commit_held=rebuilt,
            n_transfers=8,
        )
        self.assertEqual(fh.commit_held.player_ids, rebuilt.player_ids)
        self.assertEqual(fh.next_state.held.player_ids, before.player_ids)
        self.assertEqual(fh.next_state.held.bank, 80)
        self.assertEqual(fh.ft_after, 3)
        self.assertEqual(fh.hit_points, 0)

        wc = advance_planner_state(
            PlannerState(held=before),
            gw=10,
            chip="wildcard",
            commit_held=rebuilt,
            n_transfers=8,
        )
        self.assertEqual(wc.next_state.held.player_ids, rebuilt.player_ids)
        self.assertEqual(wc.next_state.held.bank, 20)
        self.assertEqual(wc.ft_after, 3)
        self.assertEqual(wc.hit_points, 0)

    def test_illegal_chip_raises(self) -> None:
        held = HeldState(
            tuple(f"p{i}" for i in range(15)),
            {f"p{i}": 40 for i in range(15)},
            bank=100,
            ft=1,
        )
        st = PlannerState(held=held)
        st.wallet.play(5, "wildcard")
        with self.assertRaises(ValueError):
            advance_planner_state(
                st, gw=6, chip="wildcard", commit_held=held_copy(held), n_transfers=0
            )


if __name__ == "__main__":
    unittest.main()
