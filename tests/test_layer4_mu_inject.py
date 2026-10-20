"""L4 μ injection — cascade / scenario boards without FD projection."""

from __future__ import annotations

import unittest

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS
from src.strategy.layer3_week import held_from_frame, solve_week_transfers
from src.strategy.layer4_mu_inject import mu_frame_to_week_pool, pools_from_mu_boards
from src.strategy.layer4_week import solve_horizon_transfers


def _toy_week0() -> pd.DataFrame:
    # Minimal legal-ish pool for L3/L4 (solver will pick what it can).
    rows = []
    eid = 1
    for pos, n, club0 in (
        ("GKP", 2, "a"),
        ("DEF", 5, "b"),
        ("MID", 5, "c"),
        ("FWD", 3, "d"),
    ):
        for i in range(n):
            rows.append(
                {
                    "element": eid,
                    "player_id": f"toy:{eid}",
                    "name": f"{pos}{i}",
                    "position": pos,
                    "club": f"club{club0}{i % 3}",
                    "value": 45 + i,
                    "mu_post": 2.0 + 0.1 * eid,
                    "score_xp_v2": 2.0 + 0.1 * eid,
                    "xp_se": 0.5,
                    "total_points": 0.0,
                }
            )
            eid += 1
    return pd.DataFrame(rows)


class Layer4MuInject(unittest.TestCase):
    def test_mu_frame_overwrites_by_element(self) -> None:
        w0 = _toy_week0()
        mu = pd.DataFrame(
            {
                "element": [1, 2, 99],
                "mu": [9.0, 8.0, 7.0],
                "xp_se": [0.1, 0.2, 0.3],
            }
        )
        out = mu_frame_to_week_pool(w0, mu)
        by = out.set_index("element")
        self.assertAlmostEqual(float(by.loc[1, "mu_post"]), 9.0)
        self.assertAlmostEqual(float(by.loc[2, "mu_post"]), 8.0)
        # Missing from μ frame → 0
        self.assertAlmostEqual(float(by.loc[3, "mu_post"]), 0.0)
        self.assertAlmostEqual(float(by.loc[1, "xp_se"]), 0.1)

    def test_pools_from_sequence_and_mapping(self) -> None:
        w0 = _toy_week0()
        b0 = pd.DataFrame({"element": w0["element"], "score_xp_v2": w0["mu_post"]})
        b1 = b0.copy()
        b1["score_xp_v2"] = b1["score_xp_v2"] + 1.0
        seq = pools_from_mu_boards(w0, [b0, b1])
        self.assertEqual(len(seq), 2)
        self.assertGreater(
            float(seq[1]["mu_post"].mean()), float(seq[0]["mu_post"].mean())
        )
        mapped = pools_from_mu_boards(w0, {0: b0, 1: b1})
        self.assertEqual(len(mapped), 2)
        pd.testing.assert_series_equal(
            seq[1]["mu_post"], mapped[1]["mu_post"], check_names=False
        )

    def test_h1_injected_matches_l3(self) -> None:
        w0 = _toy_week0()
        # Seed a held squad from top of each position.
        picks = []
        for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            part = w0.loc[w0["position"] == pos].nlargest(n, "mu_post")
            picks.append(part)
        squad = pd.concat(picks, ignore_index=True)
        squad["purchase"] = squad["value"].astype(int)
        bank = int(BUDGET_TENTHS - squad["purchase"].sum())
        held = held_from_frame(squad, bank=max(0, bank), ft=1)

        board = pd.DataFrame(
            {
                "element": w0["element"],
                "mu_post": w0["mu_post"],
                "xp_se": w0["xp_se"],
            }
        )
        pools = pools_from_mu_boards(w0, [board])
        l3 = solve_week_transfers(pools[0], held, mu_col="mu_post", max_hits=2)
        h1 = solve_horizon_transfers(
            pools,
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
