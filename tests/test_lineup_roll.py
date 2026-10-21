"""Lineup / minutes covariance worlds."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.attack_roll import build_side_dice, dice_for_lineup, roll_side_attack
from src.markets.lineup_roll import (
    CompetitorPair,
    gate_dice_weights,
    joint_minutes_for_ids,
    minutes_corr_pairs,
    minutes_panel,
    roll_lineup_world,
    select_disjoint_pairs,
)


class LineupRollTest(unittest.TestCase):
    def test_joint_resample_can_share_minutes(self) -> None:
        """Historical 60/30 and 90/0 both appear — no P_SUB_ON coin-flip."""
        joints = (
            (90.0, 0.0),
            (0.0, 90.0),
            (60.0, 30.0),
            (70.0, 20.0),
            (65.0, 25.0),
        )
        pairs = [
            CompetitorPair(
                0, 1, -0.9, "Gyok", "Hav", id_i=1, id_j=2, joints=joints
            )
        ]
        rng = np.random.default_rng(0)
        seen_share = 0
        seen_exclusive = 0
        for _ in range(300):
            world = roll_lineup_world(
                mean_m=[80.0, 40.0, 90.0],
                names=["Gyok", "Hav", "Saka"],
                pairs=pairs,
                rng=rng,
            )
            m0, m1 = float(world.minutes[0]), float(world.minutes[1])
            self.assertIn((m0, m1), joints)
            if m0 > 0 and m1 > 0:
                seen_share += 1
            if (m0 >= 90 and m1 == 0) or (m1 >= 90 and m0 == 0):
                seen_exclusive += 1
            self.assertTrue(world.pair_choices[0][2].startswith("joint:"))
        self.assertGreater(seen_share, 0)
        self.assertGreater(seen_exclusive, 0)

    def test_thin_joint_keeps_mean_minutes(self) -> None:
        pairs = [
            CompetitorPair(
                0, 1, -0.9, "A", "B", joints=((90.0, 0.0),)  # only 1 obs
            )
        ]
        world = roll_lineup_world(
            mean_m=[55.0, 35.0, 90.0],
            names=["A", "B", "C"],
            pairs=pairs,
            rng=np.random.default_rng(0),
            min_joint_obs=3,
        )
        self.assertAlmostEqual(float(world.minutes[0]), 55.0)
        self.assertAlmostEqual(float(world.minutes[1]), 35.0)
        self.assertTrue(world.pair_choices[0][2].startswith("mean_fallback:"))

    def test_gate_renorms_to_eligible(self) -> None:
        pi = np.array([0.5, 0.3, 0.2])
        el = np.array([True, False, True])
        out = gate_dice_weights(pi, el)
        self.assertAlmostEqual(float(out.sum()), 1.0)
        self.assertEqual(float(out[1]), 0.0)
        self.assertGreater(float(out[0]), float(out[2]))

    def test_dice_allows_sub_with_under_60_minutes(self) -> None:
        dice = build_side_dice(
            ["Gyok", "Hav", "Saka"],
            sG=[0.4, 0.4, 0.2],
            sA=[0.2, 0.2, 0.6],
            alpha_g=1.0,
            alpha_a=1.0,
            G_team=2.0,
        )
        world = roll_lineup_world(
            mean_m=[80.0, 40.0, 90.0],
            names=["Gyok", "Hav", "Saka"],
            pairs=[
                CompetitorPair(
                    0,
                    1,
                    -0.8,
                    "Gyok",
                    "Hav",
                    joints=((60.0, 30.0), (60.0, 30.0), (60.0, 30.0)),
                )
            ],
            rng=np.random.default_rng(3),
        )
        gated = dice_for_lineup(dice, world)
        self.assertGreater(float(gated.pi[0]), 0.0)
        self.assertGreater(float(gated.pi[1]), 0.0)
        self.assertTrue(bool(world.eligible[0]))
        self.assertFalse(bool(world.eligible[1]))

    def test_dice_for_lineup_zeros_fully_benched(self) -> None:
        dice = build_side_dice(
            ["Gyok", "Hav", "Saka"],
            sG=[0.4, 0.4, 0.2],
            sA=[0.2, 0.2, 0.6],
            alpha_g=1.0,
            alpha_a=1.0,
            G_team=2.0,
        )
        world = roll_lineup_world(
            mean_m=[90, 90, 90],
            names=["Gyok", "Hav", "Saka"],
            pairs=[
                CompetitorPair(
                    0,
                    1,
                    -0.8,
                    "Gyok",
                    "Hav",
                    joints=((90.0, 0.0), (90.0, 0.0), (90.0, 0.0)),
                )
            ],
            rng=np.random.default_rng(1),
        )
        gated = dice_for_lineup(dice, world)
        self.assertGreater(float(gated.pi[0]), 0.0)
        self.assertEqual(float(gated.pi[1]), 0.0)
        roll = roll_side_attack(
            dice, goals_for=30, rng=np.random.default_rng(2), lineup=world
        )
        self.assertNotIn("Hav", roll.scorers)

    def test_minutes_corr_pairs_detects_mutex_and_attaches_joints(self) -> None:
        rows = []
        for gw in range(1, 9):
            rows.append(
                {
                    "player_id": 1,
                    "player_name": "A",
                    "position": "FWD",
                    "gw": gw,
                    "minutes": 90 if gw % 2 else 0,
                }
            )
            rows.append(
                {
                    "player_id": 2,
                    "player_name": "B",
                    "position": "FWD",
                    "gw": gw,
                    "minutes": 0 if gw % 2 else 90,
                }
            )
            rows.append(
                {
                    "player_id": 3,
                    "player_name": "C",
                    "position": "MID",
                    "gw": gw,
                    "minutes": 90,
                }
            )
        # One shared-minutes GW pattern mixed in via extra rows on gw 9-11
        for gw, (ma, mb) in enumerate([(60, 30), (70, 20), (65, 25)], start=9):
            rows.append(
                {
                    "player_id": 1,
                    "player_name": "A",
                    "position": "FWD",
                    "gw": gw,
                    "minutes": ma,
                }
            )
            rows.append(
                {
                    "player_id": 2,
                    "player_name": "B",
                    "position": "FWD",
                    "gw": gw,
                    "minutes": mb,
                }
            )
            rows.append(
                {
                    "player_id": 3,
                    "player_name": "C",
                    "position": "MID",
                    "gw": gw,
                    "minutes": 90,
                }
            )
        hist = pd.DataFrame(rows)
        table = minutes_corr_pairs(hist, corr_thresh=-0.3)
        self.assertFalse(table.empty)
        panel = minutes_panel(hist)
        joints = joint_minutes_for_ids(panel, 1, 2)
        self.assertIn((60.0, 30.0), joints)
        self.assertIn((90.0, 0.0), joints)
        pairs = select_disjoint_pairs([1, 2, 3], table, panel=panel)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0].name_i, pairs[0].name_j}, {"A", "B"})
        self.assertGreaterEqual(len(pairs[0].joints), 3)


if __name__ == "__main__":
    unittest.main()
