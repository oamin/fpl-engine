"""Lineup / minutes covariance worlds."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.attack_roll import build_side_dice, dice_for_lineup, roll_side_attack
from src.markets.lineup_roll import (
    CompetitorPair,
    gate_dice_weights,
    minutes_corr_pairs,
    roll_lineup_world,
    select_disjoint_pairs,
)


class LineupRollTest(unittest.TestCase):
    def test_exclusive_pair_never_both_eligible(self) -> None:
        pairs = [CompetitorPair(0, 1, -0.9, "A", "B")]
        rng = np.random.default_rng(0)
        both = 0
        for _ in range(200):
            world = roll_lineup_world(
                mean_m=[60.0, 60.0, 90.0],
                names=["A", "B", "C"],
                pairs=pairs,
                rng=rng,
            )
            if world.eligible[0] and world.eligible[1]:
                both += 1
            self.assertTrue(world.eligible[0] ^ world.eligible[1])
            self.assertTrue(world.eligible[2])
        self.assertEqual(both, 0)

    def test_gate_renorms_to_eligible(self) -> None:
        pi = np.array([0.5, 0.3, 0.2])
        el = np.array([True, False, True])
        out = gate_dice_weights(pi, el)
        self.assertAlmostEqual(float(out.sum()), 1.0)
        self.assertEqual(float(out[1]), 0.0)
        self.assertGreater(float(out[0]), float(out[2]))

    def test_dice_for_lineup_zeros_benched_scorer(self) -> None:
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
            pairs=[CompetitorPair(0, 1, -0.8, "Gyok", "Hav")],
            rng=np.random.default_rng(1),
        )
        gated = dice_for_lineup(dice, world)
        self.assertTrue((gated.pi[0] > 0) ^ (gated.pi[1] > 0))
        self.assertAlmostEqual(float(gated.pi.sum()), 1.0, places=6)
        loser_idx = 1 if world.eligible[0] else 0
        roll = roll_side_attack(
            dice, goals_for=30, rng=np.random.default_rng(2), lineup=world
        )
        self.assertNotIn(dice.names[loser_idx], roll.scorers)

    def test_minutes_corr_pairs_detects_mutex(self) -> None:
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
        hist = pd.DataFrame(rows)
        table = minutes_corr_pairs(hist, corr_thresh=-0.5)
        self.assertFalse(table.empty)
        ab = table[
            ((table["name_a"] == "A") & (table["name_b"] == "B"))
            | ((table["name_a"] == "B") & (table["name_b"] == "A"))
        ]
        self.assertFalse(ab.empty)
        self.assertLess(float(ab.iloc[0]["corr"]), -0.9)
        pairs = select_disjoint_pairs([1, 2, 3], table)
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs[0].name_i, pairs[0].name_j}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
