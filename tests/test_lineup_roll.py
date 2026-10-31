"""Lineup / minutes worlds — club matchday resample (default)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.attack_roll import build_side_dice, dice_for_lineup
from src.markets.lineup_roll import (
    CompetitorPair,
    gate_dice_weights,
    joint_minutes_for_ids,
    minutes_corr_pairs,
    minutes_panel,
    roll_club_gw_lineup,
    roll_lineup_world,
    select_disjoint_pairs,
)


class ClubMatchdayResampleTest(unittest.TestCase):
    def _hist(self) -> pd.DataFrame:
        rows = []
        # GW1: A 90, B 0, C 90
        # GW2: A 60, B 30, C 90  (shared)
        # GW3: A 0, B 90, C 90
        specs = {
            1: {1: 90, 2: 0, 3: 90},
            2: {1: 60, 2: 30, 3: 90},
            3: {1: 0, 2: 90, 3: 90},
        }
        names = {1: "Gyok", 2: "Hav", 3: "Saka"}
        pos = {1: "FWD", 2: "FWD", 3: "MID"}
        for gw, mins in specs.items():
            for eid, m in mins.items():
                rows.append(
                    {
                        "element": eid,
                        "player_id": eid,
                        "name": names[eid],
                        "player_name": names[eid],
                        "position": pos[eid],
                        "gw": gw,
                        "GW": gw,
                        "minutes": m,
                    }
                )
        return pd.DataFrame(rows)

    def _club(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "element": [1, 2, 3],
                "name": ["Gyok", "Hav", "Saka"],
                "position": ["FWD", "FWD", "MID"],
                "m": [50.0, 40.0, 90.0],
            }
        )

    def test_resample_draws_real_matchday_vector(self) -> None:
        hist = self._hist()
        club = self._club()
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(60):
            world = roll_club_gw_lineup(club, hist, rng)
            key = tuple(float(x) for x in world.minutes)
            seen.add(key)
            self.assertTrue(world.pair_choices[0][2] == "matchday_resample")
        # All three historical GWs should appear
        self.assertIn((90.0, 0.0, 90.0), seen)
        self.assertIn((60.0, 30.0, 90.0), seen)
        self.assertIn((0.0, 90.0, 90.0), seen)

    def test_shared_minutes_path_exists(self) -> None:
        world = roll_club_gw_lineup(
            self._club(), self._hist(), np.random.default_rng(1)
        )
        # force until we see 60/30 or check distribution over many draws
        saw_share = False
        rng = np.random.default_rng(2)
        for _ in range(40):
            w = roll_club_gw_lineup(self._club(), self._hist(), rng)
            if w.minutes[0] > 0 and w.minutes[1] > 0:
                saw_share = True
                self.assertAlmostEqual(float(w.minutes[0] + w.minutes[1]), 90.0)
        self.assertTrue(saw_share)

    def test_thin_history_falls_back_to_mean(self) -> None:
        hist = self._hist()
        hist = hist.loc[hist["gw"] == 1]  # only 1 GW
        world = roll_club_gw_lineup(
            self._club(), hist, np.random.default_rng(0), min_club_gws=3
        )
        np.testing.assert_allclose(world.minutes, [50.0, 40.0, 90.0])
        self.assertEqual(world.pair_choices[0][1], "mean_fallback")

    def test_attack_dice_keeps_sub_under_60(self) -> None:
        dice = build_side_dice(
            ["Gyok", "Hav", "Saka"],
            sG=[0.4, 0.4, 0.2],
            sA=[0.2, 0.2, 0.6],
            alpha_g=1.0,
            alpha_a=1.0,
            G_team=2.0,
        )
        # Manually build a 60/30 world via resampling until hit
        rng = np.random.default_rng(0)
        world = None
        for _ in range(100):
            w = roll_club_gw_lineup(self._club(), self._hist(), rng)
            if abs(float(w.minutes[0]) - 60) < 1e-6 and abs(float(w.minutes[1]) - 30) < 1e-6:
                world = w
                break
        self.assertIsNotNone(world)
        gated = dice_for_lineup(dice, world)
        self.assertGreater(float(gated.pi[0]), 0.0)
        self.assertGreater(float(gated.pi[1]), 0.0)
        self.assertTrue(bool(world.eligible[0]))
        self.assertFalse(bool(world.eligible[1]))

    def test_gate_renorms(self) -> None:
        out = gate_dice_weights(np.array([0.5, 0.3, 0.2]), np.array([True, False, True]))
        self.assertAlmostEqual(float(out.sum()), 1.0)
        self.assertEqual(float(out[1]), 0.0)


class LegacyPairPathTest(unittest.TestCase):
    """Parked pair utilities still behave for archaeology."""

    def test_legacy_joint_pair_still_importable(self) -> None:
        joints = ((90.0, 0.0), (60.0, 30.0), (0.0, 90.0))
        pairs = [CompetitorPair(0, 1, -0.9, "A", "B", joints=joints)]
        world = roll_lineup_world(
            mean_m=[50.0, 40.0],
            names=["A", "B"],
            pairs=pairs,
            rng=np.random.default_rng(0),
        )
        self.assertIn(
            (float(world.minutes[0]), float(world.minutes[1])),
            joints,
        )


if __name__ == "__main__":
    unittest.main()
