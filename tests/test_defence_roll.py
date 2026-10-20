"""CS / GC from a rolled scoreline (cascade_short defence)."""

from __future__ import annotations

import unittest

import numpy as np

from src.markets.defence_roll import defence_from_scoreline, xp_defence_on_roll


class DefenceFromScoreline(unittest.TestCase):
    def test_clean_sheet_when_ga_zero(self) -> None:
        d = defence_from_scoreline(0)
        self.assertTrue(d.clean_sheet)
        self.assertEqual(d.gc_bins, 0)

    def test_no_cs_and_gc_bins(self) -> None:
        d = defence_from_scoreline(3)
        self.assertFalse(d.clean_sheet)
        self.assertEqual(d.goals_against, 3)
        self.assertEqual(d.gc_bins, 1)  # ⌊3/2⌋


class XpDefenceOnRoll(unittest.TestCase):
    def test_def_cs_and_gc(self) -> None:
        pos = ["GKP", "DEF", "MID", "FWD"]
        minutes = [90.0, 90.0, 90.0, 90.0]
        xp_cs, xp_gc, d = xp_defence_on_roll(pos, minutes, goals_against=0)
        self.assertTrue(d.clean_sheet)
        self.assertEqual(list(xp_cs), [4.0, 4.0, 1.0, 0.0])
        self.assertTrue(np.allclose(xp_gc, 0.0))

        xp_cs2, xp_gc2, d2 = xp_defence_on_roll(pos, minutes, goals_against=2)
        self.assertFalse(d2.clean_sheet)
        self.assertTrue(np.allclose(xp_cs2, 0.0))
        self.assertEqual(list(xp_gc2), [-1.0, -1.0, 0.0, 0.0])

    def test_cs_requires_sixty_minutes(self) -> None:
        xp_cs, xp_gc, _ = xp_defence_on_roll(
            ["DEF", "DEF"], [59.0, 60.0], goals_against=0
        )
        self.assertEqual(float(xp_cs[0]), 0.0)
        self.assertEqual(float(xp_cs[1]), 4.0)
        self.assertTrue(np.allclose(xp_gc, 0.0))

    def test_gc_for_any_minutes_played(self) -> None:
        # Locked fpl_2026: GC if minutes > 0 (not 60' gate).
        xp_cs, xp_gc, d = xp_defence_on_roll(
            ["DEF", "DEF"], [1.0, 0.0], goals_against=4
        )
        self.assertEqual(d.gc_bins, 2)
        self.assertEqual(float(xp_gc[0]), -2.0)
        self.assertEqual(float(xp_gc[1]), 0.0)
        self.assertTrue(np.allclose(xp_cs, 0.0))


if __name__ == "__main__":
    unittest.main()
