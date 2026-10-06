"""The four shared names, scored with the published xp."""

from __future__ import annotations

import unittest

from src.models.core_xp import run


class CoreXpTest(unittest.TestCase):
    def test_gross_haul_and_the_blank_week(self) -> None:
        rows = run()
        gross = [row for row in rows if row["player"] == "Groß"]
        self.assertEqual([int(row["gw"]) for row in gross], [1, 2, 3, 4, 5])
        self.assertEqual(sum(row["points"] for row in gross), 47)
        week4 = next(row for row in gross if row["gw"] == 4)
        self.assertAlmostEqual(week4["score_xp"], 3.907856, places=4)
        self.assertEqual(week4["points"], 17)
        blank = next(row for row in rows if row["player"] == "João Pedro" and row["gw"] == 5)
        self.assertIsNone(blank["score_xp"])
        self.assertEqual(blank["points"], 0)
        self.assertAlmostEqual(blank["carried_xp"], 5.304665, places=4)
        self.assertAlmostEqual(blank["fixture_xp"], 3.754574, places=4)


if __name__ == "__main__":
    unittest.main()
