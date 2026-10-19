"""Rank tracker expanding pots."""

from __future__ import annotations

import unittest

from src.eval.rank_tracker import expanding_match_pots


class ExpandingPotsTest(unittest.TestCase):
    def test_first_match_uses_neutral_then_updates(self) -> None:
        rows = expanding_match_pots("E0_2526.csv")
        self.assertGreater(len(rows), 100)
        first = rows[0]
        self.assertIn("lam_h", first)
        self.assertIn("pot_home", first)
        # Later fixtures should have moved off pure neutral for clubs that played.
        late = rows[50]
        self.assertNotEqual(late["h_gf"], first["h_gf"])


if __name__ == "__main__":
    unittest.main()
