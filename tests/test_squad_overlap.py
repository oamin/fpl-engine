"""The 14 stored fifteens, counted without a new score."""

from __future__ import annotations

import unittest

from src.models.squad_overlap import _cohort, _pair_sizes, _weeks


class SquadOverlapTest(unittest.TestCase):
    def test_gameweek_one_is_a_partial_overlap(self) -> None:
        specs = _cohort()
        labels = [str(row["label"]) for row in specs]
        squads, elevens, names = _weeks(specs)
        sizes = _pair_sizes(squads[1], labels)
        self.assertEqual(len(labels), 14)
        self.assertEqual(len(sizes), 91)
        self.assertEqual(min(sizes), 3)
        self.assertEqual(max(sizes), 12)
        self.assertAlmostEqual(sum(sizes) / len(sizes), 5.90, places=2)
        pedro = next(pid for pid, row in names.items() if row[0] == "João Pedro")
        self.assertEqual(sum(pedro in squads[1][label] for label in labels), 14)
        self.assertEqual(sum(pedro in squads[5][label] for label in labels), 7)
        self.assertEqual(sum(pedro in elevens[5][label] for label in labels), 0)
        for gw in squads:
            pairs = (
                squads[gw][left] == squads[gw][right]
                for left_i, left in enumerate(labels)
                for right in labels[left_i + 1 :]
            )
            self.assertFalse(any(pairs))


if __name__ == "__main__":
    unittest.main()
