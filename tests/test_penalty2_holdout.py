"""The penalty 2.0 replace rule is fixed before the earlier seasons are read."""

from __future__ import annotations

import unittest

from src.models.stage_37_penalty2 import judge


def _row(
    season: str,
    delta: float,
    *,
    oos: bool = True,
    blanks_pen: float = 3.0,
    blanks_base: float = 8.0,
    missing: bool = False,
    climbed: bool = True,
) -> dict:
    return {
        "season": season,
        "delta": delta,
        "oos": oos,
        "blanks_pen": blanks_pen,
        "blanks_base": blanks_base,
        "missing": missing,
        "climbed": climbed,
    }


class Penalty2RuleTests(unittest.TestCase):
    def test_out_of_sample_gains_replace_the_default(self) -> None:
        rows = [
            _row("2025-26", 46, oos=False),
            _row("2024-25", 10),
            _row("2023-24", 5),
            _row("2022-23", 1),
        ]
        self.assertEqual(judge(rows), "REPLACE")

    def test_the_known_season_cannot_subsidise_a_loss(self) -> None:
        rows = [
            _row("2025-26", 46, oos=False),
            _row("2024-25", 0),
            _row("2023-24", 0),
            _row("2022-23", -25),
        ]
        self.assertEqual(judge(rows), "PARKED")

    def test_a_loss_of_34_is_parked(self) -> None:
        rows = [
            _row("2025-26", 46, oos=False),
            _row("2024-25", 40),
            _row("2023-24", 40),
            _row("2022-23", -34),
        ]
        self.assertEqual(judge(rows), "PARKED")

    def test_more_blank_starters_parks_a_points_gain(self) -> None:
        rows = [
            _row("2025-26", 46, oos=False),
            _row("2024-25", 10, blanks_pen=20, blanks_base=8),
            _row("2023-24", 10),
            _row("2022-23", 10),
        ]
        self.assertEqual(judge(rows), "PARKED")

    def test_a_missing_season_is_inconclusive(self) -> None:
        rows = [
            _row("2025-26", 46, oos=False),
            _row("2024-25", 10),
            _row("2023-24", 10, missing=True, climbed=False),
            _row("2022-23", 10),
        ]
        self.assertEqual(judge(rows), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
