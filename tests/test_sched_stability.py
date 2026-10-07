"""The stability rule is fixed before any new season total is read."""

from __future__ import annotations

import unittest

from src.models.stage_36_sched_stability import judge


def _row(
    season: str,
    *,
    ft_gap: float = 0.0,
    climbed: bool = True,
    killed: bool = False,
    missing: bool = False,
) -> dict:
    return {
        "season": season,
        "ft_gap": ft_gap,
        "climbed": climbed,
        "killed": killed,
        "missing": missing,
    }


class StabilityRuleTests(unittest.TestCase):
    def test_small_gains_on_every_season_are_stable(self) -> None:
        rows = [
            _row("2025-26", ft_gap=9),
            _row("2024-25", ft_gap=4),
            _row("2023-24", ft_gap=1),
            _row("2022-23", ft_gap=2),
        ]
        self.assertEqual(judge(rows), "STABLE")

    def test_a_loss_of_34_is_unstable(self) -> None:
        rows = [
            _row("2025-26", ft_gap=9),
            _row("2024-25", ft_gap=-34),
            _row("2023-24", ft_gap=20),
            _row("2022-23", ft_gap=20),
        ]
        self.assertEqual(judge(rows), "UNSTABLE")

    def test_a_killed_screen_is_unstable(self) -> None:
        rows = [
            _row("2025-26", ft_gap=9),
            _row("2024-25", killed=True, climbed=False, ft_gap=float("nan")),
            _row("2023-24", ft_gap=5),
            _row("2022-23", ft_gap=5),
        ]
        self.assertEqual(judge(rows), "UNSTABLE")

    def test_fewer_than_three_climbs_is_inconclusive(self) -> None:
        rows = [
            _row("2025-26", ft_gap=9),
            _row("2024-25", missing=True, climbed=False),
            _row("2023-24", missing=True, climbed=False),
            _row("2022-23", ft_gap=5),
        ]
        self.assertEqual(judge(rows), "INCONCLUSIVE")

    def test_three_non_negative_seasons_can_still_sum_to_nothing(self) -> None:
        rows = [
            _row("2025-26", ft_gap=0),
            _row("2024-25", ft_gap=0),
            _row("2023-24", ft_gap=0),
            _row("2022-23", ft_gap=-10),
        ]
        self.assertEqual(judge(rows), "UNSTABLE")


if __name__ == "__main__":
    unittest.main()
