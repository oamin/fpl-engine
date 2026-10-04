"""Crowd Gameweek 1 fifteens stay inside the ownership pool and the squad law."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.crowd_openings import (
    SEASONS,
    SQUAD_NAMES,
    build_all,
    build_season,
    exclusions,
    gameweek_one,
    ownership_pool,
)
from src.rules.fpl_2026 import BUDGET_TENTHS, MAX_PER_CLUB, SQUAD_QUOTA


def _row(element: int, position: str, team: str, selected: int, value: int, gw: int = 1) -> dict:
    return {
        "name": f"P{element}",
        "position": position,
        "team": team,
        "element": element,
        "selected": selected,
        "value": value,
        "GW": gw,
        "kickoff_time": "2024-08-16T14:00:00Z",
    }


def _legal_universe() -> pd.DataFrame:
    """Enough distinct clubs and prices to fill four different fifteens."""
    rows = []
    element = 1
    clubs = [f"C{i}" for i in range(12)]
    for position, count, base in (
        ("GKP", 10, 45),
        ("DEF", 20, 40),
        ("MID", 20, 55),
        ("FWD", 12, 70),
    ):
        for i in range(count):
            rows.append(
                _row(
                    element,
                    position,
                    clubs[i % len(clubs)],
                    selected=1000 - i * 10,
                    value=base + (i % 5) * 5,
                )
            )
            element += 1
    # A later week with a huge selected count must not enter the ranking.
    rows.append(_row(9000, "FWD", "C0", selected=10**9, value=40, gw=2))
    return pd.DataFrame(rows)


class RuleTests(unittest.TestCase):
    def test_a_later_week_does_not_enter(self) -> None:
        gw1 = gameweek_one(_legal_universe())
        self.assertNotIn(9000, set(gw1["element"]))

    def test_the_pool_is_the_top_of_each_position(self) -> None:
        gw1 = gameweek_one(_legal_universe())
        pool = ownership_pool(gw1)
        self.assertEqual(len(pool), 8 + 16 + 16 + 10)
        for position, depth in (("GKP", 8), ("DEF", 16), ("MID", 16), ("FWD", 10)):
            top = set(
                gw1.loc[gw1["position"] == position]
                .sort_values(["own", "element"], ascending=[False, True])
                .head(depth)["element"]
            )
            self.assertEqual(set(pool.loc[pool["position"] == position, "element"]), top)

    def test_exclusions_follow_the_gameweek_ranking(self) -> None:
        gw1 = gameweek_one(_legal_universe())
        banned = exclusions(gw1)
        top = int(gw1.sort_values(["own", "element"], ascending=[False, True]).iloc[0]["element"])
        self.assertEqual(banned["A"], set())
        self.assertEqual(banned["B"], {top})
        self.assertEqual(len(banned["C"]), 4)
        self.assertEqual(len(banned["D"]), 8)
        self.assertTrue(banned["C"] <= banned["D"])

    def test_a_short_pool_is_missing_rather_than_padded(self) -> None:
        rows = []
        element = 1
        for position in ("GKP", "DEF", "MID", "FWD"):
            for _ in range(3):
                rows.append(_row(element, position, "Only", selected=100, value=40))
                element += 1
        squads = build_season(pd.DataFrame(rows))
        self.assertTrue(all(squad is None for squad in squads.values()))

    def test_built_squads_are_legal_and_distinct(self) -> None:
        squads = build_season(_legal_universe())
        seen: list[frozenset[int]] = []
        pool_ids = set(ownership_pool(gameweek_one(_legal_universe()))["element"])
        banned = exclusions(gameweek_one(_legal_universe()))
        for name in SQUAD_NAMES:
            squad = squads[name]
            self.assertIsNotNone(squad)
            assert squad is not None
            ids = [int(x) for x in squad["element"]]
            self.assertEqual(len(ids), 15)
            self.assertEqual(len(set(ids)), 15)
            self.assertTrue(set(ids) <= pool_ids)
            self.assertTrue(set(ids).isdisjoint(banned[name]))
            self.assertLessEqual(int(squad["value"].sum()), BUDGET_TENTHS)
            for position, count in SQUAD_QUOTA.items():
                self.assertEqual(int((squad["position"] == position).sum()), count)
            self.assertLessEqual(int(squad.groupby("team").size().max()), MAX_PER_CLUB)
            seen.append(frozenset(ids))
        self.assertEqual(len(set(seen)), 4)


class SeasonTests(unittest.TestCase):
    def test_each_finished_season_has_four_different_fifteens(self) -> None:
        table = build_all()
        self.assertEqual(set(table["season"]), set(SEASONS))
        for season in SEASONS:
            block = table.loc[table["season"] == season]
            letters = []
            for name in SQUAD_NAMES:
                squad = block.loc[block["squad"] == name]
                self.assertEqual(len(squad), 15, season + name)
                self.assertLessEqual(int(squad["value"].sum()), BUDGET_TENTHS)
                self.assertLessEqual(int(squad.groupby("team").size().max()), MAX_PER_CLUB)
                for position, count in SQUAD_QUOTA.items():
                    self.assertEqual(int((squad["position"] == position).sum()), count)
                letters.append(frozenset(squad["element"]))
            self.assertEqual(len(set(letters)), 4)


if __name__ == "__main__":
    unittest.main()
