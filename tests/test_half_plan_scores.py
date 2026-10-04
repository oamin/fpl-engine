"""Chip side report. No season climb."""

from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src.live.half_plan import SquadOutlook
from src.models.half_plan_scores import (
    BASELINE_CSV,
    SCORE_CSV,
    SCORE_SEASONS,
    club_steps,
    free_hit_points,
    score_all,
    spread_outlooks,
    week_inputs,
    write_outputs,
)
from src.models.season_climb_ft import SquadState, run_ft_season


def _outlook(xi: float) -> SquadOutlook:
    return SquadOutlook(xi_xp=xi, bench_xp=xi / 10.0, cap_xp=xi / 10.0)


def _owned_row(pid: str, position: str, club: str, score: float) -> dict:
    return {
        "player_id": pid,
        "position": position,
        "team": club,
        "team_norm": club,
        "value": 50,
        "score_xp": score,
        "eligible": True,
        "minutes": 90,
        "total_points": 2.0,
    }


def _fifteen() -> list[dict]:
    rows = []
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    clubs = list("abcdefghijklmno")
    for index, (position, club) in enumerate(zip(positions, clubs)):
        rows.append(_owned_row(f"p{index}", position, club, 1.0))
    return rows


def _stars() -> list[dict]:
    rows = []
    spec = [("GKP", 1), ("DEF", 3), ("MID", 4), ("FWD", 3)]
    start = 0
    for position, count in spec:
        for offset in range(count):
            club = f"s{start + offset}"
            rows.append(
                {
                    "player_id": f"star{start + offset}",
                    "position": position,
                    "team": club,
                    "team_norm": club,
                    "value": 120,
                    "score_xp": 40.0,
                    "eligible": True,
                    "minutes": 90,
                    "total_points": 40.0,
                }
            )
        start += count
    return rows


def _toy_season() -> tuple[pd.DataFrame, SquadState]:
    spec = (
        [("GKP", "a"), ("GKP", "b")]
        + [("DEF", club) for club in "cdefg"]
        + [("MID", club) for club in "hijkl"]
        + [("FWD", club) for club in "mno"]
    )
    rows = []
    ids = []
    for index, (position, club) in enumerate(spec, start=1):
        pid = f"p{index}"
        ids.append(pid)
        for gw in (1, 2):
            rows.append(
                {
                    "gw": gw,
                    "player_id": pid,
                    "player_name": pid,
                    "position": position,
                    "team": club,
                    "team_norm": club,
                    "value": 50,
                    "eligible": True,
                    "score_xp": 5.0,
                    "total_points": 2.0,
                    "minutes": 90.0,
                }
            )
    frame = pd.DataFrame(rows)
    opening = SquadState(purchase={pid: 50 for pid in ids}, bank=250, ft=1)
    return frame, opening


class OutlookCopyTest(unittest.TestCase):
    def test_a_blank_is_kept_and_later_weeks_copy_the_third_priced_step(self) -> None:
        clubs = {gw: {"ars"} for gw in (5, 6, 8, 9, 11)}
        self.assertEqual(club_steps(5, clubs), [5, 6, 8])
        priced = {
            5: (_outlook(5), _outlook(15), 50.0),
            6: (_outlook(6), _outlook(16), 60.0),
            8: (_outlook(8), _outlook(18), 80.0),
        }
        rows = spread_outlooks(5, clubs, priced)
        by_gw = {row.gw: row for row in rows}
        self.assertEqual([row.gw for row in rows], list(range(5, 20)))
        self.assertEqual(by_gw[7].held.xi_xp, 0.0)
        self.assertEqual(by_gw[7].fh_xi, 0.0)
        self.assertEqual(by_gw[7].rebuilt.bench_xp, 0.0)
        self.assertEqual(by_gw[9].held.xi_xp, 8.0)
        self.assertEqual(by_gw[9].fh_xi, 80.0)
        self.assertEqual(by_gw[10].held.xi_xp, 0.0)
        self.assertEqual(by_gw[10].fh_xi, 0.0)
        self.assertEqual(by_gw[11].held.xi_xp, 8.0)
        self.assertEqual(by_gw[11].rebuilt.xi_xp, 18.0)
        self.assertNotEqual(by_gw[9].held.xi_xp, by_gw[7].held.xi_xp)


class DecisionWeekFreeHitTest(unittest.TestCase):
    def test_the_decision_week_uses_the_rebuilt_eleven(self) -> None:
        owned = _fifteen()
        pool = pd.DataFrame(owned + _stars())
        state = SquadState(
            purchase={row["player_id"]: 50 for row in owned},
            bank=0,
            ft=1,
        )
        clubs = {1: {"ars"}, 2: {"ars"}}
        scores_now = {row["player_id"]: float(row["score_xp"]) for row in owned + _stars()}
        scores_next = dict(scores_now)
        step_scores = {1: scores_now, 2: scores_next}
        weeks = week_inputs(1, state, pool, clubs, step_scores)
        by_gw = {row.gw: row for row in weeks}
        self.assertEqual(by_gw[1].fh_xi, by_gw[1].rebuilt.xi_xp)
        self.assertLess(by_gw[1].fh_xi, 30.0)
        unconstrained = free_hit_points(pool, scores_now)
        self.assertGreater(unconstrained, by_gw[1].fh_xi)
        self.assertGreater(by_gw[2].fh_xi, by_gw[2].rebuilt.xi_xp)
        closed = pool.loc[~pool["player_id"].astype(str).str.startswith("star")].copy()
        closed["eligible"] = False
        fallback = free_hit_points(closed, scores_now)
        self.assertEqual(fallback, by_gw[1].fh_xi)
        self.assertEqual(by_gw[7].held.xi_xp, 0.0)
        self.assertEqual(by_gw[7].fh_xi, 0.0)
        self.assertIn(7, by_gw)


class ChipPolicyHookTest(unittest.TestCase):
    def test_a_gameweek_1_wildcard_fails_the_squad(self) -> None:
        frame, opening = _toy_season()

        def policy(gw, state, pool, gws):
            del state, pool, gws
            if int(gw) == 1:
                return "wildcard", None
            return None, None

        with self.assertRaises(RuntimeError) as caught:
            run_ft_season(
                frame,
                {"xp": "score_xp"},
                [1, 2],
                roster=frame,
                opening=opening,
                chip_policy=policy,
            )
        self.assertIn("wildcard", str(caught.exception))
        self.assertIn("GW1", str(caught.exception))

    def test_a_policy_and_a_chip_map_cannot_both_be_set(self) -> None:
        frame, opening = _toy_season()
        with self.assertRaises(RuntimeError):
            run_ft_season(
                frame,
                {"xp": "score_xp"},
                [1],
                roster=frame,
                opening=opening,
                chips={1: "bench_boost"},
                chip_policy=lambda gw, state, pool, gws: (None, None),
            )

    def test_bench_boost_on_gameweek_1_scores_the_bench_and_transfers_nobody(self) -> None:
        frame, opening = _toy_season()

        def policy(gw, state, pool, gws):
            del state, pool, gws
            if int(gw) == 1:
                return "bench_boost", 1
            return None, None

        quiet = run_ft_season(
            frame, {"xp": "score_xp"}, [1, 2], roster=frame, opening=opening
        )
        boosted = run_ft_season(
            frame,
            {"xp": "score_xp"},
            [1, 2],
            roster=frame,
            opening=opening,
            chip_policy=policy,
        )
        quiet_gw1 = quiet.loc[quiet["gw"] == 1].iloc[0]
        boost_gw1 = boosted.loc[boosted["gw"] == 1].iloc[0]
        quiet_chip = quiet_gw1["chip"]
        self.assertTrue(
            quiet_chip is None
            or str(quiet_chip) in {"None", "nan"}
        )
        self.assertEqual(boost_gw1["chip"], "bench_boost")
        self.assertEqual(int(boost_gw1["n_transfers"]), 0)
        self.assertEqual(
            float(boost_gw1["xi_points_cap"]) - float(quiet_gw1["xi_points_cap"]),
            8.0,
        )


class SideReportScopeTest(unittest.TestCase):
    def test_2024_25_is_absent_and_rejected(self) -> None:
        seasons = [season for season, _code in SCORE_SEASONS]
        self.assertEqual(seasons, ["2022-23", "2023-24", "2025-26"])
        self.assertNotIn("2024-25", seasons)
        with self.assertRaises(ValueError):
            score_all(
                openings=pd.DataFrame(),
                seasons=[("2024-25", "2425")],
                baseline=pd.DataFrame(),
            )

    def test_the_baseline_file_is_not_the_write_target(self) -> None:
        self.assertEqual(SCORE_CSV.name, "half_plan_scores.csv")
        self.assertEqual(BASELINE_CSV.name, "crowd_opening_scores.csv")
        self.assertNotEqual(SCORE_CSV.resolve(), BASELINE_CSV.resolve())
        self.assertNotEqual(SCORE_CSV.name, "season_climb_ft.csv")
        source = inspect.getsource(write_outputs)
        self.assertIn("SCORE_CSV", source)
        self.assertNotIn("BASELINE_CSV", source)
        self.assertNotIn("crowd_opening_scores", source)
        self.assertNotIn("season_climb_ft", source)
