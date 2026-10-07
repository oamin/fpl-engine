"""Decision-layer rules, locked before any replay total is read."""

from __future__ import annotations

import inspect
import unittest
from collections import Counter

import pandas as pd

from src.eval.decision import (
    Squad,
    armband,
    assert_open_score,
    calibrate,
    captain_gap,
    chips_fired,
    collapse_gameweek,
    greedy_step,
    pool_columns,
    realised_over,
    replay_season,
    template_squad,
)
from src.eval.decision_spec import HORIZON, TEMPLATE_SLOTS
from src.rules.fpl_2026 import SQUAD_QUOTA


def _row(
    pid: str,
    pos: str,
    club: str,
    value: int,
    score: float,
    points: float,
    gw: int,
    *,
    eligible: bool = True,
    exp: float | None = None,
    official: float = 0.0,
) -> dict[str, object]:
    return {
        "player_id": pid,
        "position": pos,
        "team_norm": club,
        "value": value,
        "score_xp": score,
        "score_exp_points": score if exp is None else exp,
        "total_points": points,
        "gw": gw,
        "eligible": eligible,
        "official_xp": official,
        "date": f"2024-08-{int(gw):02d}",
        "fixture_id": f"gw{gw}:{pid}",
    }


def _core(gw: int, score: float = 1.0, points: float = 2.0, eligible: bool = True) -> list[dict[str, object]]:
    spec = [
        ("1", "GKP", "a", 45),
        ("2", "GKP", "a", 40),
        ("3", "DEF", "b", 65),
        ("4", "DEF", "b", 50),
        ("5", "DEF", "b", 45),
        ("6", "DEF", "c", 40),
        ("7", "DEF", "c", 40),
        ("8", "MID", "d", 90),
        ("9", "MID", "d", 65),
        ("10", "MID", "d", 55),
        ("11", "MID", "e", 50),
        ("12", "MID", "e", 45),
        ("13", "FWD", "f", 110),
        ("14", "FWD", "f", 75),
        ("15", "FWD", "g", 55),
    ]
    return [
        _row(pid, pos, club, value, score, points, gw, eligible=eligible)
        for pid, pos, club, value in spec
    ]


def _hand_squad() -> Squad:
    return Squad(
        purchase={pid: 40 for pid in [str(i) for i in range(1, 16)]},
        position={
            "1": "GKP",
            "2": "GKP",
            "3": "DEF",
            "4": "DEF",
            "5": "DEF",
            "6": "DEF",
            "7": "DEF",
            "8": "MID",
            "9": "MID",
            "10": "MID",
            "11": "MID",
            "12": "MID",
            "13": "FWD",
            "14": "FWD",
            "15": "FWD",
        },
        club={pid: f"c{pid}" for pid in [str(i) for i in range(1, 16)]},
        bank=400,
    )


class DecisionRuleTest(unittest.TestCase):
    def test_greedy_at_t_does_not_see_the_next_score(self) -> None:
        rows: list[dict[str, object]] = []
        rows.extend(_core(5, score=1.0, points=2.0))
        rows.extend(_core(6, score=0.0, points=1.0))
        rows.extend(_core(7, score=0.0, points=1.0))
        rows.append(_row("20", "FWD", "h", 55, 0.0, 0.0, 5, eligible=False))
        rows.append(_row("21", "FWD", "h", 55, 0.0, 0.0, 5, eligible=False))
        rows.append(_row("20", "FWD", "h", 55, 3.0, 4.0, 6))
        rows.append(_row("21", "FWD", "h", 55, 0.0, 0.0, 6))
        rows.append(_row("20", "FWD", "h", 55, 0.0, 0.0, 7))
        rows.append(_row("21", "FWD", "h", 55, 100.0, 50.0, 7))
        played = replay_season(pd.DataFrame(rows), "score_xp", gw_start=5, gw_end=7)
        first = played["transfers"][0]
        self.assertEqual(first["gw"], 6)
        self.assertEqual(first["player_in"], "20")
        self.assertEqual(first["predicted"], 3.0)
        self.assertNotEqual(first["player_in"], "21")
        self.assertEqual(played["chips"], 0)
        self.assertEqual(chips_fired(), 0)

    def test_scraped_xp_does_not_change_the_replay(self) -> None:
        rows = _core(5) + _core(6)
        base = pd.DataFrame(rows)
        other = base.copy()
        other["official_xp"] = -999.0
        left = replay_season(base, gw_start=5, gw_end=6)
        right = replay_season(other, gw_start=5, gw_end=6)
        self.assertEqual(left["weeks"], right["weeks"])
        self.assertNotIn("official_xp", inspect.getsource(greedy_step))

    def test_realised_horizon_stops_at_three_and_never_looks_back(self) -> None:
        points = {
            36: {"in": 100.0, "out": 0.0},
            37: {"in": 1.0, "out": 0.0},
            38: {"in": 1.0, "out": 0.0},
        }
        self.assertEqual(HORIZON, 3)
        self.assertEqual(realised_over(points, 36, "in", "out"), 102.0)
        self.assertEqual(realised_over(points, 37, "in", "out"), 2.0)
        self.assertEqual(realised_over(points, 38, "in", "out"), 1.0)

    def test_captain_is_not_the_oracle(self) -> None:
        xi = pd.DataFrame(
            [
                {"player_id": "10", "position": "FWD", "score_xp": 2.0, "total_points": 9.0},
                {"player_id": "2", "position": "FWD", "score_xp": 2.0, "total_points": 1.0},
            ]
        )
        self.assertEqual(armband(xi, "score_xp"), "2")
        with self.assertRaises(RuntimeError):
            armband(xi, "total_points")
        squad = _hand_squad()
        week = pd.DataFrame(_core(5, score=1.0, points=0.0))
        week.loc[week["player_id"] == "13", "score_xp"] = 10.0
        week.loc[week["player_id"] == "13", "total_points"] = 4.0
        week.loc[week["player_id"] == "13", "score_exp_points"] = 0.0
        week.loc[week["player_id"] == "14", "score_exp_points"] = 10.0
        week.loc[week["player_id"] == "14", "total_points"] = 1.0
        week.loc[week["player_id"] == "15", "total_points"] = 50.0
        self.assertEqual(captain_gap(squad, week, "score_xp", "score_exp_points"), 3.0)
        with self.assertRaises(RuntimeError):
            captain_gap(squad, week, "score_xp", "total_points")

    def test_greedy_does_not_use_a_fitted_threshold(self) -> None:
        source = inspect.getsource(greedy_step)
        self.assertNotIn("threshold", source)
        self.assertNotIn("shrinkage", source)
        self.assertNotIn("HIT_COST", source)
        self.assertNotIn("calibrate", inspect.signature(greedy_step).parameters)
        squad = Squad({"1": 40}, {"1": "DEF"}, {"1": "a"}, 100)
        week = pd.DataFrame(
            [
                _row("2", "DEF", "b", 40, 1.0, 0.0, 6),
                _row("3", "DEF", "c", 40, 5.0, 0.0, 6),
                _row("4", "MID", "d", 40, 9.0, 0.0, 6),
            ]
        )
        nxt, move = greedy_step(squad, week, "score_xp")
        self.assertEqual(move["player_in"], "3")
        self.assertEqual(set(nxt.purchase), {"3"})

    def test_template_slots_position_uniqueness_and_budget(self) -> None:
        built = template_squad(pd.DataFrame(_core(5)), "score_xp")
        counts: dict[str, int] = {}
        for pos in built.position.values():
            counts[pos] = counts.get(pos, 0) + 1
        self.assertEqual(counts, dict(SQUAD_QUOTA))
        self.assertEqual(len(built.purchase), len(TEMPLATE_SLOTS))
        self.assertEqual(len(set(built.purchase)), 15)

        pricey = []
        for index, (pos, _target) in enumerate(TEMPLATE_SLOTS[:12], start=1):
            pricey.append(_row(str(index), pos, f"c{index}", 70 if index > 2 else 50, 1.0, 0.0, 5))
        # 2*50 + 10*70 = 800. Two forwards that match their targets, then a trap.
        pricey.append(_row("13", "FWD", "c13", 100, 1.0, 0.0, 5))
        pricey.append(_row("14", "FWD", "c14", 70, 1.0, 0.0, 5))
        pricey.append(_row("50", "FWD", "c50", 50, 3.0, 0.0, 5))
        pricey.append(_row("30", "FWD", "c30", 30, 1.0, 0.0, 5))
        trapped = template_squad(pd.DataFrame(pricey), "score_xp")
        self.assertNotIn("50", trapped.purchase)
        self.assertIn("30", trapped.purchase)
        self.assertEqual(trapped.bank, 0)
        self.assertLessEqual(sum(trapped.purchase.values()), 1000)

        crowded = _core(5)
        crowded.append(_row("16", "DEF", "y", 90, 0.1, 0.0, 5))
        for row in crowded:
            if row["player_id"] in {"3", "4", "5", "6"}:
                row["team_norm"] = "b"
                row["value"] = 45
                row["score_xp"] = 5.0
            if row["player_id"] == "7":
                row["team_norm"] = "z"
                row["value"] = 90
                row["score_xp"] = 0.1
        capped = template_squad(pd.DataFrame(crowded), "score_xp")
        arsenal = [pid for pid, club in capped.club.items() if club == "b"]
        self.assertEqual(len(arsenal), 3)
        self.assertNotIn("6", capped.purchase)
        self.assertIn("7", capped.purchase)
        self.assertIn("16", capped.purchase)

    def test_sell_price_club_cap_and_an_absent_player(self) -> None:
        squad = Squad({"1": 50}, {"1": "DEF"}, {"1": "a"}, 0)
        week = pd.DataFrame(
            [
                _row("1", "DEF", "a", 60, 0.0, 0.0, 6, eligible=False),
                _row("2", "DEF", "b", 55, 2.0, 0.0, 6),
                _row("3", "DEF", "b", 56, 3.0, 0.0, 6),
            ]
        )
        _nxt, move = greedy_step(squad, week, "score_xp")
        self.assertEqual(move["player_in"], "2")
        self.assertEqual(move["proceeds"], 55)

        owned = Squad(
            {"1": 40, "2": 40, "3": 40, "4": 40},
            {"1": "DEF", "2": "DEF", "3": "DEF", "4": "DEF"},
            {"1": "a", "2": "b", "3": "b", "4": "b"},
            100,
        )
        market = pd.DataFrame(
            [
                _row("5", "DEF", "b", 40, 9.0, 0.0, 6),
                _row("6", "DEF", "c", 40, 2.0, 0.0, 6),
            ]
        )
        bought, deal = greedy_step(owned, market, "score_xp")
        self.assertEqual(deal["player_in"], "5")
        self.assertLessEqual(max(Counter(bought.club.values()).values()), 3)

        absent = Squad({"1": 40}, {"1": "DEF"}, {"1": "a"}, 50)
        only = pd.DataFrame([_row("2", "DEF", "b", 40, 2.0, 0.0, 6)])
        _replaced, deal = greedy_step(absent, only, "score_xp")
        self.assertEqual(deal["player_out"], "1")
        self.assertEqual(deal["player_in"], "2")
        self.assertEqual(deal["out_score"], 0.0)

    def test_a_double_is_one_step_and_a_missing_row_stays_in_the_week(self) -> None:
        rows = [
            _row("1", "MID", "a", 50, 1.0, 3.0, 10),
            _row("1", "MID", "a", 50, 1.0, 4.0, 10),
        ]
        rows[1]["fixture_id"] = "second"
        collapsed = collapse_gameweek(pd.DataFrame(rows), ("total_points", "score_xp"))
        self.assertEqual(len(collapsed), 1)
        self.assertAlmostEqual(float(collapsed["total_points"].iloc[0]), 7.0)
        self.assertAlmostEqual(float(collapsed["score_xp"].iloc[0]), 2.0)
        played = replay_season(pd.DataFrame(_core(5) + _core(6)[:14]), gw_start=5, gw_end=6)
        self.assertEqual([row["gw"] for row in played["weeks"]], [5, 6])

    def test_a_short_season_is_not_pooled(self) -> None:
        rows = pd.DataFrame(
            {
                "season": ["2022-23"] * 19 + ["2023-24"] * 20,
                "greedy_minus_hold": [1.0] * 39,
            }
        )
        pooled = pool_columns(
            rows, ("greedy_minus_hold",), minimum=20, n_boot=20, seed=0
        )
        summary = pooled["greedy_minus_hold"]
        self.assertNotIn("2022-23", summary["n_gws"])
        self.assertEqual(summary["incomplete_seasons"]["2022-23"], 19)
        self.assertIn("2023-24", summary["n_gws"])

    def test_no_winner_before_twenty_live_weeks(self) -> None:
        assert_open_score(0, None)
        assert_open_score(19, None)
        with self.assertRaises(RuntimeError):
            assert_open_score(19, "score_xp")
        assert_open_score(20, "score_xp")

    def test_calibration_is_not_applied_by_construction(self) -> None:
        transfers = pd.DataFrame(
            {
                "season": ["2022-23"] * 8,
                "predicted": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
                "realised": [1.0, 1.0, 2.0, 2.0, 1.5, 2.5, 3.0, 4.0],
            }
        )
        fit = calibrate(transfers, n_boot=30, seed=0)
        self.assertIn("b", fit)
        squad = Squad({"1": 40}, {"1": "DEF"}, {"1": "a"}, 20)
        week = pd.DataFrame([_row("9", "DEF", "z", 40, 1.5, 0.0, 8)])
        before = dict(squad.purchase)
        _nxt, _move = greedy_step(squad, week, "score_xp")
        self.assertEqual(squad.purchase, before)


if __name__ == "__main__":
    unittest.main()
