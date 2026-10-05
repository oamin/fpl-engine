"""Close-call gates for five midfielders. The slices themselves are not fixtures."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.midfield_close import (
    analyse_history,
    analyse_window,
    batch_reading,
    club_legal,
    collapse_player,
    compare_pool,
    history_reading,
    is_spike,
    season_clears,
    window_reading,
)
from src.models.season_climb import FORMATIONS


def _week(
    group: str,
    sacrifice: float,
    gain: float,
    *,
    named: str = "4-4-2",
    feasible: bool = True,
    alternate: str = "five midfielders",
) -> dict:
    return {
        "group": group,
        "alternate": alternate,
        "feasible": feasible,
        "named_form": named,
        "sacrifice": sacrifice,
        "gain": gain,
    }


def _cohort() -> list[dict]:
    """70 five-midfielder rows: 24 already named, 46 close calls, plus the reference line."""
    rows = []
    for _index in range(12):
        rows.append(_week("veteran", 0.0, 0.0, named="3-5-2"))
        rows.append(_week("rank", 0.0, 0.0, named="4-5-1"))
    for index in range(23):
        rows.append(_week("veteran", 0.5, 2.0))
        rows.append(_week("rank", 0.5, 2.0))
    rows.append(_week("reference", 0.0, 0.0, named="3-5-2"))
    for _index in range(4):
        rows.append(_week("reference", 0.4, 1.0))
    return rows


def _season(name: str, mean: float, n: int = 20, skips: int = 0) -> dict:
    return {
        "season": name,
        "clears": season_clears(n_skips=skips, n_decision=n, mean_decision=mean),
        "decision": {"mean_gain": mean, "n": n},
    }


def _player(
    pid: str,
    position: str,
    score: float,
    points: float,
    team: str,
) -> dict:
    return {
        "player_id": pid,
        "position": position,
        "score_xp": score,
        "total_points": points,
        "team": team,
    }


class GateTest(unittest.TestCase):
    def test_present_when_the_close_call_and_the_control_agree(self) -> None:
        call = window_reading(
            n_infeasible=0,
            n_decision_14=8,
            n_decision_veterans=4,
            mean_decision_14=1.5,
            mean_decision_veterans=0.5,
            n_control_14=8,
            mean_control_14=1.5,
        )
        self.assertEqual(call["reading"], "present in this window")
        self.assertEqual(call["control"], "compared")

    def test_veterans_at_zero_are_absent(self) -> None:
        call = window_reading(
            n_infeasible=0,
            n_decision_14=20,
            n_decision_veterans=10,
            mean_decision_14=2.0,
            mean_decision_veterans=0.0,
            n_control_14=8,
            mean_control_14=-1.0,
        )
        self.assertEqual(call["reading"], "absent in this window")

    def test_a_higher_control_mean_is_absent(self) -> None:
        call = window_reading(
            n_infeasible=0,
            n_decision_14=8,
            n_decision_veterans=4,
            mean_decision_14=1.0,
            mean_decision_veterans=1.0,
            n_control_14=8,
            mean_control_14=1.01,
        )
        self.assertEqual(call["reading"], "absent in this window")

    def test_a_thin_control_can_still_be_present(self) -> None:
        call = window_reading(
            n_infeasible=0,
            n_decision_14=8,
            n_decision_veterans=4,
            mean_decision_14=1.0,
            mean_decision_veterans=1.0,
            n_control_14=7,
            mean_control_14=9.0,
        )
        self.assertEqual(call["reading"], "present in this window")
        self.assertEqual(call["control"], "too thin")

    def test_too_many_infeasible_weeks_are_inconclusive(self) -> None:
        call = window_reading(
            n_infeasible=11,
            n_decision_14=8,
            n_decision_veterans=4,
            mean_decision_14=3.0,
            mean_decision_veterans=3.0,
            n_control_14=8,
            mean_control_14=0.0,
        )
        self.assertEqual(call["reading"], "inconclusive")

    def test_a_short_decision_band_is_absent(self) -> None:
        call = window_reading(
            n_infeasible=0,
            n_decision_14=7,
            n_decision_veterans=4,
            mean_decision_14=3.0,
            mean_decision_veterans=3.0,
            n_control_14=0,
            mean_control_14=None,
        )
        self.assertEqual(call["reading"], "absent in this window")

    def test_both_screens_are_required(self) -> None:
        self.assertEqual(batch_reading("present in this window", "kept"), "kept")
        self.assertEqual(batch_reading("absent in this window", "kept"), "retired")
        self.assertEqual(batch_reading("present in this window", "retired"), "retired")
        self.assertEqual(batch_reading("inconclusive", "kept"), "retired")


class WindowSliceTest(unittest.TestCase):
    def test_weeks_already_on_five_midfielders_stay_out_of_the_band(self) -> None:
        result = analyse_window(_cohort())
        self.assertEqual(result["n_already"], 24)
        self.assertEqual(result["n_omega"], 46)
        self.assertEqual(result["tables"]["14"]["cumulative"]["0.25"]["n"], 0)
        self.assertEqual(result["tables"]["14"]["cumulative"]["1.00"]["n"], 46)
        self.assertEqual(result["tables"]["14"]["cumulative"]["1.00"]["mean_gain"], 2.0)
        self.assertEqual(result["reading"], "present in this window")

    def test_a_score_tie_stays_in_the_lowest_band(self) -> None:
        rows = [
            row
            for row in _cohort()
            if not (row["group"] == "veteran" and row["named_form"] == "3-5-2")
        ]
        rows.extend(_week("veteran", 0.0, 0.0, named="3-5-2") for _index in range(11))
        rows.append(_week("veteran", 0.0, 4.0, named="3-4-3"))
        result = analyse_window(rows)
        self.assertEqual(result["n_already"], 23)
        self.assertEqual(result["tables"]["14"]["cumulative"]["0.25"]["n"], 1)
        self.assertEqual(result["tables"]["14"]["cumulative"]["0.25"]["mean_gain"], 4.0)
        self.assertEqual(result["tables"]["14"]["cumulative"]["0.50"]["n"], 47)

    def test_an_infeasible_week_is_not_a_zero(self) -> None:
        rows = []
        for _index in range(12):
            rows.append(_week("veteran", 0.0, 0.0, named="3-5-2"))
        for _index in range(35):
            rows.append(_week("rank", 0.0, 0.0, named="4-5-1"))
        for _index in range(11):
            rows.append(_week("veteran", None, None, feasible=False))
        for _index in range(11):
            rows.append(_week("rank", 0.4, 3.0))
        rows.append(_week("veteran", 0.4, 3.0))
        for _index in range(5):
            rows.append(_week("reference", 0.2, 1.0))
        result = analyse_window(rows)
        self.assertEqual(result["reading"], "inconclusive")
        self.assertEqual(result["n_infeasible"], 11)
        self.assertEqual(result["tables"]["rank"]["cumulative"]["1.00"]["mean_gain"], 3.0)

    def test_three_forwards_are_ignored(self) -> None:
        rows = _cohort()
        rows.extend(
            _week("veteran", 0.1, -20.0, alternate="three forwards") for _index in range(10)
        )
        result = analyse_window(rows)
        self.assertEqual(result["tables"]["14"]["cumulative"]["1.00"]["mean_gain"], 2.0)

    def test_the_one_point_cut_is_closed_on_the_right(self) -> None:
        rows = []
        for _index in range(12):
            rows.append(_week("veteran", 0.0, 0.0, named="3-5-2"))
            rows.append(_week("rank", 0.0, 0.0, named="4-5-1"))
        for _index in range(4):
            rows.append(_week("veteran", 1.0, 2.0))
        for _index in range(19):
            rows.append(_week("veteran", 2.0, -4.0))
        for _index in range(23):
            rows.append(_week("rank", 2.0, -4.0))
        for _index in range(5):
            rows.append(_week("reference", 0.2, 1.0))
        result = analyse_window(rows)
        self.assertEqual(result["tables"]["14"]["cumulative"]["1.00"]["n"], 4)
        self.assertEqual(result["tables"]["14"]["control"]["n"], 42)
        self.assertEqual(result["reading"], "absent in this window")


class HistoryGateTest(unittest.TestCase):
    def test_three_clear_seasons_keep_the_hypothesis(self) -> None:
        seasons = [
            _season("2022-23", 0.4),
            _season("2023-24", 0.2),
            _season("2024-25", 0.1),
            _season("2025-26", -0.3),
        ]
        pooled = {
            "decision": {"mean_gain": 0.2, "n": 80},
            "control": {"mean_gain": 0.1, "n": 20},
        }
        call = history_reading(seasons, pooled)
        self.assertEqual(call["reading"], "kept")

    def test_two_clear_seasons_retire_it(self) -> None:
        seasons = [
            _season("2022-23", 0.4),
            _season("2023-24", 0.2),
            _season("2024-25", -0.1),
            _season("2025-26", -0.3),
        ]
        pooled = {
            "decision": {"mean_gain": 0.2, "n": 80},
            "control": {"mean_gain": 0.0, "n": 20},
        }
        self.assertEqual(history_reading(seasons, pooled)["reading"], "retired")

    def test_a_one_season_spike_retires_it(self) -> None:
        means = [3.0, -0.2, -0.1, 0.0]
        self.assertTrue(is_spike(means))
        seasons = [
            _season("2022-23", 3.0),
            _season("2023-24", -0.2),
            _season("2024-25", -0.1),
            _season("2025-26", 0.0),
        ]
        pooled = {
            "decision": {"mean_gain": 0.5, "n": 80},
            "control": {"mean_gain": 0.0, "n": 20},
        }
        self.assertEqual(history_reading(seasons, pooled)["reading"], "retired")

    def test_four_club_skips_fail_the_season(self) -> None:
        self.assertFalse(season_clears(n_skips=4, n_decision=20, mean_decision=1.0))
        self.assertTrue(season_clears(n_skips=3, n_decision=20, mean_decision=1.0))

    def test_nineteen_weeks_do_not_clear(self) -> None:
        self.assertFalse(season_clears(n_skips=0, n_decision=19, mean_decision=1.0))

    def test_a_thin_pooled_control_is_skipped(self) -> None:
        seasons = [_season(str(year), 0.5) for year in range(4)]
        pooled = {
            "decision": {"mean_gain": 0.5, "n": 80},
            "control": {"mean_gain": 4.0, "n": 19},
        }
        call = history_reading(seasons, pooled)
        self.assertEqual(call["reading"], "kept")
        self.assertEqual(call["control"], "too thin")

    def test_a_higher_pooled_control_retires_it(self) -> None:
        seasons = [_season(str(year), 0.5) for year in range(4)]
        pooled = {
            "decision": {"mean_gain": 0.5, "n": 80},
            "control": {"mean_gain": 0.51, "n": 20},
        }
        self.assertEqual(history_reading(seasons, pooled)["reading"], "retired")

    def test_history_rows_use_only_the_close_calls(self) -> None:
        rows = []
        for season in ("2022-23", "2023-24", "2024-25", "2025-26"):
            for gw in range(5, 25):
                rows.append(
                    {
                        "season": season,
                        "gw": gw,
                        "status": "omega",
                        "sacrifice": 0.4,
                        "gain": 1.0,
                    }
                )
            rows.append(
                {
                    "season": season,
                    "gw": 25,
                    "status": "already",
                    "sacrifice": 0.0,
                    "gain": 0.0,
                }
            )
        result = analyse_history(rows, ["2022-23", "2023-24", "2024-25", "2025-26"])
        self.assertEqual(result["pooled"]["decision"]["n"], 80)
        self.assertEqual(result["pooled"]["decision"]["mean_gain"], 1.0)
        self.assertEqual(result["reading"], "kept")


class PoolTest(unittest.TestCase):
    def _frame(self, mids: list[tuple[str, float, float, str]]) -> pd.DataFrame:
        rows = [
            _player("gk", "GKP", 5, 2, "Brentford"),
            _player("d1", "DEF", 6, 2, "Burnley"),
            _player("d2", "DEF", 5, 2, "Everton"),
            _player("d3", "DEF", 4, 2, "Fulham"),
            _player("d4", "DEF", 3, 2, "Leeds"),
            _player("d5", "DEF", 1, 0, "Wolves"),
            _player("f1", "FWD", 7, 2, "Newcastle"),
            _player("f2", "FWD", 6, 2, "Brighton"),
            _player("f3", "FWD", 5, 2, "Bournemouth"),
        ]
        for pid, score, points, team in mids:
            rows.append(_player(pid, "MID", score, points, team))
        return pd.DataFrame(rows)

    def test_points_are_the_named_eleven(self) -> None:
        frame = self._frame(
            [
                ("m1", 6, 2, "Aston Villa"),
                ("m2", 5, 2, "Crystal Palace"),
                ("m3", 4, 2, "Nottingham"),
                ("m4", 3, 2, "West Ham"),
                ("m5", 2.9, 20, "Sunderland"),
            ]
        )
        compared = compare_pool(frame)
        self.assertEqual(compared["status"], "omega")
        self.assertEqual(compared["named_form"], "3-4-3")
        self.assertAlmostEqual(compared["sacrifice"], 2.1)
        self.assertAlmostEqual(compared["gain"], 18.0)
        self.assertNotIn((5, 2, 3), FORMATIONS)

    def test_four_from_one_club_is_skipped(self) -> None:
        frame = self._frame(
            [
                ("m1", 6, 2, "Arsenal"),
                ("m2", 5, 2, "Arsenal"),
                ("m3", 4, 2, "Arsenal"),
                ("m4", 3, 2, "Chelsea"),
                ("m5", 2.9, 20, "Arsenal"),
            ]
        )
        self.assertEqual(compare_pool(frame)["status"], "club")

    def test_a_double_adds_the_points_and_keeps_the_earlier_score(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "player_id": "m1",
                    "position": "MID",
                    "score_xp": 4.0,
                    "total_points": 2,
                    "team": "Arsenal",
                    "date": "2024-08-16",
                },
                {
                    "player_id": "m1",
                    "position": "MID",
                    "score_xp": 9.0,
                    "total_points": 5,
                    "team": "Arsenal",
                    "date": "2024-08-19",
                },
            ]
        )
        collapsed = collapse_player(frame)
        self.assertEqual(len(collapsed), 1)
        self.assertEqual(float(collapsed.iloc[0]["score_xp"]), 4.0)
        self.assertEqual(float(collapsed.iloc[0]["total_points"]), 7.0)

    def test_three_from_one_club_is_legal(self) -> None:
        xi = pd.DataFrame({"team": ["Arsenal", "Arsenal", "Arsenal", "Chelsea"]})
        self.assertTrue(club_legal(xi))


if __name__ == "__main__":
    unittest.main()
