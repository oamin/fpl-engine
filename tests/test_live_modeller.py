"""Live minutes, chip policy, and the official XI list."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.live.fpl_snapshot import fixture_counts, next_event, week_flags
from src.live.odds import fetch_odds_api, load_odds_snapshot
from src.live.plan import captain_extra, live_xi
from src.live.policy import FH_MARGIN, WC_MARGIN, WeekOutlook, recommend_chip
from src.live.xmi import LiveInputError, apply_supplied_xmi, load_xmi
from src.models.season_climb import pick_xi


def _outlook(gw: int, **kwargs: object) -> WeekOutlook:
    base = dict(
        gw=gw,
        is_blank=False,
        is_double=False,
        best_xi=40.0,
        bench_xp=0.0,
        best_player_xp=0.0,
    )
    base.update(kwargs)
    return WeekOutlook(**base)  # type: ignore[arg-type]


def _player(pid: str, pos: str, score: float) -> dict[str, object]:
    return {"player_id": pid, "position": pos, "score": score}


class XmiTest(unittest.TestCase):
    def test_missing_file_does_not_fall_back(self) -> None:
        with self.assertRaises(LiveInputError):
            load_xmi(Path("/tmp/does-not-exist-xmi.csv"), 6)

    def test_bounds_and_duplicates(self) -> None:
        path = Path("/tmp/live_xmi_test.csv")
        path.write_text("player_id,gw,xmi\n1,6,90\n1,6,10\n", encoding="utf-8")
        with self.assertRaises(LiveInputError):
            load_xmi(path, 6)
        path.write_text("player_id,gw,xmi\n1,6,181\n", encoding="utf-8")
        with self.assertRaises(LiveInputError):
            load_xmi(path, 6)

    def test_omitted_player_is_zero_minutes(self) -> None:
        supplied = pd.DataFrame({"player_id": ["1"], "gw": [6], "xmi": [90.0]})
        features = pd.DataFrame({"player_id": ["1", "2"], "xmi": [30.0, 80.0]})
        out = apply_supplied_xmi(features, supplied)
        self.assertEqual(list(out["xmi"]), [90.0, 0.0])


class OddsTest(unittest.TestCase):
    def test_paid_endpoint_is_refused(self) -> None:
        with self.assertRaises(LiveInputError):
            fetch_odds_api()

    def test_missing_snapshot_is_refused(self) -> None:
        with self.assertRaises(LiveInputError):
            load_odds_snapshot(Path("/tmp/no-odds-snapshot.csv"))


class FormationTest(unittest.TestCase):
    def test_live_xi_can_pick_5_2_3_and_the_climb_list_cannot(self) -> None:
        rows = [_player("g", "GKP", 1)]
        rows += [_player(f"d{i}", "DEF", 10) for i in range(5)]
        rows += [_player(f"m{i}", "MID", 10) for i in range(2)]
        rows += [_player(f"f{i}", "FWD", 10) for i in range(3)]
        squad = pd.DataFrame(rows)
        with self.assertRaises(RuntimeError):
            pick_xi(squad, "score")
        plan = live_xi(squad, "score")
        self.assertEqual(plan["formation"], (5, 2, 3))

    def test_captain_is_the_highest_score_and_triple_captain_falls_through(self) -> None:
        rows = [
            _player("g", "GKP", 1),
            _player("d1", "DEF", 4),
            _player("d2", "DEF", 3),
            _player("d3", "DEF", 2),
            _player("m1", "MID", 9),
            _player("m2", "MID", 6),
            _player("m3", "MID", 5),
            _player("m4", "MID", 1),
            _player("f1", "FWD", 8),
            _player("f2", "FWD", 7),
            _player("f3", "FWD", 1),
            _player("g2", "GKP", 0),
            _player("d4", "DEF", 0.5),
            _player("d5", "DEF", 0.4),
            _player("m5", "MID", 0.2),
        ]
        plan = live_xi(pd.DataFrame(rows), "score")
        self.assertEqual(plan["captain"], "m1")
        self.assertEqual(plan["vice"], "f1")
        self.assertEqual(captain_extra(9, 8, captain_played=False, vice_played=True, chip="triple_captain"), 16.0)


class ChipPolicyTest(unittest.TestCase):
    def test_single_week_does_not_take_triple_captain(self) -> None:
        weeks = [_outlook(6, best_player_xp=12, bench_xp=8)]
        self.assertIsNone(recommend_chip(6, weeks))

    def test_triple_captain_only_on_the_best_double(self) -> None:
        weeks = [
            _outlook(6, is_double=True, best_player_xp=8),
            _outlook(10, is_double=True, best_player_xp=11),
        ]
        self.assertIsNone(recommend_chip(6, weeks))
        self.assertEqual(recommend_chip(10, weeks), "triple_captain")

    def test_free_hit_needs_the_locked_margin_on_a_blank(self) -> None:
        short = [_outlook(6, is_blank=True, best_xi=20, fh_xi=20 + FH_MARGIN - 0.1)]
        self.assertIsNone(recommend_chip(6, short))
        clear = [_outlook(6, is_blank=True, best_xi=20, fh_xi=20 + FH_MARGIN)]
        self.assertEqual(recommend_chip(6, clear), "free_hit")

    def test_wildcard_in_gw1_is_illegal(self) -> None:
        weeks = [_outlook(1, wc_delta=WC_MARGIN)]
        self.assertIsNone(recommend_chip(1, weeks))

    def test_back_to_back_free_hit_is_illegal(self) -> None:
        weeks = [_outlook(6, is_blank=True, best_xi=10, fh_xi=30)]
        self.assertIsNone(recommend_chip(6, weeks, played={5: "free_hit"}))

    def test_equal_gains_play_nothing(self) -> None:
        weeks = [_outlook(8, is_double=True, bench_xp=9, best_player_xp=9)]
        self.assertIsNone(recommend_chip(8, weeks))

    def test_bench_boost_in_gw1_is_legal_on_a_double(self) -> None:
        weeks = [_outlook(1, is_double=True, bench_xp=7, best_player_xp=4)]
        self.assertEqual(recommend_chip(1, weeks), "bench_boost")


class FixtureCountTest(unittest.TestCase):
    def test_counts_and_next_deadline(self) -> None:
        fixtures = [
            {"event": 6, "team_h": 1, "team_a": 2},
            {"event": 6, "team_h": 1, "team_a": 3},
            {"event": 7, "team_h": 2, "team_a": 3},
        ]
        counts = fixture_counts(fixtures, 6)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[2], 1)
        self.assertNotIn(4, counts)
        blank, double = week_flags(counts, [1, 4])
        self.assertTrue(blank)
        self.assertTrue(double)
        event = next_event(
            {
                "events": [
                    {"id": 5, "is_current": True, "is_next": False},
                    {"id": 6, "is_current": False, "is_next": True, "deadline_time": "2026-10-10T10:00:00Z"},
                ]
            }
        )
        self.assertEqual(event["id"], 6)


if __name__ == "__main__":
    unittest.main()
