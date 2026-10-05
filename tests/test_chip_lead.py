"""The wildcard lead's reading, locked before the carry is opened."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.chip_lead import (
    analyse,
    analyse_rebuild,
    bug_status,
    buy_tag,
    decide,
    follow_call,
    pair_tag,
    place_signing,
    reconstruct_choice,
)
from src.models.reset_chips import StepOutlook, choose_chip
from src.rules.fpl_2026 import ChipWallet


def _outlook(gw: int, wc: float, *, fh: float = 1.0, cap: float = 0.0, bench: float = 0.0) -> dict:
    return {
        "gw": gw,
        "wc_sum": wc,
        "fh_margin": fh,
        "cap_xp": cap,
        "bench_xp": bench,
        "later_cap": -1.0,
        "later_bench": -1.0,
        "chip": None,
    }


def _managers(wc_late: float, n: int = 12) -> list[dict]:
    managers = []
    for index in range(n):
        weeks = []
        for gw, lead in ((1, 0.0), (2, 1.0), (3, 1.0), (4, wc_late), (5, wc_late)):
            weeks.append(_outlook(gw, lead))
        managers.append(
            {"group": "veteran" if index < 6 else "rank", "finished": True, "entry_id": index, "weeks": weeks}
        )
    return managers


def _chip(tag: str, points: float) -> dict:
    return {"kind": "chip_squad", "tag": tag, "points": points, "group": "veteran"}


class ChoiceTest(unittest.TestCase):
    def test_the_stored_outlook_matches_the_chip_rule(self) -> None:
        row = _outlook(4, 18.0)
        row["chip"] = "wildcard"
        played = reconstruct_choice(row, ("wildcard", "free_hit", "bench_boost", "triple_captain"))
        direct, _gain = choose_chip(
            [
                StepOutlook(4, 0.0, 0.0, 0.0, 18.0, 1.0),
                StepOutlook(5, 0.0, -1.0, -1.0, 0.0, 0.0),
            ],
            ("wildcard", "free_hit", "bench_boost", "triple_captain"),
        )
        self.assertEqual(played, direct)
        self.assertEqual(played, "wildcard")

    def test_a_larger_bench_outranks_the_wildcard(self) -> None:
        row = _outlook(4, 17.0, bench=19.0)
        row["chip"] = "bench_boost"
        status = bug_status(row, ("wildcard", "bench_boost", "triple_captain", "free_hit"))
        self.assertEqual(status, "outranked")

    def test_a_cleared_wildcard_that_is_not_played_is_a_bug(self) -> None:
        row = _outlook(4, 18.0)
        status = bug_status(row, ("wildcard", "bench_boost", "triple_captain", "free_hit"))
        self.assertEqual(status, "bug")

    def test_a_tie_that_plays_nothing_is_allowed(self) -> None:
        row = _outlook(4, 16.0, fh=16.0)
        status = bug_status(row, ("wildcard", "free_hit"))
        self.assertEqual(status, "tie")

    def test_gameweek_1_cannot_make_a_wildcard_bug(self) -> None:
        row = _outlook(1, 30.0)
        wallet = ChipWallet()
        self.assertNotIn("wildcard", wallet.available(1))
        self.assertEqual(bug_status(row, wallet.available(1)), "clear")


class ReadingTest(unittest.TestCase):
    def test_half_the_chip_points_blocked_is_the_pool(self) -> None:
        explained = [_chip("absent", -200.0), _chip("eligible", -129.0)]
        result = analyse(_managers(14.0), explained)
        self.assertGreaterEqual(result["share"], 0.5)
        self.assertEqual(result["call"], "pool")

    def test_a_late_lead_between_12_and_16_is_the_margin(self) -> None:
        explained = [_chip("eligible", -329.0)]
        result = analyse(_managers(13.0), explained)
        self.assertEqual(result["call"], "margin")

    def test_a_late_lead_under_12_does_not_move_the_hurdle(self) -> None:
        explained = [_chip("eligible", -329.0), {"kind": "ranked_lower", "tag": "open", "points": -10.0, "group": "rank"}]
        result = analyse(_managers(4.0), explained)
        self.assertEqual(result["call"], "stop")

    def test_open_pairs_past_40_are_the_next_gap(self) -> None:
        explained = [
            _chip("eligible", -329.0),
            {"kind": "ranked_lower", "tag": "open", "points": -41.0, "group": "rank"},
            {"kind": "ranked_lower", "tag": "arrived", "points": -30.0, "group": "rank"},
        ]
        result = analyse(_managers(4.0), explained)
        self.assertEqual(result["call"], "open")
        self.assertEqual(result["open_points"], -41.0)

    def test_the_early_median_is_not_a_gate(self) -> None:
        call = decide(
            n_finished=14,
            bugs=0,
            share=0.1,
            n_late=8,
            late_median=4.0,
            open_points=0.0,
        )
        self.assertEqual(call, "stop")

    def test_a_chip_total_that_is_not_the_published_one_stops(self) -> None:
        result = analyse(_managers(13.0), [_chip("absent", -10.0)])
        self.assertEqual(result["call"], "inconclusive")

    def test_eleven_finished_managers_are_inconclusive(self) -> None:
        result = analyse(_managers(13.0, n=11), [_chip("absent", -329.0)])
        self.assertEqual(result["call"], "inconclusive")


class RebuildTest(unittest.TestCase):
    def _pool(self) -> dict:
        rows = [
            {"player_id": "h", "position": "MID", "score_xp": 7.0, "value": 80, "team_norm": "arsenal", "eligible": True},
            {"player_id": "q", "position": "MID", "score_xp": 4.0, "value": 45, "team_norm": "chelsea", "eligible": True},
            {"player_id": "a1", "position": "DEF", "score_xp": 4.0, "value": 40, "team_norm": "arsenal", "eligible": True},
            {"player_id": "a2", "position": "DEF", "score_xp": 4.0, "value": 40, "team_norm": "arsenal", "eligible": True},
            {"player_id": "a3", "position": "DEF", "score_xp": 4.0, "value": 40, "team_norm": "arsenal", "eligible": True},
        ]
        frame = pd.DataFrame(rows)
        return {str(row.player_id): row for row in frame.itertuples()}

    def test_a_player_already_in_the_rebuild_is_in(self) -> None:
        placed = place_signing("h", self._pool(), ["h", "q"], 0, {})
        self.assertEqual(placed["place"], "in")

    def test_half_the_points_inside_stops_at_the_squad(self) -> None:
        rows = [
            {"kind": "chip_squad", "tag": "eligible", "place": "in", "points": -200.0, "gap": None, "block": ""},
            {"kind": "chip_squad", "tag": "eligible", "place": "out", "points": -129.0, "gap": 2.0, "block": "neither"},
        ]
        result = analyse_rebuild(rows)
        self.assertGreaterEqual(result["share_in"], 0.5)
        self.assertEqual(result["call"], "in_squad")

    def test_a_negative_median_is_a_score_miss(self) -> None:
        rows = [
            {"kind": "chip_squad", "tag": "eligible", "place": "out", "points": -329.0, "gap": -1.0, "block": "neither"},
        ]
        self.assertEqual(analyse_rebuild(rows)["call"], "score_miss")

    def test_a_zero_median_is_a_score_miss(self) -> None:
        self.assertEqual(follow_call(0.2, 0.0, 0.0), "score_miss")

    def test_club_before_price(self) -> None:
        placed = place_signing("h", self._pool(), ["q", "a1", "a2", "a3"], 100, {})
        self.assertEqual(placed["place"], "out")
        self.assertEqual(placed["block"], "club")
        self.assertAlmostEqual(placed["gap"], 3.0)

    def test_an_unaffordable_replacement_is_price(self) -> None:
        placed = place_signing("h", self._pool(), ["q"], 0, {"q": 45})
        self.assertEqual(placed["block"], "price")

    def test_a_legal_replacement_left_out_is_neither(self) -> None:
        placed = place_signing("h", self._pool(), ["q"], 50, {"q": 45})
        self.assertEqual(placed["block"], "neither")

    def test_constraints_on_half_the_outside_points(self) -> None:
        rows = [
            {"kind": "chip_squad", "tag": "eligible", "place": "out", "points": -200.0, "gap": 2.0, "block": "price"},
            {"kind": "chip_squad", "tag": "eligible", "place": "out", "points": -129.0, "gap": 2.0, "block": "neither"},
        ]
        result = analyse_rebuild(rows)
        self.assertEqual(result["call"], "constraint")


class TagTest(unittest.TestCase):
    def _pool(self, rows: list[dict]) -> dict:
        frame = pd.DataFrame(rows)
        return {str(row.player_id): row for row in frame.itertuples()}

    def test_a_missing_player_is_absent(self) -> None:
        self.assertEqual(buy_tag("nope", {}), "absent")

    def test_two_appearances_in_the_pool_stay_short(self) -> None:
        pool = self._pool(
            [{"player_id": "h", "eligible": False, "n_prior": 2, "value": 50, "team_norm": "arsenal"}]
        )
        self.assertEqual(buy_tag("h", pool), "appearances")

    def test_an_unaffordable_swap_is_price(self) -> None:
        pool = self._pool(
            [
                {"player_id": "h", "eligible": True, "n_prior": 5, "value": 80, "team_norm": "arsenal"},
                {"player_id": "m", "eligible": True, "n_prior": 5, "value": 50, "team_norm": "chelsea"},
            ]
        )
        tag = pair_tag("h", "m", 6.0, 4.0, pool, {"m"}, {"m": 50}, 10)
        self.assertEqual(tag, "price")

    def test_a_fourth_club_mate_is_club(self) -> None:
        rows = [
            {"player_id": "h", "eligible": True, "n_prior": 5, "value": 55, "team_norm": "arsenal"},
            {"player_id": "m", "eligible": True, "n_prior": 5, "value": 50, "team_norm": "chelsea"},
        ]
        for index in range(3):
            rows.append(
                {
                    "player_id": f"a{index}",
                    "eligible": True,
                    "n_prior": 5,
                    "value": 45,
                    "team_norm": "arsenal",
                }
            )
        pool = self._pool(rows)
        pre = {"m", "a0", "a1", "a2"}
        tag = pair_tag("h", "m", 6.0, 4.0, pool, pre, {"m": 50}, 40)
        self.assertEqual(tag, "club")

    def test_a_lead_of_one_point_is_inside_the_score_gap(self) -> None:
        pool = self._pool(
            [
                {"player_id": "h", "eligible": True, "n_prior": 5, "value": 55, "team_norm": "arsenal"},
                {"player_id": "m", "eligible": True, "n_prior": 5, "value": 50, "team_norm": "chelsea"},
            ]
        )
        self.assertEqual(
            pair_tag("h", "m", 5.0, 4.0, pool, {"m"}, {"m": 50}, 40),
            "inside 1.25",
        )
        self.assertEqual(
            pair_tag("h", "m", 6.0, 4.0, pool, {"m"}, {"m": 50}, 40),
            "open",
        )


if __name__ == "__main__":
    unittest.main()
