"""Live deadline squad. No season climb and no network."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live.deadline import (
    ENTRY_PATH,
    FORBIDDEN_NAMES,
    LOG_PATH,
    REPORT_PATH,
    apply_live_minutes,
    bench_for_transfers,
    clubs_missing_line,
    gameweek_values,
    holdings_state,
    line_status,
    player_key,
    price_and_plan,
    reconstruct_purchases,
    resolve_holdings,
    run,
    write_log,
)
from src.live.half_plan import SquadOutlook, WeekInputs
from src.models.season_climb_ft import SquadState, rebuild_squad
from src.rules.fpl_2026 import sell_price


def _row(pid: str, position: str, club: str, value: int, score: float) -> dict:
    return {
        "player_id": pid,
        "position": position,
        "team_norm": club,
        "value": value,
        "score": score,
        "eligible": True,
        "minutes": 90,
        "total_points": score,
    }


def _pool() -> pd.DataFrame:
    rows = []
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    clubs = [
        "ars", "che", "liv", "mci", "tot", "new", "avl",
        "bha", "ful", "cry", "wol", "eve", "bou", "bre", "nfo",
    ]
    for index, (pos, club) in enumerate(zip(positions, clubs)):
        rows.append(_row(f"p{index}", pos, club, 50, 1.0))
    rows.append(_row("star", "FWD", "whu", 70, 100.0))
    return pd.DataFrame(rows)


def _outlook(xi: float, bench: float, cap: float) -> SquadOutlook:
    return SquadOutlook(xi_xp=xi, bench_xp=bench, cap_xp=cap)


def _half(held: list[SquadOutlook], rebuilt: list[SquadOutlook], fh: float) -> list[WeekInputs]:
    rows = []
    for offset, gw in enumerate(range(6, 20)):
        rows.append(WeekInputs(gw=gw, held=held[offset], rebuilt=rebuilt[offset], fh_xi=fh))
    return rows


class SellingPriceTest(unittest.TestCase):
    def test_api_selling_price_can_buy_what_the_formula_cannot(self) -> None:
        pool = _pool()
        purchase = {f"p{i}": 50 for i in range(15)}
        formula = SquadState(purchase=purchase, bank=0, ft=1)
        site = SquadState(
            purchase=purchase,
            bank=0,
            ft=1,
            selling={pid: 80 for pid in purchase},
        )
        kept = rebuild_squad(formula, pool, "score")
        bought = rebuild_squad(site, pool, "score")
        self.assertNotIn("star", kept.ids())
        self.assertIn("star", bought.ids())
        self.assertEqual(bought.bank, 10)
        self.assertEqual(bought.purchase["star"], 70)

    def test_a_missing_selling_price_raises(self) -> None:
        pool = _pool()
        selling = {f"p{i}": 80 for i in range(14)}
        state = SquadState(
            purchase={f"p{i}": 50 for i in range(15)},
            bank=0,
            ft=1,
            selling=selling,
        )
        with self.assertRaises(RuntimeError):
            rebuild_squad(state, pool, "score")


class ReconstructionTest(unittest.TestCase):
    def test_a_later_buy_keeps_its_in_cost(self) -> None:
        entry = {
            "opening_squad": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
            "transfers": [
                {"gw": 3, "out_id": 2, "in_id": 3, "in_cost": 60, "out": "B", "in": "C"}
            ],
            "gameweeks": [
                {"gw": 1, "xi": [{"id": 1}, {"id": 2}], "bench": []},
                {"gw": 3, "xi": [{"id": 1, "name": "A"}, {"id": 3, "name": "C"}], "bench": []},
            ],
        }
        purchases = reconstruct_purchases(entry, {1: 55, 2: 40})
        self.assertEqual(purchases, {1: 55, 3: 60})

    def test_a_sale_of_someone_not_owned_raises(self) -> None:
        entry = {
            "opening_squad": [{"id": 1, "name": "A"}],
            "transfers": [{"gw": 2, "out_id": 9, "in_id": 3, "in_cost": 50}],
            "gameweeks": [{"gw": 1, "xi": [{"id": 1}], "bench": []}],
        }
        with self.assertRaises(Exception):
            reconstruct_purchases(entry, {1: 50})

    def test_stored_entry_matches_the_four_buys(self) -> None:
        entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
        logs = pd.read_csv(LOG_PATH)
        purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
        self.assertEqual(len(purchases), 15)
        self.assertEqual(purchases[346], 60)
        self.assertEqual(purchases[453], 60)
        self.assertEqual(purchases[40], 76)
        self.assertEqual(purchases[412], 50)
        self.assertEqual(purchases[8], 55)
        self.assertEqual(sell_price(55, 58), 56)


class MinutesTest(unittest.TestCase):
    def test_a_zero_in_the_file_stays_zero_and_an_omission_is_no_news(self) -> None:
        rows = apply_live_minutes([1, 2, 3], {1: 0.0}, {2: 78.0})
        by_id = {row["player_id"]: row for row in rows}
        self.assertEqual(by_id[1]["xmi"], 0.0)
        self.assertEqual(by_id[1]["source"], "file")
        self.assertEqual(by_id[2]["xmi"], 78.0)
        self.assertEqual(by_id[2]["source"], "no_news")
        self.assertEqual(by_id[3]["source"], "no_history")
        self.assertEqual(by_id[3]["xmi"], 0.0)


class LineTest(unittest.TestCase):
    def test_a_fixture_without_a_matching_1x2_is_a_missing_line(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "Date": "21/08/2026",
                    "HomeTeam": "Arsenal",
                    "AwayTeam": "Chelsea",
                    "AvgH": 2.0,
                    "AvgD": 3.4,
                    "AvgA": 3.6,
                    "Avg>2.5": 1.9,
                    "Avg<2.5": 1.9,
                }
            ]
        )
        fixtures = [
            {
                "event": 6,
                "kickoff_time": "2026-10-10T14:00:00Z",
                "team_h": 1,
                "team_a": 2,
            }
        ]
        names = {1: "Arsenal", 2: "Chelsea"}
        self.assertEqual(clubs_missing_line(odds, fixtures, names, 6), ["Arsenal", "Chelsea"])
        self.assertEqual(line_status(odds, fixtures, names, 6), "missing_opening_line")
        self.assertEqual(line_status(odds, fixtures, names, 7), "blank")
        self.assertEqual(clubs_missing_line(odds, fixtures, names, 7), [])
        priced = [
            {
                "event": 6,
                "kickoff_time": "2026-08-21T15:00:00Z",
                "team_h": 1,
                "team_a": 2,
            }
        ]
        self.assertEqual(line_status(odds, priced, names, 6), "priced")


class PlanHookTest(unittest.TestCase):
    def test_bench_boost_is_passed_and_wildcard_is_not(self) -> None:
        held = [_outlook(50.0, 5.0 if gw == 6 else 1.0, 8.0) for gw in range(6, 20)]
        weeks = _half(held, held, 50.0)
        plan = price_and_plan(6, weeks, played={1: "triple_captain"})
        self.assertEqual(plan.chip, "bench_boost")
        self.assertNotEqual(plan.chip, "triple_captain")
        self.assertEqual(bench_for_transfers(plan, 6), 6)

        richer = [_outlook(60.0, 1.0, 8.0) for _ in range(14)]
        wildcard = price_and_plan(6, _half(held, richer, 50.0), played={1: "triple_captain"})
        self.assertEqual(wildcard.chip, "wildcard")
        self.assertIsNone(bench_for_transfers(wildcard, 6))


class LogTargetTest(unittest.TestCase):
    def test_the_log_is_not_a_published_comparison(self) -> None:
        self.assertEqual(REPORT_PATH.name, "live_deadline_gw6.md")
        self.assertTrue(FORBIDDEN_NAMES)
        for name in FORBIDDEN_NAMES:
            with self.assertRaises(Exception):
                write_log("nope", Path(name))

    def test_the_stored_gameweek_stops_with_no_chip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "live_deadline_gw6.md"
            log = run(report_path=dest)
            text = dest.read_text(encoding="utf-8")
        self.assertIn("missing_opening_line", text)
        self.assertIn("missing_minutes", text)
        self.assertIn("not chosen", text)
        self.assertNotIn("327", text)
        self.assertIsNone(log.chip)
        self.assertEqual(len(log.holdings), 15)
        self.assertIsNone(log.state.selling)
        self.assertEqual(log.state.bank, 15)
        self.assertEqual(log.state.ft, 1)
        self.assertIn((1, "triple_captain"), log.chips_played)
        self.assertEqual(log.state.purchase[player_key(8)], 55)
        calafiori = next(row for row in log.holdings if row.element == 8)
        self.assertEqual(calafiori.formula_sell, 56)
        self.assertEqual(calafiori.selling_source, "formula")


class SitePriceTest(unittest.TestCase):
    def test_a_different_api_price_is_the_one_that_is_used(self) -> None:
        players = [{"id": 8, "name": "Calafiori", "position": "DEF", "team": "ARS"}]
        holdings = resolve_holdings(players, {8: 55}, {8: 58}, {8: 80}, {8: "gw1"})
        self.assertEqual(holdings[0].formula_sell, 56)
        self.assertEqual(holdings[0].selling, 80)
        self.assertTrue(holdings[0].mismatch)
        state = holdings_state(holdings, bank=15, ft=1)
        self.assertEqual(state.selling[player_key(8)], 80)


if __name__ == "__main__":
    unittest.main()
