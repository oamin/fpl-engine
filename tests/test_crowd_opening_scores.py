"""The crowd-opening scorer keeps the published rule and a quiet hold."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.crowd_opening_scores import (
    HOLD_NEVER,
    REPORT_PATH,
    SCORE_CSV,
    best_climbs,
    feature_season,
    load_openings,
    opening_state,
    path_result,
    render_report,
    summarise_pair,
)
from src.rules.fpl_2026 import BUDGET_TENTHS


def _week(gw: int, points: float, transfers: int = 0, hits: int = 0, chip: object = None) -> dict:
    return {
        "gw": gw,
        "method": "xp_ft",
        "xi_points_cap": points,
        "n_transfers": transfers,
        "hits": hits,
        "hit_cost": 4 * hits,
        "chip": chip,
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class OpeningTests(unittest.TestCase):
    def test_the_feature_season_uses_a_hyphen(self) -> None:
        self.assertEqual(feature_season("2022_23"), "2022-23")

    def test_purchase_price_is_the_gameweek_one_value(self) -> None:
        values = [40, 40] + [45] * 5 + [65] * 5 + [70] * 3
        rows = pd.DataFrame({"element": list(range(1, 16)), "value": values})
        state = opening_state("2022-23", rows)
        self.assertEqual(state.purchase["2022-23:1"], 40)
        self.assertEqual(state.bank, BUDGET_TENTHS - sum(values))
        self.assertEqual(len(state.ids()), 15)
        self.assertEqual(state.ft, 1)

    def test_an_over_budget_fifteen_is_refused(self) -> None:
        rows = pd.DataFrame({"element": list(range(1, 16)), "value": [80] * 15})
        with self.assertRaises(ValueError):
            opening_state("2022-23", rows)

    def test_the_committed_template_keeps_its_prices(self) -> None:
        openings = load_openings()
        block = openings.loc[
            (openings["season"] == "2022_23") & (openings["squad"] == "template")
        ]
        state = opening_state("2022-23", block)
        self.assertEqual(len(state.ids()), 15)
        self.assertEqual(state.bank, BUDGET_TENTHS - int(block["value"].sum()))
        self.assertEqual(state.purchase["2022-23:283"], 130)
        self.assertTrue(all(pid.startswith("2022-23:") for pid in state.ids()))
        self.assertGreaterEqual(HOLD_NEVER, 1.0e6)


class PathTests(unittest.TestCase):
    def test_a_full_quiet_hold_is_a_baseline(self) -> None:
        weekly = _frame([_week(gw, 10) for gw in range(1, 39)])
        result = path_result(weekly, list(range(1, 39)), hold=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["points"], 380)
        self.assertEqual(result["transfers"], 0)

    def test_a_hold_that_transfers_is_not_a_baseline(self) -> None:
        rows = [_week(gw, 10) for gw in range(1, 39)]
        rows[5] = _week(6, 10, transfers=1, hits=0)
        result = path_result(_frame(rows), list(range(1, 39)), hold=True)
        self.assertFalse(result["ok"])
        self.assertIn("transferred", str(result["error"]))

    def test_a_short_season_is_a_failure(self) -> None:
        weekly = _frame([_week(gw, 10) for gw in range(1, 10)])
        result = path_result(weekly, list(range(1, 39)), hold=False)
        self.assertFalse(result["ok"])

    def test_a_chip_fails_the_path(self) -> None:
        rows = [_week(gw, 10) for gw in range(1, 39)]
        rows[0] = _week(1, 10, chip="bench_boost")
        result = path_result(_frame(rows), list(range(1, 39)), hold=False)
        self.assertFalse(result["ok"])
        self.assertIn("chip", str(result["error"]))

    def test_the_gap_needs_both_paths(self) -> None:
        hold = path_result(
            _frame([_week(1, 10, transfers=1)]), [1], hold=True
        )
        climb = path_result(_frame([_week(1, 14)]), [1], hold=False)
        row = summarise_pair("2022-23", "template", 995, hold, climb, [1])
        self.assertFalse(row["hold_ok"])
        self.assertTrue(row["climb_ok"])
        self.assertNotEqual(row["gap"], row["gap"])

    def test_the_best_climb_stays_inside_the_season(self) -> None:
        def row(season: str, squad: str, climb: float, ok: bool) -> dict:
            return {
                "season": season,
                "squad": squad,
                "cost": 1000,
                "hold_points": 5,
                "climb_points": climb,
                "gap": climb - 5 if ok else float("nan"),
                "climb_transfers": 2,
                "climb_hits": 0,
                "hold_ok": True,
                "climb_ok": ok,
                "hold_error": "",
                "climb_error": "" if ok else "short",
            }

        table = pd.DataFrame(
            [
                row("2022-23", "template", 10, True),
                row("2022-23", "premium", 12, True),
                row("2022-23", "next", 40, False),
                row("2023-24", "third", 1, True),
                row("2023-24", "template", 1, True),
            ]
        )
        winners = best_climbs(table)
        self.assertEqual(winners["2022-23"], ["premium"])
        self.assertEqual(winners["2023-24"], ["third", "template"])
        report = render_report(table)
        self.assertIn("Best climb in 2022/23: Premium", report)
        self.assertIn("tied", report)
        self.assertNotIn("Best climb overall", report)

    def test_outputs_are_the_side_report(self) -> None:
        self.assertEqual(SCORE_CSV.name, "crowd_opening_scores.csv")
        self.assertEqual(REPORT_PATH.name, "crowd_opening_scores.md")
