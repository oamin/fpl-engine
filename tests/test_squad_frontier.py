"""The squad-frontier bars, locked before the chip weeks are opened."""

from __future__ import annotations

import unittest

from src.models.squad_frontier import (
    acquisition_cost,
    chip_call,
    classify_status,
    objective_gap,
    per_week,
    summarise,
    week_band,
)


class BandTest(unittest.TestCase):
    def test_edges(self) -> None:
        self.assertEqual(week_band(0.0), "near")
        self.assertEqual(week_band(1.0), "near")
        self.assertEqual(week_band(1.01), "middle")
        self.assertEqual(week_band(4.0), "middle")
        self.assertEqual(week_band(4.01), "far")

    def test_half_the_weeks_makes_the_call(self) -> None:
        self.assertEqual(chip_call([]), "none")
        self.assertEqual(chip_call(["near", "middle"]), "near")
        self.assertEqual(chip_call(["near", "middle", "middle"]), "middle")
        self.assertEqual(chip_call(["far", "middle"]), "far")
        self.assertEqual(chip_call(["far", "near", "middle"]), "middle")

    def test_a_free_hit_ignores_the_later_step(self) -> None:
        raw, steps = objective_gap("free_hit", [10.0, 99.0], [8.0, 0.0])
        self.assertEqual(steps, 1)
        self.assertAlmostEqual(raw, 2.0)
        self.assertEqual(week_band(per_week(raw, steps)), "middle")

    def test_a_short_wildcard_divides_by_the_steps_it_has(self) -> None:
        raw, steps = objective_gap("wildcard", [10.0, 10.0], [8.5, 8.5])
        self.assertEqual(steps, 2)
        self.assertAlmostEqual(per_week(raw, steps), 1.5)
        self.assertEqual(week_band(per_week(raw, steps)), "middle")


class StatusTest(unittest.TestCase):
    def test_pool_and_rules_stay_out_before_money(self) -> None:
        self.assertEqual(
            classify_status(in_pool=False, scored=False, shape_ok=False, human_over=None, model_over=True),
            "pool",
        )
        self.assertEqual(
            classify_status(in_pool=True, scored=False, shape_ok=True, human_over=None, model_over=False),
            "pool",
        )
        self.assertEqual(
            classify_status(in_pool=True, scored=True, shape_ok=False, human_over=None, model_over=True),
            "rules",
        )

    def test_the_human_budget_rejects_before_the_model_budget(self) -> None:
        self.assertEqual(
            classify_status(in_pool=True, scored=True, shape_ok=True, human_over=True, model_over=True),
            "rules_mismatch",
        )
        self.assertEqual(
            classify_status(in_pool=True, scored=True, shape_ok=True, human_over=None, model_over=True),
            "unreachable_money",
        )
        self.assertEqual(
            classify_status(in_pool=True, scored=True, shape_ok=True, human_over=False, model_over=False),
            "reachable",
        )

    def test_an_owned_player_costs_his_sell_price(self) -> None:
        cost = acquisition_cost(["a", "b"], {"a": 40}, {"a": 80, "b": 55})
        self.assertEqual(cost, 95)
        self.assertIsNone(acquisition_cost(["c"], {"a": 40}, {"a": 80}))


class SummaryTest(unittest.TestCase):
    def test_money_weeks_stay_out_of_the_call(self) -> None:
        rows = [
            {"group": "veteran", "chip": "wildcard", "status": "reachable", "band": "near", "gap_per": 0.4},
            {"group": "veteran", "chip": "wildcard", "status": "unreachable_money", "band": "far", "gap_per": 9.0},
            {"group": "rank", "chip": "free_hit", "status": "reachable", "band": "far", "gap_per": 5.0},
            {"group": "rank", "chip": "free_hit", "status": "reachable", "band": "far", "gap_per": 6.0},
            {"group": "rank", "chip": "free_hit", "status": "pool", "band": "", "gap_per": None},
        ]
        summary = summarise(rows)
        self.assertEqual(summary["chips"]["wildcard"]["call"], "near")
        self.assertEqual(summary["chips"]["wildcard"]["n_reachable"], 1)
        self.assertEqual(summary["chips"]["wildcard"]["n_money"], 1)
        self.assertEqual(summary["chips"]["free_hit"]["call"], "far")
        self.assertAlmostEqual(summary["chips"]["free_hit"]["mean"], 5.5)

    def test_the_reference_manager_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            summarise(
                [{"group": "reference", "chip": "wildcard", "status": "reachable", "band": "near", "gap_per": 0.0}]
            )
