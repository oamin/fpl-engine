"""The locked cohort and the batch bar. No network and no solver."""

from __future__ import annotations

import inspect
import unittest

from src.models.cohort_carry import (
    COHORT_N,
    FRIEND_ID,
    batch_ahead,
    cohort_specs,
    manager_result,
)


def _week(gap: float, chip: str | None = None, gw: int = 1, residual: float = 0.0) -> dict:
    return {
        "gw": gw,
        "gap": gap,
        "model_points": gap,
        "their_points": 0.0,
        "captain_gap": gap,
        "transfer_gap": 0.0,
        "lineup_gap": 0.0,
        "hit_gap": 0.0,
        "bench_gap": 0.0,
        "residual": residual,
        "chip": chip,
        "his_chip": None,
        "bench_xp": 6.0,
        "bench_scored": 2.0,
    }


def _spec(group: str = "veteran", entry_id: int = 1) -> dict:
    return {"entry_id": entry_id, "label": "A", "group": group}


def _row(group: str, gap: float, *, gain: bool, finished: bool = True) -> dict:
    return {
        "group": group,
        "finished": finished,
        "identity_ok": finished,
        "gain": gain,
        "gap": gap,
    }


class CohortTest(unittest.TestCase):
    def test_the_friend_is_absent_and_the_reference_is_separate(self) -> None:
        specs = cohort_specs()
        ids = [int(row["entry_id"]) for row in specs]
        self.assertNotIn(FRIEND_ID, ids)
        self.assertEqual(len([row for row in specs if row["group"] != "reference"]), COHORT_N)
        self.assertEqual(sum(row["group"] == "reference" for row in specs), 1)
        self.assertEqual(sum(row["group"] == "veteran" for row in specs), 7)
        self.assertEqual(sum(row["group"] == "rank" for row in specs), 7)

    def test_a_zero_residual_and_three_level_weeks_is_a_gain(self) -> None:
        weeks = [_week(2), _week(0, gw=2), _week(0, gw=3), _week(-1, gw=4), _week(1, gw=5)]
        result = manager_result(_spec(), weeks)
        self.assertTrue(result["gain"])
        self.assertTrue(result["residual_ok"])

    def test_a_non_zero_residual_is_not_a_gain(self) -> None:
        weeks = [_week(5, residual=1.0, gw=gw) for gw in range(1, 6)]
        result = manager_result(_spec(), weeks)
        self.assertFalse(result["gain"])
        self.assertFalse(result["residual_ok"])

    def test_a_late_wildcard_is_marked(self) -> None:
        weeks = [_week(1, gw=gw, chip="wildcard" if gw == 4 else None) for gw in range(1, 6)]
        result = manager_result(_spec(), weeks)
        self.assertEqual(result["late_wildcard"], [4])

    def test_eight_gains_and_a_positive_mean_is_ahead(self) -> None:
        rows = [_row("veteran", 1.0, gain=True) for _ in range(8)]
        rows += [_row("rank", -0.5, gain=False) for _ in range(6)]
        rows.append(_row("reference", -20.0, gain=False))
        self.assertTrue(batch_ahead(rows))

    def test_seven_gains_is_not_ahead(self) -> None:
        rows = [_row("veteran", 3.0, gain=True) for _ in range(7)]
        rows += [_row("rank", 1.0, gain=False) for _ in range(7)]
        self.assertFalse(batch_ahead(rows))

    def test_a_failure_blocks_the_batch(self) -> None:
        rows = [_row("veteran", 1.0, gain=True) for _ in range(13)]
        rows.append(_row("rank", None, gain=False, finished=False))
        self.assertFalse(batch_ahead(rows))


class SourceTest(unittest.TestCase):
    def test_the_module_does_not_copy_the_half_or_call_the_odds_api(self) -> None:
        import src.models.cohort_carry as cohort_carry

        source = inspect.getsource(cohort_carry)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("friend_start_gw15", source)
        self.assertNotIn("run_ft_season", source)


if __name__ == "__main__":
    unittest.main()
