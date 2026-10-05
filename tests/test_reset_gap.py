"""Reset-to-his-squad measurement. No season climb and no network."""

from __future__ import annotations

import inspect
import unittest

from src.models.reset_gap import (
    assert_not_carried,
    bank_before,
    beats_five,
    classify_gap,
    horizon_gws,
    model_chip,
    phrases,
    purchases_before,
)


def _entry() -> dict:
    opening = [{"id": i, "name": f"P{i}"} for i in range(1, 16)]
    gw1_xi = [{"id": i, "name": f"P{i}"} for i in range(1, 12)]
    gw1_bench = [{"id": i, "name": f"P{i}"} for i in range(12, 16)]
    gw2_xi = [{"id": i, "name": f"P{i}"} for i in range(2, 12)] + [
        {"id": 16, "name": "New"}
    ]
    return {
        "opening_squad": opening,
        "transfers": [
            {
                "gw": 2,
                "out_id": 1,
                "in_id": 16,
                "in_cost": 55,
                "out_cost": 40,
            }
        ],
        "gameweeks": [
            {
                "gw": 1,
                "bank": 0,
                "ft_available": 0,
                "chip": "triple_captain",
                "points": 50,
                "transfer_cost": 0,
                "xi": gw1_xi,
                "bench": gw1_bench,
            },
            {
                "gw": 2,
                "bank": 5,
                "ft_available": 1,
                "chip": None,
                "points": 40,
                "transfer_cost": 0,
                "xi": gw2_xi,
                "bench": [{"id": i, "name": f"P{i}"} for i in range(12, 16)],
            },
        ],
    }


class BarTest(unittest.TestCase):
    def test_a_positive_sum_with_three_weeks_is_a_gain(self) -> None:
        self.assertTrue(beats_five([5, 1, 1, -2, -1]))

    def test_two_winning_weeks_are_not_enough(self) -> None:
        self.assertFalse(beats_five([20, 20, -1, -1, -1]))

    def test_a_non_positive_sum_is_not_a_gain(self) -> None:
        self.assertFalse(beats_five([0, 0, 0, 0, 0]))
        self.assertFalse(beats_five([5, 5, 5, -20, 0]))


class HorizonTest(unittest.TestCase):
    def test_a_blank_is_skipped(self) -> None:
        clubs = {1: {"a"}, 3: {"a"}, 4: {"a"}}
        self.assertEqual(horizon_gws(1, clubs), [1, 3, 4])


class StateTest(unittest.TestCase):
    def test_a_buy_is_owned_only_after_that_deadline(self) -> None:
        entry = _entry()
        self.assertEqual(bank_before(entry, 1), 0)
        self.assertEqual(bank_before(entry, 2), 0)
        prices = {i: 50 for i in range(1, 16)}
        before = purchases_before(entry, prices, 2)
        self.assertIn("2026-27:1", before)
        self.assertNotIn("2026-27:16", before)
        entry["gameweeks"].append(
            {
                "gw": 3,
                "bank": 5,
                "ft_available": 1,
                "chip": None,
                "points": 40,
                "transfer_cost": 0,
                "xi": entry["gameweeks"][1]["xi"],
                "bench": entry["gameweeks"][1]["bench"],
            }
        )
        after = purchases_before(entry, prices, 3)
        self.assertEqual(after["2026-27:16"], 55)
        self.assertNotIn("2026-27:1", after)
        self.assertEqual(len(after), 15)

    def test_a_gameweek_1_transfer_is_reversed_out_of_the_bank(self) -> None:
        entry = _entry()
        entry["transfers"].append(
            {"gw": 1, "out_id": 2, "in_id": 20, "in_cost": 70, "out_cost": 50}
        )
        entry["gameweeks"][0]["bank"] = 10
        # 10 - 50 + 70 = 30 before the deals
        self.assertEqual(bank_before(entry, 1), 30)

    def test_only_triple_captain_is_mirrored(self) -> None:
        self.assertEqual(model_chip("triple_captain"), "triple_captain")
        self.assertIsNone(model_chip(None))
        with self.assertRaises(RuntimeError):
            model_chip("wildcard")
        with self.assertRaises(RuntimeError):
            model_chip("bench_boost")
        with self.assertRaises(RuntimeError):
            model_chip("free_hit")


class SplitTest(unittest.TestCase):
    def test_a_buy_is_a_transfer_and_a_shared_player_is_a_lineup(self) -> None:
        pre = {"a", "b", "d"}
        model_squad = {"a", "c", "d"}
        their_squad = {"a", "b", "d"}
        model_final = {"c": 6.0, "d": 3.0}
        their_final = {"b": 1.0, "a": 4.0}
        split = classify_gap(pre, model_squad, their_squad, model_final, their_final)
        self.assertEqual(split["transfer_model"], ["c"])
        self.assertEqual(split["transfer_their"], ["b"])
        self.assertEqual(split["lineup_model"], ["d"])
        self.assertEqual(split["lineup_their"], ["a"])
        self.assertEqual(split["transfer_gap"], 5.0)
        self.assertEqual(split["lineup_gap"], -1.0)

    def test_the_sentences_sum_to_the_gap(self) -> None:
        row = {
            "captain_gap": 6.0,
            "transfer_gap": -3.0,
            "lineup_gap": 2.0,
            "hit_gap": -4.0,
            "residual": 0.0,
            "model_captain": "Haaland",
            "their_captain": "Salah",
            "transfer_model": ["c"],
            "transfer_their": ["b"],
            "lineup_model": ["a"],
            "lineup_their": ["d"],
            "gap": 1.0,
        }
        names = {"a": "Rogers", "b": "Pedro", "c": "Watkins", "d": "Gordon"}
        text = phrases(row, names)
        signed = []
        for line in text:
            token = [part for part in line.replace(".", " ").split() if part[:1] in "+-"][-1]
            signed.append(float(token))
        self.assertEqual(signed, [6.0, -3.0, 2.0, -4.0])
        self.assertAlmostEqual(sum(signed), row["gap"])
        self.assertTrue(text[0].startswith("Captaincy on Haaland vs Salah"))
        self.assertIn("Transferring Watkins in for Pedro", text[1])
        self.assertIn("Starting Rogers over Gordon", text[2])

    def test_a_carried_squad_fails(self) -> None:
        rows = [
            {"gw": 1, "pre_ids": ["a"], "model_ids": ["b"], "their_ids": ["a"]},
            {"gw": 2, "pre_ids": ["b"], "model_ids": ["b"], "their_ids": ["a"]},
        ]
        with self.assertRaises(RuntimeError):
            assert_not_carried(rows)

    def test_his_squad_is_accepted(self) -> None:
        rows = [
            {"gw": 1, "pre_ids": ["a"], "model_ids": ["b"], "their_ids": ["a"]},
            {"gw": 2, "pre_ids": ["a"], "model_ids": ["c"], "their_ids": ["a"]},
        ]
        assert_not_carried(rows)


class SourceTest(unittest.TestCase):
    def test_the_module_does_not_climb_or_call_the_odds_api(self) -> None:
        import src.models.reset_gap as reset_gap

        source = inspect.getsource(reset_gap)
        self.assertNotIn("run_ft_season", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("refresh_lines", source)
        self.assertNotIn("half_plan_scores.csv", source)
        self.assertNotIn("matrix.json", source)


if __name__ == "__main__":
    unittest.main()
