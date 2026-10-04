"""Transfer-gap accounting. No squad is downloaded."""

from __future__ import annotations

import inspect
import unittest

import pandas as pd

from src.models.stage_46_transfer_gap import (
    COHORT,
    CONSISTENT_MIN,
    RANK_SLOTS,
    REFERENCE,
    VETERANS,
    assign_week_net,
    bought_held,
    chip_sale_split,
    consistency,
    decision_score,
    iter_squad_diffs,
    lookup_from_frame,
    means_by_manager,
    pair_gross,
    pair_within_position,
    pairs_from_entry_transfers,
    prior_inside,
    repeated_pairs,
    rows_for_week,
    score_paired_move,
    select_group,
    summarise_sales,
    week_net,
    window_gws,
    window_points,
)


class CohortLockTests(unittest.TestCase):
    def test_the_fifteen_ids_are_unique(self) -> None:
        ids = [row["entry_id"] for row in COHORT]
        self.assertEqual(len(ids), 15)
        self.assertEqual(len(set(ids)), 15)

    def test_each_veteran_has_two_top_10000_finishes(self) -> None:
        self.assertEqual(len(VETERANS), 7)
        for row in VETERANS:
            self.assertGreaterEqual(prior_inside(row["prior_ranks"]), 2)

    def test_rank_slots_are_the_locked_sort_keys(self) -> None:
        self.assertEqual(
            {row["rank_sort"] for row in RANK_SLOTS},
            {100, 500, 1000, 5000, 10000, 25000, 50000},
        )

    def test_the_reference_row_is_ojaminfc(self) -> None:
        self.assertEqual(REFERENCE["entry_id"], 2632584)
        self.assertEqual(REFERENCE["group"], "reference")

    def test_the_reference_row_stays_out_of_a_group(self) -> None:
        rows = [
            {"group": "veteran", "entry_id": 1},
            {"group": "reference", "entry_id": 2632584},
        ]
        chosen = select_group(rows, "veteran")
        self.assertEqual([row["entry_id"] for row in chosen], [1])


class PairingTests(unittest.TestCase):
    def test_the_highest_scores_are_paired(self) -> None:
        sold = [
            {"player_id": "low", "position": "MID", "score_xp": 1.0},
            {"player_id": "high", "position": "MID", "score_xp": 5.0},
        ]
        bought = [
            {"player_id": "b2", "position": "MID", "score_xp": 2.0},
            {"player_id": "b9", "position": "MID", "score_xp": 9.0},
        ]
        pairs = pair_within_position(sold, bought)
        self.assertEqual(pairs[0]["sold"]["player_id"], "high")
        self.assertEqual(pairs[0]["bought"]["player_id"], "b9")
        self.assertEqual(pairs[1]["sold"]["player_id"], "low")
        self.assertEqual(pairs[1]["bought"]["player_id"], "b2")

    def test_an_equal_score_breaks_on_player_id(self) -> None:
        sold = [
            {"player_id": "b", "position": "MID", "score_xp": 3.0},
            {"player_id": "a", "position": "MID", "score_xp": 3.0},
        ]
        bought = [{"player_id": "z", "position": "MID", "score_xp": 4.0}]
        pairs = pair_within_position(sold, bought)
        self.assertEqual(pairs[0]["sold"]["player_id"], "a")
        self.assertIsNone(pairs[1]["bought"])

    def test_a_cross_position_move_stays_unpaired(self) -> None:
        pairs = pair_within_position(
            [{"player_id": "m", "position": "MID", "score_xp": 3.0}],
            [{"player_id": "f", "position": "FWD", "score_xp": 5.0}],
        )
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(pair["sold"] is None or pair["bought"] is None for pair in pairs))

    def test_a_manager_move_stays_the_pair_the_entry_recorded(self) -> None:
        pairs = pairs_from_entry_transfers(
            [
                {
                    "gw": 2,
                    "sold_id": "2026-27:1",
                    "bought_id": "2026-27:2",
                    "sold_name": "A",
                    "bought_name": "B",
                    "sold_position": "MID",
                    "bought_position": "FWD",
                }
            ]
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["sold"]["position"], "MID")
        self.assertEqual(pairs[0]["bought"]["position"], "FWD")


class WindowTests(unittest.TestCase):
    def test_the_window_stops_at_the_last_gameweek(self) -> None:
        self.assertEqual(window_gws(2, 5, 3), [2, 3, 4])
        self.assertEqual(window_gws(4, 5, 3), [4, 5])
        self.assertEqual(window_gws(5, 5, 3), [5])

    def test_a_missing_week_scores_zero(self) -> None:
        self.assertEqual(window_points("p", [2, 3, 4], {("p", 2): 5.0}), 5.0)

    def test_a_sale_carries_the_latest_earlier_score(self) -> None:
        scores = {("p", 1): 2.0, ("p", 3): 4.0}
        self.assertEqual(decision_score("p", 3, scores), 4.0)
        self.assertEqual(decision_score("p", 2, scores), 2.0)
        self.assertEqual(decision_score("q", 2, scores), 0.0)

    def test_the_buy_gate_score_overwrites_the_early_score(self) -> None:
        feat = pd.DataFrame({"player_id": ["p"], "gw": [2], "score_xp": [5.0]})
        feat.attrs["early_scores"] = (("p", 2, 1.0), ("p", 3, 1.5))
        lookup = lookup_from_frame(feat)
        self.assertEqual(lookup[("p", 2)], 5.0)
        self.assertEqual(lookup[("p", 3)], 1.5)

    def test_a_gap_in_the_trace_is_not_a_transfer(self) -> None:
        diffs = iter_squad_diffs([(1, {"a", "b"}), (3, {"a", "c"})])
        self.assertTrue(diffs[0]["gap"])
        self.assertEqual(diffs[0]["sold"], set())

    def test_consecutive_weeks_record_the_sale(self) -> None:
        diffs = iter_squad_diffs([(1, {"a", "b"}), (2, {"a", "c"})])
        self.assertEqual(diffs[0]["sold"], {"b"})
        self.assertEqual(diffs[0]["bought"], {"c"})
        self.assertEqual(iter_squad_diffs([(1, {"a"}), (2, {"a"})]), [])


class AccountingTests(unittest.TestCase):
    def test_the_hit_is_added_once_for_the_week(self) -> None:
        net = week_net([10.0, 4.0], [3.0, 1.0], 4.0)
        self.assertEqual(net, 14.0)
        self.assertEqual(pair_gross(10.0, 3.0) + pair_gross(4.0, 1.0), 10.0)
        rows = [{}, {}]
        assign_week_net(rows, [10.0, 4.0], [3.0, 1.0], 4.0)
        self.assertEqual(sum(row["hit_on_row"] for row in rows), 4.0)
        self.assertEqual(rows[0]["week_net"], 14.0)
        self.assertIsNone(rows[1]["week_net"])

    def test_a_short_window_is_flagged_and_the_kept_player_is_marked(self) -> None:
        row = score_paired_move(
            gw=4,
            last_gw=5,
            sold_id="s",
            bought_id="b",
            points={("s", 4): 3.0, ("s", 5): 2.0, ("b", 4): 1.0},
            later_squads={5: {"b"}},
            manager_squad={"s"},
        )
        self.assertEqual(row["gross"], 4.0)
        self.assertTrue(row["short"])
        self.assertEqual(row["window_len"], 2)
        self.assertTrue(row["manager_kept"])
        self.assertTrue(row["bought_held"])

    def test_the_sale_week_itself_counts_as_held(self) -> None:
        row = score_paired_move(
            gw=5,
            last_gw=5,
            sold_id="s",
            bought_id="b",
            points={("s", 5): 8.0, ("b", 5): 1.0},
            later_squads={},
            manager_squad=set(),
        )
        self.assertTrue(row["bought_held"])
        self.assertFalse(row["manager_kept"])
        self.assertEqual(row["gross"], 7.0)
        self.assertTrue(bought_held("p", [], {}))
        self.assertFalse(bought_held("p", [3], {3: set()}))

    def test_an_unpaired_move_stays_out_of_the_mean(self) -> None:
        rows, week = rows_for_week(
            gw=2,
            last_gw=5,
            pairs=pair_within_position(
                [{"player_id": "m", "position": "MID", "score_xp": 3.0, "name": "M"}],
                [{"player_id": "f", "position": "FWD", "score_xp": 5.0, "name": "F"}],
            ),
            sold_ids=["m"],
            bought_ids=["f"],
            points={("m", 2): 10.0, ("f", 2): 4.0},
            later_squads={3: {"f"}, 4: {"f"}},
            manager_squad={"m"},
            hit_cost=0.0,
        )
        summary = summarise_sales(rows)
        self.assertEqual(summary["n_sales"], 0)
        self.assertEqual(summary["n_unpaired"], 2)
        self.assertIsNone(summary["mean_gross"])
        self.assertEqual(week["week_net"], 6.0)

    def test_the_kept_subset_is_separate_from_the_pooled_mean(self) -> None:
        rows = [
            {"paired": True, "gross": 4.0, "manager_kept": True, "short": False, "bought_held": True},
            {"paired": True, "gross": -2.0, "manager_kept": False, "short": True, "bought_held": False},
            {"paired": False, "gross": None, "manager_kept": True, "short": False, "bought_held": None},
        ]
        summary = summarise_sales(rows)
        self.assertEqual(summary["n_sales"], 2)
        self.assertEqual(summary["mean_gross"], 1.0)
        self.assertEqual(summary["n_kept"], 1)
        self.assertEqual(summary["mean_gross_kept"], 4.0)
        self.assertEqual(summary["n_short"], 1)
        self.assertEqual(summary["n_churned"], 1)

    def test_manager_means_ignore_a_manager_with_no_pair(self) -> None:
        rows = [
            {"entry_id": 1, "paired": True, "gross": 4.0},
            {"entry_id": 1, "paired": True, "gross": 0.0},
            {"entry_id": 2, "paired": True, "gross": -3.0},
            {"entry_id": 3, "paired": False, "gross": None},
        ]
        self.assertEqual(means_by_manager(rows), [2.0, -3.0])


class ConsistencyTests(unittest.TestCase):
    def test_five_against_two_is_consistent(self) -> None:
        self.assertEqual(
            consistency([1, 1, 1, 1, 1, -1, -1]),
            "consistent sold ahead",
        )

    def test_four_against_three_is_mixed(self) -> None:
        self.assertEqual(consistency([1, 1, 1, 1, -1, -1, -1]), "mixed")

    def test_three_managers_are_too_small(self) -> None:
        self.assertEqual(consistency([1, 1, 1, -1]), "too small")
        self.assertEqual(consistency([]), "too small")
        self.assertGreaterEqual(CONSISTENT_MIN, 4)

    def test_a_one_sided_group_of_four_is_consistent(self) -> None:
        self.assertEqual(consistency([1, 1, 1, 1]), "consistent sold ahead")
        self.assertEqual(consistency([-1, -1, -1, -1, 1]), "consistent bought ahead")


class ChipFlagTests(unittest.TestCase):
    def test_wildcard_and_free_hit_are_split_from_the_bank(self) -> None:
        rows = [
            {"paired": True, "gross": 2.0, "chip": "wildcard"},
            {"paired": True, "gross": -4.0, "chip": "free_hit"},
            {"paired": True, "gross": -6.0, "chip": None},
            {"paired": True, "gross": -8.0, "chip": "triple_captain"},
        ]
        split = chip_sale_split(rows)
        self.assertEqual(split["n_rebuild"], 2)
        self.assertEqual(split["mean_rebuild"], -1.0)
        self.assertEqual(split["n_other"], 2)
        self.assertEqual(split["mean_other"], -7.0)

    def test_a_sale_from_two_squads_is_listed_once(self) -> None:
        rows = [
            {
                "group": "veteran",
                "side": "model",
                "paired": True,
                "sold_name": "Isak",
                "bought_name": "Thiago",
            },
            {
                "group": "rank",
                "side": "model",
                "paired": True,
                "sold_name": "Isak",
                "bought_name": "Thiago",
            },
            {
                "group": "reference",
                "side": "model",
                "paired": True,
                "sold_name": "Cherki",
                "bought_name": "Szoboszlai",
            },
            {
                "group": "veteran",
                "side": "manager",
                "paired": True,
                "sold_name": "Isak",
                "bought_name": "Thiago",
            },
        ]
        self.assertEqual(repeated_pairs(rows), [("Isak", "Thiago", 2)])


class WriterGuardTests(unittest.TestCase):
    def test_the_diagnostic_does_not_call_the_benchmark_writer(self) -> None:
        import src.models.stage_46_transfer_gap as diagnostic

        source = inspect.getsource(diagnostic)
        self.assertNotIn("benchmark.run", source)
        self.assertNotIn("live_benchmark", source)


if __name__ == "__main__":
    unittest.main()
