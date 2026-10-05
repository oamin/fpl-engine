"""The price-reach reading, locked before the carry is opened."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.sale_reach import (
    classify_pair,
    decide,
    funded_sales,
    two_transfer_funds,
)
from src.models.season_climb_ft import HOLD_EPS, SquadState, _one_swap_candidates
from src.rules.fpl_2026 import sell_price


def _row(pid: str, position: str, value: int, score: float, club: str, *, eligible: bool) -> dict:
    return {
        "player_id": pid,
        "position": position,
        "value": value,
        "score_xp": score,
        "team_norm": club,
        "eligible": eligible,
    }


def _squad(
    *,
    purchase_mid: int = 40,
    mid_value: int = 40,
    bank: int = 0,
) -> tuple[SquadState, list[dict]]:
    """Fifteen owned players, one midfielder whose purchase price is set by the caller."""
    rows: list[dict] = []
    purchase: dict[str, int] = {}
    index = 0
    for position, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        for _ in range(count):
            pid = f"o{index}"
            value = mid_value if pid == "o7" else 40
            paid = purchase_mid if pid == "o7" else 40
            rows.append(_row(pid, position, value, 1.0, f"c{index}", eligible=False))
            purchase[pid] = paid
            index += 1
    return SquadState(purchase=purchase, bank=bank, ft=1), rows


def _pool(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class PurchaseTest(unittest.TestCase):
    def test_the_price_he_paid_is_what_funds_the_sale(self) -> None:
        state, rows = _squad(purchase_mid=40, mid_value=60, bank=0)
        self.assertEqual(sell_price(40, 60), 50)
        rows.append(_row("buy", "MID", 50, 8.0, "c99", eligible=True))
        pool = _pool(rows)
        sales = funded_sales(state, "buy", {str(r.player_id): r for r in pool.itertuples()}, {"o7": 1.0, "buy": 8.0})
        self.assertIn("o7", [sale for sale, _delta in sales])
        wiped = SquadState(purchase={pid: 0 for pid in state.purchase}, bank=0, ft=1)
        self.assertEqual(sell_price(0, 60), 30)
        missed = funded_sales(
            wiped, "buy", {str(r.player_id): r for r in pool.itertuples()}, {"o7": 1.0, "buy": 8.0}
        )
        self.assertNotIn("o7", [sale for sale, _delta in missed])

    def test_a_full_squad_at_his_club_is_not_a_funded_sale(self) -> None:
        state, rows = _squad(bank=20)
        for row in rows:
            if row["player_id"] in {"o8", "o9"}:
                row["team_norm"] = "c7"
        rows.append(_row("buy", "MID", 50, 8.0, "c7", eligible=True))
        pool = _pool(rows)
        by_id = {str(row.player_id): row for row in pool.itertuples()}
        scores = {str(row.player_id): float(row.score_xp) for row in pool.itertuples()}
        sales = [sale for sale, _delta in funded_sales(state, "buy", by_id, scores)]
        self.assertNotIn("o10", sales)
        self.assertIn("o7", sales)


class BeamTest(unittest.TestCase):
    def test_the_search_list_splits_seen_truncated_and_unfunded(self) -> None:
        state, rows = _squad(purchase_mid=40, mid_value=60, bank=0)
        rows.append(_row("buy", "MID", 50, 8.0, "c99", eligible=True))
        seen = classify_pair("buy", "o7", -4.0, state, _pool(rows))
        self.assertEqual(seen["klass"], "seen")
        self.assertGreater(float(seen["delta"]), HOLD_EPS)
        listed = _one_swap_candidates(state, _pool(rows), "score_xp", top_n=35)
        self.assertIn(("o7", "buy"), [(sold, bought) for sold, bought, _squad, _delta in listed])

        crowded = list(rows)
        for index in range(21):
            crowded.append(_row(f"b{index}", "MID", 40, 30.0 - index, f"x{index}", eligible=True))
        truncated = classify_pair("buy", "o7", -4.0, state, _pool(crowded))
        self.assertEqual(truncated["klass"], "truncated")
        self.assertGreater(int(truncated["rank"]), 20)

        dear, dear_rows = _squad(bank=0)
        dear_rows.append(_row("buy", "MID", 200, 8.0, "c99", eligible=True))
        blocked = classify_pair("buy", "o7", -4.0, dear, _pool(dear_rows))
        self.assertEqual(blocked["klass"], "no_fund")
        self.assertFalse(
            any(bought == "buy" for _sold, bought, _squad, _delta in _one_swap_candidates(dear, _pool(dear_rows), "score_xp", top_n=35))
        )

    def test_a_gap_of_1_25_stays_inside_the_hold(self) -> None:
        state, rows = _squad(bank=20)
        rows.append(_row("buy", "MID", 50, 1.0 + HOLD_EPS, "c99", eligible=True))
        labelled = classify_pair("buy", "o7", -2.0, state, _pool(rows))
        self.assertEqual(labelled["klass"], "delta_le_125")
        self.assertEqual(labelled["delta"], HOLD_EPS)

    def test_an_absent_player_is_the_gate(self) -> None:
        state, rows = _squad(bank=20)
        labelled = classify_pair("missing", "o7", -3.0, state, _pool(rows))
        self.assertEqual(labelled["klass"], "gate_block")
        self.assertEqual(labelled["status"], "absent")

    def test_an_owned_name_is_a_defect(self) -> None:
        state, rows = _squad()
        labelled = classify_pair("o7", "o8", -1.0, state, _pool(rows))
        self.assertTrue(labelled["defect"])
        self.assertEqual(decide([labelled]), "unverified")


class TwoSaleTest(unittest.TestCase):
    def test_a_second_sale_can_fund_what_one_sale_cannot(self) -> None:
        state, rows = _squad(bank=0)
        rows.append(_row("buy", "MID", 70, 8.0, "c99", eligible=True))
        rows.append(_row("cheap", "DEF", 10, 0.5, "c98", eligible=True))
        pool = _pool(rows)
        self.assertEqual(classify_pair("buy", "o7", -5.0, state, pool)["klass"], "no_fund")
        self.assertTrue(two_transfer_funds(state, "buy", pool))


class ReadingTest(unittest.TestCase):
    def _rows(self, pairs: list[tuple[str, float]]) -> list[dict]:
        return [{"klass": name, "points": points, "defect": False} for name, points in pairs]

    def test_the_order_is_gate_then_budget_then_truncation(self) -> None:
        self.assertEqual(decide(self._rows([("gate_block", -30.0), ("seen", -9.0)])), "gate")
        self.assertEqual(
            decide(self._rows([("no_fund", -15.0), ("delta_le_125", -10.0), ("seen", -14.0)])),
            "budget",
        )
        self.assertEqual(
            decide(self._rows([("truncated", -20.0), ("seen", -19.0)])),
            "truncation",
        )
        self.assertEqual(decide(self._rows([("seen", -30.0), ("truncated", -9.0)])), "seen")

    def test_a_sum_away_from_minus_39_is_inconclusive(self) -> None:
        self.assertEqual(decide(self._rows([("seen", -10.0)])), "inconclusive")

    def test_half_the_points_is_enough(self) -> None:
        self.assertEqual(
            decide(self._rows([("gate_block", -19.5), ("seen", -19.5)])),
            "gate",
        )
