"""Carry and move accounting for the string agent. No live model, no ledger write."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from src.rules.fpl_2026 import sell_price
from src.str_agent import freeze as fz
from src.str_agent.carry import CarryState, validate_move
from src.str_agent.extractor import build_context
from src.str_agent.validator import validate_draft
from tests.test_str_agent import _directory, _draft


OWNED = [
    "g1", "g2",
    "d0", "d1", "d2", "d3", "d4",
    "m0", "m1", "m2", "m3", "m4",
    "f0", "f1", "f2",
]
XI = ["g1", "d0", "d1", "d2", "d3", "m0", "m1", "m2", "m3", "f0", "f1"]
BENCH = ["g2", "d4", "m4", "f2"]


def _market() -> dict[str, dict]:
    rows = [
        ("g1", "GKP", "A", 45),
        ("g2", "GKP", "B", 45),
        ("gX", "GKP", "Z", 45),
        ("d0", "DEF", "C", 40),
        ("d1", "DEF", "C", 40),
        ("d2", "DEF", "C", 40),
        ("d3", "DEF", "D", 40),
        ("d4", "DEF", "D", 40),
        ("dX", "DEF", "Z", 40),
        ("dY", "DEF", "Y", 40),
        ("m0", "MID", "E", 55),
        ("m1", "MID", "E", 55),
        ("m2", "MID", "E", 55),
        ("m3", "MID", "F", 55),
        ("m4", "MID", "F", 55),
        ("mX", "MID", "Y", 55),
        ("f0", "FWD", "G", 60),
        ("f1", "FWD", "G", 60),
        ("f2", "FWD", "H", 60),
        ("fX", "FWD", "Z", 60),
        ("fY", "FWD", "Y", 70),
    ]
    return {
        pid: {"position": pos, "club": club, "now_cost": cost}
        for pid, pos, club, cost in rows
    }


def _carry(
    *,
    ft: int = 1,
    bank: int = 15,
    purchase: dict[str, int] | None = None,
    selling: dict[str, int] | None = None,
    chips: dict[int, str] | None = None,
    gw: int = 6,
) -> CarryState:
    prices = {pid: _market()[pid]["now_cost"] for pid in OWNED}
    if purchase:
        prices.update(purchase)
    return CarryState(
        gw=gw,
        squad=tuple(OWNED),
        purchase_prices=prices,
        bank=bank,
        ft_before=ft,
        chips_played=chips or {},
        selling_prices=selling or {},
    )


def _move(
    *,
    outs: list[str],
    ins: list[str],
    chip: str | None = None,
    captain: str = "g1",
    vice: str = "d0",
) -> dict:
    squad = [pid for pid in OWNED if pid not in set(outs)] + list(ins)
    xi = [ins[outs.index(pid)] if pid in outs else pid for pid in XI]
    bench = [ins[outs.index(pid)] if pid in outs else pid for pid in BENCH]
    return {
        "decision": {
            "chip_played": chip,
            "squad_15": squad,
            "starting_11": xi,
            "captain": captain,
            "vice_captain": vice,
            "bench_order": bench,
            "transfers_in": list(ins),
            "transfers_out": list(outs),
        }
    }


class CarryMove(unittest.TestCase):
    def test_free_hit_reverts_squad_bank_and_price(self) -> None:
        directory = _market()
        directory["f2"] = {**directory["f2"], "now_cost": 60}
        directory["fX"] = {**directory["fX"], "now_cost": 70}
        carry = _carry(ft=2, bank=15, purchase={"f2": 50})
        result = validate_move(_move(outs=["f2"], ins=["fX"], chip="free_hit"), carry, directory)
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.bank_after, 0)
        self.assertEqual(result.hits, 0)
        nxt = result.carry_after
        assert nxt is not None
        self.assertIn("f2", nxt["squad"])
        self.assertNotIn("fX", nxt["squad"])
        self.assertEqual(nxt["bank"], 15)
        self.assertEqual(nxt["purchase_prices"]["f2"], 50)
        self.assertEqual(nxt["ft_before"], 2)

    def test_free_hit_five_swaps_do_not_spend_ft(self) -> None:
        outs = ["g2", "d4", "d3", "m4", "f2"]
        ins = ["gX", "dX", "dY", "mX", "fX"]
        result = validate_move(
            _move(outs=outs, ins=ins, chip="free_hit"),
            _carry(ft=2, bank=15),
            _market(),
        )
        self.assertEqual(result.errors, [], result.errors)
        assert result.carry_after is not None
        self.assertEqual(result.carry_after["ft_before"], 2)
        self.assertEqual(result.carry_after["squad"], OWNED)

    def test_half_profit_rounds_down(self) -> None:
        directory = _market()
        directory["f2"] = {**directory["f2"], "now_cost": 59}
        directory["fX"] = {**directory["fX"], "now_cost": 54}
        self.assertEqual(sell_price(50, 59), 54)
        result = validate_move(
            _move(outs=["f2"], ins=["fX"]),
            _carry(bank=0, purchase={"f2": 50}),
            directory,
        )
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.bank_after, 0)

    def test_price_fall_sells_at_current(self) -> None:
        directory = _market()
        directory["f2"] = {**directory["f2"], "now_cost": 58}
        directory["fX"] = {**directory["fX"], "now_cost": 59}
        result = validate_move(
            _move(outs=["f2"], ins=["fX"]),
            _carry(bank=0, purchase={"f2": 60}),
            directory,
        )
        self.assertEqual(result.bank_after, -1)
        self.assertTrue(any("bank_after is -1" in error for error in result.errors))
        self.assertIsNone(result.carry_after)

    def test_missing_price_fails_closed(self) -> None:
        carry = _carry()
        prices = dict(carry.purchase_prices)
        del prices["f2"]
        carry = CarryState(
            gw=carry.gw,
            squad=carry.squad,
            purchase_prices=prices,
            bank=carry.bank,
            ft_before=carry.ft_before,
            chips_played={},
            selling_prices={},
        )
        result = validate_move(_move(outs=["f2"], ins=["fX"]), carry, _market())
        self.assertTrue(any("no purchase price" in error for error in result.errors))
        self.assertIsNone(result.carry_after)

    def test_same_player_in_and_out(self) -> None:
        draft = _move(outs=[], ins=[])
        draft["decision"]["transfers_in"] = ["f2"]
        draft["decision"]["transfers_out"] = ["f2"]
        result = validate_move(draft, _carry(), _market())
        self.assertTrue(any("both" in error for error in result.errors))

    def test_duplicate_transfer_in(self) -> None:
        draft = _move(outs=["f2"], ins=["fX"])
        draft["decision"]["transfers_in"] = ["fX", "fX"]
        result = validate_move(draft, _carry(), _market())
        self.assertTrue(any("duplicate" in error for error in result.errors))

    def test_undeclared_swap(self) -> None:
        draft = _move(outs=["f2"], ins=["fX"])
        draft["decision"]["transfers_in"] = []
        draft["decision"]["transfers_out"] = []
        result = validate_move(draft, _carry(), _market())
        self.assertTrue(any("must match" in error for error in result.errors))

    def test_captain_cannot_also_be_vice(self) -> None:
        draft = _move(outs=[], ins=[])
        draft["decision"]["vice_captain"] = "g1"
        errors = validate_draft(draft, directory=_market(), budget=1000)
        self.assertTrue(any("must differ" in error for error in errors))

    def test_vice_on_the_bench_fails(self) -> None:
        draft = _move(outs=[], ins=[])
        draft["decision"]["vice_captain"] = "f2"
        errors = validate_draft(draft, directory=_market(), budget=1000)
        self.assertTrue(any("vice_captain must be in starting_11" in error for error in errors))

    def test_hits_and_next_ft_on_an_ordinary_week(self) -> None:
        outs = ["g2", "d4", "m4", "f2"]
        ins = ["gX", "dX", "mX", "fX"]
        result = validate_move(_move(outs=outs, ins=ins), _carry(ft=2), _market())
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.hits, 8)
        assert result.carry_after is not None
        self.assertEqual(result.carry_after["ft_before"], 1)
        self.assertEqual(set(result.carry_after["squad"]), set(OWNED) - set(outs) | set(ins))

    def test_wildcard_is_free_and_does_not_grant_a_transfer(self) -> None:
        outs = ["g2", "d4", "d3", "m4", "f2"]
        ins = ["gX", "dX", "dY", "mX", "fX"]
        result = validate_move(
            _move(outs=outs, ins=ins, chip="wildcard"),
            _carry(ft=1),
            _market(),
        )
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.hits, 0)
        assert result.carry_after is not None
        self.assertEqual(result.carry_after["ft_before"], 1)
        self.assertIn("fX", result.carry_after["squad"])
        self.assertEqual(result.carry_after["purchase_prices"]["fX"], 60)
        self.assertEqual(result.carry_after["purchase_prices"]["f0"], _market()["f0"]["now_cost"])

    def test_bench_boost_still_charges_hits(self) -> None:
        outs = ["g2", "d4", "f2"]
        ins = ["gX", "dX", "fX"]
        result = validate_move(
            _move(outs=outs, ins=ins, chip="bench_boost"),
            _carry(ft=1),
            _market(),
        )
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.hits, 8)
        assert result.carry_after is not None
        self.assertIn("fX", result.carry_after["squad"])

    def test_back_to_back_free_hit_is_illegal(self) -> None:
        result = validate_move(
            _move(outs=["f2"], ins=["fX"], chip="free_hit"),
            _carry(chips={5: "free_hit"}),
            _market(),
        )
        self.assertTrue(any("not available" in error for error in result.errors))
        self.assertIsNone(result.carry_after)

    def test_context_includes_rolling_avg_and_not_the_numeric_score(self) -> None:
        text = build_context(
            gw=6,
            deadline_utc="2026-10-10T10:00:00Z",
            roster=[
                {
                    "player_id": 1,
                    "name": "Hold",
                    "position": "MID",
                    "club": "MCI",
                    "now_cost": 50,
                    "owned": True,
                }
            ],
            bank=15,
            ft=1,
            chips_left=["wildcard"],
            focus_ids=[1],
            directory={1: {"name": "Hold", "position": "MID", "club": "MCI", "prior": None}},
            packets=[],
            minutes={1: [90, 0, 90]},
        )
        self.assertIn("rolling avg", text)
        self.assertIn("60.0", text)
        self.assertNotIn("score_xp", text)

    def test_official_ledger_stays_closed(self) -> None:
        with self.assertRaises(fz.StringFreezeError):
            fz.write_string_freeze(_draft())

    def test_cli_hold_is_legal_and_commit_is_refused(self) -> None:
        from src.str_agent.__main__ import main

        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["--entry", "2632584", "--hold"])
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["is_legal"])
        self.assertEqual(payload["hits"], 0)
        self.assertEqual(payload["bank_after"], 15)
        self.assertEqual(payload["carry_after"]["ft_before"], 2)
        self.assertEqual(payload["carry_after"]["squad"], payload["carry"]["squad"])
        self.assertEqual(payload["decision"]["transfers_in"], [])
        err = io.StringIO()
        with redirect_stderr(err):
            refused = main(["--entry", "2632584", "--hold", "--commit"])
        self.assertEqual(refused, 2)
        self.assertIn("closed", err.getvalue())

    def test_static_budget_check_still_uses_the_cap(self) -> None:
        errors = validate_draft(
            {"decision": _draft()["decision"]},
            directory=_directory(),
            budget=100,
        )
        self.assertTrue(any("budget" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
