"""Three-week string plan. No live model and no official ledger write."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.str_agent import freeze as fz
from src.str_agent import runner as rn
from src.str_agent.carry import CarryState
from src.str_agent.horizon import (
    FIRST_PAPER_GW,
    MissingPriorPlanError,
    followup_context,
    resolve_start,
    validate_horizon,
    write_plan,
)
from tests.test_str_agent import _draft
from tests.test_str_agent_carry import BENCH, OWNED, XI, _carry, _market, _move


def _week_from_move(move: dict, gw: int) -> dict:
    week = dict(move["decision"])
    week["gw"] = gw
    return week


def _hold(gw: int, chip: str | None = None) -> dict:
    return _week_from_move(_move(outs=[], ins=[], chip=chip), gw)


def _shift(current: list[str], xi: list[str], bench: list[str], outs: list[str], ins: list[str], gw: int, chip: str | None = None) -> tuple[list[str], list[str], list[str], dict]:
    squad = [pid for pid in current if pid not in set(outs)] + list(ins)
    xi2 = [ins[outs.index(pid)] if pid in outs else pid for pid in xi]
    bench2 = [ins[outs.index(pid)] if pid in outs else pid for pid in bench]
    week = {
        "gw": gw,
        "chip_played": chip,
        "squad_15": squad,
        "starting_11": xi2,
        "captain": "g1",
        "vice_captain": "d0",
        "bench_order": bench2,
        "transfers_in": list(ins),
        "transfers_out": list(outs),
    }
    return squad, xi2, bench2, week


class HorizonChain(unittest.TestCase):
    def test_week_zero_must_match_the_decision(self) -> None:
        hold = _hold(6)
        decision = dict(hold)
        decision.pop("gw")
        bad = dict(hold)
        bad["captain"] = "d0"
        result = validate_horizon(decision, [bad, _hold(7), _hold(8)], _carry(), _market())
        self.assertTrue(result.errors)
        self.assertEqual(result.steps, ())

    def test_an_illegal_later_week_saves_nothing(self) -> None:
        directory = _market()
        directory["fY"] = {**directory["fY"], "now_cost": 76}
        w0 = _hold(6)
        decision = dict(w0)
        decision.pop("gw")
        _squad, _xi, _bench, w2 = _shift(OWNED, XI, BENCH, ["f2"], ["fY"], 8)
        result = validate_horizon(decision, [w0, _hold(7), w2], _carry(), directory)
        self.assertTrue(any("bank_after" in error for error in result.errors))
        self.assertEqual(result.steps, ())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(Exception):
                write_plan(
                    gw=6,
                    deadline_utc="2026-10-10T10:00:00Z",
                    frozen_at_utc="2026-10-10T09:00:00Z",
                    model_id="test",
                    prompt_sha256="a" * 64,
                    context="notes",
                    rationale="no",
                    decision=decision,
                    horizon=[w0, _hold(7), w2],
                    carry=_carry(),
                    result=result,
                    root=root,
                )
            self.assertEqual(list(root.glob("*.json")), [])

    def test_free_hit_restores_the_sold_player(self) -> None:
        w0 = _week_from_move(_move(outs=["f0"], ins=["fX"], chip="free_hit"), 6)
        decision = dict(w0)
        decision.pop("gw")
        result = validate_horizon(decision, [w0, _hold(7), _hold(8)], _carry(bank=15), _market())
        self.assertEqual(result.errors, [])
        restored = result.steps[0].carry_after
        self.assertIn("f0", restored["squad"])
        self.assertNotIn("fX", restored["squad"])
        self.assertEqual(restored["bank"], 15)

    def test_free_transfers_roll_then_reset_after_three_moves(self) -> None:
        w0 = _hold(6)
        decision = dict(w0)
        decision.pop("gw")
        _squad, _xi, _bench, w2 = _shift(OWNED, XI, BENCH, ["g2", "d4", "m4"], ["gX", "dX", "mX"], 8)
        result = validate_horizon(decision, [w0, _hold(7), w2], _carry(ft=1), _market())
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.steps[0].carry_after["ft_before"], 2)
        self.assertEqual(result.steps[1].carry_after["ft_before"], 3)
        self.assertEqual(result.steps[2].hits, 0)
        self.assertEqual(result.steps[2].carry_after["ft_before"], 1)

    def test_wildcard_hits_do_not_leak_into_the_next_week(self) -> None:
        directory = _market()
        directory["mZ"] = {"position": "MID", "club": "W", "now_cost": 55}
        outs = ["g2", "d4", "m4", "f1", "f2"]
        ins = ["gX", "dY", "mX", "fX", "fY"]
        squad, xi, bench, w0 = _shift(OWNED, XI, BENCH, outs, ins, 6, "wildcard")
        decision = dict(w0)
        decision.pop("gw")
        # gX stays. Selling him would leave one goalkeeper.
        _squad, _xi, _bench, w1 = _shift(squad, xi, bench, ["fY", "mX"], ["f2", "mZ"], 7)
        result = validate_horizon(decision, [w0, w1, _hold_from(8, _squad, _xi, _bench)], _carry(bank=15), directory)
        self.assertEqual(result.errors, [], result.errors)
        self.assertEqual(result.steps[0].hits, 0)
        self.assertEqual(result.steps[0].carry_after["ft_before"], 1)
        self.assertEqual(result.steps[1].hits, 4)

    def test_the_same_chip_cannot_be_played_twice_in_the_horizon(self) -> None:
        w0 = _hold(6, "bench_boost")
        decision = dict(w0)
        decision.pop("gw")
        result = validate_horizon(decision, [w0, _hold(7, "bench_boost"), _hold(8)], _carry(), _market())
        self.assertTrue(result.errors)
        self.assertEqual(result.steps, ())


def _hold_from(gw: int, squad: list[str], xi: list[str], bench: list[str]) -> dict:
    _squad, _xi, _bench, week = _shift(squad, xi, bench, [], [], gw)
    return week


class PaperTeam(unittest.TestCase):
    def test_a_later_week_ignores_the_human_entry(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = {
                "carry_after": _carry().as_dict() | {"gw": 7, "squad": ["paper-player"]},
            }
            path = root / "gw06.json"
            path.write_text(json.dumps(saved), encoding="utf-8")
            entry = _carry(gw=7)
            start = resolve_start(7, entry, root)
        self.assertEqual(list(start.squad), ["paper-player"])
        self.assertNotEqual(list(start.squad), list(entry.squad))

    def test_a_missing_plan_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(MissingPriorPlanError):
                resolve_start(7, _carry(gw=7), Path(folder))

    def test_gameweek_six_can_start_from_the_entry(self) -> None:
        entry = _carry()
        self.assertIs(resolve_start(FIRST_PAPER_GW, entry, Path(tempfile.mkdtemp())), entry)


class LedgerGuards(unittest.TestCase):
    def _row(self) -> dict:
        return _draft()

    def test_a_late_commit_writes_nothing(self) -> None:
        row = self._row()
        row["frozen_at_utc"] = "2026-10-10T10:00:00Z"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "string_agent_freeze.jsonl"
            with self.assertRaises(fz.StringFreezeError):
                fz.commit_official(row, plan_sha256="c" * 64, path=path)
            self.assertFalse(path.exists())

    def test_a_second_commit_for_the_week_is_refused(self) -> None:
        row = self._row()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "string_agent_freeze.jsonl"
            fz.commit_official(row, plan_sha256="c" * 64, path=path)
            with self.assertRaises(fz.DuplicateFreezeError):
                fz.commit_official(row, plan_sha256="c" * 64, path=path)

    def test_a_saved_plan_does_not_touch_the_official_ledger(self) -> None:
        w0 = _hold(6)
        decision = dict(w0)
        decision.pop("gw")
        result = validate_horizon(decision, [w0, _hold(7), _hold(8)], _carry(), _market())
        self.assertEqual(result.errors, [])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _path, digest = write_plan(
                gw=6,
                deadline_utc="2026-10-10T10:00:00Z",
                frozen_at_utc="2026-10-10T09:00:00Z",
                model_id="test",
                prompt_sha256="a" * 64,
                context="Press notes only.",
                rationale="Hold.",
                decision=decision,
                horizon=[w0, _hold(7), _hold(8)],
                carry=_carry(),
                result=result,
                root=root,
            )
            self.assertEqual(len(digest), 64)
            self.assertFalse(fz.LEDGER.exists())
            text = followup_context(json.loads((root / "gw06.json").read_text(encoding="utf-8")))
            for token in ("score_xp", "xp_on_pot", "lam_scored"):
                self.assertNotIn(token, text)
                self.assertNotIn(token, (root / "gw06.json").read_text(encoding="utf-8"))

    def test_a_score_column_in_the_context_is_refused(self) -> None:
        w0 = _hold(6)
        decision = dict(w0)
        decision.pop("gw")
        result = validate_horizon(decision, [w0, _hold(7), _hold(8)], _carry(), _market())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(Exception):
                write_plan(
                    gw=6,
                    deadline_utc="2026-10-10T10:00:00Z",
                    frozen_at_utc="2026-10-10T09:00:00Z",
                    model_id="test",
                    prompt_sha256="a" * 64,
                    context="his score_xp is 4",
                    rationale="no",
                    decision=decision,
                    horizon=[w0, _hold(7), _hold(8)],
                    carry=_carry(),
                    result=result,
                    root=root,
                )
            self.assertEqual(list(root.glob("*")), [])


class SavedPlan(unittest.TestCase):
    def _payload(self) -> dict:
        w0 = _hold(6)
        decision = dict(w0)
        decision.pop("gw")
        return {
            "rationale": "Hold.",
            "decision": decision,
            "horizon": [w0, _hold(7), _hold(8)],
        }

    def test_a_run_saves_the_plan_and_the_next_week_reads_it(self) -> None:
        payload = self._payload()

        def call_model(_system: str, _user: str) -> str:
            return json.dumps(payload)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = rn.run_once(
                gw=6,
                deadline_utc="2026-10-10T10:00:00Z",
                frozen_at_utc="2026-10-10T09:00:00Z",
                context_markdown="Press notes only.",
                directory=_market(),
                budget=1000,
                model_id="test",
                call_model=call_model,
                carry=_carry(),
                plan_root=root,
                save_plan=True,
            )
            self.assertTrue(row["accounting"]["is_legal"])
            self.assertEqual(len(row["plan_sha256"]), 64)
            self.assertTrue((root / "gw06.json").is_file())
            self.assertFalse(fz.LEDGER.exists())
            nxt = resolve_start(7, _carry(gw=7), root)
            self.assertEqual(nxt.gw, 7)
            self.assertEqual(nxt.bank, row["accounting"]["bank_after"])
            self.assertEqual(list(nxt.squad), list(_carry().squad))

    def test_a_missing_horizon_is_repaired_before_anything_is_saved(self) -> None:
        good = self._payload()
        calls = {"n": 0}

        def call_model(_system: str, _user: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                return json.dumps({"rationale": "no plan", "decision": good["decision"]})
            return json.dumps(good)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = rn.run_once(
                gw=6,
                deadline_utc="2026-10-10T10:00:00Z",
                frozen_at_utc="2026-10-10T09:00:00Z",
                context_markdown="Press notes only.",
                directory=_market(),
                budget=1000,
                model_id="test",
                call_model=call_model,
                carry=_carry(),
                plan_root=root,
                save_plan=True,
            )
            self.assertEqual(calls["n"], 2)
            self.assertTrue((root / "gw06.json").is_file())
            self.assertTrue(row["accounting"]["is_legal"])

    def test_an_unsaved_illegal_horizon_writes_no_file(self) -> None:
        decision = dict(_hold(6))
        decision.pop("gw")

        def call_model(_system: str, _user: str) -> str:
            return json.dumps({"rationale": "no plan", "decision": decision})

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = rn.run_once(
                gw=6,
                deadline_utc="2026-10-10T10:00:00Z",
                frozen_at_utc="2026-10-10T09:00:00Z",
                context_markdown="Press notes only.",
                directory=_market(),
                budget=1000,
                model_id="test",
                call_model=call_model,
                max_repairs=0,
                carry=_carry(),
                plan_root=root,
                save_plan=True,
            )
            self.assertFalse(row["accounting"]["is_legal"])
            self.assertEqual(list(root.glob("*")), [])

    def test_an_official_commit_stores_the_plan_hash(self) -> None:
        payload = self._payload()

        def call_model(_system: str, _user: str) -> str:
            return json.dumps(payload)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ledger = root / "string_agent_freeze.jsonl"
            row = rn.run_once(
                gw=6,
                deadline_utc="2026-10-10T10:00:00Z",
                frozen_at_utc="2026-10-10T09:00:00Z",
                context_markdown="Press notes only.",
                directory=_market(),
                budget=1000,
                model_id="test",
                call_model=call_model,
                carry=_carry(),
                plan_root=root,
                save_plan=True,
                commit=True,
                ledger=ledger,
            )
            self.assertEqual(row["provenance"]["plan_sha256"], row["plan_sha256"])
            stored = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(stored["provenance"]["plan_sha256"], row["plan_sha256"])
            self.assertIsNone(stored["realised"]["actual_points"])
            self.assertFalse(fz.LEDGER.exists())
