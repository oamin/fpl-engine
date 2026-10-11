"""Stragent live session. No network, no model, no official ledger."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.str_agent.freeze import DuplicateFreezeError
from src.str_agent.horizon import MissingPriorPlanError
from src.str_agent.notebook import notebook_section
from src.str_agent.session import SessionError, apply_reply, run_stragent
from src.str_agent.trial import TrialPack
from tests.test_str_agent_carry import _carry, _market
from tests.test_str_agent_horizon import _hold


DEADLINE = "2026-10-17T10:00:00Z"
FROZEN = "2026-10-17T09:00:00Z"


def _pack(context: str = "The squad you own.") -> TrialPack:
    return TrialPack(
        entry_id=1,
        team_name="toy",
        deadline_utc=DEADLINE,
        carry=_carry(gw=7),
        directory=_market(),
        names={pid: pid for pid in _market()},
        context=context,
    )


def _reply(notes: str = "") -> dict:
    week = _hold(7)
    decision = dict(week)
    decision.pop("gw")
    return {
        "model_id": "test",
        "rationale": "Hold the fifteen.",
        "notes": notes,
        "adjustments": "",
        "decision": decision,
        "horizon": [_hold(7), _hold(8), _hold(9)],
    }


class StringSession(unittest.TestCase):
    def test_a_missing_prior_plan_stops_before_the_entry(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_root = root / "plans"
            plan_root.mkdir()
            called = {"trawl": 0, "prepare": 0}

            def trawl_fn(**_kwargs: object) -> object:
                called["trawl"] += 1
                raise AssertionError("feeds polled")

            def prepare_fn(**_kwargs: object) -> TrialPack:
                called["prepare"] += 1
                raise AssertionError("entry read")

            with self.assertRaises(MissingPriorPlanError):
                run_stragent(
                    7,
                    deadline_utc=DEADLINE,
                    root=root,
                    plan_root=plan_root,
                    trawl_fn=trawl_fn,
                    prepare_fn=prepare_fn,
                )
            self.assertEqual(called["trawl"], 0)
            self.assertEqual(called["prepare"], 0)
            self.assertEqual(list(plan_root.iterdir()), [])

    def test_context_waits_and_keeps_earlier_notes_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_root = root / "plans"
            plan_root.mkdir()
            (plan_root / "notebook.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "gw": 5,
                                "written_at_utc": "2026-10-03T12:00:00Z",
                                "notes": "Earlier note.",
                                "adjustments": "none",
                            }
                        ),
                        json.dumps(
                            {
                                "gw": 7,
                                "written_at_utc": "2026-10-17T08:00:00Z",
                                "notes": "This week stays out.",
                                "adjustments": "none",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (plan_root / "gw06.json").write_text('{"carry_after": {}}\n', encoding="utf-8")

            def prepare_fn(**_kwargs: object) -> TrialPack:
                return _pack(notebook_section(7, plan_root))

            def trawl_fn(_gw: int, **_kwargs: object) -> object:
                class _Result:
                    written = 1

                return _Result()

            payload = run_stragent(
                7,
                deadline_utc=DEADLINE,
                root=root,
                plan_root=plan_root,
                trawl_fn=trawl_fn,
                prepare_fn=prepare_fn,
                frozen_at_utc=FROZEN,
            )
            context = (plan_root / "gw07_context.md").read_text(encoding="utf-8")
            self.assertEqual(payload["exit_code"], 2)
            self.assertFalse(payload["locked"])
            self.assertIsNone(payload["realised"])
            self.assertIn("Gameweek 5", context)
            self.assertNotIn("This week stays out", context)
            self.assertFalse((plan_root / "gw07.json").exists())
            manifest = json.loads(
                (root / "data" / "predictions" / "2026-27" / "gw07" / "session_stragent.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(manifest["locked"])
            self.assertIsNone(manifest["realised"])

    def test_a_banned_token_writes_no_plan_or_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            plan_root = Path(folder)
            with self.assertRaises(SessionError):
                apply_reply(
                    _pack(),
                    _reply(notes="The score_xp column says start him."),
                    frozen_at_utc=FROZEN,
                    plan_root=plan_root,
                )
            self.assertFalse((plan_root / "gw07.json").exists())
            self.assertFalse((plan_root / "notebook.jsonl").exists())

    def test_an_illegal_horizon_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            plan_root = Path(folder)
            payload = _reply()
            payload["horizon"] = payload["horizon"][:1]
            with self.assertRaises(SessionError):
                apply_reply(
                    _pack(),
                    payload,
                    frozen_at_utc=FROZEN,
                    plan_root=plan_root,
                )
            self.assertEqual(list(plan_root.iterdir()), [])

    def test_a_late_lock_writes_no_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ledger = root / "string_agent_freeze.jsonl"
            with self.assertRaises(SessionError):
                apply_reply(
                    _pack(),
                    _reply(),
                    frozen_at_utc=DEADLINE,
                    plan_root=root / "plans",
                    lock=True,
                    ledger=ledger,
                )
            self.assertFalse(ledger.exists())
            self.assertFalse((root / "plans").exists())

    def test_lock_commits_week_zero_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_root = root / "plans"
            ledger = root / "string_agent_freeze.jsonl"
            result = apply_reply(
                _pack(),
                _reply(notes="Kept the same eleven."),
                frozen_at_utc=FROZEN,
                plan_root=plan_root,
                lock=True,
                ledger=ledger,
            )
            lines = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["gw"], 7)
            self.assertNotIn("horizon", row)
            self.assertEqual(row["decision"]["captain"], _hold(7)["captain"])
            self.assertTrue(row["accounting"]["is_legal"])
            plan = json.loads((plan_root / "gw07.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["horizon"]), 3)
            self.assertEqual(row["provenance"]["plan_sha256"], result["plan_sha256"])
            before = ledger.read_bytes()
            with self.assertRaises(DuplicateFreezeError):
                apply_reply(
                    _pack(),
                    _reply(notes="Kept the same eleven."),
                    frozen_at_utc=FROZEN,
                    plan_root=plan_root,
                    lock=True,
                    ledger=ledger,
                )
            self.assertEqual(ledger.read_bytes(), before)


class MorningStringSession(unittest.TestCase):
    def test_the_saturday_manifest_matches_the_ledger_and_stops_another_run(self) -> None:
        from src.str_agent.freeze import LEDGER as STRING_LEDGER
        from src.str_agent.horizon import PLAN_DIR

        manifest = json.loads(
            Path("data/predictions/2026-27/gw06/session_stragent.json").read_text(encoding="utf-8")
        )
        row = json.loads(STRING_LEDGER.read_text(encoding="utf-8").splitlines()[0])
        self.assertTrue(manifest["locked"])
        self.assertIsNone(manifest["realised"])
        self.assertEqual(manifest["hashes"]["plan_sha256"], row["provenance"]["plan_sha256"])
        self.assertEqual(manifest["hashes"]["context_sha256"], row["provenance"]["context_sha256"])
        self.assertEqual(manifest["steps"][-1]["status"], "locked")

        def trawl_fn(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("feeds polled")

        with self.assertRaises(SessionError):
            run_stragent(
                6,
                deadline_utc="2026-10-10T10:00:00Z",
                plan_root=PLAN_DIR,
                trawl_fn=trawl_fn,
            )


if __name__ == "__main__":
    unittest.main()
