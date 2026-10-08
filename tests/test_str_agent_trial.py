"""The Gameweek 6 dry run starts from the Gameweek 5 fifteen."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.live.entry import load_entry
from src.str_agent.horizon import MissingPriorPlanError
from src.str_agent.trial import judge, prepare


class Trial(unittest.TestCase):
    def test_starting_squad_is_the_last_played_fifteen(self) -> None:
        pack = prepare(2632584)
        entry = load_entry(2632584)
        week = next(row for row in entry["gameweeks"] if int(row["gw"]) == 5)
        ids = [str(player["id"]) for player in week["xi"] + week["bench"]]
        self.assertEqual(list(pack.carry.squad), ids)
        self.assertEqual(pack.carry.gw, 6)
        self.assertEqual(pack.carry.ft_before, 1)
        self.assertEqual(pack.carry.bank, 15)
        self.assertEqual(len(pack.carry.purchase_prices), 15)
        self.assertIn("Starting squad: the Gameweek 5 fifteen", pack.context)
        self.assertNotIn("score_xp", pack.context)

    def test_saved_gw6_choice_is_a_legal_move(self) -> None:
        pack = prepare(2632584)
        payload = json.loads(
            Path("reports/string_agent_gw6_decision.json").read_text(encoding="utf-8")
        )
        result = judge(pack, payload["decision"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["is_legal"])
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["bank_after"], 2)
        self.assertEqual(result["named"]["transfers_out"], ["Scarlett"])
        self.assertEqual(result["named"]["transfers_in"], ["Wood"])
        self.assertEqual(result["named"]["captain"], "Ødegaard")

    def test_a_later_week_starts_from_the_saved_plan(self) -> None:
        entry = load_entry(2632584)
        week = next(row for row in entry["gameweeks"] if int(row["gw"]) == 5)
        ids = [str(player["id"]) for player in week["xi"] + week["bench"]]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = {
                "frozen_at_utc": "2026-10-10T09:00:00Z",
                "deadline_utc": "2026-10-10T10:00:00Z",
                "rationale": "Hold the paper squad.",
                "horizon": [],
                "carry_after": {
                    "gw": 7,
                    "squad": ids,
                    "purchase_prices": {pid: 40 for pid in ids},
                    "bank": 77,
                    "ft_before": 3,
                    "chips_played": {"1": "triple_captain"},
                    "selling_prices": {},
                },
            }
            (root / "gw06.json").write_text(json.dumps(plan), encoding="utf-8")
            empty = Path(folder) / "empty"
            empty.mkdir()
            with self.assertRaises(MissingPriorPlanError):
                prepare(2632584, plan_root=empty, next_gw=7)
            pack = prepare(2632584, plan_root=root, next_gw=7)
        self.assertEqual(pack.carry.gw, 7)
        self.assertEqual(pack.carry.bank, 77)
        self.assertEqual(pack.carry.ft_before, 3)
        self.assertEqual(list(pack.carry.squad), ids)
        self.assertIn("It is not the live entry.", pack.context)
        self.assertNotIn("the Gameweek 5 fifteen", pack.context)
        self.assertNotIn("score_xp", pack.context)

    def test_commit_saved_writes_the_chosen_ledger_only(self) -> None:
        from src.str_agent.__main__ import main
        from src.str_agent.carry import hold_draft_from_entry
        from src.str_agent.freeze import LEDGER

        entry = load_entry(2632584)
        decision = hold_draft_from_entry(entry)["decision"]
        horizon = []
        for gw in (6, 7, 8):
            week = dict(decision)
            week["gw"] = gw
            horizon.append(week)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            draft = root / "draft.json"
            draft.write_text(
                json.dumps(
                    {
                        "model_id": "test",
                        "rationale": "Hold the fifteen.",
                        "decision": decision,
                        "horizon": horizon,
                    }
                ),
                encoding="utf-8",
            )
            ledger = root / "freeze.jsonl"
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "--entry",
                        "2632584",
                        "--save-plan",
                        str(draft),
                        "--plan-root",
                        str(root),
                        "--commit-saved",
                        "--ledger",
                        str(ledger),
                    ]
                )
            self.assertEqual(code, 0, out.getvalue())
            saved = json.loads(out.getvalue())
            self.assertEqual(len(saved["plan_sha256"]), 64)
            self.assertTrue((root / "gw06.json").is_file())
            self.assertTrue(ledger.is_file())
            row = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(row["provenance"]["plan_sha256"], saved["plan_sha256"])
            self.assertIsNone(row["realised"]["actual_points"])
            self.assertFalse(LEDGER.exists())
            err = io.StringIO()
            with redirect_stderr(err):
                again = main(
                    [
                        "--entry",
                        "2632584",
                        "--save-plan",
                        str(draft),
                        "--plan-root",
                        str(root / "second"),
                        "--commit-saved",
                        "--ledger",
                        str(ledger),
                    ]
                )
            self.assertEqual(again, 1)
            self.assertIn("already frozen", err.getvalue())


if __name__ == "__main__":
    unittest.main()
