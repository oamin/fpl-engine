"""String-agent scaffold: legality and freeze (no live LLM)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.str_agent import freeze as fz
from src.str_agent import runner as rn
from src.str_agent import validator as val


def _directory() -> dict[str, dict]:
    # 15 legal ids with costs summing under 1000.
    rows = []
    # 2 GKP
    rows += [("g1", "GKP", "A", 40), ("g2", "GKP", "B", 40)]
    # 5 DEF
    rows += [(f"d{i}", "DEF", "C" if i < 3 else "D", 45) for i in range(5)]
    # 5 MID
    rows += [(f"m{i}", "MID", "E" if i < 3 else "F", 50) for i in range(5)]
    # 3 FWD
    rows += [(f"f{i}", "FWD", "G", 55) for i in range(3)]
    return {
        pid: {"position": pos, "club": club, "now_cost": cost}
        for pid, pos, club, cost in rows
    }


def _draft() -> dict:
    d = _directory()
    squad = list(d)
    # Legal 4-4-2: 1 GKP + 4 DEF + 4 MID + 2 FWD (not squad[:11]).
    xi = ["g1", "d0", "d1", "d2", "d3", "m0", "m1", "m2", "m3", "f0", "f1"]
    bench = [pid for pid in squad if pid not in set(xi)]
    return {
        "gw": 6,
        "deadline_utc": "2026-10-10T10:00:00Z",
        "frozen_at_utc": "2026-10-10T09:00:00Z",
        "provenance": {
            "model_id": "test",
            "prompt_sha256": "a" * 64,
            "context_sha256": "b" * 64,
        },
        "rationale": "unit test draft",
        "decision": {
            "chip_played": None,
            "squad_15": squad,
            "starting_11": xi,
            "captain": xi[0],
            "vice_captain": xi[1],
            "bench_order": bench,
            "transfers_in": [],
            "transfers_out": [],
        },
        "accounting": {"bank_remaining": 10, "total_cost": 700, "is_legal": True},
        "realised": fz.empty_realised(),
    }


class StrAgent(unittest.TestCase):
    def test_legal_draft_passes(self) -> None:
        draft = _draft()
        errors = val.validate_draft(
            draft, directory=_directory(), budget=1000
        )
        self.assertEqual(errors, [])

    def test_budget_breach_fails(self) -> None:
        draft = _draft()
        errors = val.validate_draft(draft, directory=_directory(), budget=100)
        self.assertTrue(any("budget" in e or "squad fails" in e for e in errors))

    def test_freeze_requires_pre_deadline(self) -> None:
        row = _draft()
        row["frozen_at_utc"] = "2026-10-10T11:00:00Z"
        with self.assertRaises(fz.StringFreezeError):
            fz.validate_freeze_row(row)

    def test_freeze_append_once(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "string_agent_freeze.jsonl"
            row = _draft()
            fz.write_string_freeze(row, path)
            with self.assertRaises(fz.StringFreezeError):
                fz.write_string_freeze(row, path)

    def test_runner_repairs_then_commits(self) -> None:
        good = _draft()
        bad_decision = dict(good["decision"])
        bad_decision["starting_11"] = list(good["decision"]["squad_15"])[:11]
        calls = {"n": 0}

        def call_model(_system: str, user: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                payload = {
                    "rationale": "bad xi",
                    "decision": bad_decision,
                }
            else:
                payload = {
                    "rationale": good["rationale"],
                    "decision": good["decision"],
                }
            return "```json\n" + __import__("json").dumps(payload) + "\n```"

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "string_agent_freeze.jsonl"
            row = rn.run_once(
                gw=6,
                deadline_utc=good["deadline_utc"],
                frozen_at_utc=good["frozen_at_utc"],
                context_markdown="# ctx",
                directory=_directory(),
                budget=1000,
                model_id="test",
                call_model=call_model,
                max_repairs=2,
                ledger=path,
                commit=True,
            )
            self.assertTrue(row["accounting"]["is_legal"])
            self.assertEqual(calls["n"], 2)
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
