"""Stack audit arms are sensitivity only; Betfair arms skip when folder absent."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.live import stack_audit as sa


class StackAudit(unittest.TestCase):
    def test_betfair_ready_needs_lines(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertFalse(sa.betfair_ready(root))
            (root / "gw_lines.csv").write_text("Date\n", encoding="utf-8")
            self.assertTrue(sa.betfair_ready(root))

    def test_render_mentions_sensitivity(self) -> None:
        text = sa.render_report(
            {
                "capture": "x.csv",
                "betfair_dir": "missing",
                "betfair_ready": False,
                "arms": [
                    {
                        "key": "arm2_betfair",
                        "label": "Betfair",
                        "skipped": True,
                        "reason": "missing",
                    }
                ],
                "deltas_vs_arm0": {},
            }
        )
        self.assertIn("sensitivity", text.lower())
        self.assertIn("Sync needed", text)

    def test_artifacts_for_arm_blocks_autodiscover(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty = root / "empty"
            bf = root / "bf"
            empty.mkdir()
            bf.mkdir()
            self.assertEqual(
                sa._artifacts_for_arm(sa.ARMS[0], bf, empty),
                empty,
            )
            self.assertEqual(
                sa._artifacts_for_arm(sa.ARMS[3], bf, empty),
                bf,
            )

    def test_delta_owned_sorted(self) -> None:
        rows = sa._delta_owned({"a": 1.0, "b": 2.0}, {"a": 1.5, "b": 2.0, "c": 3.0})
        self.assertEqual(rows[0]["player"], "c")
        self.assertAlmostEqual(rows[1]["delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
