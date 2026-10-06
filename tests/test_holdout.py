"""The 2026/27 snapshots stay at the hashes in the freeze manifest."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.eval.holdout import MANIFEST_PATH, verify_freeze


class HoldoutFreezeTest(unittest.TestCase):
    def test_tracked_snapshots_match_the_manifest(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["season"], "2026-27")
        errors = verify_freeze(manifest)
        self.assertEqual(errors, [])

    def test_a_changed_hash_is_reported(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["tracked"] = dict(manifest["tracked"])
        key = next(iter(manifest["tracked"]))
        manifest["tracked"][key] = "0" * 64
        errors = verify_freeze(manifest)
        self.assertTrue(any(key in item for item in errors))

    def test_an_absent_optional_file_is_skipped(self) -> None:
        manifest = {
            "tracked": {},
            "present_at_freeze": {"data/live/not_a_real_snapshot.json": "abc"},
        }
        self.assertEqual(verify_freeze(manifest, root=Path("/workspace")), [])


if __name__ == "__main__":
    unittest.main()
