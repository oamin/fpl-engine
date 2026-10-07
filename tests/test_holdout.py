"""The 2026/27 snapshots stay at the hashes in the freeze manifest."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.eval.holdout import GROWING_CACHE, MANIFEST_PATH, sha256_file, verify_freeze


class HoldoutFreezeTest(unittest.TestCase):
    def test_tracked_snapshots_match_the_manifest(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["season"], "2026-27")
        errors = verify_freeze(manifest)
        self.assertEqual(errors, [])
        self.assertNotIn(GROWING_CACHE, manifest["tracked"])
        retired = manifest["retired_growing_cache"]
        self.assertEqual(retired["path"], GROWING_CACHE)
        self.assertEqual(
            retired["sha256"],
            "ffe911b65b5019529f8a7a322ad058e013a0e6146677c35d796f075e2572f09b",
        )
        self.assertEqual(sha256_file(Path("/workspace") / GROWING_CACHE), retired["sha256"])
        snaps = [
            key
            for key in manifest["tracked"]
            if str(key).startswith("data/holdout/2026-27/player_gw/")
        ]
        self.assertGreaterEqual(len(snaps), 5)
        self.assertEqual(
            manifest["tracked"]["data/cache/E0_2627.csv"],
            "8e21d4821be776332f3504f55f85b4277f326f03513b2c7501fbdedf92b8b1a2",
        )
        self.assertEqual(
            manifest["present_at_freeze"]["data/live/bootstrap.json"],
            "4bdb565c1c9841691252069c87a6c4e1c8f137b43a96fc15a718026d5b26a61d",
        )

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
