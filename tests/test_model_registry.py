"""Model iterations. The registry file is append-only, and the hashes are the commits."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.live.model_registry import (
    FILES,
    NUMAGENT_FILES,
    REGISTRY,
    ROOT,
    STRAGENT_FILES,
    RegistryError,
    _blobs_at,
    _digest_blobs,
    append_iteration,
    current_iteration,
    head_commit,
    read_registry,
    source_sha256,
)


class ModelRegistry(unittest.TestCase):
    def test_each_row_matches_the_commit_that_introduced_it(self) -> None:
        rows = read_registry()
        self.assertEqual([row["id"] for row in rows], ["numagent-gw6-lock", "numagent-anytime-full", "stragent-gw6"])
        for row in rows:
            blobs = _blobs_at(row["commit"], row["files"], ROOT)
            self.assertEqual(_digest_blobs(row["files"], blobs), row["source_sha256"])
        current = current_iteration("numagent")
        saturday = rows[0]
        self.assertEqual(current["id"], "numagent-anytime-full")
        self.assertEqual(tuple(current["files"]), NUMAGENT_FILES)
        self.assertEqual(source_sha256(current["files"], ROOT), current["source_sha256"])
        self.assertNotEqual(saturday["source_sha256"], current["source_sha256"])
        self.assertEqual(saturday["commit"], "6638dce8baee02071c3f1461f065face9bd50863")
        string = current_iteration("stragent")
        self.assertEqual(string["id"], "stragent-gw6")
        self.assertEqual(tuple(string["files"]), STRAGENT_FILES)
        self.assertEqual(source_sha256(string["files"], ROOT), string["source_sha256"])
        self.assertEqual(string["commit"], "756491ed3d458fb19e595d46ae62cf5468836a6a")
        for files in FILES.values():
            self.assertEqual(files, tuple(sorted(files)))

    def test_a_repeated_hash_or_id_leaves_the_registry_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_registry.jsonl"
            path.write_bytes(REGISTRY.read_bytes())
            before = path.read_bytes()
            commit = head_commit()
            self.assertIsNotNone(commit)
            with self.assertRaises(RegistryError):
                append_iteration(
                    iteration_id="numagent-again",
                    manager="numagent",
                    commit=str(commit),
                    summary="The same source hash.",
                    registry=path,
                )
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(RegistryError):
                append_iteration(
                    iteration_id="numagent-anytime-full",
                    manager="numagent",
                    commit=str(commit),
                    summary="A repeated id.",
                    registry=path,
                )
            self.assertEqual(path.read_bytes(), before)

    def test_an_older_commit_is_not_appended_over_the_working_tree(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model_registry.jsonl"
            with self.assertRaises(RegistryError):
                append_iteration(
                    iteration_id="numagent-replay",
                    manager="numagent",
                    commit="6638dce8baee02071c3f1461f065face9bd50863",
                    summary="The Saturday tree.",
                    registry=path,
                )
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
