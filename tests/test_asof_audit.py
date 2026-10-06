"""The as-of audit fails when the pool or a prior reads the week being scored."""

from __future__ import annotations

import unittest

from src.eval.gates import run_asof_audit


class AsofAuditTest(unittest.TestCase):
    def test_audit_passes_on_the_repaired_path(self) -> None:
        audit = run_asof_audit()
        self.assertTrue(audit["passed"], audit["failures"])


if __name__ == "__main__":
    unittest.main()
