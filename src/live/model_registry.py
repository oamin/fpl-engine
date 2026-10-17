"""Append-only iterations of the two live managers.

The current iteration is the last row for that manager in
``data/predictions/2026-27/model_registry.jsonl``. A lock stores that row's
id, the commit that introduced it, the git SHA at lock time, and the
sha256 of a fixed file list. When the working tree of those files differs
from the stored hash, the lock stops and the ledger is not written.

Session modules, manifests, and docs are outside the hash. An older row is
left as it was written. A formula or prompt change appends a new row.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data" / "predictions" / "2026-27" / "model_registry.jsonl"

# Lexicographic. The hash walks this order: path, NUL, bytes, NUL.
NUMAGENT_FILES = (
    "src/eval/decision_spec.py",
    "src/live/betfair_props.py",
    "src/live/half_plan.py",
    "src/live/plan.py",
    "src/live/policy.py",
    "src/live/scorer.py",
    "src/models/forecast_xp.py",
    "src/models/half_plan_scores.py",
    "src/models/season_climb.py",
    "src/models/season_climb_budget.py",
    "src/models/season_climb_ft.py",
    "src/models/xp_engine.py",
    "src/rules/fpl_2026.py",
)
STRAGENT_FILES = (
    "src/rules/fpl_2026.py",
    "src/str_agent/carry.py",
    "src/str_agent/horizon.py",
    "src/str_agent/notebook.py",
    "src/str_agent/prompt.py",
    "src/str_agent/trial.py",
    "src/str_agent/validator.py",
)
FILES = {"numagent": NUMAGENT_FILES, "stragent": STRAGENT_FILES}


class RegistryError(RuntimeError):
    """The iteration registry cannot be read or extended."""


class ModelDriftError(RegistryError):
    """The working tree is not the registered iteration."""


def _sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _commit_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def source_sha256(files: Sequence[str], root: Path) -> str:
    """Hash ``files`` in the given order from ``root``."""
    digest = hashlib.sha256()
    base = Path(root)
    for rel in files:
        path = base / rel
        if not path.is_file():
            raise ModelDriftError(f"model file is missing: {rel}")
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _digest_blobs(files: Sequence[str], blobs: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for rel in files:
        digest.update(str(rel).encode("utf-8"))
        digest.update(b"\0")
        digest.update(blobs[rel])
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        raise RegistryError("iteration id must be a non-empty string")
    if row.get("manager") not in FILES:
        raise RegistryError("iteration manager must be numagent or stragent")
    if not _commit_hex(row.get("commit")):
        raise RegistryError("iteration commit must be a 40-hex git sha")
    if not isinstance(row.get("summary"), str) or not row["summary"].strip():
        raise RegistryError("iteration summary must be a non-empty string")
    if not _sha256_hex(row.get("source_sha256")):
        raise RegistryError("iteration source_sha256 must be a 64-hex sha256")
    files = row.get("files")
    if not isinstance(files, list) or not files:
        raise RegistryError("iteration files must be a non-empty list")
    for rel in files:
        if not isinstance(rel, str) or rel.startswith("/") or ".." in Path(rel).parts:
            raise RegistryError(f"iteration file path is not relative: {rel}")


def read_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Load every iteration. A missing file is an empty registry."""
    ledger = Path(path) if path is not None else REGISTRY
    if not ledger.is_file():
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{ledger}:{line_no} is not JSON") from exc
        if not isinstance(row, dict):
            raise RegistryError(f"{ledger}:{line_no} is not an object")
        _validate_row(row)
        if row["id"] in seen:
            raise RegistryError(f"duplicate iteration id {row['id']}")
        seen.add(str(row["id"]))
        rows.append(row)
    return rows


def current_iteration(manager: str, path: Path | None = None) -> dict[str, Any]:
    """The last registered row for ``manager``."""
    if manager not in FILES:
        raise RegistryError("iteration manager must be numagent or stragent")
    rows = [row for row in read_registry(path) if row["manager"] == manager]
    if not rows:
        raise RegistryError(f"no {manager} iteration is registered")
    return rows[-1]


def head_commit(root: Path | None = None) -> str | None:
    """``git rev-parse HEAD``, or null when git cannot answer."""
    base = Path(root) if root is not None else ROOT
    try:
        done = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = done.stdout.strip()
    if not _commit_hex(text):
        return None
    return text


def _blobs_at(commit: str, files: Sequence[str], root: Path) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    for rel in files:
        try:
            blobs[rel] = subprocess.check_output(
                ["git", "-C", str(root), "show", f"{commit}:{rel}"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RegistryError(f"cannot read {rel} at {commit}") from exc
    return blobs


def record_for_session(
    manager: str,
    *,
    root: Path | None = None,
    registry: Path | None = None,
) -> dict[str, Any]:
    """Describe the working tree against the current iteration.

    A mismatch leaves ``iteration_id`` null and does not raise. The lock
    path uses :func:`record_for_lock`.
    """
    row = current_iteration(manager, registry)
    base = Path(root) if root is not None else ROOT
    observed = source_sha256(list(row["files"]), base)
    match = observed == row["source_sha256"]
    payload: dict[str, Any] = {
        "manager": manager,
        "iteration_id": row["id"] if match else None,
        "iteration_commit": row["commit"] if match else None,
        "source_sha256": observed,
        "matches_registered": match,
    }
    if not match:
        payload["registered_iteration_id"] = row["id"]
        payload["registered_source_sha256"] = row["source_sha256"]
    return payload


def record_for_lock(
    manager: str,
    *,
    root: Path | None = None,
    registry: Path | None = None,
) -> dict[str, Any]:
    """The iteration a lock may store. A hash mismatch raises."""
    record = record_for_session(manager, root=root, registry=registry)
    if record["matches_registered"] is not True:
        registered = record.get("registered_iteration_id")
        raise ModelDriftError(
            f"{manager} source hash does not match {registered}. "
            "Register the new iteration before locking."
        )
    base = Path(root) if root is not None else ROOT
    record["locked_commit"] = head_commit(base)
    return record


def append_iteration(
    *,
    iteration_id: str,
    manager: str,
    commit: str,
    summary: str,
    root: Path | None = None,
    registry: Path | None = None,
    files: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Append one row whose source hash is the working tree at ``commit``.

    The tree and ``commit`` must contain the same bytes. A repeated id, or
    a hash that is already the current row, leaves the file unchanged.
    """
    if not isinstance(iteration_id, str) or not iteration_id.strip():
        raise RegistryError("iteration id must be a non-empty string")
    if manager not in FILES:
        raise RegistryError("iteration manager must be numagent or stragent")
    if not _commit_hex(commit):
        raise RegistryError("iteration commit must be a 40-hex git sha")
    if not isinstance(summary, str) or not summary.strip():
        raise RegistryError("iteration summary must be a non-empty string")
    path = Path(registry) if registry is not None else REGISTRY
    before = path.read_bytes() if path.is_file() else b""
    rows = read_registry(path)
    if any(row["id"] == iteration_id for row in rows):
        raise RegistryError(f"iteration {iteration_id} is already registered")
    chosen = list(files) if files is not None else list(FILES[manager])
    base = Path(root) if root is not None else ROOT
    tree_hash = source_sha256(chosen, base)
    commit_hash = _digest_blobs(chosen, _blobs_at(commit, chosen, base))
    if tree_hash != commit_hash:
        raise RegistryError(
            f"{iteration_id} working tree does not match {commit} for the model files"
        )
    previous = [row for row in rows if row["manager"] == manager]
    if previous and previous[-1]["source_sha256"] == tree_hash:
        raise RegistryError(f"{manager} source hash is already {previous[-1]['id']}")
    row = {
        "id": iteration_id,
        "manager": manager,
        "commit": commit,
        "summary": summary.strip(),
        "source_sha256": tree_hash,
        "files": chosen,
    }
    _validate_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    if before and not path.read_bytes().startswith(before):
        path.write_bytes(before)
        raise RegistryError("registry prefix changed")
    return row


def main(argv: list[str] | None = None) -> int:
    """Print the current iteration of each manager and whether the tree matches."""
    import sys

    del argv
    for manager in ("numagent", "stragent"):
        row = current_iteration(manager)
        observed = source_sha256(list(row["files"]), ROOT)
        match = observed == row["source_sha256"]
        sys.stdout.write(
            f"{manager} {row['id']} {row['commit']} match={str(match).lower()}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
