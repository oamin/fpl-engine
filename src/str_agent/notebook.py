"""The manager's own notes, carried from one deadline to the next.

One row per gameweek. A later deadline sees every earlier row. This week's
row, and any later row, stay out of the prompt. The dossier of articles is
not replayed. A banned score token refuses the write and leaves the file
untouched.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

_BANNED = ("score_xp", "xp_on_pot", "lam_scored")
_CANONICAL = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_PLAN_DIR = Path(__file__).resolve().parents[2] / "data" / "predictions" / "2026-27" / "string_plans"


class NotebookError(ValueError):
    """The notebook cannot be read or written."""


class NotebookContaminationError(NotebookError):
    """Notes name a score column. Nothing is written."""


def notebook_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else _PLAN_DIR
    return base / "notebook.jsonl"


def _banned(text: str) -> str | None:
    lowered = text.lower()
    for token in _BANNED:
        if token in lowered:
            return token
    return None


def assert_clean(notes: str, adjustments: str) -> None:
    """Refuse a score token. Does not touch the file."""
    token = _banned(notes) or _banned(adjustments)
    if token is not None:
        raise NotebookContaminationError(f"notebook text contains {token}")


def _row(raw: Mapping[str, Any]) -> dict[str, Any]:
    if "gw" not in raw:
        raise NotebookError("notebook row has no gameweek")
    gw = int(raw["gw"])
    written = str(raw.get("written_at_utc") or "").strip()
    if not _CANONICAL.match(written):
        raise NotebookError("written_at_utc must be YYYY-MM-DDTHH:MM:SSZ")
    notes = str(raw.get("notes") or "")
    adjustments = str(raw.get("adjustments") or "")
    assert_clean(notes, adjustments)
    return {
        "gw": gw,
        "written_at_utc": written,
        "notes": notes,
        "adjustments": adjustments,
    }


def load_notebook(root: Path | None = None) -> list[dict[str, Any]]:
    """Rows sorted by gameweek. A repeated gameweek keeps the later line."""
    path = notebook_path(root)
    if not path.is_file():
        return []
    by_gw: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise NotebookError("notebook row is not an object")
        row = _row(parsed)
        by_gw[int(row["gw"])] = row
    return [by_gw[gw] for gw in sorted(by_gw)]


def validate_note(
    *,
    gw: int,
    written_at_utc: str,
    notes: str,
    adjustments: str,
) -> dict[str, Any]:
    """Check one row. Contamination and a bad clock write nothing."""
    assert_clean(notes, adjustments)
    written = str(written_at_utc).strip()
    if not _CANONICAL.match(written):
        raise NotebookError("written_at_utc must be YYYY-MM-DDTHH:MM:SSZ")
    return {
        "gw": int(gw),
        "written_at_utc": written,
        "notes": str(notes),
        "adjustments": str(adjustments),
    }


def upsert_note(
    *,
    gw: int,
    written_at_utc: str,
    notes: str,
    adjustments: str,
    root: Path | None = None,
) -> None:
    """Replace this gameweek's row, or append it. Contamination writes nothing."""
    fresh = validate_note(
        gw=gw,
        written_at_utc=written_at_utc,
        notes=notes,
        adjustments=adjustments,
    )
    path = notebook_path(root)
    existing = load_notebook(root) if path.is_file() else []
    by_gw = {int(row["gw"]): row for row in existing}
    by_gw[int(gw)] = fresh
    lines = [
        json.dumps(by_gw[key], ensure_ascii=False, sort_keys=True)
        for key in sorted(by_gw)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def notebook_section(gw: int, root: Path | None = None) -> str:
    """Earlier notes only. An empty notebook adds nothing to the prompt."""
    prior = [row for row in load_notebook(root) if int(row["gw"]) < int(gw)]
    if not prior:
        return ""
    lines = [
        "# Manager notebook",
        "Notes you wrote after earlier deadlines. They are not this week's dossier.",
        "",
    ]
    for row in prior:
        lines.append(f"## Gameweek {row['gw']}")
        lines.append(f"Written {row['written_at_utc']}.")
        lines.append(str(row["notes"]).strip())
        lines.append("Adjustments: " + str(row["adjustments"]).strip())
        lines.append("")
    return "\n".join(lines)
