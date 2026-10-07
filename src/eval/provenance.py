"""Official xP is allowed only with a pre-deadline capture.

Vaastav ``xP`` is ``ep_this`` scraped after the gameweek. A historical sheet
has no capture time, so it is not a benchmark and not a feature. A file
mtime is not a capture time. The deadline is the canonical calendar, not
the value written on the row.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEADLINES_PATH = ROOT / "data" / "predictions" / "2026-27" / "deadlines.json"

OFFICIAL_XP_COLUMNS = ("official_xp", "score_official_xp", "xP", "ep_this", "ep_next")


def load_deadlines(path: Path | None = None) -> dict[str, str]:
    raw = json.loads((path or DEADLINES_PATH).read_text(encoding="utf-8"))
    return {str(gw): str(deadline) for gw, deadline in raw["deadlines"].items()}


def aware_utc(value: object) -> datetime:
    """Parse one timestamp. A naive clock is refused."""
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if text.lower() in {"", "nan", "nat", "none", "na"}:
        raise ValueError("missing timestamp")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("naive timestamp")
    return parsed.astimezone(timezone.utc)


def assert_predeadline_xp(frame: pd.DataFrame, deadlines: Mapping[str, str]) -> None:
    """Raise unless every row's official xP was captured strictly before its deadline.

    One bad row fails the whole frame. Rows are not dropped.
    """
    present = [name for name in OFFICIAL_XP_COLUMNS if name in frame.columns]
    if not present:
        raise RuntimeError("refusing xP: no official xP column")
    if frame.empty:
        raise RuntimeError("refusing xP: the frame has no rows")
    for column in ("captured_at", "deadline", "gw", "source_field", "event_role"):
        if column not in frame.columns:
            raise RuntimeError(f"refusing xP: rows have no {column}")
    for row in frame.itertuples(index=False):
        gw = str(int(row.gw))
        canonical = deadlines.get(gw)
        if canonical is None:
            raise RuntimeError(f"refusing xP: no canonical deadline for gameweek {gw}")
        try:
            captured = aware_utc(row.captured_at)
            stated = aware_utc(row.deadline)
            calendar = aware_utc(canonical)
        except ValueError as exc:
            raise RuntimeError(f"refusing xP: {exc}") from exc
        if stated != calendar:
            raise RuntimeError("refusing xP: deadline does not match the canonical calendar")
        if captured >= calendar:
            raise RuntimeError("refusing xP: capture is not strictly before the deadline")
        source = str(row.source_field)
        role = str(row.event_role)
        if source == "ep_next" and role == "next":
            continue
        if source == "ep_this" and role == "current":
            continue
        raise RuntimeError(
            "refusing xP: source_field and event_role do not name the captured gameweek"
        )


def commit_is_before(commit_at: object, deadline: object) -> bool:
    return aware_utc(commit_at) < aware_utc(deadline)
