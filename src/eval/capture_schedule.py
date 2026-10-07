"""Pre-deadline FPL bootstrap captures.

The clock is the HTTP Date header on the bootstrap response. The runner's
clock is not a capture time. A snapshot is written in the T−24h window or
the T−1h window, once per gameweek and slot. The raw JSON is kept.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

from src.eval.decision_spec import T1_MINUTES, T24_HOURS
from src.eval.provenance import aware_utc
from src.live.deadline import SEASON

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
SLOT_T24 = "t24"
SLOT_T1 = "t1"


def captured_at_from_date_header(date_header: object) -> datetime:
    """Parse the response Date header. A missing header is refused."""
    if date_header is None or not str(date_header).strip():
        raise RuntimeError("refusing capture: the response has no Date header")
    parsed = parsedate_to_datetime(str(date_header))
    if parsed.tzinfo is None:
        raise RuntimeError("refusing capture: the Date header has no timezone")
    return parsed.astimezone(timezone.utc)


def choose_slot(captured: datetime, deadline: datetime) -> str | None:
    """Return t24, t1, or nothing when the gap is outside both windows."""
    delta = (deadline - captured).total_seconds()
    lo_h, hi_h = T24_HOURS
    if lo_h * 3600 <= delta <= hi_h * 3600:
        return SLOT_T24
    lo_m, hi_m = T1_MINUTES
    if lo_m * 60 <= delta <= hi_m * 60:
        return SLOT_T1
    return None


def next_open_event(events: list[dict[str, Any]], captured: datetime) -> dict[str, Any] | None:
    """Earliest unfinished deadline still strictly after the capture."""
    pending: list[tuple[datetime, dict[str, Any]]] = []
    for event in events:
        if event.get("finished"):
            continue
        deadline = aware_utc(event["deadline_time"])
        if deadline <= captured:
            continue
        pending.append((deadline, event))
    if not pending:
        return None
    pending.sort(key=lambda item: item[0])
    return pending[0][1]


def stamp_of(captured: datetime) -> str:
    return captured.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def captured_text(captured: datetime) -> str:
    return captured.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def slot_marker(root: Path, gw: int, slot: str) -> Path:
    return root / "data" / "predictions" / SEASON / f"gw{int(gw):02d}" / f"slot_{slot}.json"


def write_snapshot(
    payload: dict[str, Any],
    captured: datetime,
    *,
    gw: int,
    slot: str | None,
    root: Path | None = None,
) -> Path:
    """Write the raw bootstrap and the extracted official columns. Do not overwrite."""
    from src.eval.predictions import official_frame_from_elements, write_deadlines, write_prediction

    base = root or ROOT
    directory = base / "data" / "predictions" / SEASON / f"gw{int(gw):02d}"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = stamp_of(captured)
    suffix = f"_{slot}" if slot else ""
    raw_path = directory / f"bootstrap_{stamp}{suffix}.json"
    if raw_path.exists():
        raise FileExistsError(f"refusing to overwrite {raw_path}")
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    frame = official_frame_from_elements(
        payload["elements"],
        payload["events"],
        gw=gw,
        captured_at=captured_text(captured),
    )
    csv_path = directory / f"official_{stamp}{suffix}.csv"
    write_prediction(csv_path, frame)
    deadlines_path = directory / "deadlines.json" if root is not None else None
    write_deadlines(payload["events"], deadlines_path)
    if slot:
        marker = slot_marker(base, gw, slot)
        marker.write_text(
            json.dumps(
                {
                    "gw": int(gw),
                    "slot": slot,
                    "captured_at": captured_text(captured),
                    "stamp": stamp,
                    "raw": raw_path.name,
                    "official": csv_path.name,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return csv_path


def fetch_bootstrap(opener: Callable[..., Any] | None = None) -> tuple[dict[str, Any], datetime]:
    """GET bootstrap-static. The capture time is the Date header."""
    import urllib.request

    open_url = opener or urllib.request.urlopen
    with open_url(BOOTSTRAP_URL, timeout=60) as resp:
        body = resp.read()
        header = resp.headers.get("Date")
    captured = captured_at_from_date_header(header)
    payload = json.loads(body)
    return payload, captured


def run(opener: Callable[..., Any] | None = None, root: Path | None = None) -> int:
    """Write one snapshot when the next deadline sits in a capture window."""
    payload, captured = fetch_bootstrap(opener)
    event = next_open_event(payload["events"], captured)
    if event is None:
        return 0
    gw = int(event["id"])
    slot = choose_slot(captured, aware_utc(event["deadline_time"]))
    if slot is None:
        return 0
    base = root or ROOT
    if slot_marker(base, gw, slot).exists():
        return 0
    write_snapshot(payload, captured, gw=gw, slot=slot, root=root)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
