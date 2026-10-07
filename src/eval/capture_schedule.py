"""Pre-deadline FPL bootstrap captures.

The clock is the HTTP Date header on the bootstrap response. The runner's
clock is not a capture time. A snapshot is written in the T−24h window or
the T−1h window, once per gameweek and slot. The raw JSON is kept.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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


def _pair_engine(
    payload: dict[str, Any],
    captured: datetime,
    *,
    gw: int,
    official_path: Path,
    raw_path: Path,
    root: Path,
) -> Path | None:
    """Write ``score_xp`` on this bootstrap and the shadow beside ``ep_next``.

    A toy payload, or a checkout without the live files, leaves the official
    file in place and writes no engine score. Odds are not fetched.
    """
    elements = payload.get("elements") or []
    if "teams" not in payload or len(elements) < 15:
        return None
    from src.eval.holdout import sha256_file
    from src.eval.predictions import MINUTES_PATH, export_deadline_scores
    from src.live.deadline import ENTRY_PATH, FIXTURES_PATH, LOG_PATH, ODDS_PATH
    from src.live.scorer import write_shadow_log

    needed = (ENTRY_PATH, LOG_PATH, ODDS_PATH, FIXTURES_PATH, MINUTES_PATH)
    if any(not path.is_file() for path in needed):
        return None
    stamp = stamp_of(captured)
    engine_path = export_deadline_scores(
        gw=gw,
        dest_root=root,
        stamp=stamp,
        bootstrap=payload,
        created_at=captured_text(captured),
        bootstrap_hash=sha256_file(raw_path),
    )
    shadow = engine_path.parent / f"shadow_{stamp}.csv"
    write_shadow_log(shadow, engine_path, official_path)
    return shadow


def write_snapshot(
    payload: dict[str, Any],
    captured: datetime,
    *,
    gw: int,
    slot: str | None,
    root: Path | None = None,
    pair_engine: bool = False,
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
    if pair_engine:
        _pair_engine(payload, captured, gw=gw, official_path=csv_path, raw_path=raw_path, root=base)
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


def deadlines_from_events(events: list[dict[str, Any]]) -> dict[str, str]:
    return {str(int(event["id"])): str(event["deadline_time"]) for event in events}


def has_capture(root: Path, gw: int) -> bool:
    """True when this gameweek already has an official snapshot."""
    directory = root / "data" / "predictions" / SEASON / f"gw{int(gw):02d}"
    if not directory.is_dir():
        return False
    return any(directory.glob("official_*.csv")) or any(directory.glob("slot_*.json"))


def missing_capture_gws(
    root: Path,
    now: datetime,
    deadlines: dict[str, str],
    *,
    live_from: int = 6,
) -> list[int]:
    """Gameweeks whose capture window has closed and which have no snapshot.

    ``now`` is the clock of the check. A capture file's own timestamp still
    comes from the response Date header.
    """
    close_minutes = int(T1_MINUTES[0])
    missing: list[int] = []
    for gw, text in deadlines.items():
        number = int(gw)
        if number < int(live_from):
            continue
        deadline = aware_utc(text)
        if now < deadline - timedelta(minutes=close_minutes):
            continue
        if not has_capture(root, number):
            missing.append(number)
    return sorted(missing)


def run(opener: Callable[..., Any] | None = None, root: Path | None = None) -> int:
    """Write one snapshot when the next deadline sits in a capture window.

    After the attempt, a closed gameweek with no snapshot raises.
    """
    payload, captured = fetch_bootstrap(opener)
    event = next_open_event(payload["events"], captured)
    base = root or ROOT
    if event is not None:
        gw = int(event["id"])
        slot = choose_slot(captured, aware_utc(event["deadline_time"]))
        if slot is not None and not slot_marker(base, gw, slot).exists():
            write_snapshot(payload, captured, gw=gw, slot=slot, root=root, pair_engine=True)
    missing = missing_capture_gws(base, captured, deadlines_from_events(payload["events"]))
    if missing:
        names = ", ".join(str(gw) for gw in missing)
        raise RuntimeError(f"no pre-deadline capture for gameweeks {names}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
