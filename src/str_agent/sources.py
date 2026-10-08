"""Forums, YouTube, and other outlets for the string agent.

These notes are not press packets. ``src/live/news_packets.py`` does not
read this folder, and ``compile_player_xmi`` does not see them. A note
still has to predate the deadline, carry a real URL, and hash its text.

The publish time is the outlet's own clock, re-parsed from ``raw_published_at``.
A bare date or a naive datetime is refused. ``observed_at_utc`` is when the
note was captured. The hash covers both clocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

OUTLET_CLASSES = frozenset({"press", "forum", "youtube", "other"})
BODY_CAP = 2000
CLOCKS = frozenset(
    {
        "youtube_published_at",
        "reddit_created_utc",
        "html_time",
        "json_ld_datePublished",
        "http_last_modified",
        "feed_updated",
    }
)
_CANONICAL = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_PLACEHOLDER_HOSTS = frozenset(
    {
        "example",
        "example.com",
        "localhost",
        "test.com",
    }
)


class StringSourceError(ValueError):
    """A forum, video, or other note failed the string-agent gate."""


@dataclass(frozen=True)
class StringSource:
    """One pre-deadline note from an open outlet."""

    source_id: str
    source: str
    outlet_class: str
    url: str
    published_at_utc: str
    observed_at_utc: str
    clock: str
    raw_published_at: str
    headline: str
    body: str
    sha256: str
    player_ids: tuple[int, ...] = ()
    recorded_at_utc: str | None = None

    def quote(self) -> str:
        text = self.headline.strip()
        body = self.body.strip()
        if text and body:
            return f"{text}. {body}"
        return text or body


def sources_dir(gw: int, root: Path | None = None) -> Path:
    from src.live.news_packets import ROOT

    base = root or ROOT
    return base / "data" / "predictions" / "2026-27" / f"gw{int(gw):02d}" / "string_sources"


def _check_url(url: str) -> None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise StringSourceError("url must be http(s)")
    host = (parsed.hostname or "").lower().rstrip(".")
    blocked = host in _PLACEHOLDER_HOSTS or host.endswith(
        (".example.com", ".test.com", ".localhost")
    )
    lowered = url.lower()
    if blocked or "example" in lowered or "localhost" in lowered:
        raise StringSourceError("placeholder url is not admissible")


def _instant(value: str, label: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    elif text.endswith("z"):
        raise StringSourceError(f"{label} uses a lowercase z")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StringSourceError(f"{label} is not a timestamp") from exc
    if parsed.tzinfo is None:
        raise StringSourceError(f"{label} is a naive timestamp")
    return parsed.astimezone(timezone.utc)


def _canonical_from_aware(parsed: datetime) -> str:
    """UTC to the second. Subseconds are truncated, not rounded up."""
    utc = parsed.astimezone(timezone.utc).replace(microsecond=0)
    text = utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not _CANONICAL.match(text):
        raise StringSourceError(f"timestamp {text} is not canonical UTC")
    return text


def _require_canonical(value: str, label: str) -> datetime:
    if not _CANONICAL.match(value):
        raise StringSourceError(f"{label} must be YYYY-MM-DDTHH:MM:SSZ")
    return _instant(value, label)


def parse_outlet_clock(clock: str, raw_published_at: str) -> str:
    """Re-read the outlet's own clock and return canonical UTC."""
    if clock not in CLOCKS:
        raise StringSourceError(f"unknown clock {clock!r}")
    raw = str(raw_published_at).strip()
    if not raw or raw.lower() in {"none", "nan"}:
        raise StringSourceError("raw_published_at is required")
    if clock == "reddit_created_utc":
        try:
            seconds = float(raw)
        except ValueError as exc:
            raise StringSourceError("reddit_created_utc is not a unix time") from exc
        if not math.isfinite(seconds) or seconds < 0:
            raise StringSourceError("reddit_created_utc is not a unix time")
        parsed = datetime.fromtimestamp(int(seconds), tz=timezone.utc)
        return _canonical_from_aware(parsed)
    if clock == "http_last_modified":
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError) as exc:
            raise StringSourceError("http_last_modified is not an HTTP date") from exc
        if parsed is None or parsed.tzinfo is None:
            raise StringSourceError("http_last_modified is not an HTTP date")
        return _canonical_from_aware(parsed)
    if "T" not in raw or len(raw) <= 10:
        raise StringSourceError("bare date is not a timestamp")
    lowered = raw.lower()
    if "ago" in lowered or "yesterday" in lowered or "today" in lowered:
        raise StringSourceError("relative time is not a timestamp")
    return _canonical_from_aware(_instant(raw, clock))


def source_sha256(
    *,
    headline: str,
    body: str,
    url: str,
    clock: str,
    raw_published_at: str,
    published_at_utc: str,
    observed_at_utc: str,
    recorded_at_utc: str = "",
) -> str:
    """Hash the quote and every clock field. Order is fixed."""
    payload = json.dumps(
        [
            headline,
            body,
            url,
            clock,
            raw_published_at,
            published_at_utc,
            observed_at_utc,
            recorded_at_utc,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_string_source(
    raw: Mapping[str, Any],
    *,
    deadline_utc: str,
) -> StringSource:
    """Parse one open-outlet note. Clocks must be the outlet's, in UTC."""
    source = str(raw.get("source") or "").strip()
    if not source:
        raise StringSourceError("source is required")
    outlet = str(raw.get("outlet_class") or "").strip()
    if outlet not in OUTLET_CLASSES:
        raise StringSourceError(f"outlet_class {outlet!r} is not press, forum, youtube, or other")
    clock = str(raw.get("clock") or "").strip()
    raw_published = raw.get("raw_published_at")
    if raw_published is None or str(raw_published).strip() == "":
        raise StringSourceError("raw_published_at is required")
    raw_published = str(raw_published).strip()
    derived = parse_outlet_clock(clock, raw_published)
    published = str(raw.get("published_at_utc") or "").strip()
    published_at = _require_canonical(published, "published_at_utc")
    if published != derived:
        raise StringSourceError(
            f"published_at_utc {published} does not match {clock} raw {raw_published}"
        )
    observed = str(raw.get("observed_at_utc") or "").strip()
    observed_at = _require_canonical(observed, "observed_at_utc")
    recorded_field = raw.get("recorded_at_utc")
    recorded: str | None
    recorded_at: datetime | None
    if recorded_field in (None, ""):
        recorded = None
        recorded_at = None
    else:
        recorded = str(recorded_field).strip()
        recorded_at = _require_canonical(recorded, "recorded_at_utc")
    deadline = _instant(str(deadline_utc), "deadline")
    if recorded_at is not None and recorded_at > published_at:
        raise StringSourceError("recorded_at_utc is after published_at_utc")
    if observed_at < published_at:
        raise StringSourceError("observed_at_utc is before published_at_utc")
    if not observed_at < deadline or not published_at < deadline:
        raise StringSourceError(
            f"timestamp is not before deadline {deadline_utc}"
        )
    headline = str(raw.get("headline") or "").strip()
    body = str(raw.get("body") or raw.get("text") or "").strip()
    if not headline and not body:
        raise StringSourceError("headline or body is required")
    if len(body) > BODY_CAP:
        raise StringSourceError(f"body is longer than {BODY_CAP} characters")
    url = str(raw.get("url") or "").strip()
    _check_url(url)
    digest = source_sha256(
        headline=headline,
        body=body,
        url=url,
        clock=clock,
        raw_published_at=raw_published,
        published_at_utc=published,
        observed_at_utc=observed,
        recorded_at_utc=recorded or "",
    )
    claimed = str(raw.get("sha256") or "").strip()
    if claimed != digest:
        raise StringSourceError("sha256 does not match the quote and its clocks")
    source_id = str(raw.get("source_id") or "").strip() or f"{source}:{outlet}"
    return StringSource(
        source_id=source_id,
        source=source,
        outlet_class=outlet,
        url=url,
        published_at_utc=published,
        observed_at_utc=observed,
        clock=clock,
        raw_published_at=raw_published,
        headline=headline,
        body=body,
        sha256=digest,
        player_ids=tuple(int(pid) for pid in (raw.get("player_ids") or ())),
        recorded_at_utc=recorded,
    )


def load_string_sources(
    gw: int,
    *,
    deadline_utc: str,
    root: Path | None = None,
) -> list[StringSource]:
    """Load ``string_sources/*.json``. Missing directory is an empty list."""
    directory = sources_dir(gw, root)
    if not directory.is_dir():
        return []
    notes: list[StringSource] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or "outlet_class" not in raw:
            continue
        notes.append(validate_string_source(raw, deadline_utc=deadline_utc))
    return notes


def render_source(note: StringSource) -> str:
    """One labelled quote. The outlet class is part of the sentence."""
    when = (
        f"published {note.published_at_utc} via {note.clock}, "
        f"observed {note.observed_at_utc}"
    )
    if note.recorded_at_utc:
        when = f"{when}, recorded {note.recorded_at_utc}"
    return (
        f"- **[{note.outlet_class}]** *{note.source}* ({when})\n"
        f"  > {note.quote()}\n"
        f"  > {note.url}"
    )
