"""Forums, YouTube, and other outlets for the string agent.

These notes are not press packets. ``src/live/news_packets.py`` does not
read this folder, and ``compile_player_xmi`` does not see them. A note
still has to predate the deadline, carry a real URL, and hash its text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from src.live.news_packets import content_sha256
from src.live.news_tags import published_before

OUTLET_CLASSES = frozenset({"press", "forum", "youtube", "other"})
BODY_CAP = 2000

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


def validate_string_source(
    raw: Mapping[str, Any],
    *,
    deadline_utc: str,
) -> StringSource:
    """Parse one open-outlet note. Both clocks must predate the deadline."""
    source = str(raw.get("source") or "").strip()
    if not source:
        raise StringSourceError("source is required")
    outlet = str(raw.get("outlet_class") or "").strip()
    if outlet not in OUTLET_CLASSES:
        raise StringSourceError(f"outlet_class {outlet!r} is not press, forum, youtube, or other")
    published = str(raw.get("published_at_utc") or "").strip()
    if not published:
        raise StringSourceError("published_at_utc is required")
    if not published_before(published, deadline_utc):
        raise StringSourceError(
            f"published_at {published} is not before deadline {deadline_utc}"
        )
    recorded_raw = raw.get("recorded_at_utc")
    recorded: str | None
    if recorded_raw in (None, ""):
        recorded = None
    else:
        recorded = str(recorded_raw).strip()
        if not published_before(recorded, deadline_utc):
            raise StringSourceError(
                f"recorded_at {recorded} is not before deadline {deadline_utc}"
            )
    headline = str(raw.get("headline") or "").strip()
    body = str(raw.get("body") or raw.get("text") or "").strip()
    if not headline and not body:
        raise StringSourceError("headline or body is required")
    if len(body) > BODY_CAP:
        raise StringSourceError(f"body is longer than {BODY_CAP} characters")
    url = str(raw.get("url") or "").strip()
    _check_url(url)
    digest = content_sha256(
        headline=headline, body=body, url=url, published_at_utc=published
    )
    claimed = str(raw.get("sha256") or "").strip()
    if claimed != digest:
        raise StringSourceError("sha256 does not match headline/body/url/published_at")
    source_id = str(raw.get("source_id") or "").strip() or f"{source}:{outlet}"
    return StringSource(
        source_id=source_id,
        source=source,
        outlet_class=outlet,
        url=url,
        published_at_utc=published,
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
    when = note.published_at_utc
    if note.recorded_at_utc:
        when = f"{when}, recorded {note.recorded_at_utc}"
    return (
        f"- **[{note.outlet_class}]** *{note.source}* ({when})\n"
        f"  > {note.quote()}\n"
        f"  > {note.url}"
    )
