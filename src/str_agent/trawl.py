"""Fill one gameweek's string dossier from a fixed list of feeds.

The collector stores the feed title and the feed summary. It does not fetch
the article, the thread page, or the video. Each note is written under
``string_sources/`` and has to pass the existing clock gate. ``news_packets``
is not written, so the minutes compile does not see these files.

A second run skips a URL that is already on disk and does not refresh its
capture time. A feed that fails is skipped. Files already written in the
run stay.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse, urlunparse

from src.str_agent.sources import (
    BODY_CAP,
    StringSourceError,
    parse_outlet_clock,
    source_sha256,
    sources_dir,
    validate_string_source,
)

_BANNED = ("score_xp", "xp_on_pot", "lam_scored")
_USER_AGENT = "fpl-engine-dossier/1.0 (pre-deadline text collector)"
_FETCH_CAP = 1_000_000

Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class Feed:
    """One named outlet. ``kind`` selects the clock, not a general crawler."""

    source: str
    outlet_class: str
    url: str
    kind: str


# Press RSS, two forums, and three channels. Nothing else is polled.
FEEDS: tuple[Feed, ...] = (
    Feed(
        "bbc_sport",
        "press",
        "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "rss",
    ),
    Feed(
        "guardian",
        "press",
        "https://www.theguardian.com/football/premierleague/rss",
        "rss",
    ),
    Feed(
        "skysports",
        "press",
        "https://www.skysports.com/rss/12040",
        "rss",
    ),
    Feed(
        "reddit_fantasypl",
        "forum",
        "https://www.reddit.com/r/FantasyPL/new.rss?limit=15",
        "rss",
    ),
    Feed(
        "reddit_premierleague",
        "forum",
        "https://www.reddit.com/r/PremierLeague/new.rss?limit=15",
        "rss",
    ),
    Feed(
        "premierleague",
        "youtube",
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCG5qGWdu8nIRZqJ_GgDwQ-w",
        "youtube",
    ),
    Feed(
        "skysports_football",
        "youtube",
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCZ7wY7MRDSygp63HIEfdQZA",
        "youtube",
    ),
    Feed(
        "fantasy_football_scout",
        "youtube",
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCKxYKQ8pgJ7V8wwh4hLsSXQ",
        "youtube",
    ),
)


@dataclass(frozen=True)
class RawItem:
    headline: str
    body: str
    url: str
    clock: str
    raw_published_at: str


@dataclass(frozen=True)
class TrawlResult:
    written: int
    skipped_duplicate: int
    dropped: int
    feed_errors: tuple[str, ...]


class TrawlError(ValueError):
    """The run itself was refused. A single feed failure is not this error."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local(child.tag) == name and child.text:
            return str(child.text).strip()
    return ""


def _link(element: ET.Element) -> str:
    plain = _text(element, "link")
    if plain.startswith("http://") or plain.startswith("https://"):
        return plain
    alternate = ""
    for child in element:
        if _local(child.tag) != "link":
            continue
        href = str(child.attrib.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        rel = str(child.attrib.get("rel") or "alternate")
        if rel == "alternate":
            return href
        alternate = alternate or href
    return alternate


def canonical_url(url: str) -> str:
    parsed = urlparse(url.strip())
    return urlunparse(parsed._replace(fragment=""))


def source_filename(url: str) -> str:
    digest = hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:16]
    return f"src_{digest}.json"


def _strip_markup(text: str) -> str:
    out: list[str] = []
    inside = False
    for char in text:
        if char == "<":
            inside = True
            continue
        if char == ">":
            inside = False
            out.append(" ")
            continue
        if not inside:
            out.append(char)
    collapsed = " ".join("".join(out).split())
    return collapsed


def _clip(text: str) -> str:
    body = _strip_markup(text).strip()
    if len(body) > BODY_CAP:
        return body[:BODY_CAP]
    return body


def _banned(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _BANNED)


def _clock_for_rss(raw: str) -> str | None:
    text = raw.strip()
    if not text or "ago" in text.lower() or "yesterday" in text.lower():
        return None
    if re.match(r"\d{4}-\d{2}-\d{2}T", text):
        return "feed_updated"
    candidate = text[:-4] + " +0100" if text.endswith(" BST") else text
    try:
        parsed = parsedate_to_datetime(candidate)
    except (TypeError, ValueError):
        return None
    if parsed is None or parsed.tzinfo is None:
        return None
    return "http_last_modified"


def _entries(payload: bytes) -> list[ET.Element]:
    root = ET.fromstring(payload)
    found = [child for child in root.iter() if _local(child.tag) in {"item", "entry"}]
    return found


def parse_rss(payload: bytes) -> list[RawItem]:
    items: list[RawItem] = []
    for element in _entries(payload):
        raw = _text(element, "pubDate") or _text(element, "updated") or _text(element, "published")
        url = _link(element)
        if not raw or not url:
            continue
        clock = _clock_for_rss(raw) or "feed_updated"
        summary = (
            _text(element, "description")
            or _text(element, "summary")
            or _text(element, "content")
        )
        items.append(
            RawItem(
                headline=_strip_markup(_text(element, "title")),
                body=_clip(summary),
                url=canonical_url(url),
                clock=clock,
                raw_published_at=raw.strip(),
            )
        )
    return items


def parse_youtube(payload: bytes) -> list[RawItem]:
    items: list[RawItem] = []
    for element in _entries(payload):
        raw = _text(element, "published")
        url = _link(element)
        if not raw or not url:
            continue
        summary = _text(element, "summary") or _text(element, "description")
        items.append(
            RawItem(
                headline=_strip_markup(_text(element, "title")),
                body=_clip(summary),
                url=canonical_url(url),
                clock="youtube_published_at",
                raw_published_at=raw.strip(),
            )
        )
    return items


def _unix_raw(value: object) -> str:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number < 0:
        raise ValueError("created_utc is not a unix time")
    return format(number, "f").rstrip("0").rstrip(".") or "0"


def parse_reddit_json(payload: bytes) -> list[RawItem]:
    parsed = json.loads(payload.decode("utf-8"))
    children = ((parsed.get("data") or {}).get("children") or [])
    items: list[RawItem] = []
    for child in children:
        data = (child or {}).get("data") or {}
        permalink = str(data.get("permalink") or "")
        if not permalink.startswith("/"):
            continue
        if "created_utc" not in data:
            continue
        try:
            raw = _unix_raw(data.get("created_utc"))
        except (TypeError, ValueError):
            continue
        items.append(
            RawItem(
                headline=_strip_markup(str(data.get("title") or "")),
                body=_clip(str(data.get("selftext") or "")),
                url=canonical_url("https://www.reddit.com" + permalink),
                clock="reddit_created_utc",
                raw_published_at=raw,
            )
        )
    return items


def parse_feed(kind: str, payload: bytes) -> list[RawItem]:
    if kind == "rss":
        return parse_rss(payload)
    if kind == "youtube":
        return parse_youtube(payload)
    if kind == "reddit_json":
        return parse_reddit_json(payload)
    raise TrawlError(f"unknown feed kind {kind!r}")


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read(_FETCH_CAP)


def _known_urls(directory: Path) -> set[str]:
    found: set[str] = set()
    if not directory.is_dir():
        return found
    for path in directory.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict) and raw.get("url"):
            found.add(canonical_url(str(raw["url"])))
    return found


def _admit(
    item: RawItem,
    *,
    feed: Feed,
    observed_at_utc: str,
    deadline_utc: str,
) -> dict[str, object] | None:
    if _banned(item.headline) or _banned(item.body):
        return None
    if not item.headline and not item.body:
        return None
    try:
        published = parse_outlet_clock(item.clock, item.raw_published_at)
    except StringSourceError:
        return None
    body = item.body[:BODY_CAP]
    url = canonical_url(item.url)
    digest = source_sha256(
        headline=item.headline,
        body=body,
        url=url,
        clock=item.clock,
        raw_published_at=item.raw_published_at,
        published_at_utc=published,
        observed_at_utc=observed_at_utc,
        recorded_at_utc="",
    )
    raw = {
        "source_id": f"{feed.source}:{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}",
        "source": feed.source,
        "outlet_class": feed.outlet_class,
        "url": url,
        "clock": item.clock,
        "raw_published_at": item.raw_published_at,
        "published_at_utc": published,
        "observed_at_utc": observed_at_utc,
        "headline": item.headline,
        "body": body,
        "player_ids": [],
        "sha256": digest,
    }
    try:
        validate_string_source(raw, deadline_utc=deadline_utc)
    except StringSourceError:
        return None
    return raw


def trawl(
    gw: int,
    *,
    deadline_utc: str,
    observed_at_utc: str,
    root: Path | None = None,
    feeds: tuple[Feed, ...] | None = None,
    fetch: Fetch | None = None,
) -> TrawlResult:
    """Poll ``feeds`` (the named list, unless a test passes another)."""
    chosen = FEEDS if feeds is None else feeds
    getter = _fetch if fetch is None else fetch
    directory = sources_dir(int(gw), root)
    directory.mkdir(parents=True, exist_ok=True)
    known = _known_urls(directory)
    written = 0
    skipped = 0
    dropped = 0
    errors: list[str] = []
    for feed in chosen:
        try:
            payload = getter(feed.url)
            items = parse_feed(feed.kind, payload)
        except (OSError, ET.ParseError, json.JSONDecodeError, UnicodeError, ValueError, TrawlError) as exc:
            errors.append(f"{feed.source}: {type(exc).__name__}")
            continue
        for item in items:
            url = canonical_url(item.url)
            path = directory / source_filename(url)
            if url in known or path.is_file():
                skipped += 1
                continue
            note = _admit(
                item,
                feed=feed,
                observed_at_utc=observed_at_utc,
                deadline_utc=deadline_utc,
            )
            if note is None:
                dropped += 1
                continue
            path.write_text(
                json.dumps(note, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            known.add(url)
            written += 1
    return TrawlResult(written, skipped, dropped, tuple(errors))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description="Fill a gameweek string dossier")
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--deadline", type=str, required=True)
    parser.add_argument("--observed", type=str, default="")
    parser.add_argument("--root", type=str, default="")
    args = parser.parse_args(argv)
    observed = args.observed or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    root = Path(args.root) if args.root else None
    result = trawl(
        int(args.gw),
        deadline_utc=args.deadline,
        observed_at_utc=observed,
        root=root,
    )
    json.dump(
        {
            "written": result.written,
            "skipped_duplicate": result.skipped_duplicate,
            "dropped": result.dropped,
            "feed_errors": list(result.feed_errors),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
