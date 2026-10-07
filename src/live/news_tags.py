"""Pre-deadline news tags for the live season.

A tag is used only when the note is dated before that gameweek's deadline.
Expected minutes then follow the tag. The text does not get to invent a
number, and this module does not call the climb or the published score.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

TAGS = ("firm_starter", "injured", "transferred", "benched")
BULK_LIMIT = 10
SUPPORT = {
    "transferred": ("joined", "loan", "sold", "permanently", "returned to"),
    "benched": ("dropped", "benched", "not in the squad", "not in the matchday", "left out", "not included"),
    "injured": ("injur", "suspend", "doubt", "knock"),
    "firm_starter": ("started", "will start", "first choice", "named in the team", "regular starter"),
}
_TRANSFER = ("joined", "loan", "sold", "permanently", "returned to")
_BENCH = ("not included", "not in the squad", "not in the matchday", "left out", "dropped")
_SENTENCE = re.compile(r"[.!?](?:\s+|$)")


class NewsTagError(ValueError):
    """The completion cannot become a tag."""


@dataclass(frozen=True)
class PacketDoc:
    """One note the classifier is allowed to cite."""

    doc_id: str
    text: str
    about_player: bool = False


@dataclass(frozen=True)
class TagDecision:
    """The tag that survived the date check, and the minutes that follow."""

    player_id: int
    gw: int
    tag: str | None
    xmi: float | None
    source: str
    note: str
    prior: float | None
    rejected: str = ""


def parse_time(value: str) -> datetime:
    """An ISO date or timestamp. A bare date is midnight UTC."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def published_before(published_at: str, deadline: str) -> bool:
    """True when the note is known to predate the deadline.

    A bare date has no hour. It counts only when that whole day is before
    the deadline's day. A timestamp must be strictly earlier.
    """
    stamp = published_at.strip()
    cut = parse_time(deadline)
    if len(stamp) == 10:
        return parse_time(stamp).date() < cut.date()
    return parse_time(stamp) < cut


def bulk_seconds(elements: Sequence[Mapping[str, Any]], limit: int = BULK_LIMIT) -> frozenset[str]:
    """Seconds shared by too many players to be a real news time."""
    counts: Counter[str] = Counter()
    for element in elements:
        added = element.get("news_added")
        if added:
            counts[str(added)[:19]] += 1
    return frozenset(second for second, count in counts.items() if count >= limit)


def fpl_record(
    element: Mapping[str, Any],
    deadline: str,
    bulk: frozenset[str],
) -> dict[str, Any] | None:
    """The status line, if its timestamp is usable before this deadline.

    The snapshot's current status is not the status at an earlier deadline.
    A missing time, a bulk stamp, or a time after the deadline returns None.
    """
    added = element.get("news_added")
    if not added or str(added)[:19] in bulk:
        return None
    if not published_before(str(added), deadline):
        return None
    news = str(element.get("news") or "").strip()
    status = str(element.get("status") or "")
    chance = _chance(element.get("chance_of_playing_next_round"))
    if status == "a" and not news and (chance is None or chance >= 100.0):
        return None
    return {"status": status, "chance": chance, "news": news, "news_added": str(added)}


def tag_from_fpl(record: Mapping[str, Any]) -> str:
    """The tag for a dated FPL line. An unclear line is ask."""
    news = str(record.get("news") or "").lower()
    status = str(record.get("status") or "")
    chance = record.get("chance")
    if any(phrase in news for phrase in _TRANSFER):
        return "transferred"
    if any(phrase in news for phrase in _BENCH):
        return "benched"
    if status in {"s", "u", "i"} or chance == 0:
        return "injured"
    if status == "d" or (chance is not None and 0 < float(chance) < 100):
        return "injured"
    return "ask"


def prior_minutes(history: Sequence[float]) -> float | None:
    """Mean of the last three appearances. A zero-minute row is not an appearance."""
    played = [float(minutes) for minutes in history if float(minutes) > 0.0]
    if not played:
        return None
    window = played[-3:]
    return sum(window) / len(window)


def minutes_for_tag(
    tag: str | None,
    position: str,
    prior: float | None,
    chance: float | None,
    status: str | None,
) -> float | None:
    """Minutes for a tag. None means the old minutes stay.

    A suspended or unavailable player, or a chance of 0, is 0 even if the
    tag said otherwise. A firm starter stays in the range 65 to 90.
    """
    if tag in {None, "ask"}:
        return None
    if tag not in TAGS:
        raise NewsTagError(f"unknown tag {tag}")
    if status in {"s", "u"} or chance == 0:
        return 0.0
    base = 90.0 if prior is None else float(prior)
    if tag == "firm_starter":
        if status == "i":
            return 0.0
        return min(90.0, max(65.0, base))
    if tag == "injured":
        if chance is not None and 0.0 < float(chance) < 100.0:
            return (float(chance) / 100.0) * base
        return 0.0
    if tag == "transferred":
        return 0.0
    if position == "GKP":
        return 0.0
    return 15.0


def docs_before(docs: Sequence[Mapping[str, Any]], deadline: str) -> list[Mapping[str, Any]]:
    """Documents whose publication time is before the deadline."""
    return [doc for doc in docs if published_before(str(doc["published_at"]), deadline)]


def build_tag_prompt(
    cases: Sequence[Mapping[str, Any]],
) -> str:
    """One completion for every packet. Later documents are already removed."""
    lines = [
        "Classify each player for that gameweek's deadline.",
        "Use only the documents in that player's packet.",
        "Do not use memory of who played, of a later transfer, or of the score.",
        "Return a JSON array and nothing else. One object per case.",
        '{"player_id": number, "gw": number, "tag": string, "doc_ids": [string], "note": string}.',
        "tag is firm_starter, injured, transferred, benched, or ask.",
        "firm_starter means the documents say he started or is the regular starter.",
        "injured means the documents say he is hurt, suspended, or doubtful.",
        "transferred means the documents say he has left the club.",
        "benched means he is still at the club and the documents say he was left out.",
        "If the documents say he has joined another club, transferred wins over benched.",
        "A deal that is agreed, or a player who is set to sign, is not a transfer and not a bench.",
        "If the documents do not establish one tag, return ask and an empty doc_ids list.",
        "A tag other than ask must cite doc_ids from that packet only.",
        "",
    ]
    for case in cases:
        lines.append(
            f"CASE player_id {case['player_id']} | {case['name']} | {case['position']} | "
            f"gw {case['gw']} | deadline {case['deadline']}"
        )
        if not case["docs"]:
            lines.append("documents: none")
        for doc in case["docs"]:
            lines.append(f"doc {doc.doc_id}: {doc.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_tag_response(text: str) -> list[Any]:
    """A JSON array, or one fenced block whose body is that array."""
    body = text.strip()
    if body.startswith("```"):
        fence = body.splitlines()
        if len(fence) < 3 or not fence[-1].strip().startswith("```"):
            raise NewsTagError("the completion is not a JSON array")
        body = "\n".join(fence[1:-1]).strip()
    if not body.startswith("["):
        raise NewsTagError("the completion is not a JSON array")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise NewsTagError("the completion is not valid JSON") from exc
    if not isinstance(data, list):
        raise NewsTagError("the completion is not a JSON array")
    return data


def accept_tag(
    proposal: Mapping[str, Any],
    docs: Sequence[PacketDoc],
    *,
    position: str,
    prior: float | None,
    status: str | None,
    chance: float | None,
    name: str,
) -> TagDecision:
    """Accept a cited tag. A late document, or text that does not support it, fails."""
    player_id = int(proposal["player_id"])
    gw = int(proposal["gw"])
    tag = str(proposal.get("tag") or "")
    note = str(proposal.get("note") or "")
    if tag == "ask":
        return TagDecision(player_id, gw, "ask", None, "llm", note, prior)
    if tag not in TAGS:
        raise NewsTagError(f"unknown tag {tag}")
    raw_ids = proposal.get("doc_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise NewsTagError("a tag needs a document from the packet")
    by_id = {doc.doc_id: doc for doc in docs}
    cited: list[PacketDoc] = []
    for doc_id in raw_ids:
        key = str(doc_id)
        if key not in by_id:
            raise NewsTagError(f"{key} is not in the packet")
        cited.append(by_id[key])
    if not _cited_supports(tag, cited, name):
        raise NewsTagError("the cited text does not support the tag")
    if tag == "firm_starter" and (status in {"s", "u", "i"} or chance == 0):
        raise NewsTagError("a ruled-out player cannot be a firm starter")
    xmi = minutes_for_tag(tag, position, prior, chance, status)
    return TagDecision(player_id, gw, tag, xmi, "llm", note, prior)


def fold(text: str) -> str:
    """Lower case, accents removed."""
    normal = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in normal if not unicodedata.combining(char))
    return stripped.lower()


def _cited_supports(tag: str, cited: Sequence[PacketDoc], name: str) -> bool:
    words = SUPPORT[tag]
    folded_name = fold(name)
    for doc in cited:
        if doc.about_player:
            if any(word in fold(doc.text) for word in words):
                return True
            continue
        for sentence in _sentences(doc.text):
            folded = fold(sentence)
            if folded_name in folded and any(word in folded for word in words):
                return True
    return False


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text.strip()) if part.strip()]


def _chance(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number
