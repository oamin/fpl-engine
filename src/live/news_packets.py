"""Audited news packets for deeper live tags.

Gemini 2026-10-08 (bc-90156d5d): whitelist sources only; published_at must
predate the deadline; cite-or-reject via ``news_tags.PacketDoc``. Numeric ``xmi`` compile is
``compile_player_xmi`` / ``compile_xmi_from_numeric`` / ``compile_high_profile_test``
(sidecar only; tag path remains default).

Packets live under ``data/predictions/2026-27/gwNN/news_packets/*.json``.
When that directory is empty, ``load_gameweek_packets`` synthesises docs from
FPL bootstrap ``news`` / ``news_added`` so the live path never regresses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.live.news_tags import PacketDoc, published_before

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data" / "predictions" / "2026-27"

# v1 whitelist: club media + PL + tier-1 outlets + synthetic FPL bootstrap.
SOURCE_WHITELIST = frozenset(
    {
        "fpl_bootstrap",
        "pl_official",
        "bbc_sport",
        "the_athletic",
        "skysports",
        "guardian",
        "arsenal_fc",
        "astonvilla_fc",
        "bournemouth_fc",
        "brentford_fc",
        "brighton_fc",
        "chelsea_fc",
        "crystalpalace_fc",
        "everton_fc",
        "fulham_fc",
        "ipswich_fc",
        "leeds_fc",
        "liverpool_fc",
        "mancity_fc",
        "manutd_fc",
        "newcastle_fc",
        "nottinghamforest_fc",
        "southampton_fc",
        "tottenham_fc",
        "west_ham_fc",
        "wolves_fc",
        "burnley_fc",
        "sunderland_fc",
        "coventry_fc",
        # PI 2026-10-08: local and specialist reports for the GW6 table.
        "manchestereveningnews",
        "sportsmole",
        "football_london",
        "the_standard",
        "coventry_telegraph",
        "chronicle_live",
        "haaglanden_voetbal",
        # Local desk for every 2026/27 club, same class as MEN / Chronicle.
        "birmingham_mail",
        "bournemouth_echo",
        "the_argus",
        "liverpool_echo",
        "hull_daily_mail",
        "east_anglian_daily_times",
        "yorkshire_evening_post",
        "nottingham_post",
        "sunderland_echo",
    }
)

# Club name in the FPL bootstrap → local press keys. League-wide outlets
# (bbc_sport, sportsmole, the_standard, guardian, skysports) sit outside this map.
CLUB_PRESS: dict[str, tuple[str, ...]] = {
    "Arsenal": ("football_london", "the_standard"),
    "Aston Villa": ("birmingham_mail",),
    "Bournemouth": ("bournemouth_echo",),
    "Brentford": ("football_london",),
    "Brighton": ("the_argus",),
    "Chelsea": ("football_london",),
    "Coventry City": ("coventry_telegraph",),
    "Crystal Palace": ("football_london",),
    "Everton": ("liverpool_echo",),
    "Fulham": ("football_london",),
    "Hull City": ("hull_daily_mail",),
    "Ipswich Town": ("east_anglian_daily_times",),
    "Leeds": ("yorkshire_evening_post",),
    "Liverpool": ("liverpool_echo",),
    "Man City": ("manchestereveningnews",),
    "Man Utd": ("manchestereveningnews",),
    "Newcastle": ("chronicle_live",),
    "Nott'm Forest": ("nottingham_post",),
    "Spurs": ("football_london",),
    "Sunderland": ("sunderland_echo",),
}

_SLUG = re.compile(r"[^a-z0-9]+")


class PacketError(ValueError):
    """Packet schema or provenance failed."""


class LeakageError(PacketError):
    """Document is on or after the deadline."""


class WhitelistError(PacketError):
    """Source is not on the v1 whitelist."""


@dataclass(frozen=True)
class NewsPacket:
    """One audited note. ``sha256`` covers headline+body+url+published_at."""

    packet_id: str
    source: str
    url: str
    published_at_utc: str
    club: str | None
    player_ids: tuple[int, ...]
    headline: str
    body: str
    sha256: str

    def to_packet_doc(self) -> PacketDoc:
        """Shape used by ``news_tags.build_tag_prompt`` / ``accept_tag``."""
        text = self.headline.strip()
        body = self.body.strip()
        if body:
            text = f"{text}. {body}" if text else body
        return PacketDoc(self.packet_id, text, about_player=bool(self.player_ids))


def packets_dir(gw: int, root: Path | None = None) -> Path:
    base = root or ROOT
    return base / "data" / "predictions" / "2026-27" / f"gw{int(gw):02d}" / "news_packets"


def content_sha256(
    *,
    headline: str,
    body: str,
    url: str,
    published_at_utc: str,
) -> str:
    payload = json.dumps(
        {
            "headline": headline,
            "body": body,
            "url": url,
            "published_at_utc": published_at_utc,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _slug(text: str) -> str:
    return _SLUG.sub("-", text.lower()).strip("-")[:48] or "note"


def make_packet_id(source: str, gw: int, headline: str) -> str:
    return f"{source}:gw{int(gw):02d}:{_slug(headline)}"


def _reject_placeholder_url(url: str, *, source: str) -> None:
    """Refuse invented demo links. ``fpl:bootstrap`` is the only non-http URL.

    A path or host containing ``example`` is not a dated article. The GW6
    Haaland file used one of these and quoted Pep Guardiola after he had
    left Manchester City (Enzo Maresca appointed 29 June 2026).
    """
    if source == "fpl_bootstrap":
        if url != "fpl:bootstrap":
            raise PacketError("fpl_bootstrap packets must use url fpl:bootstrap")
        return
    lowered = url.lower()
    if not lowered.startswith(("http://", "https://")):
        raise PacketError("url must be http(s) for a non-FPL source")
    if "example" in lowered or "localhost" in lowered:
        raise PacketError(
            "placeholder url is not admissible provenance "
            "(host or path contains 'example' or 'localhost')"
        )


def validate_packet(
    raw: Mapping[str, Any],
    *,
    deadline_utc: str,
    require_whitelist: bool = True,
) -> NewsPacket:
    """Parse and check one packet dict."""
    source = str(raw.get("source") or "").strip()
    if require_whitelist and source not in SOURCE_WHITELIST:
        raise WhitelistError(f"source {source!r} is not on the v1 whitelist")
    published = str(raw.get("published_at_utc") or raw.get("published_at") or "").strip()
    if not published:
        raise PacketError("published_at_utc is required")
    if not published_before(published, deadline_utc):
        raise LeakageError(
            f"packet published_at {published} is not before deadline {deadline_utc}"
        )
    headline = str(raw.get("headline") or "").strip()
    body = str(raw.get("body") or raw.get("text") or "").strip()
    url = str(raw.get("url") or "").strip()
    if not headline and not body:
        raise PacketError("headline or body is required")
    _reject_placeholder_url(url, source=source)
    player_ids = tuple(int(pid) for pid in (raw.get("player_ids") or ()))
    club = raw.get("club")
    club_s = None if club is None or str(club).strip() == "" else str(club)
    digest = str(raw.get("sha256") or "").strip()
    expected = content_sha256(
        headline=headline, body=body, url=url, published_at_utc=published
    )
    if digest and digest != expected:
        raise PacketError("sha256 does not match headline/body/url/published_at")
    if not digest:
        digest = expected
    packet_id = str(raw.get("packet_id") or "").strip()
    if not packet_id:
        raise PacketError("packet_id is required")
    return NewsPacket(
        packet_id=packet_id,
        source=source,
        url=url,
        published_at_utc=published,
        club=club_s,
        player_ids=player_ids,
        headline=headline,
        body=body,
        sha256=digest,
    )


def write_packet(packet: NewsPacket, directory: Path) -> Path:
    """Write one packet JSON (pretty). Does not trust caller sha without check."""
    directory.mkdir(parents=True, exist_ok=True)
    expected = content_sha256(
        headline=packet.headline,
        body=packet.body,
        url=packet.url,
        published_at_utc=packet.published_at_utc,
    )
    if packet.sha256 != expected:
        raise PacketError("refusing to write packet with mismatched sha256")
    path = directory / f"{_slug(packet.packet_id)}.json"
    path.write_text(json.dumps(asdict(packet), indent=2) + "\n", encoding="utf-8")
    return path


def load_packet_files(
    directory: Path,
    *,
    deadline_utc: str,
) -> list[NewsPacket]:
    """Load every ``*.json`` packet under ``directory`` (missing dir → empty)."""
    if not directory.is_dir():
        return []
    out: list[NewsPacket] = []
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PacketError(f"{path.name} is not a JSON object")
        # Sidecar compile dumps are not packets (no source / sha256).
        if "source" not in raw or "sha256" not in raw:
            continue
        out.append(validate_packet(raw, deadline_utc=deadline_utc))
    return out


def synthetic_fpl_packets(
    bootstrap: Mapping[str, Any],
    *,
    gw: int,
    deadline_utc: str,
    owned: set[int] | None = None,
) -> list[NewsPacket]:
    """Build packets from dated FPL element news lines."""
    from src.live.news_tags import bulk_seconds, fpl_record

    elements = list(bootstrap.get("elements") or [])
    bulk = bulk_seconds(elements)
    teams = {int(row["id"]): str(row["name"]) for row in bootstrap.get("teams") or []}
    packets: list[NewsPacket] = []
    for element in elements:
        pid = int(element["id"])
        if owned is not None and pid not in owned:
            status = str(element.get("status") or "")
            chance = element.get("chance_of_playing_next_round")
            if status == "a" and (chance is None or float(chance) >= 100.0):
                news = str(element.get("news") or "").strip()
                if not news:
                    continue
        record = fpl_record(element, deadline_utc, bulk)
        if record is None:
            continue
        news = str(record["news"] or "").strip()
        if not news:
            continue
        published = str(record["news_added"])
        if len(published) == 10:
            published = published + "T00:00:00Z"
        elif published.endswith("+00:00"):
            published = published[:-6] + "Z"
        chance = record.get("chance")
        chance_text = "blank" if chance is None else f"{float(chance):g}"
        headline = f"{element.get('web_name') or pid}: {news}"
        body = (
            f"status {record['status']}. chance {chance_text}. "
            f"club {teams.get(int(element['team']), '')}."
        )
        digest = content_sha256(
            headline=headline, body=body, url="fpl:bootstrap", published_at_utc=published
        )
        packets.append(
            NewsPacket(
                packet_id=make_packet_id("fpl_bootstrap", gw, f"{pid}-{news[:32]}"),
                source="fpl_bootstrap",
                url="fpl:bootstrap",
                published_at_utc=published,
                club=teams.get(int(element["team"])),
                player_ids=(pid,),
                headline=headline,
                body=body,
                sha256=digest,
            )
        )
    return packets


def load_gameweek_packets(
    gw: int,
    *,
    deadline_utc: str,
    bootstrap: Mapping[str, Any] | None = None,
    root: Path | None = None,
    owned: set[int] | None = None,
    include_synthetic_fpl: bool = True,
) -> list[NewsPacket]:
    """External packets plus optional synthetic FPL news."""
    external = load_packet_files(packets_dir(gw, root), deadline_utc=deadline_utc)
    if not include_synthetic_fpl or bootstrap is None:
        return external
    synthetic = synthetic_fpl_packets(
        bootstrap, gw=gw, deadline_utc=deadline_utc, owned=owned
    )
    by_id = {packet.packet_id: packet for packet in synthetic}
    for packet in external:
        by_id[packet.packet_id] = packet
    return sorted(by_id.values(), key=lambda row: (row.published_at_utc, row.packet_id))


def to_packet_docs(packets: Sequence[NewsPacket]) -> list[PacketDoc]:
    return [packet.to_packet_doc() for packet in packets]


_INJURY_NEGATION = (
    "not nursing an injury",
    "not an injury",
    "no injury",
    "not a serious injury",
    "injury eased",
    "concerns over a serious injury eased",
)


def _clears_doubt(text: str) -> bool:
    """True when the note says the player should still be available."""
    return any(
        phrase in text
        for phrase in (
            "available",
            "not nursing an injury",
            "not an injury",
            "injury eased",
            "back training",
            "back in training",
            "could return this weekend",
        )
    )


_FOLD_EXTRA = str.maketrans({"ø": "o", "ö": "o", "æ": "ae", "å": "a", "ł": "l", "đ": "d", "ß": "ss"})
_NAME_DROP = frozenset({"jr", "sr", "ii", "iii"})


def name_tokens(text: str) -> list[str]:
    """Fold a display name to comparable tokens. ``ø`` becomes ``o``; ``Jr`` drops."""
    from src.live.news_tags import fold

    folded = fold(text).translate(_FOLD_EXTRA)
    parts = re.sub(r"[^a-z0-9]+", " ", folded).split()
    return [part for part in parts if part not in _NAME_DROP]


def match_element_id(
    name: str,
    club: str,
    elements: Sequence[Mapping[str, Any]],
    team_names: Mapping[int, str],
) -> int | None:
    """Map an article name to one FPL element id at ``club``.

    Last-name hits stay only when one player at the club has that surname.
    A first name breaks a tie (Lewis Miley, not Mason Miley). ``None`` means
    no unique match.
    """
    parts = name_tokens(name)
    if not parts:
        return None
    scored: list[tuple[int, int]] = []
    for element in elements:
        if team_names.get(int(element["team"])) != club:
            continue
        web = name_tokens(str(element.get("web_name") or ""))
        first = name_tokens(str(element.get("first_name") or ""))
        second = name_tokens(str(element.get("second_name") or ""))
        full = first + second
        score = 0
        if parts in (web, full, second):
            score = 100
        elif web and parts[-1] == web[-1]:
            score = 80 if first and parts[0] == first[0] else 40
        elif second and parts[-1] in {second[0], second[-1]}:
            # Surname may be the last token of a compound second_name (Rúben Dias).
            score = 80 if first and parts[0] == first[0] else 40
        if score:
            scored.append((score, int(element["id"])))
    if not scored:
        return None
    best = max(score for score, _pid in scored)
    winners = {pid for score, pid in scored if score == best}
    if len(winners) != 1:
        return None
    return next(iter(winners))


def _tag_one_packet(packet: NewsPacket, *, player_name: str) -> str:
    """Closed tag for a single packet. Negated injury lines do not count.

    ``could return this weekend`` is logged ``50/50`` (half the prior), not ``none``.
    """
    from src.live.news_tags import SUPPORT, fold

    name = fold(player_name)
    text = fold(f"{packet.headline} {packet.body}")
    if "could return this weekend" in text:
        return "50/50"
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    hits: set[str] = set()
    for tag, phrases in SUPPORT.items():
        for sentence in sentences:
            if tag == "injured" and any(phrase in sentence for phrase in _INJURY_NEGATION):
                continue
            matched = [phrase for phrase in phrases if fold(phrase) in sentence]
            if tag == "injured" and matched == ["doubt"] and _clears_doubt(text):
                continue
            if not matched:
                continue
            if tag == "firm_starter" and name and name not in text:
                continue
            hits.add(tag)
            break
    # Override tags only for compile; firm_starter is diagnostic → none (rolling).
    for tag in ("transferred", "injured", "benched"):
        if tag in hits:
            return tag
    return "none"


def classify_packets_deterministic(
    packets: Sequence[NewsPacket],
    *,
    player_name: str = "",
) -> tuple[str, list[str]]:
    """Map packet text to a closed tag via ``news_tags.SUPPORT`` phrases.

    The latest ``published_at_utc`` wins, including ``none`` (rolling default).
    An older injury line does not outvote a later note that the player is available.
    Within that packet: transferred > injured > benched; else none.
    """
    if not packets:
        return "none", []
    deciding = max(packets, key=lambda packet: (packet.published_at_utc, packet.packet_id))
    tag = _tag_one_packet(deciding, player_name=player_name)
    return tag, [deciding.packet_id]


def render_player_context(
    *,
    player_id: int,
    name: str,
    position: str,
    club: str,
    prior: float | None,
    status: str | None,
    chance: float | None,
    fpl_news: str,
    packets: Sequence[NewsPacket],
) -> str:
    """Markdown block showing the contextual evidence the compile sees."""
    prior_text = "none" if prior is None else f"{float(prior):.1f}"
    chance_text = "blank" if chance is None else f"{float(chance):g}"
    lines = [
        f"### Player: {name} (id {player_id})",
        f"- Club: {club} | Position: {position}",
        f"- Prior minutes (compile base): {prior_text}",
        f"- FPL flag: status `{status or ''}`, chance {chance_text}",
        f"- FPL news: {fpl_news or '(none)'}",
        "",
        "#### Audited packets (pre-deadline)",
    ]
    if not packets:
        lines.append("- (none)")
    for packet in packets:
        lines.append(
            f"- **[{packet.packet_id}]** *{packet.source}* ({packet.published_at_utc})"
        )
        lines.append(f"  > {packet.headline}")
        if packet.body:
            lines.append(f"  > {packet.body}")
    return "\n".join(lines) + "\n"


def rolling_game_minutes(
    history: Sequence[float],
    window: int = 3,
    *,
    rho: float = 1.0,
) -> float | None:
    """Mean of the last ``window`` gameweek rows, including zeros.

    ``rho=1`` is a flat mean (legacy default). ``rho<1`` is geometric decay
    with the most recent row weight 1 and older rows ``rho^k``
    (``protocol.score_xp_v2_minutes_decay``).

    A short history uses every row it has. An empty history returns None.
    """
    rows = [float(minutes) for minutes in history]
    if not rows:
        return None
    sample = rows[-int(window) :]
    n = len(sample)
    if n == 0:
        return None
    rho = float(rho)
    if rho >= 1.0 - 1e-12:
        return sum(sample) / n
    # Chronological oldest→newest: newest gets rho^0.
    weights = [rho ** (n - 1 - i) for i in range(n)]
    wsum = sum(weights)
    if wsum <= 0:
        return sum(sample) / n
    return sum(w * m for w, m in zip(weights, sample, strict=True)) / wsum


def rolling_game_minutes_sigma(history: Sequence[float], window: int = 3) -> float:
    """Sample SD of the last ``window`` gameweek rows, zeros included.

    Needs at least two rows in the window; otherwise 0.
    """
    rows = [float(minutes) for minutes in history]
    if not rows:
        return 0.0
    sample = rows[-int(window) :]
    if len(sample) < 2:
        return 0.0
    mean = sum(sample) / len(sample)
    var = sum((x - mean) ** 2 for x in sample) / (len(sample) - 1)
    return float(var**0.5)


def _sample_sigma(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return float(var**0.5)


def sigma_m_for_regime(
    *,
    tag: str,
    xmi: float | None,
    minutes: Sequence[float] | None,
    prior: float | None = None,
    window: int = 3,
) -> float:
    """Minutes uncertainty: tag sets the regime, history sets the spread.

    - Hard zero ``xmi`` → σ = 0.
    - ``50/50`` → mixture SD of {0, base} (regime is play-or-not).
    - ``firm_starter`` → SD of ≥60′ apps in the window when ≥2 exist, else
      full-window SD (zeros included).
    - ``rolling avg`` / ``benched`` / other → SD of the same 3-GW window.
    """
    if xmi is None:
        return 0.0
    if float(xmi) <= 0.0:
        return 0.0

    series = [float(m) for m in (minutes or ())]
    window = max(1, int(window))
    hist = rolling_game_minutes_sigma(series, window=window)

    if tag == "50/50":
        base = 90.0 if prior is None else float(prior)
        mu = 0.5 * base
        # Two-point mixture: 0 or base with equal weight.
        return float((0.5 * (0.0 - mu) ** 2 + 0.5 * (base - mu) ** 2) ** 0.5)

    if tag == "firm_starter":
        sample = series[-window:] if series else []
        starts = [m for m in sample if m >= 60.0]
        if len(starts) >= 2:
            return _sample_sigma(starts)
        return hist

    return hist


# Tags that replace the rolling default. ``none`` / ``firm_starter`` / empty → rolling.
XMI_OVERRIDE_TAGS = frozenset({"injured", "transferred", "benched", "50/50"})

# Predicted-XI blobs: need enough resolved names to trust the list.
_XI_MARKER = re.compile(
    r"(?:predicted|expected|likely)\s+(?:starting\s+)?"
    r"(?:xi|eleven|line[\s-]?up|team)\s*[:\-–]\s*",
    re.IGNORECASE,
)
_MIN_PREDICTED_XI = 9


def extract_predicted_xi_names(text: str) -> list[str] | None:
    """Pull player-name tokens from the last ``predicted XI: …`` list in ``text``.

    Returns ``None`` when no marker is found or the list is too short to trust.
    """
    matches = list(_XI_MARKER.finditer(text or ""))
    if not matches:
        return None
    blob = text[matches[-1].end() :]
    # Stop before a new prose sentence when possible.
    blob = re.split(r"\n\n|\.\s+(?=[A-Z][a-z]+\s+(?:says|said|was|is|has|may|could))", blob, maxsplit=1)[0]
    names: list[str] = []
    for part in re.split(r"[;,]", blob):
        piece = part.strip()
        piece = re.sub(r"^(?:and|&)\s+", "", piece, flags=re.IGNORECASE)
        piece = piece.rstrip(".").strip()
        if len(piece) < 2:
            continue
        if len(piece.split()) > 4:
            continue
        names.append(piece)
    return names if len(names) >= 8 else None


def resolve_predicted_xi(
    packet: NewsPacket,
    bootstrap: Mapping[str, Any],
) -> tuple[int, ...] | None:
    """Map a packet's predicted-XI name list to FPL ids at ``packet.club``.

    Needs a club and at least ``_MIN_PREDICTED_XI`` unique resolutions.
    """
    club = (packet.club or "").strip()
    if not club:
        return None
    names = extract_predicted_xi_names(f"{packet.headline}. {packet.body}")
    if not names:
        return None
    elements = list(bootstrap.get("elements") or [])
    team_names = {int(row["id"]): str(row["name"]) for row in bootstrap.get("teams") or []}
    # Packet club may be short ("Man City") — accept exact bootstrap name only via
    # match_element_id; also try short_name → name.
    short_to_name = {
        str(row.get("short_name") or ""): str(row["name"]) for row in bootstrap.get("teams") or []
    }
    club_key = club
    if club not in team_names.values() and club in short_to_name:
        club_key = short_to_name[club]
    resolved: list[int] = []
    seen: set[int] = set()
    for name in names:
        pid = match_element_id(name, club_key, elements, team_names)
        if pid is None and club_key != club:
            pid = match_element_id(name, club, elements, team_names)
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        resolved.append(pid)
    if len(resolved) < _MIN_PREDICTED_XI:
        return None
    return tuple(resolved)


def predicted_lineup_roles(
    packets: Sequence[NewsPacket],
    bootstrap: Mapping[str, Any],
) -> dict[int, tuple[str, str]]:
    """Latest predicted XI per club → ``xi`` / ``rotated`` roles.

    A player named in the latest XI for his club is ``xi``. A player who
    appeared in an earlier predicted XI for that club but not the latest is
    ``rotated`` (treated as benched). Squad mates never named in any predicted
    XI are omitted (rolling default). Availability SUPPORT tags still beat
    these roles inside ``compile_player_xmi``.
    """
    by_club: dict[str, list[tuple[str, str, tuple[int, ...]]]] = {}
    for packet in packets:
        xi = resolve_predicted_xi(packet, bootstrap)
        if not xi:
            continue
        club = (packet.club or "").strip()
        if not club:
            continue
        by_club.setdefault(club, []).append(
            (packet.published_at_utc, packet.packet_id, xi)
        )
    roles: dict[int, tuple[str, str]] = {}
    for _club, rows in by_club.items():
        rows_sorted = sorted(rows, key=lambda row: (row[0], row[1]))
        ever: set[int] = set()
        for _ts, _pid, xi in rows_sorted:
            ever.update(xi)
        latest_ts, latest_id, latest_xi = rows_sorted[-1]
        latest_set = set(latest_xi)
        for pid in latest_xi:
            roles[int(pid)] = ("xi", latest_id)
        for pid in ever - latest_set:
            # Earlier XI only — left out of the latest predicted side.
            roles[int(pid)] = ("rotated", latest_id)
    return roles


def compile_xmi_from_numeric(
    *,
    player_id: int,
    name: str,
    position: str,
    prior: float | None,
    status: str | None,
    chance: float | None,
    p_start: float,
    m_if_start: float,
    m_if_bench: float = 0.0,
    minutes: Sequence[float] | None = None,
    cite: Sequence[str] | None = None,
) -> dict[str, Any]:
    """LLM / structured context → continuous minutes (live sidecar).

    ``m = clip(p_start,0,1) · m_if_start + (1−p) · m_if_bench``, then hard FPL
    flags force 0. Does not invent numbers from free text — caller supplies the
    triple. Closed-season likelihood is not claimed without packet archives.

    Pre-registered role: ``protocol.live_xmi_numeric_sidecar``.
    """
    series = list(minutes) if minutes is not None else []
    p = float(min(1.0, max(0.0, float(p_start))))
    m_xi = float(max(0.0, float(m_if_start)))
    m_bn = float(max(0.0, float(m_if_bench)))
    xmi = p * m_xi + (1.0 - p) * m_bn
    tag = "numeric"
    source = "llm_numeric"
    if status in {"s", "u"} or chance == 0:
        xmi = 0.0
        tag = "injured"
        source = "override:hard_flag"
    # Mixture SD when p in (0,1); else regime from numeric point.
    if source == "llm_numeric" and 0.0 < p < 1.0:
        mu = xmi
        sigma = float(
            (p * (m_xi - mu) ** 2 + (1.0 - p) * (m_bn - mu) ** 2) ** 0.5
        )
    else:
        sigma = sigma_m_for_regime(
            tag="rolling avg" if xmi and xmi > 0 else "injured",
            xmi=xmi,
            minutes=series,
            prior=prior,
            window=3,
        )
    return {
        "player_id": int(player_id),
        "name": name,
        "position": position,
        "prior": prior,
        "status": status,
        "chance": chance,
        "context_tag": tag,
        "cited_packet_ids": list(cite or ()),
        "xmi_source": source,
        "p_start": p,
        "m_if_start": m_xi,
        "m_if_bench": m_bn,
        "xmi_compiled": float(xmi),
        "xmi_sigma": float(sigma),
    }


def compile_player_xmi(
    *,
    player_id: int,
    name: str,
    position: str,
    prior: float | None,
    status: str | None,
    chance: float | None,
    packets: Sequence[NewsPacket],
    minutes: Sequence[float] | None = None,
    lineup_role: str | None = None,
    lineup_packet_id: str | None = None,
) -> dict[str, Any]:
    """Deterministic packet → minutes (sidecar, not live CSV).

    **Default:** rolling mean of the last three gameweek rows (zeros included);
    if the series is empty, the appearance ``prior``.

    **Override** only for clearly flagged context: ``injured``, ``transferred``,
    ``benched`` (known rotation), ``50/50`` (could-return). ``none`` (and any
    non-override tag) keeps the rolling default — there is no ``ask``. Hard FPL
    flags (suspended / unavailable / chance 0) always force 0.

    **Predicted XI** (from ``predicted_lineup_roles``): ``lineup_role='xi'``
    applies the firm-starter floor; ``'rotated'`` applies benched minutes when
    the player left an earlier predicted XI. Availability SUPPORT tags beat
    lineup roles. Do not attach full-XI packets via ``player_ids`` just to
    drive classify — ``left out`` phrases about another name would leak.

    ``xmi_sigma`` is the regime spread (3-GW history, or 50/50 mixture).
    """
    from src.live.news_tags import minutes_for_tag

    series = list(minutes) if minutes is not None else []
    rolling = rolling_game_minutes(series)
    default_xmi: float | None = rolling if rolling is not None else prior

    context_tag, cited = classify_packets_deterministic(packets, player_name=name)

    # Hard availability flags win before any tag.
    if status in {"s", "u"} or chance == 0:
        tag = (
            "injured"
            if context_tag in {None, "ask", "none", "firm_starter", "rolling avg"}
            else context_tag
        )
        xmi: float | None = 0.0
        source = "override:hard_flag"
    elif context_tag in XMI_OVERRIDE_TAGS:
        if context_tag == "50/50":
            base = 90.0 if prior is None else float(prior)
            xmi = 0.5 * base
            tag = "50/50"
        else:
            xmi = minutes_for_tag(context_tag, position, prior, chance, status)
            tag = context_tag
        source = f"override:{context_tag}"
    elif lineup_role == "xi":
        xmi = minutes_for_tag("firm_starter", position, prior, chance, status)
        if xmi is None:
            xmi = 75.0
        tag = "lineup_xi"
        source = "lineup:xi"
        if lineup_packet_id:
            cited = [lineup_packet_id]
    elif lineup_role == "rotated":
        xmi = minutes_for_tag("benched", position, prior, chance, status)
        if xmi is None:
            xmi = 15.0
        tag = "benched"
        source = "lineup:rotated"
        if lineup_packet_id:
            cited = [lineup_packet_id]
    else:
        # none / firm_starter / empty → rolling default (no ask)
        xmi = default_xmi
        tag = "rolling avg"
        source = "rolling"

    sigma = sigma_m_for_regime(
        tag=tag,
        xmi=xmi,
        minutes=series,
        prior=prior,
        window=3,
    )
    return {
        "player_id": int(player_id),
        "name": name,
        "position": position,
        "tag": tag,
        "context_tag": context_tag,
        "xmi_source": source,
        "cited_packet_ids": cited,
        "prior": prior,
        "status": status,
        "chance": chance,
        "xmi_compiled": xmi,
        "xmi_sigma": sigma,
        "n_packets": len(packets),
        "lineup_role": lineup_role,
    }


def compile_high_profile_test(
    *,
    gw: int,
    deadline_utc: str,
    bootstrap: Mapping[str, Any],
    player_ids: Sequence[int],
    history: Mapping[int, Sequence[float]] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Compile a small set of players; write context + results for review."""
    from src.live.fpl_snapshot import ELEMENT
    from src.live.news_tags import prior_minutes

    packets = load_gameweek_packets(
        gw,
        deadline_utc=deadline_utc,
        bootstrap=bootstrap,
        root=root,
        include_synthetic_fpl=True,
    )
    lineup_roles = predicted_lineup_roles(packets, bootstrap)
    elements = {int(row["id"]): row for row in bootstrap.get("elements") or []}
    teams = {int(row["id"]): str(row["name"]) for row in bootstrap.get("teams") or []}
    hist = history or {}
    rows: list[dict[str, Any]] = []
    contexts: list[str] = []
    for pid in player_ids:
        element = elements[int(pid)]
        mine = packets_for_player(packets, int(pid))
        prior = prior_minutes(list(hist.get(int(pid), [])))
        status = str(element.get("status") or "")
        chance_raw = element.get("chance_of_playing_next_round")
        try:
            chance = None if chance_raw is None else float(chance_raw)
        except (TypeError, ValueError):
            chance = None
        name = str(element.get("web_name") or pid)
        position = ELEMENT[int(element["element_type"])]
        club = teams.get(int(element["team"]), "")
        role, role_pkt = lineup_roles.get(int(pid), (None, None))
        contexts.append(
            render_player_context(
                player_id=int(pid),
                name=name,
                position=position,
                club=club,
                prior=prior,
                status=status,
                chance=chance,
                fpl_news=str(element.get("news") or ""),
                packets=mine,
            )
        )
        rows.append(
            compile_player_xmi(
                player_id=int(pid),
                name=name,
                position=position,
                prior=prior,
                status=status,
                chance=chance,
                packets=mine,
                minutes=list(hist.get(int(pid), [])),
                lineup_role=role,
                lineup_packet_id=role_pkt,
            )
        )
    return {
        "gw": int(gw),
        "deadline_utc": deadline_utc,
        "compile": "SUPPORT phrases + predicted-XI lineup roles",
        "players": rows,
        "context_markdown": "\n".join(contexts),
    }


def packets_for_player(packets: Sequence[NewsPacket], player_id: int) -> list[NewsPacket]:
    return [packet for packet in packets if int(player_id) in packet.player_ids]


def build_player_cases(
    packets: Sequence[NewsPacket],
    bootstrap: Mapping[str, Any],
    *,
    gw: int,
    deadline_utc: str,
) -> list[dict[str, Any]]:
    """Cases shaped for ``news_tags.build_tag_prompt`` (docs as PacketDoc)."""
    from src.live.fpl_snapshot import ELEMENT

    elements = {int(row["id"]): row for row in bootstrap.get("elements") or []}
    teams = {int(row["id"]): str(row["name"]) for row in bootstrap.get("teams") or []}
    wanted = sorted({pid for packet in packets for pid in packet.player_ids})
    cases: list[dict[str, Any]] = []
    for pid in wanted:
        element = elements.get(pid)
        if element is None:
            continue
        docs = [packet.to_packet_doc() for packet in packets_for_player(packets, pid)]
        if not docs:
            continue
        cases.append(
            {
                "player_id": pid,
                "name": str(element.get("web_name") or pid),
                "position": ELEMENT[int(element["element_type"])],
                "team": teams.get(int(element["team"]), ""),
                "gw": int(gw),
                "deadline": deadline_utc,
                "docs": docs,
            }
        )
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--deadline", type=str, default="", help="ISO deadline UTC")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--with-fpl",
        action="store_true",
        help="also synthesise packets from data/live/bootstrap.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    deadline = args.deadline
    if not deadline:
        # Prefer fixtures/bootstrap event deadline when present.
        bootstrap_path = ROOT / "data" / "live" / "bootstrap.json"
        if bootstrap_path.is_file():
            boot = json.loads(bootstrap_path.read_text(encoding="utf-8"))
            for event in boot.get("events") or []:
                if int(event.get("id") or 0) == int(args.gw):
                    deadline = str(event.get("deadline_time") or "")
                    break
        if not deadline:
            parser.error("--deadline is required when bootstrap has no event")
    bootstrap = None
    if args.with_fpl:
        bootstrap = json.loads((ROOT / "data" / "live" / "bootstrap.json").read_text(encoding="utf-8"))
    packets = load_gameweek_packets(
        int(args.gw),
        deadline_utc=deadline,
        bootstrap=bootstrap,
        include_synthetic_fpl=bool(args.with_fpl),
    )
    if args.list and not args.validate:
        for packet in packets:
            print(packet.packet_id)
        return 0
    print(f"GW{args.gw}: {len(packets)} packets ok before {deadline}")
    for packet in packets:
        print(
            f"  {packet.packet_id} | {packet.source} | {packet.published_at_utc} | "
            f"players={list(packet.player_ids)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
