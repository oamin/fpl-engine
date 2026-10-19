"""Fantasy Football Scout public team-news (predicted XIs).

What is available without a members login
----------------------------------------
``https://www.fantasyfootballscout.co.uk/team-news`` (public):

* Predicted starting XI (11 names) per Premier League club
* Out / Doubts (with % when given) / Banned lists
* Short club news blurb + next opponent

There is **no** public API field labelled ``nailed_on`` / ``firm``. Those
words are community shorthand for the predicted XI. Members tools
(projections, Rate My Team) sit behind ``members-v2.fantasyfootballscout.co.uk``
and are not called here.

Mapping used by the selectable pool
-----------------------------------
* ``firm`` / ``nailed_on``: in predicted XI, and **not** listed Out, Banned,
  or Doubts.
* ``firm_selection``: the club's predicted XI minus Out/Banned (doubts may
  still appear here as lower-conviction starters — pool eligibility still
  requires the firm/nailed flag above).

YouTube / RSS mentions of FFScout in the string-agent trawl are titles only;
they do not replace this page.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.teams import norm_team

TEAM_NEWS_URL = "https://www.fantasyfootballscout.co.uk/team-news"

# FFScout data-team-code → FPL-ish club label
CODE_TO_CLUB = {
    "ars": "Arsenal",
    "avl": "Aston Villa",
    "bou": "Bournemouth",
    "bre": "Brentford",
    "bha": "Brighton",
    "che": "Chelsea",
    "cov": "Coventry",
    "cry": "Crystal Palace",
    "eve": "Everton",
    "ful": "Fulham",
    "hul": "Hull",
    "ips": "Ipswich",
    "lee": "Leeds",
    "liv": "Liverpool",
    "mci": "Man City",
    "mun": "Man Utd",
    "new": "Newcastle",
    "nfo": "Nott'm Forest",
    "sun": "Sunderland",
    "tot": "Spurs",
}


@dataclass
class ClubTeamNews:
    code: str
    club: str
    club_norm: str
    opponent: str
    predicted_xi: list[str]
    out: list[str]
    doubts: list[dict[str, str | None]]
    banned: list[str]
    firm: list[str]
    firm_selection: list[str]
    news: str


def fetch_team_news_html(
    *, url: str = TEAM_NEWS_URL, timeout: float = 30.0
) -> str:
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; fpl-engine/1.0; research)"},
    )
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


_TRANSLIT = str.maketrans(
    {
        "ø": "o",
        "Ø": "o",
        "ð": "d",
        "Ð": "d",
        "þ": "th",
        "Þ": "th",
        "æ": "ae",
        "Æ": "ae",
        "ł": "l",
        "Ł": "l",
        "ß": "ss",
        "ı": "i",  # Turkish dotless i (NFKD leaves it)
        "İ": "i",
        "ş": "s",
        "Ş": "s",
        "ğ": "g",
        "Ğ": "g",
        "ç": "c",
        "Ç": "c",
        "ü": "u",
        "Ü": "u",
        "ö": "o",
        "Ö": "o",
        "’": "'",
        "‘": "'",
        "ʼ": "'",
        "`": "'",
        "´": "'",
    }
)


def _norm_player(name: str) -> str:
    """Lowercase, strip accents/apostrophes; keep parenthetical tags."""
    import unicodedata

    text = str(name).translate(_TRANSLIT).lower().replace(".", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("'", "")
    return " ".join(text.split())


def _compact(name: str) -> str:
    return _norm_player(name).replace(" ", "")


def parse_team_news(html: str) -> list[ClubTeamNews]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    best: dict[str, ClubTeamNews] = {}
    for el in soup.select("[data-team-code]"):
        code = str(el.get("data-team-code") or "").strip().lower()
        if code not in CODE_TO_CLUB:
            continue
        lines = [
            ln.strip()
            for ln in el.get_text("\n", strip=True).split("\n")
            if ln.strip()
        ]
        try:
            nm_i = lines.index("Next Match:")
            out_i = lines.index("Out:")
            doubt_i = lines.index("Doubts:")
            ban_i = lines.index("Banned:")
            news_i = lines.index("Latest News:")
        except ValueError:
            continue
        club = CODE_TO_CLUB[code]
        opponent = lines[nm_i + 1] if nm_i + 1 < out_i else ""
        xi = lines[nm_i + 2 : out_i]
        outs = lines[out_i + 1 : doubt_i]
        banned = lines[ban_i + 1 : news_i]
        news = " ".join(lines[news_i + 1 :])
        doubts: list[dict[str, str | None]] = []
        chunk = lines[doubt_i + 1 : ban_i]
        j = 0
        while j < len(chunk):
            name = chunk[j]
            if name.endswith("%"):
                j += 1
                continue
            chance: str | None = None
            if j + 1 < len(chunk) and chunk[j + 1].endswith("%"):
                chance = chunk[j + 1]
                j += 2
            else:
                j += 1
            doubts.append({"name": name, "chance": chance})

        out_set = {_norm_player(n) for n in outs}
        ban_set = {_norm_player(n) for n in banned}
        doubt_set = {_norm_player(d["name"]) for d in doubts}
        firm_selection = [
            n
            for n in xi
            if _norm_player(n) not in out_set and _norm_player(n) not in ban_set
        ]
        firm = [
            n
            for n in firm_selection
            if _norm_player(n) not in doubt_set
        ]
        row = ClubTeamNews(
            code=code,
            club=club,
            club_norm=norm_team(club),
            opponent=opponent,
            predicted_xi=xi,
            out=outs,
            doubts=doubts,
            banned=banned,
            firm=firm,
            firm_selection=firm_selection,
            news=news[:500],
        )
        prev = best.get(code)
        if prev is None or len(row.predicted_xi) >= len(prev.predicted_xi):
            best[code] = row
    return [best[k] for k in sorted(best)]


def pull_team_news(*, out_dir: Path | None = None) -> dict[str, Any]:
    """Fetch + parse; optionally snapshot HTML/JSON under ``out_dir``."""
    html = fetch_team_news_html()
    clubs = parse_team_news(html)
    payload = {
        "source": TEAM_NEWS_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_clubs": len(clubs),
        "availability": {
            "predicted_xi": True,
            "out_doubt_banned": True,
            "nailed_on_label": False,
            "firm_label": False,
            "members_projections": False,
            "mapping": (
                "firm/nailed_on := predicted XI minus Out, Banned, and Doubts; "
                "firm_selection := predicted XI minus Out/Banned"
            ),
        },
        "clubs": [asdict(c) for c in clubs],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ffscout_team_news.html").write_text(html, encoding="utf-8")
        (out_dir / "ffscout_team_news.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def match_ffs_name_to_fpl(
    ffs_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Best-effort name match onto FPL bootstrap elements for one club.

    Handles common FFScout ↔ FPL gaps: Ø/ı transliteration, ``B.Fernandes``-
    style web names, compound surnames (``Borges Fernandes``), parenthetical
    disambiguators (``Murphy (Jacob)``), spaced particles (``Jay da Silva`` /
    ``Dasilva``), and first-name-only FFS labels (``Alisson``).
    """
    key = _norm_player(ffs_name)
    base = re.sub(r"\s*\(.*?\)\s*", " ", key).strip()
    paren = re.search(r"\((.*?)\)", key)
    given_hint = paren.group(1).strip() if paren else None
    keys = {key, base}
    if given_hint:
        keys.add(given_hint)
        if base:
            keys.add(f"{given_hint} {base.split()[0]}".strip())
    key_tokens = [t for t in base.split() if t]
    key_last = key_tokens[-1] if key_tokens else ""
    key_compact = _compact(base)

    scored: list[tuple[int, dict[str, Any]]] = []
    for el in candidates:
        web = _norm_player(el.get("web_name") or "")
        second = _norm_player(el.get("second_name") or "")
        first = _norm_player(el.get("first_name") or "")
        full = f"{first} {second}".strip()
        web_compact = web.replace(" ", "")
        second_compact = second.replace(" ", "")
        full_compact = full.replace(" ", "")
        second_tokens = second.split()
        second_last = second_tokens[-1] if second_tokens else ""
        names = {web, second, full, first, f"{first} {web}".strip()}
        names.discard("")

        rank: int | None = None

        surname_hit = bool(key_last) and (
            key_last == second
            or key_last == second_last
            or key_last in second_tokens
            or second.endswith(key_last)
            or second.startswith(f"{key_last} ")
            or key_last in web_compact
            or web_compact.endswith(key_last)
        )

        # Parenthetical given-name + surname (Jacob Murphy)
        if given_hint and first == given_hint and surname_hit:
            rank = 0
        elif keys & names:
            inter = keys & names
            # "jacob" alone must not match Jacob Ramsey when FFS is Murphy (Jacob)
            given_only = bool(given_hint) and inter <= {given_hint}
            if given_only and not surname_hit:
                rank = None
            elif given_hint and first != given_hint and first not in keys:
                # Bare surname hit on the wrong namesake
                rank = 3 if surname_hit else None
            else:
                rank = 0
        elif first and first in keys and len(first) >= 4 and not given_hint:
            # "Alisson" → first_name only (skip when a surname hint exists)
            rank = 1
        elif key_compact and key_compact in {
            web_compact,
            second_compact,
            full_compact,
            f"{first}{second_compact}",
        }:
            # "jay da silva" ↔ "dasilva"
            rank = 0
        elif len(key_tokens) >= 2 and first and second and key_tokens[0] == first:
            if surname_hit:
                # "bruno fernandes" / "bruno guimaraes" ↔ compound surnames
                rank = 0
            elif key_last == second_last and key_tokens[0][:1] == (first[:1] or ""):
                rank = 1
        elif surname_hit and key_last and len(key_last) > 2:
            # "Pino" ↔ "Pino Santos"; "Kadioglu"; "Chema Andrés" ↔ Andrés Baixauli
            rank = 1 if not given_hint else 2

        if rank is None:
            continue
        # Soft penalty when given hint disagrees with first name
        if given_hint and first and first != given_hint and rank < 3:
            rank = min(rank + 2, 4)
        scored.append((rank, el))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    best_rank = scored[0][0]
    tops = [el for r, el in scored if r == best_rank]
    if len(tops) == 1:
        return tops[0]
    # Tie-break: prefer first-name agreement with given hint / first token
    prefer = given_hint or (key_tokens[0] if key_tokens else "")
    if prefer:
        narrowed = [
            el
            for el in tops
            if _norm_player(el.get("first_name") or "") == prefer
        ]
        if len(narrowed) == 1:
            return narrowed[0]
    return None
