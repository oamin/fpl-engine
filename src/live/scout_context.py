"""Scout context labels from FFScout + FPL news + club-news blurbs.

Combines:

* FFScout buckets (Out / Banned / Doubts / predicted-XI omission)
* FPL bootstrap ``news`` / status / chance (return dates when present)
* Free-text FFScout ``Latest News`` blurbs (return horizons, ban length)

Emits human ``context`` strings such as::

  injured, expected back in 2 weeks
  injured, unknown return
  suspended, expected back in 1 week
  suspected single-week rotation
  sustained rotation

This module does **not** invent minutes numbers beyond the existing tag→xmi
rules; it sets ``return_gws`` / ``force_steps`` pins when a horizon is known.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from src.live.ffscout_team_news import _norm_player, match_ffs_name_to_fpl
from src.teams import norm_team

# Stable machine labels (also used in JSON).
CTX_INJURED_BACK = "injured_expected_back"
CTX_INJURED_UNKNOWN = "injured_unknown_return"
CTX_SUSPENDED = "suspended"
CTX_DOUBT = "doubt"
CTX_ROTATION_SINGLE = "suspected_single_week_rotation"
CTX_ROTATION_SUSTAINED = "sustained_rotation"
CTX_TRANSFERRED = "transferred"

MONTHS = {
    m.lower(): i
    for i, m in enumerate(calendar.month_abbr)
    if m
} | {
    m.lower(): i
    for i, m in enumerate(calendar.month_name)
    if m
}

_EXPECTED_BACK = re.compile(
    r"expected\s+back\s+(\d{1,2})\s+([A-Za-z]+)",
    re.I,
)
_UNKNOWN_RETURN = re.compile(r"unknown\s+return\s+date", re.I)
_UNTIL_GW = re.compile(
    r"(?:until|before|for)\s+Gameweek\s+(\d+)\s*(?:at\s+the\s+earliest)?",
    re.I,
)
_NEXT_N_GW = re.compile(
    r"out\s+of\s+the\s+next\s+(\w+)\s+Gameweeks?",
    re.I,
)
_N_MATCH_BAN = re.compile(
    r"(\w+)[-\s]?match\s+ban",
    re.I,
)
_GAME_K_OF_N = re.compile(
    r"game\s+(\d+)\s+of\s+a\s+(\d+)[-\s]?match\s+ban",
    re.I,
)
_FEW_WEEKS = re.compile(
    r"(?:out\s+for\s+)?a\s+few\s+more\s+weeks|(?:out\s+for\s+)?a\s+few\s+weeks",
    re.I,
)
_LONG_TERM = re.compile(
    r"long[-\s]?term\s+absentee|long[-\s]?term\s+injury|out\s+for\s+the\s+season",
    re.I,
)
_INTERNATIONALS = re.compile(
    r"(?:before|until|ahead of)\s+the\s+([A-Za-z]+)\s+internationals?",
    re.I,
)
_WONT_PLAY_AGAIN = re.compile(
    r"won'?t\s+play\s+again\s+before\s+the\s+([A-Za-z]+)\s+internationals?",
    re.I,
)

_WORD_NUM = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def _word_or_int(token: str) -> int | None:
    t = str(token).strip().lower()
    if t.isdigit():
        return int(t)
    return _WORD_NUM.get(t)


@dataclass(frozen=True)
class ReturnHint:
    """Parsed return horizon relative to ``decision_gw``."""

    return_gws: int | None
    """GWs to pin at zero from step 0 (inclusive count). None = unknown."""

    return_date: str | None = None
    return_gw: int | None = None
    source: str = ""
    snippet: str = ""


@dataclass
class ScoutContextRow:
    """One scout-context annotation for inspect / overlay enrichment."""

    element: int
    name: str
    club: str
    bucket: str
    context: str
    context_detail: str
    tag: str
    m0: float
    chance: float | None = None
    return_gws: int | None = None
    return_date: str | None = None
    force_steps: dict[int, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    expand_m: float | None = None
    recent_minutes: list[float] = field(default_factory=list)
    fpl_news: str = ""
    fpl_status: str = ""
    blurb_snippet: str = ""

    def human_label(self) -> str:
        if self.context == CTX_INJURED_BACK and self.return_gws is not None:
            w = int(self.return_gws)
            if w <= 0:
                return "injured, listed out (return date due / this GW)"
            unit = "week" if w == 1 else "weeks"
            return f"injured, expected back in {w} {unit}"
        if self.context == CTX_INJURED_UNKNOWN:
            return "injured, unknown return"
        if self.context == CTX_SUSPENDED:
            if self.return_gws is not None and int(self.return_gws) > 0:
                w = int(self.return_gws)
                unit = "week" if w == 1 else "weeks"
                return f"suspended, expected back in {w} {unit}"
            return "suspended"
        if self.context == CTX_DOUBT:
            if self.chance is not None:
                return f"doubt ({self.chance:.0f}% chance)"
            return "doubt"
        if self.context == CTX_ROTATION_SINGLE:
            return "suspected single-week rotation"
        if self.context == CTX_ROTATION_SUSTAINED:
            return "sustained rotation"
        if self.context == CTX_TRANSFERRED:
            return "transferred / unavailable"
        return self.context_detail or self.context


def event_deadlines(bootstrap: Mapping[str, Any]) -> dict[int, date]:
    out: dict[int, date] = {}
    for ev in bootstrap.get("events") or []:
        try:
            gid = int(ev["id"])
            raw = str(ev.get("deadline_time") or "")
            if not raw:
                continue
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            dt = datetime.fromisoformat(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            out[gid] = dt.date()
        except (KeyError, TypeError, ValueError):
            continue
    return out


def parse_fpl_return_date(news: str, *, year_hint: int) -> date | None:
    m = _EXPECTED_BACK.search(news or "")
    if not m:
        return None
    day = int(m.group(1))
    mon = MONTHS.get(m.group(2).lower()[:3]) or MONTHS.get(m.group(2).lower())
    if mon is None:
        return None
    try:
        return date(int(year_hint), int(mon), int(day))
    except ValueError:
        return None


def first_gw_on_or_after(when: date, deadlines: Mapping[int, date]) -> int | None:
    for gid in sorted(deadlines):
        if deadlines[gid] >= when:
            return int(gid)
    return None


def return_gws_from_date(
    when: date,
    *,
    decision_gw: int,
    deadlines: Mapping[int, date],
) -> tuple[int, int | None]:
    """GWs to miss from ``decision_gw``, and absolute return GW id."""
    rgw = first_gw_on_or_after(when, deadlines)
    if rgw is None:
        return 0, None
    n = max(0, int(rgw) - int(decision_gw))
    return n, int(rgw)


def _sentences_about(name: str, blurb: str) -> list[str]:
    """Rough sentence / clause filter mentioning a player surname."""
    if not blurb:
        return []
    key = _norm_player(name)
    tokens = [t for t in key.split() if len(t) >= 3]
    if not tokens:
        return []
    # Prefer last token (surname); keep full key for short web names.
    needles = {tokens[-1], key}
    if len(tokens) >= 2:
        needles.add(tokens[0])
    # Split hard stops and common compound joiners so ban clauses for
    # another player do not contaminate an injury clause.
    parts = re.split(
        r"(?<=[.!;])\s+|\n+|\s+while\s+|\s+whereas\s+|\s+,\s+so\s+|"
        r",\s*plus\s+|\s+plus\s+",
        blurb,
        flags=re.I,
    )
    hits: list[str] = []
    for part in parts:
        pnorm = _norm_player(part)
        if any(n in pnorm for n in needles):
            hits.append(part.strip())
    return hits


def _name_near_span(text: str, name: str, start: int, *, window: int = 60) -> bool:
    """True when a surname/token of ``name`` sits near ``start`` in ``text``."""
    key = _norm_player(name)
    tokens = [t for t in key.split() if len(t) >= 3]
    if not tokens:
        return False
    needles = {tokens[-1], key}
    lo = max(0, start - window)
    hi = min(len(text), start + window)
    window_norm = _norm_player(text[lo:hi])
    return any(n in window_norm for n in needles)


def parse_blurb_return(
    text: str,
    *,
    decision_gw: int,
    deadlines: Mapping[int, date],
    player_name: str | None = None,
) -> ReturnHint | None:
    """Extract a return horizon from a blurb snippet (player-scoped or club)."""
    if not text:
        return None
    raw = str(text)

    m = _GAME_K_OF_N.search(raw)
    if m and (player_name is None or _name_near_span(raw, player_name, m.start())):
        k, n = int(m.group(1)), int(m.group(2))
        left = max(0, n - k + 1)  # including current GW of the ban
        return ReturnHint(
            return_gws=left,
            return_gw=int(decision_gw) + left,
            source="blurb_ban_progress",
            snippet=m.group(0),
        )

    m = _N_MATCH_BAN.search(raw)
    if m and (player_name is None or _name_near_span(raw, player_name, m.start())):
        n = _word_or_int(m.group(1))
        if n is not None:
            return ReturnHint(
                return_gws=int(n),
                return_gw=int(decision_gw) + int(n),
                source="blurb_n_match_ban",
                snippet=m.group(0),
            )

    m = _NEXT_N_GW.search(raw)
    if m:
        n = _word_or_int(m.group(1))
        if n is not None:
            return ReturnHint(
                return_gws=int(n),
                return_gw=int(decision_gw) + int(n),
                source="blurb_next_n_gw",
                snippet=m.group(0),
            )

    m = _UNTIL_GW.search(raw)
    if m:
        until = int(m.group(1))
        # "won't return until GW10" → miss decision..until-1
        n = max(0, until - int(decision_gw))
        return ReturnHint(
            return_gws=n,
            return_gw=until,
            source="blurb_until_gw",
            snippet=m.group(0),
        )

    m = _WONT_PLAY_AGAIN.search(raw) or _INTERNATIONALS.search(raw)
    if m:
        # November internationals sit between GW10 and GW11 in 2026/27.
        month = m.group(1).lower()
        if month.startswith("nov"):
            return_gw = 11
        elif month.startswith("oct"):
            return_gw = 9
        elif month.startswith("dec"):
            return_gw = 17
        else:
            return_gw = int(decision_gw) + 3
        n = max(0, return_gw - int(decision_gw))
        return ReturnHint(
            return_gws=n,
            return_gw=return_gw,
            source="blurb_internationals",
            snippet=m.group(0),
        )

    m = _LONG_TERM.search(raw)
    if m and (player_name is None or _name_near_span(raw, player_name, m.start(), window=80)):
        return ReturnHint(
            return_gws=None,
            source="blurb_long_term",
            snippet=m.group(0),
        )

    # Clause is usually already player-scoped by ``_sentences_about``; do not
    # require tight proximity (Wan-Bissaka … few weeks can be >60 chars).
    m = _FEW_WEEKS.search(raw)
    if m and (
        player_name is None
        or _norm_player(player_name).split()[-1] in _norm_player(raw)
        or _name_near_span(raw, player_name, m.start(), window=120)
    ):
        return ReturnHint(
            return_gws=3,
            return_gw=int(decision_gw) + 3,
            source="blurb_few_weeks",
            snippet="a few weeks",
        )

    m = _EXPECTED_BACK.search(raw)
    if m and deadlines:
        year = max(deadlines[g].year for g in deadlines) if deadlines else date.today().year
        when = parse_fpl_return_date(m.group(0), year_hint=year)
        if when is not None:
            n, rgw = return_gws_from_date(
                when, decision_gw=decision_gw, deadlines=deadlines
            )
            return ReturnHint(
                return_gws=n,
                return_date=when.isoformat(),
                return_gw=rgw,
                source="blurb_expected_back_date",
                snippet=m.group(0),
            )

    return None


def parse_fpl_hint(
    element: Mapping[str, Any],
    *,
    decision_gw: int,
    deadlines: Mapping[int, date],
) -> ReturnHint | None:
    news = str(element.get("news") or "").strip()
    status = str(element.get("status") or "")
    if not news and status in {"a", ""}:
        return None
    if _UNKNOWN_RETURN.search(news):
        return ReturnHint(
            return_gws=None,
            source="fpl_unknown_return",
            snippet="Unknown return date",
        )
    year = max((deadlines[g].year for g in deadlines), default=date.today().year)
    when = parse_fpl_return_date(news, year_hint=year)
    if when is not None:
        n, rgw = return_gws_from_date(
            when, decision_gw=decision_gw, deadlines=deadlines
        )
        return ReturnHint(
            return_gws=n,
            return_date=when.isoformat(),
            return_gw=rgw,
            source="fpl_expected_back",
            snippet=_EXPECTED_BACK.search(news).group(0) if _EXPECTED_BACK.search(news) else news[:80],
        )
    # Suspended until date sometimes in news ("Suspended until 21 Nov")
    m = re.search(r"suspended\s+until\s+(\d{1,2})\s+([A-Za-z]+)", news, re.I)
    if m:
        mon = MONTHS.get(m.group(2).lower()[:3]) or MONTHS.get(m.group(2).lower())
        if mon:
            try:
                when = date(year, int(mon), int(m.group(1)))
                n, rgw = return_gws_from_date(
                    when, decision_gw=decision_gw, deadlines=deadlines
                )
                return ReturnHint(
                    return_gws=n,
                    return_date=when.isoformat(),
                    return_gw=rgw,
                    source="fpl_suspended_until",
                    snippet=m.group(0),
                )
            except ValueError:
                pass
    return None


def force_steps_for_out(return_gws: int | None, *, m0: float = 0.0) -> dict[int, float]:
    if return_gws is None or return_gws <= 0:
        return {}
    return {h: float(m0) for h in range(int(return_gws))}


def _recent_minutes(series: Sequence[float], n: int = 3) -> list[float]:
    xs = [float(v) for v in series]
    return xs[-n:] if xs else []


def classify_context(
    *,
    bucket: str,
    tag: str,
    chance: float | None,
    fpl_status: str,
    fpl_news: str,
    hint: ReturnHint | None,
    expand_m: float,
    recent: Sequence[float],
) -> tuple[str, str]:
    """Return (context_code, context_detail)."""
    news_l = (fpl_news or "").lower()
    if "joined" in news_l or "departed" in news_l or "loan" in news_l:
        if fpl_status == "u" or "permanently" in news_l or "loan" in news_l:
            return CTX_TRANSFERRED, "FPL unavailable / transferred"

    if bucket == "banned" or fpl_status == "s" or tag == "injured" and "suspend" in news_l:
        if hint and hint.return_gws is not None:
            return CTX_SUSPENDED, f"suspended; pin {hint.return_gws} GW"
        return CTX_SUSPENDED, "suspended / banned"

    if bucket in {"out", "fpl_out"} or (bucket == "doubt" and chance == 0):
        if hint and hint.return_gws is not None:
            return CTX_INJURED_BACK, f"return_gws={hint.return_gws} via {hint.source}"
        if hint and hint.source == "fpl_unknown_return":
            return CTX_INJURED_UNKNOWN, "FPL unknown return date"
        if fpl_status in {"i", "u"} or bucket == "out":
            return CTX_INJURED_UNKNOWN, "out; no parsed return horizon"
        return CTX_INJURED_UNKNOWN, "out"

    if bucket == "doubt":
        return CTX_DOUBT, f"doubt chance={chance}"

    if bucket in {"xi_omit", "rotation"}:
        low = [m for m in recent if m <= 30.0]
        if expand_m >= 45.0 and len(recent) >= 3 and len(low) >= 3:
            return CTX_ROTATION_SUSTAINED, "regular left out of XI; last 3 GWs ≤30′"
        return CTX_ROTATION_SINGLE, "regular left out of predicted XI (not listed Out)"

    if hint and hint.return_gws is not None:
        return CTX_INJURED_BACK, f"return_gws={hint.return_gws} via {hint.source}"
    return CTX_INJURED_UNKNOWN, bucket or tag


def enrich_overlays_with_context(
    overlays: dict[int, Any],
    *,
    ffs_payload: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    decision_gw: int,
    baseline_m: Mapping[int, float],
    series_by_element: Mapping[int, Sequence[float]],
    include_xi_omissions: bool = True,
    rotation_expand_min: float = 45.0,
) -> tuple[dict[int, Any], list[ScoutContextRow], dict[str, Any]]:
    """Mutate/extend scout overlays with context + optional XI-omit rotations.

    ``overlays`` values are ``ScoutOverlay``-like objects with attributes
    ``element, name, club, tag, m0, reason, chance, force_steps`` and optional
    context fields set here.
    """
    deadlines = event_deadlines(bootstrap)
    els = {int(e["id"]): e for e in bootstrap.get("elements") or []}
    teams = {int(t["id"]): str(t["name"]) for t in bootstrap.get("teams") or []}
    by_club: dict[str, list[dict[str, Any]]] = {}
    for el in els.values():
        club = teams.get(int(el["team"]), "")
        by_club.setdefault(norm_team(club), []).append(el)

    # Club blurbs + predicted XI sets (norm names)
    club_blurbs: dict[str, str] = {}
    club_xi: dict[str, set[str]] = {}
    club_listed: dict[str, set[str]] = {}  # out∪banned∪doubts
    for club in ffs_payload.get("clubs") or []:
        cnorm = str(club.get("club_norm") or norm_team(str(club.get("club") or "")))
        club_blurbs[cnorm] = str(club.get("news") or "")
        xi = {_norm_player(n) for n in (club.get("predicted_xi") or [])}
        club_xi[cnorm] = xi
        listed: set[str] = set()
        for n in club.get("out") or []:
            listed.add(_norm_player(str(n)))
        for n in club.get("banned") or []:
            listed.add(_norm_player(str(n)))
        for d in club.get("doubts") or []:
            name = d.get("name") if isinstance(d, dict) else d
            listed.add(_norm_player(str(name)))
        club_listed[cnorm] = listed

    rows: list[ScoutContextRow] = []
    counts: dict[str, int] = {}

    def _bucket_for_overlay(ov: Any) -> str:
        reason = str(getattr(ov, "reason", "") or "").lower()
        if "banned" in reason:
            return "banned"
        if "doubt" in reason:
            return "doubt"
        if "out" in reason:
            return "out"
        return "overlay"

    for eid, ov in list(overlays.items()):
        el = els.get(int(eid)) or {}
        fpl_news = str(el.get("news") or "").strip()
        fpl_status = str(el.get("status") or "")
        expand = float(baseline_m.get(int(eid), 0.0))
        recent = _recent_minutes(series_by_element.get(int(eid)) or [])
        club_norm = norm_team(str(getattr(ov, "club", "") or ""))
        blurb = club_blurbs.get(club_norm, "")
        about = _sentences_about(str(getattr(ov, "name", "") or ""), blurb)
        # Also try surname from FFScout reason ("FFScout Out: Ruggeri")
        reason_name = str(getattr(ov, "reason", "") or "")
        if ":" in reason_name:
            about = about or _sentences_about(reason_name.split(":", 1)[1].strip(), blurb)
        hint = parse_fpl_hint(el, decision_gw=decision_gw, deadlines=deadlines)
        blurb_hint = None
        snippet = ""
        # Only player-scoped sentences — never the whole club blurb (avoids
        # pinning every Out to "a few weeks" meant for two named players).
        player_for_blurb = str(getattr(ov, "name", "") or "")
        if ":" in reason_name:
            player_for_blurb = reason_name.split(":", 1)[1].strip() or player_for_blurb
        for sent in about:
            blurb_hint = parse_blurb_return(
                sent,
                decision_gw=decision_gw,
                deadlines=deadlines,
                player_name=player_for_blurb,
            )
            if blurb_hint is not None:
                snippet = sent[:220]
                break
        # Prefer explicit numeric horizon; FPL date beats vague blurb if both.
        chosen = hint
        if blurb_hint is not None:
            if chosen is None or chosen.return_gws is None:
                chosen = blurb_hint
            elif (
                blurb_hint.return_gws is not None
                and chosen.source.startswith("fpl")
                and chosen.return_gws is not None
            ):
                # Keep FPL date; retain blurb as evidence only.
                pass
            elif blurb_hint.source.startswith("blurb_ban") or blurb_hint.source.startswith(
                "blurb_next"
            ):
                chosen = blurb_hint

        bucket = _bucket_for_overlay(ov)
        # Past-due FPL return date while still listed Out → unknown horizon.
        if (
            chosen is not None
            and chosen.source.startswith("fpl")
            and chosen.return_gws == 0
            and bucket in {"out", "banned", "overlay"}
        ):
            chosen = ReturnHint(
                return_gws=None,
                return_date=chosen.return_date,
                source="fpl_return_date_due_still_out",
                snippet=chosen.snippet,
            )
        ctx, detail = classify_context(
            bucket=bucket,
            tag=str(getattr(ov, "tag", "")),
            chance=getattr(ov, "chance", None),
            fpl_status=fpl_status,
            fpl_news=fpl_news,
            hint=chosen,
            expand_m=expand,
            recent=recent,
        )
        rgw = chosen.return_gws if chosen else None
        # Doubts: don't hard-pin zeros unless chance is 0.
        if bucket == "doubt" and (getattr(ov, "chance", None) or 50) > 0:
            force: dict[int, float] = {}
        else:
            force = force_steps_for_out(rgw, m0=float(getattr(ov, "m0", 0.0)))
            if force:
                ov.force_steps = {**dict(getattr(ov, "force_steps", {}) or {}), **force}

        label = ScoutContextRow(
            element=int(eid),
            name=str(getattr(ov, "name", "")),
            club=str(getattr(ov, "club", "")),
            bucket=bucket,
            context=ctx,
            context_detail=detail,
            tag=str(getattr(ov, "tag", "")),
            m0=float(getattr(ov, "m0", 0.0)),
            chance=getattr(ov, "chance", None),
            return_gws=rgw,
            return_date=chosen.return_date if chosen else None,
            force_steps=dict(getattr(ov, "force_steps", {}) or {}),
            evidence=[
                x
                for x in [
                    f"ffscout:{getattr(ov, 'reason', '')}",
                    f"fpl:{fpl_news}" if fpl_news else "",
                    f"blurb:{snippet}" if snippet else "",
                    f"hint:{chosen.source}:{chosen.snippet}" if chosen else "",
                ]
                if x
            ],
            expand_m=expand,
            recent_minutes=recent,
            fpl_news=fpl_news,
            fpl_status=fpl_status,
            blurb_snippet=snippet,
        )
        # Attach onto overlay
        ov.context = ctx
        ov.context_label = label.human_label()
        ov.context_detail = detail
        ov.return_gws = rgw
        ov.return_date = chosen.return_date if chosen else None
        ov.evidence = list(label.evidence)
        rows.append(label)
        counts[ctx] = counts.get(ctx, 0) + 1

    # Predicted-XI omissions → rotation contexts (not already overlaid)
    n_rot = 0
    if include_xi_omissions:
        overlaid = set(overlays)
        for club in ffs_payload.get("clubs") or []:
            club_name = str(club.get("club") or "")
            cnorm = str(club.get("club_norm") or norm_team(club_name))
            xi = club_xi.get(cnorm) or set()
            listed = club_listed.get(cnorm) or set()
            cands = by_club.get(cnorm) or []
            for el in cands:
                eid = int(el["id"])
                if eid in overlaid:
                    continue
                expand = float(baseline_m.get(eid, 0.0))
                if expand < float(rotation_expand_min):
                    continue
                web = str(el.get("web_name") or "")
                second = str(el.get("second_name") or "")
                # Must be identifiable on the FFS page: match against XI∪listed fails
                # and player is a regular — treat as left out of predicted XI.
                names = {_norm_player(web), _norm_player(second)}
                first = _norm_player(str(el.get("first_name") or ""))
                if first and second:
                    names.add(_norm_player(f"{first} {second}"))
                names.discard("")
                if names & xi:
                    continue
                if names & listed:
                    continue
                # Also try FFScout-style match against XI names
                xi_hit = False
                for xn in club.get("predicted_xi") or []:
                    if match_ffs_name_to_fpl(str(xn), [el]) is not None:
                        xi_hit = True
                        break
                if xi_hit:
                    continue
                # Skip transferred / unavailable FPL
                st = str(el.get("status") or "")
                news = str(el.get("news") or "")
                if st == "u" or "joined" in news.lower():
                    continue
                recent = _recent_minutes(series_by_element.get(eid) or [])
                # Only flag if they were recently involved OR strong expand regular
                if not recent or max(recent[-3:] or [0.0]) < 45.0:
                    # still allow expand≥60′
                    if expand < 60.0:
                        continue
                ctx, detail = classify_context(
                    bucket="xi_omit",
                    tag="benched",
                    chance=None,
                    fpl_status=st,
                    fpl_news=news,
                    hint=None,
                    expand_m=expand,
                    recent=recent,
                )
                # Soft m0: rotation → benched minutes via prior path left to caller;
                # here we only annotate context rows + optional soft overlay.
                from src.live.news_tags import minutes_for_tag, prior_minutes

                pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(
                    int(el.get("element_type") or 3), "MID"
                )
                prior = prior_minutes(list(series_by_element.get(eid) or []))
                m_val = minutes_for_tag("benched", pos, prior, None, st or "a")
                if m_val is None:
                    m_val = 15.0
                # Local import avoids cycle at module load (scout_minutes → scout_context).
                from src.live.scout_minutes import ScoutOverlay

                ov = ScoutOverlay(
                    element=eid,
                    name=web,
                    club=club_name,
                    tag="benched",
                    m0=float(m_val),
                    reason=f"FFScout XI omit (regular): {web}",
                    source="ffscout_xi_omit",
                    chance=None,
                )
                ov.context = ctx
                ov.context_label = (
                    "sustained rotation"
                    if ctx == CTX_ROTATION_SUSTAINED
                    else "suspected single-week rotation"
                )
                ov.context_detail = detail
                ov.return_gws = 1 if ctx == CTX_ROTATION_SINGLE else None
                ov.return_date = None
                ov.evidence = [
                    f"expand_m={expand:.1f}",
                    f"recent={recent}",
                    "not in predicted XI; not Out/Banned/Doubt",
                ]
                if ctx == CTX_ROTATION_SINGLE:
                    ov.force_steps = {0: float(m_val)}
                overlays[eid] = ov
                rows.append(
                    ScoutContextRow(
                        element=eid,
                        name=web,
                        club=club_name,
                        bucket="xi_omit",
                        context=ctx,
                        context_detail=detail,
                        tag="benched",
                        m0=float(m_val),
                        return_gws=ov.return_gws,
                        force_steps=dict(ov.force_steps),
                        evidence=list(ov.evidence),
                        expand_m=expand,
                        recent_minutes=recent,
                        fpl_news=news,
                        fpl_status=st,
                    )
                )
                counts[ctx] = counts.get(ctx, 0) + 1
                n_rot += 1
                overlaid.add(eid)

    diag = {
        "n_context_rows": len(rows),
        "n_rotation_overlays": n_rot,
        "context_counts": dict(sorted(counts.items())),
        "n_with_return_gws": sum(1 for r in rows if r.return_gws is not None),
        "n_with_blurb_evidence": sum(1 for r in rows if r.blurb_snippet),
        "n_with_fpl_news": sum(1 for r in rows if r.fpl_news),
    }
    return overlays, rows, diag
