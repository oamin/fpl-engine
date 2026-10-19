"""Team-news LLM → contextual tags + optional expected lineups.

Default minutes = rolling 3-GW average. The model may:

1. Emit closed **override tags** per player (``injured`` / ``transferred`` /
   ``benched`` / ``50/50`` / ``none``).
2. Emit **expected lineups** by club (``xi`` of 11 ids, optional ``bench``).

There is no ``ask``. Unclear availability → ``none`` → rolling, unless a
lineup places the player. Hard FPL flags always force 0. Availability tags
(``injured`` / ``transferred``) beat lineup; lineup beats soft rotation tags
and the rolling default.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.live.fpl_snapshot import ELEMENT
from src.live.news_packets import rolling_game_minutes, sigma_m_for_regime
from src.live.news_tags import minutes_for_tag

XMI_MAX = 90.0
COLUMNS = ("player_id", "gw", "xmi", "tag", "source")
CONTEXT_TAGS = frozenset({"injured", "transferred", "benched", "50/50", "none"})
# Beat lineups (availability). Soft rotation tags yield to a club lineup.
AVAILABILITY_TAGS = frozenset({"injured", "transferred"})
SOFT_TAGS = frozenset({"benched", "50/50"})
_REFUSED = frozenset(
    {
        "crowd_opening_scores.csv",
        "half_plan_scores.csv",
        "season_climb_ft.csv",
        "live_benchmark_2026.md",
        "live_deadline_gw6.md",
    }
)


class MinutesLLMError(ValueError):
    """The completion cannot become a minutes file."""


@dataclass(frozen=True)
class PlayerCase:
    """One player in the completion set (focus + club squad mates)."""

    player_id: int
    web_name: str
    position: str
    club: str
    status: str
    chance: float | None
    news: str
    last_minutes: float | None
    minutes_history: tuple[float, ...] = ()
    focus: bool = True

    @property
    def must_be_zero(self) -> bool:
        return self.status in {"s", "u"} or self.chance == 0.0

    def history(self) -> list[float]:
        if self.minutes_history:
            return list(self.minutes_history)
        if self.last_minutes is None:
            return []
        return [float(self.last_minutes)]


@dataclass(frozen=True)
class ExpectedLineup:
    """Predicted matchday XI (+ optional bench) for one club."""

    club: str
    xi: tuple[int, ...]
    bench: tuple[int, ...] = ()


@dataclass(frozen=True)
class ParsedCompletion:
    """Normalised LLM payload."""

    players: tuple[dict[str, Any], ...]
    lineups: tuple[ExpectedLineup, ...] = ()


@dataclass(frozen=True)
class MinutesResult:
    """Validation outcome. ``ok`` is the only state that writes a CSV."""

    ok: bool
    errors: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    lineups: tuple[ExpectedLineup, ...] = ()


def required_cases(
    bootstrap: Mapping[str, Any],
    owned: set[int],
    last_minutes: Mapping[int, float],
    minutes_history: Mapping[int, Sequence[float]] | None = None,
    *,
    expand_club_squads: bool = True,
) -> list[PlayerCase]:
    """Focus set (owned + flagged), then full squads for those clubs (lineups)."""
    hist = minutes_history or {}
    names = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    focus_ids: set[int] = set()
    focus_clubs: set[str] = set()
    for element in bootstrap["elements"]:
        pid = int(element["id"])
        status = str(element.get("status") or "")
        chance = _chance(element.get("chance_of_playing_next_round"))
        if pid not in owned and status == "a" and (chance is None or chance >= 100.0):
            continue
        focus_ids.add(pid)
        focus_clubs.add(names[int(element["team"])])

    cases: list[PlayerCase] = []
    for element in bootstrap["elements"]:
        pid = int(element["id"])
        club = names[int(element["team"])]
        in_focus = pid in focus_ids
        if not in_focus and not (expand_club_squads and club in focus_clubs):
            continue
        status = str(element.get("status") or "")
        chance = _chance(element.get("chance_of_playing_next_round"))
        observed = last_minutes.get(pid)
        series = tuple(float(m) for m in hist.get(pid, ()))
        cases.append(
            PlayerCase(
                player_id=pid,
                web_name=str(element.get("web_name") or pid),
                position=ELEMENT[int(element["element_type"])],
                club=club,
                status=status,
                chance=chance,
                news=str(element.get("news") or "").strip(),
                last_minutes=None if observed is None else float(observed),
                minutes_history=series,
                focus=in_focus,
            )
        )
    cases.sort(key=lambda row: (row.club, not row.focus, row.player_id))
    return cases


def build_prompt(cases: Sequence[PlayerCase], gw: int) -> str:
    """Tags for every listed player; optional expected lineups by club."""
    clubs = sorted({case.club for case in cases})
    lines = [
        f"Tag availability and, when team news supports it, expected lineups for FPL Gameweek {int(gw)}.",
        "Return a JSON object and nothing else:",
        '{"players": [...], "lineups": [...]}',
        "",
        "players — one object per player listed below:",
        '{"player_id": number, "tag": string, "note": string}',
        "tag is one of: injured, transferred, benched, 50/50, none.",
        "injured — hurt, suspended, or doubtful with a meaningful minutes cut.",
        "transferred — left the club / not available for this club.",
        "benched — known rotation / left out when you are NOT emitting a lineup for that club.",
        "50/50 — news says he could return this weekend (play-or-not).",
        "none — no clear availability override; rolling minutes apply unless a lineup places him.",
        "Do not invent a numeric xmi. Do not use ask. Do not use firm_starter.",
        "If status is s or u, or the chance is 0, tag must be injured.",
        "",
        "lineups — optional. Zero or more club predicted XIs from team news:",
        '{"club": string, "xi": [11 player_ids], "bench": [player_ids]}',
        "club must match a club name in the table. xi must list exactly 11 distinct ids from that club.",
        "bench is optional (named substitutes). Players in that club's list but neither xi nor bench",
        "are treated as not in the matchday squad (0 minutes) when a lineup is provided.",
        "Omit lineups (or use []) when the news does not support a predicted XI.",
        "Availability tags injured/transferred override lineup placement.",
        "",
        "A bare JSON array of player objects is also accepted (no lineups).",
        "",
        "player_id | name | pos | club | status | chance | last_minutes | news | focus",
    ]
    for case in cases:
        chance = "" if case.chance is None else f"{case.chance:g}"
        observed = "" if case.last_minutes is None else f"{case.last_minutes:g}"
        news = case.news.replace("|", "/")
        focus = "yes" if case.focus else "squad"
        lines.append(
            f"{case.player_id} | {case.web_name} | {case.position} | {case.club} | "
            f"{case.status} | {chance} | {observed} | {news} | {focus}"
        )
    lines.append("")
    lines.append("Clubs in this prompt: " + ", ".join(clubs))
    return "\n".join(lines) + "\n"


def parse_response(text: str) -> ParsedCompletion:
    """Object ``{players, lineups}`` or a bare players array."""
    body = text.strip()
    if body.startswith("```"):
        fence = body.splitlines()
        if len(fence) < 3 or not fence[-1].strip().startswith("```"):
            raise MinutesLLMError("the completion is not JSON")
        body = "\n".join(fence[1:-1]).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MinutesLLMError("the completion is not valid JSON") from exc

    if isinstance(data, list):
        return ParsedCompletion(players=tuple(data), lineups=())
    if not isinstance(data, dict):
        raise MinutesLLMError("the completion must be a JSON object or array")

    players = data.get("players")
    if players is None and "player_id" in data:
        raise MinutesLLMError("wrap player rows in a players array")
    if not isinstance(players, list):
        raise MinutesLLMError("players must be a JSON array")

    lineups_raw = data.get("lineups") or []
    if not isinstance(lineups_raw, list):
        raise MinutesLLMError("lineups must be a JSON array")
    lineups: list[ExpectedLineup] = []
    for item in lineups_raw:
        if not isinstance(item, dict):
            raise MinutesLLMError("a lineup row must be an object")
        club = str(item.get("club") or "").strip()
        if not club:
            raise MinutesLLMError("a lineup needs a club name")
        try:
            xi = tuple(int(x) for x in (item.get("xi") or []))
        except (TypeError, ValueError) as exc:
            raise MinutesLLMError(f"lineup xi for {club} is not a list of ids") from exc
        bench_raw = item.get("bench") or []
        try:
            bench = tuple(int(x) for x in bench_raw)
        except (TypeError, ValueError) as exc:
            raise MinutesLLMError(f"lineup bench for {club} is not a list of ids") from exc
        lineups.append(ExpectedLineup(club=club, xi=xi, bench=bench))
    return ParsedCompletion(players=tuple(players), lineups=tuple(lineups))


def _lineup_index(
    lineups: Sequence[ExpectedLineup],
    cases: Sequence[PlayerCase],
) -> tuple[dict[str, ExpectedLineup], list[str]]:
    """Map club → lineup; collect structural errors."""
    by_club_ids: dict[str, set[int]] = {}
    for case in cases:
        by_club_ids.setdefault(case.club, set()).add(case.player_id)
    errors: list[str] = []
    out: dict[str, ExpectedLineup] = {}
    for lu in lineups:
        if lu.club not in by_club_ids:
            errors.append(f"lineup club {lu.club!r} is not in the prompt")
            continue
        if lu.club in out:
            errors.append(f"duplicate lineup for {lu.club}")
            continue
        if len(lu.xi) != 11:
            errors.append(f"lineup for {lu.club} needs 11 xi ids, got {len(lu.xi)}")
            continue
        if len(set(lu.xi)) != 11:
            errors.append(f"lineup for {lu.club} has duplicate xi ids")
            continue
        unknown = [pid for pid in lu.xi if pid not in by_club_ids[lu.club]]
        if unknown:
            errors.append(f"lineup for {lu.club} xi ids not in club squad: {unknown}")
            continue
        bench_bad = [pid for pid in lu.bench if pid not in by_club_ids[lu.club]]
        if bench_bad:
            errors.append(f"lineup for {lu.club} bench ids not in club squad: {bench_bad}")
            continue
        overlap = set(lu.xi) & set(lu.bench)
        if overlap:
            errors.append(f"lineup for {lu.club} has ids in both xi and bench: {sorted(overlap)}")
            continue
        gkp = [c for c in cases if c.player_id in lu.xi and c.position == "GKP"]
        if len(gkp) != 1:
            errors.append(f"lineup for {lu.club} must include exactly one GKP in xi, got {len(gkp)}")
            continue
        out[lu.club] = lu
    return out, errors


def _resolve_xmi(
    case: PlayerCase,
    tag: str,
    *,
    lineup_role: str | None,
) -> tuple[float, str, str]:
    """Return (xmi, decision_tag, source)."""
    series = case.history()
    rolling = rolling_game_minutes(series)
    prior = rolling if rolling is not None else case.last_minutes

    if case.must_be_zero:
        return 0.0, "injured", "override:hard_flag"

    if tag in AVAILABILITY_TAGS:
        xmi = minutes_for_tag(tag, case.position, prior, case.chance, case.status)
        if xmi is None:
            xmi = 0.0
        return float(xmi), tag, f"override:{tag}"

    if lineup_role == "xi":
        # Starter floor (≥65), not depressed rolling after injury blanks.
        xmi = minutes_for_tag("firm_starter", case.position, prior, case.chance, case.status)
        return float(xmi if xmi is not None else 75.0), "lineup_xi", "lineup:xi"
    if lineup_role == "bench":
        xmi = minutes_for_tag("benched", case.position, prior, case.chance, case.status)
        return float(xmi if xmi is not None else 0.0), "lineup_bench", "lineup:bench"
    if lineup_role == "out":
        return 0.0, "lineup_out", "lineup:out"

    if tag in SOFT_TAGS:
        if tag == "50/50":
            base = 90.0 if prior is None else float(prior)
            return 0.5 * base, "50/50", "override:50/50"
        xmi = minutes_for_tag(tag, case.position, prior, case.chance, case.status)
        if xmi is None:
            xmi = rolling if rolling is not None else 0.0
        return float(xmi), tag, f"override:{tag}"

    if rolling is not None:
        return float(rolling), "rolling avg", "rolling"
    if prior is not None:
        return float(prior), "rolling avg", "rolling"
    return 0.0, "rolling avg", "rolling"


def validate(cases: Sequence[PlayerCase], parsed: ParsedCompletion) -> MinutesResult:
    """Accept tags (+ optional lineups); compile each row to xmi."""
    by_id = {case.player_id: case for case in cases}
    errors: list[str] = []
    lineup_by_club, lineup_errors = _lineup_index(parsed.lineups, cases)
    errors.extend(lineup_errors)

    seen: dict[int, int] = {}
    tags: dict[int, tuple[str, str]] = {}  # pid -> (tag, note)
    for item in parsed.players:
        if not isinstance(item, dict) or "player_id" not in item:
            errors.append("a player row has no player_id")
            continue
        try:
            pid = int(item["player_id"])
        except (TypeError, ValueError):
            errors.append("a player row has a player_id that is not an integer")
            continue
        seen[pid] = seen.get(pid, 0) + 1
        if pid not in by_id:
            errors.append(f"{pid} is not in the required set")
            continue
        raw_tag = str(item.get("tag") or "").strip()
        if raw_tag in {"", "ask"} or str(item.get("source") or "") == "ask":
            raw_tag = "none"
        if raw_tag not in CONTEXT_TAGS:
            errors.append(f"{pid} has unknown tag {raw_tag or 'blank'}")
            continue
        case = by_id[pid]
        if case.must_be_zero and raw_tag not in {"injured", "none"}:
            raw_tag = "injured"
        tags[pid] = (raw_tag, str(item.get("note") or ""))

    for pid, count in sorted(seen.items()):
        if count > 1:
            errors.append(f"{pid} is listed {count} times")
    missing = [pid for pid in sorted(by_id) if pid not in seen]
    if missing:
        errors.append("missing player_id " + ", ".join(str(pid) for pid in missing))

    if errors:
        return MinutesResult(ok=False, errors=tuple(errors), rows=(), lineups=tuple(lineup_by_club.values()))

    accepted: dict[int, dict[str, Any]] = {}
    for pid, case in by_id.items():
        raw_tag, note = tags[pid]
        lu = lineup_by_club.get(case.club)
        role: str | None = None
        if lu is not None:
            if pid in lu.xi:
                role = "xi"
            elif pid in lu.bench:
                role = "bench"
            else:
                role = "out"
        xmi, decision_tag, source = _resolve_xmi(case, raw_tag, lineup_role=role)
        if case.must_be_zero and xmi > 0.0:
            return MinutesResult(
                ok=False,
                errors=(f"{case.web_name} cannot play and has xmi {xmi:g}",),
                rows=(),
                lineups=tuple(lineup_by_club.values()),
            )
        if xmi < 0.0 or xmi > XMI_MAX:
            return MinutesResult(
                ok=False,
                errors=(f"{pid} has xmi {xmi:g} outside 0 to {XMI_MAX:g}",),
                rows=(),
                lineups=tuple(lineup_by_club.values()),
            )
        sigma_tag = (
            "firm_starter"
            if decision_tag == "lineup_xi"
            else ("benched" if decision_tag == "lineup_bench" else decision_tag)
        )
        accepted[pid] = {
            "player_id": pid,
            "xmi": xmi,
            "tag": decision_tag,
            "source": source,
            "note": note,
            "xmi_sigma": sigma_m_for_regime(
                tag=sigma_tag,
                xmi=xmi,
                minutes=case.history(),
                prior=case.last_minutes,
            ),
        }
    rows = tuple(accepted[pid] for pid in sorted(accepted))
    return MinutesResult(
        ok=True,
        errors=(),
        rows=rows,
        lineups=tuple(lineup_by_club.values()),
    )


def apply_completion(
    text: str,
    cases: Sequence[PlayerCase],
    *,
    gw: int,
    csv_path: Path,
    report_path: Path,
) -> MinutesResult:
    """Validate one completion. Write the CSV only when every required row passes."""
    try:
        parsed = parse_response(text)
        result = validate(cases, parsed)
    except MinutesLLMError as exc:
        result = MinutesResult(ok=False, errors=(str(exc),), rows=())
    if result.ok:
        write_minutes(csv_path, result.rows, gw)
    write_report(report_path, result, gw=gw, n_required=len(cases), csv_path=csv_path)
    return result


def write_minutes(path: Path, rows: Sequence[Mapping[str, Any]], gw: int) -> None:
    """The sheet ``load_xmi`` can read. ``tag`` / ``source`` kept beside xmi."""
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "player_id": int(row["player_id"]),
                    "gw": int(gw),
                    "xmi": _format_xmi(float(row["xmi"])),
                    "tag": str(row.get("tag") or "rolling avg"),
                    "source": str(row.get("source") or "rolling"),
                }
            )


def write_report(
    path: Path,
    result: MinutesResult,
    *,
    gw: int,
    n_required: int,
    csv_path: Path,
) -> None:
    """Errors, lineups, or the rows that were written. The scorer stays unrun."""
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Gameweek {int(gw)} minutes tags",
        "",
        (
            f"Required players: {int(n_required)}. "
            f"The completion {'passed' if result.ok else 'was rejected'}."
        ),
        "",
        "Default minutes = rolling average; tags and expected lineups override context.",
        "The scorer was not run. No chip was chosen.",
        "",
    ]
    if result.lineups:
        lines.append("## Expected lineups")
        lines.append("")
        for lu in result.lineups:
            bench = ",".join(str(x) for x in lu.bench) if lu.bench else "(none)"
            xi = ",".join(str(x) for x in lu.xi)
            lines.append(f"- **{lu.club}** xi=[{xi}] bench=[{bench}]")
        lines.append("")
    if result.ok:
        lines.append(f"Minutes file: `{csv_path}`.")
        lines.append("")
        lines.append("| player_id | xmi | tag | source |")
        lines.append("| --- | --- | --- | --- |")
        for row in result.rows:
            lines.append(
                f"| {int(row['player_id'])} | {_format_xmi(float(row['xmi']))} | "
                f"{row.get('tag')} | {row.get('source')} |"
            )
    else:
        lines.append("No minutes file was written.")
        lines.append("")
        if result.errors:
            lines.append("## Errors")
            lines.append("")
            for error in result.errors:
                lines.append(f"- {error}")
            lines.append("")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _chance(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number


def _format_xmi(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _refuse(path: Path) -> None:
    if path.name in _REFUSED:
        raise MinutesLLMError(f"refusing to write {path.name}")
