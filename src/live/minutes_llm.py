"""Minutes sheet from one LLM completion.

The model sees status, chance, news, and last observed minutes. It returns
a number or a question. A question, a missing id, or minutes above 0 for a
player who cannot play means no file is written. This module does not call
the scorer, the half-season plan, or the Odds API.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.live.fpl_snapshot import ELEMENT

XMI_MAX = 90.0
COLUMNS = ("player_id", "gw", "xmi", "source")
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
    """One player the model has to answer."""

    player_id: int
    web_name: str
    position: str
    club: str
    status: str
    chance: float | None
    news: str
    last_minutes: float | None

    @property
    def must_be_zero(self) -> bool:
        """Suspended, unavailable, or a stated chance of 0."""
        return self.status in {"s", "u"} or self.chance == 0.0


@dataclass(frozen=True)
class MinutesResult:
    """Validation outcome. ``ok`` is the only state that writes a CSV."""

    ok: bool
    errors: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    asks: tuple[dict[str, Any], ...]


def required_cases(
    bootstrap: Mapping[str, Any],
    owned: set[int],
    last_minutes: Mapping[int, float],
) -> list[PlayerCase]:
    """Owned players, plus anyone unavailable, doubtful, or below a full chance."""
    names = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    cases: list[PlayerCase] = []
    for element in bootstrap["elements"]:
        pid = int(element["id"])
        status = str(element.get("status") or "")
        chance = _chance(element.get("chance_of_playing_next_round"))
        if pid not in owned and status == "a" and (chance is None or chance >= 100.0):
            continue
        observed = last_minutes.get(pid)
        cases.append(
            PlayerCase(
                player_id=pid,
                web_name=str(element.get("web_name") or pid),
                position=ELEMENT[int(element["element_type"])],
                club=names[int(element["team"])],
                status=status,
                chance=chance,
                news=str(element.get("news") or "").strip(),
                last_minutes=None if observed is None else float(observed),
            )
        )
    cases.sort(key=lambda row: row.player_id)
    return cases


def build_prompt(cases: Sequence[PlayerCase], gw: int) -> str:
    """The single completion prompt. Last observed minutes are context only."""
    lines = [
        f"Assign expected minutes for FPL Gameweek {int(gw)}.",
        "Return a JSON array and nothing else. One object per player listed below.",
        'Each object is {"player_id": number, "xmi": number or null, "source": "llm" or "ask", "note": string}.',
        "xmi is the expected minutes from 0 to 90.",
        "Use source llm when the status, the chance, and the news map to a number.",
        "Use source ask and xmi null when they do not map. Do not guess a number for that player.",
        "If status is s or u, or the chance is 0, xmi is 0 and source is llm.",
        "The last observed minutes are context. They are not the default answer.",
        "Do not copy them onto every row, and do not set every available player to 90.",
        "",
        "player_id | name | pos | club | status | chance | last_minutes | news",
    ]
    for case in cases:
        chance = "" if case.chance is None else f"{case.chance:g}"
        observed = "" if case.last_minutes is None else f"{case.last_minutes:g}"
        news = case.news.replace("|", "/")
        lines.append(
            f"{case.player_id} | {case.web_name} | {case.position} | {case.club} | "
            f"{case.status} | {chance} | {observed} | {news}"
        )
    return "\n".join(lines) + "\n"


def parse_response(text: str) -> list[Any]:
    """A JSON array, or one fenced block whose body is that array."""
    body = text.strip()
    if body.startswith("```"):
        fence = body.splitlines()
        if len(fence) < 3 or not fence[-1].strip().startswith("```"):
            raise MinutesLLMError("the completion is not a JSON array")
        body = "\n".join(fence[1:-1]).strip()
    if not body.startswith("["):
        raise MinutesLLMError("the completion is not a JSON array")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise MinutesLLMError("the completion is not valid JSON") from exc
    if not isinstance(data, list):
        raise MinutesLLMError("the completion is not a JSON array")
    return data


def validate(cases: Sequence[PlayerCase], parsed: Sequence[Any]) -> MinutesResult:
    """Accept a complete llm answer. Any ask or illegal row leaves ``ok`` false."""
    by_id = {case.player_id: case for case in cases}
    errors: list[str] = []
    seen: dict[int, int] = {}
    asks: list[dict[str, Any]] = []
    accepted: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict) or "player_id" not in item:
            errors.append("a row has no player_id")
            continue
        try:
            pid = int(item["player_id"])
        except (TypeError, ValueError):
            errors.append("a row has a player_id that is not an integer")
            continue
        seen[pid] = seen.get(pid, 0) + 1
        if pid not in by_id:
            errors.append(f"{pid} is not in the required set")
            continue
        source = str(item.get("source") or "")
        note = str(item.get("note") or "")
        if source == "ask":
            asks.append({"player_id": pid, "note": note, "name": by_id[pid].web_name})
            continue
        if source != "llm":
            errors.append(f"{pid} has source {source or 'blank'}")
            continue
        xmi = _xmi(item.get("xmi"))
        if xmi is None:
            errors.append(f"{pid} has no numeric xmi")
            continue
        if xmi < 0.0 or xmi > XMI_MAX:
            errors.append(f"{pid} has xmi {xmi:g} outside 0 to {XMI_MAX:g}")
            continue
        case = by_id[pid]
        if case.must_be_zero and xmi > 0.0:
            errors.append(f"{case.web_name} cannot play and has xmi {xmi:g}")
            continue
        accepted[pid] = {"player_id": pid, "xmi": xmi, "source": "llm", "note": note}
    for pid, count in sorted(seen.items()):
        if count > 1:
            errors.append(f"{pid} is listed {count} times")
    missing = [pid for pid in sorted(by_id) if pid not in seen]
    if missing:
        errors.append("missing player_id " + ", ".join(str(pid) for pid in missing))
    if asks:
        names = ", ".join(row["name"] for row in asks)
        errors.append(f"asked instead of a number: {names}")
    rows = tuple(accepted[pid] for pid in sorted(accepted))
    ok = not errors and not asks and len(accepted) == len(by_id)
    return MinutesResult(ok=ok, errors=tuple(errors), rows=rows, asks=tuple(asks))


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
        result = MinutesResult(ok=False, errors=(str(exc),), rows=(), asks=())
    if result.ok:
        write_minutes(csv_path, result.rows, gw)
    write_report(report_path, result, gw=gw, n_required=len(cases), csv_path=csv_path)
    return result


def write_minutes(path: Path, rows: Sequence[Mapping[str, Any]], gw: int) -> None:
    """The sheet ``load_xmi`` can read. Extra ``source`` is kept beside xmi."""
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
                    "source": "llm",
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
    """Questions and errors, or the rows that were written. The scorer stays unrun."""
    _refuse(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Gameweek {int(gw)} minutes call",
        "",
        (
            f"Required players: {int(n_required)}. "
            f"The completion {'passed' if result.ok else 'was rejected'}."
        ),
        "",
        "The scorer was not run. No chip was chosen.",
        "",
    ]
    if result.ok:
        lines.append(f"Minutes file: `{csv_path}`.")
        lines.append("")
        lines.append("| player_id | xmi | source |")
        lines.append("| --- | --- | --- |")
        for row in result.rows:
            lines.append(f"| {int(row['player_id'])} | {_format_xmi(float(row['xmi']))} | llm |")
    else:
        lines.append("No minutes file was written.")
        lines.append("")
        if result.errors:
            lines.append("## Errors")
            lines.append("")
            for error in result.errors:
                lines.append(f"- {error}")
            lines.append("")
        if result.asks:
            lines.append("## Questions")
            lines.append("")
            for ask in result.asks:
                note = ask.get("note") or "no note"
                lines.append(f"- {ask['name']} ({ask['player_id']}): {note}")
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


def _xmi(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _format_xmi(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _refuse(path: Path) -> None:
    if path.name in _REFUSED:
        raise MinutesLLMError(f"refusing to write {path.name}")
