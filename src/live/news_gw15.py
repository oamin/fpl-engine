"""Apply the live news tags to Gameweeks 1–5.

FPL lines are used when their own timestamp is before the deadline.
Prose is classified only from documents dated before that deadline.
The published climb is not run.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.live.fpl_snapshot import ELEMENT
from src.live.news_tags import (
    PacketDoc,
    TagDecision,
    NewsTagError,
    accept_tag,
    bulk_seconds,
    build_tag_prompt,
    docs_before,
    fpl_record,
    minutes_for_tag,
    parse_tag_response,
    prior_minutes,
    tag_from_fpl,
)

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / "data" / "live" / "bootstrap.json"
DOCS = ROOT / "data" / "live" / "news_docs_gw15.json"
HISTORY = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
SQUAD = ROOT / "data" / "processed" / "own_squad_gw15.csv"
OUT_CSV = ROOT / "data" / "processed" / "news_tags_gw15.csv"
OUT_REPORT = ROOT / "reports" / "news_tags_gw15.md"
GWS = (1, 2, 3, 4, 5)


def run(completion_text: str | None = None) -> dict[str, Any]:
    """Write the Gameweek 1–5 tag sheet. A missing completion leaves the prose rows open."""
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    documents = json.loads(DOCS.read_text(encoding="utf-8"))
    deadlines = {
        int(event["id"]): str(event["deadline_time"])
        for event in bootstrap["events"]
        if int(event["id"]) in GWS
    }
    bulk = bulk_seconds(bootstrap["elements"])
    history = _history()
    squad = _squad()
    elements = {int(element["id"]): element for element in bootstrap["elements"]}
    teams = {int(team["id"]): str(team["short_name"]) for team in bootstrap["teams"]}
    fpl_rows = _fpl_rows(elements, teams, deadlines, bulk, history)
    packets = _packets(elements, teams, documents, deadlines, bulk, history)
    prompt = build_tag_prompt(packets)
    llm_rows, llm_error = _llm_rows(packets, completion_text)
    merged = _merge(fpl_rows, llm_rows, packets, squad, history)
    _write_csv(merged)
    _write_report(
        merged,
        deadlines=deadlines,
        bulk=bulk,
        elements=elements,
        prompt=prompt,
        llm_error=llm_error,
        classified=completion_text is not None and llm_error is None,
    )
    return {
        "rows": merged,
        "prompt": prompt,
        "llm_error": llm_error,
        "deadlines": deadlines,
    }


def _fpl_rows(
    elements: Mapping[int, Mapping[str, Any]],
    teams: Mapping[int, str],
    deadlines: Mapping[int, str],
    bulk: frozenset[str],
    history: Mapping[int, list[tuple[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gw, deadline in deadlines.items():
        for player_id, element in elements.items():
            record = fpl_record(element, deadline, bulk)
            if record is None:
                continue
            tag = tag_from_fpl(record)
            prior = prior_minutes([minutes for week, minutes in history[player_id] if week < gw])
            xmi = minutes_for_tag(tag, _position(element), prior, record["chance"], record["status"])
            rows.append(
                {
                    "gw": gw,
                    "player_id": player_id,
                    "name": str(element.get("web_name") or player_id),
                    "position": _position(element),
                    "team": teams[int(element["team"])],
                    "tag": tag,
                    "xmi": xmi,
                    "prior": prior,
                    "source": "fpl",
                    "note": record["news"],
                    "news_added": record["news_added"],
                    "rejected": "",
                }
            )
    return rows


def _packets(
    elements: Mapping[int, Mapping[str, Any]],
    teams: Mapping[int, str],
    documents: Sequence[Mapping[str, Any]],
    deadlines: Mapping[int, str],
    bulk: frozenset[str],
    history: Mapping[int, list[tuple[int, float]]],
) -> list[dict[str, Any]]:
    wanted: set[int] = set()
    for doc in documents:
        wanted.update(int(player_id) for player_id in doc["player_ids"])
    packets: list[dict[str, Any]] = []
    for gw in GWS:
        deadline = deadlines[gw]
        visible_docs = docs_before(documents, deadline)
        for player_id in sorted(wanted):
            element = elements[player_id]
            prose = [
                PacketDoc(str(doc["doc_id"]), str(doc["text"]))
                for doc in visible_docs
                if player_id in {int(item) for item in doc["player_ids"]}
            ]
            record = fpl_record(element, deadline, bulk)
            docs = list(prose)
            status = None
            chance = None
            if record is not None:
                status = str(record["status"])
                chance = record["chance"]
                chance_text = "blank" if chance is None else f"{float(chance):g}"
                docs.append(
                    PacketDoc(
                        f"fpl-{player_id}",
                        f"status {status}. chance {chance_text}. {record['news']}",
                        about_player=True,
                    )
                )
            if not docs:
                continue
            packets.append(
                {
                    "player_id": player_id,
                    "name": str(element.get("web_name") or player_id),
                    "position": _position(element),
                    "team": teams[int(element["team"])],
                    "gw": gw,
                    "deadline": deadline,
                    "docs": docs,
                    "status": status,
                    "chance": chance,
                    "prior": prior_minutes([minutes for week, minutes in history[player_id] if week < gw]),
                }
            )
    return packets


def _llm_rows(
    packets: Sequence[Mapping[str, Any]],
    completion_text: str | None,
) -> tuple[dict[tuple[int, int], dict[str, Any]], str | None]:
    if completion_text is None:
        return {}, None
    try:
        parsed = parse_tag_response(completion_text)
    except NewsTagError as exc:
        return {}, str(exc)
    proposals: dict[tuple[int, int], Mapping[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            return {}, "a completion row is not an object"
        try:
            key = (int(item["player_id"]), int(item["gw"]))
        except (KeyError, TypeError, ValueError):
            return {}, "a completion row has no player_id and gw"
        if key in proposals:
            return {}, f"{key[0]} gw {key[1]} is listed twice"
        proposals[key] = item
    decisions: dict[tuple[int, int], dict[str, Any]] = {}
    for packet in packets:
        key = (int(packet["player_id"]), int(packet["gw"]))
        proposal = proposals.get(key)
        identity = _identity(packet)
        if proposal is None:
            decisions[key] = identity | {
                "decision": TagDecision(
                    key[0], key[1], "ask", None, "llm", "missing from the completion", packet["prior"], "missing"
                )
            }
            continue
        try:
            decision = accept_tag(
                proposal,
                packet["docs"],
                position=str(packet["position"]),
                prior=packet["prior"],
                status=packet["status"],
                chance=packet["chance"],
                name=str(packet["name"]),
            )
        except NewsTagError as exc:
            decision = TagDecision(
                key[0],
                key[1],
                "ask",
                None,
                "llm",
                str(proposal.get("note") or ""),
                packet["prior"],
                str(exc),
            )
        decisions[key] = identity | {"decision": decision}
    return decisions, None


def _merge(
    fpl_rows: Sequence[Mapping[str, Any]],
    llm_rows: Mapping[tuple[int, int], Mapping[str, Any]],
    packets: Sequence[Mapping[str, Any]],
    squad: Mapping[tuple[int, int], Mapping[str, Any]],
    history: Mapping[int, list[tuple[int, float]]],
) -> list[dict[str, Any]]:
    """FPL wins when it has a dated line. Prose fills only a week FPL does not cover."""
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in fpl_rows:
        key = (int(row["gw"]), int(row["player_id"]))
        item = dict(row)
        item["in_squad"] = key in squad
        item["played"] = _played(history, int(row["player_id"]), int(row["gw"]))
        item["llm_tag"] = ""
        item["rejected"] = row.get("rejected") or ""
        by_key[key] = item
    for key, wrapped in llm_rows.items():
        decision = wrapped["decision"]
        gw, player_id = key
        existing = by_key.get((gw, player_id))
        if existing is not None and existing["tag"] not in {None, "ask"}:
            existing["llm_tag"] = decision.tag or ""
            existing["rejected"] = decision.rejected
            continue
        if existing is not None and decision.tag == "ask":
            existing["llm_tag"] = "ask"
            existing["rejected"] = decision.rejected
            existing["note"] = decision.note or existing["note"]
            continue
        by_key[(gw, player_id)] = _llm_item(gw, player_id, decision, wrapped, squad, history)
    for packet in packets:
        key = (int(packet["gw"]), int(packet["player_id"]))
        if key in by_key:
            _fill_identity(by_key[key], packet)
            continue
        by_key[key] = {
            "gw": key[0],
            "player_id": key[1],
            "name": packet["name"],
            "position": packet["position"],
            "team": packet["team"],
            "tag": None,
            "xmi": None,
            "prior": packet["prior"],
            "source": "pending",
            "note": "prose packet not classified",
            "news_added": "",
            "rejected": "",
            "in_squad": key in squad,
            "played": _played(history, key[1], key[0]),
            "llm_tag": "",
        }
    for (gw, player_id), squad_row in squad.items():
        if (gw, player_id) in by_key:
            by_key[(gw, player_id)]["name"] = squad_row["name"]
            by_key[(gw, player_id)]["position"] = squad_row["position"]
            by_key[(gw, player_id)]["team"] = squad_row["team"]
            by_key[(gw, player_id)]["lineup"] = squad_row["lineup"]
            continue
        by_key[(gw, player_id)] = {
            "gw": gw,
            "player_id": player_id,
            "name": squad_row["name"],
            "position": squad_row["position"],
            "team": squad_row["team"],
            "lineup": squad_row["lineup"],
            "tag": None,
            "xmi": None,
            "prior": prior_minutes([minutes for week, minutes in history[player_id] if week < gw]),
            "source": "none",
            "note": "",
            "news_added": "",
            "rejected": "",
            "in_squad": True,
            "played": _played(history, player_id, gw),
            "llm_tag": "",
        }
    rows = list(by_key.values())
    rows.sort(key=lambda row: (int(row["gw"]), str(row["name"]), int(row["player_id"])))
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "gw",
        "player_id",
        "name",
        "position",
        "team",
        "tag",
        "xmi",
        "prior",
        "source",
        "lineup",
        "played",
        "news_added",
        "note",
        "rejected",
        "llm_tag",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})


def _write_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    deadlines: Mapping[int, str],
    bulk: frozenset[str],
    elements: Mapping[int, Mapping[str, Any]],
    prompt: str,
    llm_error: str | None,
    classified: bool,
) -> None:
    sanchez = [row for row in rows if int(row["player_id"]) == 140 and int(row["gw"]) in GWS]
    martinez = [row for row in rows if int(row["player_id"]) == 28 and int(row["gw"]) in GWS]
    changed = [
        row
        for row in rows
        if row.get("in_squad") and row.get("xmi") is not None and int(row["player_id"]) != 140
    ]
    lines = [
        "# News tags, Gameweeks 1–5",
        "",
        "A tag is used only from a note dated before that gameweek's deadline. "
        "The current FPL status is not treated as the status at an earlier deadline. "
        "Minutes then follow the tag. The published climb was not run.",
        "",
        "A firm starter keeps his old minutes, and the result stays between 65 and 90. "
        "A ruled-out player is 0. A doubtful player keeps the chance times his old minutes. "
        "A transfer is 0 at the old club. A benched goalkeeper is 0. A benched outfielder is 15. "
        "An unclear note leaves the minutes unchanged.",
        "",
        f"Deadlines: {_deadline_line(deadlines)}.",
        "",
        (
            f"{_bulk_phrase(bulk, elements)} "
            "Those rows are not used, because one shared second is not a time a story was published."
        ),
        "",
        "## Sánchez",
        "",
        "Gameweek 2's deadline is Friday 28 Aug 17:30 UTC. "
        "The Athletic and Guardian pieces that day say a deal for Martínez is agreed and he is due to sign. "
        "They do not say Sánchez is dropped for Sunday. "
        "Sky's report that the signing was announced hours before the Brighton match is Sunday 30 Aug, after the deadline. "
        "The Como loan is 1 Sep on the BBC and 2 Sep on the FPL line, both before the Gameweek 3 deadline.",
        "",
        _player_table(sanchez),
        "",
        "## Martínez",
        "",
        "The same dates apply. He is not tagged as the Chelsea starter until a document dated before the deadline says so.",
        "",
        _player_table(martinez),
        "",
        "## The model's squad",
        "",
        "The fifteen are the published climber's own squad. "
        "A row with no pre-deadline note keeps the minutes the score already uses. "
        "Played minutes are the result after the deadline. They are not an input.",
        "",
    ]
    if changed:
        lines.append("Other squad rows whose minutes the tag would change:")
        lines.append("")
        lines.append(_player_table(changed))
        lines.append("")
    else:
        lines.append("No other player in that fifteen has a pre-deadline note that changes his minutes.")
        lines.append("")
    lines.extend(_pool_section(rows))
    lines.extend(
        [
            "## The prose packets",
            "",
            (
                "The prose completion was applied."
                if classified
                else "The prose completion has not been applied. FPL lines above stand on their own timestamps."
            ),
            "",
        ]
    )
    if llm_error:
        lines.append(f"The completion was rejected: {llm_error}")
        lines.append("")
    lines.append("Cases in the prompt:")
    lines.append("")
    for block in prompt.split("CASE ")[1:]:
        first = block.splitlines()[0]
        lines.append(f"- {first}")
    lines.append("")
    lines.append(f"Sheet: `{OUT_CSV}`.")
    lines.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def _pool_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("source") == "fpl":
            grouped[int(row["gw"])].append(row)
    lines = [
        "## Dated FPL lines",
        "",
        "Each count is a player whose own `news_added` is before that deadline and is not the shared stamp.",
        "",
        "| GW | Transferred, 0 | Ruled out, 0 | Doubtful, scaled | Bench, 0 | Bench, 15 | Unclear |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for gw in GWS:
        bucket = grouped.get(gw, [])
        lines.append(
            "| {gw} | {moved} | {out} | {scaled} | {bench0} | {bench15} | {unclear} |".format(
                gw=gw,
                moved=_count(bucket, lambda row: row["tag"] == "transferred"),
                out=_count(bucket, lambda row: row["tag"] == "injured" and row["xmi"] == 0),
                scaled=_count(bucket, lambda row: row["tag"] == "injured" and row["xmi"] not in {None, 0}),
                bench0=_count(bucket, lambda row: row["tag"] == "benched" and row["xmi"] == 0),
                bench15=_count(bucket, lambda row: row["tag"] == "benched" and row["xmi"] == 15),
                unclear=_count(bucket, lambda row: row["tag"] == "ask"),
            )
        )
    lines.append("")
    doubtful = [
        row
        for row in rows
        if row.get("source") == "fpl" and row.get("tag") == "injured" and row.get("xmi") not in {None, 0}
    ]
    if doubtful:
        lines.append(
            "Doubtful players keep a share of their old minutes. They are not zeroed. "
            "A player with no earlier appearance uses 90 as the full match, then the chance."
        )
        lines.append("")
        lines.append("| GW | Player | Chance line | Old minutes | Tagged minutes | Played |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for row in doubtful:
            lines.append(
                f"| {row['gw']} | {row['name']} | {row['note']} | {_old(row['prior'])} | {_num(row['xmi'])} | {_num(row['played'])} |"
            )
        lines.append("")
    return lines


def _player_table(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: int(row["gw"]))
    body = [
        "| GW | Tag | Source | Tagged minutes | Old minutes | Played | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in ordered:
        note = str(row.get("note") or "")
        rejected = str(row.get("rejected") or "")
        if rejected:
            note = f"{note} Rejected: {rejected}.".strip()
        body.append(
            f"| {row['gw']} | {row.get('tag') or 'none'} | {row.get('source')} | "
            f"{_minutes(row.get('xmi'))} | {_num(row.get('prior'))} | {_num(row.get('played'))} | {note} |"
        )
    return "\n".join(body)


def _history() -> dict[int, list[tuple[int, float]]]:
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with HISTORY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["player_id"])].append((int(row["gw"]), float(row["minutes"])))
    for weeks in grouped.values():
        weeks.sort()
    return grouped


def _squad() -> dict[tuple[int, int], dict[str, Any]]:
    squad: dict[tuple[int, int], dict[str, Any]] = {}
    with SQUAD.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            player_id = int(str(row["player_id"]).split(":")[-1])
            squad[(int(row["gw"]), player_id)] = {
                "name": row["name"],
                "position": row["position"],
                "team": row["team"],
                "lineup": row["lineup"],
            }
    return squad


def _identity(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": packet["name"], "position": packet["position"], "team": packet["team"]}


def _fill_identity(row: dict[str, Any], packet: Mapping[str, Any]) -> None:
    for field in ("name", "position", "team"):
        if not row.get(field):
            row[field] = packet[field]


def _llm_item(
    gw: int,
    player_id: int,
    decision: TagDecision,
    wrapped: Mapping[str, Any],
    squad: Mapping[tuple[int, int], Mapping[str, Any]],
    history: Mapping[int, list[tuple[int, float]]],
) -> dict[str, Any]:
    squad_row = squad.get((gw, player_id), {})
    return {
        "gw": gw,
        "player_id": player_id,
        "name": squad_row.get("name") or wrapped["name"],
        "position": squad_row.get("position") or wrapped["position"],
        "team": squad_row.get("team") or wrapped["team"],
        "tag": decision.tag,
        "xmi": decision.xmi,
        "prior": decision.prior,
        "source": "llm",
        "note": decision.note,
        "news_added": "",
        "rejected": decision.rejected,
        "in_squad": (gw, player_id) in squad,
        "played": _played(history, player_id, gw),
        "llm_tag": decision.tag or "",
    }


def _played(history: Mapping[int, list[tuple[int, float]]], player_id: int, gw: int) -> float | None:
    for week, minutes in history.get(player_id, []):
        if week == gw:
            return minutes
    return None


def _position(element: Mapping[str, Any]) -> str:
    return ELEMENT[int(element["element_type"])]


def _deadline_line(deadlines: Mapping[int, str]) -> str:
    return ", ".join(f"GW{gw} {deadlines[gw]}" for gw in GWS)


def _bulk_phrase(bulk: frozenset[str], elements: Mapping[int, Mapping[str, Any]]) -> str:
    if not bulk:
        return "No news timestamp is shared widely enough to drop."
    second = sorted(bulk)[0]
    count = sum(1 for element in elements.values() if str(element.get("news_added") or "")[:19] == second)
    return f"{count} players share the second {second}."


def _count(rows: Sequence[Mapping[str, Any]], test: Any) -> int:
    return sum(1 for row in rows if test(row))


def _num(value: object) -> str:
    if value is None or value == "":
        return ""
    number = float(value)  # type: ignore[arg-type]
    if number == int(number):
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _old(value: object) -> str:
    if value is None or value == "":
        return "90, no appearance"
    return _num(value)


def _minutes(value: object) -> str:
    if value is None or value == "":
        return "unchanged"
    return _num(value)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Apply live news tags to Gameweeks 1–5.")
    parser.add_argument("--completion", type=Path, help="JSON array from the prose classifier")
    parser.add_argument("--prompt-only", action="store_true", help="Print the prose prompt and skip the report")
    args = parser.parse_args()
    text = args.completion.read_text(encoding="utf-8") if args.completion else None
    result = run(text)
    if args.prompt_only:
        sys.stdout.write(result["prompt"])
    else:
        print(f"Wrote {OUT_REPORT}")
