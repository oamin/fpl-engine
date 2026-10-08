"""A fresh fifteen whose every player is under 15% owned.

This is not a transfer from the carried squad. The context has no ``score_xp``.
The official string ledger is not written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.live.fpl_snapshot import ELEMENT
from src.live.news_packets import load_packet_files, packets_dir
from src.rules.fpl_2026 import BUDGET_TENTHS
from src.str_agent.extractor import other_outlets_markdown, packets_markdown
from src.str_agent.sources import load_string_sources
from src.str_agent.trial import _fixtures, _minutes_by_gw, last_window
from src.str_agent.validator import validate_draft

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
BOOTSTRAP = ROOT / "data" / "live" / "bootstrap.json"
FIXTURES = ROOT / "data" / "live" / "fixtures.json"
DEADLINE = "2026-10-10T10:00:00Z"
OWN_LIMIT = 15.0


@dataclass(frozen=True)
class DifferentialPack:
    """Context and the directory the validator will use."""

    deadline_utc: str
    gw: int
    directory: dict[str, dict[str, Any]]
    names: dict[str, str]
    ownership: dict[str, float]
    context: str


def _ownership(raw: object) -> float | None:
    from src.live.news_dry_run import parse_ownership

    return parse_ownership(raw)


def prepare_differential(
    *,
    gw: int = 6,
    deadline_utc: str = DEADLINE,
    own_limit: float = OWN_LIMIT,
    logs_path: Path | None = None,
    bootstrap_path: Path | None = None,
    fixtures_path: Path | None = None,
) -> DifferentialPack:
    """List only players whose bootstrap ownership is strictly under the limit."""
    bootstrap = json.loads((bootstrap_path or BOOTSTRAP).read_text(encoding="utf-8"))
    fixtures = json.loads((fixtures_path or FIXTURES).read_text(encoding="utf-8"))
    from src.live.news_tags import prior_minutes

    logs = pd.read_csv(logs_path or LOGS)
    history = _minutes_by_gw(logs)
    teams = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    short = {int(row["id"]): str(row["short_name"]) for row in bootstrap["teams"]}
    fixture_by_team = _fixtures(gw, fixtures, teams)
    directory: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    ownership: dict[str, float] = {}
    lines: list[str] = []
    dossier_ids: list[int] = []
    dossier_directory: dict[int, dict[str, Any]] = {}
    minutes: dict[int, list[float]] = {}
    for element in bootstrap["elements"]:
        percent = _ownership(element.get("selected_by_percent"))
        if percent is None or percent >= float(own_limit):
            continue
        pid = int(element["id"])
        position = ELEMENT[int(element["element_type"])]
        club = teams.get(int(element["team"]), "")
        name = str(element["web_name"])
        window = last_window(history.get(pid, []), gw - 1)
        news = str(element.get("news") or "").replace("\n", " ").strip()
        if len(news) > 120:
            news = news[:117] + "..."
        chance = element.get("chance_of_playing_next_round")
        chance_text = "" if chance is None else str(chance)
        fixture = fixture_by_team.get(int(element["team"]), "no game")
        window_text = ", ".join(str(int(value)) for value in window) or "none"
        directory[str(pid)] = {
            "position": position,
            "club": club,
            "name": name,
            "now_cost": int(element["now_cost"]),
        }
        names[str(pid)] = name
        ownership[str(pid)] = float(percent)
        lines.append(
            f"| {pid} | {name} | {position} | {short.get(int(element['team']), club)} | "
            f"{int(element['now_cost'])} | {percent:.1f} | {element.get('status') or ''} | "
            f"{chance_text} | {window_text} | {fixture} | {news} |"
        )
        series = [pair[1] for pair in history.get(pid, []) if pair[0] < gw]
        minutes[pid] = window
        dossier_directory[pid] = {
            "name": name,
            "position": position,
            "club": club,
            "prior": prior_minutes(series) if series else None,
            "status": element.get("status"),
            "chance": chance,
            "news": str(element.get("news") or ""),
        }
    packets = load_packet_files(packets_dir(gw), deadline_utc=deadline_utc)
    named = {int(pid) for packet in packets for pid in packet.player_ids}
    sources = load_string_sources(gw, deadline_utc=deadline_utc)
    for note in sources:
        named.update(int(pid) for pid in note.player_ids)
    dossier_ids = sorted(pid for pid in named if str(pid) in directory)
    lines.sort()
    counts = {pos: 0 for pos in ("GKP", "DEF", "MID", "FWD")}
    for meta in directory.values():
        counts[str(meta["position"])] = counts.get(str(meta["position"]), 0) + 1
    context = (
        f"# Gameweek {gw} differential string context\n"
        f"Deadline: {deadline_utc}\n\n"
        "Build a fresh fifteen. Do not start from a current squad. "
        f"Every player must have selected_by_percent strictly under {own_limit:g}. "
        "A player with no ownership figure is not allowed. "
        f"The sum of now_cost must be at most {int(BUDGET_TENTHS)} tenths. "
        "Quota is 2 GKP, 5 DEF, 5 MID, 3 FWD, and at most 3 from one club. "
        "Play no chip. Leave transfers_in and transfers_out as empty lists. "
        "Captain and vice must be two different players in the starting eleven. "
        "Use the numeric id from the table.\n\n"
        f"Listed players: {len(directory)} "
        f"(GKP {counts.get('GKP', 0)}, DEF {counts.get('DEF', 0)}, "
        f"MID {counts.get('MID', 0)}, FWD {counts.get('FWD', 0)}).\n\n"
        "| id | name | pos | club | cost | own% | status | chance | last minutes | fixture | news |\n"
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |\n"
        + "\n".join(lines)
        + "\n\n## Quotes for listed players who have a filed note\n"
        + packets_markdown(dossier_ids, dossier_directory, packets, minutes, sources)
        + "\n"
        + other_outlets_markdown(sources, dossier_ids)
    )
    if "score_xp" in context or "ep_next" in context:
        raise RuntimeError("differential context leaked a score column")
    return DifferentialPack(
        deadline_utc=deadline_utc,
        gw=int(gw),
        directory=directory,
        names=names,
        ownership=ownership,
        context=context,
    )


def judge_differential(
    pack: DifferentialPack,
    decision: Mapping[str, Any],
    *,
    own_limit: float = OWN_LIMIT,
) -> dict[str, Any]:
    """Legal fresh fifteen, every id strictly under the ownership limit."""
    errors = validate_draft(
        {"decision": decision},
        directory=pack.directory,
        budget=int(BUDGET_TENTHS),
    )
    if decision.get("chip_played") not in {None, ""}:
        errors.append("this dry run does not play a chip")
    if list(decision.get("transfers_in") or []) or list(decision.get("transfers_out") or []):
        errors.append("fresh squad: leave transfers empty")
    for pid in [str(row) for row in decision.get("squad_15") or []]:
        percent = pack.ownership.get(pid)
        if percent is None:
            errors.append(f"{pid} has no ownership under {own_limit:g}%")
        elif float(percent) >= float(own_limit):
            errors.append(f"{pid} is owned by {percent:g}%, at or above {own_limit:g}")
    squad = [str(pid) for pid in decision.get("squad_15") or []]
    cost = sum(int(pack.directory[pid]["now_cost"]) for pid in squad if pid in pack.directory)

    def _name(pid: object) -> str:
        return pack.names.get(str(pid), str(pid))

    return {
        "gw": pack.gw,
        "is_legal": not errors,
        "errors": errors,
        "total_cost": cost,
        "bank_after": int(BUDGET_TENTHS) - cost,
        "decision": decision,
        "named": {
            "squad_15": [_name(pid) for pid in squad],
            "starting_11": [_name(pid) for pid in decision.get("starting_11") or []],
            "bench_order": [_name(pid) for pid in decision.get("bench_order") or []],
            "captain": _name(decision.get("captain")),
            "vice_captain": _name(decision.get("vice_captain")),
            "chip_played": decision.get("chip_played"),
        },
    }
