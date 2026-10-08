"""Build a Gameweek choice from the entry's last played fifteen.

The last played week is the starting squad. This module does not call a
model and does not write the official string ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.live.deadline import gameweek_values, reconstruct_purchases
from src.live.entry import load_entry
from src.live.fpl_snapshot import ELEMENT
from src.live.news_packets import (
    load_gameweek_packets,
    synthetic_fpl_packets,
)
from src.live.news_tags import prior_minutes
from src.str_agent.carry import CarryState, carry_from_entry, validate_move
from src.str_agent.extractor import (
    other_outlets_markdown,
    packets_markdown,
    roster_markdown,
)
from src.str_agent.sources import load_string_sources

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
BOOTSTRAP = ROOT / "data" / "live" / "bootstrap.json"
FIXTURES = ROOT / "data" / "live" / "fixtures.json"
DEADLINE = "2026-10-10T10:00:00Z"


@dataclass(frozen=True)
class TrialPack:
    """Everything a model needs, and the carry the validator will use."""

    entry_id: int
    team_name: str
    deadline_utc: str
    carry: CarryState
    directory: dict[str, dict[str, Any]]
    names: dict[str, str]
    context: str


def _minutes_by_gw(logs: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    frame = logs.copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["gw"] = pd.to_numeric(frame["gw"], errors="coerce")
    frame["minutes"] = pd.to_numeric(frame["minutes"], errors="coerce").fillna(0.0)
    frame = frame.dropna(subset=["player_id", "gw"])
    grouped = frame.groupby(["player_id", "gw"], as_index=False)["minutes"].sum()
    out: dict[int, list[tuple[int, float]]] = {}
    for row in grouped.itertuples(index=False):
        out.setdefault(int(row.player_id), []).append((int(row.gw), float(row.minutes)))
    for pid in out:
        out[pid].sort()
    return out


def last_window(
    series: list[tuple[int, float]], through_gw: int, window: int = 3
) -> list[float]:
    """Minutes in the last ``window`` stored gameweeks up to ``through_gw``."""
    played = [minutes for gw, minutes in series if gw <= int(through_gw)]
    return played[-window:]


def _fixtures(gw: int, fixtures: list[dict[str, Any]], teams: Mapping[int, str]) -> dict[int, str]:
    lines: dict[int, list[str]] = {}
    for row in fixtures:
        if int(row.get("event") or 0) != int(gw):
            continue
        home = int(row["team_h"])
        away = int(row["team_a"])
        lines.setdefault(home, []).append(f"home to {teams.get(away, away)}")
        lines.setdefault(away, []).append(f"away at {teams.get(home, home)}")
    return {
        team: "; ".join(parts) if parts else "no game"
        for team, parts in lines.items()
    }


def prepare(
    entry_id: int = 2632584,
    *,
    deadline_utc: str = DEADLINE,
    logs_path: Path | None = None,
    bootstrap_path: Path | None = None,
    fixtures_path: Path | None = None,
) -> TrialPack:
    """Gameweek context whose starting fifteen is the entry's last week."""
    entry = load_entry(int(entry_id))
    carry = carry_from_entry(entry)
    logs = pd.read_csv(logs_path or LOGS)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    owned_ids = [int(pid) for pid in carry.squad]
    if set(purchases) != set(owned_ids):
        raise ValueError("purchase reconstruction does not match the Gameweek 5 squad")
    history = _minutes_by_gw(logs)
    through = int(entry["played_through"])
    minutes = {
        pid: last_window(history.get(pid, []), through)
        for pid in owned_ids
    }
    bootstrap = json.loads((bootstrap_path or BOOTSTRAP).read_text(encoding="utf-8"))
    fixtures = json.loads((fixtures_path or FIXTURES).read_text(encoding="utf-8"))
    teams = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    short = {int(row["id"]): str(row["short_name"]) for row in bootstrap["teams"]}
    team_of_name = {str(row["name"]): int(row["id"]) for row in bootstrap["teams"]}
    elements = {int(row["id"]): row for row in bootstrap["elements"]}
    directory: dict[str, dict[str, Any]] = {}
    dossier_directory: dict[int, dict[str, Any]] = {}
    names: dict[str, str] = {}
    roster: list[dict[str, Any]] = []
    owned = set(owned_ids)
    for element in bootstrap["elements"]:
        pid = int(element["id"])
        position = ELEMENT[int(element["element_type"])]
        club = teams.get(int(element["team"]), "")
        name = str(element["web_name"])
        meta = {
            "position": position,
            "club": club,
            "name": name,
            "now_cost": int(element["now_cost"]),
        }
        directory[str(pid)] = meta
        names[str(pid)] = name
        roster.append(
            {
                "player_id": pid,
                "name": name,
                "position": position,
                "club": short.get(int(element["team"]), club),
                "now_cost": int(element["now_cost"]),
                "owned": pid in owned,
            }
        )
    roster.sort(key=lambda row: (not row["owned"], row["position"], row["name"]))
    for pid in owned_ids:
        element = elements[pid]
        series = [pair[1] for pair in history.get(pid, []) if pair[0] <= through]
        dossier_directory[pid] = {
            "name": names[str(pid)],
            "position": directory[str(pid)]["position"],
            "club": directory[str(pid)]["club"],
            "prior": prior_minutes(series),
            "status": element.get("status"),
            "chance": element.get("chance_of_playing_next_round"),
            "news": str(element.get("news") or ""),
        }
    packets = load_gameweek_packets(
        carry.gw,
        deadline_utc=deadline_utc,
        include_synthetic_fpl=False,
    )
    packets.extend(
        synthetic_fpl_packets(
            bootstrap,
            gw=carry.gw,
            deadline_utc=deadline_utc,
            owned=owned,
        )
    )
    sources = load_string_sources(carry.gw, deadline_utc=deadline_utc)
    fixture_by_team = _fixtures(carry.gw, fixtures, teams)
    week = next(row for row in entry["gameweeks"] if int(row["gw"]) == through)
    slot_lines = []
    for group, label in ((week["xi"], "XI"), (week["bench"], "bench")):
        for player in group:
            pid = int(player["id"])
            team_id = team_of_name.get(str(player["team"]))
            # Entry stores the short name in ``team``. Fall back to the bootstrap club.
            if team_id is None:
                club_name = directory[str(pid)]["club"]
                team_id = team_of_name.get(club_name)
            fixture = fixture_by_team.get(int(team_id), "no game") if team_id else "no game"
            window = minutes.get(pid) or []
            window_text = ", ".join(str(int(value)) for value in window) or "none"
            sell = purchases[pid]
            slot_lines.append(
                f"- {label} {player['name']} (id {pid}), {player['position']}, "
                f"{player['team']}, cost {directory[str(pid)]['now_cost']}, "
                f"bought at {sell}, last minutes {window_text}. Fixture: {fixture}."
            )
    chips = ", ".join(entry.get("chips_left") or []) or "none"
    context = (
        f"# Gameweek {carry.gw} string-agent context\n"
        f"Deadline: {deadline_utc}\n"
        f"Starting squad: the Gameweek {through} fifteen below. "
        f"That is the squad you already own.\n\n"
        f"Bank: {int(entry['bank'])} tenths. Free transfers: {int(entry['ft_for_next'])}. "
        f"Chips left: {chips}.\n"
        "A transfer beyond the free-transfer bank costs 4 points. "
        "Selling price is the buy price plus half of any rise, rounded down, "
        "or the current price if it has fallen.\n"
        "Captain and vice must be two different players in the starting eleven. "
        "transfers_in and transfers_out must be exactly the players who join and leave.\n\n"
        "## Gameweek 5 squad\n"
        + "\n".join(slot_lines)
        + "\n\n## Market\n"
        + roster_markdown(
            roster,
            bank=int(entry["bank"]),
            ft=int(entry["ft_for_next"]),
            chips_left=list(entry.get("chips_left") or []),
        )
        + "\n## Your fifteen, with notes\n"
        + packets_markdown(
            owned_ids,
            dossier_directory,
            packets,
            minutes,
            sources,
        )
        + "\n"
        + other_outlets_markdown(sources, owned_ids)
    )
    priced = {
        str(pid): int(purchases[pid])
        for pid in owned_ids
    }
    carry = CarryState(
        gw=carry.gw,
        squad=carry.squad,
        purchase_prices=priced,
        bank=carry.bank,
        ft_before=carry.ft_before,
        chips_played=dict(carry.chips_played),
        selling_prices={},
    )
    return TrialPack(
        entry_id=int(entry["entry_id"]),
        team_name=str(entry.get("team_name") or ""),
        deadline_utc=deadline_utc,
        carry=carry,
        directory=directory,
        names=names,
        context=context,
    )


def judge(pack: TrialPack, decision: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one model decision against the Gameweek 5 carry. No ledger write."""
    move = validate_move({"decision": decision}, pack.carry, pack.directory)
    def _name(pid: object) -> str:
        return pack.names.get(str(pid), str(pid))

    return {
        "entry_id": pack.entry_id,
        "team_name": pack.team_name,
        "gw": pack.carry.gw,
        "starting_gw": pack.carry.gw - 1,
        "is_legal": not move.errors,
        "errors": move.errors,
        "hits": move.hits,
        "bank_after": move.bank_after,
        "decision": decision,
        "named": {
            "transfers_out": [_name(pid) for pid in decision.get("transfers_out") or []],
            "transfers_in": [_name(pid) for pid in decision.get("transfers_in") or []],
            "starting_11": [_name(pid) for pid in decision.get("starting_11") or []],
            "bench_order": [_name(pid) for pid in decision.get("bench_order") or []],
            "captain": _name(decision.get("captain")),
            "vice_captain": _name(decision.get("vice_captain")),
            "chip_played": decision.get("chip_played"),
        },
        "carry_after": move.carry_after,
    }
