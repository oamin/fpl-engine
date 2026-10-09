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
from src.rules.fpl_2026 import ChipWallet
from src.str_agent.carry import CarryState, carry_from_entry, validate_move
from src.str_agent.dossier import club_calendar, deadline_brief
from src.str_agent.extractor import (
    other_outlets_markdown,
    packets_markdown,
    roster_markdown,
)
from src.str_agent.horizon import (
    followup_context,
    load_plan,
    resolve_start,
    validate_horizon,
    write_plan,
)
from src.str_agent.notebook import notebook_section
from src.str_agent.prompt import SYSTEM
from src.str_agent.runner import sha256_text
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


def chips_remaining(played: Mapping[int, str], gw: int) -> list[str]:
    """Chips still available at ``gw`` after the ones already played."""
    wallet = ChipWallet()
    for week in sorted(int(key) for key in played):
        wallet.play(week, str(played[week]))
    return list(wallet.available(int(gw)))


def prepare(
    entry_id: int = 2632584,
    *,
    deadline_utc: str = DEADLINE,
    logs_path: Path | None = None,
    bootstrap_path: Path | None = None,
    fixtures_path: Path | None = None,
    plan_root: Path | None = None,
    next_gw: int | None = None,
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
        + deadline_brief(
            gw=int(carry.gw),
            deadline_utc=deadline_utc,
            bank=int(entry["bank"]),
            ft=int(entry["ft_for_next"]),
            chips_played=dict(carry.chips_played),
        )
        + "\n"
        + club_calendar(fixtures, short, int(carry.gw))
        + notebook_section(int(carry.gw), plan_root)
        + "\n## Gameweek 5 squad\n"
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
    asked = int(next_gw) if next_gw is not None else int(carry.gw)
    start = resolve_start(asked, carry, plan_root)
    pack = TrialPack(
        entry_id=int(entry["entry_id"]),
        team_name=str(entry.get("team_name") or ""),
        deadline_utc=deadline_utc,
        carry=carry,
        directory=directory,
        names=names,
        context=context,
    )
    if start is carry and asked == int(carry.gw):
        return pack
    if int(start.gw) != asked:
        raise ValueError(f"saved plan continues at GW{start.gw}, not GW{asked}")
    return _paper_pack(
        pack,
        start,
        roster=roster,
        elements=elements,
        history=history,
        team_of_name=team_of_name,
        fixtures=fixtures,
        teams=teams,
        short=short,
        bootstrap=bootstrap,
        plan_root=plan_root,
    )


def _paper_pack(
    entry_pack: TrialPack,
    start: CarryState,
    *,
    roster: list[dict[str, Any]],
    elements: Mapping[int, Mapping[str, Any]],
    history: Mapping[int, list[tuple[int, float]]],
    team_of_name: Mapping[str, int],
    fixtures: list[dict[str, Any]],
    teams: Mapping[int, str],
    short: Mapping[int, str],
    bootstrap: Mapping[str, Any],
    plan_root: Path | None,
) -> TrialPack:
    """Context for a later week, from the saved plan rather than the live entry."""
    owned_ids = [int(pid) for pid in start.squad]
    missing = [pid for pid in owned_ids if pid not in elements]
    if missing:
        raise ValueError(f"paper squad is not in the bootstrap: {missing}")
    prices = {int(key): int(value) for key, value in start.purchase_prices.items()}
    missing_prices = [pid for pid in owned_ids if pid not in prices]
    if missing_prices:
        raise ValueError(f"paper squad has no purchase price: {missing_prices}")
    through = int(start.gw) - 1
    minutes = {
        pid: last_window(list(history.get(pid, [])), through) for pid in owned_ids
    }
    owned = set(owned_ids)
    paper_roster = [dict(row) for row in roster]
    for row in paper_roster:
        row["owned"] = int(row["player_id"]) in owned
    paper_roster.sort(key=lambda row: (not row["owned"], row["position"], row["name"]))
    dossier: dict[int, dict[str, Any]] = {}
    for pid in owned_ids:
        element = elements[pid]
        series = [pair[1] for pair in history.get(pid, []) if pair[0] <= through]
        dossier[pid] = {
            "name": entry_pack.names[str(pid)],
            "position": entry_pack.directory[str(pid)]["position"],
            "club": entry_pack.directory[str(pid)]["club"],
            "prior": prior_minutes(series),
            "status": element.get("status"),
            "chance": element.get("chance_of_playing_next_round"),
            "news": str(element.get("news") or ""),
        }
    packets = load_gameweek_packets(
        start.gw,
        deadline_utc=entry_pack.deadline_utc,
        include_synthetic_fpl=False,
    )
    packets.extend(
        synthetic_fpl_packets(
            bootstrap,
            gw=start.gw,
            deadline_utc=entry_pack.deadline_utc,
            owned=owned,
        )
    )
    sources = load_string_sources(start.gw, deadline_utc=entry_pack.deadline_utc)
    fixture_by_team = _fixtures(start.gw, fixtures, teams)
    slot_lines = []
    for pid in owned_ids:
        meta = entry_pack.directory[str(pid)]
        team_id = team_of_name.get(str(meta["club"]))
        fixture = fixture_by_team.get(int(team_id), "no game") if team_id else "no game"
        window = minutes.get(pid) or []
        window_text = ", ".join(str(int(value)) for value in window) or "none"
        slot_lines.append(
            f"- owned {meta['name']} (id {pid}), {meta['position']}, "
            f"{meta['club']}, cost {meta['now_cost']}, "
            f"bought at {prices[pid]}, last minutes {window_text}. Fixture: {fixture}."
        )
    left = chips_remaining(dict(start.chips_played), start.gw)
    chips = ", ".join(left) or "none"
    prior = followup_context(load_plan(int(start.gw) - 1, plan_root))
    context = (
        f"# Gameweek {start.gw} string-agent context\n"
        f"Deadline: {entry_pack.deadline_utc}\n"
        f"Starting squad: the saved plan after Gameweek {start.gw - 1}. "
        f"That is the squad you already own. It is not the live entry.\n\n"
        + prior
        + notebook_section(int(start.gw), plan_root)
        + f"Bank: {int(start.bank)} tenths. Free transfers: {int(start.ft_before)}. "
        f"Chips left: {chips}.\n"
        "A transfer beyond the free-transfer bank costs 4 points. "
        "Selling price is the buy price plus half of any rise, rounded down, "
        "or the current price if it has fallen.\n"
        "Captain and vice must be two different players in the starting eleven. "
        "transfers_in and transfers_out must be exactly the players who join and leave.\n\n"
        + deadline_brief(
            gw=int(start.gw),
            deadline_utc=entry_pack.deadline_utc,
            bank=int(start.bank),
            ft=int(start.ft_before),
            chips_played=dict(start.chips_played),
        )
        + "\n"
        + club_calendar(fixtures, short, int(start.gw))
        + "\n## Owned fifteen\n"
        + "\n".join(slot_lines)
        + "\n\n## Market\n"
        + roster_markdown(
            paper_roster,
            bank=int(start.bank),
            ft=int(start.ft_before),
            chips_left=left,
        )
        + "\n## Your fifteen, with notes\n"
        + packets_markdown(owned_ids, dossier, packets, minutes, sources)
        + "\n"
        + other_outlets_markdown(sources, owned_ids)
    )
    return TrialPack(
        entry_id=entry_pack.entry_id,
        team_name=entry_pack.team_name,
        deadline_utc=entry_pack.deadline_utc,
        carry=start,
        directory=entry_pack.directory,
        names=entry_pack.names,
        context=context,
    )


def save_submitted_plan(
    pack: TrialPack,
    payload: Mapping[str, Any],
    *,
    frozen_at_utc: str,
    model_id: str,
    root: Path | None = None,
) -> tuple[Path, str]:
    """Validate a three-week plan and write it. Does not touch the official ledger."""
    decision = dict(payload["decision"])
    horizon = [dict(week) for week in payload["horizon"]]
    result = validate_horizon(decision, horizon, pack.carry, pack.directory)
    return write_plan(
        gw=int(pack.carry.gw),
        deadline_utc=pack.deadline_utc,
        frozen_at_utc=frozen_at_utc,
        model_id=model_id,
        prompt_sha256=sha256_text(SYSTEM),
        context=pack.context,
        rationale=str(payload.get("rationale") or ""),
        decision=decision,
        horizon=horizon,
        carry=pack.carry,
        result=result,
        root=root,
        notes=str(payload.get("notes") or ""),
        adjustments=str(payload.get("adjustments") or ""),
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
