"""Deadline, chip bank, and fixture calendar for the string agent.

The validated plan stays three weeks. This block is the rest of the half.
It uses the chip and calendar rules in ``src/rules/fpl_2026`` and does not
copy them. It does not include a score column or a future price.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.rules.fpl_2026 import (
    FIRST_HALF_END_GW,
    N_GAMEWEEKS,
    ChipWallet,
    half_for_gw,
)


def _half_end(gw: int) -> int:
    return FIRST_HALF_END_GW if half_for_gw(gw) == "H1" else N_GAMEWEEKS


def season_header(gw: int, deadline_utc: str) -> str:
    """Current gameweek, half, and when this half's chips expire."""
    week = int(gw)
    half = half_for_gw(week)
    end = _half_end(week)
    lines = [
        f"Gameweek {week} of {N_GAMEWEEKS}. Half {half} (through Gameweek {end}).",
        f"Deadline: {deadline_utc}.",
    ]
    if half == "H1":
        lines.append(
            f"Unused H1 chips expire at the Gameweek {FIRST_HALF_END_GW} deadline. "
            f"They cannot be played from Gameweek {FIRST_HALF_END_GW + 1}."
        )
    else:
        lines.append(
            f"H1 chips expired at the Gameweek {FIRST_HALF_END_GW} deadline. "
            "Only an unused H2 chip can be played."
        )
    return "\n".join(lines)


def deadline_brief(
    *,
    gw: int,
    deadline_utc: str,
    bank: int,
    ft: int,
    chips_played: Mapping[Any, str],
) -> str:
    """Current gameweek, half, expiry, money, and the chip bank."""
    week = int(gw)
    wallet = ChipWallet()
    played_map = {int(key): str(value) for key, value in chips_played.items()}
    played: list[str] = []
    for key in sorted(played_map):
        name = played_map[key]
        wallet.play(key, name)
        played.append(f"GW{key} {name}")
    available = ", ".join(wallet.available(week)) or "none"
    lines = ["## This deadline", season_header(week, deadline_utc)]
    lines.append(f"Bank: {int(bank)} tenths. Free transfers: {int(ft)}.")
    lines.append("Chips already played: " + (", ".join(played) or "none") + ".")
    lines.append(f"Chips still available: {available}.")
    lines.append(
        "One chip per gameweek. A free hit cannot be played the week after a free hit. "
        "Wildcard and free hit are not available in Gameweek 1."
    )
    return "\n".join(lines) + "\n"


def _cell(where: str, opponent: str, difficulty: object) -> str:
    if isinstance(difficulty, bool) or difficulty is None:
        return f"{where} {opponent}"
    try:
        rank = int(difficulty)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"{where} {opponent}"
    return f"{where} {opponent} {rank}"


def club_calendar(
    fixtures: Sequence[Mapping[str, Any]],
    short_names: Mapping[int, str],
    gw: int,
) -> str:
    """Every club from this gameweek through the end of the half.

    A week with other matches and none for this club is ``blank``. A week
    missing from the file is ``unknown``. A double lists both matches.
    """
    start = int(gw)
    end = _half_end(start)
    present: set[int] = set()
    by_week: dict[int, dict[int, list[str]]] = {}
    for row in fixtures:
        event = row.get("event")
        if event is None:
            continue
        week = int(event)
        if week < 1:
            continue
        present.add(week)
        if week < start or week > end:
            continue
        home = int(row["team_h"])
        away = int(row["team_a"])
        home_name = short_names.get(home, str(home))
        away_name = short_names.get(away, str(away))
        cells = by_week.setdefault(week, {})
        cells.setdefault(home, []).append(
            _cell("home", away_name, row.get("team_h_difficulty"))
        )
        cells.setdefault(away, []).append(
            _cell("away", home_name, row.get("team_a_difficulty"))
        )
    missing = [week for week in range(start, end + 1) if week not in present]
    lines = [
        "## Fixture calendar",
        "This calendar is for strategy. It is not a transfer plan. "
        "The horizon you return is still three weeks.",
        f"Clubs from Gameweek {start} through Gameweek {end}.",
    ]
    if missing:
        listed = ", ".join(f"GW{week}" for week in missing)
        lines.append(f"These gameweeks are not in the fixture file: {listed}. They are unknown.")
    for team_id in sorted(short_names, key=lambda key: short_names[key]):
        cells: list[str] = []
        for week in range(start, end + 1):
            if week not in present:
                cells.append(f"GW{week} unknown")
                continue
            matches = by_week.get(week, {}).get(team_id)
            if not matches:
                cells.append(f"GW{week} blank")
            else:
                cells.append(f"GW{week} " + ", ".join(matches))
        lines.append(f"- {short_names[team_id]}: " + "; ".join(cells))
    return "\n".join(lines) + "\n"
