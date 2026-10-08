"""Build markdown context for the string agent from roster + audited packets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.live.news_packets import (
    NewsPacket,
    compile_player_xmi,
    packets_for_player,
    render_player_context,
)
from src.str_agent.sources import StringSource, render_source


def roster_markdown(
    players: Sequence[Mapping[str, Any]],
    *,
    bank: int,
    ft: int,
    chips_left: Sequence[str],
) -> str:
    """Owned fifteen / market candidates as readable context."""
    lines = [
        f"Bank: {bank} tenths. Free transfers: {ft}. Chips left: {', '.join(chips_left) or 'none'}.",
        "",
        "| id | name | pos | club | cost | owned |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in players:
        lines.append(
            f"| {row['player_id']} | {row['name']} | {row['position']} | {row['club']} | "
            f"{row['now_cost']} | {'yes' if row.get('owned') else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def _notes_for(
    notes: Sequence[StringSource], player_id: int
) -> list[StringSource]:
    return [note for note in notes if int(player_id) in note.player_ids]


def packets_markdown(
    focus_ids: Sequence[int],
    directory: Mapping[int, Mapping[str, Any]],
    packets: Sequence[NewsPacket],
    minutes: Mapping[int, Sequence[float]] | None = None,
    string_sources: Sequence[StringSource] | None = None,
) -> str:
    """Press packets, open-outlet quotes, then a minutes footnote."""
    history = minutes or {}
    notes = list(string_sources or [])
    blocks: list[str] = []
    for pid in focus_ids:
        meta = directory.get(int(pid))
        if meta is None:
            continue
        block = render_player_context(
            player_id=int(pid),
            name=str(meta.get("name") or pid),
            position=str(meta.get("position") or ""),
            club=str(meta.get("club") or ""),
            prior=meta.get("prior"),
            status=meta.get("status"),
            chance=meta.get("chance"),
            fpl_news=str(meta.get("news") or ""),
            packets=packets_for_player(packets, int(pid)),
        )
        extra = _notes_for(notes, int(pid))
        if extra:
            quoted = "\n".join(render_source(note) for note in extra)
            block = block.rstrip() + "\n\n#### Forums, video, and other outlets\n" + quoted + "\n"
        compiled = compile_player_xmi(
            player_id=int(pid),
            name=str(meta.get("name") or pid),
            position=str(meta.get("position") or ""),
            prior=meta.get("prior"),
            status=meta.get("status"),
            chance=meta.get("chance"),
            packets=packets_for_player(packets, int(pid)),
            minutes=list(history.get(int(pid), [])),
        )
        xmi = compiled["xmi_compiled"]
        xmi_text = "none" if xmi is None else f"{float(xmi):.1f}"
        blocks.append(
            block.rstrip()
            + f"\n- Sidecar summary: {compiled['tag']}. xmi: {xmi_text}.\n"
        )
    return "\n".join(blocks)


def other_outlets_markdown(
    notes: Sequence[StringSource],
    focus_ids: Sequence[int],
) -> str:
    """Notes with no player, or a player outside the dossier list."""
    focus = {int(pid) for pid in focus_ids}
    loose = [
        note
        for note in notes
        if not note.player_ids or not set(note.player_ids) & focus
    ]
    if not loose:
        return ""
    lines = ["## Other outlets", ""]
    lines.extend(render_source(note) for note in loose)
    return "\n".join(lines) + "\n"


def build_context(
    *,
    gw: int,
    deadline_utc: str,
    roster: Sequence[Mapping[str, Any]],
    bank: int,
    ft: int,
    chips_left: Sequence[str],
    focus_ids: Sequence[int],
    directory: Mapping[int, Mapping[str, Any]],
    packets: Sequence[NewsPacket],
    minutes: Mapping[int, Sequence[float]] | None = None,
    string_sources: Sequence[StringSource] | None = None,
) -> str:
    """Full prompt context block (no model call)."""
    notes = list(string_sources or [])
    return (
        f"# Gameweek {int(gw)} string-agent context\n"
        f"Deadline: {deadline_utc}\n\n"
        "## Squad state\n"
        + roster_markdown(roster, bank=bank, ft=ft, chips_left=chips_left)
        + "\n## Audited news packets\n"
        + packets_markdown(focus_ids, directory, packets, minutes, notes)
        + "\n"
        + other_outlets_markdown(notes, focus_ids)
    )
