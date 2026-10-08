"""Build markdown context for the string agent from roster + audited packets."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.live.news_packets import NewsPacket, packets_for_player, render_player_context


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


def packets_markdown(
    focus_ids: Sequence[int],
    directory: Mapping[int, Mapping[str, Any]],
    packets: Sequence[NewsPacket],
) -> str:
    """Packet evidence for a shortlist of players."""
    blocks: list[str] = []
    for pid in focus_ids:
        meta = directory.get(int(pid))
        if meta is None:
            continue
        blocks.append(
            render_player_context(
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
        )
    return "\n".join(blocks)


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
) -> str:
    """Full prompt context block (no model call)."""
    return (
        f"# Gameweek {int(gw)} string-agent context\n"
        f"Deadline: {deadline_utc}\n\n"
        "## Squad state\n"
        + roster_markdown(roster, bank=bank, ft=ft, chips_left=chips_left)
        + "\n## Audited news packets\n"
        + packets_markdown(focus_ids, directory, packets)
    )
