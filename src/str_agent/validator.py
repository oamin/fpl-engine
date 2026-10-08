"""Hard legality checks for string-agent drafts (2026/27 rules)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.rules.fpl_2026 import (
    CHIPS,
    MAX_PER_CLUB,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    normalize_position,
    squad_legal,
    xi_legal,
)


class DraftError(ValueError):
    """The draft breaks a hard FPL rule."""


def _lookup(directory: Mapping[str, Mapping[str, Any]], pid: str) -> Mapping[str, Any] | None:
    if pid in directory:
        return directory[pid]
    if pid.isdigit() and int(pid) in directory:
        return directory[int(pid)]  # type: ignore[index]
    return None


def validate_draft(
    draft: Mapping[str, Any],
    *,
    directory: Mapping[str, Mapping[str, Any]],
    budget: int,
    skip_price_cap: bool = False,
) -> list[str]:
    """Return error strings; empty means legal enough to freeze.

    ``directory`` maps player_id → {position, club, now_cost}.
    ``skip_price_cap`` is for a carried squad: team value may exceed 1000,
    and solvency is the bank check in ``carry.validate_move``.
    """
    errors: list[str] = []
    decision = draft.get("decision") or {}
    squad = [str(pid) for pid in decision.get("squad_15") or []]
    xi = [str(pid) for pid in decision.get("starting_11") or []]
    bench = [str(pid) for pid in decision.get("bench_order") or []]
    if len(squad) != len(set(squad)):
        errors.append("squad_15 has duplicate players")
    if len(squad) != SQUAD_SIZE:
        errors.append(f"squad_15 must have {SQUAD_SIZE} players")
    if len(xi) != 11:
        errors.append("starting_11 must have 11 players")
    if len(bench) != 4:
        errors.append("bench_order must have 4 players")
    if squad and xi and bench and set(xi) | set(bench) != set(squad):
        errors.append("starting_11 and bench_order must partition squad_15")
    resolved = {pid: _lookup(directory, pid) for pid in squad}
    missing = [pid for pid, meta in resolved.items() if meta is None]
    if missing:
        errors.append(f"unknown players: {', '.join(missing[:5])}")
        return errors
    positions = [str(resolved[pid]["position"]) for pid in squad]
    clubs = [str(resolved[pid]["club"]) for pid in squad]
    if skip_price_cap:
        legal_squad = squad_legal(positions, clubs)
        cap_note = "squad fails quota or club cap"
    else:
        values = [int(resolved[pid]["now_cost"]) for pid in squad]
        legal_squad = squad_legal(positions, clubs, values, budget=int(budget))
        cap_note = "squad fails quota, club cap, or budget"
    if not legal_squad:
        errors.append(cap_note)
    xi_meta = {pid: _lookup(directory, pid) for pid in xi}
    if any(meta is None for meta in xi_meta.values()):
        errors.append("starting_11 has a player outside squad_15")
        return errors
    xi_pos = [str(xi_meta[pid]["position"]) for pid in xi]
    if xi and not xi_legal(xi_pos):
        errors.append("starting_11 is not a legal formation")
    captain = str(decision.get("captain") or "")
    vice = str(decision.get("vice_captain") or "")
    if captain and captain not in xi:
        errors.append("captain must be in starting_11")
    if vice and vice not in xi:
        errors.append("vice_captain must be in starting_11")
    if captain and vice and captain == vice:
        errors.append("captain and vice_captain must differ")
    chip = decision.get("chip_played")
    if chip not in {None, ""} and str(chip) not in CHIPS:
        errors.append(f"unknown chip {chip}")
    # silence unused import warning path for SQUAD_QUOTA in docs
    _ = (MAX_PER_CLUB, SQUAD_QUOTA, normalize_position)
    return errors


def assert_legal(
    draft: Mapping[str, Any],
    *,
    directory: Mapping[str, Mapping[str, Any]],
    budget: int,
) -> None:
    errors = validate_draft(draft, directory=directory, budget=budget)
    if errors:
        raise DraftError("; ".join(errors))
