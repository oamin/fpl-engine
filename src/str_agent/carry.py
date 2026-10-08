"""Transfer accounting from a carried squad to a string-agent draft.

Uses ``sell_price``, ``hit_cost``, ``advance_ft``, and ``ChipWallet`` from
``src/rules/fpl_2026``. Does not copy those constants and does not suggest
transfers. A free hit reverts the next carry to the pre-chip squad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.rules.fpl_2026 import (
    MAX_FT,
    ChipWallet,
    advance_ft,
    hit_cost,
    sell_price,
)
from src.str_agent.validator import validate_draft


class CarryError(ValueError):
    """The carried squad cannot be read."""


@dataclass(frozen=True)
class CarryState:
    """Squad and bank as they stand before this deadline."""

    gw: int
    squad: tuple[str, ...]
    purchase_prices: Mapping[str, int]
    bank: int
    ft_before: int
    chips_played: Mapping[int, str]
    selling_prices: Mapping[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gw": int(self.gw),
            "squad": list(self.squad),
            "purchase_prices": {str(k): int(v) for k, v in self.purchase_prices.items()},
            "bank": int(self.bank),
            "ft_before": int(self.ft_before),
            "chips_played": {str(k): str(v) for k, v in self.chips_played.items()},
            "selling_prices": {str(k): int(v) for k, v in self.selling_prices.items()},
        }


@dataclass(frozen=True)
class MoveResult:
    """Errors plus the next carry when the move is legal."""

    errors: list[str]
    hits: int
    bank_after: int | None
    carry_after: dict[str, Any] | None


def _ids(values: Any) -> list[str]:
    return [str(pid) for pid in values or []]


def _meta(directory: Mapping[Any, Mapping[str, Any]], pid: str) -> Mapping[str, Any] | None:
    if pid in directory:
        return directory[pid]
    if pid.isdigit() and int(pid) in directory:
        return directory[int(pid)]
    return None


def _sell_proceeds(
    pid: str,
    carry: CarryState,
    directory: Mapping[Any, Mapping[str, Any]],
) -> tuple[int | None, str | None]:
    """Selling price in tenths, or an error if the price cannot be known."""
    if pid in carry.selling_prices:
        return int(carry.selling_prices[pid]), None
    if pid not in carry.purchase_prices:
        return None, f"{pid} has no purchase price and no selling price"
    meta = _meta(directory, pid)
    if meta is None or meta.get("now_cost") is None:
        return None, f"{pid} has no current price"
    return sell_price(int(carry.purchase_prices[pid]), int(meta["now_cost"])), None


def _last_week(entry: Mapping[str, Any]) -> dict[str, Any]:
    played = int(entry["played_through"])
    for row in entry.get("gameweeks") or []:
        if int(row["gw"]) == played:
            return row
    raise CarryError(f"entry has no gameweek {played}")


def carry_from_entry(entry: Mapping[str, Any]) -> CarryState:
    """Seed the next deadline from the last played week of an entry snapshot.

    A free-hit week is refused: those picks are temporary and the snapshot
    does not keep the reverted fifteen.
    """
    week = _last_week(entry)
    if week.get("chip") == "free_hit":
        raise CarryError(
            f"GW{week['gw']} was a free hit; the snapshot squad is temporary"
        )
    players = list(week.get("xi") or []) + list(week.get("bench") or [])
    squad = tuple(str(player["id"]) for player in players)
    if len(squad) != len(set(squad)):
        raise CarryError("entry squad has duplicate players")
    purchase: dict[str, int] = {}
    selling: dict[str, int] = {}
    for player in players:
        pid = str(player["id"])
        if player.get("purchase_price") is not None:
            purchase[pid] = int(player["purchase_price"])
        if player.get("selling_price") is not None:
            selling[pid] = int(player["selling_price"])
    chips = {
        int(row["gw"]): str(row["chip"])
        for row in entry.get("chips_played") or []
        if row.get("chip")
    }
    return CarryState(
        gw=int(entry["next_gw"]),
        squad=squad,
        purchase_prices=purchase,
        bank=int(entry["bank"]),
        ft_before=int(entry["ft_for_next"]),
        chips_played=chips,
        selling_prices=selling,
    )


def directory_from_entry(entry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Position and club from the last played week. Price only if the row has one."""
    week = _last_week(entry)
    directory: dict[str, dict[str, Any]] = {}
    for player in list(week.get("xi") or []) + list(week.get("bench") or []):
        meta: dict[str, Any] = {
            "position": player["position"],
            "club": player["team"],
            "name": player.get("name"),
        }
        if player.get("now_cost") is not None:
            meta["now_cost"] = int(player["now_cost"])
        directory[str(player["id"])] = meta
    return directory


def hold_draft_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The same fifteen, XI, and armbands, with no transfers and no chip."""
    week = _last_week(entry)
    players = list(week.get("xi") or []) + list(week.get("bench") or [])
    by_name: dict[str, str] = {}
    for player in players:
        name = str(player["name"])
        pid = str(player["id"])
        if name in by_name and by_name[name] != pid:
            raise CarryError(f"duplicate name {name}")
        by_name[name] = pid
    captain = by_name.get(str(week.get("captain") or ""))
    vice = by_name.get(str(week.get("vice") or ""))
    if not captain or not vice:
        raise CarryError("entry week is missing a captain or vice-captain")
    return {
        "decision": {
            "chip_played": None,
            "squad_15": [str(player["id"]) for player in players],
            "starting_11": [str(player["id"]) for player in week.get("xi") or []],
            "captain": captain,
            "vice_captain": vice,
            "bench_order": [str(player["id"]) for player in week.get("bench") or []],
            "transfers_in": [],
            "transfers_out": [],
        }
    }


def _carry_after(
    carry: CarryState,
    *,
    new_squad: list[str],
    ins: set[str],
    chip: str | None,
    n_transfers: int,
    bank_after: int,
    directory: Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    played = {int(gw): str(name) for gw, name in carry.chips_played.items()}
    if chip:
        played[int(carry.gw)] = chip
    ft_next = advance_ft(int(carry.ft_before), int(n_transfers), chip)
    if chip == "free_hit":
        nxt = CarryState(
            gw=int(carry.gw) + 1,
            squad=tuple(carry.squad),
            purchase_prices=dict(carry.purchase_prices),
            bank=int(carry.bank),
            ft_before=ft_next,
            chips_played=played,
            selling_prices={},
        )
        return nxt.as_dict()
    purchase: dict[str, int] = {}
    for pid in new_squad:
        if pid in ins:
            meta = _meta(directory, pid)
            if meta is None or meta.get("now_cost") is None:
                raise CarryError(f"{pid} has no current price")
            purchase[pid] = int(meta["now_cost"])
        elif pid in carry.purchase_prices:
            purchase[pid] = int(carry.purchase_prices[pid])
    nxt = CarryState(
        gw=int(carry.gw) + 1,
        squad=tuple(new_squad),
        purchase_prices=purchase,
        bank=int(bank_after),
        ft_before=ft_next,
        chips_played=played,
        selling_prices={},
    )
    return nxt.as_dict()


def validate_move(
    draft: Mapping[str, Any],
    carry: CarryState,
    directory: Mapping[Any, Mapping[str, Any]],
) -> MoveResult:
    """Check the draft is the carried squad plus a legal set of transfers.

    Returns errors. ``carry_after`` is set only when the move is legal.
    """
    errors: list[str] = []
    if int(carry.ft_before) < 0 or int(carry.ft_before) > MAX_FT:
        errors.append(f"ft_before {carry.ft_before} is outside 0..{MAX_FT}")
    decision = draft.get("decision") or {}
    squad = _ids(decision.get("squad_15"))
    declared_in = _ids(decision.get("transfers_in"))
    declared_out = _ids(decision.get("transfers_out"))
    if len(squad) != len(set(squad)):
        errors.append("squad_15 has duplicate players")
    if len(declared_in) != len(set(declared_in)):
        errors.append("transfers_in has duplicate players")
    if len(declared_out) != len(set(declared_out)):
        errors.append("transfers_out has duplicate players")
    owned = [str(pid) for pid in carry.squad]
    if len(owned) != len(set(owned)):
        errors.append("carry squad has duplicate players")
    ins = set(squad) - set(owned)
    outs = set(owned) - set(squad)
    if set(declared_in) & set(declared_out) or ins & outs:
        errors.append("a player is in both transfers_in and transfers_out")
    if set(declared_in) != ins or set(declared_out) != outs:
        errors.append("transfers_in and transfers_out must match the squad change")
    if len(ins) != len(outs):
        errors.append("transfers in and out must be the same count")

    chip_raw = decision.get("chip_played")
    chip: str | None
    if chip_raw in (None, ""):
        chip = None
    else:
        chip = str(chip_raw)
    wallet = ChipWallet()
    try:
        for gw in sorted(int(week) for week in carry.chips_played):
            if gw >= int(carry.gw):
                raise ValueError(f"chips_played includes GW{gw}")
            wallet.play(gw, str(carry.chips_played[gw]))
        if chip is not None and chip not in wallet.available(int(carry.gw)):
            errors.append(f"{chip} is not available in GW{carry.gw}")
    except ValueError as exc:
        errors.append(f"chip wallet: {exc}")

    n_transfers = len(ins)
    hits = 0
    try:
        hits = hit_cost(int(carry.ft_before), n_transfers, chip)
    except ValueError as exc:
        errors.append(str(exc))

    bank_after: int | None = None
    if not (ins & outs) and len(owned) == len(set(owned)) and len(squad) == len(set(squad)):
        proceeds = 0
        spend = 0
        priced = True
        for pid in sorted(outs):
            amount, problem = _sell_proceeds(pid, carry, directory)
            if problem is not None or amount is None:
                errors.append(problem or f"{pid} has no selling price")
                priced = False
                continue
            proceeds += amount
        for pid in sorted(ins):
            meta = _meta(directory, pid)
            if meta is None or meta.get("now_cost") is None:
                errors.append(f"{pid} has no current price")
                priced = False
                continue
            spend += int(meta["now_cost"])
        if priced:
            bank_after = int(carry.bank) + proceeds - spend
            if bank_after < 0:
                errors.append(f"bank_after is {bank_after}")

    errors.extend(
        validate_draft(
            draft,
            directory=directory,
            budget=0,
            skip_price_cap=True,
        )
    )
    # validate_draft also reports duplicate squads; keep one copy.
    deduped: list[str] = []
    for error in errors:
        if error not in deduped:
            deduped.append(error)
    if deduped or bank_after is None:
        return MoveResult(deduped, hits, bank_after, None)
    try:
        nxt = _carry_after(
            carry,
            new_squad=squad,
            ins=ins,
            chip=chip,
            n_transfers=n_transfers,
            bank_after=bank_after,
            directory=directory,
        )
    except CarryError as exc:
        deduped.append(str(exc))
        return MoveResult(deduped, hits, bank_after, None)
    return MoveResult([], hits, bank_after, nxt)
