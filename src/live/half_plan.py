"""Half-season chip schedule.

The caller prices each week. This module does not pick a squad, change
``score_xp``, or run a climb. Chip law is ``ChipWallet``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.live.policy import FH_MARGIN, WC_MARGIN
from src.rules.fpl_2026 import (
    CHIPS,
    FIRST_HALF_END_GW,
    N_GAMEWEEKS,
    ChipWallet,
    half_for_gw,
)

_VALUE_TIE = 1e-6


class HalfPlanError(ValueError):
    """The week table does not cover the half being planned."""


@dataclass(frozen=True)
class SquadOutlook:
    """Published single-fixture points for one squad in one week."""

    xi_xp: float
    bench_xp: float
    cap_xp: float


@dataclass(frozen=True)
class WeekInputs:
    """One week in the remaining half. ``cap_xp`` is one extra captain copy."""

    gw: int
    held: SquadOutlook
    rebuilt: SquadOutlook | None = None
    fh_xi: float | None = None


@dataclass(frozen=True)
class HalfPlan:
    """``chip`` is the action this week. ``schedule`` matches that action."""

    chip: str | None
    schedule: dict[str, int | None]
    value: float


def bench_week(plan: HalfPlan, current_gw: int) -> int | None:
    """Bench Boost week still ahead, including this week. A used chip is None."""
    target = plan.schedule.get("bench_boost")
    if target is not None and int(target) >= int(current_gw):
        return int(target)
    return None


def half_end(gw: int) -> int:
    """Last gameweek of the half that contains ``gw``."""
    return FIRST_HALF_END_GW if half_for_gw(gw) == "H1" else N_GAMEWEEKS


def plan_half(
    current_gw: int,
    weeks: Sequence[WeekInputs],
    played: Mapping[int, str] | None = None,
) -> HalfPlan:
    """Best legal chip schedule through the end of this half.

    A wildcard week switches the squad in view to ``rebuilt`` from that week
    on. A free-hit week scores ``fh_xi`` only. Any other week scores the
    eleven, plus the bench on Bench Boost, plus ``cap_xp`` on Triple Captain.
    Unused chips add nothing. The sum is not discounted.

    Bench Boost and Triple Captain play this week when it is strictly the
    best week left for that chip. Free Hit also needs a lead of ``FH_MARGIN``
    on this week alone. Wildcard also needs the rebuilt eleven to lead the
    held eleven by ``WC_MARGIN`` across the rest of the half. Two chips with
    the same best value play nothing.
    """
    table = _week_table(current_gw, weeks)
    already = _played(played)
    if current_gw in already:
        raise HalfPlanError(f"GW{current_gw} already has a chip")
    gws = [row.gw for row in table]
    rebuilt_ok = all(row.rebuilt is not None for row in table)
    legal = [
        item
        for item in _candidates(gws)
        if _legal(item, already, table)
    ]
    if not legal:
        raise HalfPlanError("no legal chip schedule")
    scored = [(item, _value(item, table)) for item in legal]
    best = max(value for _, value in scored)
    optimal = [item for item, value in scored if abs(value - best) <= _VALUE_TIE]
    chips_now = {_chip_now(item, current_gw) for item in optimal}
    real = {chip for chip in chips_now if chip is not None}
    quiet = _best_quiet(scored, current_gw)
    if len(real) != 1 or None in chips_now:
        return quiet
    chip = real.pop()
    if not _hurdle(chip, table, rebuilt_ok):
        return quiet
    chosen = _best_with(scored, current_gw, chip)
    return HalfPlan(chip=chip, schedule=_as_map(chosen[0]), value=chosen[1])


def _week_table(current_gw: int, weeks: Sequence[WeekInputs]) -> tuple[WeekInputs, ...]:
    end = half_end(current_gw)
    expected = list(range(int(current_gw), end + 1))
    found = [int(row.gw) for row in weeks]
    if found != expected:
        raise HalfPlanError(
            f"weeks must be GW{expected[0]}–GW{expected[-1]} in order"
        )
    rebuilt = [row.rebuilt is not None for row in weeks]
    if any(rebuilt) and not all(rebuilt):
        raise HalfPlanError("a wildcard rebuild must be priced on every week")
    return tuple(weeks)


def _played(played: Mapping[int, str] | None) -> dict[int, str]:
    if not played:
        return {}
    wallet = ChipWallet()
    ordered = {int(gw): str(chip) for gw, chip in played.items()}
    for gw in sorted(ordered):
        wallet.play(gw, ordered[gw])
    return ordered


def _candidates(gws: list[int]):
    slots = (None, *gws)
    for combo in itertools.product(slots, repeat=len(CHIPS)):
        used = [gw for gw in combo if gw is not None]
        if len(used) != len(set(used)):
            continue
        yield combo


def _legal(combo: tuple, played: dict[int, str], table: tuple[WeekInputs, ...]) -> bool:
    by_gw = {int(row.gw): row for row in table}
    wallet = ChipWallet()
    for gw in sorted(played):
        wallet.play(gw, played[gw])
    assigned = [(gw, CHIPS[index]) for index, gw in enumerate(combo) if gw is not None]
    for gw, chip in sorted(assigned):
        row = by_gw[gw]
        if chip == "wildcard" and row.rebuilt is None:
            return False
        if chip == "free_hit" and row.fh_xi is None:
            return False
        if chip not in wallet.available(gw):
            return False
        wallet.play(gw, chip)
    return True


def _value(combo: tuple, table: tuple[WeekInputs, ...]) -> float:
    assigned = {CHIPS[index]: gw for index, gw in enumerate(combo)}
    wildcard = assigned["wildcard"]
    total = 0.0
    for row in table:
        if assigned["free_hit"] == row.gw:
            total += float(row.fh_xi or 0.0)
            continue
        squad = row.held
        if wildcard is not None and row.gw >= wildcard:
            squad = row.rebuilt if row.rebuilt is not None else row.held
        total += float(squad.xi_xp)
        if assigned["bench_boost"] == row.gw:
            total += float(squad.bench_xp)
        if assigned["triple_captain"] == row.gw:
            total += float(squad.cap_xp)
    return total


def _chip_now(combo: tuple, current_gw: int) -> str | None:
    for index, gw in enumerate(combo):
        if gw == current_gw:
            return CHIPS[index]
    return None


def _as_map(combo: tuple) -> dict[str, int | None]:
    return {chip: gw for chip, gw in zip(CHIPS, combo, strict=True)}


def _best_quiet(scored: list[tuple[tuple, float]], current_gw: int) -> HalfPlan:
    quiet = [
        (combo, value)
        for combo, value in scored
        if _chip_now(combo, current_gw) is None
    ]
    combo, value = max(quiet, key=lambda item: item[1])
    return HalfPlan(chip=None, schedule=_as_map(combo), value=value)


def _best_with(
    scored: list[tuple[tuple, float]], current_gw: int, chip: str
) -> tuple[tuple, float]:
    matching = [
        (combo, value)
        for combo, value in scored
        if _chip_now(combo, current_gw) == chip and abs(value - max(v for _, v in scored)) <= _VALUE_TIE
    ]
    return max(matching, key=lambda item: item[1])


def _hurdle(chip: str, table: tuple[WeekInputs, ...], rebuilt_ok: bool) -> bool:
    row = table[0]
    if chip == "bench_boost":
        return float(row.held.bench_xp) > 0.0
    if chip == "triple_captain":
        return float(row.held.cap_xp) > 0.0
    if chip == "free_hit":
        if row.fh_xi is None:
            return False
        return float(row.fh_xi) - float(row.held.xi_xp) >= FH_MARGIN
    if chip == "wildcard":
        if not rebuilt_ok:
            return False
        gap = sum(
            float(week.rebuilt.xi_xp) - float(week.held.xi_xp)  # type: ignore[union-attr]
            for week in table
        )
        return gap >= WC_MARGIN
    return False
