"""His chip weeks, with the illegal ones left unset.

The squad is not read. ``score_xp``, the hold, and both chip margins stay
where they are. A week this filter drops is reported and not played.
"""

from __future__ import annotations

from typing import Mapping

from src.rules.fpl_2026 import ChipWallet

# Locked before the forced calendar is scored. Chip value is the forced
# gap minus the autonomous gap. A non-positive mean, or a mean forced gap
# at or below the floor, kills the claim that his chip weeks explain the
# deficit. Neither number is a new margin.
CHIP_VALUE_FLOOR = 0.0
FORCED_GAP_FLOOR = -20.0


def legal_calendar(
    played: Mapping[int, str],
) -> tuple[dict[int, str], tuple[tuple[int, str], ...]]:
    """Keep the chips the wallet allows, in week order.

    A Gameweek 1 wildcard or free hit is dropped. So is a second chip in
    the same half, a second chip in a week, and a free hit the week after
    a free hit. The returned map is the one the carry may play. The
    dropped pairs are ``(gameweek, chip)``.
    """
    wallet = ChipWallet()
    kept: dict[int, str] = {}
    dropped: list[tuple[int, str]] = []
    for gw in sorted(int(week) for week in played):
        chip = str(played[gw])
        if chip not in wallet.available(gw):
            dropped.append((gw, chip))
            continue
        wallet.play(gw, chip)
        kept[gw] = chip
    return kept, tuple(dropped)


def hypothesis_killed(mean_chip_value: float, mean_forced_gap: float) -> bool:
    """True when his chip weeks do not explain the deficit."""
    if float(mean_chip_value) <= CHIP_VALUE_FLOOR:
        return True
    return float(mean_forced_gap) <= FORCED_GAP_FLOOR
