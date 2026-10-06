"""Official FPL 2026/27 constraints for solvers, simulators, and backtests.

Match-scoring figures are the long-standing Fantasy Premier League awards.
Bonus-point *changes* for 2026/27 follow the Premier League note of 20 Jul 2026
(CBI, tackles against, goalkeeper saves). The rest of the Opta BPS action table
is unchanged this season and is not re-encoded here.

Historical Vaastav ``total_points`` already use the rules of the season they
were scored in. Use these functions when simulating a gameweek from events,
not to overwrite published points.
"""

from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping

# --- squad -----------------------------------------------------------------

SQUAD_SIZE = 15
SQUAD_QUOTA = MappingProxyType({"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3})
BUDGET_M = 100.0
BUDGET_TENTHS = 1000  # Vaastav / FPL API store price in tenths of £m
MAX_PER_CLUB = 3

# Starting XI: 1 GKP, and outfield counts that sum to 10 inside these bounds.
MIN_DEF, MAX_DEF = 3, 5
MIN_MID, MAX_MID = 2, 5
MIN_FWD, MAX_FWD = 1, 3

# Every legal XI, including 5-2-3. The climb list is this tuple.
OFFICIAL_FORMATIONS: tuple[tuple[int, int, int], ...] = tuple(
    (n_def, n_mid, n_fwd)
    for n_def in range(MIN_DEF, MAX_DEF + 1)
    for n_mid in range(MIN_MID, MAX_MID + 1)
    for n_fwd in range(MIN_FWD, MAX_FWD + 1)
    if n_def + n_mid + n_fwd == 10
)

# --- transfers -------------------------------------------------------------

MAX_FT = 5
HIT_COST = 4  # points per transfer beyond the free-transfer bank
GW1_FT = 1

# --- chips -----------------------------------------------------------------

CHIPS = ("wildcard", "free_hit", "bench_boost", "triple_captain")
FIRST_HALF_END_GW = 19  # unused H1 chips expire at this deadline
N_GAMEWEEKS = 38
# Chips that make every transfer free and do not spend the FT bank.
FREE_TRANSFER_CHIPS = frozenset({"wildcard", "free_hit"})

# --- match points (not BPS) ------------------------------------------------

APPEARANCE_UNDER_60 = 1
APPEARANCE_60 = 2
MINUTES_FOR_APPEARANCE_BONUS = 60

# Official: goalkeeper and defender goals are both 6.
GOAL_POINTS = MappingProxyType({"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4})
ASSIST_POINTS = 3
CS_POINTS = MappingProxyType({"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0})
SAVES_PER_POINT = 3
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2
GOALS_CONCEDED_PER_DEDUCTION = 2
GOALS_CONCEDED_DEDUCTION = -1

# Defensive contributions, retained for 2026/27.
# DEF: clearances + blocks + interceptions + tackles (CBIT).
# MID/FWD: CBIT + recoveries (CBIRT). Goalkeepers are not eligible.
DEFCON_THRESHOLD = MappingProxyType({"DEF": 10, "MID": 12, "FWD": 12})
DEFCON_POINTS = 2

# --- BPS deltas (2026/27 only) ---------------------------------------------

CBI_PER_BPS = 3  # was 2 in 2025/26
BPS_PER_ANY_SAVE = 2
BPS_PER_INSIDE_BOX_SAVE = 1  # on top of BPS_PER_ANY_SAVE
BPS_PER_BIG_CHANCE_SAVE = 1
# Penalty-save line item fell 8 → 7 because a penalty is also a big chance (+1).
# Net for one penalty save stays 8. Do not also add the open-play save awards.
PENALTY_SAVE_BPS_LINE = 7
TACKLED_BPS = 0  # "tackled" deduction removed

# Bonus from the match BPS ranking.
BONUS_BY_PLACE = (3, 2, 1)


def normalize_position(position: str) -> str:
    """Map ``GK`` onto ``GKP``. Other codes must already be GKP/DEF/MID/FWD."""
    key = str(position).strip().upper()
    if key == "GK":
        return "GKP"
    if key not in SQUAD_QUOTA:
        raise ValueError(f"unknown position: {position}")
    return key


def half_for_gw(gw: int) -> str:
    """``H1`` is GW1–19. ``H2`` is GW20–38. H1 chips are dead in H2."""
    if not 1 <= int(gw) <= N_GAMEWEEKS:
        raise ValueError(f"gameweek out of range: {gw}")
    return "H1" if int(gw) <= FIRST_HALF_END_GW else "H2"


def sell_price(purchase: int, current: int) -> int:
    """Selling price in tenths of £m: half the rise, rounded down; full fall."""
    purchase = int(purchase)
    current = int(current)
    if current >= purchase:
        return purchase + (current - purchase) // 2
    return current


def hit_cost(ft_before: int, n_transfers: int, chip: str | None = None) -> int:
    """Point hit for transfers beyond the bank. Wildcard and Free Hit are free."""
    if chip in FREE_TRANSFER_CHIPS:
        return 0
    if n_transfers < 0 or ft_before < 0:
        raise ValueError("transfer counts must be non-negative")
    extra = max(0, int(n_transfers) - int(ft_before))
    return HIT_COST * extra


def advance_ft(ft_before: int, n_transfers: int, chip: str | None = None) -> int:
    """FT bank at the next deadline.

    Unused transfers roll, capped at 5, then the next gameweek grants one.
    A hit spends the bank down to zero before that grant.
    Wildcard and Free Hit do not spend the bank and do not grant another
    transfer: two saved free transfers are still two the next gameweek.
    """
    if ft_before < 0 or n_transfers < 0:
        raise ValueError("transfer counts must be non-negative")
    if chip in FREE_TRANSFER_CHIPS:
        return min(MAX_FT, int(ft_before))
    remaining = ft_before - n_transfers if n_transfers <= ft_before else 0
    return min(MAX_FT, remaining + 1)


class ChipWallet:
    """One of each chip per half. At most one chip in a gameweek.

    A chip played in GW1–19 is the first-half copy. The same name played in
    GW20–38 is the second-half copy. An unused first-half chip cannot be
    played from GW20. Wildcard and Free Hit are not available in GW1.
    A Free Hit cannot be played in the gameweek after another Free Hit.
    """

    def __init__(self) -> None:
        self.used: set[tuple[str, str]] = set()
        self.played_gw: dict[int, str] = {}

    def _free_hit_blocked(self, gw: int) -> bool:
        return any(
            played == "free_hit" and abs(int(prev) - int(gw)) == 1
            for prev, played in self.played_gw.items()
        )

    def available(self, gw: int) -> tuple[str, ...]:
        half = half_for_gw(gw)
        if int(gw) in self.played_gw:
            return ()
        chips = [chip for chip in CHIPS if (half, chip) not in self.used]
        if int(gw) == 1:
            chips = [chip for chip in chips if chip not in FREE_TRANSFER_CHIPS]
        if self._free_hit_blocked(gw):
            chips = [chip for chip in chips if chip != "free_hit"]
        return tuple(chips)

    def play(self, gw: int, chip: str) -> None:
        if chip not in CHIPS:
            raise ValueError(f"unknown chip: {chip}")
        if chip not in self.available(gw):
            raise ValueError(f"{chip} is not available in GW{gw}")
        self.used.add((half_for_gw(gw), chip))
        self.played_gw[int(gw)] = chip


def validate_chip_map(chips: Mapping[int, str] | None) -> dict[int, str]:
    """Check a caller-supplied week → chip map. An empty map plays nothing.

    The map is not a search. Weeks outside 1–38, unknown names, two chips
    in one week, a repeated chip inside one half, a GW1 wildcard or free
    hit, and back-to-back free hits all raise.
    """
    if not chips:
        return {}
    plan = {int(gw): str(chip) for gw, chip in chips.items()}
    wallet = ChipWallet()
    for gw in sorted(plan):
        wallet.play(gw, plan[gw])
    return plan


def captain_multiplier(chip: str | None = None) -> int:
    """Captain points factor. Triple Captain is ×3; otherwise ×2."""
    if chip == "triple_captain":
        return 3
    return 2


def captain_extra_points(
    captain_points: float,
    vice_points: float,
    *,
    captain_played: bool,
    vice_played: bool,
    chip: str | None = None,
) -> float:
    """Points added on top of the one copy already in the XI.

    A blank captain passes the armband to the vice-captain. Triple Captain
    triples that effective captain. If neither played, nothing is added.
    """
    mult = captain_multiplier(chip)
    if captain_played:
        return (mult - 1) * float(captain_points)
    if vice_played:
        return (mult - 1) * float(vice_points)
    return 0.0


def squad_legal(
    positions: list[str],
    clubs: list[str],
    values: list[int] | None = None,
    *,
    budget: int = BUDGET_TENTHS,
) -> bool:
    """15 players, exact 2/5/5/3, ≤3 per club, optional price cap in tenths."""
    if len(positions) != SQUAD_SIZE or len(clubs) != SQUAD_SIZE:
        return False
    try:
        counts = Counter(normalize_position(p) for p in positions)
    except ValueError:
        return False
    if dict(counts) != dict(SQUAD_QUOTA):
        return False
    if any(n > MAX_PER_CLUB for n in Counter(clubs).values()):
        return False
    if values is not None:
        if len(values) != SQUAD_SIZE:
            return False
        if sum(int(v) for v in values) > budget:
            return False
    return True


def xi_legal(positions: list[str]) -> bool:
    """1 GKP and a legal outfield formation."""
    if len(positions) != 11:
        return False
    try:
        counts = Counter(normalize_position(p) for p in positions)
    except ValueError:
        return False
    n_def = counts["DEF"]
    n_mid = counts["MID"]
    n_fwd = counts["FWD"]
    return (
        counts["GKP"] == 1
        and MIN_DEF <= n_def <= MAX_DEF
        and MIN_MID <= n_mid <= MAX_MID
        and MIN_FWD <= n_fwd <= MAX_FWD
        and n_def + n_mid + n_fwd == 10
    )


def appearance_points(minutes: int) -> int:
    if minutes <= 0:
        return 0
    if minutes < MINUTES_FOR_APPEARANCE_BONUS:
        return APPEARANCE_UNDER_60
    return APPEARANCE_60


def goal_points(position: str, goals: int) -> int:
    if goals < 0:
        raise ValueError("goals must be non-negative")
    return GOAL_POINTS[normalize_position(position)] * int(goals)


def assist_points(assists: int) -> int:
    if assists < 0:
        raise ValueError("assists must be non-negative")
    return ASSIST_POINTS * int(assists)


def clean_sheet_points(position: str, minutes: int, clean_sheet: bool) -> int:
    """Clean-sheet points require 60 minutes."""
    if not clean_sheet or minutes < MINUTES_FOR_APPEARANCE_BONUS:
        return 0
    return CS_POINTS[normalize_position(position)]


def goals_conceded_points(position: str, minutes: int, goals_conceded: int) -> int:
    """−1 per two goals conceded for a goalkeeper or defender who played."""
    if minutes <= 0 or goals_conceded < 0:
        if goals_conceded < 0:
            raise ValueError("goals_conceded must be non-negative")
        return 0
    pos = normalize_position(position)
    if pos not in ("GKP", "DEF"):
        return 0
    return GOALS_CONCEDED_DEDUCTION * (int(goals_conceded) // GOALS_CONCEDED_PER_DEDUCTION)


def save_points(saves: int) -> int:
    """1 point per three goalkeeper saves. Separate from BPS."""
    if saves < 0:
        raise ValueError("saves must be non-negative")
    return int(saves) // SAVES_PER_POINT


def defcon_points(position: str, actions: int) -> int:
    """2 points at the position threshold. Pass CBIT for DEF, CBIRT for MID/FWD.

    Goalkeepers are not eligible. There is no 60-minute gate: the award is the
    action count. ``actions`` must already be the right sum for the position.
    """
    if actions < 0:
        raise ValueError("actions must be non-negative")
    pos = normalize_position(position)
    threshold = DEFCON_THRESHOLD.get(pos)
    if threshold is None:
        return 0
    return DEFCON_POINTS if int(actions) >= threshold else 0


def card_points(yellows: int, reds: int) -> int:
    if yellows < 0 or reds < 0:
        raise ValueError("cards must be non-negative")
    return YELLOW_CARD_POINTS * int(yellows) + RED_CARD_POINTS * int(reds)


def own_goal_points(own_goals: int) -> int:
    if own_goals < 0:
        raise ValueError("own_goals must be non-negative")
    return OWN_GOAL_POINTS * int(own_goals)


def penalty_points(saves: int, misses: int) -> int:
    """FPL points for penalty saves (+5) and misses (−2), not BPS."""
    if saves < 0 or misses < 0:
        raise ValueError("penalty counts must be non-negative")
    return PENALTY_SAVE_POINTS * int(saves) + PENALTY_MISS_POINTS * int(misses)


def cbi_bps(cbi: int) -> int:
    """1 BPS per three clearances, blocks, and interceptions (2026/27)."""
    if cbi < 0:
        raise ValueError("cbi must be non-negative")
    return int(cbi) // CBI_PER_BPS


def tackled_bps(times_tackled: int) -> int:
    """The tackled deduction was removed for 2026/27."""
    if times_tackled < 0:
        raise ValueError("times_tackled must be non-negative")
    return TACKLED_BPS


def gk_save_bps(
    *,
    open_play_saves: int,
    open_play_inside_box: int,
    open_play_big_chances: int,
    penalty_saves: int,
) -> int:
    """Goalkeeper save BPS for 2026/27.

    Open-play saves: +2 each, +1 more if the shot was inside the box, +1 more
    if it was a big chance. Outside-the-box no longer has its own award.

    Penalty saves are not included in the open-play counts. Each is worth 7 BPS
    on the penalty line plus 1 for the big-chance save (net 8, matching
    2025/26). A penalty is inside the box; that +1 is not stacked on top.
    """
    if min(open_play_saves, open_play_inside_box, open_play_big_chances, penalty_saves) < 0:
        raise ValueError("save counts must be non-negative")
    if open_play_inside_box > open_play_saves or open_play_big_chances > open_play_saves:
        raise ValueError("inside-box and big-chance saves cannot exceed open-play saves")
    return (
        BPS_PER_ANY_SAVE * int(open_play_saves)
        + BPS_PER_INSIDE_BOX_SAVE * int(open_play_inside_box)
        + BPS_PER_BIG_CHANCE_SAVE * int(open_play_big_chances)
        + (PENALTY_SAVE_BPS_LINE + BPS_PER_BIG_CHANCE_SAVE) * int(penalty_saves)
    )


def bonus_points(bps: Mapping[str, int]) -> dict[str, int]:
    """Map player id → bonus (3/2/1) from match BPS totals.

    Official tie breaks (Premier League, 20 Jul 2026):

    - two tied for first: both get 3, the next player gets 1
    - two or more tied for second: they get 2, and third place is not paid
    - two or more tied for third: they get 1

    Three or more tied for first each get 3 and nobody else is paid. That
    case is not spelled out in the 2026 note; it is the long-standing rule.
    """
    if not bps:
        return {}
    groups: dict[int, list[str]] = {}
    for pid, score in bps.items():
        groups.setdefault(int(score), []).append(str(pid))
    ordered = sorted(groups.items(), key=lambda kv: kv[0], reverse=True)

    awarded: dict[str, int] = {}
    rank = 1
    for _score, pids in ordered:
        if rank > 3:
            break
        if rank == 1 and len(pids) >= 3:
            for pid in pids:
                awarded[pid] = 3
            break
        if rank == 1 and len(pids) == 2:
            for pid in pids:
                awarded[pid] = 3
            rank = 3
            continue
        if rank == 1:
            awarded[pids[0]] = 3
        elif rank == 2:
            for pid in pids:
                awarded[pid] = 2
            if len(pids) >= 2:
                break
        else:
            for pid in pids:
                awarded[pid] = 1
            break
        rank += len(pids)
    return awarded
