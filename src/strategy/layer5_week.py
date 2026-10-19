"""Layer 5 — chip helpers, scoring, WC/FH state (Steps 1–3).

Step 1: ``ChipWallet`` / FT / hit identity from ``fpl_2026`` (no re-encode).
Step 2: thin TC/BB **scoring** on a given 15+XI+captain+bench.
Step 3: WC/FH **state machine** — commit squad vs next-deadline squad
(FH reverts; WC sticks; FT preserved under both). Joint foresight is Step 4.

Margins are diagnostics later — never play thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from src.rules.fpl_2026 import (
    BUDGET_TENTHS,
    CHIPS,
    FREE_TRANSFER_CHIPS,
    ChipWallet,
    advance_ft,
    captain_multiplier,
    hit_cost,
    validate_chip_map,
)
from src.strategy.layer3_week import HeldState, held_from_frame

# Decision action: no chip, or one legal chip name.
NO_CHIP = "none"
SCORING_CHIPS = frozenset({NO_CHIP, "triple_captain", "bench_boost"})
TRANSFER_CHIPS = frozenset({"wildcard", "free_hit"})


@dataclass
class ChipDecisionState:
    """Held squad + chip wallet before a deadline."""

    held: HeldState
    wallet: ChipWallet = field(default_factory=ChipWallet)

    def available_chips(self, gw: int) -> tuple[str, ...]:
        return self.wallet.available(int(gw))

    def actions(self, gw: int) -> tuple[str, ...]:
        """Ordered decision set: ``none`` then each available chip."""
        return (NO_CHIP,) + self.available_chips(gw)


@dataclass(frozen=True)
class PlannerState:
    """Full planner node entering a deadline (Step 3+)."""

    held: HeldState
    wallet: ChipWallet = field(default_factory=ChipWallet)

    def decision(self) -> ChipDecisionState:
        return ChipDecisionState(held=self.held, wallet=self.wallet)


@dataclass(frozen=True)
class DeadlineTransition:
    """Outcome of one deadline under a chip + post-move squad."""

    chip: str
    commit_held: HeldState
    next_state: PlannerState
    n_transfers: int
    hit_points: int
    ft_after: int


def _norm_chip(chip: str | None) -> str:
    return NO_CHIP if chip in (None, NO_CHIP) else str(chip)


def held_copy(held: HeldState, *, ft: int | None = None) -> HeldState:
    """Copy HeldState, optionally overriding FT."""
    return HeldState(
        held.player_ids,
        dict(held.purchase),
        bank=held.bank,
        ft=int(held.ft if ft is None else ft),
    )


def held_from_squad_frame(
    squad: pd.DataFrame,
    *,
    ft: int,
    purchase_col: str = "purchase",
    bank: int | None = None,
) -> HeldState:
    """Build HeldState from a 15-row rebuild (WC/FH). Purchase defaults to value."""
    frame = squad.copy()
    if purchase_col not in frame.columns:
        frame[purchase_col] = frame["value"].astype(int)
    if bank is None:
        spent = int(frame[purchase_col].sum())
        bank = max(0, int(BUDGET_TENTHS - spent))
    return held_from_frame(frame, bank=int(bank), ft=int(ft), purchase_col=purchase_col)


def advance_planner_state(
    state: PlannerState,
    *,
    gw: int,
    chip: str | None,
    commit_held: HeldState,
    n_transfers: int = 0,
) -> DeadlineTransition:
    """Apply chip physics after a commit-week squad is chosen.

    ``commit_held`` is the 15 used for this week's XI (rebuild under WC/FH,
    or post-transfer squad under none/TC/BB).

    Next-deadline squad:
      - ``free_hit``: revert to ``state.held`` (pre-chip), FT preserved
      - ``wildcard``: keep ``commit_held``, FT preserved
      - otherwise: keep ``commit_held``, FT via ordinary ``advance_ft``

    Hits are always 0 under WC/FH. Wallet is updated when a real chip is played.
    """
    c = _norm_chip(chip)
    if c != NO_CHIP and c not in state.wallet.available(int(gw)):
        raise ValueError(f"{c} not available in GW{gw}: {state.wallet.available(int(gw))}")

    n_tx = int(n_transfers)
    if c in TRANSFER_CHIPS:
        # Rebuilds are free; transfer count does not spend FT or generate hits
        n_tx_ft = 0
        hits = 0
    else:
        n_tx_ft = n_tx
        hits = hit_points_for(state.held.ft, n_tx_ft, chip=c)

    ft_next = ft_after_deadline(state.held.ft, n_tx_ft, chip=c)

    if c == "free_hit":
        next_held = held_copy(state.held, ft=ft_next)
    else:
        next_held = held_copy(commit_held, ft=ft_next)

    next_wallet = play_chip(state.wallet, int(gw), c)
    return DeadlineTransition(
        chip=c,
        commit_held=held_copy(commit_held, ft=state.held.ft),
        next_state=PlannerState(held=next_held, wallet=next_wallet),
        n_transfers=n_tx,
        hit_points=int(hits),
        ft_after=int(ft_next),
    )


def wallet_from_played(played: Mapping[int, str] | None) -> ChipWallet:
    """Replay a validated week→chip map into a fresh wallet."""
    plan = validate_chip_map(played)
    wallet = ChipWallet()
    for gw in sorted(plan):
        wallet.play(gw, plan[gw])
    return wallet


def ft_after_deadline(
    ft_before: int,
    n_transfers: int,
    *,
    chip: str | None,
) -> int:
    """Next-deadline FT bank. ``chip=None`` or ``none`` = ordinary transfers."""
    c = None if chip in (None, NO_CHIP) else str(chip)
    return advance_ft(int(ft_before), int(n_transfers), chip=c)


def hit_points_for(
    ft_before: int,
    n_transfers: int,
    *,
    chip: str | None,
) -> int:
    """Hit points for this deadline under optional chip."""
    c = None if chip in (None, NO_CHIP) else str(chip)
    return hit_cost(int(ft_before), int(n_transfers), chip=c)


def captain_factor(chip: str | None) -> int:
    """×3 under TC; else ×2. ``none`` treated as no chip."""
    c = None if chip in (None, NO_CHIP) else str(chip)
    return captain_multiplier(c)


def play_chip(wallet: ChipWallet, gw: int, chip: str) -> ChipWallet:
    """Return a copy of ``wallet`` after playing ``chip`` in ``gw``."""
    if chip == NO_CHIP:
        return wallet
    out = ChipWallet()
    out.used = set(wallet.used)
    out.played_gw = dict(wallet.played_gw)
    out.play(int(gw), str(chip))
    return out


@dataclass(frozen=True)
class CommitWeekScore:
    """Commit-week expected points under a scoring chip (Step 2)."""

    chip: str
    xi_mu: float
    captain_mu: float
    bench_mu: float
    hit_points: float
    objective: float

    def summary(self) -> dict[str, Any]:
        return {
            "chip": self.chip,
            "xi_mu": float(self.xi_mu),
            "captain_mu": float(self.captain_mu),
            "bench_mu": float(self.bench_mu),
            "hit_points": float(self.hit_points),
            "objective": float(self.objective),
        }


def bench_from_squad(squad: pd.DataFrame, xi: pd.DataFrame) -> pd.DataFrame:
    """Four non-XI rows from a 15-row squad."""
    xi_ids = set(xi["player_id"].astype(str))
    bench = squad.loc[~squad["player_id"].astype(str).isin(xi_ids)].copy()
    if len(bench) != 4:
        raise ValueError(f"bench must have 4 players, got {len(bench)}")
    return bench.reset_index(drop=True)


def score_commit_week(
    *,
    xi: pd.DataFrame,
    captain_id: str,
    bench: pd.DataFrame | None = None,
    chip: str | None = NO_CHIP,
    hit_points: float = 0.0,
    mu_col: str = "mu",
) -> CommitWeekScore:
    """Commit-week value under ``none`` / ``triple_captain`` / ``bench_boost``.

    Ordinary L3/L4 commit piece is ``xi_mu + captain_mu - hits`` (captain ×2).
    TC adds one more captain copy (×3). BB adds Σ bench μ. WC/FH do not change
    this scorer — they change the squad before scoring (Step 3).
    """
    c = NO_CHIP if chip in (None, NO_CHIP) else str(chip)
    if c not in SCORING_CHIPS:
        raise ValueError(f"score_commit_week expects scoring chip, got {c!r}")
    if mu_col not in xi.columns:
        raise ValueError(f"XI missing μ column {mu_col!r}")

    xi_mu = float(pd.to_numeric(xi[mu_col], errors="coerce").fillna(0.0).sum())
    cap_rows = xi.loc[xi["player_id"].astype(str) == str(captain_id)]
    if cap_rows.empty:
        raise ValueError(f"captain_id {captain_id!r} not in XI")
    captain_mu = float(pd.to_numeric(cap_rows.iloc[0][mu_col], errors="coerce") or 0.0)

    bench_mu = 0.0
    if c == "bench_boost":
        if bench is None or bench.empty:
            raise ValueError("bench_boost requires a 4-player bench frame")
        if mu_col not in bench.columns:
            raise ValueError(f"bench missing μ column {mu_col!r}")
        if len(bench) != 4:
            raise ValueError(f"bench must have 4 players, got {len(bench)}")
        bench_mu = float(pd.to_numeric(bench[mu_col], errors="coerce").fillna(0.0).sum())

    # none/BB: +1× captain on top of XI; TC: +2× captain on top of XI
    cap_extra = 2.0 * captain_mu if c == "triple_captain" else captain_mu
    objective = xi_mu + cap_extra + bench_mu - float(hit_points)
    return CommitWeekScore(
        chip=c,
        xi_mu=xi_mu,
        captain_mu=captain_mu,
        bench_mu=bench_mu,
        hit_points=float(hit_points),
        objective=float(objective),
    )


def audit_legality_cases() -> list[dict[str, Any]]:
    """Six fixed legality checks (no MILP, no data load)."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(ok), "detail": detail})

    # 1. GW1: WC/FH banned; TC/BB ok
    w0 = ChipWallet()
    a1 = w0.available(1)
    add(
        "gw1_no_wc_fh",
        "wildcard" not in a1 and "free_hit" not in a1,
        f"available={a1}",
    )
    add(
        "gw1_tc_bb_ok",
        "triple_captain" in a1 and "bench_boost" in a1,
        f"available={a1}",
    )

    # 2. One chip per week; second play same GW empty
    w1 = ChipWallet()
    w1.play(5, "triple_captain")
    add("one_chip_per_week", w1.available(5) == (), f"after TC available={w1.available(5)}")

    # 3. H1 chip not reusable in H1; H2 copy still free
    add(
        "h1_tc_spent",
        "triple_captain" not in w1.available(10),
        f"GW10={w1.available(10)}",
    )
    add(
        "h2_tc_fresh",
        "triple_captain" in w1.available(20),
        f"GW20={w1.available(20)}",
    )

    # 4. Unused H1 chips expire — cannot play H1 WC after GW19
    w2 = ChipWallet()
    # Never played H1 wildcard; GW20 should still offer H2 wildcard only once
    # (ChipWallet treats H2 as a separate copy — H1 unused is simply gone.)
    a20 = w2.available(20)
    add(
        "h2_has_full_set",
        set(a20) == set(CHIPS),
        f"GW20={a20}",
    )
    # Playing WC in H2 uses H2 copy; H1 was never used — still only one H2 play
    w2.play(20, "wildcard")
    add(
        "h2_wc_once",
        "wildcard" not in w2.available(25),
        f"GW25={w2.available(25)}",
    )

    # 5. Free Hit adjacency
    w3 = ChipWallet()
    w3.play(19, "free_hit")
    add(
        "fh_adjacent_blocked",
        "free_hit" not in w3.available(20),
        f"GW20 after H1 FH={w3.available(20)}",
    )
    # H2 FH still available later
    add(
        "fh_h2_later_ok",
        "free_hit" in w3.available(21),
        f"GW21={w3.available(21)}",
    )

    # 6. hit / FT identity under transfer chips
    add(
        "wc_zero_hits",
        hit_points_for(1, 5, chip="wildcard") == 0,
        f"hits={hit_points_for(1, 5, chip='wildcard')}",
    )
    add(
        "fh_ft_preserved",
        ft_after_deadline(2, 4, chip="free_hit") == 2,
        f"ft_after={ft_after_deadline(2, 4, chip='free_hit')}",
    )
    add(
        "none_ordinary_hit",
        hit_points_for(1, 3, chip=NO_CHIP) == 8,
        f"hits={hit_points_for(1, 3, chip=NO_CHIP)}",
    )
    add(
        "tc_captain_x3",
        captain_factor("triple_captain") == 3 and captain_factor(NO_CHIP) == 2,
        f"tc={captain_factor('triple_captain')} none={captain_factor(NO_CHIP)}",
    )

    # Actions helper
    st = ChipDecisionState(
        held=HeldState(
            player_ids=tuple(f"p{i}" for i in range(15)),
            purchase={f"p{i}": 50 for i in range(15)},
            bank=100,
            ft=1,
        )
    )
    acts = st.actions(8)
    add(
        "actions_prefix_none",
        acts[0] == NO_CHIP and set(acts[1:]) == set(st.available_chips(8)),
        f"actions={acts}",
    )

    # validate_chip_map round-trip
    try:
        validate_chip_map({6: "wildcard", 12: "bench_boost", 20: "free_hit"})
        map_ok = True
        map_detail = "ok"
    except ValueError as exc:
        map_ok = False
        map_detail = str(exc)
    add("validate_legal_map", map_ok, map_detail)

    # --- Step 2: TC/BB scoring identities on a toy XI+bench ---
    xi = pd.DataFrame(
        {
            "player_id": [f"x{i}" for i in range(11)],
            "mu": [2.0] * 10 + [5.0],
        }
    )
    bench = pd.DataFrame(
        {
            "player_id": [f"b{i}" for i in range(4)],
            "mu": [1.0, 1.5, 0.5, 2.0],
        }
    )
    base = score_commit_week(xi=xi, captain_id="x10", chip=NO_CHIP, hit_points=0.0)
    # xi_mu = 2*10+5=25; + captain 5 → 30
    add("score_none_identity", abs(base.objective - 30.0) < 1e-9, f"obj={base.objective}")
    tc = score_commit_week(xi=xi, captain_id="x10", chip="triple_captain", hit_points=0.0)
    add(
        "score_tc_ge_none",
        tc.objective >= base.objective - 1e-12 and abs(tc.objective - 35.0) < 1e-9,
        f"tc={tc.objective} none={base.objective}",
    )
    bb = score_commit_week(
        xi=xi, captain_id="x10", bench=bench, chip="bench_boost", hit_points=0.0
    )
    add(
        "score_bb_adds_bench",
        abs(bb.objective - (30.0 + 5.0)) < 1e-9 and abs(bb.bench_mu - 5.0) < 1e-9,
        f"bb={bb.objective} bench_mu={bb.bench_mu}",
    )
    hit = score_commit_week(xi=xi, captain_id="x10", chip=NO_CHIP, hit_points=4.0)
    add("score_hits_subtract", abs(hit.objective - 26.0) < 1e-9, f"obj={hit.objective}")

    # --- Step 3: WC / FH state machine ---
    before = HeldState(
        tuple(f"a{i}" for i in range(15)),
        {f"a{i}": 50 for i in range(15)},
        bank=50,
        ft=2,
    )
    rebuilt = HeldState(
        tuple(f"b{i}" for i in range(15)),
        {f"b{i}": 50 for i in range(15)},
        bank=40,
        ft=2,
    )
    st0 = PlannerState(held=before, wallet=ChipWallet())

    fh = advance_planner_state(
        st0, gw=8, chip="free_hit", commit_held=rebuilt, n_transfers=10
    )
    add(
        "fh_commit_is_rebuild",
        fh.commit_held.player_ids == rebuilt.player_ids,
        f"commit={fh.commit_held.player_ids[:2]}…",
    )
    add(
        "fh_next_reverts",
        fh.next_state.held.player_ids == before.player_ids
        and fh.next_state.held.bank == before.bank,
        f"next_ids0={fh.next_state.held.player_ids[0]} bank={fh.next_state.held.bank}",
    )
    add(
        "fh_ft_and_hits",
        fh.ft_after == 2 and fh.hit_points == 0,
        f"ft={fh.ft_after} hits={fh.hit_points}",
    )
    add(
        "fh_wallet_spent",
        "free_hit" not in fh.next_state.wallet.available(10),
        f"avail={fh.next_state.wallet.available(10)}",
    )

    st1 = PlannerState(held=before, wallet=ChipWallet())
    wc = advance_planner_state(
        st1, gw=8, chip="wildcard", commit_held=rebuilt, n_transfers=10
    )
    add(
        "wc_next_keeps_rebuild",
        wc.next_state.held.player_ids == rebuilt.player_ids
        and wc.next_state.held.bank == rebuilt.bank,
        f"bank={wc.next_state.held.bank}",
    )
    add(
        "wc_ft_preserved",
        wc.ft_after == 2 and wc.hit_points == 0,
        f"ft={wc.ft_after} hits={wc.hit_points}",
    )

    # Ordinary transfer: 1 FT used, ft_after = min(5, 0+1) = 1 from ft_before=1
    st2 = PlannerState(held=held_copy(before, ft=1), wallet=ChipWallet())
    moved_ids = tuple(["b0"] + [f"a{i}" for i in range(1, 15)])
    moved = HeldState(moved_ids, {pid: 50 for pid in moved_ids}, bank=50, ft=1)
    ord_ = advance_planner_state(
        st2, gw=8, chip=NO_CHIP, commit_held=moved, n_transfers=1
    )
    add(
        "none_keeps_moves",
        ord_.next_state.held.player_ids == moved.player_ids,
        f"ids0={ord_.next_state.held.player_ids[0]}",
    )
    add(
        "none_ft_rolls",
        ord_.ft_after == 1 and ord_.hit_points == 0,
        f"ft={ord_.ft_after} hits={ord_.hit_points}",
    )

    return checks


def summarise_audit(checks: list[dict[str, Any]]) -> dict[str, Any]:
    n_pass = sum(1 for c in checks if c["pass"])
    return {
        "n_checks": len(checks),
        "n_pass": n_pass,
        "passed": n_pass == len(checks),
        "checks": checks,
        "defs": {
            "chips": list(CHIPS),
            "free_transfer_chips": sorted(FREE_TRANSFER_CHIPS),
            "margins": "report-only; no play thresholds",
            "inner_l4": "priced mu_post; gamma=1; unconstrained hits",
            "step": 3,
            "step_2": "TC/BB commit-week scoring identity",
            "step_3": "WC keeps rebuild; FH reverts; FT preserved under both",
        },
    }


__all__ = [
    "NO_CHIP",
    "SCORING_CHIPS",
    "TRANSFER_CHIPS",
    "ChipDecisionState",
    "CommitWeekScore",
    "DeadlineTransition",
    "PlannerState",
    "advance_planner_state",
    "audit_legality_cases",
    "bench_from_squad",
    "captain_factor",
    "ft_after_deadline",
    "held_copy",
    "held_from_squad_frame",
    "hit_points_for",
    "play_chip",
    "score_commit_week",
    "summarise_audit",
    "wallet_from_played",
]
