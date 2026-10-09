"""A three-week plan for the string agent, and the paper team that follows it.

The imminent week is the decision. The next two weeks are intentions at
today's prices. The next deadline starts from the carry after the imminent
week, not from the live FPL entry. ``score_xp`` is not stored or shown.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.str_agent.carry import CarryState, validate_move
from src.str_agent.notebook import upsert_note, validate_note

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "data" / "predictions" / "2026-27" / "string_plans"
HORIZON_WEEKS = 3
FIRST_PAPER_GW = 6
_FIELDS = (
    "chip_played",
    "squad_15",
    "starting_11",
    "captain",
    "vice_captain",
    "bench_order",
    "transfers_in",
    "transfers_out",
)
_BANNED = ("score_xp", "xp_on_pot", "lam_scored")


class HorizonError(ValueError):
    """The horizon cannot be saved."""


class MissingPriorPlanError(HorizonError):
    """An in-season week has no saved plan to continue from."""


@dataclass(frozen=True)
class HorizonStep:
    gw: int
    hits: int
    bank_after: int | None
    carry_after: dict[str, Any]


@dataclass(frozen=True)
class HorizonResult:
    errors: list[str]
    steps: tuple[HorizonStep, ...]


def plan_path(gw: int, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else PLAN_DIR
    return base / f"gw{int(gw):02d}.json"


def context_path(gw: int, root: Path | None = None) -> Path:
    return plan_path(gw, root).with_name(f"gw{int(gw):02d}_context.md")


def _ids(values: Any) -> list[str]:
    return [str(pid) for pid in values or []]


def _draft(week: Mapping[str, Any]) -> dict[str, Any]:
    return {"decision": {key: week.get(key) for key in _FIELDS}}


def _matches_decision(decision: Mapping[str, Any], week: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(_ids(decision.get("squad_15"))) != set(_ids(week.get("squad_15"))):
        errors.append("horizon week 0 squad does not match the decision")
    if _ids(decision.get("starting_11")) != _ids(week.get("starting_11")):
        errors.append("horizon week 0 starting eleven does not match the decision")
    if str(decision.get("captain") or "") != str(week.get("captain") or ""):
        errors.append("horizon week 0 captain does not match the decision")
    if str(decision.get("vice_captain") or "") != str(week.get("vice_captain") or ""):
        errors.append("horizon week 0 vice does not match the decision")
    if _ids(decision.get("bench_order")) != _ids(week.get("bench_order")):
        errors.append("horizon week 0 bench does not match the decision")
    left_chip = decision.get("chip_played") or None
    right_chip = week.get("chip_played") or None
    if left_chip != right_chip:
        errors.append("horizon week 0 chip does not match the decision")
    if set(_ids(decision.get("transfers_in"))) != set(_ids(week.get("transfers_in"))):
        errors.append("horizon week 0 transfers in do not match the decision")
    if set(_ids(decision.get("transfers_out"))) != set(_ids(week.get("transfers_out"))):
        errors.append("horizon week 0 transfers out do not match the decision")
    return errors


def validate_horizon(
    decision: Mapping[str, Any],
    horizon: list[Mapping[str, Any]] | None,
    carry: CarryState,
    directory: Mapping[Any, Mapping[str, Any]],
) -> HorizonResult:
    """Check three successive legal weeks. Any error drops the whole chain."""
    weeks = list(horizon or [])
    if len(weeks) != HORIZON_WEEKS:
        return HorizonResult([f"horizon must have {HORIZON_WEEKS} weeks"], ())
    errors = _matches_decision(decision, weeks[0])
    state = carry
    steps: list[HorizonStep] = []
    for index, week in enumerate(weeks):
        expected = int(carry.gw) + index
        if int(week.get("gw") or 0) != expected:
            errors.append(f"horizon week {index} must be GW{expected}")
        move = validate_move(_draft(week), state, directory)
        errors.extend(f"GW{expected}: {item}" for item in move.errors)
        if move.carry_after is None:
            return HorizonResult(errors, ())
        steps.append(
            HorizonStep(
                gw=expected,
                hits=int(move.hits),
                bank_after=move.bank_after,
                carry_after=move.carry_after,
            )
        )
        state = CarryState.from_dict(move.carry_after)
    if errors:
        return HorizonResult(errors, ())
    return HorizonResult([], tuple(steps))


def _contaminated(text: str) -> str | None:
    lowered = text.lower()
    for token in _BANNED:
        if token in lowered:
            return token
    return None


def write_plan(
    *,
    gw: int,
    deadline_utc: str,
    frozen_at_utc: str,
    model_id: str,
    prompt_sha256: str,
    context: str,
    rationale: str,
    decision: Mapping[str, Any],
    horizon: list[Mapping[str, Any]],
    carry: CarryState,
    result: HorizonResult,
    root: Path | None = None,
    notes: str = "",
    adjustments: str = "",
) -> tuple[Path, str]:
    """Write the plan and its context. Returns the plan path and its sha256.

    Refuses an illegal horizon and any text that names a score column.
    Does not touch the official string ledger.
    """
    if result.errors or not result.steps:
        raise HorizonError("refusing to save an illegal horizon: " + "; ".join(result.errors))
    manager_notes = str(notes or "")
    manager_adjustments = str(adjustments or "")
    pending_note = None
    if manager_notes.strip() or manager_adjustments.strip():
        pending_note = validate_note(
            gw=int(gw),
            written_at_utc=frozen_at_utc,
            notes=manager_notes,
            adjustments=manager_adjustments,
        )
    payload = {
        "gw": int(gw),
        "deadline_utc": deadline_utc,
        "frozen_at_utc": frozen_at_utc,
        "model_id": model_id,
        "prompt_sha256": prompt_sha256,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "rationale": rationale,
        "decision": dict(decision),
        "horizon": [dict(week) for week in horizon],
        "carry": carry.as_dict(),
        "carry_after": result.steps[0].carry_after,
    }
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    banned = _contaminated(text) or _contaminated(context)
    if banned is not None:
        raise HorizonError(f"plan text contains {banned}")
    path = plan_path(gw, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    context_path(gw, root).write_text(context, encoding="utf-8")
    if pending_note is not None:
        upsert_note(
            gw=int(pending_note["gw"]),
            written_at_utc=str(pending_note["written_at_utc"]),
            notes=str(pending_note["notes"]),
            adjustments=str(pending_note["adjustments"]),
            root=root,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def load_plan(gw: int, root: Path | None = None) -> dict[str, Any]:
    path = plan_path(gw, root)
    if not path.is_file():
        raise MissingPriorPlanError(f"no string plan for GW{gw}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_start(gw: int, entry_carry: CarryState, root: Path | None = None) -> CarryState:
    """Paper squad for this deadline.

    A saved plan for the previous week wins. Gameweek 6 may start from the
    entry. A later week with no plan fails closed.
    """
    previous = plan_path(int(gw) - 1, root)
    if previous.is_file():
        plan = json.loads(previous.read_text(encoding="utf-8"))
        return CarryState.from_dict(plan["carry_after"])
    if int(gw) > FIRST_PAPER_GW:
        raise MissingPriorPlanError(
            f"GW{gw} has no string plan for GW{int(gw) - 1}"
        )
    return entry_carry


def followup_context(plan: Mapping[str, Any]) -> str:
    """The previous plan, for the next deadline's prompt. No score column."""
    owned = plan["carry_after"]
    lines = [
        "# Previous string plan",
        f"Frozen at {plan['frozen_at_utc']} for a deadline of {plan['deadline_utc']}.",
        "The squad below is the one you own now. It is the squad after the week you froze.",
        "The later weeks were intentions at the prices of that deadline. Update them.",
        "",
        f"Gameweek {owned['gw']}. Bank {owned['bank']} tenths. "
        f"Free transfers {owned['ft_before']}.",
        "Squad: " + ", ".join(str(pid) for pid in owned["squad"]) + ".",
        "",
        "## Intentions you wrote",
    ]
    for week in plan.get("horizon") or []:
        chip = week.get("chip_played") or "none"
        lines.append(
            f"- GW{week.get('gw')}: chip {chip}, "
            f"in {', '.join(_ids(week.get('transfers_in'))) or 'nobody'}, "
            f"out {', '.join(_ids(week.get('transfers_out'))) or 'nobody'}, "
            f"captain {week.get('captain')}."
        )
    lines.append("")
    lines.append(str(plan.get("rationale") or ""))
    text = "\n".join(lines) + "\n"
    banned = _contaminated(text)
    if banned is not None:
        raise HorizonError(f"follow-up context contains {banned}")
    return text
