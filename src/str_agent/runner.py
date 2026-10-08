"""LLM call → validate → optional repair → freeze.

No live model call in unit tests. Pass ``call_model`` for a real run.
Does not touch ``score_xp``, pots, or MILP.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from src.str_agent import freeze as fz
from src.str_agent import prompt as pr
from src.str_agent import validator as val

CallModel = Callable[[str, str], str]

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object from model text (fence or bare)."""
    match = _JSON_FENCE.search(text)
    blob = match.group(1) if match else text
    start = blob.find("{")
    end = blob.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(blob[start : end + 1])


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assemble_freeze_row(
    *,
    gw: int,
    deadline_utc: str,
    frozen_at_utc: str,
    model_id: str,
    system_prompt: str,
    context_markdown: str,
    model_payload: Mapping[str, Any],
    directory: Mapping[str, Mapping[str, Any]],
    budget: int,
) -> dict[str, Any]:
    """Build a freeze-ready row from model JSON + accounting."""
    decision = dict(model_payload.get("decision") or {})
    squad = [str(pid) for pid in decision.get("squad_15") or []]
    total_cost = sum(int(directory[pid]["now_cost"]) for pid in squad if pid in directory)
    bank = int(budget) - total_cost
    errors = val.validate_draft(
        {"decision": decision},
        directory=directory,
        budget=budget,
    )
    return {
        "gw": int(gw),
        "deadline_utc": deadline_utc,
        "frozen_at_utc": frozen_at_utc,
        "provenance": {
            "model_id": model_id,
            "prompt_sha256": sha256_text(system_prompt),
            "context_sha256": sha256_text(context_markdown),
        },
        "rationale": str(model_payload.get("rationale") or ""),
        "decision": decision,
        "accounting": {
            "bank_remaining": bank,
            "total_cost": total_cost,
            "is_legal": not errors,
            "errors": errors,
        },
        "realised": fz.empty_realised(),
    }


def run_once(
    *,
    gw: int,
    deadline_utc: str,
    frozen_at_utc: str,
    context_markdown: str,
    directory: Mapping[str, Mapping[str, Any]],
    budget: int,
    model_id: str,
    call_model: CallModel,
    max_repairs: int = 2,
    ledger: Path | None = None,
    commit: bool = False,
) -> dict[str, Any]:
    """Call the model, repair illegal drafts, optionally freeze.

    Returns the freeze row (``accounting.is_legal`` may be false if repairs
    exhausted). Raises ``fz.StringFreezeError`` only when ``commit`` is true
    and the row fails freeze rules.
    """
    system = pr.SYSTEM
    user = pr.wrap_user_context(context_markdown)
    last_errors: list[str] = []
    row: dict[str, Any] | None = None
    for attempt in range(max_repairs + 1):
        if attempt == 0:
            raw = call_model(system, user)
        else:
            repair_user = (
                user
                + "\n\nPrevious draft failed legality:\n- "
                + "\n- ".join(last_errors)
                + "\n\nReturn a corrected JSON draft only."
            )
            raw = call_model(system, repair_user)
        payload = extract_json_object(raw)
        row = assemble_freeze_row(
            gw=gw,
            deadline_utc=deadline_utc,
            frozen_at_utc=frozen_at_utc,
            model_id=model_id,
            system_prompt=system,
            context_markdown=context_markdown,
            model_payload=payload,
            directory=directory,
            budget=budget,
        )
        last_errors = list(row["accounting"]["errors"])
        if not last_errors:
            break
    assert row is not None
    if commit:
        if not row["accounting"]["is_legal"]:
            raise fz.StringFreezeError(
                "illegal after repairs: " + "; ".join(last_errors)
            )
        # Drop errors list before ledger write (schema is accounting.is_legal).
        clean = dict(row)
        clean["accounting"] = {
            "bank_remaining": row["accounting"]["bank_remaining"],
            "total_cost": row["accounting"]["total_cost"],
            "is_legal": True,
        }
        fz.write_string_freeze(clean, path=ledger)
        return clean
    return row
