"""Append-only freeze ledger for the string competitor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.live.news_tags import published_before

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "predictions" / "2026-27" / "string_agent_freeze.jsonl"

REQUIRED = (
    "gw",
    "deadline_utc",
    "frozen_at_utc",
    "provenance",
    "rationale",
    "decision",
    "accounting",
    "realised",
)


class StringFreezeError(RuntimeError):
    """String-agent freeze rule broken."""


def empty_realised() -> dict[str, None]:
    return {"actual_points": None, "evaluated_at_utc": None}


def validate_freeze_row(row: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED if key not in row]
    if missing:
        raise StringFreezeError(f"missing keys: {', '.join(missing)}")
    if not published_before(str(row["frozen_at_utc"]), str(row["deadline_utc"])):
        # allow equality failure: frozen must be strictly before deadline
        raise StringFreezeError("frozen_at_utc must be before deadline_utc")
    decision = row["decision"]
    for key in ("squad_15", "starting_11", "captain", "vice_captain", "bench_order"):
        if key not in decision:
            raise StringFreezeError(f"decision missing {key}")
    accounting = row["accounting"]
    if accounting.get("is_legal") is not True:
        raise StringFreezeError("accounting.is_legal must be true to freeze")
    prov = row["provenance"]
    for key in ("model_id", "prompt_sha256", "context_sha256"):
        if key not in prov:
            raise StringFreezeError(f"provenance missing {key}")


def write_string_freeze(row: Mapping[str, Any], path: Path | None = None) -> Path:
    """Append one legal, pre-deadline freeze row."""
    payload = dict(row)
    if "realised" not in payload:
        payload["realised"] = empty_realised()
    validate_freeze_row(payload)
    ledger = Path(path) if path is not None else LEDGER
    ledger.parent.mkdir(parents=True, exist_ok=True)
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            existing = json.loads(line)
            if int(existing["gw"]) == int(payload["gw"]):
                raise StringFreezeError(f"GW{payload['gw']} already frozen")
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return ledger
