"""Dry-run a hold of the current entry. Does not call a model or write the ledger.

    python -m src.str_agent --entry 2632584 --hold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.live.entry import load_entry
from src.str_agent.carry import (
    carry_from_entry,
    directory_from_entry,
    hold_draft_from_entry,
    validate_move,
)
from src.str_agent.freeze import LEDGER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="String-agent dry run")
    parser.add_argument("--entry", type=int, required=True)
    parser.add_argument("--hold", action="store_true", help="Keep the current fifteen")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--ledger", type=str, default="")
    args = parser.parse_args(argv)
    ledger = Path(args.ledger).resolve() if args.ledger else None
    if args.commit or ledger == LEDGER.resolve():
        print(
            "official string_agent_freeze.jsonl is closed until the deadline commit",
            file=sys.stderr,
        )
        return 2
    if not args.hold:
        print("this slice only dry-runs --hold", file=sys.stderr)
        return 2
    entry = load_entry(int(args.entry))
    carry = carry_from_entry(entry)
    draft = hold_draft_from_entry(entry)
    move = validate_move(draft, carry, directory_from_entry(entry))
    payload = {
        "entry_id": int(entry["entry_id"]),
        "team_name": entry.get("team_name") or "",
        "gw": carry.gw,
        "hits": move.hits,
        "bank_after": move.bank_after,
        "is_legal": not move.errors,
        "errors": move.errors,
        "decision": draft["decision"],
        "carry": carry.as_dict(),
        "carry_after": move.carry_after,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if payload["is_legal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
