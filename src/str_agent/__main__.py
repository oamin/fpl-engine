"""Dry-run a hold, or save a three-week plan. Does not call a model.

    python3 -m src.str_agent --entry 2632584 --hold
    python3 -m src.str_agent --entry 2632584 --save-plan draft.json
    python3 -m src.str_agent --entry 2632584 --save-plan draft.json --commit-saved

``--commit`` stays refused. ``--commit-saved`` is the pre-deadline append,
and only together with ``--save-plan``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.live.entry import load_entry
from src.live.news_tags import published_before
from src.str_agent.carry import (
    carry_from_entry,
    directory_from_entry,
    hold_draft_from_entry,
    validate_move,
)
from src.str_agent.freeze import LEDGER, StringFreezeError, commit_official
from src.str_agent.horizon import HorizonError, PLAN_DIR
from src.str_agent.prompt import SYSTEM
from src.str_agent.runner import assemble_freeze_row
from src.str_agent.trial import prepare, save_submitted_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="String-agent dry run")
    parser.add_argument("--entry", type=int, required=True)
    parser.add_argument("--hold", action="store_true", help="Keep the current fifteen")
    parser.add_argument("--save-plan", type=str, default="", help="JSON with rationale, decision, and horizon")
    parser.add_argument("--plan-root", type=str, default="")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument(
        "--commit-saved",
        action="store_true",
        help="After a legal --save-plan, append the imminent week to the official ledger",
    )
    parser.add_argument("--ledger", type=str, default="")
    args = parser.parse_args(argv)
    ledger = Path(args.ledger).resolve() if args.ledger else None
    if args.commit or (ledger == LEDGER.resolve() and not args.commit_saved):
        print(
            "official string_agent_freeze.jsonl is closed until the deadline commit",
            file=sys.stderr,
        )
        return 2
    if args.commit_saved and not args.save_plan:
        print("--commit-saved needs --save-plan", file=sys.stderr)
        return 2
    if args.save_plan:
        plan_root = Path(args.plan_root) if args.plan_root else PLAN_DIR
        payload = json.loads(Path(args.save_plan).read_text(encoding="utf-8"))
        pack = prepare(int(args.entry), plan_root=plan_root)
        frozen = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if not published_before(frozen, pack.deadline_utc):
            print("deadline has passed", file=sys.stderr)
            return 2
        try:
            path, digest = save_submitted_plan(
                pack,
                payload,
                frozen_at_utc=frozen,
                model_id=str(payload.get("model_id") or "file"),
                root=plan_root,
            )
        except (HorizonError, KeyError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        committed = None
        if args.commit_saved:
            row = assemble_freeze_row(
                gw=pack.carry.gw,
                deadline_utc=pack.deadline_utc,
                frozen_at_utc=frozen,
                model_id=str(payload.get("model_id") or "file"),
                system_prompt=SYSTEM,
                context_markdown=pack.context,
                model_payload=payload,
                directory=pack.directory,
                budget=1000,
                carry=pack.carry,
            )
            if not row["accounting"]["is_legal"]:
                print("; ".join(row["accounting"]["errors"]), file=sys.stderr)
                return 1
            clean = dict(row)
            clean["accounting"] = {
                "bank_remaining": row["accounting"]["bank_remaining"],
                "total_cost": row["accounting"]["total_cost"],
                "is_legal": True,
            }
            if "hits" in row["accounting"]:
                clean["accounting"]["hits"] = row["accounting"]["hits"]
            if "bank_after" in row["accounting"]:
                clean["accounting"]["bank_after"] = row["accounting"]["bank_after"]
            try:
                committed = str(commit_official(clean, plan_sha256=digest, path=ledger))
            except StringFreezeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
        json.dump(
            {
                "plan": str(path),
                "plan_sha256": digest,
                "gw": pack.carry.gw,
                "committed": committed,
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 0
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
