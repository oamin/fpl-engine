"""One live stragent run for the next deadline.

The order is fixed. From Gameweek 7 the saved plan for the previous week has to be on disk
before any feed is polled. The dossier is then filled from the named feeds.
The prompt is that dossier plus the notebook rows from earlier gameweeks,
and the starting squad is that plan's ``carry_after``. The process stops
until ``--reply`` supplies the agent's JSON. ``--lock`` appends week 0 of a
legal plan to the string ledger, and only while the clock is before the
deadline.

This module does not call a model. A banned score token, an illegal horizon,
or a missing prior plan writes no plan and no notebook row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.live.model_registry import RegistryError, record_for_lock, record_for_session
from src.live.news_tags import published_before
from src.str_agent.freeze import DuplicateFreezeError, commit_official
from src.str_agent.horizon import (
    FIRST_PAPER_GW,
    HorizonError,
    MissingPriorPlanError,
    plan_path,
)
from src.str_agent.notebook import NotebookError, notebook_section
from src.str_agent.prompt import SYSTEM
from src.str_agent.runner import assemble_freeze_row, sha256_text
from src.str_agent.trial import TrialPack, prepare, save_submitted_plan
from src.str_agent.trawl import trawl

ROOT = Path(__file__).resolve().parents[2]


class SessionError(RuntimeError):
    """The string session stopped before a ledger write."""


TrawlFn = Callable[..., Any]
PrepareFn = Callable[..., TrialPack]


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gw_folder(gw: int, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else ROOT
    return base / "data" / "predictions" / "2026-27" / f"gw{int(gw):02d}"


def manifest_path(gw: int, root: Path | None = None) -> Path:
    return gw_folder(gw, root) / "session_stragent.json"


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SessionError(f"{path} is not an object")
    return payload


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    existing = _read_manifest(path)
    if existing is not None and existing.get("locked") is True:
        raise SessionError(f"locked manifest is not overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_prior_plan(gw: int, plan_root: Path | None) -> None:
    """Gameweek 7 and later start from the previous plan. The entry is not a substitute."""
    if int(gw) <= int(FIRST_PAPER_GW):
        return
    path = plan_path(int(gw) - 1, plan_root)
    if not path.is_file():
        raise MissingPriorPlanError(f"GW{int(gw)} has no string plan for GW{int(gw) - 1}")


def _ledger_has_gw(path: Path, gw: int) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and int(row.get("gw") or 0) == int(gw):
            return True
    return False


def _find_slate(gw: int, root: Path | None) -> tuple[Path, Path]:
    folder = gw_folder(gw, root)
    bootstrap = None
    fixtures = None
    for name in ("fpl_t1", "betfair_t1"):
        boot = folder / name / "bootstrap.json"
        fix = folder / name / "fixtures.json"
        if bootstrap is None and boot.is_file():
            bootstrap = boot
        if fixtures is None and fix.is_file():
            fixtures = fix
    if bootstrap is None or fixtures is None:
        raise SessionError("bootstrap and fixtures must sit in the gameweek folder")
    live = (ROOT / "data" / "live").resolve()
    for path in (bootstrap, fixtures):
        resolved = path.resolve()
        if resolved == live or live in resolved.parents:
            raise SessionError("the string session does not read data/live")
    return bootstrap, fixtures


def apply_reply(
    pack: TrialPack,
    payload: Mapping[str, Any],
    *,
    frozen_at_utc: str,
    plan_root: Path,
    lock: bool = False,
    ledger: Path | None = None,
    model_id: str | None = None,
    code_root: Path | None = None,
    registry: Path | None = None,
) -> dict[str, Any]:
    """Validate the three-week reply. An illegal reply writes no plan and no note.

    ``--lock`` commits week 0 only, and only before the deadline. A second
    row for the gameweek raises ``DuplicateFreezeError`` before the plan is
    rewritten. A model tree that is not the current iteration writes no plan
    and no ledger row.
    """
    if not published_before(frozen_at_utc, pack.deadline_utc):
        raise SessionError("frozen_at is not before the deadline")
    if lock and ledger is not None and _ledger_has_gw(ledger, pack.carry.gw):
        raise DuplicateFreezeError(f"GW{pack.carry.gw} already frozen")
    model_name = model_id or str(payload.get("model_id") or "file")
    iteration = None
    if lock:
        try:
            iteration = record_for_lock("stragent", root=code_root, registry=registry)
        except RegistryError as exc:
            raise SessionError(str(exc)) from exc
        iteration["model_id"] = model_name
        iteration["prompt_sha256"] = sha256_text(SYSTEM)
    try:
        path, digest = save_submitted_plan(
            pack,
            payload,
            frozen_at_utc=frozen_at_utc,
            model_id=model_name,
            root=plan_root,
        )
    except (HorizonError, NotebookError, KeyError, ValueError) as exc:
        raise SessionError(str(exc)) from exc
    committed = None
    if lock:
        row = assemble_freeze_row(
            gw=int(pack.carry.gw),
            deadline_utc=pack.deadline_utc,
            frozen_at_utc=frozen_at_utc,
            model_id=model_name,
            system_prompt=SYSTEM,
            context_markdown=pack.context,
            model_payload=payload,
            directory=pack.directory,
            budget=1000,
            carry=pack.carry,
        )
        if not row["accounting"]["is_legal"]:
            raise SessionError("; ".join(row["accounting"]["errors"]))
        if iteration is None:
            raise SessionError("a lock has no model iteration")
        row["provenance"]["model_iteration_id"] = iteration["iteration_id"]
        row["provenance"]["model_iteration_commit"] = iteration["iteration_commit"]
        row["provenance"]["model_locked_commit"] = iteration["locked_commit"]
        row["provenance"]["model_source_sha256"] = iteration["source_sha256"]
        clean = dict(row)
        clean["accounting"] = {
            "bank_remaining": row["accounting"]["bank_remaining"],
            "total_cost": row["accounting"]["total_cost"],
            "is_legal": True,
        }
        for key in ("hits", "bank_after"):
            if key in row["accounting"]:
                clean["accounting"][key] = row["accounting"][key]
        # Week 0 is ``decision``. Later horizon weeks stay on the plan file.
        committed = str(commit_official(clean, plan_sha256=digest, path=ledger))
    return {
        "plan": str(path),
        "plan_sha256": digest,
        "committed": committed,
        "gw": int(pack.carry.gw),
        "model": iteration,
    }


def run_stragent(
    gw: int,
    *,
    deadline_utc: str,
    entry_id: int = 2632584,
    root: Path | None = None,
    plan_root: Path,
    reply: Mapping[str, Any] | None = None,
    lock: bool = False,
    frozen_at_utc: str | None = None,
    ledger: Path | None = None,
    trawl_fn: TrawlFn | None = None,
    prepare_fn: PrepareFn | None = None,
    observed_at_utc: str | None = None,
    code_root: Path | None = None,
    registry: Path | None = None,
) -> dict[str, Any]:
    """Trawl, build the prompt, and stop until a reply is passed in.

    Returns a manifest payload. Exit handling lives in ``main``: no reply
    is exit 2, and the manifest records ``locked`` false.
    """
    manifest = manifest_path(gw, root)
    existing = _read_manifest(manifest)
    if existing is not None and existing.get("locked") is True:
        raise SessionError(f"locked manifest is not overwritten: {manifest}")
    require_prior_plan(gw, plan_root)
    if lock:
        try:
            record_for_lock("stragent", root=code_root, registry=registry)
        except RegistryError as exc:
            raise SessionError(str(exc)) from exc
    observed = observed_at_utc or utc_now()
    if trawl_fn is None:
        trawl_result = trawl(int(gw), deadline_utc=deadline_utc, observed_at_utc=observed, root=root)
    else:
        trawl_result = trawl_fn(int(gw), deadline_utc=deadline_utc, observed_at_utc=observed, root=root)
    steps = [
        {
            "name": "trawl",
            "status": "ok",
            "written": int(getattr(trawl_result, "written", 0)),
        }
    ]
    if prepare_fn is None:
        bootstrap_path, fixtures_path = _find_slate(gw, root)
        pack = prepare(
            entry_id=int(entry_id),
            deadline_utc=deadline_utc,
            bootstrap_path=bootstrap_path,
            fixtures_path=fixtures_path,
            plan_root=plan_root,
            next_gw=int(gw),
        )
    else:
        pack = prepare_fn(
            entry_id=int(entry_id),
            deadline_utc=deadline_utc,
            plan_root=plan_root,
            next_gw=int(gw),
        )
    if int(pack.carry.gw) != int(gw):
        raise SessionError(f"context is for GW{pack.carry.gw}, not GW{int(gw)}")
    # The notebook section already drops this week and later weeks.
    prior = notebook_section(int(gw), plan_root)
    context_file = plan_root / f"gw{int(gw):02d}_context.md"
    context_file.parent.mkdir(parents=True, exist_ok=True)
    context_file.write_text(pack.context, encoding="utf-8")
    context_sha = hashlib.sha256(pack.context.encode("utf-8")).hexdigest()
    steps.append({"name": "context", "status": "ok", "prior_notes": bool(prior.strip())})
    frozen = frozen_at_utc or utc_now()
    locked = False
    plan_sha = None
    if reply is None:
        steps.append({"name": "reply", "status": "waiting"})
        payload = _manifest(
            gw,
            steps,
            locked=False,
            plan_sha=None,
            context_sha=context_sha,
            model=_session_model(code_root=code_root, registry=registry, model_id=None),
        )
        _write_manifest(manifest, payload)
        payload["exit_code"] = 2
        return payload
    result = apply_reply(
        pack,
        reply,
        frozen_at_utc=frozen,
        plan_root=plan_root,
        lock=lock,
        ledger=ledger,
        code_root=code_root,
        registry=registry,
    )
    plan_sha = result["plan_sha256"]
    steps.append({"name": "reply", "status": "ok"})
    if lock:
        locked = True
        model = result["model"]
        steps.append({"name": "lock", "status": "locked", "gw": int(pack.carry.gw)})
    else:
        named = str(reply.get("model_id") or "file")
        model = _session_model(code_root=code_root, registry=registry, model_id=named)
        steps.append({"name": "lock", "status": "skipped"})
    payload = _manifest(
        gw,
        steps,
        locked=locked,
        plan_sha=plan_sha,
        context_sha=context_sha,
        model=model,
    )
    _write_manifest(manifest, payload)
    payload["exit_code"] = 0
    payload["plan"] = result["plan"]
    return payload


def _session_model(
    *,
    code_root: Path | None,
    registry: Path | None,
    model_id: str | None,
) -> dict[str, Any]:
    try:
        record = record_for_session("stragent", root=code_root, registry=registry)
    except RegistryError as exc:
        raise SessionError(str(exc)) from exc
    record["model_id"] = model_id
    record["prompt_sha256"] = sha256_text(SYSTEM)
    return record


def _manifest(
    gw: int,
    steps: list[dict[str, Any]],
    *,
    locked: bool,
    plan_sha: str | None,
    context_sha: str | None,
    model: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "gw": int(gw),
        "steps": steps,
        "hashes": {"plan_sha256": plan_sha, "context_sha256": context_sha},
        "locked": locked,
        "model": dict(model),
        "realised": None,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Stragent live session")
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--deadline", type=str, required=True)
    parser.add_argument("--entry", type=int, default=2632584)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--plan-root", type=str, default="")
    parser.add_argument("--reply", type=str, default="")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--frozen-at", type=str, default="")
    parser.add_argument("--ledger", type=str, default="")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else None
    plan_root = Path(args.plan_root) if args.plan_root else (ROOT / "data" / "predictions" / "2026-27" / "string_plans")
    ledger = Path(args.ledger) if args.ledger else None
    reply = None
    if args.reply:
        reply = json.loads(Path(args.reply).read_text(encoding="utf-8"))
    try:
        payload = run_stragent(
            int(args.gw),
            deadline_utc=args.deadline,
            entry_id=int(args.entry),
            root=root,
            plan_root=plan_root,
            reply=reply,
            lock=bool(args.lock),
            frozen_at_utc=args.frozen_at or None,
            ledger=ledger,
        )
    except (SessionError, MissingPriorPlanError, DuplicateFreezeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    code = int(payload.get("exit_code") or 0)
    json.dump({"gw": payload["gw"], "locked": payload["locked"], "exit_code": code}, sys.stdout)
    sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
