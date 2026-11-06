"""Freeze a Numagent live pick the moment the PI says the team is locked.

**Hard rule for agents:** when the user says the team is locked / submitted /
entered into FPL, run this module **immediately**, then ``git add`` + ``commit``
+ ``push`` the GW artifacts in the same turn. Do not defer.

Writes under ``data/predictions/2026-27/gwNN/``:

- ``numagent_lock.json`` (primary entry) or ``numagent_lock_<entry>.json``
- copies of the week-t / t+1 boards as ``numagent_board_locked.csv`` (+ ``_t1``)
- short report ``reports/numagent_lock_gwNN[_entry].md``

  PYTHONPATH=. python3 -m src.live.numagent_lock --gw 10 --entry 2632584 --arm l3
  PYTHONPATH=. python3 -m src.live.numagent_lock --gw 10 --entry 3271409 --arm l3 \\
      --from-pick data/predictions/2026-27/gw10/numagent_pick_3271409.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "data" / "predictions" / "2026-27"
REPORTS = ROOT / "reports"
PRIMARY_ENTRY = 2632584


def _gw_dir(gw: int) -> Path:
    return PRED / f"gw{int(gw):02d}"


def _resolve_pick_path(gw: int, entry_id: int, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    d = _gw_dir(gw)
    for cand in (
        d / f"numagent_pick_{int(entry_id)}.json",
        d / "numagent_pick.json",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No numagent_pick for GW{gw} entry {entry_id}. Run numagent_live_pick first."
    )


def _plan_from_doc(doc: dict[str, Any], arm: str) -> dict[str, Any]:
    arm_l = arm.lower().strip()
    if arm_l == "recommendation":
        plan = doc.get("recommendation")
    elif arm_l == "l3":
        plan = doc.get("l3")
    elif arm_l == "l4":
        plan = doc.get("l4")
    else:
        raise ValueError("arm must be recommendation|l3|l4")
    if not isinstance(plan, dict) or not plan:
        raise ValueError(f"pick doc has no usable plan for arm={arm!r}")
    return plan


def _xi_ids(plan: dict[str, Any]) -> list[int]:
    xi = plan.get("xi") or []
    out: list[int] = []
    for row in xi:
        if isinstance(row, dict):
            out.append(int(row.get("player_id") or row.get("element") or 0))
        else:
            out.append(int(row))
    return [e for e in out if e]


def _bench_ids(plan: dict[str, Any]) -> list[int]:
    bench = plan.get("bench") or []
    out: list[int] = []
    for row in bench:
        if isinstance(row, dict):
            out.append(int(row.get("player_id") or row.get("element") or 0))
        else:
            out.append(int(row))
    return [e for e in out if e]


def _normalize_xi_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in plan.get("xi") or []:
        if not isinstance(row, dict):
            continue
        eid = int(row.get("player_id") or row.get("element") or 0)
        rows.append(
            {
                "element": eid,
                "name": row.get("name"),
                "position": row.get("position") or row.get("pos"),
                "club": row.get("club"),
                "mu_post": float(row.get("mu_post") or row.get("mu") or 0.0),
                "captain": bool(row.get("captain")),
            }
        )
    return rows


def _normalize_bench_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in plan.get("bench") or []:
        if not isinstance(row, dict):
            continue
        eid = int(row.get("player_id") or row.get("element") or 0)
        rows.append(
            {
                "element": eid,
                "name": row.get("name"),
                "position": row.get("position") or row.get("pos"),
                "club": row.get("club"),
                "mu_post": float(row.get("mu_post") or row.get("mu") or 0.0),
            }
        )
    return rows


def build_lock_doc(
    *,
    doc: dict[str, Any],
    arm: str,
    note: str,
    vice_captain: int | None = None,
    vice_captain_name: str | None = None,
    chip_played: str | None = None,
    deadline_utc: str | None = None,
) -> dict[str, Any]:
    plan = _plan_from_doc(doc, arm)
    xi = _xi_ids(plan)
    bench = _bench_ids(plan)
    squad = list(xi) + list(bench)
    cap = int(plan.get("captain_id") or plan.get("captain") or 0)
    if not cap:
        for row in plan.get("xi") or []:
            if isinstance(row, dict) and row.get("captain"):
                cap = int(row.get("player_id") or row.get("element") or 0)
                break
    locked_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "kind": "numagent_live_lock",
        "gw": int(doc.get("gw") or 0),
        "season": "2026-27",
        "deadline_utc": deadline_utc,
        "locked_at_utc": locked_at,
        "entry_id": int(doc.get("entry_id") or 0),
        "team_name": doc.get("team_name"),
        "status": "locked",
        "note": note
        or "PI locked selection into FPL. Frozen for post-GW μ-vs-actual.",
        "arm_locked": arm.lower(),
        "decision": {
            "chip_played": chip_played,
            "transfers": {
                "n": int(plan.get("n_transfers") or 0),
                "hits": int(plan.get("hits") or 0),
                "hit_points": int(plan.get("hit_points") or 0),
                "sold": list(plan.get("sold_names") or []),
                "bought": list(plan.get("bought_names") or []),
            },
            "formation": plan.get("formation"),
            "squad_15": squad,
            "starting_11": xi,
            "captain": cap,
            "captain_name": plan.get("captain_name"),
            "vice_captain": vice_captain,
            "vice_captain_name": vice_captain_name,
            "bench_order": bench,
            "xi": _normalize_xi_rows(plan),
            "bench": _normalize_bench_rows(plan),
            "bank_after_tenths": plan.get("bank_after"),
            "ft_after": plan.get("ft_after"),
            "horizon_objective": plan.get("horizon_objective"),
            "week0_objective": plan.get("objective"),
        },
        "model": {
            "primary_solver": doc.get("primary"),
            "scorer": "score_xp_v2",
            "odds_source": doc.get("odds_source"),
            "odds_fetched_at": doc.get("odds_fetched_at"),
            "n_odds_ok": doc.get("n_odds_ok"),
            "board_path": doc.get("board_path"),
            "board_t1_path": doc.get("board_t1_path"),
            "xmi_path": doc.get("xmi_path"),
            "pick_source_arm": arm.lower(),
        },
        "source_pick": {
            "entry_id": doc.get("entry_id"),
            "primary": doc.get("primary"),
            "bank": doc.get("bank"),
            "ft": doc.get("ft"),
        },
    }


def write_lock(
    *,
    gw: int,
    entry_id: int,
    lock_doc: dict[str, Any],
    pick_doc: dict[str, Any],
) -> dict[str, Path]:
    d = _gw_dir(gw)
    d.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if int(entry_id) == PRIMARY_ENTRY:
        lock_path = d / "numagent_lock.json"
        report_path = REPORTS / f"numagent_lock_gw{int(gw):02d}.md"
        board_locked = d / "numagent_board_locked.csv"
        board_t1_locked = d / "numagent_board_t1_locked.csv"
    else:
        lock_path = d / f"numagent_lock_{int(entry_id)}.json"
        report_path = REPORTS / f"numagent_lock_gw{int(gw):02d}_{int(entry_id)}.md"
        board_locked = d / f"numagent_board_locked_{int(entry_id)}.csv"
        board_t1_locked = d / f"numagent_board_t1_locked_{int(entry_id)}.csv"

    if lock_path.is_file():
        raise FileExistsError(
            f"{lock_path} already exists — refuse to overwrite a lock. "
            "Delete deliberately only if the PI voids the lock."
        )

    lock_path.write_text(json.dumps(lock_doc, indent=2) + "\n", encoding="utf-8")

    # Freeze board copies used for post-GW scoring.
    src_board = pick_doc.get("board_path")
    src_t1 = pick_doc.get("board_t1_path")
    if src_board and Path(src_board).is_file():
        shutil.copy2(src_board, board_locked)
    if src_t1 and Path(src_t1).is_file():
        shutil.copy2(src_t1, board_t1_locked)

    # Also keep the pick JSON next to the lock under a stable name.
    pick_stable = (
        d / "numagent_pick_locked.json"
        if int(entry_id) == PRIMARY_ENTRY
        else d / f"numagent_pick_locked_{int(entry_id)}.json"
    )
    pick_stable.write_text(json.dumps(pick_doc, indent=2, default=str) + "\n", encoding="utf-8")

    dec = lock_doc["decision"]
    lines = [
        f"# Numagent lock — GW{gw} · entry {entry_id}",
        "",
        f"Locked at `{lock_doc['locked_at_utc']}`. Arm: **{lock_doc['arm_locked']}**.",
        "",
        lock_doc.get("note") or "",
        "",
        f"- Transfers: {dec['transfers']['n']} "
        f"(hits {dec['transfers']['hits']}, −{dec['transfers']['hit_points']})",
        f"- Sold: {', '.join(dec['transfers']['sold']) or '—'}",
        f"- Bought: {', '.join(dec['transfers']['bought']) or '—'}",
        f"- Captain: **{dec.get('captain_name')}**"
        + (
            f" · VC: {dec.get('vice_captain_name')}"
            if dec.get("vice_captain_name")
            else ""
        ),
        f"- Formation: {dec.get('formation')}",
        "",
        f"Lock JSON: `{lock_path.as_posix()}`",
        f"Board: `{board_locked.as_posix() if board_locked.is_file() else '—'}`",
        "",
        "## XI",
        "",
        "| name | pos | club | μ | C |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in dec.get("xi") or []:
        lines.append(
            f"| {row.get('name')} | {row.get('position')} | {row.get('club')} | "
            f"{float(row.get('mu_post') or 0):.2f} | "
            f"{'C' if row.get('captain') else ''} |"
        )
    lines += ["", "## Bench", "", "| name | pos | club | μ |", "| --- | --- | --- | ---: |"]
    for row in dec.get("bench") or []:
        lines.append(
            f"| {row.get('name')} | {row.get('position')} | {row.get('club')} | "
            f"{float(row.get('mu_post') or 0):.2f} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "lock": lock_path,
        "report": report_path,
        "pick_stable": pick_stable,
        "board": board_locked,
        "board_t1": board_t1_locked,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--entry", type=int, default=PRIMARY_ENTRY)
    ap.add_argument(
        "--arm",
        choices=("recommendation", "l3", "l4"),
        default="l3",
        help="which solver arm to freeze (default l3 — typical deadline submit)",
    )
    ap.add_argument("--from-pick", type=Path, default=None)
    ap.add_argument("--note", type=str, default="")
    ap.add_argument("--vice-id", type=int, default=None)
    ap.add_argument("--vice-name", type=str, default=None)
    ap.add_argument("--chip", type=str, default=None)
    ap.add_argument("--deadline-utc", type=str, default=None)
    args = ap.parse_args(argv)

    pick_path = _resolve_pick_path(int(args.gw), int(args.entry), args.from_pick)
    pick_doc = json.loads(pick_path.read_text(encoding="utf-8"))
    if int(pick_doc.get("entry_id") or 0) not in {0, int(args.entry)}:
        raise SystemExit(
            f"pick entry_id {pick_doc.get('entry_id')} != --entry {args.entry}"
        )

    lock_doc = build_lock_doc(
        doc=pick_doc,
        arm=str(args.arm),
        note=str(args.note or ""),
        vice_captain=args.vice_id,
        vice_captain_name=args.vice_name,
        chip_played=args.chip,
        deadline_utc=args.deadline_utc,
    )
    paths = write_lock(
        gw=int(args.gw),
        entry_id=int(args.entry),
        lock_doc=lock_doc,
        pick_doc=pick_doc,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "gw": int(args.gw),
                "entry_id": int(args.entry),
                "arm": args.arm,
                "locked_at_utc": lock_doc["locked_at_utc"],
                "paths": {k: str(v) for k, v in paths.items()},
                "agent_next": (
                    "git add the lock paths + boards, commit 'Lock GW{gw} entry "
                    "{entry}', push immediately."
                ).format(gw=int(args.gw), entry=int(args.entry)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
