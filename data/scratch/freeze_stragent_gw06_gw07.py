"""Promote GW6–7 stragent plans to official freeze rows (locked comparison).

Uses each plan's pre-deadline ``frozen_at_utc``. Fills realised points for
closed weeks. Rewrites ``string_agent_freeze.jsonl`` in GW order (6,7,8).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

from src.models.season_climb import apply_autosubs
from src.rules.fpl_2026 import hit_cost
from src.str_agent.carry import CarryState
from src.str_agent.freeze import LEDGER, empty_realised, validate_freeze_row
from src.str_agent.prompt import SYSTEM
from src.str_agent.runner import assemble_freeze_row, sha256_text

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "data/predictions/2026-27/string_plans"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def directory_from_bootstrap() -> dict[str, dict]:
    boot = json.loads((ROOT / "data/live/bootstrap.json").read_text(encoding="utf-8"))
    teams = {int(t["id"]): t["name"] for t in boot["teams"]}
    pos = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    return {
        str(e["id"]): {
            "position": pos[int(e["element_type"])],
            "club": teams[int(e["team"])],
            "name": e["web_name"],
            "now_cost": int(e["now_cost"]),
        }
        for e in boot["elements"]
    }


def live_stats(gw: int) -> dict[int, dict[str, float]]:
    live = httpx.get(
        f"https://fantasy.premierleague.com/api/event/{int(gw)}/live/", timeout=60
    ).json()
    out: dict[int, dict[str, float]] = {}
    for block in live.get("elements") or []:
        st = block.get("stats") or {}
        out[int(block["id"])] = {
            "minutes": float(st.get("minutes") or 0),
            "total_points": float(st.get("total_points") or 0),
        }
    return out


def score_decision(decision: dict, *, gw: int, ft_before: int) -> dict:
    stats = live_stats(gw)
    boot_dir = directory_from_bootstrap()
    xi = [int(x) for x in decision["starting_11"]]
    bench = [int(x) for x in decision["bench_order"]]
    rows = []
    for pid in xi + bench:
        meta = boot_dir[str(pid)]
        s = stats.get(pid, {"minutes": 0.0, "total_points": 0.0})
        rows.append(
            {
                "player_id": str(pid),
                "name": meta["name"],
                "position": meta["position"],
                **s,
            }
        )
    frame = pd.DataFrame(rows)
    final, n_subs = apply_autosubs(frame.iloc[:11].copy(), frame.iloc[11:].copy())
    pts = {str(r.player_id): float(r.total_points) for r in final.itertuples(index=False)}
    mins = {str(r.player_id): float(r.minutes) for r in final.itertuples(index=False)}
    cap = str(decision["captain"])
    vc = str(decision["vice_captain"])
    doubled = vc if mins.get(cap, 0) <= 0 and mins.get(vc, 0) > 0 else cap
    actual = sum(pts.values()) + pts.get(doubled, 0.0)
    n_transfers = len(decision.get("transfers_in") or [])
    hits = hit_cost(int(ft_before), n_transfers, decision.get("chip_played"))
    return {
        "actual_points": int(actual - hits),
        "actual_xi_plus_captain": float(actual),
        "hit_points": int(hits),
        "n_autosubs": int(n_subs),
        "evaluated_at_utc": NOW,
        "comparison_locked": True,
    }


def row_from_plan(plan: dict, directory: dict) -> tuple[dict, str]:
    path = PLAN / f"gw{int(plan['gw']):02d}.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Prefer stored context file for sha; fall back to empty.
    ctx_path = path.with_name(f"gw{int(plan['gw']):02d}_context.md")
    context = ctx_path.read_text(encoding="utf-8") if ctx_path.is_file() else ""
    carry = CarryState.from_dict(plan["carry"])
    payload = {
        "rationale": plan.get("rationale") or "",
        "decision": plan["decision"],
    }
    row = assemble_freeze_row(
        gw=int(plan["gw"]),
        deadline_utc=str(plan["deadline_utc"]),
        frozen_at_utc=str(plan["frozen_at_utc"]),
        model_id=str(plan.get("model_id") or "locked-comparison"),
        system_prompt=SYSTEM,
        context_markdown=context,
        model_payload=payload,
        directory=directory,
        budget=1000,
        carry=carry,
    )
    if not row["accounting"]["is_legal"]:
        raise RuntimeError(f"GW{plan['gw']} illegal: {row['accounting']['errors']}")
    clean = {
        "gw": row["gw"],
        "deadline_utc": row["deadline_utc"],
        "frozen_at_utc": row["frozen_at_utc"],
        "provenance": {
            "model_id": row["provenance"]["model_id"],
            "prompt_sha256": row["provenance"]["prompt_sha256"],
            "context_sha256": row["provenance"]["context_sha256"],
            "plan_sha256": digest,
            "comparison_lock": True,
            "note": (
                "Locked-in stragent vs numagent comparison for this GW. "
                "Frozen at plan frozen_at_utc (pre-deadline)."
            ),
        },
        "rationale": row["rationale"],
        "decision": row["decision"],
        "accounting": {
            "bank_remaining": row["accounting"]["bank_remaining"],
            "total_cost": row["accounting"]["total_cost"],
            "is_legal": True,
            "hits": row["accounting"].get("hits"),
            "bank_after": row["accounting"].get("bank_after"),
        },
        "carry": row.get("carry"),
        "carry_after": row.get("carry_after"),
        "realised": empty_realised(),
    }
    validate_freeze_row(clean)
    return clean, digest


def main() -> None:
    directory = directory_from_bootstrap()
    plans = {
        gw: json.loads((PLAN / f"gw{gw:02d}.json").read_text(encoding="utf-8"))
        for gw in (6, 7, 8)
    }
    # Soften plan model_id labels to locked-comparison language
    for gw, note in (
        (6, "Locked comparison freeze (GW6 horizon decision)"),
        (7, "Locked comparison freeze (GW7 intention from GW6 horizon)"),
    ):
        plans[gw]["model_id"] = note
        plans[gw]["comparison_lock"] = True
        (PLAN / f"gw{gw:02d}.json").write_text(
            json.dumps(plans[gw], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    rows = []
    for gw in (6, 7, 8):
        plan = json.loads((PLAN / f"gw{gw:02d}.json").read_text(encoding="utf-8"))
        row, digest = row_from_plan(plan, directory)
        if gw in (6, 7):
            scored = score_decision(
                plan["decision"], gw=gw, ft_before=int(plan["carry"]["ft_before"])
            )
            row["realised"] = {
                "actual_points": scored["actual_points"],
                "evaluated_at_utc": scored["evaluated_at_utc"],
                "actual_xi_plus_captain": scored["actual_xi_plus_captain"],
                "hit_points": scored["hit_points"],
                "n_autosubs": scored["n_autosubs"],
                "comparison_locked": True,
            }
        rows.append(row)
        print(
            f"GW{gw} freeze ok frozen_at={row['frozen_at_utc']} "
            f"realised={row['realised'].get('actual_points')} plan={digest[:12]}"
        )

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
        encoding="utf-8",
    )
    print("wrote", LEDGER, "rows", len(rows))

    # Update GW7 stragent_results status
    path7 = ROOT / "data/predictions/2026-27/gw07/stragent_results.json"
    doc7 = json.loads(path7.read_text(encoding="utf-8"))
    doc7["official_freeze"] = "data/predictions/2026-27/string_agent_freeze.jsonl"
    doc7["status"] = "frozen_comparison"
    doc7["note"] = (
        "Locked-in stragent vs numagent comparison for GW7 "
        "(official freeze row; realised attached)."
    )
    path7.write_text(json.dumps(doc7, indent=2) + "\n", encoding="utf-8")

    # GW6 results artifact
    r6 = next(r for r in rows if r["gw"] == 6)
    dec6 = r6["decision"]
    boot_dir = directory_from_bootstrap()
    gw6_results = {
        "kind": "stragent_gw_results",
        "gw": 6,
        "evaluated_at_utc": NOW,
        "official_freeze": "data/predictions/2026-27/string_agent_freeze.jsonl",
        "status": "frozen_comparison",
        "selection": {
            "transfers_in": dec6.get("transfers_in"),
            "transfers_out": dec6.get("transfers_out"),
            "captain": boot_dir[str(dec6["captain"])]["name"],
            "vice_captain": boot_dir[str(dec6["vice_captain"])]["name"],
        },
        "realised": r6["realised"],
        "note": "Locked-in stragent vs numagent comparison for GW6.",
    }
    (ROOT / "data/predictions/2026-27/gw06").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/predictions/2026-27/gw06/stragent_results.json").write_text(
        json.dumps(gw6_results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
