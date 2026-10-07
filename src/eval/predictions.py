"""Timestamped pre-deadline scores. An existing file is not overwritten.

The scores are the current live formula for the named gameweek. They do not
replace the published Gameweeks 1–5 total. This module writes only under
``data/predictions/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.eval.holdout import sha256_file
from src.eval.provenance import aware_utc, load_deadlines
from src.live.deadline import (
    BOOTSTRAP_PATH,
    DECISION_GW,
    ENTRY_PATH,
    FIXTURES_PATH,
    LOG_PATH,
    ODDS_PATH,
    SEASON,
    api_selling,
    apply_live_minutes,
    current_costs,
    deadline_text,
    final_players,
    gameweek_values,
    holdings_state,
    last_observed_minutes,
    line_status,
    load_minutes,
    load_odds_frame,
    purchase_source,
    readiness,
    reconstruct_purchases,
    resolve_holdings,
    team_names,
)
from src.live.lines import LINES_PATH
from src.live.scorer import player_key, price_half

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data" / "predictions"
MINUTES_PATH = ROOT / "data" / "live" / "xmi_gw6.csv"
NOTE = (
    "Current formula at this deadline. This file does not replace the published "
    "Gameweeks 1-5 total of 280."
)


def official_frame_from_elements(
    elements: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    gw: int,
    captured_at: str,
) -> pd.DataFrame:
    """Build a GW capture from bootstrap elements. ``ep_this`` is the current event only."""
    event = next(item for item in events if int(item["id"]) == int(gw))
    deadline = str(event["deadline_time"])
    if aware_utc(captured_at) >= aware_utc(deadline):
        raise RuntimeError("refusing xP: the capture is not before the deadline")
    if event.get("is_next") and not event.get("is_current"):
        source, role = "ep_next", "next"
    elif event.get("is_current") and not event.get("finished"):
        source, role = "ep_this", "current"
    else:
        raise RuntimeError(f"refusing xP: GW{gw} is not an open pre-deadline event")
    rows = []
    for element in elements:
        rows.append(
            {
                "player_id": f"{SEASON}:{int(element['id'])}",
                "gw": int(gw),
                "official_xp": float(element[source]),
                "source_field": source,
                "event_role": role,
                "captured_at": captured_at,
                "deadline": deadline,
            }
        )
    return pd.DataFrame(rows)


def write_deadlines(events: list[dict[str, Any]], path: Path | None = None) -> Path:
    dest = path or (ROOT / "data" / "predictions" / SEASON / "deadlines.json")
    payload = {
        "season": SEASON,
        "source": "FPL bootstrap-static events.deadline_time",
        "deadlines": {str(int(event["id"])): str(event["deadline_time"]) for event in events},
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def capture_official_ep(gw: int = DECISION_GW) -> Path:
    """Write one official file. The clock is the response Date header, not the local clock."""
    from src.eval.capture_schedule import fetch_bootstrap, write_snapshot

    payload, captured = fetch_bootstrap()
    path = write_snapshot(payload, captured, gw=gw, slot=None)
    load_deadlines()
    return path


def prediction_path(root: Path, season: str, gw: int, stamp: str) -> Path:
    return root / "data" / "predictions" / season / f"gw{int(gw):02d}" / f"{stamp}.csv"


def write_prediction(path: Path, frame: pd.DataFrame) -> None:
    """Write once. A second call on the same path raises."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def export_deadline_scores(
    *,
    gw: int = DECISION_GW,
    minutes_path: Path = MINUTES_PATH,
    dest_root: Path | None = None,
    stamp: str | None = None,
    bootstrap: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    bootstrap_hash: str | None = None,
) -> Path:
    """Score one deadline from the stored files and write a new timestamped CSV.

    ``bootstrap`` and ``created_at`` are the capture this score belongs to.
    The odds file is the one already stored. This function does not fetch odds.
    """
    root = dest_root or ROOT
    created = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    logs = pd.read_csv(LOG_PATH)
    odds = load_odds_frame(ODDS_PATH, LINES_PATH if LINES_PATH.is_file() else None)
    if bootstrap is None:
        bootstrap = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    if not minutes_path.is_file():
        raise RuntimeError(f"missing minutes file {minutes_path}")
    names = team_names(bootstrap)
    status = line_status(odds, fixtures, names, gw)
    ready, reasons = readiness(status, True)
    if not ready:
        raise RuntimeError("deadline is not priced: " + ", ".join(reasons))
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    holdings = resolve_holdings(
        players,
        purchases,
        current_costs(bootstrap),
        api_selling(entry),
        sources,
    )
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    supplied = load_minutes(minutes_path, gw)
    resolved = apply_live_minutes(
        [int(element["id"]) for element in bootstrap["elements"]],
        supplied,
        last_observed_minutes(logs),
    )
    minute_map = {player_key(int(row["player_id"])): float(row["xmi"]) for row in resolved}
    played = {
        int(row["gw"]): str(row["chip"]) for row in entry.get("chips_played") or []
    }
    scored = price_half(
        gw=int(gw),
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=bootstrap,
        state=state,
        minutes=minute_map,
        played=played,
    )
    scores = scored.step_scores.get(int(gw))
    if not scores:
        raise RuntimeError(f"GW{int(gw)} produced no scores")
    hashes = {
        "logs_hash": sha256_file(LOG_PATH),
        "odds_hash": sha256_file(ODDS_PATH),
        "bootstrap_hash": bootstrap_hash or sha256_file(BOOTSTRAP_PATH),
        "fixtures_hash": sha256_file(FIXTURES_PATH),
        "minutes_hash": sha256_file(minutes_path),
        "entry_hash": sha256_file(ENTRY_PATH),
    }
    deadline = deadline_text(bootstrap, gw)
    rows: list[dict[str, Any]] = []
    for pid, score in sorted(scores.items()):
        rows.append(
            {
                "player_id": pid,
                "gw": int(gw),
                "score": float(score),
                "created_at": created,
                "deadline": deadline,
                "note": NOTE,
                **hashes,
            }
        )
    path = prediction_path(root, SEASON, gw, stamp)
    write_prediction(path, pd.DataFrame(rows))
    return path
