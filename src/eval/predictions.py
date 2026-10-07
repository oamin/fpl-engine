"""Timestamped pre-deadline scores. An existing file is not overwritten.

The scores are the current live formula for the named gameweek. They do not
replace the published Gameweeks 1–5 total. This module writes only under
``data/predictions/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.holdout import sha256_file
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
) -> Path:
    """Score one deadline from the stored files and write a new timestamped CSV."""
    root = dest_root or ROOT
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    logs = pd.read_csv(LOG_PATH)
    odds = load_odds_frame(ODDS_PATH, LINES_PATH if LINES_PATH.is_file() else None)
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
        "bootstrap_hash": sha256_file(BOOTSTRAP_PATH),
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
