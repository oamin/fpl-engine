"""Free FPL API snapshot: bootstrap and fixtures. No odds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "live"
BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
ELEMENT = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def refresh(cache: Path | None = None, client: httpx.Client | None = None) -> dict[str, Any]:
    """Download bootstrap and fixtures into ``data/live``. Returns both payloads."""
    folder = cache or CACHE
    folder.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=60.0, headers={"User-Agent": "fpl-engine-live"})
    try:
        bootstrap = client.get(BOOTSTRAP_URL).json()
        fixtures = client.get(FIXTURES_URL).json()
    finally:
        if own:
            client.close()
    (folder / "bootstrap.json").write_text(json.dumps(bootstrap), encoding="utf-8")
    (folder / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")
    return {"bootstrap": bootstrap, "fixtures": fixtures}


def load(cache: Path | None = None) -> dict[str, Any]:
    folder = cache or CACHE
    bootstrap = json.loads((folder / "bootstrap.json").read_text(encoding="utf-8"))
    fixtures = json.loads((folder / "fixtures.json").read_text(encoding="utf-8"))
    return {"bootstrap": bootstrap, "fixtures": fixtures}


def next_event(bootstrap: dict[str, Any]) -> dict[str, Any]:
    """The deadline still ahead. Falls back to the current event if none is next."""
    events = bootstrap["events"]
    nxt = next((row for row in events if row.get("is_next")), None)
    if nxt is not None:
        return nxt
    current = next((row for row in events if row.get("is_current")), None)
    if current is None:
        raise ValueError("bootstrap has no current or next event")
    return current


def fixture_counts(fixtures: list[dict[str, Any]], gw: int) -> dict[int, int]:
    """How many fixtures each club has in this gameweek. Absent clubs have zero."""
    counts: dict[int, int] = {}
    for row in fixtures:
        if int(row.get("event") or 0) != int(gw):
            continue
        for side in ("team_h", "team_a"):
            club = int(row[side])
            counts[club] = counts.get(club, 0) + 1
    return counts


def week_flags(counts: dict[int, int], clubs: list[int]) -> tuple[bool, bool]:
    """(any listed club is blank, any listed club has two or more fixtures)."""
    played = [counts.get(int(club), 0) for club in clubs]
    return (any(n <= 0 for n in played), any(n >= 2 for n in played))


def players_table(bootstrap: dict[str, Any]) -> list[dict[str, Any]]:
    """Prices in tenths, position codes, and status. Scoring constants stay in the rules module."""
    teams = {int(row["id"]): row["name"] for row in bootstrap["teams"]}
    rows = []
    for el in bootstrap["elements"]:
        rows.append(
            {
                "player_id": str(el["id"]),
                "web_name": el["web_name"],
                "team_id": int(el["team"]),
                "team": teams[int(el["team"])],
                "position": ELEMENT[int(el["element_type"])],
                "now_cost": int(el["now_cost"]),
                "status": el["status"],
                "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            }
        )
    return rows
