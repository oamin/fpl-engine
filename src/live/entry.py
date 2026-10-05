"""A manager's 2026/27 squad, transfers, and chips from the public FPL entry API.

Gameweek 1 is week 0: the initial 15. No login is required. The paid Odds API
is not used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from src.rules.fpl_2026 import CHIPS, ChipWallet, advance_ft, hit_cost

ROOT = Path(__file__).resolve().parents[2]
ENTRY_DIR = ROOT / "data" / "entry"
ENTRY_URL = "https://fantasy.premierleague.com/api/entry/{entry_id}/"

# FPL chip codes on the public history payload.
CHIP_FROM_API = {
    "3xc": "triple_captain",
    "bboost": "bench_boost",
    "freehit": "free_hit",
    "wildcard": "wildcard",
}


class EntryError(ValueError):
    """The public entry payload is missing a field the backfill needs."""


def chip_name(code: str | None) -> str | None:
    if code in (None, "", "null"):
        return None
    if code not in CHIP_FROM_API:
        raise EntryError(f"unknown FPL chip code: {code}")
    return CHIP_FROM_API[code]


def roll_free_transfers(
    weeks: list[dict[str, Any]],
) -> tuple[dict[int, int], int]:
    """Free transfers available at the start of each played week, and the next bank.

    Gameweek 1 is the initial squad, so it spends no bank. ``advance_ft(0, 0)``
    is the one free transfer awarded for gameweek 2. Later weeks use the
    rules module, including a triple-captain week, which does spend the bank.
    """
    available: dict[int, int] = {}
    carried = 0
    for row in sorted(weeks, key=lambda item: int(item["event"])):
        gw = int(row["event"])
        made = int(row["transfers"])
        chip = row.get("chip")
        if gw == 1:
            available[gw] = 0
            carried = advance_ft(0, made, chip)
            continue
        available[gw] = carried
        carried = advance_ft(carried, made, chip)
    return available, carried


def chips_remaining(played: dict[int, str], gw: int) -> tuple[str, ...]:
    """Chips still legal in ``gw`` after the weeks already played."""
    wallet = ChipWallet()
    for week in sorted(played):
        wallet.play(int(week), played[week])
    return wallet.available(gw)


def build_entry(
    entry: dict[str, Any],
    history: dict[str, Any],
    transfers: list[dict[str, Any]],
    picks_by_gw: dict[int, dict[str, Any]],
    names: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """Normalise the public payloads. ``names`` maps element id to web name, position, team."""
    chip_by_gw = {int(row["event"]): chip_name(row["name"]) for row in history.get("chips") or []}
    if any(name not in CHIPS for name in chip_by_gw.values() if name):
        raise EntryError("chip map contains an unknown name")
    weeks_in = []
    for row in history.get("current") or []:
        gw = int(row["event"])
        picks = picks_by_gw.get(gw)
        if picks is None:
            raise EntryError(f"missing picks for GW{gw}")
        active = chip_name(picks.get("active_chip"))
        if active != chip_by_gw.get(gw):
            raise EntryError(f"GW{gw} chip on the picks does not match history")
        weeks_in.append(
            {
                "event": gw,
                "transfers": int(row["event_transfers"]),
                "chip": active,
                "points": int(row["points"]),
                "total_points": int(row["total_points"]),
                "transfer_cost": int(row["event_transfers_cost"]),
                "points_on_bench": int(row["points_on_bench"]),
                "bank": int(row["bank"]),
                "value": int(row["value"]),
                "picks": picks.get("picks") or [],
            }
        )
    ft_at_gw, ft_next = roll_free_transfers(weeks_in)
    for row in weeks_in:
        cost = hit_cost(ft_at_gw[row["event"]], row["transfers"], row["chip"])
        if cost != row["transfer_cost"]:
            raise EntryError(
                f"GW{row['event']} hit cost {row['transfer_cost']} does not match "
                f"a bank of {ft_at_gw[row['event']]}"
            )
    last_gw = max(row["event"] for row in weeks_in)
    played = {row["event"]: row["chip"] for row in weeks_in if row["chip"]}
    gameweeks = []
    for row in weeks_in:
        xi, bench, captain, vice = _slots(row["picks"], names)
        gameweeks.append(
            {
                "gw": row["event"],
                "chip": row["chip"],
                "points": row["points"],
                "total_points": row["total_points"],
                "transfers": row["transfers"],
                "transfer_cost": row["transfer_cost"],
                "points_on_bench": row["points_on_bench"],
                "bank": row["bank"],
                "value": row["value"],
                "ft_available": ft_at_gw[row["event"]],
                "captain": captain,
                "vice": vice,
                "xi": xi,
                "bench": bench,
            }
        )
    named_transfers = []
    for row in transfers:
        inn = names[int(row["element_in"])]
        out = names[int(row["element_out"])]
        named_transfers.append(
            {
                "gw": int(row["event"]),
                "out": out["web_name"],
                "out_id": int(row["element_out"]),
                "in": inn["web_name"],
                "in_id": int(row["element_in"]),
                "in_cost": int(row["element_in_cost"]),
                "out_cost": int(row["element_out_cost"]),
            }
        )
    opening = next(row for row in gameweeks if row["gw"] == 1)
    return {
        "entry_id": int(entry["id"]),
        "team_name": entry.get("name") or "",
        "season": "2026-27",
        "played_through": last_gw,
        "points": int(entry["summary_overall_points"]),
        "overall_rank": int(entry["summary_overall_rank"]),
        "bank": int(weeks_in[-1]["bank"]),
        "value": int(weeks_in[-1]["value"]),
        "ft_for_next": ft_next,
        "next_gw": last_gw + 1,
        "chips_played": [{"gw": gw, "chip": played[gw]} for gw in sorted(played)],
        "chips_left": list(chips_remaining(played, last_gw + 1)),
        "opening_squad": opening["xi"] + opening["bench"],
        "gameweeks": gameweeks,
        "transfers": named_transfers,
    }


def stamp_matchday_teams(
    payload: dict[str, Any], clubs: dict[tuple[int, int], str]
) -> dict[str, Any]:
    """Use the club a player played for that week.

    The bootstrap club is the current one. A mid-season transfer leaves the
    old club on earlier gameweeks. ``clubs`` maps ``(element id, gw)`` to the
    short name. A week with no appearance keeps the stored club.
    """
    out = json.loads(json.dumps(payload))
    for week in out.get("gameweeks") or []:
        gw = int(week["gw"])
        for group in ("xi", "bench"):
            for player in week.get(group) or []:
                club = clubs.get((int(player["id"]), gw))
                if club:
                    player["team"] = club
    gw1 = next((week for week in out.get("gameweeks") or [] if int(week["gw"]) == 1), None)
    if gw1 is not None and out.get("opening_squad"):
        by_id = {int(p["id"]): p["team"] for p in gw1["xi"] + gw1["bench"]}
        for player in out["opening_squad"]:
            if int(player["id"]) in by_id:
                player["team"] = by_id[int(player["id"])]
    return out


def _slots(
    picks: list[dict[str, Any]], names: dict[int, dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    xi: list[dict[str, Any]] = []
    bench: list[dict[str, Any]] = []
    captain = vice = ""
    for pick in sorted(picks, key=lambda row: int(row["position"])):
        meta = names[int(pick["element"])]
        player = {
            "id": int(pick["element"]),
            "name": meta["web_name"],
            "position": meta["position"],
            "team": meta["team"],
            "slot": int(pick["position"]),
        }
        if pick.get("purchase_price") is not None:
            player["purchase_price"] = int(pick["purchase_price"])
        if pick.get("selling_price") is not None:
            player["selling_price"] = int(pick["selling_price"])
        if int(pick["position"]) <= 11:
            xi.append(player)
        else:
            bench.append(player)
        if pick.get("is_captain"):
            captain = meta["web_name"]
        if pick.get("is_vice_captain"):
            vice = meta["web_name"]
    if not captain or not vice:
        raise EntryError("picks are missing a captain or vice-captain")
    return xi, bench, captain, vice


def fetch_entry(entry_id: int, client: httpx.Client | None = None) -> dict[str, Any]:
    """Download history, transfers, and played-week picks, then normalise them."""
    own = client is None
    client = client or httpx.Client(timeout=40.0, headers={"User-Agent": "fpl-engine-live"})
    try:
        base = ENTRY_URL.format(entry_id=int(entry_id))
        entry = client.get(base).json()
        history = client.get(base + "history/").json()
        transfers = client.get(base + "transfers/").json()
        bootstrap = client.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
        teams = {int(row["id"]): row["short_name"] for row in bootstrap["teams"]}
        positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
        names = {
            int(el["id"]): {
                "web_name": el["web_name"],
                "position": positions[int(el["element_type"])],
                "team": teams[int(el["team"])],
            }
            for el in bootstrap["elements"]
        }
        picks_by_gw = {}
        for row in history.get("current") or []:
            gw = int(row["event"])
            response = client.get(base + f"event/{gw}/picks/")
            response.raise_for_status()
            picks_by_gw[gw] = response.json()
    finally:
        if own:
            client.close()
    return build_entry(entry, history, transfers, picks_by_gw, names)


def save_entry(payload: dict[str, Any], folder: Path | None = None) -> Path:
    dest = (folder or ENTRY_DIR) / f"{payload['entry_id']}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def load_entry(entry_id: int, folder: Path | None = None) -> dict[str, Any]:
    path = (folder or ENTRY_DIR) / f"{int(entry_id)}.json"
    return json.loads(path.read_text(encoding="utf-8"))
