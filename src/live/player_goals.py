"""Gameweek 6 goal rates from anytime goalscorer prices.

Historical ``score_xp`` is unchanged. A listed player replaces only his
Gameweek 6 goal rate. The book probability is turned into a Poisson mean,
scaled by minutes over 90, then capped so the priced players on a club
cannot exceed the share of the match line that those players represent.
A player with no price keeps share times team λ. Assists and the other
components stay on the match line.

The sports list has no Premier League winner, top-four, or relegation
market, so there is no odds ranking to carry past Gameweek 7.
"""

from __future__ import annotations

import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping

import httpx

from src.live.lines import redact

MARKET = "player_goal_scorer_anytime"
REGION = "us"
ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "data" / "predictions" / "2026-27" / "gw06" / "live_20261008"
RAW_PATH = LIVE_DIR / "player_goals_gw6.json"
REPORT_PATH = ROOT / "reports" / "gw6_player_odds.md"
WATCH = (8, 411, 154, 427)


def fold_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    return "".join(ch for ch in text.lower() if ch.isalnum())


def index_players(elements: list[Mapping[str, Any]]) -> dict[str, list[int]]:
    """Full name, web name, and second name, each pointing at element ids."""
    found: dict[str, list[int]] = {}

    def add(label: str, pid: int) -> None:
        key = fold_name(label)
        if not key:
            return
        found.setdefault(key, [])
        if pid not in found[key]:
            found[key].append(pid)

    for element in elements:
        pid = int(element["id"])
        full = f"{element.get('first_name') or ''} {element.get('second_name') or ''}"
        add(full, pid)
        add(str(element.get("web_name") or ""), pid)
        add(str(element.get("second_name") or ""), pid)
    return found


def match_player(name: str, index: Mapping[str, list[int]]) -> int | None:
    """One element id, or none when the book name is missing or shared."""
    hits = index.get(fold_name(name)) or []
    if len(hits) == 1:
        return int(hits[0])
    return None


def poisson_mean(prices: list[float]) -> tuple[float, float]:
    """Mean of 1/price, clipped, then -ln(1-p)."""
    clean = [float(price) for price in prices if float(price) > 1.0]
    if not clean:
        return 0.0, 0.0
    probability = sum(1.0 / price for price in clean) / len(clean)
    probability = min(0.85, max(0.01, probability))
    return probability, -math.log(1.0 - probability)


def team_goal_rates(players: list[dict[str, Any]], lam: float) -> dict[int, float]:
    """Priced players only. Unpriced shares keep the match line outside this map.

    ``mu_raw`` is the Poisson mean before minutes. Minutes scale is xmi/90.
    If the scaled rates exceed the goal budget left by the unpriced shares,
    they are reduced to that budget and never increased.
    """
    unpriced = 0.0
    priced: list[tuple[int, float]] = []
    for player in players:
        if player.get("mu_raw") is None:
            unpriced += max(0.0, float(player.get("share") or 0.0))
            continue
        minutes = max(0.0, float(player.get("xmi") or 0.0))
        priced.append((int(player["id"]), float(player["mu_raw"]) * minutes / 90.0))
    unpriced = min(1.0, unpriced)
    budget = max(0.1, float(lam) * (1.0 - unpriced))
    total = sum(rate for _pid, rate in priced)
    scale = 1.0
    if total > budget and total > 0.0:
        scale = budget / total
    scale = min(1.0, scale)
    return {pid: rate * scale for pid, rate in priced}


def outcome_name(outcome: Mapping[str, Any]) -> str:
    description = str(outcome.get("description") or "").strip()
    name = str(outcome.get("name") or "").strip()
    if description and name.lower() in {"yes", "no", "over", "under"}:
        return description
    return name or description


def quotes_from_event(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per player name, with every US decimal price on that name."""
    grouped: dict[str, list[float]] = {}
    for book in event.get("bookmakers") or []:
        for market in book.get("markets") or []:
            if str(market.get("key")) != MARKET:
                continue
            for outcome in market.get("outcomes") or []:
                label = outcome_name(outcome)
                try:
                    price = float(outcome["price"])
                except (TypeError, ValueError, KeyError):
                    continue
                if not label or price <= 1.0:
                    continue
                grouped.setdefault(label, []).append(price)
    return [{"name": name, "prices": prices} for name, prices in grouped.items()]


def fetch_events(events: list[dict[str, str]], key: str) -> tuple[list[dict[str, Any]], list[str]]:
    """One request per fixture. Region ``us``, one market. No retry."""
    notes: list[str] = []
    payload: list[dict[str, Any]] = []
    with httpx.Client(timeout=40.0, headers={"User-Agent": "fpl-engine"}) as client:
        for event in events:
            url = (
                "https://api.the-odds-api.com/v4/sports/soccer_epl/events/"
                f"{event['id']}/odds"
            )
            response = client.get(
                url,
                params={
                    "apiKey": key,
                    "regions": REGION,
                    "markets": MARKET,
                    "oddsFormat": "decimal",
                },
            )
            last = response.headers.get("x-requests-last", "")
            remaining = response.headers.get("x-requests-remaining", "")
            notes.append(
                f"{event['home']} v {event['away']}: HTTP {response.status_code}, "
                f"last {last or '-'}, remaining {remaining or '-'}"
            )
            if response.status_code != 200:
                notes.append(redact(response.text[:200], key))
                break
            body = response.json()
            body["fixture"] = f"{event['home']} v {event['away']}"
            payload.append(body)
    return payload, notes


def gw6_events(raw_events: list[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Fixtures whose kickoff is before the Gameweek 7 slate."""
    chosen = []
    for event in raw_events:
        commence = str(event.get("commence_time") or "")
        if commence[:10] >= "2026-10-13":
            continue
        chosen.append(
            {
                "id": str(event["id"]),
                "home": str(event.get("home_team") or ""),
                "away": str(event.get("away_team") or ""),
            }
        )
    return chosen
