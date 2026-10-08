"""Live 1X2 and 2.5 totals for the deadline.

One Odds API request, and only when a key is set: EPL, markets ``h2h`` and
``totals``. Regions are ``uk``, ``eu``, and ``us``. The featured odds
endpoint bills markets times regions, so this request is 6 credits, and it
returns every fixture those books have already posted. ``us2`` is omitted:
it repeats US books. ESPN's current moneyline fills a fixture the trial
does not price, for every scheduled week, not only the next two. A total
other than 2.5 is left blank. The historical football-data file is not
modified.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import httpx
import pandas as pd

from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "data" / "live"
LINES_PATH = LIVE_DIR / "gw_lines.csv"
TRIAL_RAW = LIVE_DIR / "odds_api_trial.json"
TRIAL_META = LIVE_DIR / "odds_api_meta.json"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
# 2 markets x 3 regions = 6 credits on the featured odds endpoint.
REGIONS = "uk,eu,us"
MARKETS = "h2h,totals"

COLUMNS = (
    "Date",
    "HomeTeam",
    "AwayTeam",
    "AvgH",
    "AvgD",
    "AvgA",
    "Avg>2.5",
    "Avg<2.5",
    "source",
    "books",
    "gw",
)


def loose_team(name: str) -> str:
    """Club key shared by the bootstrap name and the book name."""
    return norm_team(str(name).replace("&", " and "))


def american_to_decimal(raw: object) -> float:
    """American moneyline to a decimal price. Even money is +100."""
    text = str(raw).strip().replace("+", "")
    if text.upper() == "EVEN":
        price = 100.0
    else:
        price = float(text)
    if price == 0:
        raise ValueError("american odds cannot be 0")
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / abs(price)


def load_key() -> str:
    """Key from the environment or `.env`. Empty when neither is set."""
    found = os.environ.get("ODDS_API_KEY", "").strip()
    if found:
        return found
    path = ROOT / ".env"
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ODDS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def redact(text: str, key: str = "") -> str:
    """Remove the key and any ``apiKey=`` query from a string."""
    out = text or ""
    if key:
        out = out.replace(key, "[redacted]")
    return re.sub(r"apiKey=[^&\s]+", "apiKey=[redacted]", out)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _fd_date(iso_day: str) -> str:
    year, month, day = iso_day.split("-")
    return f"{int(day):02d}/{int(month):02d}/{year}"


def scheduled_fixtures(
    fixtures: list[Mapping[str, Any]],
    team_names: Mapping[int, str],
    *,
    start: int = 6,
    end: int = 19,
) -> list[dict[str, Any]]:
    """Fixtures from ``start`` through ``end``, with the bootstrap club names."""
    rows = []
    for fixture in fixtures:
        event = fixture.get("event")
        kickoff = fixture.get("kickoff_time") or ""
        if event is None or not kickoff:
            continue
        gw = int(event)
        if gw < int(start) or gw > int(end):
            continue
        home = team_names.get(int(fixture["team_h"]))
        away = team_names.get(int(fixture["team_a"]))
        if not home or not away:
            continue
        day = str(kickoff)[:10]
        rows.append(
            {
                "gw": gw,
                "day": day,
                "home": home,
                "away": away,
                "key": (day, loose_team(home), loose_team(away)),
            }
        )
    return rows


def quote_from_odds_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """Mean decimal 1X2 across the returned books. The 2.5 total only."""
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    commence = str(event.get("commence_time") or "")
    if not home or not away or len(commence) < 10:
        return None
    homes: list[float] = []
    draws: list[float] = []
    aways: list[float] = []
    overs: list[float] = []
    unders: list[float] = []
    for book in event.get("bookmakers") or []:
        markets = {str(item.get("key")): item for item in book.get("markets") or []}
        h2h = markets.get("h2h")
        if h2h:
            by_name = {}
            for outcome in h2h.get("outcomes") or []:
                try:
                    by_name[str(outcome.get("name"))] = float(outcome["price"])
                except (TypeError, ValueError, KeyError):
                    continue
            if home in by_name and away in by_name and "Draw" in by_name:
                homes.append(by_name[home])
                aways.append(by_name[away])
                draws.append(by_name["Draw"])
        totals = markets.get("totals")
        over = under = None
        for outcome in (totals or {}).get("outcomes") or []:
            try:
                point = float(outcome.get("point"))
                price = float(outcome["price"])
            except (TypeError, ValueError, KeyError):
                continue
            if abs(point - 2.5) > 1e-6:
                continue
            label = str(outcome.get("name") or "").lower()
            if label == "over":
                over = price
            elif label == "under":
                under = price
        if over is not None and under is not None:
            overs.append(over)
            unders.append(under)
    if not homes:
        return None
    return {
        "key": (commence[:10], loose_team(home), loose_team(away)),
        "avg_h": _mean(homes),
        "avg_d": _mean(draws),
        "avg_a": _mean(aways),
        "over": _mean(overs) if overs else None,
        "under": _mean(unders) if unders else None,
        "source": "odds_api",
        "books": len(homes),
    }


def _close_american(side: Mapping[str, Any] | None) -> str | None:
    close = (side or {}).get("close") or {}
    odds = close.get("odds")
    if odds in (None, ""):
        return None
    return str(odds)


def quote_from_espn_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """DraftKings close. A featured total other than 2.5 is ignored."""
    competition = (event.get("competitions") or [{}])[0]
    home = away = ""
    for side in competition.get("competitors") or []:
        name = str((side.get("team") or {}).get("displayName") or "")
        if side.get("homeAway") == "home":
            home = name
        elif side.get("homeAway") == "away":
            away = name
    stamp = str(event.get("date") or "")
    odds = (competition.get("odds") or [{}])[0] or {}
    money = odds.get("moneyline") or {}
    if not home or not away or len(stamp) < 10 or not money:
        return None
    try:
        avg_h = american_to_decimal(_close_american(money.get("home")))
        avg_d = american_to_decimal(_close_american(money.get("draw")))
        avg_a = american_to_decimal(_close_american(money.get("away")))
    except (TypeError, ValueError):
        return None
    over = under = None
    try:
        featured = float(odds.get("overUnder"))
    except (TypeError, ValueError):
        featured = None
    if featured is not None and abs(featured - 2.5) < 1e-6:
        total = odds.get("total") or {}
        try:
            over = american_to_decimal(_close_american((total.get("over") or {})))
            under = american_to_decimal(_close_american((total.get("under") or {})))
        except (TypeError, ValueError):
            over = under = None
    return {
        "key": (stamp[:10], loose_team(home), loose_team(away)),
        "avg_h": avg_h,
        "avg_d": avg_d,
        "avg_a": avg_a,
        "over": over,
        "under": under,
        "source": "espn",
        "books": 1,
    }


def assemble(
    schedule: list[Mapping[str, Any]],
    odds_quotes: list[Mapping[str, Any]],
    espn_quotes: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per fixture. The Odds API 1X2 wins. ESPN fills the rest."""
    odds_by = {quote["key"]: quote for quote in odds_quotes}
    espn_by = {quote["key"]: quote for quote in espn_quotes}
    rows = []
    for fixture in schedule:
        quote = odds_by.get(fixture["key"]) or espn_by.get(fixture["key"])
        if quote is None:
            continue
        rows.append(
            {
                "Date": _fd_date(str(fixture["day"])),
                "HomeTeam": fixture["home"],
                "AwayTeam": fixture["away"],
                "AvgH": round(float(quote["avg_h"]), 4),
                "AvgD": round(float(quote["avg_d"]), 4),
                "AvgA": round(float(quote["avg_a"]), 4),
                "Avg>2.5": "" if quote["over"] is None else round(float(quote["over"]), 4),
                "Avg<2.5": "" if quote["under"] is None else round(float(quote["under"]), 4),
                "source": quote["source"],
                "books": int(quote["books"]),
                "gw": int(fixture["gw"]),
            }
        )
    return rows


def fetch_odds_api(
    key: str,
    *,
    client: httpx.Client | None = None,
    raw_path: Path = TRIAL_RAW,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One slate request. A non-200 response is not retried."""
    params = {
        "apiKey": key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": "decimal",
    }
    own = client is None
    client = client or httpx.Client(timeout=40.0, headers={"User-Agent": "fpl-engine"})
    try:
        try:
            response = client.get(ODDS_URL, params=params)
        except httpx.HTTPError as exc:
            return [], {
                "sent": True,
                "ok": False,
                "reason": "error",
                "detail": redact(str(exc), key),
                "remaining": "",
                "last": "",
                "used": "",
                "n_events": 0,
                "regions": REGIONS,
                "markets": MARKETS,
            }
    finally:
        if own:
            client.close()
    meta = {
        "sent": True,
        "ok": response.status_code == 200,
        "reason": "sent" if response.status_code == 200 else "error",
        "detail": "" if response.status_code == 200 else f"HTTP {response.status_code}",
        "remaining": response.headers.get("x-requests-remaining", ""),
        "last": response.headers.get("x-requests-last", ""),
        "used": response.headers.get("x-requests-used", ""),
        "n_events": 0,
        "regions": REGIONS,
        "markets": MARKETS,
    }
    if response.status_code != 200:
        return [], meta
    payload = response.json()
    if not isinstance(payload, list):
        meta["ok"] = False
        meta["reason"] = "error"
        meta["detail"] = "unexpected payload"
        return [], meta
    meta["n_events"] = len(payload)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, meta


def fetch_espn(dates: list[str], *, client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Public scoreboard events for the given ISO days. No key."""
    own = client is None
    client = client or httpx.Client(timeout=30.0, headers={"User-Agent": "fpl-engine"})
    events: list[dict[str, Any]] = []
    try:
        for day in sorted(set(dates)):
            response = client.get(ESPN_URL, params={"dates": day.replace("-", "")})
            response.raise_for_status()
            payload = response.json()
            events.extend(payload.get("events") or [])
    finally:
        if own:
            client.close()
    return events


def write_lines(rows: list[dict[str, Any]], path: Path = LINES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame.to_csv(path, index=False)


def write_meta(meta: Mapping[str, Any], path: Path = TRIAL_META) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta), indent=2) + "\n", encoding="utf-8")


def refresh_lines(
    *,
    fixtures: list[Mapping[str, Any]],
    team_names: Mapping[int, str],
    key: str | None = None,
    lines_path: Path = LINES_PATH,
    raw_path: Path = TRIAL_RAW,
    meta_path: Path = TRIAL_META,
    client: httpx.Client | None = None,
    espn_client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Write the live CSV. The Odds API is called once when ``key`` is non-empty."""
    schedule = scheduled_fixtures(fixtures, team_names)
    secret = load_key() if key is None else key
    if secret:
        payload, meta = fetch_odds_api(secret, client=client, raw_path=raw_path)
    else:
        payload = []
        meta = {
            "sent": False,
            "ok": False,
            "reason": "no_key",
            "detail": "",
            "remaining": "",
            "last": "",
            "used": "",
            "n_events": 0,
            "regions": REGIONS,
            "markets": MARKETS,
        }
    odds_quotes = [quote for event in payload if (quote := quote_from_odds_event(event))]
    espn_dates = [str(row["day"]) for row in schedule]
    try:
        espn_events = fetch_espn(espn_dates, client=espn_client) if espn_dates else []
    except httpx.HTTPError as exc:
        espn_events = []
        meta["espn"] = str(exc)
    espn_quotes = [quote for event in espn_events if (quote := quote_from_espn_event(event))]
    rows = assemble(schedule, odds_quotes, espn_quotes)
    write_lines(rows, lines_path)
    write_meta(meta, meta_path)
    return {"rows": len(rows), "meta": meta}


def main() -> None:
    from src.live.fpl_snapshot import load

    snap = load()
    names = {int(row["id"]): str(row["name"]) for row in snap["bootstrap"]["teams"]}
    result = refresh_lines(fixtures=snap["fixtures"], team_names=names)
    meta = result["meta"]
    print(
        f"lines {result['rows']}; odds {meta.get('reason')}; "
        f"last {meta.get('last') or '-'}; remaining {meta.get('remaining') or '-'}"
    )


if __name__ == "__main__":
    main()
