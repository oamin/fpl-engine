"""Live 1X2 and 2.5 totals for the deadline.

Pure Betfair Exchange from 2026-10-08: MATCH_ODDS and OVER_UNDER_25 only.
Fair decimal prices are the reciprocal of simplex-normalised back/lay mids,
with tiered liquidity shrinkage (Gemini bc-e75c8209). Odds API and ESPN are
not used on the live path. Frozen holdout files under ``data/live/`` are not
overwritten by callers that pass a predictions path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.live.betfair import BetfairClient, fetch_epl_line_quotes, load_secret, redact
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "data" / "live"
LINES_PATH = LIVE_DIR / "gw_lines.csv"
TRIAL_RAW = LIVE_DIR / "betfair_trial.json"
TRIAL_META = LIVE_DIR / "betfair_meta.json"
RAW_DIR = ROOT / "data" / "scratch" / "betfair"

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


def load_key() -> str:
    """Odds API key (legacy). Empty when unset. Live lines do not call it."""
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


def assemble(
    schedule: list[Mapping[str, Any]],
    betfair_quotes: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """One row per scheduled fixture that Betfair priced (or neutered)."""
    by_key = {quote["key"]: quote for quote in betfair_quotes}
    rows = []
    for fixture in schedule:
        quote = by_key.get(fixture["key"])
        if quote is None:
            # Also try matching on normalised club names ignoring Betfair's day
            # stamp drift of at most zero — keys already share the ISO day.
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
                "source": "betfair",
                "books": 1,
                "gw": int(fixture["gw"]),
            }
        )
    return rows


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
    client: Any = None,
    espn_client: Any = None,
) -> dict[str, Any]:
    """Write the live CSV from Betfair only.

    ``key``, ``client``, and ``espn_client`` are accepted for call-site
    compatibility and ignored. Odds API fallback is forbidden.
    """
    del key, client, espn_client  # pure Betfair; no bookmaker path
    schedule = scheduled_fixtures(fixtures, team_names)
    app_key = load_secret("BETFAIR_APP_KEY")
    session = load_secret("BETFAIR_SESSION_TOKEN")
    if not app_key:
        meta = {
            "sent": False,
            "ok": False,
            "reason": "no_betfair_app_key",
            "detail": "",
            "source": "betfair",
            "n_events": 0,
            "remaining": "",
            "last": "",
            "used": "",
        }
        write_lines([], lines_path)
        write_meta(meta, meta_path)
        return {"rows": 0, "meta": meta}

    bf = BetfairClient(app_key=app_key, session=session or None)
    try:
        try:
            if not session:
                bf.ensure_session()
            quotes, meta = fetch_epl_line_quotes(bf, raw_dir=RAW_DIR)
        except RuntimeError as exc:
            detail = redact(str(exc), app_key, session, bf.session)
            reason = "betfair_geo_blocked" if "HTTP 403" in detail else "betfair_error"
            meta = {
                "sent": True,
                "ok": False,
                "reason": reason,
                "detail": detail[:400],
                "source": "betfair",
                "n_events": 0,
                "remaining": "",
                "last": "",
                "used": "",
            }
            write_lines([], lines_path)
            write_meta(meta, meta_path)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps({"error": meta["detail"], "reason": reason}),
                encoding="utf-8",
            )
            return {"rows": 0, "meta": meta}
    finally:
        bf.close()

    # Persist a redacted summary next to the CSV (no secrets, no full books).
    summary = [
        {
            "day": q.get("day"),
            "home": q.get("home"),
            "away": q.get("away"),
            "tier": q.get("tier"),
            "matched": q.get("matched"),
            "avg_h": q.get("avg_h"),
            "avg_d": q.get("avg_d"),
            "avg_a": q.get("avg_a"),
            "over": q.get("over"),
            "under": q.get("under"),
        }
        for q in quotes
    ]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = assemble(schedule, quotes)
    write_lines(rows, lines_path)
    write_meta(meta, meta_path)
    return {"rows": len(rows), "meta": meta, "quotes": quotes}


def main() -> None:
    from src.live.fpl_snapshot import load

    snap = load()
    names = {int(row["id"]): str(row["name"]) for row in snap["bootstrap"]["teams"]}
    result = refresh_lines(fixtures=snap["fixtures"], team_names=names)
    meta = result["meta"]
    print(
        f"lines {result['rows']}; betfair {meta.get('reason')}; "
        f"events {meta.get('n_events') or 0}; tiers {meta.get('tiers') or {}}"
    )


if __name__ == "__main__":
    main()
