"""Live 1X2 and 2.5 totals for the deadline.

Pure Betfair Exchange: MATCH_ODDS and OVER_UNDER_25 only.
Fair decimal prices are the reciprocal of simplex-normalised back/lay mids,
with tiered liquidity shrinkage (Gemini bc-e75c8209). The frozen holdout
file ``data/live/gw_lines.csv`` is never written.
"""

from __future__ import annotations

import json
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


class FrozenSlateError(RuntimeError):
    """``data/live/gw_lines.csv`` is the holdout slate and is not a live book."""


def refuse_frozen_slate(path: Path) -> None:
    """Raise when ``path`` is the frozen holdout 1X2 file."""
    if Path(path).resolve() == LINES_PATH.resolve():
        raise FrozenSlateError(
            "data/live/gw_lines.csv is the frozen holdout slate. "
            "Live lines are written only under a Betfair predictions folder."
        )


def loose_team(name: str) -> str:
    """Club key shared by the bootstrap name and the book name."""
    return norm_team(str(name).replace("&", " and "))


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


def write_lines(rows: list[dict[str, Any]], path: Path) -> None:
    refuse_frozen_slate(path)
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
    lines_path: Path,
    raw_path: Path,
    meta_path: Path,
    client: Any = None,
    espn_client: Any = None,
) -> dict[str, Any]:
    """Write the live CSV from Betfair only.

    ``key``, ``client``, and ``espn_client`` are accepted for call-site
    compatibility and ignored. The frozen holdout CSV is never the destination.
    """
    del key, client, espn_client  # pure Betfair; no bookmaker path
    refuse_frozen_slate(lines_path)
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


def main(argv: list[str] | None = None) -> None:
    import argparse

    from src.live.fpl_snapshot import load

    parser = argparse.ArgumentParser(
        description="Write Betfair 1X2 lines into a predictions folder."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    lines_path = out / "gw_lines.csv"
    refuse_frozen_slate(lines_path)
    out.mkdir(parents=True, exist_ok=True)
    snap = load()
    names = {int(row["id"]): str(row["name"]) for row in snap["bootstrap"]["teams"]}
    result = refresh_lines(
        fixtures=snap["fixtures"],
        team_names=names,
        lines_path=lines_path,
        raw_path=out / "betfair_trial.json",
        meta_path=out / "betfair_meta.json",
    )
    meta = result["meta"]
    print(
        f"lines {result['rows']}; betfair {meta.get('reason')}; "
        f"events {meta.get('n_events') or 0}; tiers {meta.get('tiers') or {}}"
    )


if __name__ == "__main__":
    main()
