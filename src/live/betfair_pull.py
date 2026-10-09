"""Pull Betfair EPL outrights and near-term match odds as a live diagnostic.

Does not change ``score_xp`` or the Gameweek 6 decision horizon. Unpriced
weeks stay unknown for the active plan. Raw JSON is written under a
gitignored directory; the markdown report holds derived probabilities only.

Usage::

    python3 -m src.live.betfair_pull
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.live.betfair import (
    EPL_COMPETITION_ID,
    EVENT_TYPE_SOCCER,
    MIN_MATCHED_MATCH_ODDS,
    BetfairClient,
    classify_outright,
    expected_rank,
    load_secret,
    runner_mids,
    simplex,
    strength_from_rank,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "scratch" / "betfair"
REPORT_PATH = ROOT / "reports" / "betfair_overlay_20261008.md"
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_raw(name: str, payload: Any) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}_{STAMP}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def pull_outrights(client: BetfairClient) -> dict[str, Any]:
    """Catalogue and book for winner / top-6 / relegation style markets."""
    catalogue = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "competitionIds": [EPL_COMPETITION_ID],
            "marketTypeCodes": ["WINNER", "SPECIAL", "OTHER_PLACE"],
        },
        max_results=200,
    )
    # Broader text search catches "Relegation", "Top 6 Finish", etc.
    extra = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "textQuery": "Premier League",
            "marketBettingTypes": ["ODDS"],
        },
        max_results=200,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in list(catalogue) + list(extra):
        mid = str(row.get("marketId") or "")
        if not mid:
            continue
        kind = classify_outright(str(row.get("marketName") or ""))
        if kind is None:
            continue
        row = dict(row)
        row["_kind"] = kind
        by_id[mid] = row
    markets = list(by_id.values())
    books = client.list_market_book([str(m["marketId"]) for m in markets])
    book_by_id = {str(b["marketId"]): b for b in books}
    _write_raw("outright_catalogue", markets)
    _write_raw("outright_books", books)

    by_kind: dict[str, dict[str, float]] = {}
    detail: dict[str, Any] = {}
    for market in markets:
        kind = str(market["_kind"])
        book = book_by_id.get(str(market["marketId"]))
        if book is None:
            continue
        mids = runner_mids(book, market)
        mass = {"winner": 1.0, "top6": 6.0, "relegation": 3.0}[kind]
        normalised = simplex(mids, mass=mass)
        # Prefer the most-matched market per kind.
        matched = float(book.get("totalMatched") or 0.0)
        prev = detail.get(kind)
        if prev is not None and float(prev.get("totalMatched") or 0.0) >= matched:
            continue
        by_kind[kind] = normalised
        detail[kind] = {
            "marketId": market["marketId"],
            "marketName": market.get("marketName"),
            "totalMatched": matched,
            "n_runners": len(normalised),
        }
    return {"by_kind": by_kind, "detail": detail, "n_markets": len(markets)}


def pull_match_odds(client: BetfairClient) -> list[dict[str, Any]]:
    """Near-term EPL MATCH_ODDS with liquidity gate and simplex mids."""
    catalogue = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "competitionIds": [EPL_COMPETITION_ID],
            "marketTypeCodes": ["MATCH_ODDS"],
        },
        max_results=50,
    )
    books = client.list_market_book([str(m["marketId"]) for m in catalogue])
    book_by_id = {str(b["marketId"]): b for b in books}
    _write_raw("match_odds_catalogue", catalogue)
    _write_raw("match_odds_books", books)

    rows: list[dict[str, Any]] = []
    for market in catalogue:
        book = book_by_id.get(str(market["marketId"]))
        if book is None:
            continue
        matched = float(book.get("totalMatched") or 0.0)
        event = market.get("event") or {}
        mids = runner_mids(book, market)
        # Map Home/Away/The Draw when runner names are team names.
        normalised = simplex(mids, mass=1.0)
        liquid = matched >= MIN_MATCHED_MATCH_ODDS
        n_two_sided = sum(1 for v in mids.values() if v is not None)
        rows.append(
            {
                "event": event.get("name"),
                "start": market.get("marketStartTime") or event.get("openDate"),
                "marketId": market["marketId"],
                "totalMatched": matched,
                "liquid": liquid,
                "n_two_sided": n_two_sided,
                "mids_raw": mids,
                "probs": normalised if liquid and n_two_sided >= 2 else {},
            }
        )
    rows.sort(key=lambda r: str(r.get("start") or ""))
    return rows


def club_table(outrights: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build expected-rank rows from the three outright markets."""
    by_kind = outrights.get("by_kind") or {}
    clubs = set()
    for kind in ("winner", "top6", "relegation"):
        clubs.update((by_kind.get(kind) or {}).keys())
    rows = []
    for club in sorted(clubs):
        p_win = float((by_kind.get("winner") or {}).get(club, 0.0))
        p_top6 = float((by_kind.get("top6") or {}).get(club, 0.0))
        p_rel = float((by_kind.get("relegation") or {}).get(club, 0.0))
        rank = expected_rank(p_win, p_top6, p_rel)
        rows.append(
            {
                "club": club,
                "p_win": p_win,
                "p_top6": p_top6,
                "p_rel": p_rel,
                "E_rank": rank,
                "strength": strength_from_rank(rank),
            }
        )
    rows.sort(key=lambda r: r["E_rank"])
    return rows


def render_report(
    *,
    auth_ok: bool,
    auth_note: str,
    outrights: dict[str, Any] | None,
    matches: list[dict[str, Any]] | None,
    table: list[dict[str, Any]] | None,
) -> str:
    lines = [
        "# Betfair exchange overlay (diagnostic)",
        "",
        f"Pulled at `{STAMP}`. App key loaded from the environment; not printed.",
        "",
        "Gemini (bc-e75c8209): DROP outright-derived ratings from the active "
        "decision horizon and keep unpriced weeks unknown; CHANGE Betfair match "
        "and goalscorer overlays to require two-sided liquidity and simplex "
        "normalisation before entering shadow diagnostics.",
        "",
        f"Auth: {auth_note}",
        "",
    ]
    if not auth_ok:
        lines.extend(
            [
                "No books were pulled. Set `BETFAIR_USERNAME` and "
                "`BETFAIR_PASSWORD` (or a fresh `BETFAIR_SESSION_TOKEN`) in "
                "`.env` alongside the app key, then re-run "
                "`python3 -m src.live.betfair_pull`.",
                "",
            ]
        )
        return "\n".join(lines)

    assert outrights is not None and matches is not None and table is not None
    lines.append("## Outright markets")
    lines.append("")
    detail = outrights.get("detail") or {}
    if not detail:
        lines.append("No winner / top-6 / relegation markets matched the catalogue filter.")
        lines.append("")
    else:
        for kind in ("winner", "top6", "relegation"):
            row = detail.get(kind)
            if row is None:
                lines.append(f"- `{kind}`: missing")
                continue
            lines.append(
                f"- `{kind}`: {row.get('marketName')} "
                f"(matched £{float(row.get('totalMatched') or 0):,.0f}, "
                f"{row.get('n_runners')} runners)"
            )
        lines.append("")
        lines.append("| Club | P(win) | P(top 6) | P(rel) | E[rank] | strength |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in table:
            lines.append(
                f"| {row['club']} | {row['p_win']:.3f} | {row['p_top6']:.3f} | "
                f"{row['p_rel']:.3f} | {row['E_rank']:.2f} | {row['strength']:+.3f} |"
            )
        total = sum(r["E_rank"] for r in table)
        lines.append("")
        lines.append(f"Sum of E[rank] over quoted clubs: {total:.2f} (identity is 210 when all 20 are present).")
        lines.append("")

    lines.append("## Match odds (liquidity-gated)")
    lines.append("")
    lines.append(
        f"Gate: totalMatched ≥ £{MIN_MATCHED_MATCH_ODDS:,.0f} and at least two "
        "two-sided runners. A fixture under that gate is left unpriced."
    )
    lines.append("")
    lines.append("| Kickoff | Event | Matched | Liquid | Home | Draw | Away |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: |")
    for row in matches:
        probs = row.get("probs") or {}
        # Best-effort Home/Draw/Away labelling from runner names.
        draw = probs.get("The Draw")
        others = [(k, v) for k, v in probs.items() if k != "The Draw"]
        home = others[0][1] if len(others) > 0 else None
        away = others[1][1] if len(others) > 1 else None
        # Prefer event name split "A v B".
        event = str(row.get("event") or "")
        if " v " in event and probs:
            left, right = event.split(" v ", 1)
            home = probs.get(left.strip(), home)
            away = probs.get(right.strip(), away)
            draw = probs.get("The Draw", draw)

        def fmt(x: float | None) -> str:
            return f"{x:.3f}" if x is not None else "—"

        lines.append(
            f"| {row.get('start') or '—'} | {event or '—'} | "
            f"£{float(row.get('totalMatched') or 0):,.0f} | "
            f"{'yes' if row.get('liquid') else 'no'} | "
            f"{fmt(home)} | {fmt(draw)} | {fmt(away)} |"
        )
    lines.append("")
    lines.append(
        "This report does not move the Gameweek 6 squad. Raw books are under "
        "`data/scratch/betfair/` (gitignored)."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    app_key = load_secret("BETFAIR_APP_KEY")
    if not app_key:
        REPORT_PATH.write_text(
            render_report(
                auth_ok=False,
                auth_note="BETFAIR_APP_KEY missing",
                outrights=None,
                matches=None,
                table=None,
            ),
            encoding="utf-8",
        )
        print(f"wrote {REPORT_PATH} (no app key)")
        return 1

    client = BetfairClient(app_key=app_key)
    try:
        try:
            client.ensure_session()
        except RuntimeError as exc:
            note = str(exc)
            REPORT_PATH.write_text(
                render_report(
                    auth_ok=False,
                    auth_note=note,
                    outrights=None,
                    matches=None,
                    table=None,
                ),
                encoding="utf-8",
            )
            print(f"wrote {REPORT_PATH}")
            print(note)
            return 2

        outrights = pull_outrights(client)
        matches = pull_match_odds(client)
        table = club_table(outrights)
        REPORT_PATH.write_text(
            render_report(
                auth_ok=True,
                auth_note="session ok (token not printed)",
                outrights=outrights,
                matches=matches,
                table=table,
            ),
            encoding="utf-8",
        )
        print(f"wrote {REPORT_PATH}")
        print(
            f"outrights kinds={list((outrights.get('detail') or {}).keys())} "
            f"match_markets={len(matches)} "
            f"liquid={sum(1 for m in matches if m.get('liquid'))}"
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
