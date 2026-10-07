"""Live 1X2 lines. No Odds API call and no season climb."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from src.live.deadline import BOOTSTRAP_PATH, FIXTURES_PATH, run
from src.live.lines import (
    american_to_decimal,
    assemble,
    fetch_odds_api,
    loose_team,
    quote_from_espn_event,
    quote_from_odds_event,
    scheduled_fixtures,
    write_lines,
)


def _espn(home: str, away: str, day: str, total: float) -> dict:
    return {
        "date": f"{day}T14:00Z",
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": home}},
                    {"homeAway": "away", "team": {"displayName": away}},
                ],
                "odds": [
                    {
                        "overUnder": total,
                        "moneyline": {
                            "home": {"close": {"odds": "-150"}},
                            "draw": {"close": {"odds": "+250"}},
                            "away": {"close": {"odds": "+400"}},
                        },
                        "total": {
                            "over": {"close": {"odds": "-110"}},
                            "under": {"close": {"odds": "-110"}},
                        },
                    }
                ],
            }
        ],
    }


def _odds_event() -> dict:
    return {
        "home_team": "Chelsea",
        "away_team": "AFC Bournemouth",
        "commence_time": "2026-10-10T14:00:00Z",
        "bookmakers": [
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Chelsea", "price": 1.5},
                            {"name": "AFC Bournemouth", "price": 6.0},
                            {"name": "Draw", "price": 4.0},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.8, "point": 2.5},
                            {"name": "Under", "price": 2.0, "point": 2.5},
                            {"name": "Over", "price": 1.2, "point": 3.5},
                            {"name": "Under", "price": 3.0, "point": 3.5},
                        ],
                    },
                ]
            },
            {
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Chelsea", "price": 1.7},
                            {"name": "AFC Bournemouth", "price": 5.0},
                            {"name": "Draw", "price": 4.2},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 2.0, "point": 3.5},
                            {"name": "Under", "price": 1.8, "point": 3.5},
                        ],
                    },
                ]
            },
        ],
    }


class ConversionTest(unittest.TestCase):
    def test_american_prices_and_club_names(self) -> None:
        self.assertAlmostEqual(american_to_decimal("-260"), 1.0 + 100.0 / 260.0)
        self.assertAlmostEqual(american_to_decimal("+150"), 2.5)
        self.assertEqual(loose_team("Tottenham Hotspur"), loose_team("Spurs"))
        self.assertEqual(loose_team("Brighton & Hove Albion"), loose_team("Brighton"))
        self.assertEqual(loose_team("AFC Bournemouth"), loose_team("Bournemouth"))


class QuoteTest(unittest.TestCase):
    def test_a_3_5_total_is_blank_and_the_odds_api_row_wins(self) -> None:
        espn = quote_from_espn_event(_espn("Chelsea", "AFC Bournemouth", "2026-10-10", 3.5))
        self.assertIsNotNone(espn)
        assert espn is not None
        self.assertIsNone(espn["over"])
        self.assertIsNone(espn["under"])
        priced = quote_from_espn_event(_espn("Chelsea", "AFC Bournemouth", "2026-10-10", 2.5))
        assert priced is not None
        self.assertAlmostEqual(priced["over"], 1.0 + 100.0 / 110.0)

        odds = quote_from_odds_event(_odds_event())
        assert odds is not None
        self.assertAlmostEqual(odds["avg_h"], 1.6)
        self.assertAlmostEqual(odds["over"], 1.8)
        self.assertEqual(odds["books"], 2)
        self.assertEqual(odds["key"], espn["key"])

        schedule = [
            {
                "gw": 6,
                "day": "2026-10-10",
                "home": "Chelsea",
                "away": "Bournemouth",
                "key": odds["key"],
            }
        ]
        rows = assemble(schedule, [odds], [priced])
        self.assertEqual(rows[0]["source"], "odds_api")
        self.assertEqual(rows[0]["HomeTeam"], "Chelsea")
        self.assertEqual(rows[0]["Avg>2.5"], 1.8)

        odds["over"] = None
        odds["under"] = None
        blank = assemble(schedule, [odds], [priced])
        self.assertEqual(blank[0]["source"], "odds_api")
        self.assertEqual(blank[0]["Avg>2.5"], "")


class TrialRequestTest(unittest.TestCase):
    def test_one_us_request_is_saved_without_the_key(self) -> None:
        key = "test-key-not-real"
        seen: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url)
            return httpx.Response(
                200,
                json=[],
                headers={
                    "x-requests-remaining": "400",
                    "x-requests-last": "1",
                    "x-requests-used": "10",
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "odds_api_trial.json"
            _payload, meta = fetch_odds_api(key, client=client, raw_path=raw)
            text = raw.read_text(encoding="utf-8")
        client.close()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].params["regions"], "us")
        self.assertEqual(seen[0].params["markets"], "h2h,totals")
        self.assertEqual(seen[0].params["oddsFormat"], "decimal")
        self.assertNotIn(key, text)
        self.assertEqual(meta["reason"], "sent")
        self.assertEqual(meta["last"], "1")
        self.assertEqual(meta["remaining"], "400")

    def test_a_rejected_request_is_not_retried(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, json={"message": "unauthorized"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "odds_api_trial.json"
            _payload, meta = fetch_odds_api("another-test-key", client=client, raw_path=raw)
            self.assertFalse(raw.exists())
        client.close()
        self.assertEqual(calls["n"], 1)
        self.assertEqual(meta["reason"], "error")
        self.assertEqual(meta["detail"], "HTTP 401")


class LiveFileTest(unittest.TestCase):
    def test_a_complete_gameweek_still_chooses_no_chip(self) -> None:
        fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
        bootstrap = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
        names = {int(team["id"]): str(team["name"]) for team in bootstrap["teams"]}
        schedule = [row for row in scheduled_fixtures(fixtures, names) if row["gw"] == 6]
        self.assertEqual(len(schedule), 10)
        quotes = [
            {
                "key": row["key"],
                "avg_h": 2.0,
                "avg_d": 3.4,
                "avg_a": 3.6,
                "over": None,
                "under": None,
                "source": "espn",
                "books": 1,
            }
            for row in schedule
        ]
        rows = assemble(schedule, [], quotes)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gw_lines.csv"
            meta = Path(folder) / "meta.json"
            dest = Path(folder) / "live_deadline_gw6.md"
            write_lines(rows, path)
            meta.write_text(
                json.dumps({"reason": "sent", "remaining": "100", "last": "2"}),
                encoding="utf-8",
            )
            log = run(report_path=dest, live_path=path, trial_path=meta)
            text = dest.read_text(encoding="utf-8")
        self.assertEqual(log.line_status, "priced")
        self.assertIsNone(log.chip)
        self.assertIn("missing_minutes", log.reasons)
        self.assertNotIn("missing_opening_line", log.reasons)
        self.assertEqual(log.priced_weeks, ())
        self.assertIn("not chosen", text)
        self.assertIn("It cost 2 credits", text)
        self.assertNotIn("327", text)

    def test_one_missing_match_keeps_the_stop(self) -> None:
        fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
        bootstrap = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
        names = {int(team["id"]): str(team["name"]) for team in bootstrap["teams"]}
        schedule = [row for row in scheduled_fixtures(fixtures, names) if row["gw"] == 6]
        quotes = [
            {
                "key": row["key"],
                "avg_h": 2.0,
                "avg_d": 3.4,
                "avg_a": 3.6,
                "over": 1.9,
                "under": 1.9,
                "source": "odds_api",
                "books": 4,
            }
            for row in schedule[1:]
        ]
        rows = assemble(schedule, quotes, [])
        self.assertEqual(len(rows), 9)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gw_lines.csv"
            dest = Path(folder) / "note.md"
            write_lines(rows, path)
            log = run(report_path=dest, live_path=path, trial_path=Path(folder) / "absent.json")
        self.assertEqual(log.line_status, "missing_opening_line")
        self.assertIsNone(log.chip)


if __name__ == "__main__":
    unittest.main()
