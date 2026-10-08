"""Live 1X2 lines from Betfair. No Odds API call and no season climb."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.live.deadline import BOOTSTRAP_PATH, FIXTURES_PATH, run
from src.live.lines import assemble, loose_team, scheduled_fixtures, write_lines


class ConversionTest(unittest.TestCase):
    def test_club_names(self) -> None:
        self.assertEqual(loose_team("Tottenham Hotspur"), loose_team("Spurs"))
        self.assertEqual(loose_team("Brighton & Hove Albion"), loose_team("Brighton"))
        self.assertEqual(loose_team("AFC Bournemouth"), loose_team("Bournemouth"))
        self.assertEqual(loose_team("Man City"), loose_team("Manchester City"))


class AssembleTest(unittest.TestCase):
    def test_betfair_row_wins_and_blank_total(self) -> None:
        key = ("2026-10-10", loose_team("Chelsea"), loose_team("Bournemouth"))
        schedule = [
            {
                "gw": 6,
                "day": "2026-10-10",
                "home": "Chelsea",
                "away": "Bournemouth",
                "key": key,
            }
        ]
        quote = {
            "key": key,
            "avg_h": 1.55,
            "avg_d": 4.2,
            "avg_a": 6.1,
            "over": 1.9,
            "under": 1.95,
            "source": "betfair",
            "books": 1,
        }
        rows = assemble(schedule, [quote])
        self.assertEqual(rows[0]["source"], "betfair")
        self.assertEqual(rows[0]["AvgH"], 1.55)
        self.assertEqual(rows[0]["Avg>2.5"], 1.9)

        quote["over"] = None
        quote["under"] = None
        blank = assemble(schedule, [quote])
        self.assertEqual(blank[0]["Avg>2.5"], "")


class RefreshBetfairTest(unittest.TestCase):
    def test_refresh_writes_betfair_rows_without_odds_api(self) -> None:
        from src.live import lines as lines_mod

        key = ("2026-10-10", loose_team("Chelsea"), loose_team("Bournemouth"))
        fake_quotes = [
            {
                "key": key,
                "day": "2026-10-10",
                "home": "Chelsea",
                "away": "Bournemouth",
                "avg_h": 1.6,
                "avg_d": 4.0,
                "avg_a": 5.5,
                "over": 1.85,
                "under": 2.05,
                "source": "betfair",
                "books": 1,
                "tier": "tier1",
                "matched": 80_000.0,
                "spread": 0.02,
            }
        ]
        fake_meta = {
            "sent": True,
            "ok": True,
            "reason": "betfair",
            "n_events": 1,
            "tiers": {"tier1": 1},
            "source": "betfair",
            "remaining": "",
            "last": "",
            "used": "",
            "detail": "",
        }

        class FakeClient:
            def __init__(self, *a: object, **k: object) -> None:
                pass

            def ensure_session(self) -> str:
                return "tok"

            def close(self) -> None:
                return None

        fixtures = [
            {
                "event": 6,
                "kickoff_time": "2026-10-10T14:00:00Z",
                "team_h": 1,
                "team_a": 2,
            }
        ]
        names = {1: "Chelsea", 2: "Bournemouth"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lines_path = root / "gw_lines.csv"
            raw_path = root / "betfair_trial.json"
            meta_path = root / "betfair_meta.json"
            with (
                mock.patch.object(lines_mod, "load_secret", side_effect=lambda n: "x" if "KEY" in n or "TOKEN" in n else ""),
                mock.patch.object(lines_mod, "BetfairClient", FakeClient),
                mock.patch.object(
                    lines_mod,
                    "fetch_epl_line_quotes",
                    return_value=(fake_quotes, fake_meta),
                ),
                mock.patch.object(lines_mod, "RAW_DIR", root / "raw"),
            ):
                result = lines_mod.refresh_lines(
                    fixtures=fixtures,
                    team_names=names,
                    lines_path=lines_path,
                    raw_path=raw_path,
                    meta_path=meta_path,
                )
            text = lines_path.read_text(encoding="utf-8")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(result["rows"], 1)
        self.assertEqual(meta["reason"], "betfair")
        self.assertIn("betfair", text)
        self.assertIn("Chelsea", text)
        self.assertNotIn("odds_api", text)


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
                "source": "betfair",
                "books": 1,
            }
            for row in schedule
        ]
        rows = assemble(schedule, quotes)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gw_lines.csv"
            meta = Path(folder) / "meta.json"
            dest = Path(folder) / "live_deadline_gw6.md"
            write_lines(rows, path)
            meta.write_text(
                json.dumps({"reason": "betfair", "n_events": 10, "source": "betfair"}),
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
        self.assertIn("Betfair", text)
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
                "source": "betfair",
                "books": 1,
            }
            for row in schedule[1:]
        ]
        rows = assemble(schedule, quotes)
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
