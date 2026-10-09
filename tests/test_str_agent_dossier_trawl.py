"""Dossier collector and the manager notebook. No network and no minutes compile."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.live.news_packets import compile_player_xmi, load_gameweek_packets
from src.str_agent.horizon import validate_horizon, write_plan
from src.str_agent.notebook import (
    NotebookContaminationError,
    notebook_section,
    upsert_note,
)
from src.str_agent.trawl import Feed, trawl
from tests.test_str_agent_horizon import _carry, _hold, _market

DEADLINE = "2026-10-10T10:00:00Z"
OBSERVED = "2026-10-09T18:00:00Z"


def _rss(pub: str, link: str, summary: str, title: str = "Training") -> bytes:
    return (
        "<rss><channel><item>"
        f"<title>{title}</title>"
        f"<link>{link}</link>"
        f"<pubDate>{pub}</pubDate>"
        f"<description>{summary}</description>"
        "</item></channel></rss>"
    ).encode()


class TrawlFeeds(unittest.TestCase):
    def test_a_deadline_timestamp_is_dropped(self) -> None:
        feed = Feed("bbc_sport", "press", "https://feeds.example.test/bbc", "rss")
        payload = _rss(
            "Sat, 10 Oct 2026 10:00:00 GMT",
            "https://www.bbc.co.uk/sport/football/late",
            "Published on the deadline.",
        )
        with tempfile.TemporaryDirectory() as folder:
            result = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc="2026-10-10T09:00:00Z",
                root=Path(folder),
                feeds=(feed,),
                fetch=lambda _url: payload,
            )
            self.assertEqual(result.written, 0)
            self.assertEqual(result.dropped, 1)
            self.assertEqual(list(Path(folder).rglob("*.json")), [])

    def test_a_bare_date_is_dropped(self) -> None:
        feed = Feed("premierleague", "youtube", "https://feeds.example.test/yt", "youtube")
        payload = (
            "<feed><entry>"
            "<title>Roundup</title>"
            '<link rel="alternate" href="https://www.youtube.com/watch?v=baredate"/>'
            "<published>2026-10-09</published>"
            "<summary>No clock.</summary>"
            "</entry></feed>"
        ).encode()
        with tempfile.TemporaryDirectory() as folder:
            result = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc=OBSERVED,
                root=Path(folder),
                feeds=(feed,),
                fetch=lambda _url: payload,
            )
            self.assertEqual(result.written, 0)
            self.assertEqual(result.dropped, 1)

    def test_a_second_run_does_not_rewrite_the_url(self) -> None:
        feed = Feed("bbc_sport", "press", "https://feeds.example.test/bbc", "rss")
        payload = _rss(
            "Wed, 07 Oct 2026 12:00:00 GMT",
            "https://www.bbc.co.uk/sport/football/haaland",
            "He finished training.",
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc=OBSERVED,
                root=root,
                feeds=(feed,),
                fetch=lambda _url: payload,
            )
            path = next(root.rglob("src_*.json"))
            blob = path.read_bytes()
            second = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc="2026-10-09T19:00:00Z",
                root=root,
                feeds=(feed,),
                fetch=lambda _url: payload,
            )
            self.assertEqual(first.written, 1)
            self.assertEqual(second.written, 0)
            self.assertEqual(second.skipped_duplicate, 1)
            self.assertEqual(path.read_bytes(), blob)
            note = json.loads(blob)
            self.assertEqual(note["clock"], "http_last_modified")
            self.assertEqual(note["observed_at_utc"], OBSERVED)

    def test_a_forum_file_does_not_move_minutes(self) -> None:
        feed = Feed("reddit_fantasypl", "forum", "https://feeds.example.test/reddit", "reddit_json")
        payload = json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Haaland trained",
                                "permalink": "/r/FantasyPL/comments/abc/haaland/",
                                "created_utc": 1759946400,
                                "selftext": "Regulars say he was seen finishing.",
                            }
                        }
                    ]
                }
            }
        ).encode()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc=OBSERVED,
                root=root,
                feeds=(feed,),
                fetch=lambda _url: payload,
            )
            self.assertEqual(result.written, 1)
            note = json.loads(next(root.rglob("src_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(note["clock"], "reddit_created_utc")
            self.assertEqual(note["outlet_class"], "forum")
            packets = load_gameweek_packets(6, deadline_utc=DEADLINE, root=root)
            self.assertEqual(packets, [])
            compiled = compile_player_xmi(
                player_id=1,
                name="Haaland",
                position="FWD",
                prior=90.0,
                status="a",
                chance=None,
                packets=packets,
                minutes=[90, 90, 90],
            )
            self.assertNotIn("finishing", json.dumps(compiled))
            self.assertFalse((root / "data/predictions/2026-27/gw06/news_packets").exists())

    def test_only_the_feed_summary_is_stored(self) -> None:
        feed = Feed("guardian", "press", "https://feeds.example.test/guardian", "rss")
        summary = "A" * 2500
        calls: list[str] = []

        def fetch(url: str) -> bytes:
            calls.append(url)
            return _rss(
                "2026-10-08T18:00:00+00:00",
                "https://www.theguardian.com/football/2026/oct/08/wood",
                summary + " FULL ARTICLE BODY",
            )

        with tempfile.TemporaryDirectory() as folder:
            result = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc=OBSERVED,
                root=Path(folder),
                feeds=(feed,),
                fetch=fetch,
            )
            self.assertEqual(result.written, 1)
            self.assertEqual(calls, ["https://feeds.example.test/guardian"])
            note = json.loads(next(Path(folder).rglob("src_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(len(note["body"]), 2000)
            self.assertNotIn("FULL ARTICLE", note["body"])
            self.assertEqual(note["clock"], "feed_updated")

    def test_a_failed_feed_leaves_earlier_files(self) -> None:
        first = Feed("bbc_sport", "press", "https://feeds.example.test/bbc", "rss")
        second = Feed("skysports", "press", "https://feeds.example.test/sky", "rss")

        def fetch(url: str) -> bytes:
            if url.endswith("/sky"):
                raise TimeoutError("slow")
            return _rss(
                "Wed, 07 Oct 2026 12:00:00 GMT",
                "https://www.bbc.co.uk/sport/football/ok",
                "A short summary.",
            )

        with tempfile.TemporaryDirectory() as folder:
            result = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc=OBSERVED,
                root=Path(folder),
                feeds=(first, second),
                fetch=fetch,
            )
            self.assertEqual(result.written, 1)
            self.assertEqual(result.feed_errors, ("skysports: TimeoutError",))
            self.assertTrue(next(Path(folder).rglob("src_*.json")).is_file())

    def test_a_british_summer_time_pubdate_is_kept(self) -> None:
        feed = Feed("skysports", "press", "https://feeds.example.test/sky", "rss")
        payload = _rss(
            "Fri, 09 Oct 2026 16:22:00 BST",
            "https://www.skysports.com/football/news/123/sky-note",
            "A short summary.",
        )
        with tempfile.TemporaryDirectory() as folder:
            result = trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc=OBSERVED,
                root=Path(folder),
                feeds=(feed,),
                fetch=lambda _url: payload,
            )
            self.assertEqual(result.written, 1)
            note = json.loads(next(Path(folder).rglob("src_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(note["clock"], "http_last_modified")
            self.assertEqual(note["raw_published_at"], "Fri, 09 Oct 2026 16:22:00 BST")
            self.assertEqual(note["published_at_utc"], "2026-10-09T15:22:00Z")

    def test_a_youtube_clock_stays_on_the_published_field(self) -> None:
        feed = Feed("premierleague", "youtube", "https://feeds.example.test/yt", "youtube")
        payload = (
            "<feed><entry>"
            "<title>Presser</title>"
            '<link rel="alternate" href="https://www.youtube.com/watch?v=presser1"/>'
            "<published>2026-10-08T19:00:00+01:00</published>"
            "<summary>The manager named no starter.</summary>"
            "</entry></feed>"
        ).encode()
        with tempfile.TemporaryDirectory() as folder:
            trawl(
                6,
                deadline_utc=DEADLINE,
                observed_at_utc=OBSERVED,
                root=Path(folder),
                feeds=(feed,),
                fetch=lambda _url: payload,
            )
            note = json.loads(next(Path(folder).rglob("src_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(note["clock"], "youtube_published_at")
            self.assertEqual(note["published_at_utc"], "2026-10-08T18:00:00Z")
            self.assertEqual(note["raw_published_at"], "2026-10-08T19:00:00+01:00")


class ManagerNotebook(unittest.TestCase):
    def _save(self, root: Path, **kwargs: str) -> None:
        week = _hold(6)
        decision = dict(week)
        decision.pop("gw")
        result = validate_horizon(decision, [week, _hold(7), _hold(8)], _carry(), _market())
        write_plan(
            gw=6,
            deadline_utc=DEADLINE,
            frozen_at_utc="2026-10-09T18:00:00Z",
            model_id="test",
            prompt_sha256="a" * 64,
            context="Press notes only.",
            rationale="Hold.",
            decision=decision,
            horizon=[week, _hold(7), _hold(8)],
            carry=_carry(),
            result=result,
            root=root,
            notes=kwargs.get("notes", ""),
            adjustments=kwargs.get("adjustments", ""),
        )

    def test_a_banned_token_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(NotebookContaminationError):
                self._save(root, notes="expecting high score_xp")
            self.assertEqual(list(root.glob("*")), [])
            self.assertFalse((root / "notebook.jsonl").exists())

    def test_a_banned_token_leaves_an_existing_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            upsert_note(
                gw=6,
                written_at_utc="2026-10-09T18:00:00Z",
                notes="Wood is the cover.",
                adjustments="Watch the knee.",
                root=root,
            )
            blob = (root / "notebook.jsonl").read_bytes()
            with self.assertRaises(NotebookContaminationError):
                upsert_note(
                    gw=7,
                    written_at_utc="2026-10-16T18:00:00Z",
                    notes="xp_on_pot says otherwise",
                    adjustments="",
                    root=root,
                )
            self.assertEqual((root / "notebook.jsonl").read_bytes(), blob)

    def test_the_same_gameweek_replaces_its_row(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._save(root, notes="First note.", adjustments="Hold Wood.")
            self._save(root, notes="Revised note.", adjustments="Sell if he misses pressers.")
            lines = [
                line
                for line in (root / "notebook.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["notes"], "Revised note.")
            self.assertEqual(row["gw"], 6)

    def test_a_later_note_is_hidden_until_that_week_has_passed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            upsert_note(
                gw=7,
                written_at_utc="2026-10-16T18:00:00Z",
                notes="GW7 secret about the presser.",
                adjustments="Captain the home forward.",
                root=root,
            )
            earlier = notebook_section(6, root)
            later = notebook_section(8, root)
            self.assertNotIn("GW7 secret", earlier)
            self.assertEqual(earlier, "")
            self.assertIn("GW7 secret", later)
            self.assertIn("Gameweek 7", later)
