"""Open outlets for the string agent. They stay out of the minutes compile."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.live import news_packets as np
from src.live.news_packets import compile_player_xmi, content_sha256, load_gameweek_packets
from src.str_agent.extractor import build_context
from src.str_agent.sources import (
    StringSourceError,
    load_string_sources,
    validate_string_source,
)


DEADLINE = "2026-10-10T10:00:00Z"


def _note(**overrides: object) -> dict:
    raw = {
        "source_id": "reddit_fantasypl:gw06:haaland-thread",
        "source": "reddit_fantasypl",
        "outlet_class": "forum",
        "url": "https://www.reddit.com/r/FantasyPL/comments/haaland",
        "published_at_utc": "2026-10-08T18:00:00Z",
        "headline": "Haaland minutes thread",
        "body": "Several regulars say he was seen finishing training.",
        "player_ids": [411],
    }
    raw.update(overrides)
    raw["sha256"] = content_sha256(
        headline=str(raw["headline"]),
        body=str(raw["body"]),
        url=str(raw["url"]),
        published_at_utc=str(raw["published_at_utc"]),
    )
    return raw


class StringSources(unittest.TestCase):
    def test_forum_youtube_press_and_other_are_allowed(self) -> None:
        for outlet, source in (
            ("forum", "reddit_fantasypl"),
            ("youtube", "press_conference_channel"),
            ("press", "local_blog"),
            ("other", "club_podcast"),
        ):
            note = validate_string_source(
                _note(outlet_class=outlet, source=source, source_id=f"{source}:note"),
                deadline_utc=DEADLINE,
            )
            self.assertEqual(note.outlet_class, outlet)

    def test_press_whitelist_still_rejects_a_forum_name(self) -> None:
        with self.assertRaises(np.WhitelistError):
            np.validate_packet(
                {
                    "packet_id": "reddit:gw06:x",
                    "source": "reddit_fantasypl",
                    "url": "https://www.reddit.com/r/FantasyPL/x",
                    "published_at_utc": "2026-10-08T18:00:00Z",
                    "headline": "rumour",
                    "body": "rumour",
                    "player_ids": [],
                    "sha256": content_sha256(
                        headline="rumour",
                        body="rumour",
                        url="https://www.reddit.com/r/FantasyPL/x",
                        published_at_utc="2026-10-08T18:00:00Z",
                    ),
                },
                deadline_utc=DEADLINE,
            )

    def test_placeholder_and_non_http_urls_fail(self) -> None:
        for url in (
            "http://example.com/rumour",
            "http://localhost:8000/thread",
            "https://test.com/clip",
            "ftp://files.example.org/clip",
            "not a url",
        ):
            with self.assertRaises(StringSourceError):
                validate_string_source(_note(url=url), deadline_utc=DEADLINE)

    def test_published_after_the_deadline_fails(self) -> None:
        with self.assertRaises(StringSourceError):
            validate_string_source(
                _note(published_at_utc="2026-10-10T10:00:00Z"),
                deadline_utc=DEADLINE,
            )

    def test_recorded_after_the_deadline_fails(self) -> None:
        with self.assertRaises(StringSourceError):
            validate_string_source(
                _note(recorded_at_utc="2026-10-10T12:00:00Z"),
                deadline_utc=DEADLINE,
            )

    def test_body_over_the_cap_fails(self) -> None:
        with self.assertRaises(StringSourceError):
            validate_string_source(_note(body="a" * 2001), deadline_utc=DEADLINE)
        note = validate_string_source(_note(body="a" * 2000), deadline_utc=DEADLINE)
        self.assertEqual(len(note.body), 2000)

    def test_string_folder_does_not_move_xmi(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dest = root / "data" / "predictions" / "2026-27" / "gw06" / "string_sources"
            dest.mkdir(parents=True)
            (dest / "forum.json").write_text(json.dumps(_note()), encoding="utf-8")
            loaded = load_gameweek_packets(6, deadline_utc=DEADLINE, root=root, include_synthetic_fpl=False)
            self.assertEqual(loaded, [])
            notes = load_string_sources(6, deadline_utc=DEADLINE, root=root)
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].outlet_class, "forum")
        row = compile_player_xmi(
            player_id=411,
            name="Haaland",
            position="FWD",
            prior=90.0,
            status="a",
            chance=None,
            packets=[],
            minutes=[90, 90, 90],
        )
        self.assertEqual(row["tag"], "rolling avg")
        self.assertEqual(row["xmi_compiled"], 90.0)
        text = Path("src/live/news_packets.py").read_text(encoding="utf-8")
        self.assertNotIn("string_sources", text)

    def test_context_quotes_the_outlet_and_keeps_the_summary_last(self) -> None:
        forum = validate_string_source(_note(player_ids=[1]), deadline_utc=DEADLINE)
        video = validate_string_source(
            _note(
                source_id="yt:gw06:roundup",
                source="fpl_weekly",
                outlet_class="youtube",
                url="https://www.youtube.com/watch?v=abc123",
                headline="Gameweek roundup",
                body="The host talks through blanks without naming a starter.",
                player_ids=[],
            ),
            deadline_utc=DEADLINE,
        )
        text = build_context(
            gw=6,
            deadline_utc=DEADLINE,
            roster=[
                {
                    "player_id": 1,
                    "name": "Hold",
                    "position": "MID",
                    "club": "MCI",
                    "now_cost": 50,
                    "owned": True,
                }
            ],
            bank=15,
            ft=1,
            chips_left=["wildcard"],
            focus_ids=[1],
            directory={1: {"name": "Hold", "position": "MID", "club": "MCI", "prior": None}},
            packets=[],
            minutes={1: [90, 0, 90]},
            string_sources=[forum, video],
        )
        self.assertIn("[forum]", text)
        self.assertIn("seen finishing training", text)
        self.assertIn("## Other outlets", text)
        self.assertIn("[youtube]", text)
        self.assertIn("Sidecar summary: rolling avg", text)
        self.assertNotIn("score_xp", text)
        self.assertLess(text.index("seen finishing training"), text.index("Sidecar summary"))


if __name__ == "__main__":
    unittest.main()
