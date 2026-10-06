"""The live news tag uses a note only when it is dated before the deadline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.live.news_tags import (
    PacketDoc,
    accept_tag,
    bulk_seconds,
    build_tag_prompt,
    docs_before,
    fpl_record,
    minutes_for_tag,
    parse_tag_response,
    prior_minutes,
    published_before,
    tag_from_fpl,
    NewsTagError,
)

ROOT = Path(__file__).resolve().parents[1]
GW2 = "2026-08-28T17:30:00Z"
GW3 = "2026-09-04T17:30:00Z"


def _element(news_added: str, status: str = "u", chance: int | None = 0, news: str = "") -> dict:
    return {
        "news_added": news_added,
        "status": status,
        "chance_of_playing_next_round": chance,
        "news": news,
    }


class DateGateTest(unittest.TestCase):
    def test_a_timestamp_must_be_strictly_before_the_deadline(self) -> None:
        self.assertTrue(published_before("2026-08-27T18:26:00Z", GW2))
        self.assertFalse(published_before(GW2, GW2))
        self.assertFalse(published_before("2026-08-30T16:25:00Z", GW2))

    def test_a_bare_date_on_the_deadline_day_is_excluded(self) -> None:
        self.assertTrue(published_before("2026-09-01", GW3))
        self.assertFalse(published_before("2026-09-04", GW3))

    def test_the_gameweek_2_packet_drops_the_later_chelsea_notes(self) -> None:
        docs = json.loads((ROOT / "data" / "live" / "news_docs_gw15.json").read_text())
        kept = {doc["doc_id"] for doc in docs_before(docs, GW2)}
        self.assertEqual(kept, {"athletic-martinez-2026-08-27", "guardian-deal-2026-08-27"})

    def test_a_shared_second_is_not_a_news_time(self) -> None:
        elements = [_element(f"2026-07-23T12:01:23.{index:06d}Z") for index in range(10)]
        elements.append(_element("2026-09-02T16:20:41.841936Z"))
        bulk = bulk_seconds(elements)
        self.assertIn("2026-07-23T12:01:23", bulk)
        self.assertIsNone(fpl_record(elements[0], GW3, bulk))
        visible = fpl_record(elements[-1], GW3, bulk)
        self.assertIsNotNone(visible)
        self.assertIsNone(fpl_record(elements[-1], GW2, bulk))


class MinutesTest(unittest.TestCase):
    def test_the_tag_map(self) -> None:
        self.assertEqual(minutes_for_tag("firm_starter", "GKP", 90, None, "a"), 90.0)
        self.assertEqual(minutes_for_tag("firm_starter", "DEF", 40, None, "a"), 65.0)
        self.assertEqual(minutes_for_tag("firm_starter", "DEF", 95, None, "a"), 90.0)
        self.assertEqual(minutes_for_tag("injured", "FWD", 90, 0, "i"), 0.0)
        self.assertEqual(minutes_for_tag("injured", "FWD", 80, 75, "d"), 60.0)
        self.assertEqual(minutes_for_tag("transferred", "GKP", 90, None, "a"), 0.0)
        self.assertEqual(minutes_for_tag("benched", "GKP", 90, None, "a"), 0.0)
        self.assertEqual(minutes_for_tag("benched", "DEF", 90, None, "a"), 15.0)
        self.assertIsNone(minutes_for_tag("ask", "GKP", 90, None, "a"))
        self.assertIsNone(minutes_for_tag(None, "GKP", 90, None, "a"))

    def test_unavailable_forces_zero(self) -> None:
        self.assertEqual(minutes_for_tag("firm_starter", "DEF", 90, 100, "u"), 0.0)
        self.assertEqual(minutes_for_tag("benched", "DEF", 90, None, "u"), 0.0)
        self.assertEqual(minutes_for_tag("injured", "MID", 90, 0, "d"), 0.0)

    def test_a_ruled_out_fpl_line_is_not_read_before_its_timestamp(self) -> None:
        element = _element(
            "2026-09-02T16:20:41.841936Z",
            news="Has joined Como on loan for the rest of the season",
        )
        self.assertIsNone(fpl_record(element, GW2, frozenset()))
        record = fpl_record(element, GW3, frozenset())
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(tag_from_fpl(record), "transferred")
        self.assertEqual(minutes_for_tag("transferred", "GKP", 90, record["chance"], record["status"]), 0.0)

    def test_doubtful_keeps_a_share_of_the_old_minutes(self) -> None:
        record = {
            "status": "d",
            "chance": 75.0,
            "news": "Knee injury - 75% chance of playing",
        }
        self.assertEqual(tag_from_fpl(record), "injured")
        self.assertEqual(prior_minutes([90, 0, 90, 90]), 90.0)
        self.assertEqual(minutes_for_tag("injured", "FWD", 90, 75, "d"), 67.5)

    def test_not_included_is_a_bench_and_unavailable_is_still_zero(self) -> None:
        record = {"status": "u", "chance": 0, "news": "not included in squad."}
        self.assertEqual(tag_from_fpl(record), "benched")
        self.assertEqual(minutes_for_tag("benched", "FWD", 90, 0, "u"), 0.0)


class CompletionTest(unittest.TestCase):
    def test_a_late_citation_is_rejected(self) -> None:
        docs = [PacketDoc("guardian-deal-2026-08-27", "Sources believe Sanchez may need to be moved on.")]
        with self.assertRaises(NewsTagError):
            accept_tag(
                {
                    "player_id": 140,
                    "gw": 2,
                    "tag": "transferred",
                    "doc_ids": ["bbc-como-2026-09-01"],
                    "note": "left",
                },
                docs,
                position="GKP",
                prior=90,
                status=None,
                chance=None,
                name="Sánchez",
            )

    def test_an_agreed_deal_does_not_support_a_bench(self) -> None:
        docs = [
            PacketDoc(
                "guardian-deal-2026-08-27",
                "Chelsea have agreed a deal for Emiliano Martinez, who is due to sign. "
                "Sources believe Sanchez may need to be moved on before the window shuts.",
            )
        ]
        with self.assertRaises(NewsTagError):
            accept_tag(
                {"player_id": 140, "gw": 2, "tag": "benched", "doc_ids": ["guardian-deal-2026-08-27"], "note": ""},
                docs,
                position="GKP",
                prior=90,
                status=None,
                chance=None,
                name="Sánchez",
            )

    def test_the_loan_sentence_supports_a_transfer(self) -> None:
        docs = [
            PacketDoc(
                "bbc-como-2026-09-01",
                "Robert Sanchez has joined Como on a season-long loan. "
                "Emiliano Martinez started Chelsea's match against Brighton.",
            )
        ]
        decision = accept_tag(
            {
                "player_id": 140,
                "gw": 3,
                "tag": "transferred",
                "doc_ids": ["bbc-como-2026-09-01"],
                "note": "left",
            },
            docs,
            position="GKP",
            prior=90,
            status="u",
            chance=0,
            name="Sánchez",
        )
        self.assertEqual(decision.tag, "transferred")
        self.assertEqual(decision.xmi, 0.0)

    def test_the_same_article_can_name_the_other_goalkeeper_as_a_starter(self) -> None:
        docs = [
            PacketDoc(
                "bbc-como-2026-09-01",
                "Robert Sanchez has joined Como on a season-long loan. "
                "Emiliano Martinez started Chelsea's match against Brighton.",
            )
        ]
        decision = accept_tag(
            {"player_id": 28, "gw": 3, "tag": "firm_starter", "doc_ids": ["bbc-como-2026-09-01"], "note": ""},
            docs,
            position="GKP",
            prior=90,
            status=None,
            chance=None,
            name="Martinez",
        )
        self.assertEqual(decision.xmi, 90.0)
        with self.assertRaises(NewsTagError):
            accept_tag(
                {"player_id": 140, "gw": 3, "tag": "firm_starter", "doc_ids": ["bbc-como-2026-09-01"], "note": ""},
                docs,
                position="GKP",
                prior=90,
                status=None,
                chance=None,
                name="Sánchez",
            )

    def test_ask_leaves_the_minutes_unchanged(self) -> None:
        decision = accept_tag(
            {"player_id": 140, "gw": 2, "tag": "ask", "doc_ids": [], "note": "deal only"},
            [],
            position="GKP",
            prior=90,
            status=None,
            chance=None,
            name="Sánchez",
        )
        self.assertIsNone(decision.xmi)
        self.assertEqual(decision.tag, "ask")

    def test_a_fenced_array_parses(self) -> None:
        parsed = parse_tag_response('```json\n[{"player_id": 140, "gw": 2}]\n```')
        self.assertEqual(parsed, [{"player_id": 140, "gw": 2}])
        with self.assertRaises(NewsTagError):
            parse_tag_response("He is benched.")

    def test_the_prompt_contains_only_the_packet(self) -> None:
        prompt = build_tag_prompt(
            [
                {
                    "player_id": 140,
                    "name": "Sánchez",
                    "position": "GKP",
                    "gw": 2,
                    "deadline": GW2,
                    "docs": [PacketDoc("guardian-deal-2026-08-27", "agreed a deal")],
                }
            ]
        )
        self.assertIn("guardian-deal-2026-08-27", prompt)
        self.assertNotIn("bbc-como", prompt)
        self.assertIn("Do not use memory", prompt)


if __name__ == "__main__":
    unittest.main()
