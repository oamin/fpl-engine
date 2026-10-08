"""The string dossier states the deadline, the chip bank, and the half calendar."""

from __future__ import annotations

import unittest

from src.str_agent.dossier import club_calendar, deadline_brief
from src.str_agent.horizon import HORIZON_WEEKS
from src.str_agent.prompt import SYSTEM


def _line(text: str, club: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"- {club}:"):
            return line
    raise AssertionError(club)


class Dossier(unittest.TestCase):
    def test_blank_unknown_and_a_double_stay_distinct(self) -> None:
        names = {1: "ARS", 2: "LEE", 3: "MCI", 4: "CHE"}
        fixtures = [
            {
                "event": 6,
                "team_h": 1,
                "team_a": 2,
                "team_h_difficulty": 3,
                "team_a_difficulty": 5,
            },
            {
                "event": 6,
                "team_h": 1,
                "team_a": 3,
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            },
            {
                "event": 7,
                "team_h": 3,
                "team_a": 4,
                "team_h_difficulty": 2,
                "team_a_difficulty": 2,
            },
        ]
        text = club_calendar(fixtures, names, 6)
        arsenal = _line(text, "ARS")
        self.assertIn("GW6 home LEE 3, home MCI 2", arsenal)
        self.assertIn("GW7 blank", arsenal)
        self.assertIn("GW8 unknown", arsenal)
        self.assertNotIn("GW8 blank", text)
        self.assertNotIn("GW7 unknown", arsenal)
        self.assertIn("three weeks", text)
        for token in ("score_xp", "xp_on_pot", "lam_scored", "ep_next"):
            self.assertNotIn(token, text)

    def test_the_chip_bank_names_what_is_left_in_this_half(self) -> None:
        text = deadline_brief(
            gw=6,
            deadline_utc="2026-10-10T10:00:00Z",
            bank=15,
            ft=1,
            chips_played={1: "triple_captain"},
        )
        self.assertIn("Gameweek 6 of 38", text)
        self.assertIn("Half H1", text)
        self.assertIn("expire at the Gameweek 19 deadline", text)
        self.assertIn("Chips already played: GW1 triple_captain.", text)
        self.assertIn("Chips still available: wildcard, free_hit, bench_boost.", text)
        self.assertNotIn("score_xp", text)

    def test_the_prompt_still_asks_for_three_weeks(self) -> None:
        self.assertEqual(HORIZON_WEEKS, 3)
        self.assertIn("three weeks", SYSTEM)
        self.assertIn("not extra weeks", SYSTEM)
