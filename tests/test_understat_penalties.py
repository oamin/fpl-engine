"""Name join and sheet columns for Understat penalty shots."""

from __future__ import annotations

import unittest

from src.ingest.understat_penalties import (
    RosterPlayer,
    _pool,
    append_penalty_columns,
    fixture_cells,
    fold_person,
    match_element,
    penalty_rows_from_match,
)
from src.teams import norm_team


def _player(
    element: int,
    name: str,
    team: str,
    web: str,
    first: str,
    second: str,
) -> RosterPlayer:
    return RosterPlayer(element, name, norm_team(team), web, first, second)


ROSTER = [
    _player(1, "Bruno Borges Fernandes", "Man Utd", "B.Fernandes", "Bruno", "Borges Fernandes"),
    _player(2, "Lucas Tolentino Coelho de Lima", "West Ham", "L.Paquetá", "Lucas", "Tolentino Coelho de Lima"),
    _player(3, "Dominic Solanke-Mitchell", "Spurs", "Solanke", "Dominic", "Solanke-Mitchell"),
    _player(4, "Martin Ødegaard", "Arsenal", "Ødegaard", "Martin", "Ødegaard"),
    _player(5, "Matt O'Riley", "Brighton", "O'Riley", "Matt", "O'Riley"),
    _player(6, "João Pedro Junqueira de Jesus", "Brighton", "João Pedro", "João Pedro", "Junqueira de Jesus"),
    _player(7, "João Pedro Ferreira Silva", "Nott'm Forest", "Jota Silva", "João Pedro", "Ferreira Silva"),
    _player(8, "Pedro Porro", "Spurs", "Pedro Porro", "Pedro", "Porro"),
    _player(9, "Rayan Cherki", "Man City", "Cherki", "Rayan", "Cherki"),
    _player(10, "Jacob Murphy", "Newcastle", "Murphy", "Jacob", "Murphy"),
    _player(11, "Junior Kroupi", "Bournemouth", "Kroupi.Jr", "Junior", "Kroupi"),
    _player(12, "Mohamed Salah", "Liverpool", "M.Salah", "Mohamed", "Salah"),
]


class FoldAndMatchTest(unittest.TestCase):
    def test_fold_drops_accent_stroke_and_apostrophe(self) -> None:
        self.assertEqual(fold_person("Martin Ødegaard"), "martin odegaard")
        self.assertEqual(fold_person("Matt O&#039;Riley"), "matt oriley")
        self.assertEqual(fold_person("Lucas Paquetá"), "lucas paqueta")

    def test_unique_legal_and_web_names(self) -> None:
        self.assertEqual(match_element("Bruno Fernandes", "Manchester United", ROSTER), 1)
        self.assertEqual(match_element("Lucas Paquetá", "West Ham", ROSTER), 2)
        self.assertEqual(match_element("Dominic Solanke", "Tottenham", ROSTER), 3)
        self.assertEqual(match_element("Martin Odegaard", "Arsenal", ROSTER), 4)
        self.assertEqual(match_element("Matt O&#039;Riley", "Brighton", ROSTER), 5)
        self.assertEqual(match_element("João Pedro", "Brighton", ROSTER), 6)
        self.assertEqual(match_element("Mohamed Salah", "Liverpool", ROSTER), 12)

    def test_shared_token_on_another_player_is_not_a_guess(self) -> None:
        self.assertIsNone(match_element("Mathis Cherki", "Manchester City", ROSTER))
        self.assertIsNone(match_element("Jacob Ramsey", "Newcastle United", ROSTER))
        self.assertEqual(match_element("Eli Junior Kroupi", "Bournemouth", ROSTER), 11)
        crowded = ROSTER + [
            _player(13, "Pedro Cardoso de Lima", "Spurs", "Pedro Lima", "Pedro", "Cardoso de Lima"),
        ]
        self.assertIsNone(match_element("Pedro", "Spurs", crowded))

    def test_matchday_club_not_the_club_he_left(self) -> None:
        city = RosterPlayer(360, "James McAtee", "man city", "McAtee", "James", "McAtee", "2023-08-12")
        sheff = RosterPlayer(360, "James McAtee", "sheffield utd", "McAtee", "James", "McAtee", "2024-02-10")
        rows = [city, sheff]
        self.assertEqual(
            match_element("James McAtee", "Sheffield United", _pool(rows, "2024-02-10", "Sheffield United")),
            360,
        )
        self.assertIsNone(
            match_element("James McAtee", "Sheffield United", _pool(rows, "2023-08-12", "Sheffield United"))
        )

    def test_joao_pedro_stays_on_his_own_club(self) -> None:
        self.assertEqual(match_element("João Pedro", "Nottingham Forest", ROSTER), 7)
        self.assertNotEqual(match_element("João Pedro", "Brighton", ROSTER), 7)


class FixtureCellsTest(unittest.TestCase):
    def test_full_join_writes_zeroes_for_the_rest_of_the_fixture(self) -> None:
        matches = [
            {
                "ok": True,
                "date": "2024-08-16",
                "home": "Liverpool",
                "away": "Ipswich",
                "pens": [
                    {
                        "player": "Mohamed Salah",
                        "team": "Liverpool",
                        "result": "Goal",
                        "xg": 0.76,
                        "season": "2024-25",
                        "date": "2024-08-16",
                    }
                ],
            }
        ]
        rows = [
            ("2024-08-16", 12, "liverpool"),
            ("2024-08-16", 99, "liverpool"),
            ("2024-08-16", 3, "spurs"),
        ]
        cells, unmatched = fixture_cells(matches, ROSTER, rows)
        self.assertEqual(unmatched, [])
        self.assertEqual(cells[( "2024-08-16", 12)], ("1", "1", "0.76"))
        self.assertEqual(cells[("2024-08-16", 99)], ("0", "0", "0"))
        self.assertNotIn(("2024-08-16", 3), cells)

    def test_registered_spelling_joins_kroupi(self) -> None:
        matches = [
            {
                "ok": True,
                "date": "2025-08-16",
                "home": "Bournemouth",
                "away": "Liverpool",
                "pens": [
                    {
                        "player": "Eli Junior Kroupi",
                        "team": "Bournemouth",
                        "result": "Goal",
                        "xg": 0.76,
                        "season": "2025-26",
                        "date": "2025-08-16",
                    }
                ],
            }
        ]
        rows = [("2025-08-16", 11, "bournemouth"), ("2025-08-16", 12, "liverpool")]
        cells, unmatched = fixture_cells(matches, ROSTER, rows)
        self.assertEqual(unmatched, [])
        self.assertEqual(cells[("2025-08-16", 11)], ("1", "1", "0.76"))
        self.assertEqual(cells[("2025-08-16", 12)], ("0", "0", "0"))

    def test_unmatched_shot_does_not_zero_the_fixture(self) -> None:
        matches = [
            {
                "ok": True,
                "date": "2025-08-16",
                "home": "Manchester City",
                "away": "Liverpool",
                "pens": [
                    {
                        "player": "Mathis Cherki",
                        "team": "Manchester City",
                        "result": "Goal",
                        "xg": 0.76,
                        "season": "2025-26",
                        "date": "2025-08-16",
                    }
                ],
            }
        ]
        rows = [("2025-08-16", 9, "man city"), ("2025-08-16", 12, "liverpool")]
        cells, unmatched = fixture_cells(matches, ROSTER, rows)
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(cells, {})

    def test_saved_penalty_is_taken_and_not_scored(self) -> None:
        matches = [
            {
                "ok": True,
                "date": "2024-09-01",
                "home": "Arsenal",
                "away": "Brighton",
                "pens": [
                    {
                        "player": "Martin Odegaard",
                        "team": "Arsenal",
                        "result": "SavedShot",
                        "xg": 0.76,
                        "season": "2024-25",
                        "date": "2024-09-01",
                    }
                ],
            }
        ]
        rows = [("2024-09-01", 4, "arsenal")]
        cells, _unmatched = fixture_cells(matches, ROSTER, rows)
        self.assertEqual(cells[("2024-09-01", 4)], ("1", "0", "0.76"))


class SheetAppendTest(unittest.TestCase):
    def test_appends_without_rewriting_existing_cells(self) -> None:
        raw = [
            "name,element,kickoff_time,total_points",
            "Mohamed Salah,12,2024-08-16T19:00:00Z,5",
            "Other,99,2024-08-16T19:00:00Z,2",
        ]
        cells = {("2024-08-16", 12): ("1", "1", "0.76")}
        out, stats = append_penalty_columns(raw, cells)
        self.assertEqual(stats["filled"], 1)
        self.assertEqual(stats["blank"], 1)
        self.assertTrue(out[1].startswith("Mohamed Salah,12,2024-08-16T19:00:00Z,5,"))
        self.assertTrue(out[1].endswith(",1,1,0.76"))
        self.assertTrue(out[2].endswith(",,,"))
        again, _stats = append_penalty_columns(out, cells)
        self.assertEqual(again[1], out[1])


class ShotParseTest(unittest.TestCase):
    def test_keeps_only_penalties(self) -> None:
        payload = {
            "shots": {
                "h": [
                    {
                        "id": "1",
                        "situation": "OpenPlay",
                        "player": "Mohamed Salah",
                        "h_a": "h",
                        "h_team": "Liverpool",
                        "a_team": "Ipswich",
                        "result": "Goal",
                        "xG": "0.4",
                        "date": "2024-08-16 19:00:00",
                        "match_id": "9",
                        "minute": "12",
                        "player_id": "3",
                    },
                    {
                        "id": "2",
                        "situation": "Penalty",
                        "player": "Mohamed Salah",
                        "h_a": "h",
                        "h_team": "Liverpool",
                        "a_team": "Ipswich",
                        "result": "MissedShots",
                        "xG": "0.76",
                        "date": "2024-08-16 19:00:00",
                        "match_id": "9",
                        "minute": "80",
                        "player_id": "3",
                    },
                ],
                "a": [],
            }
        }
        rows = penalty_rows_from_match(payload, "2024-25")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["result"], "MissedShots")
        self.assertEqual(rows[0]["team"], "Liverpool")
        self.assertEqual(rows[0]["date"], "2024-08-16")


if __name__ == "__main__":
    unittest.main()
