"""FFScout team-news matching + selectable pool filters."""

from __future__ import annotations

import unittest

import pandas as pd

from src.live.ffscout_team_news import _norm_player, match_ffs_name_to_fpl, parse_team_news
from src.models.selectable_pool import apply_selectable_pool, build_selectable_pool


class NormPlayerTest(unittest.TestCase):
    def test_transliterates_o_slash_and_turkish_i(self) -> None:
        self.assertEqual(_norm_player("Ødegaard"), "odegaard")
        self.assertEqual(_norm_player("Odegaard"), "odegaard")
        self.assertEqual(_norm_player("Kadıoğlu"), "kadioglu")
        self.assertEqual(_norm_player("Kadioglu"), "kadioglu")


class MatchFfsNameTest(unittest.TestCase):
    def test_odegaard(self) -> None:
        cands = [
            {"id": 15, "web_name": "Ødegaard", "first_name": "Martin", "second_name": "Ødegaard"},
        ]
        hit = match_ffs_name_to_fpl("Odegaard", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 15)

    def test_bruno_fernandes_compound_surname(self) -> None:
        cands = [
            {
                "id": 426,
                "web_name": "B.Fernandes",
                "first_name": "Bruno",
                "second_name": "Borges Fernandes",
            },
        ]
        hit = match_ffs_name_to_fpl("Bruno Fernandes", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 426)

    def test_alisson_first_name(self) -> None:
        cands = [
            {"id": 350, "web_name": "A.Becker", "first_name": "Alisson", "second_name": "Becker"},
        ]
        hit = match_ffs_name_to_fpl("Alisson", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 350)

    def test_murphy_parenthetical_disambiguates(self) -> None:
        cands = [
            {"id": 451, "web_name": "A.Murphy", "first_name": "Alex", "second_name": "Murphy"},
            {"id": 457, "web_name": "J.Murphy", "first_name": "Jacob", "second_name": "Murphy"},
        ]
        hit = match_ffs_name_to_fpl("Murphy (Jacob)", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 457)

    def test_jay_da_silva_compact(self) -> None:
        cands = [
            {"id": 176, "web_name": "Dasilva", "first_name": "Jay", "second_name": "Dasilva"},
        ]
        hit = match_ffs_name_to_fpl("Jay da Silva", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 176)

    def test_pino_partial_surname(self) -> None:
        cands = [
            {
                "id": 211,
                "web_name": "Yeremy",
                "first_name": "Yéremy",
                "second_name": "Pino Santos",
            },
        ]
        hit = match_ffs_name_to_fpl("Pino", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 211)

    def test_kadioglu_turkish(self) -> None:
        cands = [
            {
                "id": 113,
                "web_name": "F.Kadıoğlu",
                "first_name": "Ferdi",
                "second_name": "Kadıoğlu",
            },
        ]
        hit = match_ffs_name_to_fpl("Kadioglu", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 113)

    def test_bruno_guimaraes_compound(self) -> None:
        cands = [
            {
                "id": 452,
                "web_name": "Bruno G.",
                "first_name": "Bruno",
                "second_name": "Guimarães Rodriguez Moura",
            },
        ]
        hit = match_ffs_name_to_fpl("Bruno Guimarães", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 452)

    def test_chema_andres_surname_token(self) -> None:
        cands = [
            {
                "id": 639,
                "web_name": "Andrés",
                "first_name": "José María",
                "second_name": "Andrés Baixauli",
            },
        ]
        hit = match_ffs_name_to_fpl("Chema Andrés", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 639)

    def test_murphy_ignores_same_first_other_surname(self) -> None:
        cands = [
            {"id": 451, "web_name": "A.Murphy", "first_name": "Alex", "second_name": "Murphy"},
            {"id": 456, "web_name": "J.Ramsey", "first_name": "Jacob", "second_name": "Ramsey"},
            {"id": 457, "web_name": "J.Murphy", "first_name": "Jacob", "second_name": "Murphy"},
        ]
        hit = match_ffs_name_to_fpl("Murphy (Jacob)", cands)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["id"], 457)


class SelectablePoolFilterTest(unittest.TestCase):
    def test_apply_restricts_eligible(self) -> None:
        feat = pd.DataFrame(
            {
                "player_id": ["1", "2", "2025-26:3"],
                "eligible": [True, True, True],
            }
        )
        out = apply_selectable_pool(feat, {"1", "3"})
        self.assertTrue(bool(out.loc[0, "eligible"]))
        self.assertFalse(bool(out.loc[1, "eligible"]))
        self.assertTrue(bool(out.loc[2, "eligible"]))

    def test_firm_and_minutes_gate(self) -> None:
        html = """
        <div data-team-code="ars">
        Next Match:
        Nottingham Forest (A)
        Raya
        Odegaard
        Out:
        Doubts:
        Banned:
        Latest News:
        none
        </div>
        """
        bootstrap = {
            "teams": [{"id": 1, "name": "Arsenal", "short_name": "ARS"}],
            "elements": [
                {
                    "id": 1,
                    "team": 1,
                    "web_name": "Raya",
                    "first_name": "David",
                    "second_name": "Raya",
                    "element_type": 1,
                    "minutes": 450,
                    "starts": 5,
                    "now_cost": 55,
                    "status": "a",
                },
                {
                    "id": 15,
                    "team": 1,
                    "web_name": "Ødegaard",
                    "first_name": "Martin",
                    "second_name": "Ødegaard",
                    "element_type": 3,
                    "minutes": 100,
                    "starts": 5,
                    "now_cost": 80,
                    "status": "a",
                },
            ],
        }
        from dataclasses import asdict

        ffscout = {"clubs": [asdict(c) for c in parse_team_news(html)]}
        result = build_selectable_pool(bootstrap=bootstrap, ffscout=ffscout, min_apps=2)
        # Raya mean mins = 90 > 60; Odegaard = 20 ≤ 60 → only Raya selectable
        self.assertEqual(set(result["selectable_ids"]), {"1"})
        self.assertEqual(result["n_clubs_with_firm"], 1)
        firm_ids = {str(p["player_id"]) for p in result["clubs"][0]["firm_players"]}
        self.assertEqual(firm_ids, {"1", "15"})
        self.assertIn("15", result["clubs"][0]["firm_selection_ids"])


if __name__ == "__main__":
    unittest.main()
