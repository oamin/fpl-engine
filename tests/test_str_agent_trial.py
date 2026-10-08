"""The Gameweek 6 dry run starts from the Gameweek 5 fifteen."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.live.entry import load_entry
from src.str_agent.trial import judge, prepare


class Trial(unittest.TestCase):
    def test_starting_squad_is_the_last_played_fifteen(self) -> None:
        pack = prepare(2632584)
        entry = load_entry(2632584)
        week = next(row for row in entry["gameweeks"] if int(row["gw"]) == 5)
        ids = [str(player["id"]) for player in week["xi"] + week["bench"]]
        self.assertEqual(list(pack.carry.squad), ids)
        self.assertEqual(pack.carry.gw, 6)
        self.assertEqual(pack.carry.ft_before, 1)
        self.assertEqual(pack.carry.bank, 15)
        self.assertEqual(len(pack.carry.purchase_prices), 15)
        self.assertIn("Starting squad: the Gameweek 5 fifteen", pack.context)
        self.assertNotIn("score_xp", pack.context)

    def test_saved_gw6_choice_is_a_legal_move(self) -> None:
        pack = prepare(2632584)
        payload = json.loads(
            Path("reports/string_agent_gw6_decision.json").read_text(encoding="utf-8")
        )
        result = judge(pack, payload["decision"])
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["is_legal"])
        self.assertEqual(result["hits"], 0)
        self.assertEqual(result["bank_after"], 2)
        self.assertEqual(result["named"]["transfers_out"], ["Scarlett"])
        self.assertEqual(result["named"]["transfers_in"], ["Wood"])
        self.assertEqual(result["named"]["captain"], "Ødegaard")


if __name__ == "__main__":
    unittest.main()
