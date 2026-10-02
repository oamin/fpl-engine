"""Rules that solvers must not drift from."""

from __future__ import annotations

import unittest

from src.models.season_climb_budget import BUDGET, MAX_PER_CLUB, SQUAD_QUOTA
from src.models.season_climb_ft import HIT_COST, MAX_FT, advance_ft, sell_price
from src.rules.fpl_2026 import (
    BUDGET_TENTHS,
    OFFICIAL_FORMATIONS,
    ChipWallet,
    captain_extra_points,
    validate_chip_map,
    bonus_points,
    cbi_bps,
    defcon_points,
    gk_save_bps,
    goal_points,
    half_for_gw,
    hit_cost,
    squad_legal,
    tackled_bps,
    xi_legal,
)
from src.rules.fpl_2026 import (
    HIT_COST as RULES_HIT,
)
from src.rules.fpl_2026 import (
    MAX_FT as RULES_MAX_FT,
)
from src.rules.fpl_2026 import (
    MAX_PER_CLUB as RULES_CLUB,
)
from src.rules.fpl_2026 import (
    SQUAD_QUOTA as RULES_QUOTA,
)
from src.rules.fpl_2026 import (
    advance_ft as rules_advance_ft,
)
from src.rules.fpl_2026 import (
    sell_price as rules_sell_price,
)


def _squad(extra_club: str = "WHU") -> tuple[list[str], list[str], list[int]]:
    positions = ["GKP", "GKP", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD"]
    clubs = ["ARS", "CHE", "ARS", "LIV", "MCI", "TOT", "NEW", "ARS", "CHE", "LIV", "MCI", "BHA", "TOT", "NEW", extra_club]
    values = [50] * 15
    return positions, clubs, values


class RulesTest(unittest.TestCase):
    def test_climb_constants_match_rules(self) -> None:
        self.assertEqual(BUDGET, BUDGET_TENTHS)
        self.assertEqual(MAX_PER_CLUB, RULES_CLUB)
        self.assertEqual(dict(SQUAD_QUOTA), dict(RULES_QUOTA))
        self.assertEqual(MAX_FT, RULES_MAX_FT)
        self.assertEqual(HIT_COST, RULES_HIT)

    def test_sell_price_and_ft_match_climb(self) -> None:
        self.assertEqual(sell_price(50, 56), rules_sell_price(50, 56))
        self.assertEqual(sell_price(50, 45), 45)
        self.assertEqual(advance_ft(2, 0), rules_advance_ft(2, 0))
        self.assertEqual(advance_ft(1, 3), 1)
        self.assertEqual(advance_ft(5, 0), 5)

    def test_wildcard_keeps_bank(self) -> None:
        self.assertEqual(hit_cost(2, 8, chip="wildcard"), 0)
        self.assertEqual(rules_advance_ft(3, 8, chip="wildcard"), 3)
        self.assertEqual(rules_advance_ft(2, 0, chip="free_hit"), 2)
        self.assertEqual(rules_advance_ft(2, 0), 3)
        self.assertEqual(hit_cost(2, 4), 8)

    def test_chip_halves_expire(self) -> None:
        wallet = ChipWallet()
        wallet.play(19, "wildcard")
        self.assertEqual(half_for_gw(19), "H1")
        self.assertNotIn("wildcard", wallet.available(19))
        self.assertIn("wildcard", wallet.available(20))
        wallet.play(20, "wildcard")
        self.assertNotIn("wildcard", wallet.available(21))
        with self.assertRaises(ValueError):
            wallet.play(20, "free_hit")
        with self.assertRaises(ValueError):
            half_for_gw(0)
        with self.assertRaises(ValueError):
            ChipWallet().available(39)

    def test_wildcard_and_free_hit_wait_until_gw2(self) -> None:
        wallet = ChipWallet()
        self.assertNotIn("wildcard", wallet.available(1))
        self.assertNotIn("free_hit", wallet.available(1))
        self.assertIn("bench_boost", wallet.available(1))
        self.assertIn("triple_captain", wallet.available(1))
        with self.assertRaises(ValueError):
            wallet.play(1, "wildcard")

    def test_free_hit_cannot_be_consecutive(self) -> None:
        wallet = ChipWallet()
        wallet.play(19, "free_hit")
        self.assertNotIn("free_hit", wallet.available(20))
        with self.assertRaises(ValueError):
            wallet.play(20, "free_hit")
        wallet.play(21, "free_hit")

    def test_validate_chip_map_rejects_a_bad_plan(self) -> None:
        self.assertEqual(validate_chip_map(None), {})
        self.assertEqual(validate_chip_map({}), {})
        self.assertEqual(validate_chip_map({12: "bench_boost"}), {12: "bench_boost"})
        with self.assertRaises(ValueError):
            validate_chip_map({6: "wildcard", 10: "wildcard"})
        with self.assertRaises(ValueError):
            validate_chip_map({19: "free_hit", 20: "free_hit"})
        with self.assertRaises(ValueError):
            validate_chip_map({4: "not_a_chip"})

    def test_triple_captain_passes_to_the_vice(self) -> None:
        self.assertEqual(
            captain_extra_points(5, 4, captain_played=True, vice_played=True),
            5,
        )
        self.assertEqual(
            captain_extra_points(
                5, 4, captain_played=True, vice_played=True, chip="triple_captain"
            ),
            10,
        )
        self.assertEqual(
            captain_extra_points(
                5, 4, captain_played=False, vice_played=True, chip="triple_captain"
            ),
            8,
        )
        self.assertEqual(
            captain_extra_points(
                5, 4, captain_played=False, vice_played=False, chip="triple_captain"
            ),
            0,
        )

    def test_h1_chip_cannot_be_replayed_in_h1(self) -> None:
        wallet = ChipWallet()
        wallet.play(6, "triple_captain")
        self.assertNotIn("triple_captain", wallet.available(12))
        self.assertIn("triple_captain", wallet.available(25))

    def test_squad_and_xi(self) -> None:
        positions, clubs, values = _squad()
        self.assertTrue(squad_legal(positions, clubs, values))
        values[-1] = 400
        self.assertFalse(squad_legal(positions, clubs, values))
        positions, clubs, values = _squad("ARS")
        # four Arsenal if extra_club is ARS and ARS already appears thrice
        self.assertFalse(squad_legal(positions, clubs, values))
        self.assertTrue(xi_legal(["GKP", "DEF", "DEF", "DEF", "DEF", "DEF", "MID", "MID", "FWD", "FWD", "FWD"]))
        self.assertIn((5, 2, 3), OFFICIAL_FORMATIONS)
        self.assertNotIn((3, 2, 5), OFFICIAL_FORMATIONS)

    def test_scoring(self) -> None:
        self.assertEqual(goal_points("GK", 1), 6)
        self.assertEqual(goal_points("FWD", 2), 8)
        self.assertEqual(defcon_points("DEF", 10), 2)
        self.assertEqual(defcon_points("DEF", 9), 0)
        self.assertEqual(defcon_points("MID", 12), 2)
        self.assertEqual(defcon_points("GKP", 20), 0)
        self.assertEqual(cbi_bps(8), 2)
        self.assertEqual(tackled_bps(156), 0)
        self.assertEqual(
            gk_save_bps(
                open_play_saves=3,
                open_play_inside_box=2,
                open_play_big_chances=1,
                penalty_saves=1,
            ),
            2 * 3 + 2 + 1 + 8,
        )

    def test_bonus_ties(self) -> None:
        self.assertEqual(bonus_points({"a": 30, "b": 20, "c": 10}), {"a": 3, "b": 2, "c": 1})
        self.assertEqual(bonus_points({"a": 30, "b": 30, "c": 10}), {"a": 3, "b": 3, "c": 1})
        self.assertEqual(bonus_points({"a": 30, "b": 20, "c": 20}), {"a": 3, "b": 2, "c": 2})
        self.assertEqual(
            bonus_points({"a": 30, "b": 20, "c": 10, "d": 10}),
            {"a": 3, "b": 2, "c": 1, "d": 1},
        )
        self.assertEqual(
            bonus_points({"a": 10, "b": 10, "c": 10, "d": 4}),
            {"a": 3, "b": 3, "c": 3},
        )


if __name__ == "__main__":
    unittest.main()
