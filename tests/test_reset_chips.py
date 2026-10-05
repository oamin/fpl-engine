"""Free chip wallet on the reset. No copied week and no network."""

from __future__ import annotations

import inspect
import unittest

from src.models.reset_chips import StepOutlook, chip_phrases, choose_chip
from src.rules.fpl_2026 import ChipWallet


def _step(
    gw: int,
    *,
    held: float = 40.0,
    bench: float = 1.0,
    cap: float = 4.0,
    rebuilt: float = 40.0,
) -> StepOutlook:
    return StepOutlook(
        gw=gw,
        held_xi=held,
        bench_xp=bench,
        cap_xp=cap,
        rebuilt_xi=rebuilt,
        fh_xi=rebuilt,
    )


class ChoiceTest(unittest.TestCase):
    def test_triple_captain_plays_on_the_best_week(self) -> None:
        steps = [_step(2, cap=8.0), _step(3, cap=3.0), _step(4, cap=3.0)]
        chip, gain = choose_chip(steps, ("triple_captain", "bench_boost"))
        self.assertEqual(chip, "triple_captain")
        self.assertEqual(gain, 8.0)

    def test_a_later_captain_keeps_the_chip(self) -> None:
        steps = [_step(2, cap=3.0), _step(3, cap=8.0), _step(4, cap=1.0)]
        chip, _gain = choose_chip(steps, ("triple_captain",))
        self.assertIsNone(chip)

    def test_gameweek_1_refuses_a_wildcard(self) -> None:
        steps = [_step(1, rebuilt=80.0), _step(2, rebuilt=80.0), _step(3, rebuilt=80.0)]
        chip, _gain = choose_chip(steps, ("wildcard", "free_hit", "triple_captain", "bench_boost"))
        self.assertNotIn(chip, ("wildcard", "free_hit"))

    def test_wildcard_needs_16_on_the_priced_weeks(self) -> None:
        short = [_step(2, held=40, rebuilt=48), _step(3, held=40, rebuilt=46), _step(4, held=40, rebuilt=41)]
        self.assertIsNone(choose_chip(short, ("wildcard",))[0])
        clear = [_step(2, held=40, rebuilt=50), _step(3, held=40, rebuilt=48), _step(4, held=40, rebuilt=46)]
        chip, gain = choose_chip(clear, ("wildcard",))
        self.assertEqual(chip, "wildcard")
        self.assertEqual(gain, 24.0)

    def test_free_hit_needs_12(self) -> None:
        shy = [_step(3, held=40, rebuilt=51)]
        self.assertIsNone(choose_chip(shy, ("free_hit",))[0])
        clear = [_step(3, held=40, rebuilt=52)]
        self.assertEqual(choose_chip(clear, ("free_hit",))[0], "free_hit")

    def test_a_tie_plays_nothing(self) -> None:
        steps = [_step(4, cap=6.0, bench=6.0), _step(5, cap=1.0, bench=1.0)]
        chip, _gain = choose_chip(steps, ("triple_captain", "bench_boost"))
        self.assertIsNone(chip)

    def test_a_played_chip_leaves_the_wallet(self) -> None:
        wallet = ChipWallet()
        wallet.play(2, "triple_captain")
        self.assertNotIn("triple_captain", wallet.available(3))
        wallet.play(3, "free_hit")
        self.assertNotIn("free_hit", wallet.available(4))


class PhraseTest(unittest.TestCase):
    def test_the_bench_piece_is_in_the_sum(self) -> None:
        row = {
            "captain_gap": 4.0,
            "transfer_gap": 0.0,
            "lineup_gap": 0.0,
            "hit_gap": 0.0,
            "bench_gap": 9.0,
            "residual": 0.0,
            "model_captain": "Haaland",
            "their_captain": "Haaland",
            "transfer_model": [],
            "transfer_their": [],
            "lineup_model": [],
            "lineup_their": [],
            "model_in": [],
            "chip": "bench_boost",
            "his_chip": None,
            "gap": 13.0,
        }
        text = chip_phrases(row, {})
        signed = []
        for line in text:
            token = [part for part in line.replace(".", " ").split() if part[:1] in "+-"][-1]
            signed.append(float(token))
        self.assertAlmostEqual(sum(signed), 13.0)
        self.assertIn("Bench Boost added +9.", text)


class SourceTest(unittest.TestCase):
    def test_the_module_does_not_copy_the_half_or_call_the_odds_api(self) -> None:
        import src.models.reset_chips as reset_chips

        source = inspect.getsource(reset_chips)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("spread_outlooks", source)
        self.assertNotIn("run_ft_season", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("half_plan_scores.csv", source)
        self.assertNotIn("reset_gap_gw15.md", source)


if __name__ == "__main__":
    unittest.main()
