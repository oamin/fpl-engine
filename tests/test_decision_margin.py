"""The decision-margin calls, locked before the histograms were read."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.decision_margin import (
    bin_name,
    curse_call,
    margin,
    ranking_call,
    read_signings,
)


def _row(
    points: float,
    gap: float,
    block: str,
    place: str = "out",
    group: str = "veteran",
) -> dict:
    return {
        "kind": "chip_squad",
        "tag": "eligible",
        "points": points,
        "gap": gap,
        "block": block,
        "place": place,
        "group": group,
    }


class MarginTest(unittest.TestCase):
    def test_stored_gap_is_human_minus_model(self) -> None:
        self.assertAlmostEqual(margin(-0.4), 0.4)
        self.assertEqual(bin_name(0.0), "<=0")
        self.assertEqual(bin_name(0.5), "0-0.5")
        self.assertEqual(bin_name(0.5001), "0.5-1.25")
        self.assertEqual(bin_name(1.25), "0.5-1.25")
        self.assertEqual(bin_name(1.2501), ">1.25")

    def test_half_the_weight_is_the_bar(self) -> None:
        self.assertEqual(ranking_call({"0-0.5": 0.5, ">1.25": 0.1}), "small-margin")
        self.assertEqual(ranking_call({"0-0.5": 0.49, ">1.25": 0.5}), "wide-margin")
        self.assertEqual(ranking_call({"0-0.5": 0.49, ">1.25": 0.49}), "inconclusive")

    def test_curse_needs_every_season(self) -> None:
        clear = {"top_bias": 0.4, "eligible_bias": 0.1}
        self.assertEqual(curse_call([clear, clear, clear, clear]), "curse")
        short = {"top_bias": 0.3, "eligible_bias": 0.1}
        self.assertEqual(curse_call([clear, clear, clear, short]), "parked")
        negative = {"top_bias": -0.1, "eligible_bias": -0.4}
        self.assertEqual(curse_call([negative, negative, negative, negative]), "parked")


class FileTest(unittest.TestCase):
    def test_published_totals_still_split(self) -> None:
        frame = pd.read_csv("data/processed/chip_lead_players_gw15.csv")
        result = read_signings(frame)
        self.assertEqual(result["n_out"], 64)
        self.assertEqual(result["points_out"], -229.0)
        self.assertEqual(result["n_held"], 19)
        self.assertEqual(result["top"], "constraint")
        self.assertGreaterEqual(result["constrained_share"], 0.5)
        self.assertEqual(sum(result["bins"][name]["n"] for name in result["bins"]), result["n_free"])
