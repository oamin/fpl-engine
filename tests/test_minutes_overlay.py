"""Availability / minutes overlay."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.minutes_overlay import (
    apply_availability_overlay,
    availability_factor,
    overlay_summary,
)


class AvailabilityFactorTest(unittest.TestCase):
    def test_chance_next_preferred(self) -> None:
        el = {
            "status": "d",
            "chance_of_playing_this_round": 25,
            "chance_of_playing_next_round": 75,
        }
        self.assertAlmostEqual(availability_factor(el, prefer_next=True), 0.75)
        self.assertAlmostEqual(availability_factor(el, prefer_next=False), 0.25)

    def test_status_defaults(self) -> None:
        self.assertAlmostEqual(availability_factor({"status": "a"}), 1.0)
        self.assertAlmostEqual(availability_factor({"status": "i"}), 0.0)
        self.assertAlmostEqual(availability_factor({"status": "d"}), 0.5)


class ApplyOverlayTest(unittest.TestCase):
    def test_scales_involvement(self) -> None:
        pool = pd.DataFrame(
            [
                {
                    "player_id": "1",
                    "name": "Fit",
                    "xmi": 90.0,
                    "share_xG": 0.4,
                    "share_xA": 0.2,
                    "exp_defcon_hit": 0.5,
                },
                {
                    "player_id": "2",
                    "name": "Hurt",
                    "xmi": 90.0,
                    "share_xG": 0.4,
                    "share_xA": 0.2,
                    "exp_defcon_hit": 0.5,
                },
            ]
        )
        boot = {
            "elements": [
                {"id": 1, "status": "a", "chance_of_playing_next_round": 100},
                {"id": 2, "status": "i", "chance_of_playing_next_round": 0},
            ]
        }
        out = apply_availability_overlay(pool, boot)
        fit = out.loc[out["player_id"] == "1"].iloc[0]
        hurt = out.loc[out["player_id"] == "2"].iloc[0]
        self.assertAlmostEqual(float(fit["xmi"]), 90.0)
        self.assertAlmostEqual(float(fit["share_xG"]), 0.4)
        self.assertAlmostEqual(float(hurt["xmi"]), 0.0)
        self.assertAlmostEqual(float(hurt["share_xG"]), 0.0)
        summary = overlay_summary(out)
        self.assertEqual(summary["n_zeroed"], 1)
        self.assertEqual(summary["n_scaled"], 1)


if __name__ == "__main__":
    unittest.main()
