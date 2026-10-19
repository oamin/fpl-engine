"""Odds API anytime props → FPL id match + score_team_side rescale."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.live.odds_props import match_anytime_props, side_moments_from_fit
from src.markets import numagent_xp
from src.markets.prop_shares import anytime_p_to_lambda

ROOT = Path(__file__).resolve().parents[1]
BOOT = ROOT / "data" / "predictions" / "2026-27" / "gw06" / "bootstrap_20261007T220914Z.json"
PROBE = ROOT / "data" / "scratch" / "gw07_odds_probe.json"


class OddsPropsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bootstrap = json.loads(BOOT.read_text(encoding="utf-8"))

    def test_match_brentford_liverpool_names(self) -> None:
        props = [
            {"player": "Igor Thiago Nascimento Rodrigues", "p_anytime": 0.46},
            {"player": "Alexander Isak", "p_anytime": 0.37},
            {"player": "Callum Wilson", "p_anytime": 0.38},
            {"player": "Not A Real Player XYZ", "p_anytime": 0.20},
        ]
        m = match_anytime_props(
            props, home="Brentford", away="Liverpool", bootstrap=self.bootstrap
        )
        self.assertTrue(m["ok"])
        self.assertEqual(m["n_matched"], 3)
        self.assertEqual(m["n_unmatched"], 1)
        self.assertIn("106", m["by_id"])  # Thiago
        self.assertIn("379", m["by_id"])  # Isak
        self.assertIn("108", m["by_id"])  # Callum Wilson (Brentford)

    def test_side_moments_fallback_xgc(self) -> None:
        mom = side_moments_from_fit(
            {
                "xg_home": 1.4,
                "xg_away": 1.5,
                "xcs_home": 0.2,
                "xcs_away": 0.25,
            }
        )
        self.assertAlmostEqual(mom["home"]["xg"], 1.4)
        self.assertAlmostEqual(mom["home"]["xgc"], 0.75)

    def test_score_team_side_prop_lifts_returner(self) -> None:
        players = pd.DataFrame(
            {
                "player_id": ["1", "2", "3", "4"],
                "position": ["FWD", "FWD", "MID", "FWD"],
                "rG": [0.20, 0.18, 0.15, 0.03],
                "drG": [0.05, 0.05, 0.05, 0.05],
                "rA": [0.10, 0.10, 0.12, 0.05],
                "drA": [0.04, 0.04, 0.04, 0.04],
                "m": [80.0, 80.0, 80.0, 70.0],
                "sigma_m": [8.0, 8.0, 8.0, 8.0],
            }
        )
        xg = 1.5
        alpha_g = 0.96
        base = numagent_xp.score_team_side(
            players, xg=xg, xcs=0.25, xgc=0.4, alpha_g=alpha_g, alpha_a=0.90
        )
        lifted = numagent_xp.score_team_side(
            players,
            xg=xg,
            xcs=0.25,
            xgc=0.4,
            alpha_g=alpha_g,
            alpha_a=0.90,
            prop_p_by_player={"4": 0.40, "1": 0.45},
        )
        self.assertGreater(float(lifted.iloc[3]["E_G"]), float(base.iloc[3]["E_G"]))
        g_team = alpha_g * xg
        # Lift-only: returner moves toward λ(p); may sit below λ if team pot scaled.
        self.assertGreaterEqual(
            float(lifted.iloc[3]["E_G"]),
            min(float(base.iloc[3]["E_G"]) * 1.5, anytime_p_to_lambda(0.40) * 0.5),
        )
        self.assertAlmostEqual(float(lifted["E_G"].sum()), g_team, places=4)
        self.assertIn("prop_covered", lifted.columns)

    @unittest.skipUnless(PROBE.is_file(), "probe snapshot missing")
    def test_probe_snapshot_match_rate(self) -> None:
        doc = json.loads(PROBE.read_text(encoding="utf-8"))
        block = (doc.get("props") or {}).get("Brentford v Liverpool") or {}
        props = block.get("all") or block.get("top") or []
        self.assertGreater(len(props), 0)
        m = match_anytime_props(
            props, home="Brentford", away="Liverpool", bootstrap=self.bootstrap
        )
        self.assertTrue(m["ok"])
        self.assertGreaterEqual(m["n_matched"], 6)


if __name__ == "__main__":
    unittest.main()
