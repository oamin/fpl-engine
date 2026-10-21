"""Historical player / positional rate baselines (Cash share collapse)."""

from __future__ import annotations

import unittest

import pandas as pd

from src.markets.numagent_xp_pilot import load_merged
from src.markets import numagent_xp
from src.markets.rate_baseline import (
    apply_historical_rate_baseline,
    norm_player_name,
    player_prior_rates,
    positional_prior_rates,
    prior_seasons_for,
    rates_for_players,
)
from src.models.short_horizon_cascade import build_frozen_side_table


class TestRateBaseline(unittest.TestCase):
    def test_prior_seasons_for_2025_26(self) -> None:
        self.assertEqual(
            prior_seasons_for("2025-26"),
            ("2022-23", "2023-24", "2024-25"),
        )

    def test_watkins_has_player_prior_not_zero(self) -> None:
        priors = player_prior_rates("2025-26")
        self.assertFalse(priors.empty)
        key = norm_player_name("Ollie Watkins")
        row = priors.loc[priors["name_key"] == key]
        self.assertFalse(row.empty, "Watkins missing from prior rates")
        self.assertGreater(float(row.iloc[0]["rG"]), 0.0)
        self.assertGreater(float(row.iloc[0]["rate_minutes"]), 1000.0)

    def test_positional_fwd_above_def(self) -> None:
        pos = positional_prior_rates("2025-26")
        self.assertIn("FWD", pos)
        self.assertIn("DEF", pos)
        self.assertGreater(float(pos["FWD"]["rG"]), float(pos["DEF"]["rG"]))

    def test_villa_share_not_cash_monopoly_after_baseline(self) -> None:
        season = "2025-26"
        merged = load_merged(season)
        hist = numagent_xp.asof_share_history(merged, before_gw=6, season=season)
        side, _, _ = build_frozen_side_table(
            hist,
            alpha_g=0.96,
            alpha_a=0.92,
            season=season,
            historical_rate_baseline=True,
        )
        villa = side.loc[side["team"].astype(str).str.contains("Villa", case=False)].copy()
        villa["rG"] = pd.to_numeric(villa["rG"], errors="coerce").fillna(0.0)
        total = float(villa["rG"].sum())
        self.assertGreater(total, 0.0)
        cash = villa.loc[villa["name"].astype(str).str.contains("Matty Cash", na=False)]
        self.assertFalse(cash.empty)
        cash_share = float(cash.iloc[0]["rG"]) / total
        # Must not absorb the whole club (was 1.0 under same-season-only).
        self.assertLess(cash_share, 0.5)
        watkins = villa.loc[villa["name"].astype(str).str.contains("Watkins", na=False)]
        self.assertFalse(watkins.empty)
        self.assertGreater(float(watkins.iloc[0]["rG"]), float(cash.iloc[0]["rG"]))

    def test_unknown_name_gets_positional(self) -> None:
        frame = pd.DataFrame(
            [{"name": "Completely Unknown Zzz", "position": "FWD"}]
        )
        out = rates_for_players(frame, season="2025-26")
        self.assertEqual(out.iloc[0]["rate_source"], "positional_prior")
        self.assertGreater(float(out.iloc[0]["rG"]), 0.0)


if __name__ == "__main__":
    unittest.main()
