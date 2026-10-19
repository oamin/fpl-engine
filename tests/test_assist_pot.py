"""Assist pot: fixed ρ control vs team xA/xG construction."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.xp_engine import (
    ASSIST_RHO_FIXED,
    NEUTRAL_TEAM_XA,
    NEUTRAL_TEAM_XG,
    add_market_pots,
    add_player_priors,
    apply_lam_assist,
    assist_rho_team_xa_xg,
    compute_xp,
)


def _row(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "player_id": "p",
        "gw": 1,
        "date": "2024-08-17",
        "position": "MID",
        "team_norm": "arsenal",
        "fixture_id": "2024-08-17:arsenal:wolves",
        "minutes": 90.0,
        "total_points": 2.0,
        "xG": 0.2,
        "xA": 0.3,
        "defcon_hit": 0.0,
        "value": 55.0,
        "goals": 0.0,
        "p_over": 0.5,
        "p_under": 0.5,
        "attack_strength": 0.6,
        "defend_threat": 0.4,
        "p_not_lose": 0.6,
    }
    base.update(kwargs)
    return base


class AssistPotTests(unittest.TestCase):
    def test_neutral_ratio_matches_fixed(self) -> None:
        self.assertAlmostEqual(
            NEUTRAL_TEAM_XA / NEUTRAL_TEAM_XG, ASSIST_RHO_FIXED, places=9
        )

    def test_team_rho_uses_xa_over_xg(self) -> None:
        rho = assist_rho_team_xa_xg([1.0, 2.0, np.nan], [2.0, 2.0, 1.0])
        self.assertAlmostEqual(float(rho[0]), 0.5, places=6)
        self.assertAlmostEqual(float(rho[1]), 1.0, places=6)
        self.assertAlmostEqual(float(rho[2]), ASSIST_RHO_FIXED, places=6)

    def test_apply_fixed_matches_market_default(self) -> None:
        frame = add_market_pots(pd.DataFrame([_row()]))
        again = apply_lam_assist(frame, mode="fixed")
        self.assertTrue(np.allclose(frame["lam_assist"], again["lam_assist"]))
        self.assertTrue((again["assist_rho"] == ASSIST_RHO_FIXED).all())

    def test_team_mode_scales_lam_assist(self) -> None:
        history = []
        for gw in range(1, 5):
            history.append(
                _row(
                    gw=gw,
                    date=f"2024-08-{gw:02d}",
                    fixture_id=f"2024-08-{gw:02d}:arsenal:wolves",
                    xG=0.4,
                    xA=0.8,
                    minutes=90.0,
                )
            )
        history.append(
            _row(
                gw=5,
                date="2024-08-05",
                fixture_id="2024-08-05:arsenal:wolves",
                xG=0.0,
                xA=0.0,
                minutes=90.0,
            )
        )
        frame = add_player_priors(add_market_pots(pd.DataFrame(history)))
        team = apply_lam_assist(frame, mode="team_xa_xg")
        week5 = team.loc[team["gw"] == 5].iloc[0]
        # Prior team xA/xG ≈ 0.8/0.4 = 2.0, clipped to ASSIST_RHO_HI.
        self.assertGreater(float(week5["assist_rho"]), ASSIST_RHO_FIXED)
        self.assertAlmostEqual(
            float(week5["lam_assist"]),
            float(week5["lam_scored"]) * float(week5["assist_rho"]),
            places=6,
        )
        scored = compute_xp(team)
        later = scored.loc[scored["gw"] >= 2]
        self.assertTrue(np.isfinite(later["xp"]).all())
        self.assertTrue(np.isfinite(later["xp_assists"]).all())

    def test_scored_week_xa_does_not_move_rho(self) -> None:
        history = []
        for gw in range(1, 5):
            history.append(
                _row(
                    player_id="a",
                    gw=gw,
                    date=f"2024-08-{gw:02d}",
                    fixture_id=f"2024-08-{gw:02d}:arsenal:wolves",
                    xG=0.5,
                    xA=0.5,
                )
            )
        history.append(
            _row(
                player_id="a",
                gw=5,
                date="2024-08-05",
                fixture_id="2024-08-05:arsenal:wolves",
                xG=0.0,
                xA=0.0,
            )
        )
        base = add_player_priors(add_market_pots(pd.DataFrame(history)))
        rho0 = float(
            apply_lam_assist(base, mode="team_xa_xg")
            .loc[lambda d: d["gw"] == 5, "assist_rho"]
            .iloc[0]
        )
        leaked = [dict(r) for r in history]
        for row in leaked:
            if row["gw"] == 5:
                row["xA"] = 5.0
                row["xG"] = 0.1
        again = add_player_priors(add_market_pots(pd.DataFrame(leaked)))
        rho1 = float(
            apply_lam_assist(again, mode="team_xa_xg")
            .loc[lambda d: d["gw"] == 5, "assist_rho"]
            .iloc[0]
        )
        self.assertAlmostEqual(rho0, rho1, places=9)


if __name__ == "__main__":
    unittest.main()
