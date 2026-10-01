"""Causal σ for u = μ / (σ + 1). Later gameweeks must not move earlier rows."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.sharpe_u import EPS, SIGMA_FILL, add_causal_sharpe_u


def _panel() -> pd.DataFrame:
    rows = []
    # Player A: residuals 1, 2, 3, then a later spike.
    for gw, mu, pts in (
        (1, 4.0, 5.0),
        (2, 4.0, 6.0),
        (3, 4.0, 7.0),
        (4, 4.0, 4.0),
        (5, 4.0, 4.0),
        (6, 4.0, 40.0),
    ):
        rows.append(
            {
                "player_id": "A",
                "gw": gw,
                "position": "MID",
                "score_xp": mu,
                "total_points": pts,
            }
        )
    # Player B has no residual history. At GW5 the only prior σ is A's GW4 value.
    rows.append(
        {
            "player_id": "B",
            "gw": 5,
            "position": "MID",
            "score_xp": 8.0,
            "total_points": 8.0,
        }
    )
    return pd.DataFrame(rows)


class SharpeUTest(unittest.TestCase):
    def test_sigma_uses_only_earlier_residuals(self) -> None:
        out = add_causal_sharpe_u(_panel())
        a4 = out.loc[(out["player_id"] == "A") & (out["gw"] == 4)].iloc[0]
        # Residuals at GW1–3 are 1, 2, 3. Sample std is 1.
        self.assertAlmostEqual(a4["sigma_xp"], 1.0)
        self.assertAlmostEqual(a4["score_u"], 4.0 / (1.0 + EPS))
        self.assertAlmostEqual(a4["score_risk"], 4.0 - 0.25 * 1.0)

    def test_future_points_do_not_move_earlier_u(self) -> None:
        base = add_causal_sharpe_u(_panel())
        shocked = _panel()
        shocked.loc[shocked["gw"] == 6, "total_points"] = 0.0
        shocked.loc[shocked["gw"] == 6, "score_xp"] = 99.0
        alt = add_causal_sharpe_u(shocked)
        for gw in (4, 5):
            b = base.loc[(base["player_id"] == "A") & (base["gw"] == gw), "score_u"].iloc[0]
            a = alt.loc[(alt["player_id"] == "A") & (alt["gw"] == gw), "score_u"].iloc[0]
            self.assertAlmostEqual(float(b), float(a))

    def test_position_median_is_strictly_earlier(self) -> None:
        out = add_causal_sharpe_u(_panel())
        a4 = float(out.loc[(out["player_id"] == "A") & (out["gw"] == 4), "sigma_xp"].iloc[0])
        b5 = out.loc[(out["player_id"] == "B") & (out["gw"] == 5)].iloc[0]
        self.assertAlmostEqual(float(b5["sigma_xp"]), a4)
        self.assertAlmostEqual(float(b5["score_u"]), 8.0 / (a4 + EPS))
        # GW4 has no earlier finite σ, so a thin history falls back to the constant.
        early = add_causal_sharpe_u(
            pd.DataFrame(
                [
                    {
                        "player_id": "C",
                        "gw": 1,
                        "position": "FWD",
                        "score_xp": 6.0,
                        "total_points": 6.0,
                    }
                ]
            )
        )
        self.assertAlmostEqual(float(early["sigma_xp"].iloc[0]), SIGMA_FILL)


if __name__ == "__main__":
    unittest.main()
