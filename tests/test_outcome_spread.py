"""As-of player outcome spread (σ_out) — not xp_se."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.markets.correlated_draws import _toy_side, draw_fixture
from src.markets.outcome_spread import BLANK_THRESHOLD, HAUL_THRESHOLD, expand_outcome_spread


def _fake_v2() -> pd.DataFrame:
    """Two MIDs: stable vs volatile; known as-of path."""
    rows = []
    # Player 1: flat residuals ~0
    for gw, pts, mu in [
        (1, 4, 4.0),
        (2, 5, 4.5),
        (3, 4, 4.2),
        (4, 6, 5.0),
        (5, 5, 4.8),
        (6, 4, 4.5),
        (7, 5, 4.6),
    ]:
        rows.append(
            {
                "season": "2025-26",
                "gw": gw,
                "element": 1,
                "name": "Stable",
                "team": "A",
                "position": "MID",
                "fixture_id": 100 + gw,
                "minutes": 90,
                "total_points": pts,
                "score_xp_v2": mu,
                "xp_se": 0.2,
            }
        )
    # Player 2: big swings
    for gw, pts, mu in [
        (1, 2, 4.0),
        (2, 14, 4.5),
        (3, 1, 4.2),
        (4, 12, 5.0),
        (5, 0, 4.8),
        (6, 15, 4.5),
        (7, 2, 4.6),
    ]:
        rows.append(
            {
                "season": "2025-26",
                "gw": gw,
                "element": 2,
                "name": "Volatile",
                "team": "A",
                "position": "MID",
                "fixture_id": 200 + gw,
                "minutes": 90,
                "total_points": pts,
                "score_xp_v2": mu,
                "xp_se": 0.2,
            }
        )
    return pd.DataFrame(rows)


class OutcomeSpread(unittest.TestCase):
    def test_asof_no_same_gw_leak(self) -> None:
        base = expand_outcome_spread(_fake_v2())
        s7_clean = float(
            base.loc[(base["element"] == 1) & (base["gw"] == 7), "sigma_out"].iloc[0]
        )
        df = _fake_v2()
        df.loc[(df["element"] == 1) & (df["gw"] == 7), "total_points"] = 99
        poisoned = expand_outcome_spread(df)
        s7_poison = float(
            poisoned.loc[
                (poisoned["element"] == 1) & (poisoned["gw"] == 7), "sigma_out"
            ].iloc[0]
        )
        self.assertAlmostEqual(s7_clean, s7_poison, places=6)
        n7 = int(
            poisoned.loc[(poisoned["element"] == 1) & (poisoned["gw"] == 7), "n_out"].iloc[0]
        )
        self.assertEqual(n7, 6)

    def test_volatile_has_larger_sigma_out(self) -> None:
        spread = expand_outcome_spread(_fake_v2())
        late = spread.loc[spread["gw"] == 7]
        s_stable = float(late.loc[late["element"] == 1, "sigma_out"].iloc[0])
        s_vol = float(late.loc[late["element"] == 2, "sigma_out"].iloc[0])
        self.assertGreater(s_vol, s_stable + 1.0)

    def test_haul_rate_tracks_hauls(self) -> None:
        spread = expand_outcome_spread(_fake_v2())
        late = spread.loc[spread["gw"] == 7]
        h_stable = float(late.loc[late["element"] == 1, "haul_rate"].iloc[0])
        h_vol = float(late.loc[late["element"] == 2, "haul_rate"].iloc[0])
        self.assertLess(h_stable, 0.2)
        self.assertGreater(h_vol, 0.3)
        self.assertEqual(HAUL_THRESHOLD, 10.0)

    def test_blank_rate_tracks_blanks(self) -> None:
        spread = expand_outcome_spread(_fake_v2())
        late = spread.loc[spread["gw"] == 7]
        b_stable = float(late.loc[late["element"] == 1, "blank_rate"].iloc[0])
        b_vol = float(late.loc[late["element"] == 2, "blank_rate"].iloc[0])
        self.assertLess(b_stable, 0.2)
        self.assertGreater(b_vol, 0.3)
        self.assertEqual(BLANK_THRESHOLD, 2.0)

    def test_mu_is_non_shrunk_score_xp_v2(self) -> None:
        spread = expand_outcome_spread(_fake_v2())
        np.testing.assert_allclose(
            spread["mu"].to_numpy(float),
            spread["score_xp_v2"].to_numpy(float),
        )

    def test_hetero_sigma_out_widens_volatile_draws(self) -> None:
        home = _toy_side("H", "H")
        away = _toy_side("A", "A")
        # H_M1 wide, H_M2 narrow
        sigma = np.array([2.5, 3.0, 5.0, 1.0, 2.5, 3.0, 3.0, 3.0])
        d = draw_fixture(
            1.7,
            1.1,
            -0.05,
            home,
            away,
            n_scenarios=800,
            seed=0,
            sigma_log_lam=0.08,
            sigma_out=sigma,
            dirichlet_shares_flag=False,
        )
        ids = list(d.player_ids)
        i1, i2 = ids.index("H_M1"), ids.index("H_M2")
        self.assertGreater(float(d.xp[:, i1].std(ddof=1)), float(d.xp[:, i2].std(ddof=1)))


if __name__ == "__main__":
    unittest.main()
