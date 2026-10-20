"""Multistage MC strand helpers (no full-season MILP in CI)."""

from __future__ import annotations

import random
import unittest

import numpy as np
import pandas as pd

from src.markets.match_outcome import match_outcome_probs_from_lambdas
from src.models.multistage_mc_milp import (
    ClubPath,
    StrandResult,
    aggregate_frequencies,
    sample_match_lams,
    sample_scoreline,
)
from src.models import cascade_long as cl


class CascadeLongIdentity(unittest.TestCase):
    def test_branch_uses_opta_match_source(self) -> None:
        self.assertEqual(cl.BRANCH, "cascade_long")
        self.assertEqual(cl.HORIZON, "t+2:t+n")
        self.assertEqual(cl.MATCH_SOURCE, "opta_rank_gf_ga")


class SampleScorelineTest(unittest.TestCase):
    def test_samples_near_centre(self) -> None:
        rng = random.Random(0)
        totals_h = totals_a = 0
        for _ in range(200):
            gh, ga, _, _ = sample_scoreline(
                1.5,
                1.0,
                lam_h_lo=1.0,
                lam_h_hi=2.0,
                lam_a_lo=0.6,
                lam_a_hi=1.4,
                rng=rng,
            )
            totals_h += gh
            totals_a += ga
            self.assertGreaterEqual(gh, 0)
            self.assertGreaterEqual(ga, 0)
        self.assertGreater(totals_h / 200, 0.8)
        self.assertLess(totals_h / 200, 2.5)

    def test_scoreline_from_m_not_independent_poisson(self) -> None:
        """Draws must be cells of M built from the sampled λ pair."""
        rng = random.Random(7)
        sh, sa = sample_match_lams(
            1.8,
            0.9,
            lam_h_lo=1.5,
            lam_h_hi=2.1,
            lam_a_lo=0.7,
            lam_a_hi=1.1,
            rng=rng,
            centre_blend=1.0,
        )
        probs = match_outcome_probs_from_lambdas(sh, sa, 0.0)
        # With centre_blend=1, λ are fixed — every roll is under that M.
        rng2 = random.Random(11)
        for _ in range(50):
            gh, ga, out_h, out_a = sample_scoreline(
                1.8,
                0.9,
                lam_h_lo=1.5,
                lam_h_hi=2.1,
                lam_a_lo=0.7,
                lam_a_hi=1.1,
                rng=rng2,
                centre_blend=1.0,
            )
            self.assertAlmostEqual(out_h, sh, places=9)
            self.assertAlmostEqual(out_a, sa, places=9)
            self.assertGreater(float(probs.matrix[gh, ga]), 0.0)
            self.assertLessEqual(gh, probs.matrix.shape[0] - 1)
            self.assertLessEqual(ga, probs.matrix.shape[1] - 1)

    def test_mean_mass_tracks_lambda(self) -> None:
        rng = random.Random(3)
        n = 800
        hs = []
        for _ in range(n):
            gh, ga, sh, sa = sample_scoreline(
                2.0,
                0.8,
                lam_h_lo=2.0,
                lam_h_hi=2.0,
                lam_a_lo=0.8,
                lam_a_hi=0.8,
                rng=rng,
                centre_blend=1.0,
            )
            hs.append(gh)
            self.assertAlmostEqual(sh, 2.0)
            self.assertAlmostEqual(sa, 0.8)
        self.assertGreater(float(np.mean(hs)), 1.5)
        self.assertLess(float(np.mean(hs)), 2.5)


class ClubPathTest(unittest.TestCase):
    def test_observe_updates_expanding_mean(self) -> None:
        path = ClubPath(att=1.5, dfn=1.0, n=0.0)
        path.observe(3, 0)
        self.assertEqual(path.n, 1.0)
        self.assertAlmostEqual(path.att, 3.0)
        self.assertAlmostEqual(path.dfn, 0.0)
        path.observe(1, 2)
        self.assertAlmostEqual(path.att, 2.0)
        self.assertAlmostEqual(path.dfn, 1.0)


def _toy_pool() -> pd.DataFrame:
    rows = []
    pid = 1
    for pos, n, base in (
        ("GKP", 2, 40),
        ("DEF", 5, 45),
        ("MID", 5, 60),
        ("FWD", 3, 70),
    ):
        for i in range(n):
            rows.append(
                {
                    "player_id": str(pid),
                    "name": f"{pos}{i}",
                    "team_norm": f"club{pid}",
                    "position": pos,
                    "value": base + i,
                    "mc_score": 2.0 + 0.1 * i,
                }
            )
            pid += 1
    rows[-1]["mc_score"] = 12.0
    rows[-1]["name"] = "Star"
    rows[-1]["value"] = 140
    # Extra midfielder to buy into.
    rows.append(
        {
            "player_id": "99",
            "name": "Upgrade",
            "team_norm": "club99",
            "position": "MID",
            "value": 70,
            "mc_score": 9.0,
        }
    )
    return pd.DataFrame(rows)


class XiCaptainMilpTest(unittest.TestCase):
    def test_prefers_premium_as_captain(self) -> None:
        from src.models.multistage_mc_milp import _pick_squad_xi_captain

        df = _toy_pool()
        picked = _pick_squad_xi_captain(df, "mc_score", time_limit=20.0)
        star = df.loc[df["name"] == "Star", "player_id"].iloc[0]
        self.assertEqual(picked["captain_id"], star)
        self.assertIn(picked["captain_id"], picked["xi_ids"])
        self.assertEqual(len(picked["squad"]), 15)
        self.assertEqual(len(picked["xi_ids"]), 11)

    def test_ft_hold_when_bank_tight(self) -> None:
        from src.models.multistage_mc_milp import _pick_squad_xi_captain_ft

        df = _toy_pool()
        from src.models.multistage_mc_milp import _pick_squad_xi_captain

        base = _pick_squad_xi_captain(df, "mc_score", time_limit=20.0)
        owned = {str(x) for x in base["squad"]["player_id"].tolist()}
        # With 0 FT and max_hits=0, must hold.
        held = _pick_squad_xi_captain_ft(
            df, "mc_score", owned_ids=owned, ft=0, max_hits=0, time_limit=20.0
        )
        self.assertEqual(held["n_transfers"], 0)
        self.assertEqual(held["hits"], 0)
        self.assertEqual(set(held["squad"]["player_id"].astype(str)), owned)


class AggregateTest(unittest.TestCase):
    def test_frequency(self) -> None:
        pool = pd.DataFrame(
            {
                "player_id": ["1", "2", "3"],
                "name": ["A", "B", "C"],
                "team_norm": ["x", "y", "z"],
                "position": ["MID", "MID", "FWD"],
                "value": [70, 60, 80],
            }
        )
        strands = [
            StrandResult(
                0,
                squads=[
                    {"gw": 8, "player_ids": ["1", "2"], "names": ["A", "B"], "score_sum": 1.0},
                    {"gw": 10, "player_ids": ["1", "3"], "names": ["A", "C"], "score_sum": 1.0},
                ],
            ),
            StrandResult(
                1,
                squads=[
                    {"gw": 8, "player_ids": ["1", "2"], "names": ["A", "B"], "score_sum": 1.0},
                ],
            ),
        ]
        freq = aggregate_frequencies(strands, pool)
        by_id = {r["player_id"]: r["freq"] for r in freq.to_dict("records")}
        self.assertAlmostEqual(by_id["1"], 1.0)
        self.assertAlmostEqual(by_id["2"], 2.0 / 3.0)
        self.assertAlmostEqual(by_id["3"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
