"""Horizon piece of transfer value with a bench weight. No network."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.season_climb import pick_xi
from src.models.season_climb_ft import transfer_value


def _squad() -> pd.DataFrame:
    """15 players. Scores make 4-4-2 the unique XI; the other four are the bench."""
    rows = [
        ("g1", "GKP", 10.0),
        ("g2", "GKP", 1.0),
        ("d1", "DEF", 9.0),
        ("d2", "DEF", 8.0),
        ("d3", "DEF", 7.0),
        ("d4", "DEF", 2.0),
        ("d5", "DEF", 1.0),
        ("m1", "MID", 9.0),
        ("m2", "MID", 8.0),
        ("m3", "MID", 7.0),
        ("m4", "MID", 6.0),
        ("m5", "MID", 1.0),
        ("f1", "FWD", 9.0),
        ("f2", "FWD", 8.0),
        ("f3", "FWD", 1.0),
    ]
    return pd.DataFrame(rows, columns=["player_id", "position", "mu"])


def _value(squad: pd.DataFrame, bench_weight: float | None) -> float:
    ids = set(squad["player_id"].astype(str))
    scores = {str(r.player_id): float(r.mu) for r in squad.itertuples()}
    meta = {
        str(r.player_id): {"position": str(r.position), "team_norm": "t"}
        for r in squad.itertuples()
    }
    return transfer_value(
        ids,
        scores,
        meta,
        0,
        1,
        [],
        {},
        "mu",
        horizon=1,
        bench_weight=bench_weight,
    )


class BenchWeightHorizonTest(unittest.TestCase):
    def test_none_matches_zero_and_quarter_adds_the_unselected_four(self) -> None:
        squad = _squad()
        xi, _form = pick_xi(squad, "mu")
        xi_ids = set(xi["player_id"].astype(str))
        bench = squad.loc[~squad["player_id"].astype(str).isin(xi_ids)]
        self.assertEqual(len(bench), 4)
        bench_sum = float(bench["mu"].sum())

        v_none = _value(squad, None)
        v_zero = _value(squad, 0)
        v_zero_float = _value(squad, 0.0)
        v_quarter = _value(squad, 0.25)

        self.assertEqual(v_none, v_zero)
        self.assertEqual(v_none, v_zero_float)
        self.assertEqual(v_none, float(xi["mu"].sum()))
        self.assertEqual(v_quarter, v_none + 0.25 * bench_sum)
