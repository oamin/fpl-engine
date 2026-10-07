"""Bench Boost weight inside the transfer value. The default path adds none."""

from __future__ import annotations

import unittest

import pandas as pd

from src.models.season_climb_ft import (
    GAMMA,
    HOLD_EPS,
    SWITCH_PENALTY,
    SquadState,
    choose_transfers,
    transfer_value,
)


def _player(pid: str, position: str, score: float, eligible: bool) -> dict:
    return {
        "player_id": pid,
        "position": position,
        "team": pid,
        "team_norm": pid,
        "value": 40,
        "score_xp": score,
        "eligible": eligible,
    }


def _rows(bench: float) -> list[dict]:
    """Eleven clear starters and four bench players at ``bench``."""
    specs = [
        ("g1", "GKP", 30.0, False),
        ("g2", "GKP", bench, False),
        ("d1", "DEF", 29.0, False),
        ("d2", "DEF", 28.0, False),
        ("d3", "DEF", 27.0, False),
        ("d4", "DEF", 26.0, False),
        ("d5", "DEF", bench, False),
        ("m1", "MID", 25.0, False),
        ("m2", "MID", 24.0, False),
        ("m3", "MID", 23.0, False),
        ("m4", "MID", 22.0, False),
        ("m5", "MID", bench, False),
        ("f1", "FWD", 21.0, False),
        ("f2", "FWD", 10.0, False),
        ("f3", "FWD", bench, False),
    ]
    return [_player(*spec) for spec in specs]


def _value(frame: pd.DataFrame, gw: int, bench_gw: int | None) -> float:
    ids = set(frame["player_id"])
    scores = {row.player_id: float(row.score_xp) for row in frame.itertuples()}
    meta = {
        row.player_id: {"position": row.position, "team_norm": row.team_norm}
        for row in frame.itertuples()
    }
    roster = {g: set(ids) for g in (gw, gw + 1, gw + 2)}
    return transfer_value(
        ids,
        scores,
        meta,
        0,
        gw,
        [gw + 1, gw + 2],
        roster,
        "score_xp",
        bench_gw=bench_gw,
    )


class BenchWeightTests(unittest.TestCase):
    def test_the_default_ignores_the_bench(self) -> None:
        cheap = pd.DataFrame(_rows(1.0))
        dear = pd.DataFrame(_rows(3.0))
        self.assertAlmostEqual(_value(cheap, 10, None), _value(dear, 10, None))
        self.assertAlmostEqual(
            _value(cheap, 10, None),
            transfer_value(
                set(cheap["player_id"]),
                {row.player_id: float(row.score_xp) for row in cheap.itertuples()},
                {
                    row.player_id: {"position": row.position, "team_norm": row.team_norm}
                    for row in cheap.itertuples()
                },
                0,
                10,
                [11, 12],
                {g: set(cheap["player_id"]) for g in (10, 11, 12)},
                "score_xp",
            ),
        )

    def test_this_week_adds_the_bench_once(self) -> None:
        cheap = pd.DataFrame(_rows(1.0))
        dear = pd.DataFrame(_rows(3.0))
        gap = _value(dear, 10, 10) - _value(cheap, 10, 10)
        self.assertAlmostEqual(gap, 4.0 * (3.0 - 1.0))

    def test_a_later_horizon_week_is_discounted(self) -> None:
        cheap = pd.DataFrame(_rows(1.0))
        dear = pd.DataFrame(_rows(3.0))
        gap = _value(dear, 10, 12) - _value(cheap, 10, 12)
        self.assertAlmostEqual(gap, 8.0 * (GAMMA**2))

    def test_past_the_horizon_uses_the_decision_week_bench(self) -> None:
        cheap = pd.DataFrame(_rows(1.0))
        dear = pd.DataFrame(_rows(3.0))
        gap = _value(dear, 10, 15) - _value(cheap, 10, 15)
        self.assertAlmostEqual(gap, 8.0 * (GAMMA**5))

    def test_a_spent_week_and_a_hole_add_nothing(self) -> None:
        cheap = pd.DataFrame(_rows(1.0))
        dear = pd.DataFrame(_rows(3.0))
        self.assertAlmostEqual(_value(dear, 10, 9), _value(cheap, 10, 9))
        ids = set(cheap["player_id"])
        scores = {row.player_id: float(row.score_xp) for row in cheap.itertuples()}
        meta = {
            row.player_id: {"position": row.position, "team_norm": row.team_norm}
            for row in cheap.itertuples()
        }
        dear_scores = {row.player_id: float(row.score_xp) for row in dear.itertuples()}
        roster = {g: set(ids) for g in (10, 12, 13)}
        kwargs = dict(
            score_now=scores,
            meta=meta,
            hits=0,
            gw=10,
            future_gws=[12, 13],
            roster_by_gw=roster,
            score_col="score_xp",
            bench_gw=11,
        )
        base = transfer_value(ids, **kwargs)
        kwargs["score_now"] = dear_scores
        self.assertAlmostEqual(transfer_value(ids, **kwargs), base)


class ChoiceTests(unittest.TestCase):
    def test_the_bench_week_keeps_a_move_the_eleven_alone_would_refuse(self) -> None:
        rows = _rows(7.0)
        # 10.5 enters the XI and the 10 drops to the bench. Across three weeks
        # that half-point does not clear the hold. On the bench week the bench
        # also rises by 3, and that does.
        rows.append(_player("f4", "FWD", 10.5, True))
        pool = pd.DataFrame(rows)
        horizon = 1.0 + GAMMA + GAMMA**2
        xi_only = 0.5 * horizon - SWITCH_PENALTY
        with_bench = xi_only + (10.0 - 7.0)
        self.assertLess(xi_only, HOLD_EPS)
        self.assertGreaterEqual(with_bench, HOLD_EPS)
        purchase = {row["player_id"]: 40 for row in rows if row["player_id"] != "f4"}
        state = SquadState(purchase=purchase, bank=40, ft=1)
        held, n_hold, _ = choose_transfers(
            state,
            pool,
            "score_xp",
            10,
            [11, 12],
            {g: set(pool["player_id"]) for g in (10, 11, 12)},
        )
        moved, n_move, _ = choose_transfers(
            state,
            pool,
            "score_xp",
            10,
            [11, 12],
            {g: set(pool["player_id"]) for g in (10, 11, 12)},
            bench_gw=10,
        )
        self.assertEqual(n_hold, 0)
        self.assertEqual(held.ids(), state.ids())
        self.assertEqual(n_move, 1)
        self.assertIn("f4", moved.ids())
        self.assertNotIn("f3", moved.ids())


if __name__ == "__main__":
    unittest.main()
