"""Unit tests for Opta-far decay target modes."""

from __future__ import annotations

import pandas as pd

from src.strategy.centreline_path_plan import apply_opta_far_decay


def _board(mu: list[float], pos: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [str(i) for i in range(len(mu))],
            "position": pos,
            "mu_post": mu,
            "score_xp_v2": mu,
            "xp": mu,
        }
    )


def test_decay_toward_zero():
    near = _board([4.0, 2.0], ["MID", "FWD"])
    far = _board([4.0, 2.0], ["MID", "FWD"])
    out = apply_opta_far_decay([near, near, far], near_depth=2, rho=0.5, target=0.0)
    assert len(out) == 3
    assert float(out[2]["mu_post"].iloc[0]) == 2.0  # 4 * 0.5^(2-2+1) wait t=2: 0.5^1=0.5 → 2.0
    assert float(out[2]["mu_post"].iloc[1]) == 1.0


def test_decay_pos_mean():
    near = _board([1.0], ["MID"])
    # two MIDs mean 3, one FWD 10
    far = _board([2.0, 4.0, 10.0], ["MID", "MID", "FWD"])
    out = apply_opta_far_decay(
        [near, near, far], near_depth=2, rho=0.0 + 0.5, target="pos_mean"
    )
    # factor=0.5; MID target=3 → 3+(2-3)*0.5=2.5, 3+(4-3)*0.5=3.5; FWD 10 stays 10
    vals = out[2]["mu_post"].to_numpy(float)
    assert abs(vals[0] - 2.5) < 1e-9
    assert abs(vals[1] - 3.5) < 1e-9
    assert abs(vals[2] - 10.0) < 1e-9


def test_near_boards_unchanged():
    b0 = _board([5.0], ["MID"])
    b1 = _board([6.0], ["MID"])
    b2 = _board([7.0], ["MID"])
    out = apply_opta_far_decay([b0, b1, b2], near_depth=2, rho=0.85, target=0.0)
    assert float(out[0]["mu_post"].iloc[0]) == 5.0
    assert float(out[1]["mu_post"].iloc[0]) == 6.0
    assert abs(float(out[2]["mu_post"].iloc[0]) - 7.0 * 0.85) < 1e-9
