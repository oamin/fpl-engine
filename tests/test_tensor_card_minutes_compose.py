"""Composition vs plug-in helpers."""

from __future__ import annotations

import numpy as np

from src.models.tensor_card_minutes_compose import _arm_from_probs, _predict_ungated


class _Fake:
    def predict_proba_pos(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), 0.2, dtype=np.float64)


def test_compose_scales_conditional_by_p_play():
    rows = [
        {"y_goal": 0, "y_assist": 0},
        {"y_goal": 1, "y_assist": 0},
    ]
    p_play = np.array([0.5, 0.25], dtype=np.float64)
    cond = np.array([0.4, 0.8], dtype=np.float64)
    out = _arm_from_probs(rows, p_play * cond, p_play * cond)
    assert abs(out["p_goal"][0] - 0.2) < 1e-12
    assert abs(out["p_goal"][1] - 0.2) < 1e-12


def test_ungated_predict_never_floors_on_zero_minutes_proxy():
    model = _Fake()
    x = np.zeros((3, 2), dtype=np.float64)
    p = _predict_ungated(model, x)
    assert np.all(p == 0.2)
