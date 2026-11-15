"""Sliced baselines on the typed tensor card.

Train on closed seasons 2022-23…2024-25; score 2025-26.
Targets are binary event indicators (count ≥ 1) for goals and assists.
Features are as-of history plus the scored week's minutes/odds/home/role
(conditional-on-minutes spine). Contemporaneous teammate minutes are included
as a shared-minutes context signal.

Metrics: log loss, Brier, calibration slope/intercept (logit regression).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.tensor_card import (
    ART,
    FEATURE_NAMES,
    REPORTS,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_tensor_cards,
    iter_slices,
)

ROOT = Path(__file__).resolve().parents[2]


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([[r["features"][name] for name in FEATURE_NAMES] for r in rows], dtype=np.float64)
    y_g = np.array([r["y_goal"] for r in rows], dtype=np.int64)
    y_a = np.array([r["y_assist"] for r in rows], dtype=np.int64)
    return x, y_g, y_a


def _cal_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if y.min() == y.max() or float(np.std(p)) < 1e-12:
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    if y_tr.min() == y_tr.max():
        return np.full(len(x_te), float(y_tr.mean()), dtype=np.float64)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", class_weight=None),
    )
    pipe.fit(x_tr, y_tr)
    return pipe.predict_proba(x_te)[:, 1]


def _score(y: np.ndarray, p: np.ndarray, name: str) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    slope, intercept = _cal_slope_intercept(y, p)
    return {
        "name": name,
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }


def run_baselines(seed: int = 0) -> dict:
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    cards = build_tensor_cards(seasons)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for card in cards:
        for row in iter_slices(card, min_history=1):
            if card.season in TRAIN_SEASONS:
                train_rows.append(row)
            elif card.season == TEST_SEASON:
                test_rows.append(row)

    x_tr, yg_tr, ya_tr = _matrix(train_rows)
    x_te, yg_te, ya_te = _matrix(test_rows)

    # Constant prevalence baseline
    p_g_const = np.full(len(yg_te), float(yg_tr.mean()))
    p_a_const = np.full(len(ya_te), float(ya_tr.mean()))

    # Logistic on as-of slice features
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_tr))
    x_tr, yg_tr, ya_tr = x_tr[order], yg_tr[order], ya_tr[order]
    p_g_log = _fit_predict(x_tr, yg_tr, x_te)
    p_a_log = _fit_predict(x_tr, ya_tr, x_te)

    results = {
        "train_seasons": list(TRAIN_SEASONS),
        "test_season": TEST_SEASON,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "n_cards": len(cards),
        "features": list(FEATURE_NAMES),
        "goal": {
            "constant": _score(yg_te, p_g_const, "constant"),
            "logistic": _score(yg_te, p_g_log, "logistic"),
        },
        "assist": {
            "constant": _score(ya_te, p_a_const, "constant"),
            "logistic": _score(ya_te, p_a_log, "logistic"),
        },
    }
    return results


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_card_baselines.md"
    g, a = results["goal"], results["assist"]

    def _fmt(v: float, nd: int = 3) -> str:
        return "—" if v != v else f"{v:.{nd}f}"

    def row(label: str, m: dict) -> str:
        return (
            f"| {label} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
            f"{_fmt(m['cal_slope'])} | {_fmt(m['cal_intercept'])} | {m['base_rate']:.4f} |"
        )

    lines = [
        "# Typed tensor card — sliced baselines",
        "",
        "Pre-registered spine check on the cemented typed card (`src/models/tensor_card.py`).",
        f"Train: {', '.join(results['train_seasons'])}. Test: {results['test_season']}.",
        f"n_train={results['n_train']}, n_test={results['n_test']}, cards={results['n_cards']}.",
        "",
        "Targets: `y = 1[count ≥ 1]` for goals and assists. Features: as-of history",
        "plus scored-week minutes / home / role / assessed 1X2 / teammate shared minutes.",
        "Baudot binary planes are not used.",
        "",
        "## Goal event",
        "",
        "| Model | log loss | Brier | cal slope | cal intercept | base rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        row("constant", g["constant"]),
        row("logistic", g["logistic"]),
        "",
        "## Assist event",
        "",
        "| Model | log loss | Brier | cal slope | cal intercept | base rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        row("constant", a["constant"]),
        row("logistic", a["logistic"]),
        "",
        "## Read",
        "",
        "Logistic beats constant prevalence on both events when Δlog_loss < 0.",
        "This is a spine sanity check, not a full-card stepped/horizon comparison.",
        "Next: full-card FOV painter on the same tensor (opacity = P(event)).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    def _clean(obj):
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    payload = _clean(results)
    json_path = REPORTS / "typed_tensor_card_baselines.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_card_baselines.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    results = run_baselines()
    path = write_report(results)
    print(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
