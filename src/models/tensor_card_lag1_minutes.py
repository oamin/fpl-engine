"""Slice spine: actual minutes vs lag-1 predicted minutes.

Control: scored-week realised minutes + same-week mate minutes (current spine).
Candidate: ̂m_t = m_{t-1} (else last present, else train role prior); no mate leak.

Same logistic, same rows, train 2022-23…2024-25 → holdout 2025-26.
Horizon card untouched. Expected to lose; still a pre-registered honesty screen.
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
    FEATURE_NAMES_LAG1,
    REPORTS,
    TEST_SEASON,
    TRAIN_SEASONS,
    build_tensor_cards,
    iter_slices,
    lag1_minutes,
    role_minutes_prior,
)

SEED = 0


def _matrix(rows: list[dict], names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([[r["features"][name] for name in names] for r in rows], dtype=np.float64)
    y_g = np.array([r["y_goal"] for r in rows], dtype=np.int64)
    y_a = np.array([r["y_assist"] for r in rows], dtype=np.int64)
    return x, y_g, y_a


def _fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    if y_tr.min() == y_tr.max():
        return np.full(len(x_te), float(y_tr.mean()), dtype=np.float64)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipe.fit(x_tr, y_tr)
    return pipe.predict_proba(x_te)[:, 1]


def _score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _arm_metrics(train_rows, test_rows, names: tuple[str, ...], seed: int) -> dict:
    x_tr, yg_tr, ya_tr = _matrix(train_rows, names)
    x_te, yg_te, ya_te = _matrix(test_rows, names)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_tr))
    x_tr, yg_tr, ya_tr = x_tr[order], yg_tr[order], ya_tr[order]
    pg = _fit_predict(x_tr, yg_tr, x_te)
    pa = _fit_predict(x_tr, ya_tr, x_te)
    # pooled event channels for a single headline too
    p_all = np.concatenate([pg, pa])
    y_all = np.concatenate([yg_te.astype(np.float64), ya_te.astype(np.float64)])
    return {
        "goal": _score(yg_te, pg),
        "assist": _score(ya_te, pa),
        "pooled": _score(y_all, p_all),
        "features": list(names),
    }


def minutes_mae(cards, role_prior) -> dict:
    """How wrong lag-1 is at predicting realised minutes on holdout present rows."""
    err, n = 0.0, 0
    for card in cards:
        for p in range(card.n_players):
            if int(card.position[p]) < 0:
                continue
            for t in range(1, card.n_fixtures):
                if not card.present[p, t]:
                    continue
                pred = lag1_minutes(card, p, t, role_prior=role_prior)
                err += abs(pred - float(card.minutes[p, t]))
                n += 1
    return {"mae": float(err / max(n, 1)), "n": n}


def run(seed: int = SEED) -> dict:
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    cards = build_tensor_cards(seasons)
    train_cards = [c for c in cards if c.season in TRAIN_SEASONS]
    test_cards = [c for c in cards if c.season == TEST_SEASON]
    prior = role_minutes_prior(train_cards)

    train_actual, test_actual = [], []
    train_lag1, test_lag1 = [], []
    for card in cards:
        bucket_a = train_actual if card.season in TRAIN_SEASONS else test_actual
        bucket_l = train_lag1 if card.season in TRAIN_SEASONS else test_lag1
        if card.season not in TRAIN_SEASONS and card.season != TEST_SEASON:
            continue
        for row in iter_slices(card, min_history=1, minutes_mode="actual"):
            bucket_a.append(row)
        for row in iter_slices(card, min_history=1, minutes_mode="lag1", role_prior=prior):
            bucket_l.append(row)

    actual = _arm_metrics(train_actual, test_actual, FEATURE_NAMES, seed)
    lag1 = _arm_metrics(train_lag1, test_lag1, FEATURE_NAMES_LAG1, seed)
    mae = minutes_mae(test_cards, prior)

    delta = {
        "goal_log": lag1["goal"]["log_loss"] - actual["goal"]["log_loss"],
        "assist_log": lag1["assist"]["log_loss"] - actual["assist"]["log_loss"],
        "pooled_log": lag1["pooled"]["log_loss"] - actual["pooled"]["log_loss"],
        "goal_brier": lag1["goal"]["brier"] - actual["goal"]["brier"],
        "assist_brier": lag1["assist"]["brier"] - actual["assist"]["brier"],
        "pooled_brier": lag1["pooled"]["brier"] - actual["pooled"]["brier"],
    }
    return {
        "train_seasons": list(TRAIN_SEASONS),
        "test_season": TEST_SEASON,
        "n_train": len(train_actual),
        "n_test": len(test_actual),
        "role_prior": {str(k): v for k, v in prior.items()},
        "lag1_minutes_mae": mae,
        "actual": actual,
        "lag1": lag1,
        "delta_lag1_minus_actual": delta,
    }


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_lag1_minutes.md"
    a, b = results["actual"], results["lag1"]
    d = results["delta_lag1_minus_actual"]

    def row(label: str, arm: dict, event: str) -> str:
        m = arm[event]
        return f"| {label} | {event} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['base_rate']:.4f} |"

    lines = [
        "# Slice spine — actual minutes vs lag-1",
        "",
        "Pre-registered honesty screen on the typed-tensor **slice** arm.",
        "Horizon FOV unchanged (still shows true minutes on the card).",
        "",
        "- **actual:** scored-week realised minutes + same-week mate minutes.",
        "- **lag1:** `m_hat_t = m_(t-1)` (else last present before t, else train role prior); "
        "mate features dropped.",
        "",
        f"Train {', '.join(results['train_seasons'])} → test {results['test_season']}. "
        f"n_train={results['n_train']}, n_test={results['n_test']}.",
        f"Lag-1 minutes MAE on holdout present rows: **{results['lag1_minutes_mae']['mae']:.2f}** "
        f"(n={results['lag1_minutes_mae']['n']}).",
        "",
        "## Holdout",
        "",
        "| arm | event | log loss | Brier | base rate |",
        "| --- | --- | ---: | ---: | ---: |",
        row("actual", a, "goal"),
        row("lag1", b, "goal"),
        row("actual", a, "assist"),
        row("lag1", b, "assist"),
        row("actual", a, "pooled"),
        row("lag1", b, "pooled"),
        "",
        "## Delta (lag1 − actual)",
        "",
        f"| | Δ log | Δ Brier |",
        f"| --- | ---: | ---: |",
        f"| goal | {d['goal_log']:+.4f} | {d['goal_brier']:+.4f} |",
        f"| assist | {d['assist_log']:+.4f} | {d['assist_brier']:+.4f} |",
        f"| pooled | {d['pooled_log']:+.4f} | {d['pooled_brier']:+.4f} |",
        "",
        "## Call",
        "",
    ]
    if d["pooled_log"] > 0.01:
        lines.append(
            f"Lag-1 **loses** as expected (pooled Δlog **{d['pooled_log']:+.4f}**). "
            "Actual minutes stay the slice spine; lag-1 is the honest pre-match reference, not a promote."
        )
    elif d["pooled_log"] > 0:
        lines.append(
            f"Lag-1 slightly worse (pooled Δlog **{d['pooled_log']:+.4f}**). "
            "Keep actual minutes as the conditional spine; document lag-1 as the as-of variant."
        )
    else:
        lines.append(
            f"Lag-1 pooled Δlog **{d['pooled_log']:+.4f}** — unexpected non-loss; re-check before any promote."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "typed_tensor_lag1_minutes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_lag1_minutes.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


def main() -> None:
    results = run()
    path = write_report(results)
    print(json.dumps(results["delta_lag1_minus_actual"], indent=2))
    print(json.dumps({"actual_pooled": results["actual"]["pooled"], "lag1_pooled": results["lag1"]["pooled"], "mae": results["lag1_minutes_mae"]}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
