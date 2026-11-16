"""Zero-gated SliceLogistic vs ungated SliceLogistic on the typed tensor spine.

Pre-registered before totals were read:
  reports/typed_tensor_zero_gate.md

Control: logistic fit on all present rows (mates off, actual minutes).
Candidate: fit logistic only on minutes > 0; minutes ≤ 0 → p = 1e-6.

Leave-one-season-out across the four closed seasons. Primary metric: paired
GW mean binary log-loss Δ = candidate − control via cluster_interval.
Not a Numagent / score_xp_v2 comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.gates import CLOSED_SEASONS, cluster_interval, run_asof_audit
from src.models.tensor_card import (
    ART,
    FEATURE_NAMES_LAG1,
    REPORTS,
    build_tensor_cards,
    iter_slices,
)

SEED = 0
N_BOOT = 1000
P_FLOOR = 1e-6
CONTRAST = "zero_gate_minus_ungated_logistic"


def _feat_matrix(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[r["features"][name] for name in FEATURE_NAMES_LAG1] for r in rows],
        dtype=np.float64,
    )


def _minutes(rows: list[dict]) -> np.ndarray:
    return np.array([r["features"]["minutes"] for r in rows], dtype=np.float64)


def _binary(rows: list[dict], channel: str) -> np.ndarray:
    key = "y_goal" if channel == "goal" else "y_assist"
    return np.array([r[key] for r in rows], dtype=np.int64)


def _clip_p(p: np.ndarray) -> np.ndarray:
    return np.clip(p, P_FLOOR, 1.0 - P_FLOOR)


def _row_nll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = _clip_p(p)
    y = y.astype(np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _fit_logistic(x_tr: np.ndarray, y_tr: np.ndarray) -> object:
    if len(y_tr) == 0 or y_tr.min() == y_tr.max():
        rate = float(y_tr.mean()) if len(y_tr) else P_FLOOR

        class _Const:
            def predict_proba_pos(self, x: np.ndarray) -> np.ndarray:
                return np.full(len(x), rate, dtype=np.float64)

        return _Const()
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    pipe.fit(x_tr, y_tr)

    class _Wrap:
        def __init__(self, model: object) -> None:
            self.model = model

        def predict_proba_pos(self, x: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(x)[:, 1]

    return _Wrap(pipe)


def _predict_ungated(model: object, x: np.ndarray) -> np.ndarray:
    return _clip_p(model.predict_proba_pos(x))


def _predict_gated(model: object, x: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    out = np.full(len(minutes), P_FLOOR, dtype=np.float64)
    active = minutes > 0
    if np.any(active):
        out[active] = _clip_p(model.predict_proba_pos(x[active]))
    return out


def _cal_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = _clip_p(p)
    if len(y) == 0 or y.min() == y.max() or float(np.std(p)) < 1e-12:
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _score_binary(y: np.ndarray, p: np.ndarray) -> dict:
    p = _clip_p(p)
    if len(y) == 0:
        return {
            "n": 0,
            "base_rate": float("nan"),
            "log_loss": float("nan"),
            "brier": float("nan"),
            "cal_slope": float("nan"),
            "cal_intercept": float("nan"),
        }
    slope, intercept = _cal_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }


def _collect_rows(cards: list) -> list[dict]:
    rows: list[dict] = []
    for card in cards:
        for row in iter_slices(
            card,
            min_history=1,
            minutes_mode="actual",
            include_mates=False,
        ):
            rows.append(row)
    return rows


def assert_mates_off(rows: list[dict]) -> None:
    for row in rows[:200]:
        assert float(row["features"]["mate_minutes"]) == 0.0
        assert float(row["features"]["mate_n"]) == 0.0


def _predict_fold(
    train_rows: list[dict],
    test_rows: list[dict],
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_rows))
    train_rows = [train_rows[i] for i in order]

    x_tr = _feat_matrix(train_rows)
    x_te = _feat_matrix(test_rows)
    minutes_tr = _minutes(train_rows)
    minutes_te = _minutes(test_rows)
    active_tr = minutes_tr > 0

    out: dict[str, np.ndarray] = {"minutes": minutes_te}
    for channel in ("goal", "assist"):
        y_tr = _binary(train_rows, channel)
        y_te = _binary(test_rows, channel).astype(np.float64)

        ungated = _fit_logistic(x_tr, y_tr)
        gated = _fit_logistic(x_tr[active_tr], y_tr[active_tr])

        out[f"p_ungated_{channel}"] = _predict_ungated(ungated, x_te)
        out[f"p_gated_{channel}"] = _predict_gated(gated, x_te, minutes_te)
        out[f"y_{channel}"] = y_te
    return out


def _gw_deltas(test_rows: list[dict], pred: dict[str, np.ndarray]) -> np.ndarray:
    gws = np.array([int(r["gw"]) for r in test_rows], dtype=np.int64)
    deltas = []
    for gw in sorted(set(gws.tolist())):
        m = gws == gw
        nll_gated = np.concatenate(
            [
                _row_nll(pred[f"y_{ch}"][m], pred[f"p_gated_{ch}"][m])
                for ch in ("goal", "assist")
            ]
        )
        nll_ungated = np.concatenate(
            [
                _row_nll(pred[f"y_{ch}"][m], pred[f"p_ungated_{ch}"][m])
                for ch in ("goal", "assist")
            ]
        )
        deltas.append(float(nll_gated.mean() - nll_ungated.mean()))
    return np.asarray(deltas, dtype=float)


def _subset_metrics(pred: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    metrics = {}
    for arm, prefix in (("ungated", "p_ungated"), ("gated", "p_gated")):
        channel_scores = {}
        ps, ys = [], []
        for channel in ("goal", "assist"):
            y = pred[f"y_{channel}"][mask]
            p = pred[f"{prefix}_{channel}"][mask]
            channel_scores[channel] = _score_binary(y, p)
            ps.append(p)
            ys.append(y)
        channel_scores["pooled"] = _score_binary(np.concatenate(ys), np.concatenate(ps))
        metrics[arm] = channel_scores
    return metrics


def run(seed: int = SEED) -> dict:
    cards = build_tensor_cards(CLOSED_SEASONS)
    by_season: dict[str, list] = {s: [] for s in CLOSED_SEASONS}
    for card in cards:
        by_season[card.season].append(card)

    deltas_by_season: dict[str, np.ndarray] = {}
    deltas_active_by_season: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    n_pred = 0

    for holdout in CLOSED_SEASONS:
        train_cards = [c for s, cs in by_season.items() if s != holdout for c in cs]
        test_cards = by_season[holdout]
        train_rows = _collect_rows(train_cards)
        test_rows = _collect_rows(test_cards)
        assert_mates_off(train_rows)
        assert_mates_off(test_rows)

        pred = _predict_fold(train_rows, test_rows, seed=seed)
        deltas_by_season[holdout] = _gw_deltas(test_rows, pred)

        minutes = pred["minutes"]
        active_idx = np.flatnonzero(minutes > 0)
        if active_idx.size:
            active_rows = [test_rows[i] for i in active_idx]
            active_pred = {
                "minutes": minutes[active_idx],
                **{
                    k: v[active_idx]
                    for k, v in pred.items()
                    if k != "minutes" and isinstance(v, np.ndarray)
                },
            }
            deltas_active_by_season[holdout] = _gw_deltas(active_rows, active_pred)
        else:
            deltas_active_by_season[holdout] = np.asarray([0.0], dtype=float)

        all_mask = np.ones(len(test_rows), dtype=bool)
        active_mask = minutes > 0
        fold_rows.append(
            {
                "holdout": holdout,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_active": int(active_mask.sum()),
                "n_zero": int((~active_mask).sum()),
                "all": _subset_metrics(pred, all_mask),
                "active": _subset_metrics(pred, active_mask),
                "mean_gw_delta": float(deltas_by_season[holdout].mean()),
                "mean_gw_delta_active": float(deltas_active_by_season[holdout].mean()),
            }
        )
        n_pred += len(test_rows)

    interval = cluster_interval(deltas_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed)
    interval_active = cluster_interval(
        deltas_active_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed
    )
    promote = float(interval["hi"]) < 0.0 and float(interval_active["hi"]) < 0.0
    if promote:
        decision = "promote_zero_gated_logistic"
    elif float(interval["lo"]) > 0.0:
        decision = "keep_ungated_logistic"
    else:
        decision = "inconclusive_keep_ungated_logistic"

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_zero_gate",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "features": list(FEATURE_NAMES_LAG1),
        "n_pred_rows": n_pred,
        "folds": fold_rows,
        "interval": interval,
        "interval_active_minutes": interval_active,
        "asof_audit_passed": bool(audit.get("passed")),
        "asof_audit_failures": list(audit.get("failures") or []),
        "mates_off": True,
        "seed": seed,
        "n_boot": N_BOOT,
    }


def _fmt(v: float, nd: int = 5) -> str:
    if v != v:
        return "—"
    return f"{v:+.{nd}f}" if nd >= 4 else f"{v:.{nd}f}"


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_zero_gate.md"
    iv = results["interval"]
    iva = results["interval_active_minutes"]
    decision = results["decision"]

    lines = [
        "# Typed tensor — zero-gated SliceLogistic",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "Follow-up to `typed_tensor_poisson_glm.md` (Co-PI diagnosis): gate structural",
        "`minutes == 0` zeros on the logistic spine.",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        (
            "| **Control** | Ungated SliceLogistic on all present rows; "
            "`FEATURE_NAMES_LAG1`; mates off. |"
        ),
        (
            "| **Candidate** | Zero-gated SliceLogistic: fit only on `minutes > 0`; "
            "`minutes ≤ 0 → p = 1e-6`. Same features. |"
        ),
        "",
        "## Protocol",
        "",
        "- Leave-one-season-out on 2022-23…2025-26. `2026-27` untouched.",
        f"- Primary: paired-by-GW mean binary log-loss, `Δ = {CONTRAST}` via "
        f"`cluster_interval` (n_boot={results['n_boot']}, seed={results['seed']}).",
        "- Promote only if 95% upper bound of Δ is strictly &lt; 0 **and** the same "
        "holds on the `minutes > 0` subset.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            f"Interval Δ log-loss (gated − ungated): mean {_fmt(iv['mean'])}, "
            f"95% [{_fmt(iv['lo'])}, {_fmt(iv['hi'])}]."
        ),
        (
            f"Active minutes subset: mean {_fmt(iva['mean'])}, "
            f"95% [{_fmt(iva['lo'])}, {_fmt(iva['hi'])}]."
        ),
        "",
        "Gameweeks per season: "
        + ", ".join(f"{s}={iv['n_gws'].get(s, 0)}" for s in CLOSED_SEASONS)
        + ".",
        "",
        f"Mates off asserted. Product as-of audit passed={results['asof_audit_passed']}"
        + (
            f" (failures: {results['asof_audit_failures']})"
            if results["asof_audit_failures"]
            else "."
        ),
        "",
        "## Fold holdouts (pooled log loss)",
        "",
        "| holdout | n_test | n_active | n_zero | ungated LL | gated LL | Δ mean GW | active Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in results["folds"]:
        u = fold["all"]["ungated"]["pooled"]["log_loss"]
        g = fold["all"]["gated"]["pooled"]["log_loss"]
        lines.append(
            f"| {fold['holdout']} | {fold['n_test']} | {fold['n_active']} | "
            f"{fold['n_zero']} | {u:.4f} | {g:.4f} | "
            f"{fold['mean_gw_delta']:+.5f} | {fold['mean_gw_delta_active']:+.5f} |"
        )

    lines += [
        "",
        "## Per-channel (n-weighted across LOSO folds)",
        "",
        "| arm | channel | scope | log loss | Brier | cal slope | base rate | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in ("all", "active"):
        for arm in ("ungated", "gated"):
            for channel in ("goal", "assist", "pooled"):
                ns, lls, brs, cals, bases = [], [], [], [], []
                for fold in results["folds"]:
                    m = fold[scope][arm][channel]
                    if m["n"] <= 0 or m["log_loss"] != m["log_loss"]:
                        continue
                    ns.append(m["n"])
                    lls.append(m["log_loss"])
                    brs.append(m["brier"])
                    cals.append(m["cal_slope"])
                    bases.append(m["base_rate"])
                if not ns:
                    continue
                w = np.asarray(ns, dtype=float)
                cal = (
                    float(np.average(cals, weights=w))
                    if np.isfinite(cals).all()
                    else float("nan")
                )
                lines.append(
                    f"| {arm} | {channel} | {scope} | "
                    f"{float(np.average(lls, weights=w)):.4f} | "
                    f"{float(np.average(brs, weights=w)):.4f} | "
                    f"{_fmt(cal, 3)} | "
                    f"{float(np.average(bases, weights=w)):.4f} | "
                    f"{int(w.sum())} |"
                )

    lines += ["", "## Call", ""]
    if decision == "promote_zero_gated_logistic":
        lines.append(
            "**promote_zero_gated_logistic.** Zero-gate wins on the GW-clustered "
            "interval including the active-minutes subset. Adopt gated SliceLogistic "
            "as the slice spine default (mates still off in this screen; mates on/off "
            "not re-opened here)."
        )
    elif decision == "keep_ungated_logistic":
        lines.append(
            "**keep_ungated_logistic.** Zero-gate loses or fails the active-subset "
            "rule. Keep ungated SliceLogistic."
        )
    else:
        lines.append(
            "**inconclusive_keep_ungated_logistic.** Interval covers zero or "
            "active-subset fails the strict promote rule. Keep ungated default; "
            "park zero-gate evidence."
        )
    lines += [
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_zero_gate`.",
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
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    payload = _clean(results)
    json_path = REPORTS / "typed_tensor_zero_gate.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_zero_gate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    results = run()
    path = write_report(results)
    print(
        json.dumps(
            {
                "decision": results["decision"],
                "interval": results["interval"],
                "interval_active_minutes": results["interval_active_minutes"],
                "promote": results["promote"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
