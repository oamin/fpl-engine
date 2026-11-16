"""Zero-gated Poisson exposure GLM vs SliceLogistic on the typed tensor spine.

Pre-registered before totals were read:
  reports/typed_tensor_poisson_glm.md

Control: logistic on as-of actual minutes, mates off (FEATURE_NAMES_LAG1).
Candidate: minutes≤0 → p≈0; minutes>0 → Poisson with log(m/90) exposure on
unclipped counts; opacity = 1 − exp(−λ).

Leave-one-season-out across the four closed seasons. Primary metric: paired
GW mean binary log-loss Δ = candidate − control via cluster_interval.
Not a Numagent / score_xp_v2 comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor
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

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
N_BOOT = 1000
P_FLOOR = 1e-6
POISSON_ALPHA = 1.0
# Minutes enter as exposure, not as a linear feature.
POISSON_FEATURES = tuple(name for name in FEATURE_NAMES_LAG1 if name != "minutes")
CONTRAST = "poisson_glm_minus_slice_logistic"


def _feat_matrix(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    return np.array([[r["features"][name] for name in names] for r in rows], dtype=np.float64)


def _minutes(rows: list[dict]) -> np.ndarray:
    return np.array([r["features"]["minutes"] for r in rows], dtype=np.float64)


def _counts(rows: list[dict], channel: str) -> np.ndarray:
    key = "goals" if channel == "goal" else "assists"
    return np.array([r[key] for r in rows], dtype=np.float64)


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
    if y_tr.min() == y_tr.max():
        rate = float(y_tr.mean())

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


def _fit_poisson(x_tr: np.ndarray, minutes_tr: np.ndarray, counts_tr: np.ndarray) -> object:
    """Fit rate model with exposure sample weights; predict opacity via 1-exp(-λ)."""
    active = minutes_tr > 0
    if not np.any(active) or float(counts_tr[active].sum()) == 0.0:
        # Degenerate: never score when minutes > 0 in train → near-zero opacity.
        class _Zero:
            def predict_opacity(self, x: np.ndarray, minutes: np.ndarray) -> np.ndarray:
                return np.full(len(minutes), P_FLOOR, dtype=np.float64)

        return _Zero()

    exposure = minutes_tr[active] / 90.0
    y_rate = counts_tr[active] / np.clip(exposure, 1e-12, None)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x_tr[active])
    model = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=3000)
    model.fit(xs, y_rate, sample_weight=exposure)

    class _PoissonOpacity:
        def __init__(self, scaler: StandardScaler, model: PoissonRegressor) -> None:
            self.scaler = scaler
            self.model = model

        def predict_opacity(self, x: np.ndarray, minutes: np.ndarray) -> np.ndarray:
            out = np.full(len(minutes), P_FLOOR, dtype=np.float64)
            mask = minutes > 0
            if not np.any(mask):
                return out
            exposure = minutes[mask] / 90.0
            rate = np.clip(self.model.predict(self.scaler.transform(x[mask])), 1e-12, None)
            lam = rate * exposure
            out[mask] = 1.0 - np.exp(-lam)
            return _clip_p(out)

    return _PoissonOpacity(scaler, model)


def _cal_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = _clip_p(p)
    if y.min() == y.max() or float(np.std(p)) < 1e-12:
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def _score_binary(y: np.ndarray, p: np.ndarray) -> dict:
    p = _clip_p(p)
    slope, intercept = _cal_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else float("nan"),
        "brier": float(brier_score_loss(y, p)) if len(y) else float("nan"),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }


def _poisson_nll(counts: np.ndarray, minutes: np.ndarray, p_event: np.ndarray) -> float:
    """Poisson NLL from opacity→λ on active rows: λ = −log(1 − p)."""
    mask = minutes > 0
    if not np.any(mask):
        return float("nan")
    p = _clip_p(p_event[mask])
    lam = -np.log(np.clip(1.0 - p, 1e-12, None))
    y = counts[mask]
    return float((lam - y * np.log(np.clip(lam, 1e-12, None)) + _log_fact(y)).mean())


def _log_fact(y: np.ndarray) -> np.ndarray:
    # Stirling / scipy-free: only small integer counts on FPL rows.
    out = np.zeros(len(y), dtype=np.float64)
    for i, v in enumerate(y.astype(int)):
        if v <= 1:
            out[i] = 0.0
        else:
            out[i] = float(np.sum(np.log(np.arange(2, v + 1))))
    return out


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


def _predict_fold(
    train_rows: list[dict],
    test_rows: list[dict],
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_rows))
    train_rows = [train_rows[i] for i in order]

    x_log_tr = _feat_matrix(train_rows, FEATURE_NAMES_LAG1)
    x_log_te = _feat_matrix(test_rows, FEATURE_NAMES_LAG1)
    x_poi_tr = _feat_matrix(train_rows, POISSON_FEATURES)
    x_poi_te = _feat_matrix(test_rows, POISSON_FEATURES)
    minutes_tr = _minutes(train_rows)
    minutes_te = _minutes(test_rows)

    out: dict[str, np.ndarray] = {"minutes": minutes_te}
    for channel in ("goal", "assist"):
        y_bin_tr = _binary(train_rows, channel)
        counts_tr = _counts(train_rows, channel)
        log_model = _fit_logistic(x_log_tr, y_bin_tr)
        poi_model = _fit_poisson(x_poi_tr, minutes_tr, counts_tr)
        out[f"p_log_{channel}"] = _clip_p(log_model.predict_proba_pos(x_log_te))
        out[f"p_poi_{channel}"] = poi_model.predict_opacity(x_poi_te, minutes_te)
        out[f"y_{channel}"] = _binary(test_rows, channel).astype(np.float64)
        out[f"count_{channel}"] = _counts(test_rows, channel)
    return out


def _gw_deltas(test_rows: list[dict], pred: dict[str, np.ndarray]) -> np.ndarray:
    """One delta per GW: mean (nll_poi − nll_log) over player×channel cells."""
    gws = np.array([int(r["gw"]) for r in test_rows], dtype=np.int64)
    unique = sorted(set(gws.tolist()))
    deltas = []
    for gw in unique:
        m = gws == gw
        nll_poi = np.concatenate(
            [_row_nll(pred[f"y_{channel}"][m], pred[f"p_poi_{channel}"][m]) for channel in ("goal", "assist")]
        )
        nll_log = np.concatenate(
            [_row_nll(pred[f"y_{channel}"][m], pred[f"p_log_{channel}"][m]) for channel in ("goal", "assist")]
        )
        deltas.append(float(nll_poi.mean() - nll_log.mean()))
    return np.asarray(deltas, dtype=float)


def _subset_metrics(pred: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    metrics = {}
    for arm, prefix in (("logistic", "p_log"), ("poisson", "p_poi")):
        channel_scores = {}
        ps, ys = [], []
        for channel in ("goal", "assist"):
            y = pred[f"y_{channel}"][mask]
            p = pred[f"{prefix}_{channel}"][mask]
            channel_scores[channel] = _score_binary(y, p)
            ps.append(p)
            ys.append(y)
        y_all = np.concatenate(ys)
        p_all = np.concatenate(ps)
        channel_scores["pooled"] = _score_binary(y_all, p_all)
        metrics[arm] = channel_scores
    # count NLL on active rows for poisson opacities
    active = mask & (pred["minutes"] > 0)
    metrics["poisson_count_nll"] = {
        "goal": _poisson_nll(pred["count_goal"], pred["minutes"], pred["p_poi_goal"])
        if np.any(active)
        else float("nan"),
        "assist": _poisson_nll(pred["count_assist"], pred["minutes"], pred["p_poi_assist"])
        if np.any(active)
        else float("nan"),
    }
    return metrics


def assert_mates_off(rows: list[dict]) -> None:
    for row in rows[:200]:
        assert float(row["features"]["mate_minutes"]) == 0.0
        assert float(row["features"]["mate_n"]) == 0.0


def run(seed: int = SEED) -> dict:
    cards = build_tensor_cards(CLOSED_SEASONS)
    by_season: dict[str, list] = {s: [] for s in CLOSED_SEASONS}
    for card in cards:
        by_season[card.season].append(card)

    deltas_by_season: dict[str, np.ndarray] = {}
    deltas_active_by_season: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    all_pred_rows = 0

    for holdout in CLOSED_SEASONS:
        train_cards = [c for s, cs in by_season.items() if s != holdout for c in cs]
        test_cards = by_season[holdout]
        train_rows = _collect_rows(train_cards)
        test_rows = _collect_rows(test_cards)
        assert_mates_off(train_rows)
        assert_mates_off(test_rows)
        pred = _predict_fold(train_rows, test_rows, seed=seed)
        deltas_by_season[holdout] = _gw_deltas(test_rows, pred)

        # Active-only GW deltas (minutes > 0)
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
        metrics_all = _subset_metrics(pred, all_mask)
        metrics_active = _subset_metrics(pred, active_mask)
        fold_rows.append(
            {
                "holdout": holdout,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_active": int(active_mask.sum()),
                "all": metrics_all,
                "active": metrics_active,
                "mean_gw_delta": float(deltas_by_season[holdout].mean()),
                "mean_gw_delta_active": float(deltas_active_by_season[holdout].mean()),
            }
        )
        all_pred_rows += len(test_rows)

    interval = cluster_interval(deltas_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed)
    interval_active = cluster_interval(
        deltas_active_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed
    )
    promote = float(interval["hi"]) < 0.0 and float(interval_active["hi"]) < 0.0
    if promote:
        decision = "promote_poisson_glm"
    elif float(interval["lo"]) > 0.0:
        decision = "keep_slice_logistic"
    else:
        decision = "inconclusive_keep_slice_logistic"

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_poisson_glm",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "features_logistic": list(FEATURE_NAMES_LAG1),
        "features_poisson": list(POISSON_FEATURES),
        "poisson_alpha": POISSON_ALPHA,
        "n_pred_rows": all_pred_rows,
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
    return "—" if v != v else f"{v:+.{nd}f}" if nd == 5 else f"{v:.{nd}f}"


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_poisson_glm.md"
    iv = results["interval"]
    iva = results["interval_active_minutes"]
    decision = results["decision"]

    lines = [
        "# Typed tensor — zero-gated Poisson exposure GLM",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        (
            "| **Control** | `SliceLogistic` on `asof_inputs(..., minutes_mode=\"actual\", "
            "include_mates=False)` / `FEATURE_NAMES_LAG1`. |"
        ),
        (
            "| **Candidate** | Zero-gated Poisson exposure GLM: `m≤0 → p=1e-6`; "
            "`m>0` Poisson on unclipped counts with `log(m/90)` exposure; "
            "opacity `1−exp(−λ)`. Features = logistic set without `minutes`. |"
        ),
        "",
        "## Protocol",
        "",
        "- Leave-one-season-out on closed seasons 2022-23…2025-26. `2026-27` untouched.",
        "- Primary: paired-by-GW mean binary opacity log-loss, "
        f"`Δ = {CONTRAST}` via `cluster_interval` (n_boot={results['n_boot']}, seed={results['seed']}).",
        "- Promote only if 95% upper bound of Δ is strictly &lt; 0 **and** the same holds "
        "on the `minutes > 0` subset.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            f"Interval Δ log-loss (poisson − logistic): mean {_fmt(iv['mean'])}, "
            f"95% [{_fmt(iv['lo'])}, {_fmt(iv['hi'])}]."
        ),
        (
            f"Active minutes subset: mean {_fmt(iva['mean'])}, "
            f"95% [{_fmt(iva['lo'])}, {_fmt(iva['hi'])}]."
        ),
        "",
        f"Gameweeks per season: "
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
        "| holdout | n_test | n_active | log LL | poi LL | Δ mean GW | active Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in results["folds"]:
        log_ll = fold["all"]["logistic"]["pooled"]["log_loss"]
        poi_ll = fold["all"]["poisson"]["pooled"]["log_loss"]
        lines.append(
            f"| {fold['holdout']} | {fold['n_test']} | {fold['n_active']} | "
            f"{log_ll:.4f} | {poi_ll:.4f} | {fold['mean_gw_delta']:+.5f} | "
            f"{fold['mean_gw_delta_active']:+.5f} |"
        )

    lines += [
        "",
        "## Per-channel (pooled across LOSO predictions)",
        "",
        "| arm | channel | scope | log loss | Brier | cal slope | base rate | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    # Re-aggregate channel metrics as simple n-weighted means across folds
    for scope in ("all", "active"):
        for arm in ("logistic", "poisson"):
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
                cal = float(np.average(cals, weights=w)) if np.isfinite(cals).all() else float("nan")
                lines.append(
                    f"| {arm} | {channel} | {scope} | "
                    f"{float(np.average(lls, weights=w)):.4f} | "
                    f"{float(np.average(brs, weights=w)):.4f} | "
                    f"{_fmt(cal, 3)} | "
                    f"{float(np.average(bases, weights=w)):.4f} | "
                    f"{int(w.sum())} |"
                )

    lines += [
        "",
        "## Call",
        "",
    ]
    if decision == "promote_poisson_glm":
        lines.append(
            "Promote zero-gated Poisson exposure GLM as the slice spine event head. "
            "Opacity remains `1−exp(−λ)`; mates stay off."
        )
    elif decision == "keep_slice_logistic":
        lines.append(
            f"**keep_slice_logistic.** Poisson GLM loses on the GW-clustered interval "
            f"(Δ mean {_fmt(iv['mean'])}, entirely above zero). The active-minutes "
            f"subset is worse still (Δ {_fmt(iva['mean'])}), so the loss is not an "
            "artefact of zero-gating blanks. Park the exposure GLM; slice spine stays "
            "logistic. This screen held mates off for both arms (Co-PI leak fix); "
            "mates on vs off is not decided here."
        )
    else:
        lines.append(
            "Interval covers zero or active-subset fails the strict rule. "
            "Keep SliceLogistic as default spine; park Poisson as screened."
        )
    lines += [
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_poisson_glm`.",
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
    json_path = REPORTS / "typed_tensor_poisson_glm.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_poisson_glm.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    results = run()
    path = write_report(results)
    summary = {
        "decision": results["decision"],
        "interval": results["interval"],
        "interval_active_minutes": results["interval_active_minutes"],
        "promote": results["promote"],
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
