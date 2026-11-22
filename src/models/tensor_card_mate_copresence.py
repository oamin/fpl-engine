"""Mate co-presence vs zero-gated SliceLogistic on the typed tensor spine.

Pre-registered before totals were read:
  reports/typed_tensor_mate_copresence.md

Control: zero-gated logistic on FEATURE_NAMES_LAG1 (no mates).
Candidate: same gate + who-shared-the-pitch features (role counts + as-of
mate history among those who played). Conditional-on-XI spine.

Not a Numagent / score_xp_v2 comparison. Not the old aggregate mate_minutes arm.
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
    COPRESENCE_FEATURE_NAMES,
    FEATURE_NAMES_LAG1,
    REPORTS,
    build_tensor_cards,
    iter_slices,
    mate_copresence_features,
)

SEED = 0
N_BOOT = 1000
P_FLOOR = 1e-6
CONTROL_FEATURES = FEATURE_NAMES_LAG1
CANDIDATE_FEATURES = FEATURE_NAMES_LAG1 + COPRESENCE_FEATURE_NAMES
CONTRAST = "mate_copresence_minus_zero_gated"


def _attach_copresence(card, rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        feats = dict(row["features"])
        # Recover seat index from player_id on this card.
        pids = card.player_id[: card.n_players]
        matches = np.flatnonzero(pids == int(row["player_id"]))
        if len(matches) != 1:
            # Fallback: skip co-presence (zeros) if seat ambiguous.
            for name in COPRESENCE_FEATURE_NAMES:
                feats[name] = 0.0
        else:
            feats.update(mate_copresence_features(card, int(matches[0]), int(row["t"])))
        out.append({**row, "features": feats})
    return out


def _collect(cards: list) -> list[dict]:
    rows: list[dict] = []
    for card in cards:
        base = list(
            iter_slices(
                card,
                min_history=1,
                minutes_mode="actual",
                include_mates=False,
            )
        )
        rows.extend(_attach_copresence(card, base))
    return rows


def _feat_matrix(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    return np.array([[r["features"][name] for name in names] for r in rows], dtype=np.float64)


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


def _predict_fold(train_rows: list[dict], test_rows: list[dict], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_rows))
    train_rows = [train_rows[i] for i in order]

    minutes_tr = _minutes(train_rows)
    minutes_te = _minutes(test_rows)
    active_tr = minutes_tr > 0

    x_ctrl_tr = _feat_matrix(train_rows, CONTROL_FEATURES)
    x_ctrl_te = _feat_matrix(test_rows, CONTROL_FEATURES)
    x_cand_tr = _feat_matrix(train_rows, CANDIDATE_FEATURES)
    x_cand_te = _feat_matrix(test_rows, CANDIDATE_FEATURES)

    out: dict[str, np.ndarray] = {"minutes": minutes_te}
    for channel in ("goal", "assist"):
        y_tr = _binary(train_rows, channel)
        y_te = _binary(test_rows, channel).astype(np.float64)
        ctrl = _fit_logistic(x_ctrl_tr[active_tr], y_tr[active_tr])
        cand = _fit_logistic(x_cand_tr[active_tr], y_tr[active_tr])
        out[f"p_ctrl_{channel}"] = _predict_gated(ctrl, x_ctrl_te, minutes_te)
        out[f"p_cand_{channel}"] = _predict_gated(cand, x_cand_te, minutes_te)
        out[f"y_{channel}"] = y_te
    return out


def _gw_deltas(test_rows: list[dict], pred: dict[str, np.ndarray]) -> np.ndarray:
    gws = np.array([int(r["gw"]) for r in test_rows], dtype=np.int64)
    deltas = []
    for gw in sorted(set(gws.tolist())):
        m = gws == gw
        nll_cand = np.concatenate(
            [_row_nll(pred[f"y_{ch}"][m], pred[f"p_cand_{ch}"][m]) for ch in ("goal", "assist")]
        )
        nll_ctrl = np.concatenate(
            [_row_nll(pred[f"y_{ch}"][m], pred[f"p_ctrl_{ch}"][m]) for ch in ("goal", "assist")]
        )
        deltas.append(float(nll_cand.mean() - nll_ctrl.mean()))
    return np.asarray(deltas, dtype=float)


def _subset_metrics(pred: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    metrics = {}
    for arm, prefix in (("control", "p_ctrl"), ("candidate", "p_cand")):
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


def _feature_diagnostics(rows: list[dict]) -> dict:
    """Show co-presence is not the old mate_minutes collinearity trap."""
    active = [r for r in rows if r["features"]["minutes"] > 0]
    if len(active) < 10:
        return {}
    minutes = np.array([r["features"]["minutes"] for r in active], dtype=float)
    # Rebuild aggregate mate_minutes for contrast only (not a model feature).
    # Approximate from co-presence n_on is wrong; use stored if present else skip.
    diag = {"n_active": len(active)}
    for name in COPRESENCE_FEATURE_NAMES:
        x = np.array([r["features"][name] for r in active], dtype=float)
        if float(np.std(x)) < 1e-12 or float(np.std(minutes)) < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(minutes, x)[0, 1])
        diag[f"corr_minutes_{name}"] = corr
        diag[f"mean_{name}"] = float(x.mean())
    return diag


def run(seed: int = SEED) -> dict:
    cards = build_tensor_cards(CLOSED_SEASONS)
    by_season: dict[str, list] = {s: [] for s in CLOSED_SEASONS}
    for card in cards:
        by_season[card.season].append(card)

    deltas_by_season: dict[str, np.ndarray] = {}
    deltas_active_by_season: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    n_pred = 0
    diag_holdout = None

    for holdout in CLOSED_SEASONS:
        train_cards = [c for s, cs in by_season.items() if s != holdout for c in cs]
        test_cards = by_season[holdout]
        train_rows = _collect(train_cards)
        test_rows = _collect(test_cards)
        if holdout == "2025-26":
            diag_holdout = _feature_diagnostics(test_rows)

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
        decision = "promote_mate_copresence"
    elif float(interval["lo"]) > 0.0:
        decision = "keep_zero_gated_no_mates"
    else:
        decision = "inconclusive_keep_zero_gated_no_mates"

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_mate_copresence",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "control_features": list(CONTROL_FEATURES),
        "candidate_features": list(CANDIDATE_FEATURES),
        "copresence_features": list(COPRESENCE_FEATURE_NAMES),
        "n_pred_rows": n_pred,
        "folds": fold_rows,
        "interval": interval,
        "interval_active_minutes": interval_active,
        "feature_diagnostics_2025_26_active": diag_holdout,
        "asof_audit_passed": bool(audit.get("passed")),
        "asof_audit_failures": list(audit.get("failures") or []),
        "seed": seed,
        "n_boot": N_BOOT,
    }


def _fmt(v: float, nd: int = 5) -> str:
    if v != v:
        return "—"
    return f"{v:+.{nd}f}" if nd >= 4 else f"{v:.{nd}f}"


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_mate_copresence.md"
    iv = results["interval"]
    iva = results["interval_active_minutes"]
    decision = results["decision"]
    diag = results.get("feature_diagnostics_2025_26_active") or {}

    lines = [
        "# Typed tensor — mate co-presence on zero-gated spine",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "PI question: does **who shares the pitch** move event rates? This is not the",
        "old aggregate `mate_minutes` arm (near-collinear with own minutes).",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        "| **Control** | Zero-gated SliceLogistic on `FEATURE_NAMES_LAG1` (no mates). |",
        (
            "| **Candidate** | Same gate + co-presence: DEF/MID/FWD pitch-mate counts, "
            "as-of sum/max of mate `mean_goals` / `mean_assists` among those who played. |"
        ),
        "",
        "Conditional-on-XI spine (same-week who-played); mate history weeks `< t` only.",
        "",
        "## Protocol",
        "",
        "- LOSO on 2022-23…2025-26. `2026-27` untouched.",
        f"- Primary: `Δ = {CONTRAST}` via `cluster_interval` "
        f"(n_boot={results['n_boot']}, seed={results['seed']}).",
        "- Promote if 95% upper bound of Δ &lt; 0 on all rows **and** on `minutes > 0`.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            f"Interval Δ log-loss (copresence − control): mean {_fmt(iv['mean'])}, "
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
        f"Product as-of audit passed={results['asof_audit_passed']}"
        + (
            f" (failures: {results['asof_audit_failures']})"
            if results["asof_audit_failures"]
            else "."
        ),
        "",
        "## Co-presence vs own minutes (2025-26 active rows)",
        "",
        "Correlation of each co-presence feature with own `minutes` "
        "(old `mate_minutes` was ≈ −0.97). Soft |r| here means a different signal.",
        "",
        "| feature | corr with own minutes | mean |",
        "| --- | ---: | ---: |",
    ]
    for name in COPRESENCE_FEATURE_NAMES:
        corr = diag.get(f"corr_minutes_{name}", float("nan"))
        mean = diag.get(f"mean_{name}", float("nan"))
        lines.append(f"| `{name}` | {_fmt(corr, 3)} | {_fmt(mean, 3)} |")

    lines += [
        "",
        "## Fold holdouts (pooled log loss)",
        "",
        "| holdout | n_test | n_active | control LL | copresence LL | Δ mean GW | active Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in results["folds"]:
        c = fold["all"]["control"]["pooled"]["log_loss"]
        k = fold["all"]["candidate"]["pooled"]["log_loss"]
        lines.append(
            f"| {fold['holdout']} | {fold['n_test']} | {fold['n_active']} | "
            f"{c:.4f} | {k:.4f} | {fold['mean_gw_delta']:+.5f} | "
            f"{fold['mean_gw_delta_active']:+.5f} |"
        )

    lines += [
        "",
        "## Per-channel (n-weighted across LOSO folds)",
        "",
        "| arm | channel | scope | log loss | Brier | cal slope | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scope in ("all", "active"):
        for arm in ("control", "candidate"):
            for channel in ("goal", "assist", "pooled"):
                ns, lls, brs, cals = [], [], [], []
                for fold in results["folds"]:
                    m = fold[scope][arm][channel]
                    if m["n"] <= 0 or m["log_loss"] != m["log_loss"]:
                        continue
                    ns.append(m["n"])
                    lls.append(m["log_loss"])
                    brs.append(m["brier"])
                    cals.append(m["cal_slope"])
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
                    f"{_fmt(cal, 3)} | {int(w.sum())} |"
                )

    lines += ["", "## Call", ""]
    if decision == "promote_mate_copresence":
        lines.append(
            "**promote_mate_copresence.** Who-shared-the-pitch features improve "
            "zero-gated opacity log-loss on all and active rows. Add co-presence "
            "to the slice spine (still conditional-on-XI / minutes)."
        )
    elif decision == "keep_zero_gated_no_mates":
        lines.append(
            "**keep_zero_gated_no_mates.** Co-presence loses on the GW-clustered "
            "interval. Pitch-mate composition does not improve the opacity spine "
            "under this feature set; keep zero-gated logistic without mates."
        )
    else:
        lines.append(
            "**inconclusive_keep_zero_gated_no_mates.** Interval covers zero or "
            "active-subset fails the strict rule. Keep zero-gated spine without "
            "co-presence; park evidence."
        )
    lines += [
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_mate_copresence`.",
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
    (REPORTS / "typed_tensor_mate_copresence.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_mate_copresence.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
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
                "feature_diagnostics_2025_26_active": results.get(
                    "feature_diagnostics_2025_26_active"
                ),
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
