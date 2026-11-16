"""Composition ``p_play · f(μ)`` vs plug-in ``f(p·μ)`` on frozen minutes head.

Pre-registered before totals were read:
  reports/typed_tensor_minutes_compose.md

Control: promoted plug-in ``p = f_event(x(p_play · μ))``.
Candidate: law-of-total-probability compose ``p = p_play · f_event(x(μ))``.
Diagnostic: ungated lag-1 into ``f_event`` (no 1e-6 on lag1==0).

Same frozen play / minutes|play / rich event models. No Odds API. Not Numagent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.gates import CLOSED_SEASONS, cluster_interval, run_asof_audit
from src.models.tensor_card import ART, REPORTS, build_tensor_cards, role_minutes_prior
from src.models.tensor_card_minutes_head import (
    EVENT_FEATURES,
    MATERIALITY,
    SEED,
    _arm_event_preds,
    _attach_event_features,
    _clip_p,
    _collect_rows,
    _fit_logistic,
    _fit_minutes_head,
    _gw_deltas,
    _mat,
    _subset_metrics,
    load_over25_table,
)

N_BOOT = 1000
CONTRAST = "compose_minus_plugin"


def _predict_play_and_mu(play_model: object, mins_model: object, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    from src.models.tensor_card_minutes_head import MINUTES_FEATURES

    x = _mat(rows, MINUTES_FEATURES, key="minutes_features")
    p_play = _clip_p(play_model.predict_proba_pos(x))
    mu = np.clip(np.asarray(mins_model.predict(x), dtype=np.float64), 0.0, 120.0)
    return p_play, mu


def _predict_ungated(model: object, x: np.ndarray) -> np.ndarray:
    return _clip_p(model.predict_proba_pos(x))


def _arm_from_probs(
    rows: list[dict],
    p_goal: np.ndarray,
    p_assist: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "y_goal": np.array([r["y_goal"] for r in rows], dtype=np.float64),
        "y_assist": np.array([r["y_assist"] for r in rows], dtype=np.float64),
        "p_goal": _clip_p(p_goal),
        "p_assist": _clip_p(p_assist),
    }


def run(seed: int = SEED) -> dict:
    over_table = load_over25_table(CLOSED_SEASONS)
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
        role_prior = role_minutes_prior(train_cards)
        train_rows = _collect_rows(train_cards, over_table, role_prior)
        test_rows = _collect_rows(test_cards, over_table, role_prior)

        play_model, mins_model = _fit_minutes_head(train_rows)
        p_play_te, mu_te = _predict_play_and_mu(play_model, mins_model, test_rows)
        hat_te = np.clip(p_play_te * mu_te, 0.0, 120.0)

        # Event head: train on actual minutes (same as minutes_head).
        train_actual = _attach_event_features(
            train_rows,
            np.array([r["actual_minutes"] for r in train_rows], dtype=np.float64),
            role_prior,
        )
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(train_actual))
        train_actual = [train_actual[i] for i in order]
        x_tr = _mat(train_actual, EVENT_FEATURES)
        m_tr = np.array([r["used_minutes"] for r in train_actual], dtype=np.float64)
        active_tr = m_tr > 0
        event_models = {}
        for channel in ("goal", "assist"):
            y_tr = np.array([r[f"y_{channel}"] for r in train_actual], dtype=np.int64)
            event_models[channel] = _fit_logistic(x_tr[active_tr], y_tr[active_tr])

        # Control: plug-in f(x(p·μ)) with gated predict (hat never 0 in practice).
        test_plugin = _attach_event_features(test_rows, hat_te, role_prior)
        pred_plugin = _arm_event_preds(event_models, test_plugin)

        # Candidate: p_play · f(x(μ)); no hard zero-gate on p_play.
        test_mu = _attach_event_features(test_rows, mu_te, role_prior)
        x_mu = _mat(test_mu, EVENT_FEATURES)
        p_goal_cond = _predict_ungated(event_models["goal"], x_mu)
        p_assist_cond = _predict_ungated(event_models["assist"], x_mu)
        pred_compose = _arm_from_probs(
            test_rows,
            p_play_te * p_goal_cond,
            p_play_te * p_assist_cond,
        )

        # Diagnostic: ungated lag-1 into f_event.
        lag1_m = np.array([r["lag1_minutes"] for r in test_rows], dtype=np.float64)
        test_lag1 = _attach_event_features(test_rows, lag1_m, role_prior)
        x_lag = _mat(test_lag1, EVENT_FEATURES)
        pred_lag1_ungated = _arm_from_probs(
            test_rows,
            _predict_ungated(event_models["goal"], x_lag),
            _predict_ungated(event_models["assist"], x_lag),
        )
        # Gated lag-1 (confounded reference from prior screen).
        pred_lag1_gated = _arm_event_preds(event_models, test_lag1)

        deltas_by_season[holdout] = _gw_deltas(test_rows, pred_compose, pred_plugin)

        actual_m = np.array([r["actual_minutes"] for r in test_rows], dtype=np.float64)
        active_mask = actual_m > 0
        inactive_mask = ~active_mask
        active_idx = np.flatnonzero(active_mask)
        if active_idx.size:
            active_rows = [test_rows[i] for i in active_idx]
            active_compose = {k: v[active_idx] for k, v in pred_compose.items()}
            active_plugin = {k: v[active_idx] for k, v in pred_plugin.items()}
            deltas_active_by_season[holdout] = _gw_deltas(active_rows, active_compose, active_plugin)
        else:
            deltas_active_by_season[holdout] = np.asarray([0.0], dtype=float)

        all_mask = np.ones(len(test_rows), dtype=bool)
        fold_rows.append(
            {
                "holdout": holdout,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_active": int(active_mask.sum()),
                "n_inactive": int(inactive_mask.sum()),
                "plugin": {
                    "all": _subset_metrics(pred_plugin, all_mask),
                    "active": _subset_metrics(pred_plugin, active_mask),
                    "inactive": _subset_metrics(pred_plugin, inactive_mask),
                },
                "compose": {
                    "all": _subset_metrics(pred_compose, all_mask),
                    "active": _subset_metrics(pred_compose, active_mask),
                    "inactive": _subset_metrics(pred_compose, inactive_mask),
                },
                "lag1_ungated": {
                    "all": _subset_metrics(pred_lag1_ungated, all_mask),
                    "active": _subset_metrics(pred_lag1_ungated, active_mask),
                    "inactive": _subset_metrics(pred_lag1_ungated, inactive_mask),
                },
                "lag1_gated": {
                    "all": _subset_metrics(pred_lag1_gated, all_mask),
                },
                "mean_gw_delta": float(deltas_by_season[holdout].mean()),
                "mean_gw_delta_active": float(deltas_active_by_season[holdout].mean()),
                "mean_p_play": float(p_play_te.mean()),
                "mean_mu": float(mu_te.mean()),
                "mean_hat": float(hat_te.mean()),
                "frac_hat_zero": float((hat_te <= 0).mean()),
            }
        )
        n_pred += len(test_rows)

    interval = cluster_interval(deltas_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed)
    interval_active = cluster_interval(
        deltas_active_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed
    )
    clears = float(interval["hi"]) < 0.0 and float(interval_active["hi"]) < 0.0
    material = float(interval["mean"]) <= MATERIALITY and float(interval_active["mean"]) <= MATERIALITY
    folds_neg = all(f["mean_gw_delta"] < 0.0 for f in fold_rows)
    if clears and material:
        decision = "promote_compose"
        promote = True
    elif clears and (folds_neg or float(interval["mean"]) < 0.0):
        decision = "soft_promote_compose"
        promote = False
    elif float(interval["lo"]) > 0.0:
        decision = "keep_plugin"
        promote = False
    else:
        decision = "inconclusive_keep_plugin"
        promote = False

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_minutes_compose",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "materiality": MATERIALITY,
        "clears_zero": clears,
        "meets_materiality": material,
        "n_pred_rows": n_pred,
        "folds": fold_rows,
        "interval": interval,
        "interval_active_actual_minutes": interval_active,
        "asof_audit_passed": bool(audit.get("passed")),
        "asof_audit_failures": list(audit.get("failures") or []),
        "seed": seed,
        "n_boot": N_BOOT,
        "minutes_head_reframe": "provisional_plugin_reference",
    }


def _fmt(v: float, nd: int = 5) -> str:
    if v != v:
        return "—"
    return f"{v:+.{nd}f}" if nd >= 4 else f"{v:.{nd}f}"


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_minutes_compose.md"
    iv = results["interval"]
    iva = results["interval_active_actual_minutes"]
    decision = results["decision"]
    lines = [
        "# Typed tensor — minutes composition `p·f(μ)` vs plug-in `f(p·μ)`",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "Co-PI follow-up after `promote_minutes_head`: the −0.034 gap vs gated lag-1 was",
        "amplified by hard-zeroing non-structural lag-1 blanks. Next lever is the",
        "**probability composition**, not a harder gate.",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        "| **Control (plug-in)** | Locked models; `p = f_event(x(ĥm))` with `ĥm = p_play·μ₊`. |",
        "| **Candidate (compose)** | Same models; `p = p_play · f_event(x(μ₊))` (no hard gate on `p_play`). |",
        "| **Diagnostic** | Ungated lag-1 into `f_event` (no `1e-6` on lag1==0). |",
        "",
        "## Protocol",
        "",
        "- Same play / minutes|play / rich event models as `typed_tensor_minutes_head`.",
        "- LOSO on 2022-23…2025-26. `2026-27` untouched.",
        f"- Primary: `Δ = {CONTRAST}` via `cluster_interval` "
        f"(n_boot={results['n_boot']}, seed={results['seed']}).",
        f"- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ {results['materiality']}.",
        "- Soft-promote if hi &lt; 0 on both but mean above materiality.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            f"Interval Δ log-loss (compose − plugin): mean {_fmt(iv['mean'])}, "
            f"95% [{_fmt(iv['lo'])}, {_fmt(iv['hi'])}]."
        ),
        (
            f"Active (actual m&gt;0) subset: mean {_fmt(iva['mean'])}, "
            f"95% [{_fmt(iva['lo'])}, {_fmt(iva['hi'])}]."
        ),
        "",
        f"Clears zero={results['clears_zero']}. Meets materiality={results['meets_materiality']}.",
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
        "## Fold holdouts (pooled log loss)",
        "",
        "| holdout | n | plugin | compose | lag1 ungated | lag1 gated | inactive plugin | inactive compose | Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in results["folds"]:
        lines.append(
            f"| {fold['holdout']} | {fold['n_test']} | "
            f"{fold['plugin']['all']['pooled']['log_loss']:.4f} | "
            f"{fold['compose']['all']['pooled']['log_loss']:.4f} | "
            f"{fold['lag1_ungated']['all']['pooled']['log_loss']:.4f} | "
            f"{fold['lag1_gated']['all']['pooled']['log_loss']:.4f} | "
            f"{fold['plugin']['inactive']['pooled']['log_loss']:.4f} | "
            f"{fold['compose']['inactive']['pooled']['log_loss']:.4f} | "
            f"{fold['mean_gw_delta']:+.5f} |"
        )

    lines += [
        "",
        "## Per-channel (n-weighted, all rows)",
        "",
        "| arm | channel | log loss | Brier | n |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm in ("plugin", "compose", "lag1_ungated"):
        for channel in ("goal", "assist", "pooled"):
            ns, lls, brs = [], [], []
            for fold in results["folds"]:
                m = fold[arm]["all"][channel]
                if m["n"] <= 0 or m["log_loss"] != m["log_loss"]:
                    continue
                ns.append(m["n"])
                lls.append(m["log_loss"])
                brs.append(m["brier"])
            if not ns:
                continue
            w = np.asarray(ns, dtype=float)
            lines.append(
                f"| {arm} | {channel} | "
                f"{float(np.average(lls, weights=w)):.4f} | "
                f"{float(np.average(brs, weights=w)):.4f} | {int(w.sum())} |"
            )

    lines += ["", "## Call", ""]
    if decision == "promote_compose":
        lines.append(
            "**promote_compose.** Law-of-total-probability composition beats the "
            "`ĥm` plug-in. Adopt `p = p_play · f_event(μ₊)` as the pre-match path."
        )
    elif decision == "soft_promote_compose":
        lines.append(
            "**soft_promote_compose.** Compose clears zero but misses materiality. "
            "Prefer compose for further pre-match work; keep plug-in as reference."
        )
    elif decision == "keep_plugin":
        lines.append(
            "**keep_plugin.** Compose loses to the plug-in. Keep `f(x(ĥm))` as the "
            "provisional pre-match path (still not a clean −0.034 vs ungated lag-1)."
        )
    else:
        lines.append(
            "**inconclusive_keep_plugin.** Interval covers zero. Keep plug-in "
            "provisional reference; compose does not earn promotion."
        )
    lines += [
        "",
        "Do **not** hard-gate `p_play` with oracle `1e-6`. Ungated lag-1 is the",
        "unconfounded diagnostic baseline for future pre-match screens.",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_minutes_compose`.",
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
    (REPORTS / "typed_tensor_minutes_compose.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_minutes_compose.json").write_text(
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
                "interval_active_actual_minutes": results["interval_active_actual_minutes"],
                "promote": results["promote"],
                "clears_zero": results["clears_zero"],
                "meets_materiality": results["meets_materiality"],
                "folds": [
                    {
                        "holdout": f["holdout"],
                        "plugin": f["plugin"]["all"]["pooled"]["log_loss"],
                        "compose": f["compose"]["all"]["pooled"]["log_loss"],
                        "lag1_ungated": f["lag1_ungated"]["all"]["pooled"]["log_loss"],
                        "inactive_plugin": f["plugin"]["inactive"]["pooled"]["log_loss"],
                        "inactive_compose": f["compose"]["inactive"]["pooled"]["log_loss"],
                        "delta": f["mean_gw_delta"],
                    }
                    for f in results["folds"]
                ],
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
