"""Pre-match minutes head + rich event spine vs lag-1 honesty.

Pre-registered before totals were read:
  reports/typed_tensor_minutes_head.md

Control: rich zero-gated event head with lag-1 minutes at inference.
Candidate: same event head with two-part as-of minutes head
  ĥm = P(m>0) · E[m | m>0].
Oracle diagnostic: actual minutes (ceiling; not a promote arm).

Pivot after fixed-mixer loss / Co-PI: attack the +0.012 lag-1 gap.
Not a Numagent comparison. No Odds API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.gates import CLOSED_SEASONS, cluster_interval, run_asof_audit
from src.models.tensor_card import (
    ART,
    REPORTS,
    TensorCard,
    asof_inputs,
    build_tensor_cards,
    iter_slices,
    lag1_minutes,
    role_minutes_prior,
)
from src.models.tensor_card_rich_baseline import (
    CANDIDATE_FEATURES,
    MATERIALITY,
    _fixture_over25,
    _m0_rates,
    _team_asof_rates,
    load_over25_table,
)

SEED = 0
N_BOOT = 1000
P_FLOOR = 1e-6
EVENT_FEATURES = CANDIDATE_FEATURES
CONTRAST = "minutes_head_minus_lag1"

MINUTES_FEATURES = (
    "role_gkp",
    "role_def",
    "role_mid",
    "role_fwd",
    "home",
    "odds_win",
    "odds_draw",
    "odds_lose",
    "games_played",
    "play_rate",
    "mean_minutes",
    "mean_minutes_played",
    "last_minutes",
    "last5_minutes",
    "lag1_minutes",
    "n_played",
)


def minutes_asof_features(
    card: TensorCard,
    player: int,
    t: int,
    role_prior: dict[int, float],
) -> dict[str, float]:
    """Strictly as-of minutes predictors (no scored-week m_t)."""
    role = int(card.position[player])
    prior = float(role_prior.get(role, 0.0))
    odds = card.odds_1x2[t]
    hist = np.flatnonzero(card.present[player, :t])
    if len(hist):
        m = card.minutes[player, hist].astype(np.float64)
        played = m > 0
        n = float(len(hist))
        play_rate = float(played.mean())
        mean_m = float(m.mean())
        mean_played = float(m[played].mean()) if np.any(played) else 0.0
        last_m = float(m[-1])
        last5 = float(m[-5:].mean())
        n_played = float(played.sum())
    else:
        n = play_rate = mean_m = mean_played = last5 = n_played = 0.0
        last_m = prior
    return {
        "role_gkp": 1.0 if role == 0 else 0.0,
        "role_def": 1.0 if role == 1 else 0.0,
        "role_mid": 1.0 if role == 2 else 0.0,
        "role_fwd": 1.0 if role == 3 else 0.0,
        "home": float(card.home[t]),
        "odds_win": float(odds[0]) if np.isfinite(odds[0]) else 1.0 / 3.0,
        "odds_draw": float(odds[1]) if np.isfinite(odds[1]) else 1.0 / 3.0,
        "odds_lose": float(odds[2]) if np.isfinite(odds[2]) else 1.0 / 3.0,
        "games_played": n,
        "play_rate": play_rate,
        "mean_minutes": mean_m,
        "mean_minutes_played": mean_played,
        "last_minutes": last_m,
        "last5_minutes": last5,
        "lag1_minutes": float(lag1_minutes(card, player, t, role_prior=role_prior)),
        "n_played": n_played,
    }


def rich_extras_for_minutes(
    card: TensorCard,
    player: int,
    t: int,
    over25: float,
    minutes: float,
) -> dict[str, float]:
    team_g, team_a = _team_asof_rates(card, t)
    m0_g, m0_a = _m0_rates(card, player, t)
    return {
        "log1p_minutes": float(np.log1p(max(minutes, 0.0))),
        "minutes_ge60": 1.0 if minutes >= 60.0 else 0.0,
        "minutes_eq90": 1.0 if minutes >= 89.5 else 0.0,
        "p_over25": float(over25),
        "team_asof_goals": team_g,
        "team_asof_assists": team_a,
        "m0_rG": m0_g,
        "m0_rA": m0_a,
    }


def event_features_for_minutes(
    card: TensorCard,
    player: int,
    t: int,
    over25: float,
    minutes: float,
    role_prior: dict[int, float],
) -> dict[str, float]:
    """Rich event features with a substituted minutes value (lag-1 / head / actual)."""
    base = asof_inputs(
        card,
        player,
        t,
        minutes_mode="lag1",
        role_prior=role_prior,
        include_mates=False,
        history_mode="present",
    )
    feats = dict(base)
    feats["minutes"] = float(max(minutes, 0.0))
    feats.update(rich_extras_for_minutes(card, player, t, over25, float(max(minutes, 0.0))))
    return feats


def _collect_rows(
    cards: list[TensorCard],
    over_table: dict,
    role_prior: dict[int, float],
) -> list[dict]:
    rows: list[dict] = []
    for card in cards:
        for row in iter_slices(
            card,
            min_history=1,
            minutes_mode="actual",
            include_mates=False,
            history_mode="present",
        ):
            pids = card.player_id[: card.n_players]
            matches = np.flatnonzero(pids == int(row["player_id"]))
            if len(matches) != 1:
                continue
            p = int(matches[0])
            t = int(row["t"])
            over = _fixture_over25(card, t, over_table)
            actual_m = float(card.minutes[p, t])
            lag1_m = float(lag1_minutes(card, p, t, role_prior=role_prior))
            rows.append(
                {
                    "season": card.season,
                    "team": card.team,
                    "player_id": int(row["player_id"]),
                    "gw": int(row["gw"]),
                    "t": t,
                    "seat": p,
                    "y_goal": int(row["y_goal"]),
                    "y_assist": int(row["y_assist"]),
                    "actual_minutes": actual_m,
                    "lag1_minutes": lag1_m,
                    "minutes_features": minutes_asof_features(card, p, t, role_prior),
                    "over25": over,
                    "card": card,
                }
            )
    return rows


def _mat(rows: list[dict], names: tuple[str, ...], key: str = "features") -> np.ndarray:
    return np.array([[r[key][name] for name in names] for r in rows], dtype=np.float64)


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
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipe.fit(x_tr, y_tr)

    class _Wrap:
        def __init__(self, model: object) -> None:
            self.model = model

        def predict_proba_pos(self, x: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(x)[:, 1]

    return _Wrap(pipe)


def _fit_minutes_head(train_rows: list[dict]) -> tuple[object, object]:
    x = _mat(train_rows, MINUTES_FEATURES, key="minutes_features")
    y_m = np.array([r["actual_minutes"] for r in train_rows], dtype=np.float64)
    y_play = (y_m > 0).astype(np.int64)
    play_model = _fit_logistic(x, y_play)
    active = y_play == 1
    if int(active.sum()) < 10:
        mu = float(y_m[active].mean()) if np.any(active) else 60.0

        class _ConstMu:
            def predict(self, x: np.ndarray) -> np.ndarray:
                return np.full(len(x), mu, dtype=np.float64)

        mins_model = _ConstMu()
    else:
        mins_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        mins_model.fit(x[active], y_m[active])
    return play_model, mins_model


def _predict_hat_m(play_model: object, mins_model: object, rows: list[dict]) -> np.ndarray:
    x = _mat(rows, MINUTES_FEATURES, key="minutes_features")
    p_play = _clip_p(play_model.predict_proba_pos(x))
    mu = np.asarray(mins_model.predict(x), dtype=np.float64)
    mu = np.clip(mu, 0.0, 120.0)
    return np.clip(p_play * mu, 0.0, 120.0)


def _attach_event_features(
    rows: list[dict],
    minutes: np.ndarray,
    role_prior: dict[int, float],
) -> list[dict]:
    out = []
    for row, m in zip(rows, minutes, strict=True):
        feats = event_features_for_minutes(
            row["card"],
            int(row["seat"]),
            int(row["t"]),
            float(row["over25"]),
            float(m),
            role_prior,
        )
        out.append({**row, "features": feats, "used_minutes": float(m)})
    return out


def _predict_gated(model: object, x: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    out = np.full(len(minutes), P_FLOOR, dtype=np.float64)
    active = minutes > 0
    if np.any(active):
        out[active] = _clip_p(model.predict_proba_pos(x[active]))
    return out


def _score_binary(y: np.ndarray, p: np.ndarray) -> dict:
    p = _clip_p(p)
    if len(y) == 0:
        return {"n": 0, "log_loss": float("nan"), "brier": float("nan")}
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _arm_event_preds(
    event_models: dict[str, object],
    rows: list[dict],
) -> dict[str, np.ndarray]:
    minutes = np.array([r["used_minutes"] for r in rows], dtype=np.float64)
    x = _mat(rows, EVENT_FEATURES)
    out = {"minutes": minutes}
    for channel in ("goal", "assist"):
        y = np.array([r[f"y_{channel}"] for r in rows], dtype=np.float64)
        out[f"y_{channel}"] = y
        out[f"p_{channel}"] = _predict_gated(event_models[channel], x, minutes)
    return out


def _subset_metrics(pred: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    metrics = {}
    ps, ys = [], []
    for channel in ("goal", "assist"):
        y = pred[f"y_{channel}"][mask]
        p = pred[f"p_{channel}"][mask]
        metrics[channel] = _score_binary(y, p)
        ps.append(p)
        ys.append(y)
    metrics["pooled"] = _score_binary(np.concatenate(ys), np.concatenate(ps))
    return metrics


def _gw_deltas(
    test_rows: list[dict],
    pred_cand: dict[str, np.ndarray],
    pred_ctrl: dict[str, np.ndarray],
) -> np.ndarray:
    gws = np.array([int(r["gw"]) for r in test_rows], dtype=np.int64)
    deltas = []
    for gw in sorted(set(gws.tolist())):
        m = gws == gw
        nll_c = np.concatenate(
            [_row_nll(pred_cand[f"y_{ch}"][m], pred_cand[f"p_{ch}"][m]) for ch in ("goal", "assist")]
        )
        nll_k = np.concatenate(
            [_row_nll(pred_ctrl[f"y_{ch}"][m], pred_ctrl[f"p_{ch}"][m]) for ch in ("goal", "assist")]
        )
        deltas.append(float(nll_c.mean() - nll_k.mean()))
    return np.asarray(deltas, dtype=float)


def run(seed: int = SEED) -> dict:
    over_table = load_over25_table(CLOSED_SEASONS)
    cards = build_tensor_cards(CLOSED_SEASONS)
    by_season: dict[str, list] = {s: [] for s in CLOSED_SEASONS}
    for card in cards:
        by_season[card.season].append(card)

    deltas_by_season: dict[str, np.ndarray] = {}
    deltas_active_by_season: dict[str, np.ndarray] = {}
    deltas_oracle_by_season: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    n_pred = 0

    for holdout in CLOSED_SEASONS:
        train_cards = [c for s, cs in by_season.items() if s != holdout for c in cs]
        test_cards = by_season[holdout]
        role_prior = role_minutes_prior(train_cards)
        train_rows = _collect_rows(train_cards, over_table, role_prior)
        test_rows = _collect_rows(test_cards, over_table, role_prior)

        play_model, mins_model = _fit_minutes_head(train_rows)
        hat_tr = _predict_hat_m(play_model, mins_model, train_rows)
        hat_te = _predict_hat_m(play_model, mins_model, test_rows)

        # Event head locked recipe: train on actual minutes.
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

        test_lag1 = _attach_event_features(
            test_rows,
            np.array([r["lag1_minutes"] for r in test_rows], dtype=np.float64),
            role_prior,
        )
        test_head = _attach_event_features(test_rows, hat_te, role_prior)
        test_oracle = _attach_event_features(
            test_rows,
            np.array([r["actual_minutes"] for r in test_rows], dtype=np.float64),
            role_prior,
        )

        pred_ctrl = _arm_event_preds(event_models, test_lag1)
        pred_cand = _arm_event_preds(event_models, test_head)
        pred_oracle = _arm_event_preds(event_models, test_oracle)

        # Rename for gw delta helper (cand - ctrl)
        deltas_by_season[holdout] = _gw_deltas(test_rows, pred_cand, pred_ctrl)
        oracle_vs_lag1 = _gw_deltas(test_rows, pred_oracle, pred_ctrl)
        deltas_oracle_by_season[holdout] = oracle_vs_lag1

        actual_m = np.array([r["actual_minutes"] for r in test_rows], dtype=np.float64)
        lag1_m = np.array([r["lag1_minutes"] for r in test_rows], dtype=np.float64)
        active_idx = np.flatnonzero(actual_m > 0)
        if active_idx.size:
            active_rows = [test_rows[i] for i in active_idx]
            active_cand = {k: v[active_idx] for k, v in pred_cand.items()}
            active_ctrl = {k: v[active_idx] for k, v in pred_ctrl.items()}
            deltas_active_by_season[holdout] = _gw_deltas(active_rows, active_cand, active_ctrl)
        else:
            deltas_active_by_season[holdout] = np.asarray([0.0], dtype=float)

        all_mask = np.ones(len(test_rows), dtype=bool)
        active_mask = actual_m > 0
        fold_rows.append(
            {
                "holdout": holdout,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_active": int(active_mask.sum()),
                "lag1": {
                    "all": _subset_metrics(pred_ctrl, all_mask),
                    "active": _subset_metrics(pred_ctrl, active_mask),
                },
                "minutes_head": {
                    "all": _subset_metrics(pred_cand, all_mask),
                    "active": _subset_metrics(pred_cand, active_mask),
                },
                "oracle": {
                    "all": _subset_metrics(pred_oracle, all_mask),
                    "active": _subset_metrics(pred_oracle, active_mask),
                },
                "mean_gw_delta": float(deltas_by_season[holdout].mean()),
                "mean_gw_delta_active": float(deltas_active_by_season[holdout].mean()),
                "mean_gw_delta_oracle_minus_lag1": float(oracle_vs_lag1.mean()),
                "minutes_mae": {
                    "lag1": float(mean_absolute_error(actual_m, lag1_m)),
                    "head": float(mean_absolute_error(actual_m, hat_te)),
                    "n": int(len(actual_m)),
                },
                "train_hat_mae": float(
                    mean_absolute_error(
                        np.array([r["actual_minutes"] for r in train_rows], dtype=np.float64),
                        hat_tr,
                    )
                ),
            }
        )
        n_pred += len(test_rows)

    interval = cluster_interval(deltas_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed)
    interval_active = cluster_interval(
        deltas_active_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed
    )
    interval_oracle = cluster_interval(
        deltas_oracle_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed
    )
    clears = float(interval["hi"]) < 0.0 and float(interval_active["hi"]) < 0.0
    material = float(interval["mean"]) <= MATERIALITY and float(interval_active["mean"]) <= MATERIALITY
    folds_neg = all(f["mean_gw_delta"] < 0.0 for f in fold_rows)
    if clears and material:
        decision = "promote_minutes_head"
        promote = True
    elif clears and (folds_neg or float(interval["mean"]) < 0.0):
        decision = "soft_promote_minutes_head"
        promote = False
    elif float(interval["lo"]) > 0.0:
        decision = "keep_lag1_honesty"
        promote = False
    else:
        decision = "inconclusive_keep_lag1_honesty"
        promote = False

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_minutes_head",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "materiality": MATERIALITY,
        "clears_zero": clears,
        "meets_materiality": material,
        "event_features": list(EVENT_FEATURES),
        "minutes_features": list(MINUTES_FEATURES),
        "n_pred_rows": n_pred,
        "folds": fold_rows,
        "interval": interval,
        "interval_active_actual_minutes": interval_active,
        "interval_oracle_minus_lag1": interval_oracle,
        "asof_audit_passed": bool(audit.get("passed")),
        "asof_audit_failures": list(audit.get("failures") or []),
        "seed": seed,
        "n_boot": N_BOOT,
        "neural_slice_parked": True,
    }


def _fmt(v: float, nd: int = 5) -> str:
    if v != v:
        return "—"
    return f"{v:+.{nd}f}" if nd >= 4 else f"{v:.{nd}f}"


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_minutes_head.md"
    iv = results["interval"]
    iva = results["interval_active_actual_minutes"]
    ivo = results["interval_oracle_minus_lag1"]
    decision = results["decision"]
    lines = [
        "# Typed tensor — pre-match minutes head vs lag-1 honesty",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "Co-PI / Cursor pivot after fixed-mixer loss: stop oracle-minutes feature search.",
        "Attack the **+0.012** lag-1 honesty gap with a pre-match minutes head feeding the",
        "locked rich zero-gated event spine.",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        "| **Control** | Rich zero-gated event head; **lag-1** minutes at inference. |",
        "| **Candidate** | Same event head; **minutes head** `ĥm = P(m>0)·E[m\\|m>0]` (as-of only). |",
        "| **Oracle (diagnostic)** | Same event head with **actual** minutes (ceiling). |",
        "",
        "## Protocol",
        "",
        "- Event head trained on actual minutes (mates off); inference substitutes minutes.",
        "- Minutes head: logistic play + Ridge minutes|play on as-of features (includes lag-1).",
        "- LOSO on 2022-23…2025-26. `2026-27` untouched. Neural slice arm parked.",
        f"- Primary: `Δ = {CONTRAST}` via `cluster_interval` "
        f"(n_boot={results['n_boot']}, seed={results['seed']}).",
        f"- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ {results['materiality']}.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            f"Interval Δ log-loss (minutes_head − lag1): mean {_fmt(iv['mean'])}, "
            f"95% [{_fmt(iv['lo'])}, {_fmt(iv['hi'])}]."
        ),
        (
            f"Active (actual m&gt;0) subset: mean {_fmt(iva['mean'])}, "
            f"95% [{_fmt(iva['lo'])}, {_fmt(iva['hi'])}]."
        ),
        (
            f"Oracle − lag1 (diagnostic ceiling): mean {_fmt(ivo['mean'])}, "
            f"95% [{_fmt(ivo['lo'])}, {_fmt(ivo['hi'])}]."
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
        "## Fold holdouts",
        "",
        "| holdout | n_test | lag1 LL | head LL | oracle LL | Δ head−lag1 | minutes MAE lag1 | MAE head |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in results["folds"]:
        lines.append(
            f"| {fold['holdout']} | {fold['n_test']} | "
            f"{fold['lag1']['all']['pooled']['log_loss']:.4f} | "
            f"{fold['minutes_head']['all']['pooled']['log_loss']:.4f} | "
            f"{fold['oracle']['all']['pooled']['log_loss']:.4f} | "
            f"{fold['mean_gw_delta']:+.5f} | "
            f"{fold['minutes_mae']['lag1']:.2f} | "
            f"{fold['minutes_mae']['head']:.2f} |"
        )

    lines += [
        "",
        "## Per-channel (n-weighted, all rows)",
        "",
        "| arm | channel | log loss | Brier | n |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm in ("lag1", "minutes_head", "oracle"):
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
    if decision == "promote_minutes_head":
        lines.append(
            "**promote_minutes_head.** Pre-match minutes head beats lag-1 on the rich "
            "event spine (clears zero + materiality). Adopt as pre-match minutes path."
        )
    elif decision == "soft_promote_minutes_head":
        lines.append(
            "**soft_promote_minutes_head.** Beats lag-1 but misses materiality. Prefer "
            "minutes head for further pre-match work; keep lag-1 as honesty reference."
        )
    elif decision == "keep_lag1_honesty":
        lines.append(
            "**keep_lag1_honesty.** Minutes head loses to lag-1. Keep lag-1 as the "
            "pre-match minutes reference on the rich event spine."
        )
    else:
        lines.append(
            "**inconclusive_keep_lag1_honesty.** Interval covers zero. Keep lag-1 "
            "honesty reference; minutes head does not earn promotion."
        )
    lines += [
        "",
        "Neural slice / FOV mixer stays parked as a product path under oracle minutes.",
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_minutes_head`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    def _clean(obj):
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "card"}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, TensorCard):
            return None
        return obj

    # Drop card objects before JSON
    payload = _clean(results)
    (REPORTS / "typed_tensor_minutes_head.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_minutes_head.json").write_text(
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
                "interval_oracle_minus_lag1": results["interval_oracle_minus_lag1"],
                "promote": results["promote"],
                "clears_zero": results["clears_zero"],
                "meets_materiality": results["meets_materiality"],
                "fold_mae": [
                    {
                        "holdout": f["holdout"],
                        "mae_lag1": f["minutes_mae"]["lag1"],
                        "mae_head": f["minutes_mae"]["head"],
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
