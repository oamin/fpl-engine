"""Rich zero-gated SliceLogistic vs plain zero-gated spine.

Pre-registered before totals were read:
  reports/typed_tensor_rich_baseline.md

Control: zero-gated logistic on FEATURE_NAMES_LAG1 (mates off).
Candidate: same + NL minutes, p_over25, team as-of GA, M0=270 own per-90.

Co-PI call after mate co-presence review. Not a Numagent comparison.
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
from src.ingest.fpl_odds import load_football_data
from src.models.tensor_card import (
    ART,
    FEATURE_NAMES_LAG1,
    REPORTS,
    SEASON_FD,
    TensorCard,
    build_tensor_cards,
    iter_slices,
)
from src.teams import norm_team

SEED = 0
N_BOOT = 1000
P_FLOOR = 1e-6
M0 = 270.0
MATERIALITY = -0.001
CONTROL_FEATURES = FEATURE_NAMES_LAG1
RICH_EXTRA = (
    "log1p_minutes",
    "minutes_ge60",
    "minutes_eq90",
    "p_over25",
    "team_asof_goals",
    "team_asof_assists",
    "m0_rG",
    "m0_rA",
)
CANDIDATE_FEATURES = FEATURE_NAMES_LAG1 + RICH_EXTRA
CONTRAST = "rich_baseline_minus_zero_gated"


def load_over25_table(seasons: tuple[str, ...]) -> dict[tuple[str, str, str], float]:
    """Map (season, home_norm, away_norm) → p_over25."""
    out: dict[tuple[str, str, str], float] = {}
    for season in seasons:
        code = SEASON_FD.get(season)
        if not code:
            continue
        try:
            frame = load_football_data(code=code)
        except FileNotFoundError:
            continue
        for row in frame.itertuples(index=False):
            key = (season, str(row.home_norm), str(row.away_norm))
            val = float(row.p_over25)
            out[key] = val if np.isfinite(val) else 0.5
    return out


def _fixture_over25(card: TensorCard, t: int, table: dict[tuple[str, str, str], float]) -> float:
    team_norm = norm_team(card.team)
    opp_norm = norm_team(card.opponent[t]) if card.opponent[t] else ""
    if int(card.home[t]):
        key = (card.season, team_norm, opp_norm)
    else:
        key = (card.season, opp_norm, team_norm)
    return float(table.get(key, 0.5))


def _team_asof_rates(card: TensorCard, t: int) -> tuple[float, float]:
    """Mean team goals / assists per prior fixture (weeks < t with any present)."""
    if t <= 0:
        return 0.0, 0.0
    goals = []
    assists = []
    for s in range(t):
        if not np.any(card.present[: card.n_players, s]):
            continue
        goals.append(float(card.goals[: card.n_players, s].sum()))
        assists.append(float(card.assists[: card.n_players, s].sum()))
    if not goals:
        return 0.0, 0.0
    return float(np.mean(goals)), float(np.mean(assists))


def _m0_rates(card: TensorCard, player: int, t: int) -> tuple[float, float]:
    hist = np.flatnonzero(card.present[player, :t])
    if len(hist) == 0:
        return 0.0, 0.0
    tot_m = float(card.minutes[player, hist].sum())
    tot_g = float(card.goals[player, hist].sum())
    tot_a = float(card.assists[player, hist].sum())
    denom = tot_m + M0
    return 90.0 * tot_g / denom, 90.0 * tot_a / denom


def rich_extras(card: TensorCard, player: int, t: int, over25: float) -> dict[str, float]:
    minutes = float(card.minutes[player, t])
    team_g, team_a = _team_asof_rates(card, t)
    m0_g, m0_a = _m0_rates(card, player, t)
    return {
        "log1p_minutes": float(np.log1p(minutes)),
        "minutes_ge60": 1.0 if minutes >= 60.0 else 0.0,
        "minutes_eq90": 1.0 if minutes >= 89.5 else 0.0,
        "p_over25": float(over25),
        "team_asof_goals": team_g,
        "team_asof_assists": team_a,
        "m0_rG": m0_g,
        "m0_rA": m0_a,
    }


def _collect(cards: list[TensorCard], over_table: dict) -> list[dict]:
    rows: list[dict] = []
    for card in cards:
        for row in iter_slices(
            card,
            min_history=1,
            minutes_mode="actual",
            include_mates=False,
        ):
            pids = card.player_id[: card.n_players]
            matches = np.flatnonzero(pids == int(row["player_id"]))
            if len(matches) != 1:
                continue
            p = int(matches[0])
            t = int(row["t"])
            over = _fixture_over25(card, t, over_table)
            feats = dict(row["features"])
            feats.update(rich_extras(card, p, t, over))
            rows.append({**row, "features": feats})
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
        train_rows = _collect(train_cards, over_table)
        test_rows = _collect(test_cards, over_table)
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
    clears = float(interval["hi"]) < 0.0 and float(interval_active["hi"]) < 0.0
    material = float(interval["mean"]) <= MATERIALITY and float(interval_active["mean"]) <= MATERIALITY
    folds_neg = all(f["mean_gw_delta"] < 0.0 for f in fold_rows)
    if clears and material:
        decision = "promote_rich_baseline"
        promote = True
    elif clears and (folds_neg or float(interval["mean"]) < 0.0):
        decision = "soft_promote_rich_baseline"
        promote = False
    elif float(interval["lo"]) > 0.0:
        decision = "keep_zero_gated_plain"
        promote = False
    else:
        decision = "inconclusive_keep_zero_gated_plain"
        promote = False

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_rich_baseline",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "materiality": MATERIALITY,
        "clears_zero": clears,
        "meets_materiality": material,
        "control_features": list(CONTROL_FEATURES),
        "candidate_features": list(CANDIDATE_FEATURES),
        "rich_extra": list(RICH_EXTRA),
        "m0": M0,
        "n_pred_rows": n_pred,
        "n_over25_keys": len(over_table),
        "folds": fold_rows,
        "interval": interval,
        "interval_active_minutes": interval_active,
        "asof_audit_passed": bool(audit.get("passed")),
        "asof_audit_failures": list(audit.get("failures") or []),
        "copresence_soft_parked": True,
        "seed": seed,
        "n_boot": N_BOOT,
    }


def _fmt(v: float, nd: int = 5) -> str:
    if v != v:
        return "—"
    return f"{v:+.{nd}f}" if nd >= 4 else f"{v:.{nd}f}"


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_rich_baseline.md"
    iv = results["interval"]
    iva = results["interval_active_minutes"]
    decision = results["decision"]

    lines = [
        "# Typed tensor — rich zero-gated baseline",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "Co-PI follow-up after mate co-presence review: soft-park co-presence;",
        "upgrade the single-player zero-gated spine (mates off).",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        "| **Control** | Zero-gated SliceLogistic on `FEATURE_NAMES_LAG1` (mates off). |",
        (
            "| **Candidate** | Same gate + `log1p(m)`, `I(m≥60)`, `I(m=90)`, `p_over25`, "
            f"team as-of G/A, M0={int(results['m0'])} own per-90. |"
        ),
        "",
        "## Protocol",
        "",
        "- LOSO on 2022-23…2025-26. `2026-27` untouched.",
        f"- Primary: `Δ = {CONTRAST}` via `cluster_interval` "
        f"(n_boot={results['n_boot']}, seed={results['seed']}).",
        f"- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ {results['materiality']} "
        "(materiality bar).",
        "- Soft-promote if hi &lt; 0 on both but mean above materiality.",
        "",
        "## Cement side-effect",
        "",
        "Mate co-presence (`typed_tensor_mate_copresence.md`) is **soft-parked** as exploratory.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            f"Interval Δ log-loss (rich − plain): mean {_fmt(iv['mean'])}, "
            f"95% [{_fmt(iv['lo'])}, {_fmt(iv['hi'])}]."
        ),
        (
            f"Active minutes subset: mean {_fmt(iva['mean'])}, "
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
        "| holdout | n_test | n_active | plain LL | rich LL | Δ mean GW | active Δ |",
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
    if decision == "promote_rich_baseline":
        lines.append(
            "**promote_rich_baseline.** Rich features clear zero and the −0.001 "
            "materiality bar. Adopt as zero-gated spine default (mates off; "
            "co-presence stays soft-parked)."
        )
    elif decision == "soft_promote_rich_baseline":
        lines.append(
            "**soft_promote_rich_baseline.** Rich features clear zero but miss the "
            "−0.001 materiality bar. Prefer rich features for further slice work; "
            "do not treat as a large locked win. Co-presence remains soft-parked."
        )
    elif decision == "keep_zero_gated_plain":
        lines.append(
            "**keep_zero_gated_plain.** Rich baseline loses. Keep plain zero-gated "
            "logistic. Co-presence remains soft-parked."
        )
    else:
        lines.append(
            "**inconclusive_keep_zero_gated_plain.** Interval covers zero. Keep "
            "plain zero-gated spine. Co-presence remains soft-parked."
        )
    lines += [
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_rich_baseline`.",
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
    (REPORTS / "typed_tensor_rich_baseline.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_rich_baseline.json").write_text(
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
                "clears_zero": results["clears_zero"],
                "meets_materiality": results["meets_materiality"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
