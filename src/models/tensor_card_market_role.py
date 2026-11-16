"""Market pot + role× interactions vs rich zero-gated spine.

Pre-registered before totals were read:
  reports/typed_tensor_market_role.md

Control: rich zero-gated logistic (mates off; present-row history).
Candidate: same + ah_line_own, Dixon–Coles xg_own, role×{log1p(m), xg_own, m0_rG, m0_rA}.

Cement open-room item. Not a Numagent comparison. No Odds API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.gates import CLOSED_SEASONS, cluster_interval, run_asof_audit
from src.models.tensor_card import (
    ART,
    REPORTS,
    ROOT,
    TensorCard,
    build_tensor_cards,
    iter_slices,
)
from src.models.tensor_card_rich_baseline import (
    CANDIDATE_FEATURES,
    MATERIALITY,
    _fixture_over25,
    load_over25_table,
    rich_extras,
)
from src.teams import norm_team

SEED = 0
N_BOOT = 1000
P_FLOOR = 1e-6
PROCESSED = ROOT / "data" / "processed"
ROLES = ("role_gkp", "role_def", "role_mid", "role_fwd")
ROLE_BASES = ("log1p_minutes", "xg_own", "m0_rG", "m0_rA")
MARKET_CORE = ("ah_line_own", "xg_own")
ROLE_X = tuple(f"{role}_x_{base}" for role in ROLES for base in ROLE_BASES)
MARKET_ROLE_EXTRA = MARKET_CORE + ROLE_X
CONTROL_FEATURES = CANDIDATE_FEATURES
CANDIDATE_FEATURES_MR = CANDIDATE_FEATURES + MARKET_ROLE_EXTRA
CONTRAST = "market_role_minus_rich"


def load_market_pot_table(
    seasons: tuple[str, ...],
) -> dict[tuple[str, str, str, int], dict[str, float]]:
    """Map (season, team_norm, opp_norm, is_home) → {ah_line_own, xg_own}.

    ``is_home`` is required: each club meets each opponent twice, so a key
    without venue silently overwrites the first leg (bug in the published
    ``typed_tensor_market_role`` screen — see that report's confound note).
    """
    out: dict[tuple[str, str, str, int], dict[str, float]] = {}
    for season in seasons:
        stem = season.replace("-", "_")
        path = PROCESSED / f"scoreline_fd_moments_{stem}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            team_n = norm_team(str(row.team))
            opp_n = norm_team(str(row.opp))
            is_home = 1 if str(row.side).lower() == "home" else 0
            ah = float(row.ah_line)
            if not np.isfinite(ah):
                ah = 0.0
            xg = float(row.xg)
            if not np.isfinite(xg):
                xg = 1.3
            out[(season, team_n, opp_n, is_home)] = {
                "ah_line_own": float(ah if is_home else -ah),
                "xg_own": float(xg),
            }
    return out


def _fixture_market(
    card: TensorCard,
    t: int,
    table: dict[tuple[str, str, str, int], dict[str, float]],
) -> dict[str, float]:
    team_n = norm_team(card.team)
    opp_n = norm_team(card.opponent[t]) if card.opponent[t] else ""
    is_home = 1 if int(card.home[t]) else 0
    hit = table.get((card.season, team_n, opp_n, is_home))
    if hit is None:
        return {"ah_line_own": 0.0, "xg_own": 1.3}
    return {"ah_line_own": float(hit["ah_line_own"]), "xg_own": float(hit["xg_own"])}


def market_role_extras(
    features: dict[str, float],
    market: dict[str, float],
) -> dict[str, float]:
    """Build ah_line_own, xg_own, and role× bases from rich + role main effects."""
    out: dict[str, float] = {
        "ah_line_own": float(market["ah_line_own"]),
        "xg_own": float(market["xg_own"]),
    }
    bases = {
        "log1p_minutes": float(features["log1p_minutes"]),
        "xg_own": float(market["xg_own"]),
        "m0_rG": float(features["m0_rG"]),
        "m0_rA": float(features["m0_rA"]),
    }
    for role in ROLES:
        r = float(features[role])
        for base_name, base_val in bases.items():
            out[f"{role}_x_{base_name}"] = r * base_val
    return out


def _collect(
    cards: list[TensorCard],
    over_table: dict,
    market_table: dict,
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
            market = _fixture_market(card, t, market_table)
            feats = dict(row["features"])
            feats.update(rich_extras(card, p, t, over))
            feats.update(market_role_extras(feats, market))
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
    x_cand_tr = _feat_matrix(train_rows, CANDIDATE_FEATURES_MR)
    x_cand_te = _feat_matrix(test_rows, CANDIDATE_FEATURES_MR)

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
    market_table = load_market_pot_table(CLOSED_SEASONS)
    cards = build_tensor_cards(CLOSED_SEASONS)
    by_season: dict[str, list] = {s: [] for s in CLOSED_SEASONS}
    for card in cards:
        by_season[card.season].append(card)

    deltas_by_season: dict[str, np.ndarray] = {}
    deltas_active_by_season: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    n_pred = 0
    n_market_hit = 0
    n_market_miss = 0

    for holdout in CLOSED_SEASONS:
        train_cards = [c for s, cs in by_season.items() if s != holdout for c in cs]
        test_cards = by_season[holdout]
        train_rows = _collect(train_cards, over_table, market_table)
        test_rows = _collect(test_cards, over_table, market_table)
        for card in test_cards:
            for t in range(card.n_fixtures):
                if not np.any(card.present[: card.n_players, t]):
                    continue
                team_n = norm_team(card.team)
                opp_n = norm_team(card.opponent[t]) if card.opponent[t] else ""
                is_home = 1 if int(card.home[t]) else 0
                if (card.season, team_n, opp_n, is_home) in market_table:
                    n_market_hit += 1
                else:
                    n_market_miss += 1

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
        decision = "promote_market_role"
        promote = True
    elif clears and (folds_neg or float(interval["mean"]) < 0.0):
        decision = "soft_promote_market_role"
        promote = False
    elif float(interval["lo"]) > 0.0:
        decision = "keep_rich_baseline"
        promote = False
    else:
        decision = "inconclusive_keep_rich_baseline"
        promote = False

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_market_role",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "materiality": MATERIALITY,
        "clears_zero": clears,
        "meets_materiality": material,
        "control_features": list(CONTROL_FEATURES),
        "candidate_features": list(CANDIDATE_FEATURES_MR),
        "market_role_extra": list(MARKET_ROLE_EXTRA),
        "n_pred_rows": n_pred,
        "n_market_keys": len(market_table),
        "n_market_fixture_hit": n_market_hit,
        "n_market_fixture_miss": n_market_miss,
        "folds": fold_rows,
        "interval": interval,
        "interval_active_minutes": interval_active,
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
    path = REPORTS / "typed_tensor_market_role.md"
    iv = results["interval"]
    iva = results["interval_active_minutes"]
    decision = results["decision"]

    lines = [
        "# Typed tensor — market pot / role interactions",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "Cement open-room item: AH + Dixon–Coles `xg_own` + role× interactions on the",
        "rich zero-gated spine (mates off; present-row history).",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        "| **Control** | Rich zero-gated SliceLogistic (`typed_tensor_rich_baseline` candidate features). |",
        (
            "| **Candidate** | Same + `ah_line_own`, Dixon–Coles `xg_own` (FD opening AH+OU → `M`), "
            "and role × {`log1p(m)`, `xg_own`, `m0_rG`, `m0_rA`}. |"
        ),
        "",
        "## Protocol",
        "",
        "- LOSO on 2022-23…2025-26. `2026-27` untouched. Mates off.",
        "- Market pots from frozen `data/processed/scoreline_fd_moments_*.csv` (no Odds API).",
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
            f"Interval Δ log-loss (market_role − rich): mean {_fmt(iv['mean'])}, "
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
        (
            f"Market fixture coverage (present weeks): hit={results['n_market_fixture_hit']}, "
            f"miss={results['n_market_fixture_miss']} "
            f"(keys={results['n_market_keys']})."
        ),
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
        "| holdout | n_test | n_active | rich LL | market+role LL | Δ mean GW | active Δ |",
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
    if decision == "promote_market_role":
        lines.append(
            "**promote_market_role.** Market pots + role× clear zero and the −0.001 "
            "materiality bar. Adopt on the rich zero-gated spine."
        )
    elif decision == "soft_promote_market_role":
        lines.append(
            "**soft_promote_market_role.** Clears zero but misses materiality. Prefer "
            "market+role features for further slice work; do not treat as a large locked win."
        )
    elif decision == "keep_rich_baseline":
        lines.append(
            "**keep_rich_baseline.** Market pot / role× loses. Keep rich zero-gated spine."
        )
    else:
        lines.append(
            "**inconclusive_keep_rich_baseline.** Interval covers zero. Keep rich "
            "zero-gated spine without market/role extras as default."
        )
    lines += [
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_market_role`.",
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
    (REPORTS / "typed_tensor_market_role.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_market_role.json").write_text(
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
                "n_market_fixture_hit": results["n_market_fixture_hit"],
                "n_market_fixture_miss": results["n_market_fixture_miss"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
