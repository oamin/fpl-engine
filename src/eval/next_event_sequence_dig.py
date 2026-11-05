"""Next-GW event sequence dig — discrete component channels (no expanding).

Ablation
--------
- **B**: position + home/away + opponent id (fixture-t only)
- **C**: B + last-K GW discrete channels (minutes + all FPL points components;
  **not** total_points)

No padding: a player-GW is eligible only once that season has K completed
prior GWs (first label GW = K+1). Shared pool for B and C.

Assess next-GW event log loss / Brier; report Δ(C−B). Chronological split:
train on earlier closed seasons, test on holdout season (default 2025-26).
Live 2026/27 holdout is not used.

  PYTHONPATH=. python3 -m src.eval.next_event_sequence_dig
  PYTHONPATH=. python3 -m src.eval.next_event_sequence_dig --k 5 --test-season 2025-26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.rules.fpl_2026 import (
    DEFCON_THRESHOLD,
    appearance_points,
    assist_points,
    card_points,
    clean_sheet_points,
    defcon_points,
    drop_assistant_managers,
    goal_points,
    goals_conceded_points,
    own_goal_points,
    penalty_points,
    save_points,
)

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

DEFAULT_TRAIN = ("2022-23", "2023-24", "2024-25")
DEFAULT_TEST = "2025-26"
DEFAULT_K = 5
RANDOM_STATE = 0

# Discrete history channels (matches all-points-component bar view; no total_points).
HISTORY_CHANNELS = (
    "minutes",
    "pts_appear",
    "pts_goals",
    "pts_assists",
    "pts_cs",
    "pts_gc",
    "pts_saves",
    "pts_defcon",
    "pts_bonus",
    "pts_cards",
    "pts_own_goal",
    "pts_pens",
)

# Next-GW binary targets.
TARGETS = (
    "y_goal",
    "y_assist",
    "y_appear60",
    "y_defcon",
    "y_bonus",
    "y_cs",
)


def merged_path(season: str) -> Path:
    return CACHE / f"merged_gw_{season.replace('-', '_')}.csv"


def _attach_point_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Vector-ish channel attach (python loop over columns still OK at season scale)."""
    out = df.copy()
    for c in (
        "saves",
        "defensive_contribution",
        "yellow_cards",
        "red_cards",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
    ):
        if c not in out.columns:
            out[c] = 0
    n = len(out)
    pts = {ch: np.zeros(n, dtype=np.float64) for ch in HISTORY_CHANNELS}
    # minutes channel = raw minutes
    pts["minutes"] = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )
    for i in range(n):
        row = out.iloc[i]
        pos = str(row["position"])
        m = int(row["minutes"] or 0)
        g = int(row["goals_scored"] or 0)
        a = int(row["assists"] or 0)
        cs = bool(int(row["clean_sheets"] or 0))
        gc = int(row["goals_conceded"] or 0)
        saves = int(row["saves"] or 0)
        defcon = int(row["defensive_contribution"] or 0)
        bonus = int(row["bonus"] or 0)
        yel = int(row["yellow_cards"] or 0)
        red = int(row["red_cards"] or 0)
        og = int(row["own_goals"] or 0)
        pen_save = int(row["penalties_saved"] or 0)
        pen_miss = int(row["penalties_missed"] or 0)
        pts["pts_appear"][i] = appearance_points(m)
        pts["pts_goals"][i] = goal_points(pos, g)
        pts["pts_assists"][i] = assist_points(a)
        pts["pts_cs"][i] = clean_sheet_points(pos, m, cs)
        pts["pts_gc"][i] = goals_conceded_points(pos, m, gc)
        pts["pts_saves"][i] = save_points(saves)
        pts["pts_defcon"][i] = defcon_points(pos, defcon)
        pts["pts_bonus"][i] = bonus
        pts["pts_cards"][i] = card_points(yel, red)
        pts["pts_own_goal"][i] = own_goal_points(og)
        pts["pts_pens"][i] = penalty_points(pen_save, pen_miss)
    for ch, arr in pts.items():
        out[ch] = arr
    return out


def load_season_player_gw(season: str) -> pd.DataFrame:
    path = merged_path(season)
    df = pd.read_csv(path)
    df = drop_assistant_managers(df)
    need = {
        "element",
        "name",
        "position",
        "team",
        "GW",
        "fixture",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "bonus",
        "was_home",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")

    df = _attach_point_channels(df)

    # Opponent: other team on the same fixture.
    fx_teams = df.drop_duplicates(["fixture", "team"])[["fixture", "team"]]
    paired = fx_teams.merge(fx_teams, on="fixture", suffixes=("", "_opp"))
    paired = paired.loc[paired["team"] != paired["team_opp"]][
        ["fixture", "team", "team_opp"]
    ].rename(columns={"team_opp": "opponent"})
    df = df.merge(paired, on=["fixture", "team"], how="left")
    df["opponent"] = df["opponent"].fillna("UNKNOWN")

    if "defensive_contribution" not in df.columns:
        df["defensive_contribution"] = 0.0

    sum_cols = list(
        dict.fromkeys(
            list(HISTORY_CHANNELS)
            + [
                "goals_scored",
                "assists",
                "clean_sheets",
                "goals_conceded",
                "bonus",
                "defensive_contribution",
            ]
        )
    )
    present_sum = [c for c in sum_cols if c in df.columns]
    first_cols = ["name", "position", "team", "opponent"]
    agg_map: dict[str, Any] = {c: "sum" for c in present_sum}
    agg_map.update({c: "first" for c in first_cols})
    agg_map["was_home"] = "max"
    agg = df.groupby(["element", "GW"], as_index=False).agg(agg_map)
    agg["season"] = season
    pos = agg["position"].astype(str)
    thr = pos.map(lambda p: float(DEFCON_THRESHOLD.get(p, 1e9)))
    agg["y_goal"] = (agg["goals_scored"] >= 1).astype(int)
    agg["y_assist"] = (agg["assists"] >= 1).astype(int)
    agg["y_appear60"] = (agg["minutes"] >= 60).astype(int)
    agg["y_defcon"] = (agg["defensive_contribution"] >= thr).astype(int)
    agg["y_bonus"] = (agg["bonus"] >= 1).astype(int)
    agg["y_cs"] = (agg["clean_sheets"] >= 1).astype(int)
    return agg.sort_values(["element", "GW"]).reset_index(drop=True)


def build_eligible_rows(panel: pd.DataFrame, *, k: int) -> pd.DataFrame:
    """One row per (season, element, t) with t >= k+1 and full prior K GWs present."""
    rows: list[dict[str, Any]] = []
    for (season, element), g in panel.groupby(["season", "element"], sort=False):
        g = g.sort_values("GW")
        by_gw = {int(r.GW): r for r in g.itertuples(index=False)}
        gws = sorted(by_gw)
        for t in gws:
            prior = [t - j for j in range(k, 0, -1)]
            if any(p not in by_gw for p in prior):
                continue  # no padding — skip
            cur = by_gw[t]
            hist = []
            for p in prior:
                pr = by_gw[p]
                hist.extend(float(getattr(pr, ch)) for ch in HISTORY_CHANNELS)
            rows.append(
                {
                    "season": season,
                    "element": int(element),
                    "GW": int(t),
                    "name": cur.name,
                    "position": cur.position,
                    "team": cur.team,
                    "opponent": cur.opponent,
                    "was_home": int(bool(cur.was_home)),
                    "hist": hist,
                    **{y: int(getattr(cur, y)) for y in TARGETS},
                }
            )
    return pd.DataFrame(rows)


def _encode_frame(
    rows: pd.DataFrame,
    *,
    positions: list[str],
    opponents: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X_B, X_C, Y."""
    pos_idx = {p: i for i, p in enumerate(positions)}
    opp_idx = {o: i for i, o in enumerate(opponents)}
    n_pos, n_opp = len(positions), len(opponents)
    n = len(rows)
    k_feats = len(HISTORY_CHANNELS)
    # infer k from hist length
    hist_len = len(rows.iloc[0]["hist"]) if n else 0
    assert hist_len % k_feats == 0
    x_b = np.zeros((n, n_pos + 1 + n_opp), dtype=np.float64)
    x_c = np.zeros((n, n_pos + 1 + n_opp + hist_len), dtype=np.float64)
    y = np.zeros((n, len(TARGETS)), dtype=np.int64)
    for i, r in enumerate(rows.itertuples(index=False)):
        p = pos_idx.get(str(r.position), 0)
        o = opp_idx.get(str(r.opponent), 0)
        x_b[i, p] = 1.0
        x_b[i, n_pos] = float(r.was_home)
        x_b[i, n_pos + 1 + o] = 1.0
        x_c[i, : x_b.shape[1]] = x_b[i]
        hist = np.asarray(r.hist, dtype=np.float64)
        x_c[i, x_b.shape[1] :] = hist
        for j, tname in enumerate(TARGETS):
            y[i, j] = int(getattr(r, tname))
    return x_b, x_c, y


def _mlp() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=RANDOM_STATE,
                    learning_rate_init=1e-3,
                ),
            ),
        ]
    )


def _safe_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if y_true.min() == y_true.max():
        # sklearn log_loss needs both classes; use Bernoulli ll
        return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    return float(log_loss(y_true, p, labels=[0, 1]))


def _safe_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)))


def fit_predict_arm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    """Independent binary MLP per target; return P(test) shape (n_test, n_targets)."""
    n_test = x_test.shape[0]
    n_t = y_train.shape[1]
    out = np.zeros((n_test, n_t), dtype=np.float64)
    for j in range(n_t):
        yj = y_train[:, j]
        if yj.min() == yj.max():
            out[:, j] = float(yj[0])
            continue
        clf = _mlp()
        clf.fit(x_train, yj)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x_test)
            # column for class 1
            classes = list(clf.named_steps["mlp"].classes_)
            if 1 in classes:
                out[:, j] = proba[:, classes.index(1)]
            else:
                out[:, j] = 0.0
        else:
            out[:, j] = clf.predict(x_test).astype(float)
    return out


def evaluate(y_true: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    metrics = {}
    for j, name in enumerate(TARGETS):
        yt = y_true[:, j]
        pj = p[:, j]
        metrics[name] = {
            "log_loss": _safe_log_loss(yt, pj),
            "brier": _safe_brier(yt, pj),
            "base_rate": float(yt.mean()),
            "mean_p": float(pj.mean()),
            "n": int(len(yt)),
        }
    metrics["mean_log_loss"] = float(np.mean([metrics[t]["log_loss"] for t in TARGETS]))
    metrics["mean_brier"] = float(np.mean([metrics[t]["brier"] for t in TARGETS]))
    return metrics


def run_dig(
    *,
    train_seasons: tuple[str, ...],
    test_season: str,
    k: int,
) -> dict[str, Any]:
    panels = []
    for s in list(train_seasons) + [test_season]:
        panels.append(load_season_player_gw(s))
    panel = pd.concat(panels, ignore_index=True)
    eligible = build_eligible_rows(panel, k=k)
    if eligible.empty:
        raise RuntimeError("no eligible rows — check K / merged data")

    train = eligible.loc[eligible["season"].isin(train_seasons)].reset_index(drop=True)
    test = eligible.loc[eligible["season"] == test_season].reset_index(drop=True)
    if train.empty or test.empty:
        raise RuntimeError(f"empty split train={len(train)} test={len(test)}")

    positions = sorted(eligible["position"].astype(str).unique())
    opponents = sorted(eligible["opponent"].astype(str).unique())

    xb_tr, xc_tr, y_tr = _encode_frame(train, positions=positions, opponents=opponents)
    xb_te, xc_te, y_te = _encode_frame(test, positions=positions, opponents=opponents)

    p_b = fit_predict_arm(xb_tr, y_tr, xb_te)
    p_c = fit_predict_arm(xc_tr, y_tr, xc_te)

    m_b = evaluate(y_te, p_b)
    m_c = evaluate(y_te, p_c)
    delta = {
        t: {
            "d_log_loss": m_c[t]["log_loss"] - m_b[t]["log_loss"],
            "d_brier": m_c[t]["brier"] - m_b[t]["brier"],
        }
        for t in TARGETS
    }
    delta["mean_d_log_loss"] = float(m_c["mean_log_loss"] - m_b["mean_log_loss"])
    delta["mean_d_brier"] = float(m_c["mean_brier"] - m_b["mean_brier"])

    return {
        "defs": {
            "k": k,
            "history_channels": list(HISTORY_CHANNELS),
            "targets": list(TARGETS),
            "no_total_points": True,
            "no_padding": True,
            "no_expanding_inputs": True,
            "arm_b": "position one-hot + was_home + opponent one-hot",
            "arm_c": "B + flat last-K discrete channel vector",
            "model": "sklearn MLPClassifier (64,32) + StandardScaler; one head per target",
            "split": f"train {list(train_seasons)} → test {test_season}",
            "note": "negative Δlog_loss means C better than B",
        },
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_positions": len(positions),
        "n_opponents": len(opponents),
        "metrics_b": m_b,
        "metrics_c": m_c,
        "delta_c_minus_b": delta,
    }


def write_report(doc: dict[str, Any], path: Path) -> None:
    d = doc["delta_c_minus_b"]
    lines = [
        "# Next-GW event sequence dig",
        "",
        "Discrete component-channel history (bar-plot ticks) vs fixture-only baseline.",
        "**No** expanding inputs, **no** padding, **no** `total_points` in the history vector.",
        "",
        f"- K = **{doc['defs']['k']}** (first label GW = K+1)",
        f"- Split: `{doc['defs']['split']}`",
        f"- Train rows: **{doc['n_train']}**; test rows: **{doc['n_test']}**",
        f"- History channels: `{', '.join(doc['defs']['history_channels'])}`",
        "",
        "## Arms",
        "",
        f"- **B**: {doc['defs']['arm_b']}",
        f"- **C**: {doc['defs']['arm_c']}",
        f"- Model: {doc['defs']['model']}",
        "",
        "## Mean metrics (test)",
        "",
        "| arm | mean log loss | mean Brier |",
        "| --- | ---: | ---: |",
        f"| B | {doc['metrics_b']['mean_log_loss']:.4f} | {doc['metrics_b']['mean_brier']:.4f} |",
        f"| C | {doc['metrics_c']['mean_log_loss']:.4f} | {doc['metrics_c']['mean_brier']:.4f} |",
        f"| Δ(C−B) | {d['mean_d_log_loss']:+.4f} | {d['mean_d_brier']:+.4f} |",
        "",
        "Negative Δ ⇒ sequence arm better.",
        "",
        "## Per target (test)",
        "",
        "| target | base rate | B ll | C ll | Δll | B brier | C brier | Δbrier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in TARGETS:
        mb, mc = doc["metrics_b"][t], doc["metrics_c"][t]
        dd = d[t]
        lines.append(
            f"| {t} | {mb['base_rate']:.3f} | {mb['log_loss']:.4f} | {mc['log_loss']:.4f} | "
            f"{dd['d_log_loss']:+.4f} | {mb['brier']:.4f} | {mc['brier']:.4f} | "
            f"{dd['d_brier']:+.4f} |"
        )
    lines += [
        "",
        "## Reading",
        "",
    ]
    mean_dll = d["mean_d_log_loss"]
    if mean_dll < -0.01:
        lines.append(
            f"Sequence arm **improves** mean log loss by {-mean_dll:.4f} vs fixture-only."
        )
    elif mean_dll > 0.01:
        lines.append(
            f"Sequence arm **worsens** mean log loss by {mean_dll:.4f} vs fixture-only "
            "(discrete history not helpful under this MLP)."
        )
    else:
        lines.append(
            "Mean log-loss gap is small — no clear win for discrete sequence under this MLP."
        )
    lines += [
        "",
        "Reproduce:",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m src.eval.next_event_sequence_dig",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--test-season", default=DEFAULT_TEST)
    ap.add_argument(
        "--train-seasons",
        nargs="+",
        default=list(DEFAULT_TRAIN),
    )
    args = ap.parse_args(argv)

    doc = run_dig(
        train_seasons=tuple(args.train_seasons),
        test_season=str(args.test_season),
        k=int(args.k),
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    tag = str(args.test_season).replace("-", "_")
    json_path = PROCESSED / f"next_event_sequence_dig_{tag}.json"
    report_path = REPORTS / f"next_event_sequence_dig_{tag}.md"
    json_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    write_report(doc, report_path)
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    d = doc["delta_c_minus_b"]
    print(
        f"n_train={doc['n_train']} n_test={doc['n_test']} "
        f"mean_d_log_loss={d['mean_d_log_loss']:+.4f} "
        f"mean_d_brier={d['mean_d_brier']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
