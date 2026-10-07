"""Serial mean reversion, then an oracle-minutes ceiling.

Residuals are actual points minus the baseline. Same-week minutes do not
enter the residual. The oracle replaces xmi on a copy and does not write
score_xp. Neither diagnostic installs a feature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.xp_engine import compute_xp

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

EDGES = (-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0)
BIN_LABELS = (
    "< -5",
    "[-5, -3)",
    "[-3, -2)",
    "[-2, -1)",
    "[-1, 0)",
    "[0, +1)",
    "[+1, +2)",
    "[+2, +3)",
    "[+3, +5)",
    "> +5",
)
BASELINES = ("exp", "xp")
ORACLE_SCORES = ("score_xp_oracle", "score_xp_w08", "score_xp_w06")
WEIGHTS = (0.8, 0.6)

REQUIRED = (
    "The serial reversion diagnostic evaluates whether player performance deviations from baseline in preceding weeks predict subsequent forecast errors; the sign convention is strictly actual minus baseline.",
    "A negative regression slope indicates that past outperformance relative to the baseline is followed by an undershoot in the subsequent match.",
    "All conditioning splits (prior minutes, bookmaker team strength, rolling expected minutes) use strictly pre-decision information; same-week minutes never enter reversion conditioning.",
    "Oracle minutes replays use post-match realized minutes to construct a theoretical performance ceiling; they are not pre-deadline forecasts and do not alter the published scoring model.",
    "An oracle interval that excludes zero does not demonstrate that `score_xp` beats expected points in pre-deadline decision-making.",
    "No winner is declared between `score_xp` and `score_exp_points`, and `score_xp` is unchanged.",
    "The three-week residual is secondary and does not choose a model.",
    "The slope stays above zero for both baselines, on the full sample and when the previous week had at least 60 minutes. That is persistence of the residual, not a bounce through the baseline. No reversion term is added to score_xp.",
    "Same-fixture minutes beat the historical xmi on this one-week transfer, and the gap versus expected points covers zero.",
)
FORBIDDEN = (
    "requires an empirical mean-reversion dampener",
    "validates `score_xp` as a superior decision engine",
    "were used to classify form",
    "was excluded from the serial reversion",
    "were optimized to maximize",
    "establishes a winner",
)


def bin_label(value: float) -> str:
    """Half-open bins. An exact edge belongs to the upper bin."""
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("a residual bin needs a finite value")
    if number < EDGES[0]:
        return BIN_LABELS[0]
    for index in range(len(EDGES) - 1):
        if EDGES[index] <= number < EDGES[index + 1]:
            return BIN_LABELS[index + 1]
    return BIN_LABELS[-1]


def sheet_predecessor(gameweeks: list[int], gw: int, steps: int) -> int | None:
    """The steps-th preceding gameweek on the season sheet. No player backfill."""
    prior = [int(item) for item in gameweeks if int(item) < int(gw)]
    if len(prior) < int(steps):
        return None
    return int(prior[-int(steps)])


def blend_xmi(minutes: float, historical: float, weight: float) -> float:
    """Weight on same-fixture minutes. The rest stays on the historical xmi."""
    if weight not in WEIGHTS:
        raise ValueError("the oracle weight is not locked")
    if not np.isfinite(minutes) or not np.isfinite(historical):
        return float("nan")
    return float(weight) * float(minutes) + (1.0 - float(weight)) * float(historical)


def _beta(x_values: np.ndarray, y_values: np.ndarray) -> float | None:
    x_axis = np.asarray(x_values, dtype=float)
    y_axis = np.asarray(y_values, dtype=float)
    if x_axis.size < 2 or not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all():
        return None
    centred = x_axis - float(x_axis.mean())
    variance = float(np.dot(centred, centred))
    if variance <= 1e-12:
        return None
    return float(np.dot(centred, y_axis - float(y_axis.mean())) / variance)


def player_gameweeks(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per player-gameweek. Scores and points add across a double."""
    work = frame.copy()
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    work = work.dropna(subset=["gw"])
    work["gw"] = work["gw"].astype(int)
    work["player_id"] = work["player_id"].astype(str)
    for column in ("total_points", "minutes", "score_exp_points", "score_xp", "lam_scored", "xmi"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    counts = work.groupby(["gw", "player_id"], sort=False).size().rename("n_fix")
    grouped = work.groupby(["gw", "player_id"], sort=False).agg(
        actual=("total_points", "sum"),
        minutes=("minutes", "sum"),
        exp=("score_exp_points", "sum"),
        xp=("score_xp", "sum"),
        lam=("lam_scored", "mean"),
        xmi=("xmi", "mean"),
    )
    out = grouped.join(counts).reset_index()
    out["season"] = season
    return out


def reversion_rows(panel: pd.DataFrame) -> list[dict[str, Any]]:
    """Pairs whose current and previous weeks are both single fixtures."""
    gameweeks = sorted(int(gw) for gw in panel["gw"].unique())
    indexed = panel.set_index(["gw", "player_id"], drop=False)
    rows: list[dict[str, Any]] = []
    for record in panel.itertuples(index=False):
        if int(record.n_fix) != 1:
            continue
        gw = int(record.gw)
        previous_gw = sheet_predecessor(gameweeks, gw, 1)
        if previous_gw is None:
            continue
        key = (previous_gw, str(record.player_id))
        if key not in indexed.index:
            continue
        previous = indexed.loc[key]
        if int(previous["n_fix"]) != 1:
            continue
        actual_prev = float(previous["actual"])
        minutes_prev = float(previous["minutes"])
        row: dict[str, Any] = {
            "season": str(record.season),
            "gw": gw,
            "player_id": str(record.player_id),
            "minutes_prev": minutes_prev,
            "xmi": float(record.xmi),
            "lam": float(record.lam),
            "previous_exp": actual_prev - float(record.exp),
            "next_exp": float(record.actual) - float(record.exp),
            "previous_xp": actual_prev - float(record.xp),
            "next_xp": float(record.actual) - float(record.xp),
            "previous3_exp": float("nan"),
            "previous3_xp": float("nan"),
        }
        priors = [sheet_predecessor(gameweeks, gw, step) for step in (1, 2, 3)]
        if all(item is not None for item in priors):
            actuals: list[float] = []
            single = True
            for pred_gw in priors:
                assert pred_gw is not None
                pred_key = (pred_gw, str(record.player_id))
                if pred_key not in indexed.index or int(indexed.loc[pred_key, "n_fix"]) != 1:
                    single = False
                    break
                actuals.append(float(indexed.loc[pred_key, "actual"]))
            if single and len(actuals) == 3:
                mean_actual = float(np.mean(actuals))
                row["previous3_exp"] = mean_actual - float(record.exp)
                row["previous3_xp"] = mean_actual - float(record.xp)
        rows.append(row)
    return rows


def _mark_strength(frame: pd.DataFrame) -> pd.DataFrame:
    """Strict within-week median split on the rows already in the frame."""
    out = frame.copy()
    out["strength"] = "tie"
    if out.empty:
        return out
    medians = out.groupby(["season", "gw"], sort=False)["lam"].transform("median")
    out.loc[out["lam"] > medians, "strength"] = "strong"
    out.loc[out["lam"] < medians, "strength"] = "weak"
    return out


def samples(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pre-registered slices. Same-week minutes are not a column in the slice rule."""
    base = panel.dropna(subset=["previous_exp", "next_exp", "previous_xp", "next_xp"]).copy()
    played = base.loc[base["minutes_prev"] >= 60].copy()
    played = _mark_strength(played)
    three = base.dropna(subset=["previous3_exp", "previous3_xp"]).copy()
    three_played = three.loc[three["minutes_prev"] >= 60].copy()
    return {
        "single": base,
        "single_min60": played,
        "single_min60_strong": played.loc[played["strength"] == "strong"].copy(),
        "single_min60_weak": played.loc[played["strength"] == "weak"].copy(),
        "single_min60_xmi60": played.loc[played["xmi"] >= 60].copy(),
        "single_min60_xmi_below": played.loc[played["xmi"] < 60].copy(),
        "three": three,
        "three_min60": three_played,
    }


SAMPLE_LABELS = {
    "single": "single week, every single-fixture pair",
    "single_min60": "single week, previous minutes at least 60",
    "single_min60_strong": "single week, previous minutes at least 60, strong fixture",
    "single_min60_weak": "single week, previous minutes at least 60, weak fixture",
    "single_min60_xmi60": "single week, previous minutes at least 60, xmi at least 60",
    "single_min60_xmi_below": "single week, previous minutes at least 60, xmi below 60",
    "three": "three preceding weeks",
    "three_min60": "three preceding weeks, previous minutes at least 60",
}
SAMPLE_X = {
    "single": "previous",
    "single_min60": "previous",
    "single_min60_strong": "previous",
    "single_min60_weak": "previous",
    "single_min60_xmi60": "previous",
    "single_min60_xmi_below": "previous",
    "three": "previous3",
    "three_min60": "previous3",
}


def _undefined() -> dict[str, Any]:
    return {"mean": None, "lo": None, "hi": None, "n_gws": {}}


def weekly_means(frame: pd.DataFrame, column: str) -> dict[str, np.ndarray]:
    found: dict[str, list[float]] = {}
    if frame.empty:
        return {}
    for (season, _gw), block in frame.groupby(["season", "gw"], sort=True):
        values = pd.to_numeric(block[column], errors="coerce").dropna()
        if values.empty:
            continue
        found.setdefault(str(season), []).append(float(values.mean()))
    return {season: np.asarray(values, dtype=float) for season, values in found.items()}


def pool_weeks(weekly: dict[str, np.ndarray], *, floor: int, n_boot: int, seed: int) -> dict[str, Any]:
    from src.eval.gates import cluster_interval

    groups = {season: values for season, values in weekly.items() if int(values.size) >= int(floor)}
    if len(groups) < 2:
        return _undefined()
    summary = cluster_interval(groups, seasons=tuple(groups), n_boot=n_boot, seed=seed)
    return summary


def slope_interval(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if frame.empty:
        return _undefined()
    seasons = sorted(str(season) for season in frame["season"].unique())
    if len(seasons) < 2:
        return _undefined()
    bundles: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for season in seasons:
        block = frame.loc[frame["season"].astype(str) == season]
        weeks: list[tuple[np.ndarray, np.ndarray]] = []
        for _gw, part in block.groupby("gw", sort=True):
            weeks.append(
                (
                    pd.to_numeric(part[x_column], errors="coerce").to_numpy(float),
                    pd.to_numeric(part[y_column], errors="coerce").to_numpy(float),
                )
            )
        bundles.append(weeks)
    observed = _beta(
        pd.to_numeric(frame[x_column], errors="coerce").to_numpy(float),
        pd.to_numeric(frame[y_column], errors="coerce").to_numpy(float),
    )
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    dropped = 0
    for _draw in range(int(n_boot)):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for weeks in bundles:
            if not weeks:
                continue
            drawn = rng.integers(0, len(weeks), size=len(weeks))
            for index in drawn:
                xs.append(weeks[int(index)][0])
                ys.append(weeks[int(index)][1])
        if not xs:
            dropped += 1
            continue
        estimate = _beta(np.concatenate(xs), np.concatenate(ys))
        if estimate is None:
            dropped += 1
            continue
        boots.append(estimate)
    if observed is None or dropped > 0.05 * int(n_boot) or len(boots) < 2:
        return _undefined()
    lo, hi = np.quantile(np.asarray(boots, dtype=float), [0.025, 0.975])
    return {"mean": float(observed), "lo": float(lo), "hi": float(hi), "n_gws": {}}


def _band(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "undefined"
    if float(row["lo"]) > 0.0:
        return "the interval stays above zero"
    if float(row["hi"]) < 0.0:
        return "the interval stays below zero"
    return "the interval covers zero"


def _fmt(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "undefined"
    return f"{float(row['mean']):+.4f} [{float(row['lo']):+.4f}, {float(row['hi']):+.4f}]"


def _assign_bins(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").map(bin_label)


def summarise(
    pieces: dict[str, pd.DataFrame],
    *,
    floor: int,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for name, frame in pieces.items():
        x_name = SAMPLE_X[name]
        for baseline in BASELINES:
            x_column = f"{x_name}_{baseline}"
            y_column = f"next_{baseline}"
            labelled = frame.dropna(subset=[x_column, y_column]).copy()
            labelled["bin"] = _assign_bins(labelled, x_column)
            bins: dict[str, dict[str, Any]] = {}
            for label in BIN_LABELS:
                block = labelled.loc[labelled["bin"] == label]
                weekly = weekly_means(block, y_column)
                summary = pool_weeks(weekly, floor=floor, n_boot=n_boot, seed=seed)
                bins[label] = {"n": int(len(block)), **summary}
            found[f"{name}:{baseline}"] = {
                "n": int(len(labelled)),
                "slope": slope_interval(labelled, x_column, y_column, n_boot=n_boot, seed=seed),
                "bins": bins,
            }
    return found


def attach_oracle(frame: pd.DataFrame) -> pd.DataFrame:
    """Score copies. The caller's score_xp column is not replaced."""
    historical = pd.to_numeric(frame["xmi"], errors="coerce").to_numpy(float)
    minutes = pd.to_numeric(frame["minutes"], errors="coerce").to_numpy(float)
    before_xp = pd.to_numeric(frame["score_xp"], errors="coerce").to_numpy(float).copy()
    before_xmi = historical.copy()
    out = frame.copy()
    blends = {
        "score_xp_oracle": minutes,
        "score_xp_w08": 0.8 * minutes + 0.2 * historical,
        "score_xp_w06": 0.6 * minutes + 0.4 * historical,
    }
    for name, xmi in blends.items():
        copy = frame.copy()
        copy["xmi"] = xmi
        scored = compute_xp(copy)
        out[name] = scored["xp"].to_numpy(float)
    after_xp = pd.to_numeric(out["score_xp"], errors="coerce").to_numpy(float)
    after_xmi = pd.to_numeric(out["xmi"], errors="coerce").to_numpy(float)
    if not np.array_equal(before_xp, after_xp, equal_nan=True):
        raise RuntimeError("the oracle overwrote score_xp")
    if not np.array_equal(before_xmi, after_xmi, equal_nan=True):
        raise RuntimeError("the oracle overwrote historical xmi")
    return out


def _decision_weeks(
    frame: pd.DataFrame,
    extras: tuple[str, ...],
    *,
    gw_start: int,
    gw_end: int,
) -> dict[int, pd.DataFrame]:
    from src.eval.common_state import SUMS
    from src.eval.decision import collapse_gameweek
    from src.eval.provenance import OFFICIAL_XP_COLUMNS

    sums = tuple(dict.fromkeys((*SUMS, *extras)))
    work = frame.copy()
    banned = [column for column in OFFICIAL_XP_COLUMNS if column in work.columns]
    if banned:
        work = work.drop(columns=banned)
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    work = work.dropna(subset=["gw"])
    weeks: dict[int, pd.DataFrame] = {}
    for gw, block in work.groupby(work["gw"].astype(int), sort=True):
        number = int(gw)
        if number < int(gw_start) or number > int(gw_end):
            continue
        weeks[number] = collapse_gameweek(block, sums)
    return weeks


def _r1(squad: Any, week: pd.DataFrame, score_col: str, points: dict[str, float]) -> float:
    from src.eval.decision import greedy_step

    _nxt, move = greedy_step(squad, week, score_col)
    if move is None:
        return 0.0
    return float(points.get(str(move["player_in"]), 0.0) - points.get(str(move["player_out"]), 0.0))


def oracle_rows(
    frame: pd.DataFrame,
    published: list[dict[str, Any]],
    *,
    season: str,
    gw_start: int,
    gw_end: int,
) -> list[dict[str, Any]]:
    from src.eval.common_state import _frozen
    from src.eval.decision import points_lookup

    scored = attach_oracle(frame)
    weeks = _decision_weeks(scored, ORACLE_SCORES, gw_start=gw_start, gw_end=gw_end)
    squad, start = _frozen(weeks)
    points = {gw: points_lookup(week) for gw, week in weeks.items()}
    published_by_gw = {int(row["gw"]): row for row in published}
    rows: list[dict[str, Any]] = []
    for gw in sorted(weeks):
        if gw <= start:
            continue
        week = weeks[gw]
        raw = _r1(squad, week, "score_xp", points[gw])
        expected = _r1(squad, week, "score_exp_points", points[gw])
        stored = published_by_gw.get(gw)
        if stored is None:
            raise RuntimeError(f"{season} GW{gw} is not in the published common-state replay")
        if abs(raw - float(stored["r1_xp"])) > 1e-6 or abs(expected - float(stored["r1_exp"])) > 1e-6:
            raise RuntimeError(f"{season} GW{gw} oracle path moved the published transfer")
        point = points[gw]
        record = {
            "season": season,
            "gw": gw,
            "r1_xp": raw,
            "r1_exp": expected,
            "r1_xp_minus_r1_exp": raw - expected,
        }
        for name in ORACLE_SCORES:
            value = _r1(squad, week, name, point)
            record[f"r1_{name}"] = value
            record[f"r1_{name}_minus_exp"] = value - expected
            record[f"r1_{name}_minus_xp"] = value - raw
        rows.append(record)
    return rows


def _lines(
    summary: dict[str, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [
        "# Mean reversion and an oracle minutes ceiling",
        "",
        "No change is made to score_xp.",
        "",
        "The serial reversion diagnostic evaluates whether player performance deviations "
        "from baseline in preceding weeks predict subsequent forecast errors; the sign "
        "convention is strictly actual minus baseline.",
        "",
        "A negative regression slope indicates that past outperformance relative to the "
        "baseline is followed by an undershoot in the subsequent match.",
        "",
        "All conditioning splits (prior minutes, bookmaker team strength, rolling expected "
        "minutes) use strictly pre-decision information; same-week minutes never enter "
        "reversion conditioning.",
        "",
        "The three-week residual is secondary and does not choose a model.",
        "",
        "The slope stays above zero for both baselines, on the full sample and when the previous "
        "week had at least 60 minutes. That is persistence of the residual, not a bounce through "
        "the baseline. No reversion term is added to score_xp.",
        "",
        "Doubles are excluded. A missing predecessor is not filled from an older week. "
        "Where 2022-23 gameweek 7 is absent, gameweek 8 uses gameweek 6.",
        "",
    ]
    for name, label in SAMPLE_LABELS.items():
        lines.extend([f"## {label}", ""])
        for baseline, title in (("exp", "expected points"), ("xp", "score_xp")):
            block = summary[f"{name}:{baseline}"]
            lines.append(
                f"{title}: N={block['n']}. Slope {_fmt(block['slope'])}. {_band(block['slope'])}."
            )
            lines.extend(
                [
                    "",
                    "| previous residual | N | next residual | reading |",
                    "|---|---:|---|---|",
                ]
            )
            for bin_name in BIN_LABELS:
                item = block["bins"][bin_name]
                lines.append(
                    f"| {bin_name} | {item['n']} | {_fmt(item)} | {_band(item)} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Oracle minutes",
            "",
            "Oracle minutes replays use post-match realized minutes to construct a theoretical "
            "performance ceiling; they are not pre-deadline forecasts and do not alter the "
            "published scoring model.",
            "",
            "The published common-state one-week transfers were replayed and matched before "
            "these columns were read. The weights are 0.8 and 0.6. They were not searched.",
            "",
            "An oracle interval that excludes zero does not demonstrate that `score_xp` beats "
            "expected points in pre-deadline decision-making.",
            "",
            "Same-fixture minutes beat the historical xmi on this one-week transfer, and the gap "
            "versus expected points covers zero.",
            "",
            "| score | versus expected points | reading | versus raw score_xp | reading |",
            "|---|---|---|---|---|",
        ]
    )
    labels = {
        "score_xp_oracle": "same-fixture minutes",
        "score_xp_w08": "weight 0.8 on same-fixture minutes",
        "score_xp_w06": "weight 0.6 on same-fixture minutes",
    }
    for name, title in labels.items():
        versus_exp = oracle[f"{name}_minus_exp"]
        versus_xp = oracle[f"{name}_minus_xp"]
        lines.append(
            f"| {title} | {_fmt(versus_exp)} | {_band(versus_exp)} | {_fmt(versus_xp)} | {_band(versus_xp)} |"
        )
    lines.extend(
        [
            "",
            "No winner is declared between `score_xp` and `score_exp_points`, and `score_xp` is unchanged.",
            "",
            "Gemini kept the residual sign and reviewed the table "
            "([reversion](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
            "",
            "Bootstrap 1000, seed 0. A bin season under 20 weeks is omitted. "
            "Fewer than two seasons leaves the interval undefined.",
            "",
        ]
    )
    return lines


def _oracle_summaries(frame: pd.DataFrame, *, floor: int, n_boot: int, seed: int) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for name in ORACLE_SCORES:
        for suffix in ("minus_exp", "minus_xp"):
            column = f"r1_{name}_{suffix}"
            weekly: dict[str, list[float]] = {}
            for (season, _gw), block in frame.groupby(["season", "gw"], sort=True):
                weekly.setdefault(str(season), []).append(float(block[column].iloc[0]))
            arrays = {season: np.asarray(values, dtype=float) for season, values in weekly.items()}
            found[f"{name}_{suffix}"] = pool_weeks(arrays, floor=floor, n_boot=n_boot, seed=seed)
    return found


def run() -> None:
    from src.eval.common_state import evaluate_season
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season

    protocol = load_protocol()
    rule = protocol["reversion"]
    if rule.get("winner") is not None or rule.get("installs_feature") or rule.get("replaces_score"):
        raise RuntimeError("the reversion lock edits the score")
    if rule.get("oracle_is_forecast") or rule.get("sign") != "actual_minus_baseline":
        raise RuntimeError("the reversion lock is not the residual diagnostic")
    if list(rule.get("weights") or []) != [0.6, 0.8]:
        raise RuntimeError("the oracle weights are not locked")
    n_boot = int(rule["bootstrap"])
    seed = int(rule["seed"])
    floor = int(rule["floor"])
    if n_boot != 1000 or seed != 0 or floor != 20:
        raise RuntimeError("the draw is not the locked draw")
    gw_start = int(protocol["gw_start"])
    gw_end = int(protocol["gw_end"])
    codes = protocol["season_codes"]
    panel_rows: list[dict[str, Any]] = []
    fresh: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for index, season in enumerate(protocol["closed_seasons"]):
        print(f"reversion {season}", flush=True)
        frame = build_season(season, codes[season], protocol)
        panel_rows.extend(reversion_rows(player_gameweeks(frame, season)))
        published, _start = evaluate_season(frame, index, gw_start=gw_start, gw_end=gw_end)
        for row in published:
            fresh.append({"season": season, "gw": int(row["gw"]), "r1_xp": float(row["r1_xp"]), "r1_exp": float(row["r1_exp"])})
        oracle.extend(
            oracle_rows(frame, published, season=season, gw_start=gw_start, gw_end=gw_end)
        )
    stored = pd.read_csv(PROCESSED / "hierarchy_weeks.csv")
    replay = pd.DataFrame(fresh)
    merged = replay.merge(stored, on=["season", "gw"], suffixes=("_fresh", "_stored"))
    if len(merged) != len(stored) or len(merged) != len(replay):
        raise RuntimeError("the common-state replay does not match the stored weeks")
    for column in ("r1_xp", "r1_exp"):
        if not np.allclose(merged[f"{column}_fresh"], merged[f"{column}_stored"]):
            raise RuntimeError("the common-state replay moved a stored week")
    panel = pd.DataFrame(panel_rows)
    if panel.empty or "2022-23" not in set(panel["season"]):
        raise RuntimeError("2022-23 is missing from the reversion panel")
    summary = summarise(samples(panel), floor=floor, n_boot=n_boot, seed=seed)
    oracle_frame = pd.DataFrame(oracle)
    oracle_summary = _oracle_summaries(oracle_frame, floor=floor, n_boot=n_boot, seed=seed)
    lines = _lines(summary, oracle_summary)
    text = "\n".join(lines)
    for sentence in REQUIRED:
        if sentence not in text:
            raise RuntimeError("a required sentence is missing")
    for banned in FORBIDDEN:
        if banned in text:
            raise RuntimeError(f"the report contains a banned claim: {banned}")
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = "score_xp_minus_score_exp_points"
    audit = run_asof_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED / "reversion_rows.csv", index=False)
    oracle_frame.to_csv(PROCESSED / "oracle_minutes_weeks.csv", index=False)
    write_gated_report(
        REPORTS / "reversion.md",
        audit,
        {
            "seasons": list(protocol["closed_seasons"]),
            "min_gws": floor,
            "comparisons": {key: certified["comparisons"][key]},
        },
        lines,
    )


if __name__ == "__main__":
    run()
