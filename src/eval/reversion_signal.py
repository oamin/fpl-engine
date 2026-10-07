"""Post-match per-90 z-score. Not a pre-deadline score and not a feature.

The lookback is the previous five active appearances. A week under 30 minutes
does not occupy a slot. The forward week is the sheet week that many steps
ahead, and it is missing when that week is inactive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
LOOKBACK = 5
MIN_MINUTES = 30.0
Z_BAR = 1.5
HORIZONS = (1, 2, 3)
BASELINES = ("exp", "xp")
SPEARMAN_BAR = -0.15
BUCKET_BAR = 0.5
HURST_BAR = 0.5
HURST_MIN_N = 16
FLOOR = 20
MIN_SEASONS = 2
N_BOOT = 1000
SEED = 0

REQUIRED = (
    "Per-90 rates divide a minutes-inclusive forecast by the realized minutes of that appearance.",
    "This is a post-match diagnostic, not a pre-deadline decision.",
    "The forward residual is missing when that later week is not a single fixture of at least 30 minutes.",
    "A bar is cleared only when the whole interval is past it.",
    "No reversion term is added to score_xp.",
    "No winner is declared between score_xp and score_exp_points.",
)
FORBIDDEN = (
    "suggestive",
    "establishes a winner",
    "requires an empirical mean-reversion dampener",
)


def sheet_successor(gameweeks: list[int], gw: int, steps: int) -> int | None:
    """The steps-th later gameweek on the season sheet. No skip of an inactive week."""
    later = [int(item) for item in gameweeks if int(item) > int(gw)]
    if len(later) < int(steps):
        return None
    return int(later[int(steps) - 1])


def season_sheets(panel: pd.DataFrame) -> dict[str, list[int]]:
    sheets: dict[str, list[int]] = {}
    for season, block in panel.groupby("season", sort=False):
        sheets[str(season)] = sorted(int(gw) for gw in block["gw"].unique())
    return sheets


def _residual(actual: pd.Series, baseline: pd.Series, minutes: pd.Series) -> pd.Series:
    return (actual - baseline) * 90.0 / minutes


def active_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Single fixtures of at least 30 minutes. Shorter weeks and doubles are gone."""
    work = panel.copy()
    work["season"] = work["season"].astype(str)
    work["player_id"] = work["player_id"].astype(str)
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    work["n_fix"] = pd.to_numeric(work["n_fix"], errors="coerce")
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce")
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce")
    work["exp"] = pd.to_numeric(work["exp"], errors="coerce")
    work["xp"] = pd.to_numeric(work["xp"], errors="coerce")
    work = work.dropna(subset=["gw", "n_fix", "minutes", "actual"])
    work["gw"] = work["gw"].astype(int)
    keep = (work["n_fix"] == 1) & (work["minutes"] >= MIN_MINUTES)
    work = work.loc[keep].copy()
    if work.duplicated(["season", "gw", "player_id"]).any():
        raise RuntimeError("an active player-week is duplicated")
    work["res_exp"] = _residual(work["actual"], work["exp"], work["minutes"])
    work["res_xp"] = _residual(work["actual"], work["xp"], work["minutes"])
    return work.sort_values(["season", "player_id", "gw"]).reset_index(drop=True)


def _past_mean(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).mean()


def _past_std(series: pd.Series) -> pd.Series:
    return series.shift(1).rolling(LOOKBACK, min_periods=LOOKBACK).std(ddof=1)


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Z uses the previous five active residuals. The current residual stays out."""
    out = frame.copy()
    for name in BASELINES:
        column = f"res_{name}"
        grouped = out.groupby(["season", "player_id"], sort=False)[column]
        mu = grouped.transform(_past_mean)
        sigma = grouped.transform(_past_std)
        z = (out[column] - mu) / sigma
        z = z.where(sigma > 0)
        out[f"mu_{name}"] = mu
        out[f"sigma_{name}"] = sigma
        out[f"z_{name}"] = z
    return out


def add_forward(frame: pd.DataFrame, sheets: dict[str, list[int]]) -> pd.DataFrame:
    """Forward residual on the sheet week h steps ahead. Inactive weeks stay missing."""
    out = frame.copy()
    lookup = out.set_index(["season", "gw", "player_id"], drop=False)
    seasons = out["season"].astype(str).tolist()
    gameweeks = out["gw"].astype(int).tolist()
    players = out["player_id"].astype(str).tolist()
    for horizon in HORIZONS:
        targets: list[int | None] = []
        for season, gw in zip(seasons, gameweeks, strict=True):
            targets.append(sheet_successor(sheets.get(season, []), gw, horizon))
        for name in BASELINES:
            column = f"res_{name}"
            found: list[float] = []
            for season, player, target in zip(seasons, players, targets, strict=True):
                if target is None:
                    found.append(float("nan"))
                    continue
                key = (season, int(target), player)
                if key not in lookup.index:
                    found.append(float("nan"))
                    continue
                found.append(float(lookup.at[key, column]))
            out[f"fwd_{name}_{horizon}"] = found
    return out


def signal_frame(panel: pd.DataFrame) -> pd.DataFrame:
    sheets = season_sheets(panel)
    return add_forward(add_history(active_frame(panel)), sheets)


def _spearman(x_axis: np.ndarray, y_axis: np.ndarray) -> float | None:
    if x_axis.size < 3 or not np.isfinite(x_axis).all() or not np.isfinite(y_axis).all():
        return None
    rx = rankdata(x_axis, method="average")
    ry = rankdata(y_axis, method="average")
    rx = rx - float(rx.mean())
    ry = ry - float(ry.mean())
    denom = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    if denom == 0.0 or not np.isfinite(denom):
        return None
    return float(np.dot(rx, ry) / denom)


def _undefined() -> dict[str, Any]:
    return {"mean": None, "lo": None, "hi": None, "point": None}


def _kept_seasons(counts: dict[str, int], floor: int) -> list[str]:
    return [season for season, n_gws in counts.items() if int(n_gws) >= int(floor)]


def spearman_interval(
    frame: pd.DataFrame,
    z_column: str,
    y_column: str,
    *,
    n_boot: int,
    seed: int,
    floor: int,
) -> dict[str, Any]:
    """Week-cluster interval. Seasons under the floor stay out of the pool."""
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[z_column, y_column])
    bundles: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for season, block in work.groupby("season", sort=False):
        weeks: list[tuple[np.ndarray, np.ndarray]] = []
        for _gw, part in block.groupby("gw", sort=True):
            weeks.append(
                (
                    part[z_column].to_numpy(float),
                    part[y_column].to_numpy(float),
                )
            )
        bundles[str(season)] = weeks
    counts = {season: len(weeks) for season, weeks in bundles.items()}
    kept = _kept_seasons(counts, floor)
    per_season = {
        season: _spearman(
            np.concatenate([pair[0] for pair in weeks]),
            np.concatenate([pair[1] for pair in weeks]),
        )
        if weeks
        else None
        for season, weeks in bundles.items()
    }
    if len(kept) < MIN_SEASONS:
        result = _undefined()
        result["n_gws"] = counts
        result["per_season"] = per_season
        return result
    x_obs = np.concatenate([pair[0] for season in kept for pair in bundles[season]])
    y_obs = np.concatenate([pair[1] for season in kept for pair in bundles[season]])
    observed = _spearman(x_obs, y_obs)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    dropped = 0
    for _draw in range(int(n_boot)):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for season in kept:
            weeks = bundles[season]
            drawn = rng.integers(0, len(weeks), size=len(weeks))
            for index in drawn:
                xs.append(weeks[int(index)][0])
                ys.append(weeks[int(index)][1])
        estimate = _spearman(np.concatenate(xs), np.concatenate(ys))
        if estimate is None:
            dropped += 1
            continue
        boots.append(estimate)
    result: dict[str, Any] = {
        "point": observed,
        "n_gws": counts,
        "per_season": per_season,
        "n_rows": int(x_obs.size),
        "dropped": int(dropped),
    }
    if observed is None or dropped > 0.05 * int(n_boot) or len(boots) < 2:
        result["mean"] = None
        result["lo"] = None
        result["hi"] = None
        return result
    lo, hi = np.quantile(np.asarray(boots, dtype=float), [0.025, 0.975])
    result["mean"] = float(observed)
    result["lo"] = float(lo)
    result["hi"] = float(hi)
    return result


def _bucket_diff(oversold: np.ndarray, overbought: np.ndarray) -> float | None:
    if oversold.size == 0 or overbought.size == 0:
        return None
    return float(oversold.mean() - overbought.mean())


def bucket_interval(
    frame: pd.DataFrame,
    z_column: str,
    y_column: str,
    *,
    n_boot: int,
    seed: int,
    floor: int,
) -> dict[str, Any]:
    """Oversold mean minus overbought mean. Z equal to 1.5 is in neither bucket."""
    work = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[z_column, y_column])
    bundles: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for season, block in work.groupby("season", sort=False):
        weeks: list[tuple[np.ndarray, np.ndarray]] = []
        for _gw, part in block.groupby("gw", sort=True):
            z_values = part[z_column].to_numpy(float)
            y_values = part[y_column].to_numpy(float)
            weeks.append((y_values[z_values < -Z_BAR], y_values[z_values > Z_BAR]))
        bundles[str(season)] = weeks
    counts = {season: len(weeks) for season, weeks in bundles.items()}

    def _season_point(weeks: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
        if not weeks:
            return {"point": None, "n_oversold": 0, "n_overbought": 0}
        oversold = np.concatenate([pair[0] for pair in weeks]) if weeks else np.array([])
        overbought = np.concatenate([pair[1] for pair in weeks]) if weeks else np.array([])
        # concatenate of empty arrays can be shape (0,) or fail if all empty with no rows
        if oversold.size == 0:
            oversold = np.array([])
        if overbought.size == 0:
            overbought = np.array([])
        return {
            "point": _bucket_diff(oversold, overbought),
            "mean_oversold": float(oversold.mean()) if oversold.size else None,
            "mean_overbought": float(overbought.mean()) if overbought.size else None,
            "n_oversold": int(oversold.size),
            "n_overbought": int(overbought.size),
        }

    per_season = {season: _season_point(weeks) for season, weeks in bundles.items()}
    kept = _kept_seasons(counts, floor)
    if len(kept) < MIN_SEASONS:
        result = _undefined()
        result["n_gws"] = counts
        result["per_season"] = per_season
        return result
    oversold = np.concatenate([pair[0] for season in kept for pair in bundles[season]])
    overbought = np.concatenate([pair[1] for season in kept for pair in bundles[season]])
    observed = _bucket_diff(oversold, overbought)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    dropped = 0
    for _draw in range(int(n_boot)):
        over_parts: list[np.ndarray] = []
        bought_parts: list[np.ndarray] = []
        for season in kept:
            weeks = bundles[season]
            drawn = rng.integers(0, len(weeks), size=len(weeks))
            for index in drawn:
                over_parts.append(weeks[int(index)][0])
                bought_parts.append(weeks[int(index)][1])
        estimate = _bucket_diff(np.concatenate(over_parts), np.concatenate(bought_parts))
        if estimate is None:
            dropped += 1
            continue
        boots.append(estimate)
    result = {
        "point": observed,
        "mean_oversold": float(oversold.mean()) if oversold.size else None,
        "mean_overbought": float(overbought.mean()) if overbought.size else None,
        "n_oversold": int(oversold.size),
        "n_overbought": int(overbought.size),
        "n_gws": counts,
        "per_season": per_season,
        "dropped": int(dropped),
    }
    if observed is None or dropped > 0.05 * int(n_boot) or len(boots) < 2:
        result["mean"] = None
        result["lo"] = None
        result["hi"] = None
        return result
    lo, hi = np.quantile(np.asarray(boots, dtype=float), [0.025, 0.975])
    result["mean"] = float(observed)
    result["lo"] = float(lo)
    result["hi"] = float(hi)
    return result


def hurst_rs(values: np.ndarray) -> float:
    """Classical rescaled-range slope on dyadic lags. Short series stay missing."""
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    n_obs = int(series.size)
    if n_obs < HURST_MIN_N:
        return float("nan")
    lags: list[int] = []
    lag = 8
    while lag <= n_obs / 2:
        lags.append(lag)
        lag *= 2
    if len(lags) < 2:
        return float("nan")
    log_lag: list[float] = []
    log_rs: list[float] = []
    for width in lags:
        n_chunks = n_obs // width
        ratios: list[float] = []
        for chunk_index in range(n_chunks):
            chunk = series[chunk_index * width : (chunk_index + 1) * width]
            sigma = float(chunk.std(ddof=1))
            if not np.isfinite(sigma) or sigma == 0.0:
                continue
            cumulative = np.cumsum(chunk - float(chunk.mean()))
            width_range = float(cumulative.max() - cumulative.min())
            if not np.isfinite(width_range) or width_range <= 0.0:
                continue
            ratios.append(width_range / sigma)
        if not ratios:
            continue
        log_lag.append(float(np.log(width)))
        log_rs.append(float(np.log(float(np.mean(ratios)))))
    if len(log_lag) < 2:
        return float("nan")
    x_axis = np.asarray(log_lag, dtype=float)
    y_axis = np.asarray(log_rs, dtype=float)
    centred = x_axis - float(x_axis.mean())
    variance = float(np.dot(centred, centred))
    if variance == 0.0:
        return float("nan")
    return float(np.dot(centred, y_axis - float(y_axis.mean())) / variance)


def hurst_interval(values: np.ndarray, *, n_boot: int, seed: int) -> dict[str, Any]:
    """Median Hurst. The draw resamples player-seasons and does not reorder a series."""
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    n_series = int(series.size)
    if n_series == 0:
        result = _undefined()
        result["n"] = 0
        result["share_below"] = None
        return result
    median = float(np.median(series))
    share = float(np.mean(series < HURST_BAR))
    result = {
        "point": median,
        "n": n_series,
        "share_below": share,
        "mean": None,
        "lo": None,
        "hi": None,
    }
    if n_series < FLOOR:
        return result
    rng = np.random.default_rng(seed)
    boots = np.empty(int(n_boot), dtype=float)
    for draw in range(int(n_boot)):
        taken = rng.choice(series, size=n_series, replace=True)
        boots[draw] = float(np.median(taken))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    result["mean"] = median
    result["lo"] = float(lo)
    result["hi"] = float(hi)
    return result


def player_hurst(frame: pd.DataFrame, column: str) -> np.ndarray:
    found: list[float] = []
    grouped = frame.groupby(["season", "player_id"], sort=False)
    for _key, block in grouped:
        ordered = block.sort_values("gw")
        estimate = hurst_rs(ordered[column].to_numpy(float))
        if np.isfinite(estimate):
            found.append(float(estimate))
    return np.asarray(found, dtype=float)


def _fmt(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        if row.get("point") is None:
            return "undefined"
        return f"point {float(row['point']):+.4f}, interval undefined"
    return f"{float(row['mean']):+.4f} [{float(row['lo']):+.4f}, {float(row['hi']):+.4f}]"


def _clears_below(row: dict[str, Any], bar: float) -> str:
    if row.get("mean") is None:
        return "the bar is not cleared"
    if float(row["hi"]) < float(bar):
        return "the interval stays below the bar"
    return "the bar is not cleared"


def _clears_above(row: dict[str, Any], bar: float) -> str:
    if row.get("mean") is None:
        return "the bar is not cleared"
    if float(row["lo"]) > float(bar):
        return "the interval stays above the bar"
    return "the bar is not cleared"


def _num(value: Any, digits: int = 4) -> str:
    if value is None or not np.isfinite(float(value)):
        return ""
    return f"{float(value):+.{digits}f}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def summarise(
    frame: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    floor: int,
) -> dict[str, Any]:
    found: dict[str, Any] = {"spearman": {}, "bucket": {}, "hurst": {}}
    for name in BASELINES:
        print(f"hurst {name}", flush=True)
        hurst_values = player_hurst(frame, f"res_{name}")
        found["hurst"][name] = hurst_interval(hurst_values, n_boot=n_boot, seed=seed)
        for horizon in HORIZONS:
            key = f"{name}:{horizon}"
            print(f"signal {name} horizon {horizon}", flush=True)
            found["spearman"][key] = spearman_interval(
                frame,
                f"z_{name}",
                f"fwd_{name}_{horizon}",
                n_boot=n_boot,
                seed=seed,
                floor=floor,
            )
            found["bucket"][key] = bucket_interval(
                frame,
                f"z_{name}",
                f"fwd_{name}_{horizon}",
                n_boot=n_boot,
                seed=seed,
                floor=floor,
            )
    return found


def _lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "# Per-90 residual z-score",
        "",
        "Per-90 rates divide a minutes-inclusive forecast by the realized minutes of that appearance.",
        "This is a post-match diagnostic, not a pre-deadline decision.",
        "The forward residual is missing when that later week is not a single fixture of at least 30 minutes.",
        "A bar is cleared only when the whole interval is past it.",
        "",
        "The residual is points per 90 minus the baseline per 90. "
        "The z-score compares that residual with the previous five active appearances in the same season. "
        "A week under 30 minutes does not occupy a slot. A double is excluded. "
        "The current residual is not inside the mean or the standard deviation. "
        "A standard deviation of zero leaves the z-score missing.",
        "",
        "Oversold means the z-score is below -1.5. Overbought means it is above +1.5. "
        "The value 1.5 itself is in neither bucket.",
        "The forward horizons are the next one, two, and three sheet gameweeks. "
        "They were locked before these numbers were read. "
        "Where 2022-23 gameweek 7 is absent, the next sheet week after gameweek 6 is gameweek 8.",
        "The Spearman bar is -0.15. The bucket bar is +0.5 per-90 points, oversold minus overbought. "
        "The Hurst bar is a median below 0.5. "
        "The draw resamples gameweeks inside each season, 1000 times, seed 0. "
        "A season with fewer than 20 signal weeks is omitted from the pool. "
        "Hurst resamples player-seasons and does not reorder a series. "
        "Short-series rescaled range is biased toward 0.5.",
        "The targets were not used to choose the lookback.",
        "",
        "No reversion term is added to score_xp.",
        "No winner is declared between score_xp and score_exp_points.",
        "",
        "## Spearman of the z-score with the forward residual",
        "",
        "| baseline | horizon | correlation | reading |",
        "|---|---:|---|---|",
    ]
    titles = {"exp": "expected points", "xp": "score_xp"}
    for name in BASELINES:
        for horizon in HORIZONS:
            row = summary["spearman"][f"{name}:{horizon}"]
            lines.append(
                f"| {titles[name]} | {horizon} | {_fmt(row)} | {_clears_below(row, SPEARMAN_BAR)} |"
            )
    lines.extend(
        [
            "",
            "## Oversold minus overbought",
            "",
            "| baseline | horizon | oversold | overbought | difference | N oversold | N overbought | reading |",
            "|---|---:|---:|---:|---|---:|---:|---|",
        ]
    )
    for name in BASELINES:
        for horizon in HORIZONS:
            row = summary["bucket"][f"{name}:{horizon}"]
            lines.append(
                "| {title} | {horizon} | {over} | {bought} | {diff} | {n_over} | {n_bought} | {reading} |".format(
                    title=titles[name],
                    horizon=horizon,
                    over=_num(row.get("mean_oversold"), 2),
                    bought=_num(row.get("mean_overbought"), 2),
                    diff=_fmt(row),
                    n_over=row.get("n_oversold", ""),
                    n_bought=row.get("n_overbought", ""),
                    reading=_clears_above(row, BUCKET_BAR),
                )
            )
    lines.extend(
        [
            "",
            "## Hurst exponent of the active residual",
            "",
            "A series needs 16 finite residuals and two dyadic lags, so the shortest series that "
            "enters is 32 appearances. The lag set is powers of two from 8 up to half the length.",
            "",
            "| baseline | series | median | share below 0.5 | reading |",
            "|---|---:|---|---:|---|",
        ]
    )
    for name in BASELINES:
        row = summary["hurst"][name]
        share = row.get("share_below")
        share_text = "" if share is None else f"{float(share):.3f}"
        lines.append(
            f"| {titles[name]} | {int(row.get('n') or 0)} | {_fmt(row)} | {share_text} | {_clears_below(row, HURST_BAR)} |"
        )
    lines.extend(["", "## By season", "", "These are point estimates. They are not a second interval.", ""])
    for name in BASELINES:
        lines.append(f"### {titles[name]}")
        lines.append("")
        lines.append("| season | horizon | spearman | bucket difference | signal weeks |")
        lines.append("|---|---:|---:|---:|---:|")
        for horizon in HORIZONS:
            spearman = summary["spearman"][f"{name}:{horizon}"]
            bucket = summary["bucket"][f"{name}:{horizon}"]
            seasons = list(spearman.get("n_gws") or {})
            for season in seasons:
                point = (spearman.get("per_season") or {}).get(season)
                diff = (bucket.get("per_season") or {}).get(season) or {}
                lines.append(
                    f"| {season} | {horizon} | {_num(point)} | {_num(diff.get('point'))} | "
                    f"{int((spearman.get('n_gws') or {}).get(season, 0))} |"
                )
        lines.append("")
    lines.extend(
        [
            "Gemini kept the formula "
            "([reversion signal](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
            "",
        ]
    )
    return lines


def _flat_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in BASELINES:
        hurst = summary["hurst"][name]
        rows.append(
            {
                "kind": "hurst",
                "baseline": name,
                "horizon": "",
                "point": hurst.get("point"),
                "lo": hurst.get("lo"),
                "hi": hurst.get("hi"),
                "n": hurst.get("n"),
                "share_below": hurst.get("share_below"),
            }
        )
        for horizon in HORIZONS:
            for kind in ("spearman", "bucket"):
                row = summary[kind][f"{name}:{horizon}"]
                rows.append(
                    {
                        "kind": kind,
                        "baseline": name,
                        "horizon": horizon,
                        "point": row.get("point"),
                        "lo": row.get("lo"),
                        "hi": row.get("hi"),
                        "n_oversold": row.get("n_oversold"),
                        "n_overbought": row.get("n_overbought"),
                        "mean_oversold": row.get("mean_oversold"),
                        "mean_overbought": row.get("mean_overbought"),
                        "n_rows": row.get("n_rows"),
                    }
                )
    return rows


def run() -> None:
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season
    from src.eval.reversion import player_gameweeks

    protocol = load_protocol()
    locked = protocol["reversion_signal"]
    if locked.get("installs_feature") or locked.get("winner") is not None:
        raise RuntimeError("the reversion signal edits the score")
    if int(locked["lookback"]) != LOOKBACK or int(locked["min_minutes"]) != int(MIN_MINUTES):
        raise RuntimeError("the reversion signal window is not the locked one")
    if int(locked["bootstrap"]) != N_BOOT or int(locked["seed"]) != SEED:
        raise RuntimeError("the reversion signal draw is not the locked one")
    frames = []
    for season in protocol["closed_seasons"]:
        print(f"signal {season}", flush=True)
        built = build_season(season, protocol["season_codes"][season], protocol)
        frames.append(player_gameweeks(built, season))
    panel = pd.concat(frames, ignore_index=True)
    frame = signal_frame(panel)
    summary = summarise(frame, n_boot=N_BOOT, seed=SEED, floor=int(locked["floor"]))
    lines = _lines(summary)
    text = "\n".join(lines)
    for sentence in REQUIRED:
        if sentence not in text:
            raise RuntimeError("a required sentence is missing")
    for banned in FORBIDDEN:
        if banned in text:
            raise RuntimeError(f"the report contains a banned claim: {banned}")
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    audit = run_asof_audit()
    pd.DataFrame(_flat_rows(summary)).to_csv(PROCESSED / "reversion_signal_summary.csv", index=False)
    (PROCESSED / "reversion_signal_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    write_gated_report(
        REPORTS / "reversion_signal.md",
        audit,
        {
            "seasons": list(protocol["closed_seasons"]),
            "min_gws": 20,
            "comparisons": {
                "score_xp_minus_score_exp_points": certified["comparisons"][
                    "score_xp_minus_score_exp_points"
                ]
            },
        },
        lines,
    )


if __name__ == "__main__":
    run()
