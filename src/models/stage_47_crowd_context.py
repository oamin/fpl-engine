"""Stage 47 — crowd flow and ownership as context, not as a score.

Leave-one-season-out check on Gameweeks 5–38. The question is whether
deadline transfer flow and ownership reduce error after ``score_xp``.
Coefficients are not written back onto the score. No XI is picked.

``log_transfer_flow`` is the difference of the two log flows, not the
log of net transfers. Ownership enters as points per 10 percentage
points, centered on the training-fold mean in both crowd models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.xp_engine import MIN_HISTORY

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEASONS: tuple[str, ...] = ("2022-23", "2023-24", "2024-25", "2025-26")
GW_START = 5
GW_END = 38
ELIGIBLE_XMI = 45.0
MAE_BAR = 0.02
MERGE_MIN = 0.999
SD_FLOOR = 1e-9


class CrowdJoinError(RuntimeError):
    """The deadline transfer file does not cover the eligible rows."""


def norm_element(values: Any) -> pd.Series:
    series = pd.Series(values).astype(str).str.strip()
    return series.str.replace(r"\.0$", "", regex=True)


def log_transfer_flow(transfers_in: np.ndarray, transfers_out: np.ndarray) -> np.ndarray:
    """Difference of log inflows and log outflows."""
    tin = np.asarray(transfers_in, dtype=float)
    tout = np.asarray(transfers_out, dtype=float)
    if np.any(tin < 0) or np.any(tout < 0):
        raise CrowdJoinError("transfer counts are negative")
    return np.log1p(tin) - np.log1p(tout)


def ownership_share(selected: pd.Series, gw: pd.Series) -> pd.Series:
    """Share of managers who own the player. Sum is one row per element."""
    sel = pd.to_numeric(selected, errors="coerce")
    totals = sel.groupby(gw).transform("sum")
    if totals.isna().any() or (totals <= 0).any():
        raise CrowdJoinError("ownership denominator is not positive")
    return 15.0 * sel / totals


def aggregate_player_weeks(feat: pd.DataFrame) -> pd.DataFrame:
    """One row per player-week. The earliest fixture anchors the buy gate."""
    work = pd.DataFrame(
        {
            "season": feat["season"].astype(str),
            "element": norm_element(feat["element"]),
            "gw": pd.to_numeric(feat["gw"], errors="coerce"),
            "date": feat["date"],
            "total_points": pd.to_numeric(feat["total_points"], errors="coerce"),
            "score_xp": pd.to_numeric(feat["score_xp"], errors="coerce"),
            "xmi": pd.to_numeric(feat["xmi"], errors="coerce"),
            "n_prior": pd.to_numeric(feat["n_prior"], errors="coerce"),
        }
    ).dropna(subset=["gw"])
    work["gw"] = work["gw"].astype(int)
    work = work.sort_values(["season", "element", "gw", "date"], kind="mergesort")
    grouped = work.groupby(["season", "element", "gw"], sort=False)
    first = grouped.first()
    summed = grouped[["total_points", "score_xp"]].sum()
    out = first[["date", "xmi", "n_prior"]].copy()
    out["total_points"] = summed["total_points"]
    out["score_xp"] = summed["score_xp"]
    out["n_fixtures"] = grouped.size()
    return out.reset_index()


def eligible_mask(weeks: pd.DataFrame) -> pd.Series:
    gw = weeks["gw"].astype(int)
    xmi = pd.to_numeric(weeks["xmi"], errors="coerce")
    n_prior = pd.to_numeric(weeks["n_prior"], errors="coerce")
    points = pd.to_numeric(weeks["total_points"], errors="coerce")
    score = pd.to_numeric(weeks["score_xp"], errors="coerce")
    return (
        gw.between(GW_START, GW_END)
        & (xmi >= ELIGIBLE_XMI)
        & (n_prior >= MIN_HISTORY)
        & np.isfinite(points)
        & np.isfinite(score)
    )


def dedupe_market(raw: pd.DataFrame, season: str) -> pd.DataFrame:
    """Deadline transfers and ownership, one row per player-week."""
    work = pd.DataFrame(
        {
            "element": norm_element(raw["element"]),
            "gw": pd.to_numeric(raw["GW"], errors="coerce"),
            "kickoff_time": raw["kickoff_time"].astype(str),
            "transfers_in": pd.to_numeric(raw["transfers_in"], errors="coerce"),
            "transfers_out": pd.to_numeric(raw["transfers_out"], errors="coerce"),
            "selected": pd.to_numeric(raw["selected"], errors="coerce"),
        }
    ).dropna(subset=["gw"])
    work["gw"] = work["gw"].astype(int)
    work = work.sort_values(["element", "gw", "kickoff_time"], kind="mergesort")
    first = work.drop_duplicates(["element", "gw"], keep="first").copy()
    finite = (
        first["selected"].notna()
        & first["transfers_in"].notna()
        & first["transfers_out"].notna()
    )
    first = first.loc[finite].copy()
    first["ow"] = ownership_share(first["selected"], first["gw"])
    first["season"] = season
    return first[
        [
            "season",
            "element",
            "gw",
            "transfers_in",
            "transfers_out",
            "selected",
            "ow",
        ]
    ]


def within_week_z(flow: pd.Series, season: pd.Series, gw: pd.Series) -> pd.Series:
    """Z-score inside the eligible rows of one season-week."""
    frame = pd.DataFrame({"flow": flow.astype(float), "season": season, "gw": gw})
    z = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby(["season", "gw"], sort=False).groups.items():
        values = frame.loc[idx, "flow"]
        mu = float(values.mean())
        sd = float(values.std(ddof=0))
        if not np.isfinite(sd) or sd < SD_FLOOR:
            z.loc[idx] = 0.0
        else:
            z.loc[idx] = (values - mu) / sd
    return z


def attach_market(eligible: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Join deadline flow. Abort when a season is short of the merge bar."""
    merged = eligible.merge(
        market[
            [
                "season",
                "element",
                "gw",
                "transfers_in",
                "transfers_out",
                "selected",
                "ow",
            ]
        ],
        on=["season", "element", "gw"],
        how="left",
    )
    ok = (
        merged["transfers_in"].notna()
        & merged["transfers_out"].notna()
        & merged["selected"].notna()
        & merged["ow"].notna()
        & (merged["transfers_in"] >= 0)
        & (merged["transfers_out"] >= 0)
    )
    rates: dict[str, float] = {}
    for season, idx in merged.groupby("season", sort=False).groups.items():
        rate = float(ok.loc[idx].mean()) if len(idx) else 0.0
        rates[str(season)] = rate
        if rate < MERGE_MIN:
            raise CrowdJoinError(
                f"{season}: transfer merge rate {rate:.4f} is below {MERGE_MIN:.3f}"
            )
    out = merged.loc[ok].copy()
    out["log_transfer_flow"] = log_transfer_flow(
        out["transfers_in"].to_numpy(float),
        out["transfers_out"].to_numpy(float),
    )
    out["own_10"] = out["ow"].astype(float) / 0.10
    out["volume_z"] = within_week_z(out["log_transfer_flow"], out["season"], out["gw"]).to_numpy()
    return out, rates


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(pred, float))))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    err = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.sqrt(np.mean(err**2)))


def ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary least squares with homoskedastic standard errors."""
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    n, k = x.shape
    if rank < k:
        raise CrowdJoinError("crowd regression is rank deficient")
    resid = y - x @ beta
    sigma2 = float(resid @ resid) / (n - k)
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    return beta, se


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    columns = [np.ones(len(df))]
    for col in cols:
        columns.append(df[col].to_numpy(float))
    return np.column_stack(columns)


def center_ownership(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Center on the training mean and apply that same mean to the holdout."""
    mu = float(train["own_10"].mean())
    train = train.copy()
    test = test.copy()
    train["own_10_c"] = train["own_10"] - mu
    test["own_10_c"] = test["own_10"] - mu
    train["interact"] = train["volume_z"] * train["own_10_c"]
    test["interact"] = test["volume_z"] * test["own_10_c"]
    return train, test, mu


def qualifies(deltas: list[float], bar: float = MAE_BAR) -> bool:
    """True only when every season improves and the mean clears the bar."""
    if len(deltas) != len(SEASONS):
        raise ValueError("the reading needs one delta per season")
    return all(delta > 0.0 for delta in deltas) and (sum(deltas) / len(deltas)) >= bar


def loso(sample: pd.DataFrame) -> dict[str, Any]:
    """Fit each left-out season on the other three. Do not refit on the holdout."""
    present = set(sample["season"].astype(str))
    missing = [season for season in SEASONS if season not in present]
    if missing:
        raise CrowdJoinError(f"missing seasons: {', '.join(missing)}")
    folds: list[dict[str, Any]] = []
    for holdout in SEASONS:
        train = sample.loc[sample["season"] != holdout]
        test = sample.loc[sample["season"] == holdout]
        train, test, own_mean = center_ownership(train, test)
        y = test["total_points"].to_numpy(float)
        base_beta, _ = ols(_design(train, ["score_xp"]), train["total_points"].to_numpy(float))
        add_beta, add_se = ols(
            _design(train, ["score_xp", "volume_z", "own_10_c"]),
            train["total_points"].to_numpy(float),
        )
        int_beta, int_se = ols(
            _design(train, ["score_xp", "volume_z", "own_10_c", "interact"]),
            train["total_points"].to_numpy(float),
        )
        base_hat = _design(test, ["score_xp"]) @ base_beta
        add_hat = _design(test, ["score_xp", "volume_z", "own_10_c"]) @ add_beta
        int_hat = _design(test, ["score_xp", "volume_z", "own_10_c", "interact"]) @ int_beta
        base_mae = mae(y, base_hat)
        add_mae = mae(y, add_hat)
        int_mae = mae(y, int_hat)
        folds.append(
            {
                "holdout": holdout,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "own_mean_train": own_mean,
                "baseline_mae": base_mae,
                "baseline_rmse": rmse(y, base_hat),
                "additive_mae": add_mae,
                "additive_rmse": rmse(y, add_hat),
                "interaction_mae": int_mae,
                "interaction_rmse": rmse(y, int_hat),
                "delta_additive": base_mae - add_mae,
                "delta_interaction": add_mae - int_mae,
                "a_base": float(base_beta[0]),
                "b_base": float(base_beta[1]),
                "a": float(add_beta[0]),
                "b": float(add_beta[1]),
                "c": float(add_beta[2]),
                "d": float(add_beta[3]),
                "se_c": float(add_se[2]),
                "se_d": float(add_se[3]),
                "e": float(int_beta[4]),
                "se_e": float(int_se[4]),
                "c_interaction": float(int_beta[2]),
                "d_interaction": float(int_beta[3]),
            }
        )
    add_deltas = [float(row["delta_additive"]) for row in folds]
    int_deltas = [float(row["delta_interaction"]) for row in folds]
    return {
        "folds": folds,
        "mean_delta_additive": float(np.mean(add_deltas)),
        "mean_delta_interaction": float(np.mean(int_deltas)),
        "additive_qualifies": qualifies(add_deltas),
        "interaction_qualifies": qualifies(int_deltas),
    }


def _corr(frame: pd.DataFrame, left: str, right: str) -> float:
    return float(frame[left].corr(frame[right]))


def write_report(path: Path, result: dict[str, Any], review: str | None = None) -> None:
    folds = result["folds"]
    add_word = "clears" if result["additive_qualifies"] else "misses"
    int_word = "clears" if result["interaction_qualifies"] else "misses"
    lines = [
        "# Stage 47 — crowd context",
        "",
        "Deadline transfer flow and ownership are checked as their own columns. "
        "`score_xp` is the only score. Nothing in this file is added to it, and "
        "no eleven is picked.",
        "",
        "The flow is `log(1 + transfers in) − log(1 + transfers out)`. "
        "That is the difference of the two log flows. Ownership is the share of "
        "managers who own the player, entered in units of 10 percentage points "
        "and centered on the training-fold mean. Volume is a within-week "
        "standard deviation on the eligible rows. The fit leaves out one season "
        "at a time, using 2022/23, 2023/24, 2024/25, and 2025/26. The holdout "
        "error is mean absolute error on Gameweeks 5–38, among players with at "
        "least 45 expected minutes and at least three prior appearances. A zero-minute "
        "week is absent, because the player log drops it.",
        "",
        "The crowd columns carry held-out information only when the error falls "
        "in every left-out season and the mean fall is at least 0.02 points per "
        "player-week. The interaction is held to that same bar against the "
        "additive fit. Either result stays out of the transfer search.",
        "",
        "## Holdout error",
        "",
        "| holdout | rows | baseline MAE | additive MAE | flow and ownership | interaction MAE | interaction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row['holdout']} | {row['n_test']} | {row['baseline_mae']:.4f} | "
            f"{row['additive_mae']:.4f} | {row['delta_additive']:+.4f} | "
            f"{row['interaction_mae']:.4f} | {row['delta_interaction']:+.4f} |"
        )
    lines += [
        "",
        f"Mean error change, flow and ownership: **{result['mean_delta_additive']:+.4f}**. "
        f"The bar {add_word}.",
        "",
        f"Mean error change, interaction against the additive fit: "
        f"**{result['mean_delta_interaction']:+.4f}**. The bar {int_word}.",
        "",
        "## Training coefficients",
        "",
        "Flow is points per within-week standard deviation. Ownership is points "
        "per 10 percentage points, at the training-fold average. The standard "
        "errors are homoskedastic and are not the gate.",
        "",
        "| training leaves out | flow c | se | ownership d | se | interaction e | se |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        lines.append(
            f"| {row['holdout']} | {row['c']:+.4f} | {row['se_c']:.4f} | "
            f"{row['d']:+.4f} | {row['se_d']:.4f} | {row['e']:+.4f} | {row['se_e']:.4f} |"
        )
    lines += [
        "",
        "## Sample",
        "",
    ]
    for season in SEASONS:
        n = result["n_by_season"][season]
        rate = result["merge_rate"][season]
        lines.append(f"- {season}: {n} eligible rows, transfer merge {rate:.4f}.")
    lines += [
        f"- Eligible weeks that contain two fixtures: {result['n_doubles']}.",
        f"- Correlation of the score with flow: {result['corr_score_flow']:+.3f}.",
        f"- Correlation of the score with ownership: {result['corr_score_own']:+.3f}.",
        f"- Correlation of flow with ownership: {result['corr_flow_own']:+.3f}.",
        "Those three correlations are not the test. The partial coefficients are.",
        "",
        "`score_xp` stays the published score.",
        "",
    ]
    if review:
        lines += [review, ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _fold_frame(result: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(result["folds"])


def run() -> dict[str, Any]:
    from src.models.ridge_multiseason import CACHE, SEASONS as SEASON_CODES, build_one_season

    frames: list[pd.DataFrame] = []
    markets: list[pd.DataFrame] = []
    for season, code in SEASON_CODES:
        if season not in SEASONS:
            continue
        frames.append(build_one_season(season, code))
        raw = pd.read_csv(CACHE / f"merged_gw_{season.replace('-', '_')}.csv")
        markets.append(dedupe_market(raw, season))
    weeks = aggregate_player_weeks(pd.concat(frames, ignore_index=True))
    eligible = weeks.loc[eligible_mask(weeks)].copy()
    sample, rates = attach_market(eligible, pd.concat(markets, ignore_index=True))
    for season in SEASONS:
        if season not in rates:
            raise CrowdJoinError(f"{season}: no eligible rows")
    result = loso(sample)
    result["n_by_season"] = {
        season: int((sample["season"] == season).sum()) for season in SEASONS
    }
    result["n_doubles"] = int((sample["n_fixtures"] > 1).sum())
    result["merge_rate"] = {season: float(rates[season]) for season in SEASONS}
    result["corr_score_flow"] = _corr(sample, "score_xp", "volume_z")
    result["corr_score_own"] = _corr(sample, "score_xp", "own_10")
    result["corr_flow_own"] = _corr(sample, "volume_z", "own_10")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    _fold_frame(result).to_csv(PROCESSED / "stage_47_crowd_context.csv", index=False)
    (PROCESSED / "stage_47_crowd_context.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    write_report(REPORTS / "stage_47_crowd_context.md", result)
    return result


if __name__ == "__main__":
    run()
