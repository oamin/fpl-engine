"""Leave-one-season-out of contrasts that are already stored.

The function reads weekly rows and resamples them. It does not replay a
season, rebuild a score, or replace the published four-season interval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.decision_spec import (
    LIVE_PRIMARY,
    LIVE_PRIMARY_WIRED,
    LIVE_SHADOW,
    LOSO_BOOTSTRAP,
    LOSO_CONDITIONAL_FLOOR,
    LOSO_FLOOR,
    LOSO_MIN_SEASONS,
    LOSO_SEED,
    SCORE_COLUMN,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")

# Point estimates already published. Used only to refuse a silent replacement.
PUBLISHED_MEANS: dict[str, float] = {
    "placebo:greedy_xp_minus_greedy_exp": -1.6667,
    "placebo:greedy_shuffled_minus_hold_shuffled": 1.8444,
    "placebo:greedy_xp_minus_greedy_shuffled": 15.0519,
    "placebo:greedy_xp_minus_hold_xp": 7.3556,
    "hierarchy:r1_xp_minus_r1_exp": -0.2901,
    "hierarchy:r3_xp_minus_r3_exp": -1.6947,
    "hierarchy:r1_xp_minus_r1_roll3": 0.2672,
    "hierarchy:r1_xp_minus_r1_shuffled": 3.9771,
    "hierarchy:r1_exp_minus_r1_shuffled": 4.2672,
    "initial:score_xp:xp_minus_exp": -2.1852,
    "initial:score_xp:xp_minus_shuffled": 7.7481,
    "initial:score_xp:xp_minus_neutral": 4.6370,
    "initial:score_exp_points:xp_minus_exp": -3.3037,
    "initial:score_exp_points:xp_minus_shuffled": 7.7556,
    "initial:score_exp_points:xp_minus_neutral": 4.1778,
    "disagreement:r1_xp_minus_r1_exp": -0.7451,
    "disagreement:r3_xp_minus_r3_exp": -4.3529,
}

CONTRASTS: tuple[dict[str, Any], ...] = (
    {
        "key": "placebo:greedy_xp_minus_greedy_exp",
        "file": "placebo_weeks.csv",
        "column": "greedy_xp_minus_greedy_exp",
        "label": "greedy score_xp − greedy expected points",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "placebo:greedy_shuffled_minus_hold_shuffled",
        "file": "placebo_weeks.csv",
        "column": "greedy_shuffled_minus_hold_shuffled",
        "label": "shuffled greedy − shuffled hold",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "placebo:greedy_xp_minus_greedy_shuffled",
        "file": "placebo_weeks.csv",
        "column": "greedy_xp_minus_greedy_shuffled",
        "label": "greedy score_xp − shuffled greedy",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "placebo:greedy_xp_minus_hold_xp",
        "file": "placebo_weeks.csv",
        "column": "greedy_xp_minus_hold_xp",
        "label": "score_xp greedy − score_xp hold",
        "strawman": True,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "hierarchy:r1_xp_minus_r1_exp",
        "file": "hierarchy_weeks.csv",
        "column": "r1_xp_minus_r1_exp",
        "label": "common state, one-week score_xp − expected points",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "hierarchy:r3_xp_minus_r3_exp",
        "file": "hierarchy_weeks.csv",
        "column": "r3_xp_minus_r3_exp",
        "label": "common state, three-week score_xp − expected points",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "hierarchy:r1_xp_minus_r1_roll3",
        "file": "hierarchy_weeks.csv",
        "column": "r1_xp_minus_r1_roll3",
        "label": "common state, one-week score_xp − rolling three-week points",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "hierarchy:r1_xp_minus_r1_shuffled",
        "file": "hierarchy_weeks.csv",
        "column": "r1_xp_minus_r1_shuffled",
        "label": "common state, one-week score_xp − shuffled score",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "hierarchy:r1_exp_minus_r1_shuffled",
        "file": "hierarchy_weeks.csv",
        "column": "r1_exp_minus_r1_shuffled",
        "label": "common state, one-week expected points − shuffled score",
        "strawman": False,
        "conditional": False,
        "deployment": None,
    },
    {
        "key": "initial:score_xp:xp_minus_exp",
        "file": "initial_squad_weeks.csv",
        "column": "xp_minus_exp",
        "label": "opening fifteen, XI by score_xp, xp − exp",
        "strawman": False,
        "conditional": False,
        "deployment": "score_xp",
    },
    {
        "key": "initial:score_xp:xp_minus_shuffled",
        "file": "initial_squad_weeks.csv",
        "column": "xp_minus_shuffled",
        "label": "opening fifteen, XI by score_xp, xp − shuffled",
        "strawman": False,
        "conditional": False,
        "deployment": "score_xp",
    },
    {
        "key": "initial:score_xp:xp_minus_neutral",
        "file": "initial_squad_weeks.csv",
        "column": "xp_minus_neutral",
        "label": "opening fifteen, XI by score_xp, xp − price ladder",
        "strawman": False,
        "conditional": False,
        "deployment": "score_xp",
    },
    {
        "key": "initial:score_exp_points:xp_minus_exp",
        "file": "initial_squad_weeks.csv",
        "column": "xp_minus_exp",
        "label": "opening fifteen, XI by expected points, xp − exp",
        "strawman": False,
        "conditional": False,
        "deployment": "score_exp_points",
    },
    {
        "key": "initial:score_exp_points:xp_minus_shuffled",
        "file": "initial_squad_weeks.csv",
        "column": "xp_minus_shuffled",
        "label": "opening fifteen, XI by expected points, xp − shuffled",
        "strawman": False,
        "conditional": False,
        "deployment": "score_exp_points",
    },
    {
        "key": "initial:score_exp_points:xp_minus_neutral",
        "file": "initial_squad_weeks.csv",
        "column": "xp_minus_neutral",
        "label": "opening fifteen, XI by expected points, xp − price ladder",
        "strawman": False,
        "conditional": False,
        "deployment": "score_exp_points",
    },
    {
        "key": "disagreement:r1_xp_minus_r1_exp",
        "file": "disagreement_weeks.csv",
        "column": "r1_xp_minus_r1_exp",
        "label": "disagreement weeks, one-week score_xp − expected points",
        "strawman": False,
        "conditional": True,
        "deployment": None,
    },
    {
        "key": "disagreement:r3_xp_minus_r3_exp",
        "file": "disagreement_weeks.csv",
        "column": "r3_xp_minus_r3_exp",
        "label": "disagreement weeks, three-week score_xp − expected points",
        "strawman": False,
        "conditional": True,
        "deployment": None,
    },
)

IDENTITY_SENTENCE = (
    "The two 2022–23 series (the opening-portfolio gap under the expected-points XI "
    "and the diverging squad gap) both average −14.45 points per gameweek and sum to −477, "
    "but agree on only 1 of 33 weeks and represent distinct processes."
)
REQUIRED_SENTENCES = (
    "The published four-season intervals remain the official evaluation benchmarks and are unchanged.",
    "Leave-one-season-out intervals measure the sensitivity of pooled estimates to individual season cohorts; they are diagnostic and do not replace the four-season results.",
    "A sign flip or shift in interval bounds when holding out a season is evidence of cohort heterogeneity, not a justification to drop any season or declare a winner.",
    IDENTITY_SENTENCE,
    "The 2022–23 deficit is distributed across the season, is present both before and after the World Cup hiatus, and is not caused by missing odds joins (0 unmatched of 26,505 rows).",
    "For live squad decisions from gameweek 6 onward, the primary score is pre-registered as `ep_next`, with `score_xp` logged alongside.",
    "Promotion of `score_xp` over `ep_next` requires the paired live interval across at least 20 pre-deadline gameweeks to stay strictly above zero; an interval covering zero is undetermined and testing continues through gameweek 38.",
    "`greedy_xp_minus_hold_xp` is reported as the previously retired strawman baseline (a static squad that never transfers) and does not measure transfer skill.",
)
FORBIDDEN = (
    "proves that `score_xp` outperforms",
    "unrepresentative outlier",
    "statistically significant positive transfer edge of +2.5",
    "declared the winner over",
    "leave-one-season-out calibrated hit hurdles",
    "explained by the World Cup break or odds data gaps",
    "has concluded no difference",
    "underlying transfer advantage",
    "definitively proves",
    "has been promoted over",
    "validates the transfer model",
)
XP_VERSUS_EXP = (
    "placebo:greedy_xp_minus_greedy_exp",
    "hierarchy:r1_xp_minus_r1_exp",
    "hierarchy:r3_xp_minus_r3_exp",
    "initial:score_xp:xp_minus_exp",
    "initial:score_exp_points:xp_minus_exp",
    "disagreement:r1_xp_minus_r1_exp",
    "disagreement:r3_xp_minus_r3_exp",
)
SHUFFLE_KEYS = (
    "placebo:greedy_xp_minus_greedy_shuffled",
    "hierarchy:r1_xp_minus_r1_shuffled",
    "hierarchy:r1_exp_minus_r1_shuffled",
)


def contrast_keys() -> tuple[str, ...]:
    return tuple(str(item["key"]) for item in CONTRASTS)


def refuse_replacement(replaces_published: bool) -> None:
    """The sensitivity is not a new published interval."""
    if replaces_published:
        raise RuntimeError("leave-one-season-out must not replace the four-season interval")


def _groups(
    rows: pd.DataFrame,
    column: str,
    *,
    minimum: int,
    seasons: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    complete: dict[str, np.ndarray] = {}
    incomplete: dict[str, int] = {}
    for season in seasons:
        values = rows.loc[rows["season"].astype(str) == season, column].to_numpy(float)
        if int(values.size) == 0:
            continue
        if int(values.size) >= int(minimum):
            complete[season] = values
        else:
            incomplete[season] = int(values.size)
    return complete, incomplete


def pool_stored(
    rows: pd.DataFrame,
    column: str,
    *,
    minimum: int = LOSO_FLOOR,
    n_boot: int = LOSO_BOOTSTRAP,
    seed: int = LOSO_SEED,
    seasons: tuple[str, ...] = SEASONS,
) -> dict[str, Any]:
    """Pool the seasons that clear the floor. The input frame is not mutated."""
    from src.eval.gates import cluster_interval

    complete, incomplete = _groups(rows, column, minimum=minimum, seasons=seasons)
    result: dict[str, Any] = {
        "incomplete_seasons": incomplete,
        "undefined": False,
        "mean": None,
        "lo": None,
        "hi": None,
        "n_gws": {},
    }
    if len(complete) < LOSO_MIN_SEASONS:
        result["undefined"] = True
        return result
    summary = cluster_interval(complete, seasons=tuple(complete), n_boot=int(n_boot), seed=int(seed))
    result.update(summary)
    result["incomplete_seasons"] = incomplete
    result["undefined"] = False
    return result


def leave_one_out(
    rows: pd.DataFrame,
    column: str,
    *,
    holdout: str,
    minimum: int = LOSO_FLOOR,
    n_boot: int = LOSO_BOOTSTRAP,
    seed: int = LOSO_SEED,
    min_seasons: int = LOSO_MIN_SEASONS,
    seasons: tuple[str, ...] = SEASONS,
) -> dict[str, Any]:
    """Drop one season and pool the rest. The input frame is not mutated."""
    from src.eval.gates import cluster_interval

    kept = rows.loc[rows["season"].astype(str) != str(holdout)]
    complete, incomplete = _groups(
        kept, column, minimum=minimum, seasons=tuple(s for s in seasons if s != str(holdout))
    )
    result: dict[str, Any] = {
        "holdout": str(holdout),
        "incomplete_seasons": incomplete,
        "undefined": False,
        "mean": None,
        "lo": None,
        "hi": None,
        "n_gws": {},
    }
    if len(complete) < int(min_seasons):
        result["undefined"] = True
        return result
    summary = cluster_interval(
        complete, seasons=tuple(complete), n_boot=int(n_boot), seed=int(seed)
    )
    result.update(summary)
    result["holdout"] = str(holdout)
    result["incomplete_seasons"] = incomplete
    result["undefined"] = False
    return result


def reading(row: dict[str, Any]) -> str:
    if row.get("undefined") or row.get("mean") is None:
        return "not identified"
    lo = float(row["lo"])
    hi = float(row["hi"])
    if lo <= 0.0 <= hi:
        return "the interval covers zero"
    if hi < 0.0:
        return "the interval stays below zero"
    return "the interval stays above zero"


def sign_differs(estimate: float | None, published: float) -> bool:
    if estimate is None or not np.isfinite(float(estimate)):
        return False
    return float(estimate) * float(published) < 0.0


def _pts(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-9:
        return f"{int(round(float(value))):+d}"
    return f"{float(value):+.2f}"


def _interval_text(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "not identified"
    return f"{float(row['mean']):+.4f} [{float(row['lo']):+.4f}, {float(row['hi']):+.4f}]"


def _load_column(spec: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / str(spec["file"]))
    if spec["deployment"] is not None:
        frame = frame.loc[frame["deployment"].astype(str) == str(spec["deployment"])].copy()
    column = str(spec["column"])
    if column not in frame.columns or "season" not in frame.columns:
        raise RuntimeError(f"stored contrast {spec['key']} is missing")
    out = frame.loc[:, ["season", "gw", column]].copy()
    out["season"] = out["season"].astype(str)
    values = pd.to_numeric(out[column], errors="coerce").to_numpy(float)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"non-finite values in {spec['key']}")
    out[column] = values
    return out


def _check_published_mean(rows: pd.DataFrame, spec: dict[str, Any]) -> float:
    column = str(spec["column"])
    minimum = LOSO_CONDITIONAL_FLOOR if spec["conditional"] else LOSO_FLOOR
    complete, incomplete = _groups(rows, column, minimum=minimum, seasons=SEASONS)
    if incomplete:
        raise RuntimeError(f"{spec['key']} has an incomplete season in the published pool")
    if tuple(complete) != SEASONS:
        raise RuntimeError(f"{spec['key']} does not cover the four closed seasons")
    observed = float(np.concatenate([complete[season] for season in SEASONS]).mean())
    published = float(PUBLISHED_MEANS[str(spec["key"])])
    if abs(observed - published) > 5e-5:
        raise RuntimeError(
            f"{spec['key']} re-aggregates to {observed:+.4f}, not the published {published:+.4f}"
        )
    return published


def _pair_2022(opening: pd.DataFrame, forensic: pd.DataFrame) -> pd.DataFrame:
    left = opening.loc[
        (opening["season"].astype(str) == "2022-23")
        & (opening["deployment"].astype(str) == "score_exp_points"),
        ["gw", "xp_minus_exp"],
    ].copy()
    right = forensic.loc[:, ["gw", "squad_gap"]].copy()
    left["gw"] = pd.to_numeric(left["gw"], errors="raise").astype(int)
    right["gw"] = pd.to_numeric(right["gw"], errors="raise").astype(int)
    paired = left.merge(right, on="gw", how="outer").sort_values("gw")
    if paired[["xp_minus_exp", "squad_gap"]].isna().any().any():
        raise RuntimeError("the two 2022-23 series do not share their gameweeks")
    values_left = paired["xp_minus_exp"].to_numpy(float)
    values_right = paired["squad_gap"].to_numpy(float)
    if not np.isfinite(values_left).all() or not np.isfinite(values_right).all():
        raise RuntimeError("a 2022-23 gap is not finite")
    n_equal = int(np.isclose(values_left, values_right, atol=1e-9).sum())
    if (
        len(paired) != 33
        or n_equal != 1
        or not np.isclose(values_left.sum(), -477.0)
        or not np.isclose(values_right.sum(), -477.0)
        or 7 in set(paired["gw"].tolist())
    ):
        raise RuntimeError(
            "the two 2022-23 series are not the pair already checked "
            f"(n={len(paired)}, equal={n_equal}, "
            f"sums={values_left.sum():.1f},{values_right.sum():.1f})"
        )
    before = paired["gw"] <= 16
    after = paired["gw"] >= 17
    for column in ("xp_minus_exp", "squad_gap"):
        if not ((paired.loc[before, column] < 0).any() and (paired.loc[after, column] < 0).any()):
            raise RuntimeError(f"{column} is not negative on both sides of the break")
    return paired


def _folds_for(spec: dict[str, Any], rows: pd.DataFrame, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    column = str(spec["column"])
    minimum = LOSO_CONDITIONAL_FLOOR if spec["conditional"] else LOSO_FLOOR
    published = float(PUBLISHED_MEANS[str(spec["key"])])
    folds = []
    for season in SEASONS:
        fold = leave_one_out(
            rows,
            column,
            holdout=season,
            minimum=minimum,
            n_boot=n_boot,
            seed=seed,
        )
        fold["key"] = str(spec["key"])
        fold["label"] = str(spec["label"])
        fold["strawman"] = bool(spec["strawman"])
        fold["conditional"] = bool(spec["conditional"])
        fold["published_mean"] = published
        fold["sign_differs"] = sign_differs(fold.get("mean"), published)
        fold["reading"] = reading(fold)
        folds.append(fold)
    return folds


def _week_lines(paired: pd.DataFrame) -> list[str]:
    left = paired["xp_minus_exp"].to_numpy(float)
    right = paired["squad_gap"].to_numpy(float)
    match = paired.loc[np.isclose(left, right, atol=1e-9), "gw"]
    corr = float(np.corrcoef(left, right)[0, 1])
    lines = [
        "Two computations for 2022-23 both average −14.45. They are not the same weeks.",
        "",
        "The opening-portfolio column is the frozen score_xp fifteen minus the frozen "
        "expected-points fifteen, with the XI chosen by expected points, and no transfer. "
        "The diverging column is each score building its own squad and then transferring. "
        "Gameweek 7 is absent from both.",
        "",
        "| GW | opening portfolio | diverging squad |",
        "|---:|---:|---:|",
    ]
    for record in paired.itertuples(index=False):
        lines.append(
            f"| {int(record.gw)} | {_pts(float(record.xp_minus_exp))} | {_pts(float(record.squad_gap))} |"
        )
    lines.extend(
        [
            "",
            IDENTITY_SENTENCE,
            f"The matching week is gameweek {int(match.iloc[0])}. "
            f"The correlation of the two weekly series is {corr:+.4f}.",
            "The 2022–23 deficit is distributed across the season, is present both before and "
            "after the World Cup hiatus, and is not caused by missing odds joins "
            "(0 unmatched of 26,505 rows).",
            "That odds-join count is the one already printed in `reports/decision_placebo.md`. "
            "It was not recomputed.",
            "",
        ]
    )
    return lines


def _fold_lines(folds: list[dict[str, Any]]) -> list[str]:
    lines = [
        "Leave-one-season-out intervals measure the sensitivity of pooled estimates to "
        "individual season cohorts; they are diagnostic and do not replace the four-season results.",
        "",
        "Each fold drops one season and resamples gameweeks inside each remaining season. "
        "The point estimate is the mean of the concatenated weeks. B=1000 and the seed is 0. "
        "A remaining season under 20 weeks is omitted. A conditional disagreement fold uses "
        "a minimum of 5 weeks instead, and that waiver is unchanged. A fold with fewer than "
        "two complete seasons is not identified. Concordance is not re-aggregated: the weekly "
        "difference is not a stored column.",
        "",
        "The published four-season intervals remain the official evaluation benchmarks and are unchanged.",
        "A sign flip or shift in interval bounds when holding out a season is evidence of cohort "
        "heterogeneity, not a justification to drop any season or declare a winner.",
        "",
        "The published unconditional three-week interval is −1.69 [−2.92, −0.52]. "
        "Its exclusion of zero does not survive when 2022-23 is held out. "
        "The claim between the engine and expected points is inconclusive.",
        "",
        "No new closed-season contrast is in this file. The certified likelihood was not re-aggregated.",
        "",
    ]
    current = ""
    for fold in folds:
        if fold["key"] != current:
            if current:
                lines.append("")
            current = str(fold["key"])
            tag = " Strawman." if fold["strawman"] else ""
            if fold["conditional"]:
                tag = " Conditional: the 20-week floor does not apply."
            lines.append(f"### {fold['label']}")
            lines.append("")
            lines.append(
                f"Published four-season mean {float(fold['published_mean']):+.4f}. "
                f"That interval was not recomputed.{tag}"
            )
            lines.append("")
            lines.append("| held out | weeks | estimate | reading | versus published mean |")
            lines.append("|---|---:|---|---|---|")
        weeks = sum(int(n) for n in (fold.get("n_gws") or {}).values())
        versus = "sign differs" if fold["sign_differs"] else "same sign"
        if fold.get("undefined"):
            versus = "not identified"
        omitted = fold.get("incomplete_seasons") or {}
        if omitted:
            versus += "; omitted " + ", ".join(f"{name} ({count})" for name, count in omitted.items())
        lines.append(
            f"| {fold['holdout']} | {weeks} | {_interval_text(fold)} | {fold['reading']} | {versus} |"
        )
    lines.append("")
    lines.append(
        "`greedy_xp_minus_hold_xp` is reported as the previously retired strawman baseline "
        "(a static squad that never transfers) and does not measure transfer skill."
    )
    lines.append("")
    return lines


def _find(folds: list[dict[str, Any]], key: str, holdout: str) -> dict[str, Any]:
    for fold in folds:
        if fold["key"] == key and fold["holdout"] == holdout:
            return fold
    raise RuntimeError(f"missing fold {key} held out {holdout}")


def _signed(value: float) -> str:
    text = f"{abs(float(value)):.4f}"
    if float(value) < 0.0:
        return f"−{text}"
    return f"+{text}"


def assert_no_promotion(folds: list[dict[str, Any]]) -> None:
    """A fold above zero for score_xp minus expected points is not a promotion."""
    for fold in folds:
        if fold["key"] not in XP_VERSUS_EXP:
            continue
        lo = fold.get("lo")
        if lo is not None and float(lo) > 0.0:
            raise RuntimeError("a fold stays above zero for score_xp minus expected points")


def assert_shuffle_separation(folds: list[dict[str, Any]]) -> None:
    for fold in folds:
        if fold["key"] not in SHUFFLE_KEYS:
            continue
        lo = fold.get("lo")
        if lo is None or float(lo) <= 0.0:
            raise RuntimeError("an informed-versus-shuffle fold does not stay above zero")


def _diagnostic(folds: list[dict[str, Any]]) -> list[str]:
    assert_no_promotion(folds)
    assert_shuffle_separation(folds)
    three = _find(folds, "hierarchy:r3_xp_minus_r3_exp", "2022-23")
    greedy = _find(folds, "placebo:greedy_xp_minus_greedy_exp", "2022-23")
    if three["reading"] != "the interval covers zero" or greedy["reading"] != "the interval covers zero":
        raise RuntimeError("the 2022-23 hold-out is not the covering-zero sensitivity already read")
    return [
        (
            "Holding out 2022–23 shifts the three-week common-state interval to "
            f"{_signed(three['mean'])} [{_signed(three['lo'])}, {_signed(three['hi'])}], "
            "which covers zero; the exclusion of zero in the published four-season estimate "
            "(−1.6947 [−2.9162, −0.5189]) is sensitive to that single cohort."
        ),
        (
            "Without 2022–23, the greedy contrast point estimate is "
            f"{_signed(greedy['mean'])} points per gameweek, but the interval "
            f"[{_signed(greedy['lo'])}, {_signed(greedy['hi'])}] covers zero; setting that "
            "season aside provides no evidence that `score_xp` outperforms expected points."
        ),
        "No leave-one-out fold justifies dropping 2022–23 from the evaluation or promoting "
        "`score_xp` over expected points.",
        "The published four-season pooled intervals remain the official benchmarks and are unchanged.",
        "Both informed scores remain strictly separated from the within-week shuffle across "
        "all leave-one-season-out folds.",
        "",
    ]


def _close(lines: list[str], folds: list[dict[str, Any]]) -> list[str]:
    lines.extend(_diagnostic(folds))
    lines.extend(
        [
            "The pattern already reported on the disagreement weeks, cheaper buys, expected minutes "
            "about 77 against about 83, higher attack strength, and a naive buy that is often Haaland "
            "or Salah, is a hypothesis for the live weeks. It is not fit on the closed seasons.",
            "",
            "A gain in the bulk of the list, such as who will not play, rather than among the best "
            "players, is a hypothesis. It is not a finding.",
            "",
            "For live squad decisions from gameweek 6 onward, the primary score is pre-registered "
            "as `ep_next`, with `score_xp` logged alongside.",
            "Promotion of `score_xp` over `ep_next` requires the paired live interval across at least "
            "20 pre-deadline gameweeks to stay strictly above zero; an interval covering zero is "
            "undetermined and testing continues through gameweek 38.",
            f"The published historical score stays `{SCORE_COLUMN}`. From gameweek 6 the live "
            "squad is chosen by captured `ep_next`, and `score_xp` is logged in shadow and does not choose.",
            "",
            "In accordance with protocol, if a paired transfer contrast interval covers zero, "
            "the result is inconclusive and no winner is declared.",
            "No winner is declared. `score_xp` is unchanged.",
            "",
            "Gemini kept the estimator and, after these folds, the reading that no fold "
            "promotes `score_xp` "
            "([leave-one-season-out](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
        ]
    )
    return lines


def _fold_row(fold: dict[str, Any]) -> dict[str, Any]:
    return {
        "contrast": fold["key"],
        "holdout": fold["holdout"],
        "conditional": fold["conditional"],
        "strawman": fold["strawman"],
        "undefined": fold["undefined"],
        "mean": fold["mean"],
        "lo": fold["lo"],
        "hi": fold["hi"],
        "n_gws": json.dumps(fold.get("n_gws") or {}),
        "incomplete_seasons": json.dumps(fold.get("incomplete_seasons") or {}),
        "published_mean": fold["published_mean"],
        "sign_differs": fold["sign_differs"],
        "reading": fold["reading"],
    }


def run() -> None:
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report

    protocol = load_protocol()
    loso = protocol["loso"]
    live = protocol["live_primary"]
    refuse_replacement(bool(loso.get("replaces_published")))
    if loso.get("winner") is not None or live.get("winner") is not None:
        raise RuntimeError("a winner is already named")
    if list(loso.get("contrasts") or []) != list(contrast_keys()):
        raise RuntimeError("the contrast list does not match the lock")
    if live.get("primary") != LIVE_PRIMARY or live.get("shadow") != LIVE_SHADOW:
        raise RuntimeError("the live primary is not the locked pair")
    if live.get("wired") is not LIVE_PRIMARY_WIRED:
        raise RuntimeError("the live wiring flag does not match the lock")
    if int(loso["bootstrap"]) != LOSO_BOOTSTRAP or int(loso["seed"]) != LOSO_SEED:
        raise RuntimeError("the draw is not the locked draw")
    placebo = (REPORTS / "decision_placebo.md").read_text(encoding="utf-8")
    if "0 of 26505" not in placebo:
        raise RuntimeError("the published odds-join count is not the one already reported")
    opening = pd.read_csv(PROCESSED / "initial_squad_weeks.csv")
    forensic = pd.read_csv(PROCESSED / "forensic_2022_23.csv")
    paired = _pair_2022(opening, forensic)
    n_boot = int(loso["bootstrap"])
    seed = int(loso["seed"])
    folds: list[dict[str, Any]] = []
    for spec in CONTRASTS:
        rows = _load_column(spec)
        _check_published_mean(rows, spec)
        folds.extend(_folds_for(spec, rows, n_boot=n_boot, seed=seed))
    lines = _week_lines(paired)
    lines.extend(_fold_lines(folds))
    _close(lines, folds)
    text = "\n".join(lines)
    for sentence in REQUIRED_SENTENCES:
        if sentence not in text:
            raise RuntimeError("a required sentence is missing")
    for banned in FORBIDDEN:
        if banned in text:
            raise RuntimeError(f"the report contains a banned claim: {banned}")
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = "score_xp_minus_score_exp_points"
    audit = run_asof_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_fold_row(fold) for fold in folds]).to_csv(PROCESSED / "loso_folds.csv", index=False)
    write_gated_report(
        REPORTS / "loso.md",
        audit,
        {
            "seasons": list(protocol["closed_seasons"]),
            "min_gws": int(protocol["min_gws_per_season"]),
            "comparisons": {key: certified["comparisons"][key]},
        },
        lines,
    )


if __name__ == "__main__":
    run()
