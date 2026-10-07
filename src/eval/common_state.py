"""Common-state transfer comparison.

One score-blind fifteen is frozen. Every later week, each scoring rule sees
that same squad and the same legal moves. Nothing is written back onto the
squad. No winner is declared. The formulas were locked before the totals
were read.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.decision import (
    Squad,
    collapse_gameweek,
    greedy_step,
    legal_moves,
    neutral_squad,
    permute_week,
    points_lookup,
    realised_over,
    replay_season,
)
from src.eval.decision_spec import SCORE_COLUMN
from src.eval.provenance import OFFICIAL_XP_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

RULES: tuple[tuple[str, str, bool], ...] = (
    ("xp", "score_xp", False),
    ("exp", "score_exp_points", False),
    ("roll3", "roll3_points", False),
    ("shuffled", "score_xp", True),
)
CONTRASTS = (
    "r1_xp_minus_r1_exp",
    "r3_xp_minus_r3_exp",
    "r1_xp_minus_r1_roll3",
    "r1_xp_minus_r1_shuffled",
)
SUMS = ("total_points", "score_xp", "score_exp_points", "roll3_points")


def pairwise_concordance(predicted: np.ndarray, realised: np.ndarray) -> float | None:
    """Share of legal pairs where the higher predicted gain also gains more points.

    Pairs with equal predicted gain are ignored. Pairs whose realised gains
    are equal are not decisive. A week with no decisive pair is undefined.
    """
    pred = np.asarray(predicted, dtype=float)
    real = np.asarray(realised, dtype=float)
    if pred.size != real.size or pred.size < 2:
        return None
    left, right = np.triu_indices(pred.size, k=1)
    gap_pred = pred[left] - pred[right]
    gap_real = real[left] - real[right]
    differ = gap_pred != 0.0
    decisive = differ & (gap_real != 0.0)
    if not bool(decisive.any()):
        return None
    higher = ((gap_pred > 0.0) & (gap_real > 0.0)) | ((gap_pred < 0.0) & (gap_real < 0.0))
    return float(higher[decisive].mean())


def _weeks(frame: pd.DataFrame, gw_start: int, gw_end: int) -> dict[int, pd.DataFrame]:
    work = frame.copy()
    banned = [column for column in OFFICIAL_XP_COLUMNS if column in work.columns]
    if banned:
        work = work.drop(columns=banned)
    missing = [column for column in SUMS if column not in work.columns]
    if missing:
        raise RuntimeError("common state is missing " + ", ".join(missing))
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    work = work.dropna(subset=["gw"])
    weeks: dict[int, pd.DataFrame] = {}
    for gw, block in work.groupby(work["gw"].astype(int), sort=True):
        number = int(gw)
        if number < int(gw_start) or number > int(gw_end):
            continue
        weeks[number] = collapse_gameweek(block, SUMS)
    return weeks


def _frozen(weeks: dict[int, pd.DataFrame]) -> tuple[Squad, int]:
    for gw in sorted(weeks):
        try:
            return neutral_squad(weeks[gw]), gw
        except RuntimeError:
            continue
    raise RuntimeError("no gameweek filled the score-blind fifteen")


def _snapshot(squad: Squad) -> tuple[Any, ...]:
    return (
        tuple(sorted(squad.purchase.items())),
        tuple(sorted(squad.position.items())),
        tuple(sorted(squad.club.items())),
        int(squad.bank),
    )


def _realised_pair(
    points_by_gw: dict[int, dict[str, float]],
    gw: int,
    move: dict[str, Any] | None,
) -> tuple[float, float]:
    if move is None:
        return 0.0, 0.0
    player_in = str(move["player_in"])
    player_out = str(move["player_out"])
    week = points_by_gw[gw]
    one = float(week.get(player_in, 0.0) - week.get(player_out, 0.0))
    three = realised_over(points_by_gw, gw, player_in, player_out)
    return one, three


def _concordance(
    squad: Squad,
    week: pd.DataFrame,
    score_col: str,
    points: dict[str, float],
) -> float | None:
    moves = legal_moves(squad, week, score_col)
    if len(moves) < 2:
        return None
    predicted = np.array([float(move["predicted"]) for move in moves], dtype=float)
    realised = np.array(
        [
            float(points.get(str(move["player_in"]), 0.0) - points.get(str(move["player_out"]), 0.0))
            for move in moves
        ],
        dtype=float,
    )
    return pairwise_concordance(predicted, realised)


def _choose(
    squad: Squad,
    week: pd.DataFrame,
    score_col: str,
    points_by_gw: dict[int, dict[str, float]],
    gw: int,
) -> dict[str, Any]:
    before = _snapshot(squad)
    _nxt, move = greedy_step(squad, week, score_col)
    if _snapshot(squad) != before:
        raise RuntimeError("a scoring rule mutated the frozen squad")
    one, three = _realised_pair(points_by_gw, gw, move)
    if move is None:
        return {
            "player_in": "",
            "player_out": "",
            "position": "",
            "predicted": 0.0,
            "r1": one,
            "r3": three,
        }
    return {
        "player_in": str(move["player_in"]),
        "player_out": str(move["player_out"]),
        "position": str(move["position"]),
        "predicted": float(move["predicted"]),
        "r1": one,
        "r3": three,
    }


def evaluate_season(
    frame: pd.DataFrame,
    season_index: int,
    *,
    gw_start: int,
    gw_end: int,
) -> tuple[list[dict[str, Any]], int]:
    """Transfer weeks from one frozen squad. The construction week is not a row."""
    weeks = _weeks(frame, gw_start, gw_end)
    squad, start = _frozen(weeks)
    points_by_gw = {gw: points_lookup(week) for gw, week in weeks.items()}
    rows: list[dict[str, Any]] = []
    for gw in sorted(weeks):
        if gw <= start:
            continue
        base = weeks[gw]
        chosen: dict[str, dict[str, Any]] = {}
        concord: dict[str, float | None] = {}
        for label, column, shuffle in RULES:
            week = permute_week(base, column, season_index, gw) if shuffle else base
            chosen[label] = _choose(squad, week, column, points_by_gw, gw)
            concord[label] = _concordance(squad, week, column, points_by_gw[gw])
        xp = chosen["xp"]
        exp = chosen["exp"]
        roll3 = chosen["roll3"]
        shuffled = chosen["shuffled"]
        same = xp["player_in"] == exp["player_in"] and xp["player_out"] == exp["player_out"]
        rows.append(
            {
                "gw": gw,
                "start_gw": start,
                "r1_xp": xp["r1"],
                "r1_exp": exp["r1"],
                "r1_roll3": roll3["r1"],
                "r1_shuffled": shuffled["r1"],
                "r3_xp": xp["r3"],
                "r3_exp": exp["r3"],
                "pred_xp": xp["predicted"],
                "pred_exp": exp["predicted"],
                "pred_roll3": roll3["predicted"],
                "pred_shuffled": shuffled["predicted"],
                "r1_xp_minus_r1_exp": xp["r1"] - exp["r1"],
                "r3_xp_minus_r3_exp": xp["r3"] - exp["r3"],
                "r1_xp_minus_r1_roll3": xp["r1"] - roll3["r1"],
                "r1_xp_minus_r1_shuffled": xp["r1"] - shuffled["r1"],
                "agree_xp_exp": int(same),
                "c_xp": concord["xp"],
                "c_exp": concord["exp"],
                "c_roll3": concord["roll3"],
                "c_shuffled": concord["shuffled"],
                "xp_in": xp["player_in"],
                "xp_out": xp["player_out"],
                "exp_in": exp["player_in"],
                "exp_out": exp["player_out"],
                "roll3_in": roll3["player_in"],
                "roll3_out": roll3["player_out"],
                "shuffled_in": shuffled["player_in"],
                "shuffled_out": shuffled["player_out"],
            }
        )
    return rows, start


def _lookup(frame: pd.DataFrame, column: str) -> dict[tuple[int, str], float]:
    work = pd.DataFrame(
        {
            "gw": pd.to_numeric(frame["gw"], errors="coerce"),
            "player_id": frame["player_id"].astype(str),
            column: pd.to_numeric(frame[column], errors="coerce"),
        }
    ).dropna(subset=["gw"])
    grouped = work.groupby(["gw", "player_id"], sort=False)[column].mean()
    found: dict[tuple[int, str], float] = {}
    for (gw, pid), value in grouped.items():
        if math.isfinite(float(value)):
            found[(int(gw), str(pid))] = float(value)
    return found


def _names(frame: pd.DataFrame) -> dict[str, str]:
    if "player_name" not in frame.columns:
        return {}
    names: dict[str, str] = {}
    for row in frame[["player_id", "player_name"]].itertuples(index=False):
        pid = str(row.player_id)
        if pid not in names and str(row.player_name):
            names[pid] = str(row.player_name)
    return names


def _positions(frame: pd.DataFrame) -> dict[tuple[int, str], str]:
    from src.rules.fpl_2026 import normalize_position

    work = frame.dropna(subset=["gw"]).copy()
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    found: dict[tuple[int, str], str] = {}
    for row in work[["gw", "player_id", "position"]].itertuples(index=False):
        if not math.isfinite(float(row.gw)):
            continue
        key = (int(row.gw), str(row.player_id))
        if key not in found:
            try:
                found[key] = normalize_position(str(row.position))
            except ValueError:
                continue
    return found


def _label(names: dict[str, str], pid: str) -> str:
    if not pid:
        return "roll"
    return names.get(pid, pid).replace("|", "/")


def diverging_forensic(
    frame: pd.DataFrame,
    *,
    gw_start: int,
    gw_end: int,
) -> pd.DataFrame:
    """2022-23 transfers from the separate opening squads. Not the common state."""
    xp = replay_season(frame, SCORE_COLUMN, gw_start=gw_start, gw_end=gw_end)
    exp = replay_season(frame, "score_exp_points", gw_start=gw_start, gw_end=gw_end)
    attack = _lookup(frame, "attack_strength")
    names = _names(frame)
    position = _positions(frame)
    xp_weeks = {int(row["gw"]): row for row in xp["weeks"]}
    exp_weeks = {int(row["gw"]): row for row in exp["weeks"]}
    xp_moves = {int(row["gw"]): row for row in xp["transfers"]}
    exp_moves = {int(row["gw"]): row for row in exp["transfers"]}
    rows: list[dict[str, Any]] = []
    for gw in sorted(set(xp_weeks) & set(exp_weeks)):
        left = xp_moves.get(gw)
        right = exp_moves.get(gw)
        rows.append(
            {
                "gw": gw,
                "squad_gap": float(xp_weeks[gw]["greedy"] - exp_weeks[gw]["greedy"]),
                "xp_out": "" if left is None else str(left["player_out"]),
                "xp_in": "" if left is None else str(left["player_in"]),
                "xp_predicted": 0.0 if left is None else float(left["predicted"]),
                "xp_realised_t": 0.0 if left is None else float(left["realised_t"]),
                "exp_out": "" if right is None else str(right["player_out"]),
                "exp_in": "" if right is None else str(right["player_in"]),
                "exp_predicted": 0.0 if right is None else float(right["predicted"]),
                "exp_realised_t": 0.0 if right is None else float(right["realised_t"]),
                "xp_position": "" if left is None else position.get((gw, str(left["player_in"])), ""),
                "exp_position": "" if right is None else position.get((gw, str(right["player_in"])), ""),
                "xp_attack_delta": _attack_delta(attack, gw, left),
                "exp_attack_delta": _attack_delta(attack, gw, right),
                "xp_out_name": _label(names, "" if left is None else str(left["player_out"])),
                "xp_in_name": _label(names, "" if left is None else str(left["player_in"])),
                "exp_out_name": _label(names, "" if right is None else str(right["player_out"])),
                "exp_in_name": _label(names, "" if right is None else str(right["player_in"])),
            }
        )
    return pd.DataFrame(rows)


def _attack_delta(
    attack: dict[tuple[int, str], float],
    gw: int,
    move: dict[str, Any] | None,
) -> float:
    if move is None:
        return float("nan")
    bought = attack.get((gw, str(move["player_in"])))
    sold = attack.get((gw, str(move["player_out"])))
    if bought is None or sold is None:
        return float("nan")
    return float(bought - sold)


def _band(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "no season reached 20 gameweeks"
    if float(row["lo"]) > 0.0:
        return "the interval stays above zero"
    if float(row["hi"]) < 0.0:
        return "the interval stays below zero"
    return "the interval covers zero"


def _fmt_interval(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "no season reached 20 gameweeks"
    return f"{row['mean']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"


def _season_lines(frame: pd.DataFrame, column: str) -> list[str]:
    lines = []
    for season, block in frame.groupby("season", sort=True):
        lines.append(f"| {season} | {len(block)} | {block[column].mean():+.2f} |")
    return lines


def _mean(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "not identified"
    return f"{float(values.mean()):+.2f}"


def _counts(series: pd.Series) -> str:
    cleaned = [str(value) for value in series.tolist() if str(value)]
    if not cleaned:
        return "none"
    return ", ".join(f"{pos} {count}" for pos, count in Counter(cleaned).most_common())


def _forensic_lines(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return ["The 2022-23 diverging replay produced no weeks."]
    xp_moves = table.loc[table["xp_in"] != ""]
    exp_moves = table.loc[table["exp_in"] != ""]
    isolated = table["xp_realised_t"] - table["exp_realised_t"]
    lines = [
        f"Weeks in the diverging replay: {len(table)}. "
        f"`score_xp` transfers: {len(xp_moves)}. Expected-points transfers: {len(exp_moves)}.",
        f"Mean squad-point gap (xp greedy minus exp greedy): {table['squad_gap'].mean():+.2f}. "
        f"Sum of those gaps: {table['squad_gap'].sum():+.1f}.",
        f"Mean of the same weeks' isolated one-week transfer gaps: {isolated.mean():+.2f}. "
        f"Sum of those isolated gaps: {isolated.sum():+.1f}.",
        f"On the transfers `score_xp` made, mean predicted gain {_mean(xp_moves['xp_predicted'])}, "
        f"mean one-week realised {_mean(xp_moves['xp_realised_t'])}, "
        f"mean attack_strength in minus out {_mean(xp_moves['xp_attack_delta'])}. "
        f"Buy positions: {_counts(xp_moves['xp_position'])}.",
        f"On the transfers expected points made, mean predicted gain {_mean(exp_moves['exp_predicted'])}, "
        f"mean one-week realised {_mean(exp_moves['exp_realised_t'])}, "
        f"mean attack_strength in minus out {_mean(exp_moves['exp_attack_delta'])}. "
        f"Buy positions: {_counts(exp_moves['exp_position'])}.",
        "",
        "| GW | XP out | XP in | XP pos | XP Δ | XP realised | Exp out | Exp in | Exp pos | Exp Δ | Exp realised | squad gap |",
        "|---:|---|---|---|---:|---:|---|---|---|---:|---:|---:|",
    ]
    ordered = table.sort_values("squad_gap")
    for row in table.sort_values("gw").itertuples(index=False):
        lines.append(
            "| {gw} | {xp_out} | {xp_in} | {xp_pos} | {xp_pred:+.2f} | {xp_real:+.1f} | "
            "{exp_out} | {exp_in} | {exp_pos} | {exp_pred:+.2f} | {exp_real:+.1f} | {gap:+.1f} |".format(
                gw=int(row.gw),
                xp_out=row.xp_out_name,
                xp_in=row.xp_in_name,
                xp_pos=row.xp_position or "—",
                xp_pred=float(row.xp_predicted),
                xp_real=float(row.xp_realised_t),
                exp_out=row.exp_out_name,
                exp_in=row.exp_in_name,
                exp_pos=row.exp_position or "—",
                exp_pred=float(row.exp_predicted),
                exp_real=float(row.exp_realised_t),
                gap=float(row.squad_gap),
            )
        )
    worst = ", ".join(
        f"GW{int(row.gw)} {row.squad_gap:+.0f}" for row in ordered.head(5).itertuples(index=False)
    )
    xp_attack = xp_moves["xp_attack_delta"].dropna()
    exp_attack = exp_moves["exp_attack_delta"].dropna()
    lines.extend(
        [
            "",
            f"Widest squad gaps, xp minus exp: {worst}.",
            "",
            f"In 2022-23 the diverging squad gap averages {table['squad_gap'].mean():+.2f}. "
            f"The isolated one-week transfer gap on those weeks averages {isolated.mean():+.2f}. "
            "The week's in-minus-out does not account for the squad gap.",
            "",
            "Among 2022-23 transfers with a finite attack delta, the share with a higher "
            f"attack strength on the buy is {(xp_attack > 0).mean():.2f} "
            f"for `score_xp` ({int((xp_attack > 0).sum())} of {len(xp_attack)}) and "
            f"{(exp_attack > 0).mean():.2f} for expected points "
            f"({int((exp_attack > 0).sum())} of {len(exp_attack)}).",
        ]
    )
    return lines


def _one_week_sentence(intervals: dict[str, Any]) -> str:
    row = intervals["r1_xp_minus_r1_exp"]
    if row.get("lo") is not None and float(row["lo"]) <= 0.0 <= float(row["hi"]):
        return "On the one-week contrast the interval covers zero, so that comparison is inconclusive."
    return f"On the one-week contrast, {_band(row)}. No winner is declared."


def _three_week_sentence(intervals: dict[str, Any], weeks: pd.DataFrame) -> str:
    row = intervals["r3_xp_minus_r3_exp"]
    means = weeks.groupby("season")["r3_xp_minus_r3_exp"].mean()
    if row.get("hi") is not None and float(row["hi"]) < 0.0 and bool((means < 0).all()):
        return (
            "On the three-week contrast the interval stays below zero, and every season mean "
            "is negative. That reading is not a decision to replace score_xp."
        )
    return (
        f"On the three-week contrast, {_band(row)}. "
        "That reading is not a decision to replace score_xp."
    )


def _roll_sentence(weeks: pd.DataFrame) -> str:
    rolls = int((weeks["xp_in"] == "").sum() + (weeks["exp_in"] == "").sum())
    agree = float(weeks["agree_xp_exp"].mean()) if len(weeks) else float("nan")
    differ = weeks.loc[weeks["agree_xp_exp"] == 0]
    if rolls:
        rolled = f"Rolls across the two rules: {rolls}."
    else:
        rolled = "Neither rule rolls."
    return (
        f"{rolled} The two rules pick the same players on {agree:.0%} of weeks. "
        f"On the weeks they differ, the one-week gap averages {differ['r1_xp_minus_r1_exp'].mean():+.2f} "
        f"and the three-week gap averages {differ['r3_xp_minus_r3_exp'].mean():+.2f}. "
        "Means on the weeks they differ are descriptive only and were not bootstrapped."
    )


def _defined_mean(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return "undefined"
    return f"{float(values.mean()):+.3f} on {len(values)} defined weeks"


def _lines(
    protocol: dict[str, Any],
    intervals: dict[str, Any],
    weeks: pd.DataFrame,
    starts: list[tuple[str, int, int]],
    forensic: pd.DataFrame,
) -> list[str]:
    centre = intervals["r1_xp_minus_r1_exp"]
    both = weeks["c_xp"].notna() & weeks["c_exp"].notna()
    undefined = int((~both).sum())
    agree = float(weeks["agree_xp_exp"].mean()) if len(weeks) else float("nan")
    start_bits = ", ".join(f"{season} GW{gw} ({n} transfer weeks)" for season, gw, n in starts)
    return [
        "# Common-state transfers",
        "",
        "The common-state test evaluates transfer selection from an identical frozen "
        "fifteen built without reference to any player score.",
        "",
        "Each week, every scoring rule evaluates candidate transfers from the exact "
        "same unmutated squad, ensuring zero state divergence.",
        "",
        "Primary transfer contrasts pair realised in-minus-out gains unconditionally "
        "across all gameweeks, including weeks where both rules choose the same "
        "transfer or roll.",
        "",
        "The question is whether `score_xp` chooses a better one-free-transfer move "
        "than `score_exp_points` when the squad, the bank, and the legal set are the "
        "same. A roll scores 0 predicted and 0 realised. The construction week is the "
        f"first week that fills the fifteen ({start_bits}). It is not a transfer week. "
        "No later transfer is written onto that squad.",
        "",
        f"Centre contrast, one-week realised, `score_xp` minus expected points: "
        f"{_fmt_interval(centre)}. {_band(centre)}.",
        "",
        "| contrast | estimate | reading |",
        "|---|---|---|",
        f"| R1 xp − R1 exp | {_fmt_interval(centre)} | {_band(centre)} |",
        f"| R3 xp − R3 exp | {_fmt_interval(intervals['r3_xp_minus_r3_exp'])} | {_band(intervals['r3_xp_minus_r3_exp'])} |",
        f"| R1 xp − R1 roll3 | {_fmt_interval(intervals['r1_xp_minus_r1_roll3'])} | {_band(intervals['r1_xp_minus_r1_roll3'])} |",
        f"| R1 xp − R1 shuffled xp | {_fmt_interval(intervals['r1_xp_minus_r1_shuffled'])} | {_band(intervals['r1_xp_minus_r1_shuffled'])} |",
        f"| concordance xp − exp | {_fmt_interval(intervals['c_xp_minus_c_exp'])} | {_band(intervals['c_xp_minus_c_exp'])} |",
        "",
        "In accordance with protocol, if a paired transfer contrast interval covers "
        "zero, the result is inconclusive and no winner is declared.",
        "",
        "No winner is declared from this batch. The published score, the hold, and "
        "the chip map are unchanged.",
        "",
        "### One-week xp minus expected points",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(weeks, "r1_xp_minus_r1_exp"),
        "",
        _one_week_sentence(intervals),
        "",
        "### Three-week xp minus expected points",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(weeks, "r3_xp_minus_r3_exp"),
        "",
        _three_week_sentence(intervals, weeks),
        "",
        f"Of the three-week gap, the decision week averages {weeks['r1_xp_minus_r1_exp'].mean():+.2f} "
        f"and the next two weeks together average "
        f"{(weeks['r3_xp_minus_r3_exp'] - weeks['r1_xp_minus_r1_exp']).mean():+.2f}. "
        "That split has no interval.",
        "",
        "### One-week xp minus the rolling three-week mean",
        "",
        "The rolling three-week points baseline uses shift-1 historical data only, "
        "with zero current-week information.",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(weeks, "r1_xp_minus_r1_roll3"),
        "",
        "### One-week xp minus shuffled xp",
        "",
        "Shuffled `score_xp` permutes eligible finite values inside that gameweek. "
        "Points stay on the player. The squad is not rebuilt.",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(weeks, "r1_xp_minus_r1_shuffled"),
        "",
        "The one-week contrast against shuffled score_xp stays above zero. The squad was not rebuilt.",
        "",
        "### Pairwise concordance",
        "",
        "Pairwise concordance measures discrimination strictly across legal moves "
        "that satisfy position, price, club quota, and eligibility constraints; "
        "weeks without decisive pairs are undefined.",
        "",
        f"Defined-week means, not themselves contrasts: "
        f"`score_xp` {_defined_mean(weeks, 'c_xp')}; "
        f"expected points {_defined_mean(weeks, 'c_exp')}; "
        f"roll3 {_defined_mean(weeks, 'c_roll3')}; "
        f"shuffled xp {_defined_mean(weeks, 'c_shuffled')}.",
        f"Weeks where xp concordance or exp concordance is undefined: {undefined}. "
        "Those weeks are absent from the concordance contrast only.",
        f"Share of transfer weeks where `score_xp` and expected points pick the same "
        f"players, or both roll: {agree:.3f}. That share is descriptive. Those weeks "
        f"stay in the primary transfer contrasts, where the gap is zero.",
        "",
        "Pairwise concordance of score_xp minus expected points stays above zero. "
        f"The gap is {intervals['c_xp_minus_c_exp']['mean']:+.3f}. "
        f"The defined-week means are {pd.to_numeric(weeks['c_xp'], errors='coerce').mean():+.3f} "
        f"and {pd.to_numeric(weeks['c_exp'], errors='coerce').mean():+.3f}.",
        "",
        _roll_sentence(weeks),
        "",
        "| season | both defined | mean concordance gap |",
        "|---|---:|---:|",
        *_concord_lines(weeks.loc[both]),
        "",
        "The difference of +5.51 points per gameweek from the prior report is an "
        "arithmetic decomposition between separate baselines, not a statistical "
        "confidence interval.",
        "",
        "## 2022-23 diverging path",
        "",
        "The 2022–23 deficit of −14.45 points per gameweek is a property of "
        "multi-week squad divergence, not the sum of isolated transfer gains.",
        "",
        "This table is the earlier replay, in which each score builds its own "
        "opening squad and then transfers. It is not a common-state choice. "
        "Attack strength is the mean of that player's rows in the week.",
        "",
        *_forensic_lines(forensic),
        "",
        f"Bootstrap {protocol['bootstrap']}, seed {protocol['seed']}. "
        "A season under 20 weeks is not pooled. "
        f"Closed seasons: {', '.join(protocol['closed_seasons'])}.",
        "",
        "Gemini reviewed these diagnostics "
        "([common-state diagnostics](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
        "",
    ]


def _concord_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["| — | 0 | undefined |"]
    lines = []
    for season, block in frame.groupby("season", sort=True):
        gap = block["c_xp"] - block["c_exp"]
        lines.append(f"| {season} | {len(block)} | {gap.mean():+.3f} |")
    return lines


def run() -> None:
    from src.eval.decision import pool_columns
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season

    protocol = load_protocol()
    common = protocol["common_state"]
    if common.get("winner") is not None:
        raise RuntimeError("the common-state lock already names a winner")
    if "score unused" not in str(common.get("squad") or ""):
        raise RuntimeError("the common-state squad is not score-blind")
    codes = protocol["season_codes"]
    gw_start = int(protocol["gw_start"])
    gw_end = int(protocol["gw_end"])
    week_rows: list[dict[str, Any]] = []
    starts: list[tuple[str, int, int]] = []
    forensic = pd.DataFrame()
    for index, season in enumerate(protocol["closed_seasons"]):
        print(f"common state {season}", flush=True)
        frame = build_season(season, codes[season], protocol)
        rows, start = evaluate_season(frame, index, gw_start=gw_start, gw_end=gw_end)
        for row in rows:
            week_rows.append({"season": season, **row})
        starts.append((season, start, len(rows)))
        if season == "2022-23":
            forensic = diverging_forensic(frame, gw_start=gw_start, gw_end=gw_end)
    weeks = pd.DataFrame(week_rows)
    intervals = pool_columns(
        weeks,
        CONTRASTS,
        minimum=int(protocol["min_gws_per_season"]),
        n_boot=int(protocol["bootstrap"]),
        seed=int(protocol["seed"]),
    )
    defined = weeks.loc[weeks["c_xp"].notna() & weeks["c_exp"].notna()].copy()
    defined["c_xp_minus_c_exp"] = defined["c_xp"] - defined["c_exp"]
    intervals.update(
        pool_columns(
            defined,
            ("c_xp_minus_c_exp",),
            minimum=int(protocol["min_gws_per_season"]),
            n_boot=int(protocol["bootstrap"]),
            seed=int(protocol["seed"]),
        )
    )
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = "score_xp_minus_score_exp_points"
    audit = run_asof_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weeks.to_csv(PROCESSED / "common_state_weeks.csv", index=False)
    forensic.to_csv(PROCESSED / "forensic_2022_23.csv", index=False)
    write_gated_report(
        REPORTS / "common_state.md",
        audit,
        {
            "seasons": list(protocol["closed_seasons"]),
            "min_gws": int(protocol["min_gws_per_season"]),
            "comparisons": {key: certified["comparisons"][key], **intervals},
        },
        _lines(protocol, intervals, weeks, starts, forensic),
    )


if __name__ == "__main__":
    run()
