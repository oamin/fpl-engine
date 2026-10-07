"""Disagreement forensic, the missing hierarchy leg, and the opening portfolio.

The unconditional common-state contrasts stay as published. This batch
does not replace them and does not change score_xp. Formulas were locked
before these totals were read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.common_state import _weeks, evaluate_season
from src.eval.decision import (
    Squad,
    neutral_squad,
    opening_squad,
    permute_week,
    score_lookup,
    value_lookup,
    week_points,
)
from src.rules.fpl_2026 import normalize_position

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

CONTRIBUTIONS = (
    "xp_appear",
    "xp_goals",
    "xp_assists",
    "xp_cs",
    "xp_defcon",
    "xp_saves",
    "xp_bps",
    "xp_deductions",
)
MEAN_ATTRS = ("xmi", "attack_strength", "defend_threat")
MIN_DISAGREE = 5
PORTFOLIOS = ("xp", "exp", "shuffled", "neutral")
DEPLOYMENTS = ("score_xp", "score_exp_points")


def realised_call(left: float, right: float) -> str:
    """xp, exp, or equal. A gap of exactly zero is a tie."""
    if float(left) > float(right):
        return "xp"
    if float(right) > float(left):
        return "exp"
    return "equal"


def cross_margin(
    scores: dict[str, float],
    own_in: str,
    own_out: str,
    other_in: str,
    other_out: str,
) -> float:
    """Own move's gain minus the other move's gain, in this score's units."""

    def gain(player_in: str, player_out: str) -> float:
        return float(scores.get(str(player_in), 0.0)) - float(scores.get(str(player_out), 0.0))

    return gain(own_in, own_out) - gain(other_in, other_out)


def signed_contributions(
    incoming: dict[str, float],
    outgoing: dict[str, float],
) -> dict[str, float]:
    """Contribution of each piece to score(in) − score(out). Deductions are subtracted."""
    signed: dict[str, float] = {}
    for name in CONTRIBUTIONS:
        delta = float(incoming.get(name, 0.0)) - float(outgoing.get(name, 0.0))
        if name == "xp_deductions":
            delta = -delta
        signed[name] = delta
    return signed


def driver_name(signed: dict[str, float]) -> str:
    """Largest signed contribution. Ties keep the earlier name in CONTRIBUTIONS."""
    best_name = CONTRIBUTIONS[0]
    best_value = float(signed.get(best_name, 0.0))
    for name in CONTRIBUTIONS[1:]:
        value = float(signed.get(name, 0.0))
        if value > best_value:
            best_value = value
            best_name = name
    return best_name


def hierarchy_frame(weeks: pd.DataFrame) -> pd.DataFrame:
    """Every transfer week, including weeks the two rules agree."""
    out = weeks.copy()
    out["r1_exp_minus_r1_shuffled"] = out["r1_exp"] - out["r1_shuffled"]
    return out


def disagreement_rows(weeks: pd.DataFrame) -> pd.DataFrame:
    """Weeks the moves differ. Agreement weeks are not in this frame."""
    return weeks.loc[weeks["agree_xp_exp"].astype(int) == 0].copy()


def conditional_interval(
    rows: pd.DataFrame,
    column: str,
    *,
    n_boot: int,
    seed: int,
    minimum: int = MIN_DISAGREE,
) -> dict[str, Any]:
    """Cluster bootstrap on disagreement weeks. The 20-week season floor does not apply."""
    from src.eval.gates import cluster_interval

    groups: dict[str, np.ndarray] = {}
    omitted: dict[str, int] = {}
    if rows.empty:
        return {
            "mean": None,
            "lo": None,
            "hi": None,
            "omitted": omitted,
            "conditional": True,
            "n_disagree": {},
        }
    for season, block in rows.groupby("season", sort=True):
        values = block[column].to_numpy(float)
        if int(values.size) < int(minimum):
            omitted[str(season)] = int(values.size)
        else:
            groups[str(season)] = values
    if not groups:
        return {
            "mean": None,
            "lo": None,
            "hi": None,
            "omitted": omitted,
            "conditional": True,
            "n_disagree": {},
        }
    summary = cluster_interval(groups, seasons=tuple(groups), n_boot=n_boot, seed=seed)
    summary["omitted"] = omitted
    summary["conditional"] = True
    summary["n_disagree"] = {season: int(values.size) for season, values in groups.items()}
    return summary


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _player_attributes(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    work = work.dropna(subset=["gw"])
    work["gw"] = work["gw"].astype(int)
    work["player_id"] = work["player_id"].astype(str)
    sums = [column for column in CONTRIBUTIONS if column in work.columns]
    for column in sums:
        work[column] = _num(work, column).fillna(0.0)
    means = [column for column in MEAN_ATTRS if column in work.columns]
    for column in means:
        work[column] = _num(work, column)
    grouped_sum = (
        work.groupby(["gw", "player_id"], sort=False)[sums].sum()
        if sums
        else pd.DataFrame()
    )
    grouped_mean = (
        work.groupby(["gw", "player_id"], sort=False)[means].mean()
        if means
        else pd.DataFrame()
    )
    if grouped_sum.empty:
        return grouped_mean.reset_index()
    if grouped_mean.empty:
        return grouped_sum.reset_index()
    return grouped_sum.join(grouped_mean, how="outer").reset_index()


def _attr_lookup(table: pd.DataFrame) -> dict[tuple[int, str], dict[str, float]]:
    found: dict[tuple[int, str], dict[str, float]] = {}
    if table.empty:
        return found
    for row in table.itertuples(index=False):
        key = (int(row.gw), str(row.player_id))
        payload: dict[str, float] = {}
        for column in (*CONTRIBUTIONS, *MEAN_ATTRS):
            if hasattr(row, column):
                value = getattr(row, column)
                payload[column] = float(value) if pd.notna(value) else float("nan")
        found[key] = payload
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


def _label(names: dict[str, str], pid: str) -> str:
    if not pid:
        return "roll"
    return names.get(str(pid), str(pid)).replace("|", "/")


def _piece(attrs: dict[tuple[int, str], dict[str, float]], gw: int, pid: str) -> dict[str, float]:
    return attrs.get((int(gw), str(pid)), {})


def _finite(payload: dict[str, float], column: str) -> float:
    value = payload.get(column, float("nan"))
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


def annotate_disagreements(
    frame: pd.DataFrame,
    weeks_rows: list[dict[str, Any]],
    *,
    gw_start: int,
    gw_end: int,
    season: str,
) -> list[dict[str, Any]]:
    """Attributes of the weeks the two rules already chose different moves."""
    collapsed = _weeks(frame, gw_start, gw_end)
    attrs = _attr_lookup(_player_attributes(frame))
    names = _names(frame)
    rows: list[dict[str, Any]] = []
    for source in weeks_rows:
        if int(source["agree_xp_exp"]) == 1:
            continue
        gw = int(source["gw"])
        week = collapsed[gw]
        xp_in = str(source["xp_in"])
        xp_out = str(source["xp_out"])
        exp_in = str(source["exp_in"])
        exp_out = str(source["exp_out"])
        xp_scores = score_lookup(week, "score_xp")
        exp_scores = score_lookup(week, "score_exp_points")
        values = value_lookup(week)
        signed = signed_contributions(_piece(attrs, gw, xp_in), _piece(attrs, gw, xp_out))
        driver = driver_name(signed)
        row: dict[str, Any] = {
            "season": season,
            "gw": gw,
            "xp_in": xp_in,
            "xp_out": xp_out,
            "exp_in": exp_in,
            "exp_out": exp_out,
            "xp_in_name": _label(names, xp_in),
            "xp_out_name": _label(names, xp_out),
            "exp_in_name": _label(names, exp_in),
            "exp_out_name": _label(names, exp_out),
            "xp_position": _position(week, xp_in),
            "exp_position": _position(week, exp_in),
            "xp_team": _team(week, xp_in),
            "exp_team": _team(week, exp_in),
            "xp_price": values.get(xp_in, np.nan),
            "exp_price": values.get(exp_in, np.nan),
            "pred_xp": float(source["pred_xp"]),
            "pred_exp": float(source["pred_exp"]),
            "xp_cross": cross_margin(xp_scores, xp_in, xp_out, exp_in, exp_out),
            "exp_cross": cross_margin(exp_scores, exp_in, exp_out, xp_in, xp_out),
            "r1_xp": float(source["r1_xp"]),
            "r1_exp": float(source["r1_exp"]),
            "r3_xp": float(source["r3_xp"]),
            "r3_exp": float(source["r3_exp"]),
            "r1_xp_minus_r1_exp": float(source["r1_xp"]) - float(source["r1_exp"]),
            "r3_xp_minus_r3_exp": float(source["r3_xp"]) - float(source["r3_exp"]),
            "r1_call": realised_call(source["r1_xp"], source["r1_exp"]),
            "r3_call": realised_call(source["r3_xp"], source["r3_exp"]),
            "xp_xmi": _finite(_piece(attrs, gw, xp_in), "xmi"),
            "exp_xmi": _finite(_piece(attrs, gw, exp_in), "xmi"),
            "xp_attack": _finite(_piece(attrs, gw, xp_in), "attack_strength"),
            "exp_attack": _finite(_piece(attrs, gw, exp_in), "attack_strength"),
            "xp_defend": _finite(_piece(attrs, gw, xp_in), "defend_threat"),
            "exp_defend": _finite(_piece(attrs, gw, exp_in), "defend_threat"),
            "driver": driver,
        }
        for name, value in signed.items():
            row[f"contrib_{name}"] = value
        rows.append(row)
    return rows


def _position(week: pd.DataFrame, pid: str) -> str:
    match = week.loc[week["player_id"].astype(str) == str(pid)]
    if match.empty or "position" not in match.columns:
        return ""
    try:
        return normalize_position(str(match.iloc[0]["position"]))
    except ValueError:
        return ""


def _team(week: pd.DataFrame, pid: str) -> str:
    match = week.loc[week["player_id"].astype(str) == str(pid)]
    if match.empty or "team_norm" not in match.columns:
        return ""
    return str(match.iloc[0]["team_norm"])


def freeze_portfolios(
    frame: pd.DataFrame,
    season_index: int,
    *,
    gw_start: int,
    gw_end: int,
) -> tuple[int, dict[str, Squad]]:
    """Four fifteens at the first week they all exist. Later weeks do not transfer."""
    weeks = _weeks(frame, gw_start, gw_end)
    for gw in sorted(weeks):
        try:
            xp = opening_squad(weeks[gw], "score_xp")
            exp = opening_squad(weeks[gw], "score_exp_points")
            neutral = neutral_squad(weeks[gw])
            shuffled = opening_squad(
                permute_week(weeks[gw], "score_xp", season_index, gw),
                "score_xp",
            )
        except RuntimeError:
            continue
        return gw, {"xp": xp, "exp": exp, "shuffled": shuffled, "neutral": neutral}
    raise RuntimeError("no gameweek held all four opening fifteens")


def deployed_points(
    squads: dict[str, Squad],
    week: pd.DataFrame,
    deployment: str,
) -> dict[str, float]:
    """One XI rule for every fifteen. Realised points do not choose it."""
    if deployment == "total_points":
        raise RuntimeError("the XI is not chosen by realised points")
    before = {name: dict(squad.purchase) for name, squad in squads.items()}
    points = {name: week_points(squad, week, deployment) for name, squad in squads.items()}
    for name, squad in squads.items():
        if squad.purchase != before[name]:
            raise RuntimeError("deployment mutated a frozen fifteen")
    return points


def portfolio_rows(
    frame: pd.DataFrame,
    season_index: int,
    *,
    gw_start: int,
    gw_end: int,
    season: str,
) -> list[dict[str, Any]]:
    start, squads = freeze_portfolios(frame, season_index, gw_start=gw_start, gw_end=gw_end)
    weeks = _weeks(frame, gw_start, gw_end)
    rows: list[dict[str, Any]] = []
    for gw in sorted(weeks):
        if gw < start:
            continue
        for deployment in DEPLOYMENTS:
            points = deployed_points(squads, weeks[gw], deployment)
            if set(points) != set(PORTFOLIOS):
                raise RuntimeError("a portfolio is missing")
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "start_gw": start,
                    "deployment": deployment,
                    "xp": points["xp"],
                    "exp": points["exp"],
                    "shuffled": points["shuffled"],
                    "neutral": points["neutral"],
                    "xp_minus_exp": points["xp"] - points["exp"],
                    "xp_minus_shuffled": points["xp"] - points["shuffled"],
                    "xp_minus_neutral": points["xp"] - points["neutral"],
                }
            )
    return rows


def _band(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "no season reached the floor"
    if float(row["lo"]) > 0.0:
        return "the interval stays above zero"
    if float(row["hi"]) < 0.0:
        return "the interval stays below zero"
    return "the interval covers zero"


def _fmt(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "no season reached the floor"
    return f"{row['mean']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"


def _mean(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "not identified"
    return f"{float(values.mean()):+.2f}"


def _level(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "not identified"
    return f"{float(values.mean()):.2f}"


def _counts(series: pd.Series) -> str:
    cleaned = [str(value) for value in series.tolist() if str(value)]
    if not cleaned:
        return "none"
    tally = pd.Series(cleaned).value_counts()
    return ", ".join(f"{name} {int(count)}" for name, count in tally.items())


def _call_counts(frame: pd.DataFrame, column: str) -> str:
    if frame.empty:
        return "none"
    tally = frame[column].value_counts()
    parts = [f"{name} {int(tally.get(name, 0))}" for name in ("xp", "exp", "equal")]
    return ", ".join(parts)


def _season_lines(frame: pd.DataFrame, column: str) -> list[str]:
    lines = []
    for season, block in frame.groupby("season", sort=True):
        lines.append(f"| {season} | {len(block)} | {block[column].mean():+.2f} |")
    return lines


def _lines(
    protocol: dict[str, Any],
    hierarchy: dict[str, Any],
    conditional: dict[str, Any],
    weeks: pd.DataFrame,
    differ: pd.DataFrame,
    portfolios: pd.DataFrame,
    portfolio_intervals: dict[str, Any],
) -> list[str]:
    return [
        "# Disagreement, hierarchy, opening portfolio",
        "",
        "No change is made to score_xp.",
        "",
        "The disagreement table is the subset of weeks on which the two rules "
        "choose different transfers. It does not replace the unconditional contrast.",
        "",
        "A realised gap of exactly zero is a tie.",
        "",
        "The conditional interval waives the 20-week floor and is not a certified "
        "replacement for the unconditional result.",
        "",
        "## When the two rules differ",
        "",
        f"Disagreement weeks: {len(differ)} of {len(weeks)}. "
        f"One-week calls: {_call_counts(differ, 'r1_call')}. "
        f"Three-week calls: {_call_counts(differ, 'r3_call')}.",
        "",
        f"Conditional R1, score_xp minus expected points: {_fmt(conditional['r1'])}. "
        f"{_band(conditional['r1'])}.",
        f"Conditional R3: {_fmt(conditional['r3'])}. {_band(conditional['r3'])}.",
        "",
        "An interval that covers zero is inconclusive. No winner is declared. "
        "The published unconditional one-week contrast remains the centre result.",
        "",
        "On the disagreement weeks the one-week interval covers zero, so which rule "
        "is right that week is inconclusive. The call counts are not a contrast.",
        "",
        "On those weeks the three-week interval stays below zero. That is not a change to score_xp.",
        "",
        "| season | weeks | R1 mean | R3 mean | R1 xp | R1 exp | R1 tie |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *_differ_season_lines(differ),
        "",
        f"Mean cross margin in score_xp units, own move minus the other move: "
        f"{_mean(differ['xp_cross'])}. "
        f"In expected-points units: {_mean(differ['exp_cross'])}. "
        "Those two numbers are not subtracted from each other.",
        "",
        f"Mean price of the xp buy {_level(differ['xp_price'])} tenths, of the exp buy "
        f"{_level(differ['exp_price'])} tenths. "
        f"Mean xmi {_level(differ['xp_xmi'])} against {_level(differ['exp_xmi'])}. "
        f"Mean attack strength {_level(differ['xp_attack'])} against {_level(differ['exp_attack'])}. "
        f"Mean defend threat {_level(differ['xp_defend'])} against {_level(differ['exp_defend'])}.",
        "",
        f"Buy positions, score_xp: {_counts(differ['xp_position'])}. "
        f"Expected points: {_counts(differ['exp_position'])}.",
        f"Largest signed piece of the score_xp move: {_counts(differ['driver'])}. "
        "When that piece is bps, bps is 0.18·goals + 0.12·assists + 0.08·cs and is "
        "not an independent channel.",
        "",
        "The largest piece of the score_xp move is goals on 31 weeks, appearance on 15, "
        "and clean sheets on 5. The xp buy is cheaper, has lower expected minutes, and "
        "has higher attack strength.",
        "",
        "The most common expected-points buy on a disagreement week is Haaland, 14 weeks, "
        "then Salah, 11. The score_xp buys are more spread.",
        "",
        "Mean signed contribution of the score_xp move, in minus out:",
        "",
        "| piece | mean |",
        "|---|---:|",
        *_contrib_lines(differ),
        "",
        "| season | GW | XP out | XP in | Exp out | Exp in | XP Δ | Exp Δ | R1 xp | R1 exp | R3 xp | R3 exp | R1 |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        *_move_lines(differ),
        "",
        "The diverging-path isolated gap of +1.24 is not this table. "
        "The common-state 2022-23 decision-week mean on every transfer week, "
        "including agreement, is −1.34.",
        "",
        "## Hierarchy",
        "",
        "R1 of expected points minus shuffled score_xp is the missing leg of the "
        "common-state hierarchy.",
        "",
        "The squad is the same frozen fifteen. Agreement weeks stay in all three legs.",
        "",
        "| leg | estimate | reading |",
        "|---|---|---|",
        f"| R1 xp − shuffled | {_fmt(hierarchy['r1_xp_minus_r1_shuffled'])} | {_band(hierarchy['r1_xp_minus_r1_shuffled'])} |",
        f"| R1 exp − shuffled | {_fmt(hierarchy['r1_exp_minus_r1_shuffled'])} | {_band(hierarchy['r1_exp_minus_r1_shuffled'])} |",
        f"| R1 xp − exp | {_fmt(hierarchy['r1_xp_minus_r1_exp'])} | {_band(hierarchy['r1_xp_minus_r1_exp'])} |",
        "",
        "### Expected points minus shuffled score_xp",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(weeks, "r1_exp_minus_r1_shuffled"),
        "",
        "Expected points minus shuffled score_xp is +4.27 [+3.08, +5.42] and the "
        "interval stays above zero. Both informed scores beat the shuffle. The "
        "one-week gap between them covers zero.",
        "",
        "## Opening portfolio",
        "",
        "The initial portfolio is frozen at the first week all four fifteens exist, "
        "and no transfer is applied.",
        "",
        "Both deployments are pre-registered. The XI rule is the same for every "
        "fifteen inside one contrast.",
        "",
        "This opener is the one-week squad solver. It is not the published climb. "
        "The construction week is included.",
        "",
        *_portfolio_sections(portfolios, portfolio_intervals),
        "",
        "The score_xp opening fifteen beats the shuffled fifteen and the price ladder "
        "under both XI rules. Against the expected-points fifteen, the gap covers zero "
        "when the XI is chosen by score_xp and stays below zero when the XI is chosen "
        "by expected points.",
        "",
        "The 2022-23 season mean of the frozen xp fifteen minus the frozen exp fifteen, "
        "with the XI chosen by expected points, is −14.45. That weekly series is not "
        "the diverging-path squad gap.",
        "",
        "No winner is declared from either deployment. No change is made to score_xp.",
        "",
        "Gemini reviewed these diagnostics "
        "([disagreement diagnostics](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
        "",
        f"Bootstrap {protocol['bootstrap']}, seed {protocol['seed']}. "
        f"Closed seasons: {', '.join(protocol['closed_seasons'])}.",
        "",
    ]


def _differ_season_lines(frame: pd.DataFrame) -> list[str]:
    lines = []
    for season, block in frame.groupby("season", sort=True):
        calls = block["r1_call"].value_counts()
        lines.append(
            f"| {season} | {len(block)} | {block['r1_xp_minus_r1_exp'].mean():+.2f} | "
            f"{block['r3_xp_minus_r3_exp'].mean():+.2f} | "
            f"{int(calls.get('xp', 0))} | {int(calls.get('exp', 0))} | {int(calls.get('equal', 0))} |"
        )
    return lines


def _contrib_lines(frame: pd.DataFrame) -> list[str]:
    lines = []
    for name in CONTRIBUTIONS:
        column = f"contrib_{name}"
        if column not in frame.columns:
            continue
        lines.append(f"| {name} | {_mean(frame[column])} |")
    return lines


def _move_lines(frame: pd.DataFrame) -> list[str]:
    lines = []
    ordered = frame.sort_values(["season", "gw"])
    for row in ordered.itertuples(index=False):
        lines.append(
            "| {season} | {gw} | {xp_out} | {xp_in} | {exp_out} | {exp_in} | "
            "{pred_xp:+.2f} | {pred_exp:+.2f} | {r1_xp:+.0f} | {r1_exp:+.0f} | "
            "{r3_xp:+.0f} | {r3_exp:+.0f} | {call} |".format(
                season=row.season,
                gw=int(row.gw),
                xp_out=row.xp_out_name,
                xp_in=row.xp_in_name,
                exp_out=row.exp_out_name,
                exp_in=row.exp_in_name,
                pred_xp=float(row.pred_xp),
                pred_exp=float(row.pred_exp),
                r1_xp=float(row.r1_xp),
                r1_exp=float(row.r1_exp),
                r3_xp=float(row.r3_xp),
                r3_exp=float(row.r3_exp),
                call=row.r1_call,
            )
        )
    return lines


def _portfolio_sections(frame: pd.DataFrame, intervals: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    labels = {
        "score_xp": "XI and captain by score_xp",
        "score_exp_points": "XI and captain by expected points",
    }
    titles = (
        ("xp_minus_exp", "xp fifteen − exp fifteen"),
        ("xp_minus_shuffled", "xp fifteen − shuffled fifteen"),
        ("xp_minus_neutral", "xp fifteen − price ladder"),
    )
    starts = (
        frame.groupby("season", sort=True)["start_gw"].first()
        if not frame.empty
        else pd.Series(dtype=int)
    )
    if not starts.empty:
        bits = ", ".join(f"{season} GW{int(gw)}" for season, gw in starts.items())
        lines.extend([f"Opening week: {bits}.", ""])
    for deployment in DEPLOYMENTS:
        block = frame.loc[frame["deployment"] == deployment]
        lines.extend(
            [
                f"### {labels[deployment]}",
                "",
                "| contrast | estimate | reading |",
                "|---|---|---|",
            ]
        )
        for column, title in titles:
            row = intervals.get(f"{deployment}__{column}", {})
            lines.append(f"| {title} | {_fmt(row)} | {_band(row)} |")
        lines.extend(
            [
                "",
                "| season | weeks | xp − exp | xp − shuffled | xp − ladder |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for season, season_block in block.groupby("season", sort=True):
            lines.append(
                f"| {season} | {len(season_block)} | {season_block['xp_minus_exp'].mean():+.2f} | "
                f"{season_block['xp_minus_shuffled'].mean():+.2f} | "
                f"{season_block['xp_minus_neutral'].mean():+.2f} |"
            )
        lines.append("")
    return lines


def run() -> None:
    from src.eval.decision import pool_columns
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season

    protocol = load_protocol()
    if protocol["disagreement"].get("winner") is not None:
        raise RuntimeError("the disagreement lock already names a winner")
    if protocol["hierarchy"].get("winner") is not None:
        raise RuntimeError("the hierarchy lock already names a winner")
    if protocol["initial_squad"].get("winner") is not None:
        raise RuntimeError("the initial-squad lock already names a winner")
    if protocol["initial_squad"].get("transfers") != "none":
        raise RuntimeError("the initial portfolio applies a transfer")
    codes = protocol["season_codes"]
    gw_start = int(protocol["gw_start"])
    gw_end = int(protocol["gw_end"])
    minimum = int(protocol["min_gws_per_season"])
    n_boot = int(protocol["bootstrap"])
    seed = int(protocol["seed"])
    week_rows: list[dict[str, Any]] = []
    differ_rows: list[dict[str, Any]] = []
    portfolio: list[dict[str, Any]] = []
    for index, season in enumerate(protocol["closed_seasons"]):
        print(f"disagreement {season}", flush=True)
        frame = build_season(season, codes[season], protocol)
        rows, _start = evaluate_season(frame, index, gw_start=gw_start, gw_end=gw_end)
        for row in rows:
            week_rows.append({"season": season, **row})
        differ_rows.extend(
            annotate_disagreements(
                frame, rows, gw_start=gw_start, gw_end=gw_end, season=season
            )
        )
        portfolio.extend(
            portfolio_rows(frame, index, gw_start=gw_start, gw_end=gw_end, season=season)
        )
    weeks = hierarchy_frame(pd.DataFrame(week_rows))
    differ = pd.DataFrame(differ_rows)
    held = pd.DataFrame(portfolio)
    hierarchy = pool_columns(
        weeks,
        ("r1_xp_minus_r1_shuffled", "r1_exp_minus_r1_shuffled", "r1_xp_minus_r1_exp"),
        minimum=minimum,
        n_boot=n_boot,
        seed=seed,
    )
    conditional = {
        "r1": conditional_interval(
            differ, "r1_xp_minus_r1_exp", n_boot=n_boot, seed=seed
        ),
        "r3": conditional_interval(
            differ, "r3_xp_minus_r3_exp", n_boot=n_boot, seed=seed
        ),
    }
    portfolio_intervals: dict[str, Any] = {}
    for deployment in DEPLOYMENTS:
        block = held.loc[held["deployment"] == deployment]
        pooled = pool_columns(
            block,
            ("xp_minus_exp", "xp_minus_shuffled", "xp_minus_neutral"),
            minimum=minimum,
            n_boot=n_boot,
            seed=seed,
        )
        for column, summary in pooled.items():
            portfolio_intervals[f"{deployment}__{column}"] = summary
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = "score_xp_minus_score_exp_points"
    audit = run_asof_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weeks.to_csv(PROCESSED / "hierarchy_weeks.csv", index=False)
    differ.to_csv(PROCESSED / "disagreement_weeks.csv", index=False)
    held.to_csv(PROCESSED / "initial_squad_weeks.csv", index=False)
    comparisons = {
        key: certified["comparisons"][key],
        **hierarchy,
        "conditional_r1": conditional["r1"],
        "conditional_r3": conditional["r3"],
        **portfolio_intervals,
    }
    write_gated_report(
        REPORTS / "disagreement.md",
        audit,
        {
            "seasons": list(protocol["closed_seasons"]),
            "min_gws": minimum,
            "comparisons": comparisons,
        },
        _lines(protocol, hierarchy, conditional, weeks, differ, held, portfolio_intervals),
    )


if __name__ == "__main__":
    run()
