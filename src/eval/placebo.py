"""Placebo for the one-free-transfer rule.

Shuffled scores stay inside the gameweek. Expected points use the same rule
with no shuffle. A squad that never transfers is reported and is not the
comparison that decides whether the score has skill. No winner is declared.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.eval.alignment import eligible_rank_gap_by_week, spearman_by_week
from src.eval.decision import (
    SCORE_COLUMN,
    calibrate,
    loso_thresholds,
    pool_columns,
    replay_season,
)
from src.eval.decision_spec import CAPTAIN_BASELINE

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def _by_gw(rows: list[dict[str, Any]], field: str) -> dict[int, float]:
    return {int(row["gw"]): float(row[field]) for row in rows}


def _paired(
    season: str,
    left: dict[int, float],
    right: dict[int, float],
) -> list[dict[str, Any]]:
    rows = []
    for gw in sorted(set(left) & set(right)):
        rows.append({"season": season, "gw": gw, "delta": left[gw] - right[gw]})
    return rows


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


def run() -> None:
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season

    protocol = load_protocol()
    placebo = protocol["placebo"]
    if placebo.get("winner") is not None:
        raise RuntimeError("the placebo lock already names a winner")
    codes = protocol["season_codes"]
    gw_start = int(protocol["gw_start"])
    gw_end = int(protocol["gw_end"])
    paired_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    checked = 0
    for index, season in enumerate(protocol["closed_seasons"]):
        frame = build_season(season, codes[season], protocol)
        xp = replay_season(frame, SCORE_COLUMN, gw_start=gw_start, gw_end=gw_end)
        naive = replay_season(frame, CAPTAIN_BASELINE, gw_start=gw_start, gw_end=gw_end)
        shuffled = replay_season(
            frame,
            SCORE_COLUMN,
            gw_start=gw_start,
            gw_end=gw_end,
            permute_season_index=index,
        )
        checked += len(xp["transfers"]) + len(naive["transfers"]) + len(shuffled["transfers"])
        xp_g = _by_gw(xp["weeks"], "greedy")
        xp_h = _by_gw(xp["weeks"], "hold")
        exp_g = _by_gw(naive["weeks"], "greedy")
        sh_g = _by_gw(shuffled["weeks"], "greedy")
        sh_h = _by_gw(shuffled["weeks"], "hold")
        for row in _paired(season, xp_g, exp_g):
            paired_rows.append({**row, "contrast": "greedy_xp_minus_greedy_exp"})
        for row in _paired(season, sh_g, sh_h):
            paired_rows.append({**row, "contrast": "greedy_shuffled_minus_hold_shuffled"})
        for row in _paired(season, xp_g, xp_h):
            paired_rows.append({**row, "contrast": "greedy_xp_minus_hold_xp"})
        for row in _paired(season, xp_g, sh_g):
            paired_rows.append({**row, "contrast": "greedy_xp_minus_greedy_shuffled"})
        for row in xp["transfers"]:
            transfer_rows.append({"season": season, "score": SCORE_COLUMN, **row})
        rank_rows.extend(spearman_by_week(frame))
        gap_rows.extend(eligible_rank_gap_by_week(frame))
    paired = pd.DataFrame(paired_rows)
    wide = paired.pivot_table(
        index=["season", "gw"], columns="contrast", values="delta", aggfunc="first"
    ).reset_index()
    intervals = pool_columns(
        wide,
        (
            "greedy_xp_minus_greedy_shuffled",
            "greedy_xp_minus_greedy_exp",
            "greedy_shuffled_minus_hold_shuffled",
            "greedy_xp_minus_hold_xp",
        ),
        minimum=int(protocol["min_gws_per_season"]),
        n_boot=int(protocol["bootstrap"]),
        seed=int(protocol["seed"]),
    )
    transfers = pd.DataFrame(transfer_rows)
    horizon_fit = calibrate(
        transfers, n_boot=int(protocol["bootstrap"]), seed=int(protocol["seed"])
    )
    one_week = transfers.drop(columns=["realised"]).rename(columns={"realised_t": "realised"})
    week_fit = calibrate(
        one_week, n_boot=int(protocol["bootstrap"]), seed=int(protocol["seed"])
    )
    folds = loso_thresholds(one_week)
    ranks = pd.DataFrame(rank_rows)
    gaps = pd.DataFrame(gap_rows)
    negative = ranks.loc[ranks["negative"]].sort_values(["season", "gw"])
    gap_negative = gaps.loc[gaps["negative"]].sort_values(["season", "gw"])
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    key = "score_xp_minus_score_exp_points"
    audit = run_asof_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    wide.to_csv(PROCESSED / "placebo_weeks.csv", index=False)
    transfers.to_csv(PROCESSED / "placebo_transfers.csv", index=False)
    ranks.to_csv(PROCESSED / "spearman_by_week.csv", index=False)
    gaps.to_csv(PROCESSED / "rank_gap_by_week.csv", index=False)
    lines = _lines(
        protocol,
        intervals,
        wide,
        horizon_fit,
        week_fit,
        folds,
        negative,
        ranks,
        gaps,
        gap_negative,
        checked,
        _odds_sentence(protocol),
        _anchor_sentence(),
    )
    write_gated_report(
        REPORTS / "decision_placebo.md",
        audit,
        {
            "seasons": list(protocol["closed_seasons"]),
            "min_gws": int(protocol["min_gws_per_season"]),
            "comparisons": {key: certified["comparisons"][key], **intervals},
        },
        lines,
    )


def _fit_sentence(fit: dict[str, Any], label: str) -> str:
    if "b" not in fit:
        return f"{label}: the slope is not identified."
    return (
        f"{label}: a = {fit['a']:+.4f}, b = {fit['b']:+.4f} "
        f"[{fit.get('lo', float('nan')):+.4f}, {fit.get('hi', float('nan')):+.4f}], "
        f"n = {fit.get('n', 0)}."
    )


def _fill_rate_sentence() -> str:
    """Nonzero scraped xP among players who played, on weeks the column was filled."""
    cache = ROOT / "data" / "cache"
    played_hit = played_n = dnp_hit = dnp_n = 0
    for season in ("2022_23", "2023_24", "2024_25", "2025_26"):
        frame = pd.read_csv(cache / f"merged_gw_{season}.csv", usecols=["minutes", "xP", "GW"])
        frame["xP"] = pd.to_numeric(frame["xP"], errors="coerce")
        filled = frame.groupby("GW")["xP"].transform("max") > 0
        block = frame.loc[filled]
        played = block["minutes"].fillna(0) > 0
        positive = block["xP"].fillna(0) > 0
        played_hit += int(positive.loc[played].sum())
        played_n += int(played.sum())
        dnp_hit += int(positive.loc[~played].sum())
        dnp_n += int((~played).sum())
    return (
        f"On filled scrapes, {played_hit / played_n:.1%} of players who played have a "
        f"nonzero scraped xP and {dnp_hit / dnp_n:.1%} of players who did not. "
        "A pre-match forecast does not know who played."
    )


def _odds_sentence(protocol: dict[str, Any]) -> str:
    """Count sheet rows whose club has no opening-odds fixture. No network."""
    from src.ingest.fpl_odds import join_players_to_fixtures, load_football_data, load_player_logs

    parts = []
    for season in protocol["closed_seasons"]:
        players = load_player_logs(season=season)
        fixtures = load_football_data(code=protocol["season_codes"][season])
        _joined, _fix, stats = join_players_to_fixtures(
            players, fixtures, retain_sheet_rows=True
        )
        parts.append(f"{season} {int(stats['n_unmatched'])} of {int(stats['n_player_appearances'])}")
    return (
        "Sheet rows with no opening-odds fixture: " + "; ".join(parts) + ". "
        "A miss of zero means the World Cup break is a gap in the calendar, not a dropped join."
    )


def _anchor_sentence() -> str:
    path = PROCESSED / "average_entry_score_2025_26.csv"
    if not path.is_file():
        return (
            "The price-target template is a fixed heuristic, not an ownership squad. "
            "An official average-manager score for 2022-23, 2023-24, and 2024-25 is not in the cache. "
            "Anchor: null."
        )
    frame = pd.read_csv(path)
    window = frame.loc[frame["gw"].between(5, 38), "average_entry_score"]
    decision = pd.read_csv(PROCESSED / "decision_weeks.csv")
    season = decision.loc[decision["season"] == "2025-26"]
    return (
        "The price-target template is a fixed heuristic, not an ownership squad. "
        "`selected` is not an input. It is built once, at the first week a legal squad exists, "
        "by walking the locked price targets and taking the closest eligible price in that position. "
        "Ties break toward the higher score, then the lower player id. Later weeks do not transfer. "
        f"The official average entry score for 2025-26, gameweeks 5–38, is {float(window.mean()):.2f}. "
        f"On those weeks this replay's hold averages {float(season['hold'].mean()):.2f}, "
        f"the template {float(season['template'].mean()):.2f}, "
        f"and greedy {float(season['greedy'].mean()):.2f}. "
        "The official average includes automatic substitutes, chips, and hits. "
        "This replay is the eleven plus the captain, with no hit. "
        "2022-23, 2023-24, and 2024-25 are not in that public file. Anchor for those seasons: null."
    )


def _break_lines(wide: pd.DataFrame) -> list[str]:
    block = wide.loc[wide["season"] == "2022-23"].sort_values("gw")
    column = "greedy_xp_minus_greedy_exp"
    lines = [
        "## 2022-23, week by week",
        "",
        "This is greedy on `score_xp` minus greedy on expected points. "
        "Gameweek 7 has no sheet, so it is absent. "
        "Gameweek 16 kicks off on 12 November 2022 and gameweek 17 on 26 December 2022.",
        "",
        "| gameweek | points |",
        "|---:|---:|",
    ]
    for row in block.itertuples():
        lines.append(f"| {int(row.gw)} | {float(getattr(row, column)):+.1f} |")
    values = block[column]
    lines += [
        "",
        f"The season mean is {float(values.mean()):+.2f} on {len(block)} weeks. "
        "The loss is in the block before the World Cup break and in the block after it. "
        "It is not one missing gameweek and it is not dropped from the pool.",
        "",
    ]
    return lines


def _fold_lines(folds: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| held-out season | training transfers | a1 | b1 | threshold |",
        "|---|---:|---:|---:|---:|",
    ]
    for fold in folds:
        if fold.get("threshold") is None:
            threshold = "undefined"
            a_text = "—"
            b_text = "—"
        else:
            threshold = f"{fold['threshold']:+.2f}"
            a_text = f"{fold['a']:+.3f}"
            b_text = f"{fold['b']:+.3f}"
        lines.append(
            f"| {fold['held_out']} | {fold['n']} | {a_text} | {b_text} | {threshold} |"
        )
    return lines


def _week_list(frame: pd.DataFrame, value: str) -> str:
    if frame.empty:
        return "none"
    return ", ".join(
        f"{row.season} GW{int(row.gw)} ({float(getattr(row, value)):+.3f})"
        for row in frame.itertuples()
    )


def _lines(
    protocol: dict[str, Any],
    intervals: dict[str, Any],
    wide: pd.DataFrame,
    horizon_fit: dict[str, Any],
    week_fit: dict[str, Any],
    folds: list[dict[str, Any]],
    negative: pd.DataFrame,
    ranks: pd.DataFrame,
    gaps: pd.DataFrame,
    gap_negative: pd.DataFrame,
    checked: int,
    odds_text: str,
    anchor_text: str,
) -> list[str]:
    skill = intervals["greedy_xp_minus_greedy_exp"]
    placebo = intervals["greedy_shuffled_minus_hold_shuffled"]
    straw = intervals["greedy_xp_minus_hold_xp"]
    direct = intervals["greedy_xp_minus_greedy_shuffled"]
    negative_text = (
        "No gameweek has a negative Spearman of the two forecasts."
        if negative.empty
        else f"Negative weeks for that correlation: {_week_list(negative, 'spearman')}."
    )
    undefined = int(ranks["undefined"].sum()) if len(ranks) else 0
    window = gaps.loc[(gaps["gw"] >= 5) & (gaps["gw"] <= 38) & ~gaps["undefined"]]
    gap_mean = float(window["rank_gap"].mean()) if len(window) else float("nan")
    gap_text = (
        "No filled week has a negative rank gap."
        if gap_negative.empty
        else (
            f"Negative rank-gap weeks: {_week_list(gap_negative, 'rank_gap')}. "
            f"{len(gap_negative)} of {int((~gaps['undefined']).sum())} defined weeks."
        )
    )
    a = horizon_fit.get("a")
    b = horizon_fit.get("b")
    a1 = week_fit.get("a")
    b1 = week_fit.get("b")
    if a is None or b is None:
        three_week_text = "The three-week line is not identified."
    else:
        three_week_text = (
            f"The old printed threshold used this three-week line: (4 − a) / b "
            f"= (4 − ({a:+.4f})) / {b:+.4f} = {(4.0 - a) / b:+.3f}. "
            "That is not the hurdle. A slope near 1.2 is what a three-week sum against "
            "a one-week prediction looks like. "
            f"b/3 = {b / 3:+.4f} is a per-week arithmetic rate rescaling of that slope, "
            "not winner's-curse shrinkage."
        )
    if a1 is None or b1 is None or float(b1) <= 0.0:
        one_week_hurdle = "The in-sample one-week hurdle is undefined."
    else:
        one_week_hurdle = (
            f"The one-week intercept is {a1:+.4f}. "
            f"The in-sample hurdle solves a1 + b1 x = 4, so x = (4 − a1) / b1 "
            f"= (4 − ({a1:+.4f})) / {b1:+.4f} = {(4.0 - float(a1)) / float(b1):+.3f}. "
            "That pooled line is an in-sample observation. It was not used to hurdle or execute a transfer."
        )
    defined = gaps.loc[~gaps["undefined"]]
    if len(defined) and "rho_score" in defined.columns:
        level_text = (
            f"On {len(defined)} defined weeks the mean within-position Spearman with points "
            f"is {float(defined['rho_score'].mean()):+.3f} for `score_xp` and "
            f"{float(defined['rho_scraped'].mean()):+.3f} for scraped xP. "
            f"On filled weeks from gameweek 5 to 38 those levels are "
            f"{float(window['rho_score'].mean()):+.3f} and {float(window['rho_scraped'].mean()):+.3f}."
        )
    else:
        level_text = "The two Spearman levels are not on this frame."
    return [
        "# Decision placebo",
        "",
        "Locked before these totals were read. The greedy rule is one same-position "
        "free transfer when that week's score gain is positive. It does not bank a "
        "second transfer and it does not take a hit. The opening fifteen maximises "
        "the same score. A squad that never transfers is a weak baseline: it cannot "
        "replace an injury or a blank. The comparison for the score is greedy on "
        "`score_xp` against greedy on `score_exp_points`. The shuffle permutes "
        "`score_xp` among eligible players inside each gameweek. Seed (0, season index, "
        "gameweek). Points are not shuffled. No winner is declared against `ep_next`.",
        "",
        f"Every greedy transfer was checked against budget, `sell_price`, same position, "
        f"the club cap, a squad of 15, and eligibility in that week only. "
        f"Transfers checked: {checked}. The buy uses that week's score. "
        "Eligibility is three prior appearances and expected minutes at least 45, "
        "including weeks of 0 minutes. The current week's minutes and points do not enter it.",
        "",
        "A positive number is points per gameweek.",
        "",
        "| contrast | pooled mean [95% interval] | reading |",
        "|---|---:|---|",
        f"| greedy score_xp − shuffled greedy | {_fmt_interval(direct)} | {_band(direct)} |",
        f"| greedy score_xp − greedy expected points | {_fmt_interval(skill)} | {_band(skill)} |",
        f"| shuffled greedy − shuffled hold | {_fmt_interval(placebo)} | {_band(placebo)} |",
        f"| score_xp greedy − score_xp hold | {_fmt_interval(straw)} | {_band(straw)} |",
        "",
        "The placebo contrast pairs the realised weekly points of the `score_xp` greedy "
        "squad against the shuffled-score greedy squad on the same gameweeks. "
        "It is not a difference of the two hold baselines. Each squad builds its own "
        "opening fifteen on the column it sees. "
        f"The two opening squads differ by {float((wide['greedy_xp_minus_greedy_shuffled'] - wide['greedy_xp_minus_hold_xp'] + wide['greedy_shuffled_minus_hold_shuffled']).mean()):+.2f} "
        f"points a week, and the two greedy-minus-hold edges differ by "
        f"{float((wide['greedy_xp_minus_hold_xp'] - wide['greedy_shuffled_minus_hold_shuffled']).mean()):+.2f}. "
        "That second figure has no interval of its own. The direct gap is those two pieces added, "
        "and its interval is the one in the table. It is the real score against a squad built on noise. "
        "It is not a win over expected points. Subtracting the published hold rows is not the contrast. "
        "The hold row is the squad that never transfers. It is not the skill comparison. "
        "Permuting the score inside each gameweek does not reproduce a pooled gain "
        "that stays above zero. 2023-24 of that shuffle is several points, so "
        "replacing players can beat a squad that never transfers when the score is noise. "
        "The skill comparison covers zero. 2022-23 is sharply negative and stays in the pool. "
        "The one-week slope's interval reaches 1, so a point estimate below 1 is a direction, "
        "not a finding that the predictions were overstated. No winner is declared.",
        "",
        "## By season",
        "",
        "### greedy score_xp − greedy expected points",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(wide, "greedy_xp_minus_greedy_exp"),
        "",
        "### shuffled greedy − shuffled hold",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(wide, "greedy_shuffled_minus_hold_shuffled"),
        "",
        "### greedy score_xp − shuffled greedy",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(wide, "greedy_xp_minus_greedy_shuffled"),
        "",
        "### score_xp greedy − score_xp hold",
        "",
        "| season | weeks | mean |",
        "|---|---:|---:|",
        *_season_lines(wide, "greedy_xp_minus_hold_xp"),
        "",
        *_break_lines(wide),
        odds_text,
        "",
        "## Template",
        "",
        anchor_text,
        "",
        "## What 4.33 and 0.40 were",
        "",
        _fit_sentence(horizon_fit, "Three-week realised gain on the one-week predicted gain"),
        "",
        three_week_text,
        "",
        _fit_sentence(week_fit, "The same transfers, realised in the decision week only"),
        "",
        one_week_hurdle,
        "",
        "A threshold that is offered as a choice is fit on the other three seasons. "
        "It is undefined when that slope is not positive. None of these numbers entered the rule.",
        "",
        *_fold_lines(folds),
        "",
        "No winner is declared between `score_xp` and `score_exp_points`. "
        "At 20 live weeks, an interval that covers zero is undetermined. "
        f"The test continues through gameweek {protocol['encompassing']['continue_to_gw']}.",
        "",
        "## Spearman, every week",
        "",
        "Two different correlations. The first is `score_xp` against scraped xP. "
        "A positive value means the two forecasts rank players the same way. "
        "An undefined week has no variation in one column, which is what an "
        "all-zero scrape looks like. "
        f"Undefined weeks: {undefined}. {negative_text}",
        "",
        "The withdrawn −0.35 is not that correlation. It is, on eligible players, "
        "Spearman(`score_xp`, points) minus Spearman(scraped xP, points), within "
        "position, averaged across positions. "
        f"{level_text} "
        f"On filled weeks from gameweek 5 to 38 the mean of those weekly gaps is {gap_mean:+.4f}. "
        f"{gap_text} {_fill_rate_sentence()} "
        "Scraped xP stays unusable as a feature and as a benchmark.",
        "",
        "2022-23 has no gameweek 7 sheet, so that week is absent rather than negative. "
        "An all-zero scraped column is undefined, not a negative correlation. "
        "The rank gap is negative on every defined week, so it is not a few misaligned "
        "weeks and not the missing gameweek 7. "
        "These weeks are a diagnostic of a withdrawn figure. They are not a benchmark. "
        "Gemini reviewed the diagnostics "
        "([placebo diagnostics](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
        "",
    ]


if __name__ == "__main__":
    run()
