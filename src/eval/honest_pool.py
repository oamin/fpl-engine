"""Player-GW log scores on the repaired pool. Four closed seasons. No 2026/27 rows.

The likelihood is the gate. The stripped XI is a sanity check and is not a pass mark.
This module does not write the player-GW frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.encompassing import live_encompassing
from src.eval.gates import (
    CLOSED_SEASONS,
    COMPARISONS,
    cluster_interval,
    comparison_key,
    gaussian_log_score,
    load_protocol,
    run_asof_audit,
    write_gated_report,
)
from src.eval.official_xp import sheet_fill, usable_gameweeks
from src.eval.provenance import OFFICIAL_XP_COLUMNS, assert_predeadline_xp, load_deadlines
from src.eval.slices import slice_rows
from src.ingest.fpl_odds import join_players_to_fixtures, load_football_data, load_player_logs
from src.models.season_climb import pick_xi
from src.models.xp_engine import (
    DEFCON_THRESH,
    MIN_MINUTES,
    add_market_pots,
    add_player_priors,
    compute_xp,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
LOGSCORE_CSV = PROCESSED / "player_gw_logscore.csv"
XI_CSV = PROCESSED / "player_gw_xi_sanity.csv"
SLICE_CSV = PROCESSED / "player_gw_slices.csv"


def eligible_mask(frame: pd.DataFrame, *, min_history: int, min_xmi: float) -> pd.Series:
    """History and expected minutes only. The scored week's minutes do not enter."""
    n_prior = pd.to_numeric(frame["n_prior"], errors="coerce")
    xmi = pd.to_numeric(frame["xmi"], errors="coerce")
    return (n_prior >= min_history) & (xmi >= min_xmi)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["gw"] = pd.to_numeric(out["gw"], errors="coerce")
    out = out.dropna(subset=["gw", "position"]).copy()
    out["gw"] = out["gw"].astype(int)
    minutes = pd.to_numeric(out["minutes"], errors="coerce").fillna(0.0)
    out["minutes"] = minutes
    out["total_points"] = pd.to_numeric(out["total_points"], errors="coerce").fillna(0.0)
    out["xG"] = pd.to_numeric(out["xG"], errors="coerce").fillna(0.0)
    out["xA"] = pd.to_numeric(out["xA"], errors="coerce").fillna(0.0)
    out["goals"] = pd.to_numeric(out.get("goals", 0.0), errors="coerce").fillna(0.0)
    out["official_xp"] = pd.to_numeric(out["official_xp"], errors="coerce")
    out["value"] = pd.to_numeric(out.get("value"), errors="coerce")
    raw = pd.to_numeric(out.get("defcon", 0.0), errors="coerce").fillna(0.0)
    thr = out["position"].map(DEFCON_THRESH)
    out["defcon_hit"] = (
        out["position"].isin(list(DEFCON_THRESH))
        & (minutes >= MIN_MINUTES)
        & (raw >= thr.fillna(999))
    ).astype(float)
    if "p_not_lose" not in out.columns:
        out["p_not_lose"] = np.nan
    out["p_not_lose"] = pd.to_numeric(out["p_win"], errors="coerce") + 0.5 * pd.to_numeric(
        out["p_draw"], errors="coerce"
    )
    if "attack_strength" not in out.columns:
        out["attack_strength"] = np.nan
    if "defend_threat" not in out.columns:
        out["defend_threat"] = np.nan
    return out


def build_season(season: str, code: str, protocol: dict[str, Any]) -> pd.DataFrame:
    if season not in CLOSED_SEASONS:
        raise RuntimeError(f"{season} is outside the closed-season comparison")
    players = load_player_logs(season=season)
    fixtures = load_football_data(code=code)
    joined, _, _stats = join_players_to_fixtures(
        players, fixtures, retain_sheet_rows=True
    )
    frame = _prepare(joined)
    frame = add_market_pots(frame)
    frame = add_player_priors(frame)
    frame = compute_xp(frame)
    frame["season"] = season
    frame["score_xp"] = frame["xp"]
    frame["score_exp_points"] = frame["exp_points"]
    frame["score_official_xp"] = frame["official_xp"]
    frame["eligible"] = eligible_mask(
        frame,
        min_history=int(protocol["min_history"]),
        min_xmi=float(protocol["min_xmi"]),
    )
    return frame


def _window(frame: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    gw = pd.to_numeric(frame["gw"], errors="coerce")
    return frame.loc[(gw >= int(protocol["gw_start"])) & (gw <= int(protocol["gw_end"]))].copy()


def logscore_rows(frame: pd.DataFrame, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per season, gameweek, and pre-registered pair. Pair on finite scores."""
    window = _window(frame, protocol)
    sigma = float(protocol["sigma"])
    rows: list[dict[str, Any]] = []
    season = str(window["season"].iloc[0])
    filled = usable_gameweeks(window)
    for left, right in COMPARISONS:
        view = window
        if left in OFFICIAL_XP_COLUMNS or right in OFFICIAL_XP_COLUMNS:
            assert_predeadline_xp(window, load_deadlines())
            view = window.loc[window["gw"].isin(filled)]
        a = pd.to_numeric(view[left], errors="coerce")
        b = pd.to_numeric(view[right], errors="coerce")
        yy = pd.to_numeric(view["total_points"], errors="coerce")
        ok = yy.notna() & a.notna() & b.notna()
        paired = view.loc[ok].copy()
        if paired.empty:
            continue
        paired["_y"] = yy.loc[ok].to_numpy(float)
        paired["_a"] = a.loc[ok].to_numpy(float)
        paired["_b"] = b.loc[ok].to_numpy(float)
        for gw, block in paired.groupby("gw", sort=True):
            log_a = gaussian_log_score(block["_y"].to_numpy(float), block["_a"].to_numpy(float), sigma)
            log_b = gaussian_log_score(block["_y"].to_numpy(float), block["_b"].to_numpy(float), sigma)
            rows.append(
                {
                    "season": season,
                    "gw": int(gw),
                    "comparison": comparison_key(left, right),
                    "n_rows": int(len(block)),
                    "mean_log_a": float(log_a.mean()),
                    "mean_log_b": float(log_b.mean()),
                    "delta": float((log_a - log_b).mean()),
                }
            )
    return rows


def _one_player(gw: pd.DataFrame) -> pd.DataFrame:
    """Sum a double gameweek onto the earlier row so the XI names each player once."""
    ordered = gw.sort_values(["date", "fixture_id"], kind="mergesort")
    base = ordered.groupby("player_id", as_index=False).first()
    sums = ordered.groupby("player_id", as_index=False)[
        ["total_points", "score_xp", "score_exp_points"]
    ].sum()
    keep = [col for col in base.columns if col not in sums.columns or col == "player_id"]
    return base[keep].merge(sums, on="player_id", how="left")


def _xi_total(pool: pd.DataFrame, score_col: str) -> float | None:
    if pool.empty or score_col not in pool.columns:
        return None
    usable = pool.loc[pd.to_numeric(pool[score_col], errors="coerce").notna()].copy()
    if usable.empty:
        return None
    try:
        sel, _form = pick_xi(usable, score_col)
    except RuntimeError:
        return None
    points = pd.to_numeric(sel["total_points"], errors="coerce").fillna(0.0)
    return float(points.sum() + points.max())


def xi_rows(frame: pd.DataFrame, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Stripped XI points. A week that one score cannot fill is dropped from all three."""
    window = _window(frame, protocol)
    window = window.loc[window["eligible"]].copy()
    rows: list[dict[str, Any]] = []
    season = str(frame["season"].iloc[0])
    for gw, block in window.groupby("gw", sort=True):
        pool = _one_player(block)
        wanted = ("score_xp", "score_exp_points")
        totals = {col: _xi_total(pool, col) for col in wanted}
        if any(value is None for value in totals.values()):
            continue
        row: dict[str, Any] = {"season": season, "gw": int(gw)}
        row.update(totals)
        rows.append(row)
    return rows


def _pool_intervals(
    rows: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    value: str,
    key_col: str | None = "comparison",
    seasons: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Pool seasons that clear the gameweek floor. A short season stays out of the interval."""
    if key_col is None:
        rows = rows.copy()
        rows["_key"] = "engine"
        key_col = "_key"
    seasons = seasons or tuple(protocol["closed_seasons"])
    minimum = int(protocol["min_gws_per_season"])
    pooled: dict[str, Any] = {}
    keys = list(dict.fromkeys(rows[key_col].tolist()))
    for key in keys:
        block = rows.loc[rows[key_col] == key]
        complete: dict[str, np.ndarray] = {}
        incomplete: dict[str, int] = {}
        for season in seasons:
            values = block.loc[block["season"] == season, value].to_numpy(float)
            if int(values.size) >= minimum:
                complete[season] = values
            else:
                incomplete[season] = int(values.size)
        if not complete:
            raise RuntimeError(f"{key} has no season with {minimum} gameweeks")
        summary = cluster_interval(
            complete,
            seasons=tuple(complete),
            n_boot=int(protocol["bootstrap"]),
            seed=int(protocol["seed"]),
        )
        summary["incomplete_seasons"] = incomplete
        if "n_rows" in block.columns:
            kept = block.loc[block["season"].isin(complete)]
            summary["n_rows"] = int(kept["n_rows"].sum()) if len(kept) else 0
        pooled[key] = summary
    return pooled


def _fmt(row: dict[str, Any]) -> str:
    return f"{row['mean']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"


def _band(row: dict[str, Any]) -> str:
    if row["lo"] > 0:
        return "the interval stays above zero"
    if row["hi"] < 0:
        return "the interval stays below zero"
    return "the interval covers zero"


def _season_cell(
    log_rows: pd.DataFrame,
    season: str,
    key: str,
    incomplete: dict[str, dict[str, int]],
) -> str:
    block = log_rows.loc[
        (log_rows["season"] == season) & (log_rows["comparison"] == key), "delta"
    ]
    if block.empty:
        return "—"
    text = f"{float(block.mean()):+.4f}"
    left_out = incomplete.get(key) or {}
    if season in left_out:
        text += f" (n={left_out[season]}, not pooled)"
    return text


def _fill_sentence(fill: dict[str, Any]) -> str:
    missing = ", ".join(str(gw) for gw in fill["unfilled"]) or "none"
    transfers = fill.get("transfers_all_zero")
    transfer_text = "every transfers_balance is 0" if transfers else "transfers_balance is not identically 0"
    return (
        f"{fill['season']}: unfilled gameweeks {missing}. "
        f"`modified` is {fill['modified']}. "
        f"Blanks with xP at least 4: {fill['high_xp_blanks']}. "
        f"value median {fill['value_median']}, max {fill['value_max']}. "
        f"selected mean {float(fill['selected_mean']):.0f}. {transfer_text}."
    )


def assert_procedure(intervals: dict[str, Any]) -> None:
    """The procedure runner refuses a survival call and a scraped-xP comparison."""
    enc = intervals.get("encompassing") or {}
    if enc.get("status") != "held" or enc.get("survives") is not None:
        raise RuntimeError("refusing report: the stop-forecasting decision is not held")
    if int(enc.get("n_gws") or 0) > 0:
        raise RuntimeError("refusing report: played pre-deadline weeks are not in this batch")
    slices = intervals.get("slices") or {}
    for name in ("spearman", "top15"):
        block = slices.get(name) or {}
        for left, right in COMPARISONS:
            key = comparison_key(left, right)
            row = block.get(key)
            if not row:
                raise RuntimeError(f"refusing report: missing {name} interval for {key}")
            for field in ("mean", "lo", "hi"):
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"refusing report: {name} {key} has no finite {field}")


def _report_lines(
    intervals: dict[str, Any],
    log_rows: pd.DataFrame,
    xi: pd.DataFrame,
    fills: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "# Procedure audit",
        "",
        "One pre-registered batch. The gate is the player-GW Gaussian log score "
        "with σ = 3, fixed before the totals were read. The cluster is a gameweek. "
        "Gameweeks are resampled inside each season, then pooled. Player rows inside "
        "a gameweek are not resampled. B = 1000, seed 0, 95% interval. Gameweeks 5–38. "
        "The certified comparison is score_xp against the shift-1 mean of points. "
        "Scraped official xP is not in this table.",
        "",
        "## Likelihood",
        "",
        "A positive mean is a higher log score for the first column.",
        "",
        "| comparison | mean [95% interval] | player-GW rows |",
        "|---|---:|---:|",
    ]
    for left, right in COMPARISONS:
        key = comparison_key(left, right)
        row = intervals["comparisons"][key]
        lines.append(f"| {left} − {right} | {_fmt(row)} | {row['n_rows']} |")
    lines += [
        "",
        "The same delta by season. Each cell is the mean of that season's gameweek deltas.",
        "",
        "| season | score_xp − exp |",
        "|---|---:|",
    ]
    order = [comparison_key(left, right) for left, right in COMPARISONS]
    incomplete = {
        key: (intervals["comparisons"][key].get("incomplete_seasons") or {})
        for key in order
    }
    for season in intervals["seasons"]:
        cells = [_season_cell(log_rows, season, key, incomplete) for key in order]
        lines.append(f"| {season} | " + " | ".join(cells) + " |")
    exp_key = comparison_key("score_xp", "score_exp_points")
    for season in intervals["seasons"]:
        present = set(
            log_rows.loc[
                (log_rows["season"] == season) & (log_rows["comparison"] == exp_key), "gw"
            ].astype(int)
        )
        missing = [gw for gw in range(5, 39) if gw not in present]
        if missing:
            joined = ", ".join(str(gw) for gw in missing)
            lines.append("")
            lines.append(
                f"{season} has no sheet rows in gameweek {joined} on the expected-points "
                f"comparison, so that week is absent. {len(present)} gameweeks remain."
            )
    xp_exp = intervals["comparisons"][comparison_key("score_xp", "score_exp_points")]
    lines += [
        "",
        f"On the pooled gameweeks, score_xp minus score_exp_points is {_fmt(xp_exp)}, "
        f"and {_band(xp_exp)}.",
        "",
        "The likelihood is the mean over every finite player-GW row, including "
        "0-minute rows. A Gaussian log score with σ fixed at 3 is mean squared error "
        "up to a constant, so non-starters dominate it. The rank slice below is the "
        "decision-relevant cut. It is not the eleven.",
    ]
    lines += [
        "",
        "## Scraped xP, withdrawn",
        "",
        "The Vaastav README says `xP` comes from the FPL field `ep_this`, and the "
        "scraper runs after each gameweek ends. If FPL updates `ep_this` after the "
        "matches, the scraped value contains information that was not available before "
        "the deadline. The maintainer says that cadence is not documented. Weekly "
        "updates of that repository stopped after 2024-25. A season now has three "
        "updates. `modified` is not a scrape clock. These sheets have no per-row "
        "capture time, so they fail the pre-deadline check. They are not a benchmark "
        "and not a feature. The figures below are the last run that used them. They "
        "are not a result, and they are not a reason to stop work on the forecast.",
        "",
        "Last scraped log score, not certified: score_xp minus official xP "
        "−0.0565 [−0.0680, −0.0442] on the three seasons with at least 20 filled "
        "weeks. Spearman on eligible rows −0.3487 [−0.3789, −0.3188]. Top-15 "
        "intersection +0.2174 [−0.0257, +0.4949]. Walk-forward coefficient on "
        "score_xp −0.0245 [−0.0356, −0.0139]. The −0.35 rank gap is unexplained. "
        "It is not evidence that the scrape is post-match, and it is not evidence "
        "that it is clean.",
        "",
        "Filled weeks in the cache, maximum xP greater than 0, are listed so the "
        "pattern can be checked. An all-zero week is an unfilled scrape. That "
        "description is not a score.",
        "",
    ]
    for fill in fills:
        lines.append(_fill_sentence(fill))
        lines.append("")
    clock = "; ".join(f"{fill['season']}: {fill['modified']}" for fill in fills)
    lines += [
        f"`modified` by season: {clock}. It is not a scrape clock. "
        "No historical news time was invented. The README warning is the provenance.",
        "",
        "`value` is the gameweek price in tenths. `selected` is ownership. "
        "`transfers_balance` is the transfer column on the same sheet. None of the "
        "three enters the sum inside `compute_xp`. `value` is copied to `baseline_value` "
        "and is not the engine score. A filled week still contains blanks whose official "
        "xP is at least 4, so a filled column is not a rewrite of the points.",
    ]
    lines += [
        "",
        "## Stripped XI, descriptive",
        "",
        "Eligible rows only: at least three prior sheet rows and expected minutes "
        "at least 45, both from shift-1 history that includes 0-minute weeks. "
        "The total is the eleven's points plus the highest points inside that eleven. "
        "That captain is the realised maximum, so the column is a sanity check. "
        "A week one score cannot fill is dropped from both columns. This table is not a pass.",
        "",
        "| season | weeks | score_xp | score_exp_points |",
        "|---|---:|---:|---:|",
    ]
    if xi.empty:
        lines.append("| — | 0 | — | — |")
    else:
        for season, block in xi.groupby("season", sort=False):
            lines.append(
                f"| {season} | {len(block)} | {block['score_xp'].sum():.0f} | "
                f"{block['score_exp_points'].sum():.0f} |"
            )
    if not xi.empty:
        lines += [
            "",
            "There is no budget and no transfer constraint, and the captain is the "
            "highest realised score in the eleven. These totals are not a pass.",
        ]
    lines += ["", "## Decision slice", ""]
    lines.append(
        "Eligible rows only. Within a season, gameweek, and position, the rank delta is "
        "Spearman(score, points) for the first column minus the same rank for the second. "
        "A position with fewer than three finite rows, or a constant score, is skipped. "
        "The top 15 by each score are paired on the players both lists name, when that "
        "intersection has at least five players and the position has at least fifteen "
        "eligible rows. The top-15 delta is the paired Gaussian log score on that intersection. "
        "A gameweek delta is the mean across positions. The bootstrap still resamples gameweeks."
    )
    lines += [
        "",
        "| slice | comparison | mean [95% interval] | pooled gameweeks | left out |",
        "|---|---|---:|---|---|",
    ]
    for name in ("spearman", "top15"):
        block = intervals["slices"][name]
        for left, right in COMPARISONS:
            key = comparison_key(left, right)
            row = block[key]
            pooled = ", ".join(f"{season} {count}" for season, count in row["n_gws"].items())
            left_out = row.get("incomplete_seasons") or {}
            omitted = ", ".join(f"{season} {count}" for season, count in left_out.items()) or "—"
            lines.append(
                f"| {name} | {left} − {right} | {_fmt(row)} | {pooled} | {omitted} |"
            )
    rank_exp = intervals["slices"]["spearman"][comparison_key("score_xp", "score_exp_points")]
    lines += [
        "",
        f"On the rank slice, score_xp minus score_exp_points is {_fmt(rank_exp)}, "
        f"and {_band(rank_exp)}. Scraped official xP is not in this slice.",
    ]
    enc = intervals["encompassing"]
    engine_notes = []
    for capture in enc.get("engine_captures") or []:
        committed = capture.get("git_commit_at") or "not committed"
        engine_notes.append(f"{capture['path']} (git {committed})")
    official_notes = ", ".join(enc.get("official_captures") or []) or "none"
    lines += [
        "",
        "## Live encompassing test",
        "",
        "The formula is unchanged: points on an intercept, official xP, and score_xp. "
        "Both inputs have to be pre-deadline captures, and the points have to be the "
        "played gameweek. Eligible weeks start at 2026-27 gameweek 6. A survival call "
        "needs 20 such gameweeks. This batch does not make that call.",
        "",
        f"Status: {enc['status']}. Evaluable gameweeks: {enc['n_gws']}. "
        f"Outcomes in the 2026-27 log run through gameweek {enc['outcomes_through_gw']}. "
        f"Engine captures: {', '.join(engine_notes) or 'none'}. "
        f"Official captures: {official_notes}. {enc['reason']} "
        "score_xp stays the forecast.",
    ]
    lines += [
        "",
        "## What was wrong, and what changed",
        "",
        "The buy pool had been the players who played the gameweek being scored. "
        "`load_player_logs` dropped `minutes <= 0`, and `build_frames` did the same "
        "before the priors. A purchase was a player who had already played. "
        "A sheet row now stays whether the minutes are 0 or not. A row that misses "
        "its fixture stays, with an empty market cell. Expected minutes, expected "
        "points, the three-week roll, expected xG, expected xA, and the defensive-"
        "contribution rate stay shift-1, and the shift includes the 0-minute weeks. "
        "A missing player prior is not filled from that season's position mean. "
        "The row is not eligible. A team with no shifted history takes xG 1.40 and "
        "xA 1.05. Those two numbers are fixed. They are not a mean of the season "
        "being scored. Eligible means at least three prior sheet rows and expected "
        "minutes at least 45. The current week's minutes, points, and the fact that "
        "a positive-minute row exists do not enter that flag. A 0-minute row with a "
        "high prior is eligible and scores 0 if it is picked. A player who appears "
        "for the first time in the scored week is not eligible.",
        "",
        "Closing odds had been preferred to opening odds. `AvgCH` of 1.2 with `AvgH` "
        "of 3.0 now uses 3.0. The 1X2 group is Avg, then B365, then Pinnacle. The "
        "2.5 total is Avg, then B365, and never `AvgC>2.5`. The asian line is `AHh`, "
        "with opening asian prices. A fixture whose opening 1X2 is missing is dropped "
        "even when the closing price is present.",
        "",
        "The season-total bar of +34 is retired. One season's paired noise was larger "
        "than that bar, and every finished season had already been used as a screen. "
        "The interval above is the comparison. A climb total is not a winner. "
        "`experiments/matrix.json` no longer carries `pass_margin`, and "
        "`search_protocol` does not label a season total as a winner.",
        "",
        "Vaastav `xP` is stored as `official_xp` and is not an input to `compute_xp`. "
        "It is not a pre-deadline forecast. The README says the scraper runs after "
        "the gameweek. `total_points` stays the outcome. `exp_points` remains the "
        "expanding mean of past points.",
        "",
        "Stage 13's information-coefficient gate failed: xP trailed expected points "
        "at horizon 8. Stage 18's ridge tied xP under the budget, a difference of "
        "−2. Both were set aside because the stripped climb was treated as the gate. "
        "That climb is the comparison most exposed to the played-only pool. They "
        "stay set aside. This batch does not reopen them as a pass.",
        "",
        "The goalkeeper goal inside `compute_xp` now reads `GOAL_POINTS` from "
        "`src/rules/fpl_2026.py`, so a goalkeeper goal is 6. Historical "
        "`total_points` are not rescaled. The default formation list is "
        "`OFFICIAL_FORMATIONS`, which includes 5-2-3. A tie keeps the earlier shape. "
        "The harness in `src/models/` was left in place. The leak is closed at "
        "`load_player_logs`, the priors, and `build_frames`, and a result has one "
        "writer, `write_gated_report`. Pytest and ruff are declared. The workflow "
        "runs the gate tests. The full player-GW frame is not committed. The "
        "committed evidence is the gameweek log-score table.",
        "",
        "Historical sheets have no chance of playing. Availability on this comparison "
        "is the minutes history, including zeros. Live news tags stay on the live "
        "path. No historical news tag was invented.",
        "",
        "## Rules now in force",
        "",
        "A result is not reported because a review sentence says so. The as-of audit "
        "has to pass, and each closed season is either inside a paired interval or "
        "marked incomplete with fewer than 20 gameweeks. `write_gated_report` raises "
        "otherwise. This runner also refuses the file when the rank slice or the "
        "encompassing coefficient is missing. Gemini may still be asked about a formula. "
        "That exchange is not the certificate.",
        "",
        "Stages 14–34, and the later climbs on the same played-only pool, are superseded. "
        "Their totals stay in `reports/` as the record of those questions. They are not "
        "the current comparison. See `reports/SUPERSEDED.md`. The transfer climb was not "
        "rerun. The published Gameweeks 1–5 total of 280 used the played-only pool and "
        "the shorter formation list. It was not recomputed, and it is not restated here "
        "as a new measurement.",
        "",
        "Pull requests 53, 54, 55, and 56 stay open as finished counts on the stack. "
        "They are not extended. A new chip simulator stays deferred. They are not merged "
        "ahead of these gates. This checkout cannot merge the stack onto `main`. That "
        "merge is a PI action on GitHub. The default comparison is this gated path.",
        "",
        "The live 2026/27 holdout is frozen. Gameweeks 1–5 of that season were already "
        "read by the chip-hurdle and horizon trials, so they are not a clean holdout. "
        "The clean holdout starts at gameweek 6. `data/live/HOLDOUT_FREEZE.json` holds "
        "the snapshot hashes. A hash of a file that keeps growing does not freeze a "
        "future week. Pre-deadline forecasts are timestamped files under "
        "`data/predictions/`, and an existing timestamp file is not overwritten. "
        "Those snapshots were not rewritten, and no parameter was fit on 2026-27. "
        "No Odds API call was made. This comparison does not include 2026-27.",
        "",
        "- `data/processed/player_gw_logscore.csv`",
        "- `data/processed/player_gw_slices.csv`",
        "- `data/processed/player_gw_xi_sanity.csv`",
        "- `experiments/protocol.json`",
        "",
    ]
    return lines


def run() -> dict[str, Any]:
    protocol = load_protocol()
    audit = run_asof_audit()
    if not audit["passed"]:
        raise RuntimeError("as-of audit failed: " + "; ".join(audit["failures"]))
    collected: list[dict[str, Any]] = []
    xi_collected: list[dict[str, Any]] = []
    slice_collected: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    for season in protocol["closed_seasons"]:
        code = protocol["season_codes"][season]
        print(f"Building {season}…", flush=True)
        fills.append(sheet_fill(season))
        frame = build_season(season, code, protocol)
        collected.extend(logscore_rows(frame, protocol))
        xi_collected.extend(xi_rows(frame, protocol))
        slice_collected.extend(slice_rows(frame, protocol))
        print(f"  sheet rows {len(frame)}", flush=True)
        del frame
    log_rows = pd.DataFrame(collected)
    xi = pd.DataFrame(xi_collected)
    slices = pd.DataFrame(slice_collected)
    if slices.empty:
        raise RuntimeError("the decision slice is empty")
    comparisons = _pool_intervals(log_rows, protocol, value="delta")
    slice_intervals = {
        name: _pool_intervals(slices.loc[slices["slice"] == name], protocol, value="delta")
        for name in ("spearman", "top15")
    }
    encompassing = live_encompassing()
    intervals = {
        "seasons": list(protocol["closed_seasons"]),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
        "slices": slice_intervals,
        "encompassing": encompassing,
        "official_xp_fill": fills,
    }
    assert_procedure(intervals)
    lines = _report_lines(intervals, log_rows, xi, fills)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    write_gated_report(REPORTS / "procedure_audit.md", audit, intervals, lines)
    log_rows.to_csv(LOGSCORE_CSV, index=False)
    slices.to_csv(SLICE_CSV, index=False)
    xi.to_csv(XI_CSV, index=False)
    (PROCESSED / "procedure_intervals.json").write_text(
        json.dumps(intervals, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {REPORTS / 'procedure_audit.md'}", flush=True)
    return {"audit": audit, "intervals": intervals, "xi": xi}


if __name__ == "__main__":
    run()
