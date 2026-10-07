"""Week-by-week case study for five 2024-25 players. Not a new interval.

The delta is the baseline at the next week minus points in the previous week.
Players are chosen from ownership, not from points. score_xp is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.reversion import player_gameweeks, sheet_predecessor

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "cache"
SEASON = "2024-25"
PERCENTILES = (10, 30, 50, 70, 90)
MIN_WEEKS = 8

REQUIRED = (
    "This table is a case study of five players in 2024-25. It is not a population interval.",
    "The delta is the baseline at the next week minus the points scored in the previous week.",
    "Minutes in the forecast week are not a filter.",
    "No reversion term is added to score_xp.",
    "does not sample the extreme ownership tail.",
)
FORBIDDEN = (
    "establishes a winner",
    "requires an empirical mean-reversion dampener",
    "population estimate",
)


def manager_percent(selected: pd.Series, gw: pd.Series) -> pd.Series:
    """Percent of squads. Each squad holds 15 players, so the share is scaled by 15."""
    counts = pd.to_numeric(selected, errors="coerce")
    totals = counts.groupby(gw).transform("sum")
    if totals.isna().any() or (totals <= 0).any():
        raise RuntimeError("ownership denominator is not positive")
    return 100.0 * 15.0 * counts / totals


def _nearest(frame: pd.DataFrame, percentiles: tuple[int, ...]) -> pd.DataFrame:
    """Players nearest each percentile of median ownership. Ties take the lower id."""
    ordered = frame.sort_values(["own_median", "player_id"]).reset_index(drop=True)
    values = ordered["own_median"].to_numpy(float)
    chosen: list[pd.Series] = []
    used: set[str] = set()
    for pct in percentiles:
        target = float(np.percentile(values, pct))
        rest = ordered.loc[~ordered["player_id"].isin(used)].copy()
        rest["distance"] = (rest["own_median"] - target).abs()
        pick = rest.sort_values(["distance", "player_id"]).iloc[0]
        chosen.append(pick)
        used.add(str(pick["player_id"]))
    return pd.DataFrame(chosen)


def _sheet(panel: pd.DataFrame) -> list[int]:
    return sorted(int(gw) for gw in panel["gw"].unique())


def qualifying_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Forecast weeks whose previous week was a single fixture of exactly 90 minutes."""
    gameweeks = _sheet(panel)
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
        if int(previous["n_fix"]) != 1 or float(previous["minutes"]) != 90.0:
            continue
        row: dict[str, Any] = {
            "gw": gw,
            "player_id": str(record.player_id),
            "actual_prev": float(previous["actual"]),
            "exp": float(record.exp),
            "xp": float(record.xp),
            "exp_minus_actual": float(record.exp) - float(previous["actual"]),
            "xp_minus_actual": float(record.xp) - float(previous["actual"]),
            "exp_minus_mean3": float("nan"),
            "xp_minus_mean3": float("nan"),
        }
        priors = [sheet_predecessor(gameweeks, gw, step) for step in (1, 2, 3)]
        if all(item is not None for item in priors):
            actuals: list[float] = []
            complete = True
            for pred_gw in priors:
                assert pred_gw is not None
                pred_key = (pred_gw, str(record.player_id))
                if pred_key not in indexed.index:
                    complete = False
                    break
                pred = indexed.loc[pred_key]
                if int(pred["n_fix"]) != 1 or float(pred["minutes"]) != 90.0:
                    complete = False
                    break
                actuals.append(float(pred["actual"]))
            if complete:
                mean_actual = float(np.mean(actuals))
                row["exp_minus_mean3"] = float(record.exp) - mean_actual
                row["xp_minus_mean3"] = float(record.xp) - mean_actual
        rows.append(row)
    return pd.DataFrame(rows)


def _ownership() -> pd.DataFrame:
    raw = pd.read_csv(
        CACHE / "merged_gw_2024_25.csv",
        usecols=["element", "GW", "selected", "name"],
    )
    raw["player_id"] = raw["element"].astype(str)
    raw["gw"] = pd.to_numeric(raw["GW"], errors="coerce")
    raw["selected"] = pd.to_numeric(raw["selected"], errors="coerce")
    raw = raw.dropna(subset=["gw", "selected"])
    raw["gw"] = raw["gw"].astype(int)
    raw = raw.sort_values(["player_id", "gw"]).drop_duplicates(["player_id", "gw"], keep="first")
    raw["own_pct"] = manager_percent(raw["selected"], raw["gw"])
    names = raw.drop_duplicates("player_id")[["player_id", "name"]]
    return raw[["player_id", "gw", "own_pct"]].merge(names, on="player_id", how="left")


def _fdr() -> pd.DataFrame:
    """Official 1–5 difficulty for the player's side. Doubles are left blank."""
    fixtures = pd.read_csv(PROCESSED / "fdr_2024_25.csv").drop_duplicates("id")
    difficulty = fixtures.set_index("id")
    raw = pd.read_csv(
        CACHE / "merged_gw_2024_25.csv",
        usecols=["element", "GW", "fixture", "was_home"],
    )
    raw["player_id"] = raw["element"].astype(str)
    raw["gw"] = pd.to_numeric(raw["GW"], errors="coerce")
    raw["fixture"] = pd.to_numeric(raw["fixture"], errors="coerce")
    raw = raw.dropna(subset=["gw", "fixture"])
    raw["gw"] = raw["gw"].astype(int)
    counts = raw.groupby(["player_id", "gw"]).size()
    raw = raw.drop_duplicates(["player_id", "gw"], keep="first")
    raw["n_fix"] = [
        int(counts.loc[(player_id, gw)])
        for player_id, gw in zip(raw["player_id"], raw["gw"], strict=True)
    ]
    home = raw["was_home"].astype(str).str.lower().isin({"true", "1"})
    home_fdr = raw["fixture"].astype(int).map(difficulty["team_h_difficulty"])
    away_fdr = raw["fixture"].astype(int).map(difficulty["team_a_difficulty"])
    raw["fdr"] = np.where(home, home_fdr, away_fdr)
    raw.loc[raw["n_fix"] != 1, "fdr"] = np.nan
    return raw[["player_id", "gw", "fdr"]]


def ownership_pool(qualifying: pd.DataFrame, ownership: pd.DataFrame) -> pd.DataFrame:
    """One row per player with at least eight qualifying weeks. Points are not used."""
    joined = qualifying.merge(ownership, on=["player_id", "gw"], how="left")
    counts = joined.groupby("player_id").size()
    eligible = counts[counts >= MIN_WEEKS].index
    pool = joined.loc[joined["player_id"].isin(eligible)]
    summary = (
        pool.groupby("player_id", sort=False)
        .agg(own_median=("own_pct", "median"), name=("name", "first"), weeks=("gw", "size"))
        .reset_index()
    )
    return summary.dropna(subset=["own_median"])


def choose_players(qualifying: pd.DataFrame, ownership: pd.DataFrame) -> pd.DataFrame:
    """Five players by ownership percentile. Points are not an argument."""
    summary = ownership_pool(qualifying, ownership)
    if len(summary) < len(PERCENTILES):
        raise RuntimeError("fewer than five players have eight 90-minute weeks")
    return _nearest(summary, PERCENTILES)


def _tail_sentence(summary: pd.DataFrame, high: pd.Series) -> str:
    ordered = summary.sort_values(["own_median", "player_id"])
    values = ordered["own_median"].to_numpy(float)
    top = ordered.iloc[-1]
    return (
        f"In the qualifying pool of {len(ordered)} players, the maximum median ownership "
        f"is {top['name']} at {float(top['own_median']):.1f}% "
        f"(with the 95th percentile at {float(np.percentile(values, 95)):.1f}% "
        f"and 99th at {float(np.percentile(values, 99)):.1f}%), "
        f"well above the 90th percentile ({high['name']}, {float(high['own_median']):.1f}%); "
        "the case study strictly follows the pre-registered percentile lattice "
        "and does not sample the extreme ownership tail."
    )


def _cell(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}f}"


def _mean_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gw, block in frame.groupby("gw", sort=True):
        def _avg(column: str) -> float:
            values = pd.to_numeric(block[column], errors="coerce").dropna()
            if values.empty:
                return float("nan")
            return float(values.mean())

        rows.append(
            {
                "gw": int(gw),
                "n": int(len(block)),
                "n3": int(pd.to_numeric(block["exp_minus_mean3"], errors="coerce").notna().sum()),
                "exp": _avg("exp"),
                "exp_minus_actual": _avg("exp_minus_actual"),
                "xp": _avg("xp"),
                "xp_minus_actual": _avg("xp_minus_actual"),
                "exp_minus_mean3": _avg("exp_minus_mean3"),
                "xp_minus_mean3": _avg("xp_minus_mean3"),
                "fdr": _avg("fdr"),
            }
        )
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame, *, with_n: bool) -> list[str]:
    if with_n:
        header = (
            "| GW(t+1) | N | N3 | exp_points(t+1) | exp_points(t+1) − actual(t) | "
            "score_xp(t+1) | score_xp(t+1) − actual(t) | "
            "exp_points(t+1) − mean(t−2:t) | score_xp(t+1) − mean(t−2:t) | FDR |"
        )
        rule = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = (
            "| GW(t+1) | exp_points(t+1) | exp_points(t+1) − actual(t) | "
            "score_xp(t+1) | score_xp(t+1) − actual(t) | "
            "exp_points(t+1) − mean(t−2:t) | score_xp(t+1) − mean(t−2:t) | FDR |"
        )
        rule = "|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, rule]
    for row in frame.itertuples(index=False):
        cells = [
            str(int(row.gw)),
            _cell(row.exp),
            _cell(row.exp_minus_actual),
            _cell(row.xp),
            _cell(row.xp_minus_actual),
            _cell(row.exp_minus_mean3),
            _cell(row.xp_minus_mean3),
            _cell(row.fdr, 1),
        ]
        if with_n:
            cells.insert(1, str(int(row.n3)))
            cells.insert(1, str(int(row.n)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _lines(
    chosen: pd.DataFrame,
    average: pd.DataFrame,
    single: pd.DataFrame,
    focus: pd.Series,
    tail: str,
) -> list[str]:
    roster = []
    for row in chosen.itertuples(index=False):
        roster.append(
            f"| {row.name} | {row.player_id} | {int(row.weeks)} | {float(row.own_median):.1f} |"
        )
    focus_name = str(focus["name"])
    return [
        "# Ninety-minute case study, 2024-25",
        "",
        "This table is a case study of five players in 2024-25. It is not a population interval.",
        "",
        "The delta is the baseline at the next week minus the points scored in the previous week.",
        "Minutes in the forecast week are not a filter.",
        "The previous week is a single fixture of exactly 90 minutes. A double is excluded.",
        "The three-week columns are blank unless each of those three weeks was also 90 minutes.",
        "Ownership percent is 15 times selected, divided by the gameweek sum of selected.",
        "The five players are the nearest to the 10th, 30th, 50th, 70th, and 90th percentiles "
        "of median ownership among players with at least eight such weeks. Ties take the lower id. "
        "Points were not used to choose them.",
        tail,
        "FDR is the official 1–5 difficulty of the player's side in the forecast week.",
        "",
        "No reversion term is added to score_xp.",
        "",
        "| player | id | 90-minute weeks | median ownership % |",
        "|---|---:|---:|---:|",
        *roster,
        "",
        "## Average of the five",
        "",
        "N is how many of the five had a 90-minute previous week. "
        "N3 is how many of those rows also have three preceding 90-minute weeks. "
        "The three-week mean uses only those rows.",
        "",
        *_markdown(average, with_n=True),
        "",
        f"## {focus_name}",
        "",
        f"{focus_name} is the player nearest the 50th percentile of median ownership, "
        f"at {float(focus['own_median']):.1f} percent.",
        "",
        *_markdown(single, with_n=False),
        "",
        "Gemini kept the display rules and reviewed the filled table "
        "([ninety-minute table](bc-e75c8209-ffd3-590a-b610-d0e34e62bbee)).",
        "",
    ]


def run() -> None:
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season

    protocol = load_protocol()
    if protocol["reversion"].get("installs_feature") or protocol["reversion"].get("winner") is not None:
        raise RuntimeError("the reversion lock edits the score")
    code = protocol["season_codes"][SEASON]
    print(f"case study {SEASON}", flush=True)
    frame = build_season(SEASON, code, protocol)
    panel = player_gameweeks(frame, SEASON)
    qualifying = qualifying_rows(panel)
    if qualifying.empty:
        raise RuntimeError("no 90-minute pairs")
    ownership = _ownership()
    difficulty = _fdr()
    pool = ownership_pool(qualifying, ownership)
    chosen = choose_players(qualifying, ownership)
    ids = set(chosen["player_id"].astype(str))
    focus = chosen.iloc[PERCENTILES.index(50)]
    high = chosen.iloc[PERCENTILES.index(90)]
    tail = _tail_sentence(pool, high)
    detail = qualifying.loc[qualifying["player_id"].isin(ids)].merge(
        difficulty, on=["player_id", "gw"], how="left"
    )
    if detail["fdr"].isna().all():
        raise RuntimeError("FDR did not join")
    average = _mean_table(detail)
    single = (
        detail.loc[detail["player_id"] == str(focus["player_id"])]
        .sort_values("gw")
        .reset_index(drop=True)
    )
    lines = _lines(chosen, average, single, focus, tail)
    text = "\n".join(lines)
    for sentence in REQUIRED:
        if sentence not in text:
            raise RuntimeError("a required sentence is missing")
    for banned in FORBIDDEN:
        if banned in text:
            raise RuntimeError(f"the report contains a banned claim: {banned}")
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    audit = run_asof_audit()
    detail.to_csv(PROCESSED / "reversion_players_2024_25.csv", index=False)
    write_gated_report(
        REPORTS / "reversion_players.md",
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
