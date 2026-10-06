"""Player-GW log scores on the repaired pool. Four closed seasons. No 2026/27 rows.

The likelihood is the gate. The stripped XI is a sanity check and is not a pass mark.
This module does not write the player-GW frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    y = pd.to_numeric(window["total_points"], errors="coerce")
    rows: list[dict[str, Any]] = []
    season = str(window["season"].iloc[0])
    for left, right in COMPARISONS:
        a = pd.to_numeric(window[left], errors="coerce")
        b = pd.to_numeric(window[right], errors="coerce")
        ok = y.notna() & a.notna() & b.notna()
        paired = window.loc[ok].copy()
        if paired.empty:
            continue
        paired["_y"] = y.loc[ok].to_numpy(float)
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
    sums = (
        ordered.groupby("player_id", as_index=False)[
            ["total_points", "score_xp", "score_exp_points", "score_official_xp"]
        ]
        .sum()
    )
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
        wanted = ("score_xp", "score_exp_points", "score_official_xp")
        totals = {col: _xi_total(pool, col) for col in wanted}
        if any(value is None for value in totals.values()):
            continue
        row: dict[str, Any] = {"season": season, "gw": int(gw)}
        row.update(totals)
        rows.append(row)
    return rows


def _intervals(log_rows: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    seasons = tuple(protocol["closed_seasons"])
    comps: dict[str, Any] = {}
    for left, right in COMPARISONS:
        key = comparison_key(left, right)
        block = log_rows.loc[log_rows["comparison"] == key]
        by_season = {
            season: block.loc[block["season"] == season, "delta"].to_numpy(float)
            for season in seasons
        }
        summary = cluster_interval(
            by_season,
            seasons=seasons,
            n_boot=int(protocol["bootstrap"]),
            seed=int(protocol["seed"]),
        )
        summary["n_rows"] = int(block["n_rows"].sum())
        comps[key] = summary
    return {
        "seasons": list(seasons),
        "min_gws": int(protocol["min_gws_per_season"]),
        "sigma": float(protocol["sigma"]),
        "bootstrap": int(protocol["bootstrap"]),
        "seed": int(protocol["seed"]),
        "comparisons": comps,
    }


def _fmt(row: dict[str, Any]) -> str:
    return f"{row['mean']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"


def _report_lines(
    intervals: dict[str, Any],
    log_rows: pd.DataFrame,
    xi: pd.DataFrame,
) -> list[str]:
    lines = [
        "# Procedure audit",
        "",
        "One pre-registered batch. The gate is the player-GW Gaussian log score "
        "with σ = 3, fixed before the totals were read. The cluster is a gameweek. "
        "Gameweeks are resampled inside each season, then pooled. B = 1000, seed 0, "
        "95% interval. Gameweeks 5–38.",
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
        "Gameweeks per season:",
        "",
    ]
    counts = log_rows.groupby(["comparison", "season"])["gw"].nunique()
    for key, season in counts.index:
        lines.append(f"- {key}, {season}: {int(counts.loc[(key, season)])} gameweeks")
    lines += [
        "",
        "## Stripped XI, descriptive",
        "",
        "Eligible rows only: at least three prior sheet rows and expected minutes "
        "at least 45, both from shift-1 history that includes 0-minute weeks. "
        "The total is the eleven's points plus the highest points inside that eleven. "
        "That captain is the realised maximum, so the column is a sanity check. "
        "A week one score cannot fill is dropped from all three. This table is not a pass.",
        "",
        "| season | weeks | score_xp | score_exp_points | score_official_xp |",
        "|---|---:|---:|---:|---:|",
    ]
    if xi.empty:
        lines.append("| — | 0 | — | — | — |")
    else:
        for season, block in xi.groupby("season", sort=False):
            lines.append(
                f"| {season} | {len(block)} | {block['score_xp'].sum():.0f} | "
                f"{block['score_exp_points'].sum():.0f} | {block['score_official_xp'].sum():.0f} |"
            )
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
        "Official xP is the Vaastav `xP` column on the same row, stored as "
        "`official_xp` and scored as `score_official_xp`. It is that week's "
        "pre-deadline forecast. It is not shifted, and it is not an input to "
        "`compute_xp`. `total_points` stays the outcome. `exp_points` remains the "
        "expanding mean of past points. It is a different baseline.",
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
        "has to pass, and the paired intervals have to exist for all four closed "
        "seasons. `write_gated_report` raises otherwise. Gemini may still be asked "
        "about a formula. That exchange is not the certificate.",
        "",
        "The next batch is one pre-registered comparison on this protocol, judged on "
        "the player-GW likelihood. Objective-function search (Sharpe, ownership, "
        "churn) stays parked. A captain model and a chip simulator stay deferred. "
        "A climb is a sanity check.",
        "",
        "The live 2026/27 holdout is frozen. `data/live/HOLDOUT_FREEZE.json` holds "
        "the hashes. Those snapshots were not rewritten, and no parameter was fit on "
        "that season. No Odds API call was made. This comparison does not include "
        "2026-27.",
        "",
        "The published Gameweeks 1–5 total of 280 used the played-only pool and the "
        "shorter formation list. It was not recomputed on the repaired pool, and it "
        "is not restated here as a new measurement.",
        "",
        "- `data/processed/player_gw_logscore.csv`",
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
    for season in protocol["closed_seasons"]:
        code = protocol["season_codes"][season]
        print(f"Building {season}…", flush=True)
        frame = build_season(season, code, protocol)
        collected.extend(logscore_rows(frame, protocol))
        xi_collected.extend(xi_rows(frame, protocol))
        print(f"  sheet rows {len(frame)}", flush=True)
        del frame
    log_rows = pd.DataFrame(collected)
    xi = pd.DataFrame(xi_collected)
    intervals = _intervals(log_rows, protocol)
    lines = _report_lines(intervals, log_rows, xi)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    write_gated_report(REPORTS / "procedure_audit.md", audit, intervals, lines)
    log_rows.to_csv(LOGSCORE_CSV, index=False)
    xi.to_csv(XI_CSV, index=False)
    (PROCESSED / "procedure_intervals.json").write_text(
        json.dumps(intervals, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {REPORTS / 'procedure_audit.md'}", flush=True)
    return {"audit": audit, "intervals": intervals, "xi": xi}


if __name__ == "__main__":
    run()
