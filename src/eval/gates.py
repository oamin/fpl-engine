"""As-of audit and the rule that a result is not written without it.

The audit fails when a pool filter or a prior reads the gameweek being scored.
A report also needs a paired interval on every closed season in the protocol.
"""

from __future__ import annotations

import inspect
import json
import math
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ingest.fpl_odds import load_player_logs, opening_prices
from src.models.season_climb import FORMATIONS
from src.models.xp_engine import (
    NEUTRAL_TEAM_XG,
    NEUTRAL_TEAM_XA,
    add_market_pots,
    add_player_priors,
    compute_xp,
)
from src.rules.fpl_2026 import GOAL_POINTS, OFFICIAL_FORMATIONS
from src.models import xp_engine

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "experiments" / "protocol.json"

CLOSED_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
HOLDOUT_SEASON = "2026-27"
COMPARISONS = (
    ("score_xp", "score_exp_points"),
    ("score_xp", "score_official_xp"),
    ("score_official_xp", "score_exp_points"),
)


def comparison_key(left: str, right: str) -> str:
    return f"{left}_minus_{right}"


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    raw = json.loads((path or PROTOCOL_PATH).read_text(encoding="utf-8"))
    seasons = tuple(raw["closed_seasons"])
    if seasons != CLOSED_SEASONS:
        raise ValueError("protocol closed seasons must be the four finished seasons")
    if HOLDOUT_SEASON in seasons:
        raise ValueError("2026-27 is a frozen holdout and is not a closed season")
    if raw.get("holdout_season") != HOLDOUT_SEASON:
        raise ValueError("protocol holdout must be 2026-27")
    pairs = [tuple(pair) for pair in raw["comparisons"]]
    if pairs != list(COMPARISONS):
        raise ValueError("protocol comparisons do not match the pre-registered three")
    if float(raw["sigma"]) != 3.0:
        raise ValueError("sigma is pre-registered at 3.0")
    return raw


def gaussian_log_score(y: np.ndarray, mu: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian log density. Sigma is fixed. It is not estimated from the sample."""
    var = float(sigma) ** 2
    yy = np.asarray(y, dtype=float)
    mm = np.asarray(mu, dtype=float)
    return -0.5 * np.log(2.0 * np.pi * var) - (yy - mm) ** 2 / (2.0 * var)


def cluster_interval(
    deltas_by_season: dict[str, np.ndarray],
    *,
    seasons: tuple[str, ...] = CLOSED_SEASONS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean of gameweek deltas, with a bootstrap that resamples gameweeks inside each season."""
    parts = [np.asarray(deltas_by_season[season], dtype=float) for season in seasons]
    for season, arr in zip(seasons, parts, strict=True):
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"missing gameweek deltas for {season}")
    observed = float(np.concatenate(parts).mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        taken = []
        for arr in parts:
            idx = rng.integers(0, arr.size, size=arr.size)
            taken.append(arr[idx])
        boots[draw] = float(np.concatenate(taken).mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "mean": observed,
        "lo": float(lo),
        "hi": float(hi),
        "n_gws": {season: int(arr.size) for season, arr in zip(seasons, parts, strict=True)},
    }


def assert_reportable(audit: dict[str, Any], intervals: dict[str, Any]) -> None:
    """Refuse a result that lacks a passed audit or a paired interval on each closed season."""
    if not audit or not audit.get("passed"):
        detail = "; ".join(audit.get("failures", []) if audit else [])
        raise RuntimeError(f"refusing report: as-of audit has not passed ({detail})")
    seasons = list(intervals.get("seasons") or [])
    if set(seasons) != set(CLOSED_SEASONS) or HOLDOUT_SEASON in seasons:
        raise RuntimeError(
            "refusing report: paired intervals must cover the four closed seasons"
        )
    min_gws = int(intervals.get("min_gws") or 20)
    comps = intervals.get("comparisons") or {}
    for left, right in COMPARISONS:
        key = comparison_key(left, right)
        row = comps.get(key)
        if not row:
            raise RuntimeError(f"refusing report: missing interval for {key}")
        for field in ("mean", "lo", "hi"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"refusing report: {key} has no finite {field}")
        counts = row.get("n_gws") or {}
        for season in CLOSED_SEASONS:
            if int(counts.get(season, 0)) < min_gws:
                raise RuntimeError(
                    f"refusing report: {season} has fewer than {min_gws} gameweeks for {key}"
                )


def write_gated_report(
    path: Path,
    audit: dict[str, Any],
    intervals: dict[str, Any],
    lines: list[str],
) -> None:
    assert_reportable(audit, intervals)
    header = [
        "Gates: the as-of audit passed. Paired intervals cover 2022-23, 2023-24, "
        "2024-25, and 2025-26. 2026-27 is not in this comparison.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(header + lines).rstrip() + "\n", encoding="utf-8")


def _row(**kwargs: object) -> dict[str, object]:
    base: dict[str, object] = {
        "player_id": "p",
        "gw": 1,
        "date": "2024-08-17",
        "position": "MID",
        "team_norm": "arsenal",
        "fixture_id": "2024-08-17:arsenal:wolves",
        "minutes": 90.0,
        "total_points": 2.0,
        "xG": 0.1,
        "xA": 0.1,
        "defcon_hit": 0.0,
        "official_xp": 2.0,
        "value": 55.0,
        "goals": 0.0,
        "p_over": 0.5,
        "p_under": 0.5,
        "attack_strength": 0.5,
        "defend_threat": 0.5,
        "p_not_lose": 0.5,
    }
    base.update(kwargs)
    return base


def _failures_sheet() -> list[str]:
    header = (
        "name,position,team,xP,minutes,element,kickoff_time,GW,total_points,value,"
        "goals_scored,assists,clean_sheets,goals_conceded,saves,bonus,bps,"
        "expected_goals,expected_assists,yellow_cards,red_cards,starts,was_home"
    )
    blank = (
        "Blank,MID,Arsenal,1.5,0,1,2024-08-17T14:00:00Z,1,0,50,0,0,0,0,0,0,0,0,0,0,0,0,True"
    )
    played = (
        "Played,MID,Arsenal,4.0,90,2,2024-08-17T14:00:00Z,1,2,55,0,0,0,0,0,0,0,0.2,0.1,0,0,1,True"
    )
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        cache.mkdir()
        (cache / "merged_gw_2099_00.csv").write_text(
            header + "\n" + blank + "\n" + played + "\n", encoding="utf-8"
        )
        with patch("src.ingest.fpl_odds.DATA", Path(tmp)):
            frame = load_player_logs(season="2099-00")
    if len(frame) != 2 or not (frame["minutes"] <= 0).any():
        failures.append("a 0-minute sheet row was dropped from the pool")
    if "official_xp" not in frame.columns:
        failures.append("official xP was not kept on the sheet row")
    return failures


def _score_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = add_market_pots(pd.DataFrame(rows))
    return add_player_priors(frame)


def _failures_priors() -> list[str]:
    failures: list[str] = []
    history = []
    for gw, day in enumerate(range(1, 6), start=1):
        history.append(
            _row(
                player_id="regular",
                gw=gw,
                date=f"2024-08-{day:02d}",
                fixture_id=f"2024-08-{day:02d}:arsenal:wolves",
                minutes=0.0 if gw == 5 else 90.0,
                total_points=0.0 if gw == 5 else 6.0,
            )
        )
    history.append(
        _row(
            player_id="debut",
            gw=5,
            date="2024-08-05",
            fixture_id="2024-08-05:arsenal:wolves",
            minutes=90.0,
            total_points=12.0,
            xG=2.0,
        )
    )
    # A later week with a large minute count must not fill the debut prior.
    history.append(
        _row(
            player_id="debut",
            gw=6,
            date="2024-08-12",
            fixture_id="2024-08-12:arsenal:wolves",
            minutes=90.0,
            total_points=20.0,
            xG=3.0,
        )
    )
    scored = _score_frame(history)
    regular = scored.loc[scored["player_id"] == "regular"].sort_values("gw")
    week5 = regular.loc[regular["gw"] == 5].iloc[0]
    if not (int(week5["n_prior"]) >= 3 and float(week5["xmi"]) >= 45):
        failures.append("a 0-minute row with three prior appearances was not eligible")
    if abs(float(week5["xmi"]) - 90.0) > 1e-9:
        failures.append("xmi on the scored week read that week's minutes")
    leaked = [dict(row) for row in history]
    for row in leaked:
        if row["player_id"] == "regular" and row["gw"] == 5:
            row["minutes"] = 90.0
            row["total_points"] = 50.0
    again = _score_frame(leaked)
    week5b = again.loc[
        (again["player_id"] == "regular") & (again["gw"] == 5)
    ].iloc[0]
    if abs(float(week5b["xmi"]) - float(week5["xmi"])) > 1e-9:
        failures.append("changing the scored week's minutes changed xmi")
    if abs(float(week5b["exp_points"]) - float(week5["exp_points"])) > 1e-9:
        failures.append("changing the scored week's points changed exp_points")
    debut = scored.loc[(scored["player_id"] == "debut") & (scored["gw"] == 5)].iloc[0]
    if pd.notna(debut["xmi"]) or int(debut["n_prior"]) >= 3:
        failures.append("a player who only just appeared took a prior from later weeks")
    from src.eval.honest_pool import eligible_mask

    mask = eligible_mask(scored, min_history=3, min_xmi=45.0)
    scored = scored.copy()
    scored["_ok"] = mask.to_numpy()
    regular_ok = bool(
        scored.loc[(scored["player_id"] == "regular") & (scored["gw"] == 5), "_ok"].iloc[0]
    )
    debut_ok = bool(
        scored.loc[(scored["player_id"] == "debut") & (scored["gw"] == 5), "_ok"].iloc[0]
    )
    if not regular_ok:
        failures.append("eligibility rejected a blank week that already had history")
    if debut_ok:
        failures.append("eligibility accepted a same-week debut")
    neutral = _score_frame(
        [_row(player_id="solo", gw=1, minutes=90.0, xG=3.0, total_points=2.0)]
    )
    if abs(float(neutral["exp_team_xg"].iloc[0]) - NEUTRAL_TEAM_XG) > 1e-9:
        failures.append("team xG was filled from the scored season")
    if abs(float(neutral["exp_team_xa"].iloc[0]) - NEUTRAL_TEAM_XA) > 1e-9:
        failures.append("team xA was filled from the scored season")
    return failures


def _failures_odds_rules() -> list[str]:
    failures: list[str] = []
    opened = opening_prices(
        {
            "AvgCH": "1.2",
            "AvgH": "3.0",
            "AvgD": "3.4",
            "AvgA": "2.4",
            "Avg>2.5": "1.9",
            "Avg<2.5": "1.9",
            "AHh": "-0.5",
            "AHCh": "-1.5",
            "AvgAHH": "1.9",
            "AvgAHA": "1.9",
            "AvgCAHH": "2.1",
        }
    )
    if opened is None or abs(float(opened["home_odds"]) - 3.0) > 1e-9:
        failures.append("opening 1X2 did not beat a closing price")
    if opened is None or abs(float(opened["ah_line"] or 0) - (-0.5)) > 1e-9:
        failures.append("the asian line was not the opening AHh")
    if opening_prices({"AvgCH": "1.2", "AvgCD": "5", "AvgCA": "8", "AvgC>2.5": "1.8", "AvgC<2.5": "2.0"}) is not None:
        failures.append("a fixture with only closing prices stayed in the pot")
    if GOAL_POINTS["GKP"] != 6 or xp_engine.GOAL_PTS is not GOAL_POINTS:
        failures.append("xp_engine does not use the rules-module goal awards")
    source = Path(xp_engine.__file__).read_text(encoding="utf-8")
    if '"GKP": 10' in source or "'GKP': 10" in source:
        failures.append("xp_engine still contains a goalkeeper goal of 10")
    if list(FORMATIONS) != list(OFFICIAL_FORMATIONS) or (5, 2, 3) not in FORMATIONS:
        failures.append("the climb formation list is not the official list")
    if "official_xp" in inspect.getsource(compute_xp):
        failures.append("compute_xp reads official xP")
    return failures


def _failures_paths() -> list[str]:
    failures: list[str] = []
    try:
        protocol = load_protocol()
    except ValueError as exc:
        return [str(exc)]
    if protocol["holdout_season"] != HOLDOUT_SEASON:
        failures.append("holdout season is not 2026-27")
    matrix = json.loads((ROOT / "experiments" / "matrix.json").read_text(encoding="utf-8"))
    if "pass_margin" in matrix:
        failures.append("pass_margin is still a season-total bar")
    if matrix.get("holdout_season") != HOLDOUT_SEASON:
        failures.append("matrix holdout is not the frozen 2026-27 season")
    import src.live.benchmark as benchmark

    body = inspect.getsource(benchmark.build_frames)
    if 'logs["minutes"] > 0' in body or "logs['minutes'] > 0" in body:
        failures.append("build_frames still drops 0-minute rows before the join")
    if "retain_sheet_rows=True" not in body:
        failures.append("build_frames drops a sheet row that misses its fixture")
    return failures


def _failures_official_xp_unused() -> list[str]:
    rows = [
        _row(player_id="k", gw=gw, date=f"2024-08-{gw:02d}", minutes=90.0, total_points=2.0)
        for gw in range(1, 5)
    ]
    base = _score_frame(rows)
    left = base.copy()
    right = base.copy()
    left["official_xp"] = 0.0
    right["official_xp"] = 999.0
    a = compute_xp(left)["xp"].to_numpy(float)
    b = compute_xp(right)["xp"].to_numpy(float)
    if not np.allclose(a, b, equal_nan=True):
        return ["changing official xP changed compute_xp"]
    return []


def run_asof_audit() -> dict[str, Any]:
    failures: list[str] = []
    failures.extend(_failures_sheet())
    failures.extend(_failures_priors())
    failures.extend(_failures_odds_rules())
    failures.extend(_failures_paths())
    failures.extend(_failures_official_xp_unused())
    return {"passed": not failures, "failures": failures}
