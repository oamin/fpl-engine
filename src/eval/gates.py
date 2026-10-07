"""As-of audit and the rule that a result is not written without it.

The audit fails when a pool filter or a prior reads the gameweek being scored.
A report also needs a paired interval on every closed season in the protocol.
"""

from __future__ import annotations

import inspect
import json
import math
import tempfile
from datetime import datetime, timezone
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
COMPARISONS = (("score_xp", "score_exp_points"),)


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
        raise ValueError("protocol comparisons do not match the registered list")
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
    for key in comps:
        if "official_xp" in str(key) or "ep_this" in str(key) or str(key).endswith("_xP"):
            raise RuntimeError(f"refusing report: {key} uses official xP without a live capture")
    for left, right in COMPARISONS:
        key = comparison_key(left, right)
        row = comps.get(key)
        if not row:
            raise RuntimeError(f"refusing report: missing interval for {key}")
        for field in ("mean", "lo", "hi"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"refusing report: {key} has no finite {field}")
        counts = row.get("n_gws") or {}
        incomplete = row.get("incomplete_seasons") or {}
        for season in CLOSED_SEASONS:
            if season in incomplete:
                if season in counts:
                    raise RuntimeError(
                        f"refusing report: {season} is incomplete and was still pooled for {key}"
                    )
                if int(incomplete[season]) >= min_gws:
                    raise RuntimeError(
                        f"refusing report: {season} was marked incomplete with {incomplete[season]} gameweeks"
                    )
                continue
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
    if int(protocol.get("clean_holdout_from_gw") or 0) != 6:
        failures.append("the clean 2026-27 holdout does not start at gameweek 6")
    if int(protocol.get("holdout_contaminated_through_gw") or 0) != 5:
        failures.append("gameweeks 1-5 are not marked as already read")
    enc = protocol.get("encompassing") or {}
    formula = str(enc.get("formula") or "")
    if "score_xp" not in formula or "official_xp" not in formula:
        failures.append("the encompassing formula is not official xP plus the engine")
    if enc.get("decision") != "held":
        failures.append("the stop-forecasting decision is not held")
    if "pre-deadline" not in str(enc.get("population") or ""):
        failures.append("the encompassing test is not limited to pre-deadline captures")
    if int(enc.get("min_gws") or 0) != 20:
        failures.append("a survival call does not wait for 20 pre-deadline gameweeks")
    if enc.get("covers_zero") != "undetermined":
        failures.append("a 20-week interval that covers zero is not marked undetermined")
    if int(enc.get("continue_to_gw") or 0) != 38:
        failures.append("an undetermined live test does not continue through gameweek 38")
    sheet = str((protocol.get("official_xp") or {}).get("historical_sheet") or "")
    if "not a benchmark" not in sheet:
        failures.append("scraped xP is still a historical benchmark")
    import src.live.benchmark as benchmark

    body = inspect.getsource(benchmark.build_frames)
    if 'logs["minutes"] > 0' in body or "logs['minutes'] > 0" in body:
        failures.append("build_frames still drops 0-minute rows before the join")
    if "retain_sheet_rows=True" not in body:
        failures.append("build_frames drops a sheet row that misses its fixture")
    from src.eval.decision_spec import (
        CAPTAIN_BASELINE,
        HORIZON,
        LOGGED_ALONGSIDE,
        MIN_LIVE_WEEKS,
        POWER_BOOTSTRAP,
        POWER_LEVEL,
        POWER_SEED,
        POWER_SIMS,
        POWER_STEP,
        POWER_WEEKS,
        LIVE_CONTINUE_TO_GW,
        LIVE_COVERS_ZERO,
        LIVE_PRIMARY,
        LIVE_PRIMARY_FROM_GW,
        LIVE_PRIMARY_WIRED,
        LIVE_SHADOW,
        LOSO_BOOTSTRAP,
        LOSO_CONDITIONAL_FLOOR,
        LOSO_FLOOR,
        LOSO_MIN_SEASONS,
        LOSO_SEED,
        SCORE_COLUMN,
        T1_MINUTES,
        T24_HOURS,
        TEMPLATE_SLOTS,
    )

    if "decision_layer" not in protocol or "capture" not in protocol:
        failures.append("the decision batch and the capture window are not locked")
        return failures
    layer = protocol["decision_layer"]
    if layer.get("score_column") != SCORE_COLUMN:
        failures.append("the decision batch is not locked on score_xp")
    if layer.get("winner") is not None:
        failures.append("a winner was declared between score_xp and ep_next")
    if layer.get("logged_alongside") != LOGGED_ALONGSIDE:
        failures.append("ep_next is not the column logged beside score_xp")
    if int(layer.get("min_live_weeks") or 0) != MIN_LIVE_WEEKS:
        failures.append("a score-column winner does not wait for 20 live weeks")
    if layer.get("captain_baseline") != CAPTAIN_BASELINE:
        failures.append("the captain baseline is not the highest score")
    if int(layer.get("horizon") or 0) != HORIZON:
        failures.append("realised transfer gain is not the three-week horizon")
    targets = [tuple(pair) for pair in layer.get("template_targets") or []]
    if targets != list(TEMPLATE_SLOTS):
        failures.append("template price targets do not match the locked slots")
    if int(layer.get("alignment_gw") or 0) != 10:
        failures.append("the alignment diagnostic is not locked to one gameweek")
    if int(layer.get("power_weeks") or 0) != POWER_WEEKS:
        failures.append("the power check is not locked to 20 gameweeks")
    if float(layer.get("power_level") or 0) != POWER_LEVEL:
        failures.append("the power target is not 80 percent")
    if int(layer.get("power_bootstrap") or 0) != POWER_BOOTSTRAP or int(layer["power_seed"]) != POWER_SEED:
        failures.append("the power bootstrap is not the locked draw")
    if int(layer.get("power_sims") or 0) != POWER_SIMS or float(layer.get("power_step") or 0) != POWER_STEP:
        failures.append("the power grid is not locked")
    capture = protocol["capture"]
    if capture.get("clock") != "HTTP Date header":
        failures.append("the capture clock is not the response Date header")
    if list(capture.get("t24_hours") or []) != list(T24_HOURS):
        failures.append("the T-24h window is not locked")
    if list(capture.get("t1_minutes") or []) != list(T1_MINUTES):
        failures.append("the T-1h window is not locked")
    if not capture.get("raw_bootstrap"):
        failures.append("the raw bootstrap is not saved")
    if int(capture.get("cadence_minutes") or 0) != 15:
        failures.append("the capture job is not every 15 minutes")
    placebo = protocol.get("placebo") or {}
    if placebo.get("winner") is not None:
        failures.append("the placebo declared a winner")
    if "within each gameweek" not in str(placebo.get("shuffle") or ""):
        failures.append("the placebo shuffle is not within the gameweek")
    if placebo.get("naive_score") != "score_exp_points":
        failures.append("the naive score in the placebo is not expected points")
    common = protocol.get("common_state") or {}
    if common.get("winner") is not None:
        failures.append("the common-state test declared a winner")
    if "within each gameweek" not in str(common.get("shuffle") or ""):
        failures.append("the common-state shuffle is not within the gameweek")
    if "score unused" not in str(common.get("squad") or ""):
        failures.append("the common-state squad is not score-blind")
    for key in ("disagreement", "hierarchy", "initial_squad"):
        block = protocol.get(key) or {}
        if block.get("winner") is not None:
            failures.append(f"the {key} block declared a winner")
    if (protocol.get("initial_squad") or {}).get("transfers") != "none":
        failures.append("the initial portfolio applies a transfer")
    if "loso" not in protocol or "live_primary" not in protocol:
        failures.append("leave-one-season-out and the live primary are not locked")
    else:
        from src.eval.loso import contrast_keys

        loso = protocol["loso"]
        live = protocol["live_primary"]
        if loso.get("winner") is not None:
            failures.append("leave-one-season-out declared a winner")
        if loso.get("replaces_published") is not False:
            failures.append("leave-one-season-out replaces the four-season interval")
        if int(loso.get("bootstrap") or 0) != LOSO_BOOTSTRAP or int(loso["seed"]) != LOSO_SEED:
            failures.append("leave-one-season-out is not the locked draw")
        if int(loso.get("floor") or 0) != LOSO_FLOOR:
            failures.append("leave-one-season-out does not keep the 20-week floor")
        if int(loso.get("conditional_floor") or 0) != LOSO_CONDITIONAL_FLOOR:
            failures.append("the conditional floor is not 5 disagreement weeks")
        if int(loso.get("min_seasons") or 0) != LOSO_MIN_SEASONS:
            failures.append("a leave-one-out fold may pool a single season")
        if list(loso.get("contrasts") or []) != list(contrast_keys()):
            failures.append("the leave-one-season-out contrast list does not match the lock")
        if live.get("primary") != LIVE_PRIMARY:
            failures.append("the live primary is not ep_next")
        if live.get("shadow") != LIVE_SHADOW:
            failures.append("score_xp is not the live shadow")
        if live.get("historical_score") != SCORE_COLUMN:
            failures.append("the published historical score is not score_xp")
        if live.get("winner") is not None:
            failures.append("the live primary declared a winner")
        if live.get("wired") is not LIVE_PRIMARY_WIRED:
            failures.append("the live primary wiring flag does not match the lock")
        if live.get("covers_zero") != LIVE_COVERS_ZERO:
            failures.append("a live interval that covers zero is not undetermined")
        if int(live.get("continue_to_gw") or 0) != LIVE_CONTINUE_TO_GW:
            failures.append("the live test does not continue through gameweek 38")
        if int(live.get("from_gw") or 0) != LIVE_PRIMARY_FROM_GW:
            failures.append("the live primary does not start at gameweek 6")
        if int(live.get("min_weeks") or 0) != MIN_LIVE_WEEKS:
            failures.append("promotion does not wait for 20 live weeks")
        import src.live.scorer as live_scorer

        if live_scorer.SCORE_COL != SCORE_COLUMN:
            failures.append("the live scorer no longer prices score_xp")
        if "live_choice" not in inspect.getsource(live_scorer.plan_deadline):
            failures.append("the live plan does not read the ep_next choice")
        import src.live.deadline as live_deadline

        if "choice=" not in inspect.getsource(live_deadline.collect):
            failures.append("the deadline does not pass the ep_next choice")
        eligible = protocol.get("eligibility") or {}
        if eligible.get("winner") is not None:
            failures.append("the eligibility rule declared a winner")
        if eligible.get("replaces_published") is not False:
            failures.append("the eligibility rule replaces the four-season interval")
        if eligible.get("repairs_score") is not False:
            failures.append("the eligibility rule repairs the score")
        if eligible.get("combine") != "and":
            failures.append("eligibility is not the conjunction of xG and a prior season")
        if eligible.get("timing") != "post-hoc" or eligible.get("primary") != "eligible":
            failures.append("the eligible pool is not the primary post-hoc result")
        if eligible.get("all_weeks") != "sensitivity":
            failures.append("the all-weeks pool is not the sensitivity")
        if live.get("decision_capture") != "t1_same_stamp" or live.get("choose_after_seeing_scores") is not False:
            failures.append("the decision capture is not the same-stamp t1 pair")
        reversion = protocol.get("reversion") or {}
        if reversion.get("sign") != "actual_minus_baseline":
            failures.append("the reversion residual is not actual minus baseline")
        if reversion.get("installs_feature") is not False or reversion.get("replaces_score") is not False:
            failures.append("the reversion diagnostic edits score_xp")
        if reversion.get("oracle_is_forecast") is not False or reversion.get("winner") is not None:
            failures.append("the oracle minutes run is treated as a forecast or a winner")
        if list(reversion.get("weights") or []) != [0.6, 0.8]:
            failures.append("the oracle weights are not the locked pair")
        if int(reversion.get("minutes_gate") or 0) != 60 or int(reversion.get("bootstrap") or 0) != 1000:
            failures.append("the reversion draw is not locked")
        if "seed" not in reversion or int(reversion["seed"]) != 0:
            failures.append("the reversion seed is not 0")
        signal = protocol.get("reversion_signal") or {}
        if signal.get("installs_feature") is not False or signal.get("replaces_score") is not False:
            failures.append("the reversion signal edits score_xp")
        if signal.get("winner") is not None:
            failures.append("the reversion signal declared a winner")
        if int(signal.get("lookback") or 0) != 5 or int(signal.get("min_minutes") or 0) != 30:
            failures.append("the reversion signal window is not locked")
        if float(signal.get("z_threshold") or 0) != 1.5:
            failures.append("the reversion signal threshold is not locked")
        if list(signal.get("horizons") or []) != [1, 2, 3]:
            failures.append("the reversion signal horizons are not locked")
        if list(signal.get("baselines") or []) != ["score_exp_points", "score_xp"]:
            failures.append("the reversion signal baselines are not locked")
        if float(signal.get("spearman_bar") or 0) != -0.15 or float(signal.get("bucket_bar") or 0) != 0.5:
            failures.append("the reversion signal bars are not locked")
        if float(signal.get("hurst_bar") or 0) != 0.5 or int(signal.get("hurst_min_n") or 0) != 16:
            failures.append("the Hurst bar is not locked")
        if int(signal.get("bootstrap") or 0) != 1000 or int(signal.get("floor") or 0) != 20:
            failures.append("the reversion signal draw is not locked")
        if "seed" not in signal or int(signal["seed"]) != 0:
            failures.append("the reversion signal seed is not 0")
    manifest = json.loads((ROOT / "data" / "live" / "HOLDOUT_FREEZE.json").read_text(encoding="utf-8"))
    tracked = manifest.get("tracked") or {}
    if "data/cache/player_gw_2026_27.csv" in tracked:
        failures.append("the growing player cache is still a single freeze hash")
    snaps = [key for key in tracked if str(key).startswith("data/holdout/2026-27/player_gw/")]
    if len(snaps) < 5:
        failures.append("gameweeks 1-5 do not each have a snapshot hash")
    from src.eval.capture_schedule import missing_capture_gws
    from src.eval.provenance import load_deadlines

    missing = missing_capture_gws(
        ROOT, datetime.now(timezone.utc), load_deadlines()
    )
    if missing:
        failures.append(
            "no pre-deadline capture for gameweeks " + ", ".join(str(gw) for gw in missing)
        )
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


def _xp_of(rows: list[dict[str, object]], player_id: str, gw: int) -> float:
    frame = compute_xp(_score_frame(rows))
    hit = frame.loc[(frame["player_id"] == player_id) & (frame["gw"] == gw)]
    if hit.empty:
        return float("nan")
    return float(hit["xp"].iloc[0])


def _xmi_of(rows: list[dict[str, object]], player_id: str, gw: int) -> float:
    frame = _score_frame(rows)
    hit = frame.loc[(frame["player_id"] == player_id) & (frame["gw"] == gw)]
    if hit.empty:
        return float("nan")
    return float(hit["xmi"].iloc[0])


def _panel() -> list[dict[str, object]]:
    rows = []
    for gw in range(1, 7):
        rows.append(
            _row(
                player_id="regular",
                gw=gw,
                date=f"2024-08-{gw:02d}",
                fixture_id=f"2024-08-{gw:02d}:arsenal:wolves",
                minutes=0.0 if gw == 5 else 90.0,
                total_points=0.0 if gw == 5 else 6.0,
                xG=0.2,
            )
        )
    rows.append(
        _row(
            player_id="mate",
            gw=5,
            date="2024-08-05",
            fixture_id="2024-08-05:arsenal:wolves",
            minutes=90.0,
            total_points=2.0,
            xG=0.1,
        )
    )
    return rows


def _failures_perturbation() -> list[str]:
    """Scores at the deadline stay put when the scored week is perturbed."""
    failures: list[str] = []
    base = _panel()
    current = _xp_of(base, "regular", 5)
    if not math.isfinite(current):
        return ["the perturbation panel did not produce a score"]
    bumped = [dict(row) for row in base]
    for row in bumped:
        if row["player_id"] == "regular" and row["gw"] == 5:
            row["minutes"] = 90.0
            row["total_points"] = 40.0
            row["xG"] = 4.0
    if abs(_xp_of(bumped, "regular", 5) - current) > 1e-9:
        failures.append("the scored week's minutes, points or xG changed that week's xp")
    without_mate = [row for row in base if not (row["player_id"] == "mate" and row["gw"] == 5)]
    if abs(_xp_of(without_mate, "regular", 5) - current) > 1e-9:
        failures.append("removing another player's scored-week row changed xp")
    phantom = base + [
        _row(
            player_id="phantom",
            gw=5,
            date="2024-08-05",
            fixture_id="2024-08-05:arsenal:wolves",
            minutes=90.0,
            total_points=90.0,
            xG=6.0,
        )
    ]
    if abs(_xp_of(phantom, "regular", 5) - current) > 1e-9:
        failures.append("a scored-week row that did not exist changed xp")
    kept = _xmi_of(base, "regular", 6)
    dropped = _xmi_of(
        [row for row in base if not (row["player_id"] == "regular" and row["gw"] == 5)],
        "regular",
        6,
    )
    if not (math.isfinite(kept) and math.isfinite(dropped)) or abs(kept - dropped) < 1e-9:
        failures.append("deleting a past 0-minute row left the next week's minutes unchanged")
    src = inspect.getsource(cluster_interval)
    if "rng.integers(0, arr.size, size=arr.size)" not in src:
        failures.append("the interval does not resample one draw per gameweek")
    return failures


def run_asof_audit() -> dict[str, Any]:
    failures: list[str] = []
    failures.extend(_failures_sheet())
    failures.extend(_failures_priors())
    failures.extend(_failures_odds_rules())
    failures.extend(_failures_paths())
    failures.extend(_failures_official_xp_unused())
    failures.extend(_failures_perturbation())
    return {"passed": not failures, "failures": failures}
