"""Deadline scores for one live half.

``xp_on_pot`` is the one-match formula, the same components as ``compute_xp``.
This module chooses the inputs. Minutes come from the file, and a zero stays
a zero. Shot shares stay on the deadline. A priced week moves only the
opponent pot, and a double uses the first pot. The next club week with no
1X2 repeats the last priced week in the outlook table. One rebuild is shared
by Wildcard and the decision-week Free Hit. ``plan_half`` counts only the
weeks that have their own opening line. A copied week stays in the table and
adds nothing.

The formula is not changed here. A missing minutes file never reaches this
module: the caller records ``missing_minutes`` and does not plan a chip.
When a captured ``ep_next`` map is passed, that map chooses the squad and
``score_xp`` stays on the shadow log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.live.fpl_snapshot import ELEMENT
from src.live.half_plan import HalfPlan, WeekInputs, half_end, plan_half
from src.models.half_plan_scores import club_steps, week_inputs
from src.models.open_horizon import opening_pots_by_team_gw, xp_on_pot
from src.models.season_climb_ft import SquadState
from src.models.xp_engine import DEFCON_THRESH, MIN_HISTORY, MIN_MINUTES, add_player_priors
from src.teams import norm_team

# The climb's buy gate. It is not a 2026/27 rules constant.
ELIGIBLE_XMI = 45.0
SEASON = "2026-27"
SCORE_COL = "score_xp"
_STUB_DATE = "2099-01-01"
_GHOST = "__team__:"


class ScorerError(RuntimeError):
    """The line is present and the scorer cannot price the half."""


@dataclass(frozen=True)
class ScorerResult:
    """The half that was priced. ``line_weeks`` have their own 1X2."""

    plan: HalfPlan
    weeks: tuple[WeekInputs, ...]
    line_weeks: tuple[int, ...]
    horizon_weeks: tuple[int, ...]
    copy_note: str
    step_scores: dict[int, dict[str, float]]


def player_key(element: int) -> str:
    """Element id as the live season key."""
    return f"{SEASON}:{int(element)}"


def score_on_line(
    *,
    position: str,
    xmi: float,
    share_xg: float,
    share_xa: float,
    exp_defcon_hit: float,
    pot: Mapping[str, float],
    fwd_goal_scale: float = 1.0,
) -> float:
    """One player, one pot. The scale stays 1 until a later lock fits it."""
    return xp_on_pot(
        position=position,
        xmi=xmi,
        share_xg=share_xg,
        share_xa=share_xa,
        exp_defcon_hit=exp_defcon_hit,
        fwd_goal_scale=fwd_goal_scale,
        pot=dict(pot),
    )


def deadline_shares(logs: pd.DataFrame, before_gw: int) -> pd.DataFrame:
    """Shot shares at the deadline. The stub's own match is not in the share.

    Rolling minutes on the stub are the historical prior. The live scorer
    does not use them. A later step overwrites the club with the current one.
    """
    empty = pd.DataFrame(
        columns=[
            "player_id",
            "n_prior",
            "share_xG",
            "share_xA",
            "exp_defcon_hit",
            "team_norm",
            "position",
        ]
    )
    if logs.empty or "gw" not in logs.columns:
        return empty
    hist = logs.loc[pd.to_numeric(logs["gw"], errors="coerce") < int(before_gw)].copy()
    if hist.empty:
        return empty
    hist["player_id"] = [_key_from_log(value) for value in hist["player_id"]]
    hist["gw"] = pd.to_numeric(hist["gw"], errors="coerce").astype(int)
    hist["date"] = hist["date"].astype(str) if "date" in hist.columns else ""
    hist["team_norm"] = hist["team_norm"].map(lambda name: norm_team(str(name)))
    hist["position"] = hist["position"].astype(str)
    for col in ("minutes", "xG", "xA", "total_points", "defcon_raw"):
        if col not in hist.columns:
            hist[col] = 0.0
        hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0.0)
    hist["defcon_hit"] = _defcon_hit(hist)
    hist["fixture_id"] = hist["date"] + ":" + hist["team_norm"]

    last = (
        hist.sort_values(["player_id", "gw", "date"], kind="mergesort")
        .groupby("player_id", as_index=False)
        .tail(1)
    )
    stubs = last.copy()
    stubs["gw"] = int(before_gw)
    stubs["date"] = _STUB_DATE
    stubs["minutes"] = 0.0
    stubs["xG"] = 0.0
    stubs["xA"] = 0.0
    stubs["total_points"] = 0.0
    stubs["defcon_hit"] = 0.0
    stubs["defcon_raw"] = 0.0
    stubs["fixture_id"] = _STUB_DATE + ":" + stubs["team_norm"]

    clubs = stubs.drop_duplicates("team_norm")
    ghosts = clubs.copy()
    ghosts["player_id"] = ghosts["team_norm"].map(lambda club: f"{_GHOST}{club}")
    ghosts["minutes"] = 90.0
    ghosts["position"] = "MID"

    frame = pd.concat([hist, stubs, ghosts], ignore_index=True)
    priors = add_player_priors(frame)
    kept = priors.loc[
        (priors["gw"] == int(before_gw))
        & ~priors["player_id"].astype(str).str.startswith(_GHOST)
    ].copy()
    kept = kept.drop_duplicates("player_id", keep="first")
    return kept.reset_index(drop=True)


def roster_from_bootstrap(bootstrap: Mapping[str, Any]) -> pd.DataFrame:
    """Current club and price. The club is the one this week's fixture uses."""
    names = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    rows = []
    for element in bootstrap["elements"]:
        team_id = int(element["team"])
        rows.append(
            {
                "player_id": player_key(int(element["id"])),
                "position": ELEMENT[int(element["element_type"])],
                "team_norm": norm_team(names[team_id]),
                "value": int(element["now_cost"]),
            }
        )
    return pd.DataFrame(rows)


def build_pool(
    roster: pd.DataFrame,
    shares: pd.DataFrame,
    minutes: Mapping[str, float],
    owned: set[str],
) -> pd.DataFrame:
    """Eligible buyers plus every owned id. A missing share is zero.

    ``minutes`` is the live map. Rolling minutes on ``shares`` are ignored.
    """
    share_rows = (
        shares.drop_duplicates("player_id", keep="first").set_index("player_id")
        if not shares.empty
        else pd.DataFrame()
    )
    rows: list[dict[str, Any]] = []
    for person in roster.itertuples(index=False):
        pid = str(person.player_id)
        if not share_rows.empty and pid in share_rows.index:
            src = share_rows.loc[pid]
            share_xg = float(src["share_xG"])
            share_xa = float(src["share_xA"])
            defcon = float(src["exp_defcon_hit"])
            n_prior = int(src["n_prior"])
        else:
            share_xg = 0.0
            share_xa = 0.0
            defcon = 0.0
            n_prior = 0
        xmi = float(minutes.get(pid, 0.0))
        eligible = n_prior >= MIN_HISTORY and xmi >= ELIGIBLE_XMI
        if not eligible and pid not in owned:
            continue
        rows.append(
            {
                "player_id": pid,
                "position": str(person.position),
                "team_norm": str(person.team_norm),
                "value": int(person.value),
                "eligible": bool(eligible),
                "minutes": xmi,
                "share_xG": share_xg,
                "share_xA": share_xa,
                "exp_defcon_hit": defcon,
                "n_prior": n_prior,
                "total_points": 0.0,
                SCORE_COL: 0.0,
            }
        )
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise ScorerError("the chip pool is empty")
    missing = set(owned) - set(pool["player_id"].astype(str))
    if missing:
        raise ScorerError("owned players missing from the chip pool")
    return pool


def priced_gameweeks(
    fixtures: Sequence[Mapping[str, Any]],
    pots: Mapping[tuple[int, str], Sequence[Mapping[str, float]]],
    names: Mapping[int, str],
    start: int,
    end: int,
    limit: int = 3,
) -> list[int]:
    """Club weeks with a 1X2 for every side, in order, at most ``limit``.

    A week with no clubs is skipped. The walk stops at the first club week
    that is missing a 1X2. A later priced week is not used in its place.
    """
    found: list[int] = []
    for gw in range(int(start), int(end) + 1):
        clubs = _clubs_in_week(fixtures, names, gw)
        if not clubs:
            continue
        if any((gw, club) not in pots for club in clubs):
            break
        found.append(gw)
        if len(found) >= int(limit):
            break
    return found


def score_steps(
    pool: pd.DataFrame,
    pots: Mapping[tuple[int, str], Sequence[Mapping[str, float]]],
    weeks: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Score each priced week from the live minutes and the first pot.

    A club with no fixture that week scores 0. A second pot is not added.
    """
    out: dict[int, dict[str, float]] = {}
    for gw in weeks:
        scores: dict[str, float] = {}
        for row in pool.itertuples(index=False):
            quotes = pots.get((int(gw), str(row.team_norm)), [])
            pid = str(row.player_id)
            if not quotes:
                scores[pid] = 0.0
                continue
            scores[pid] = score_on_line(
                position=str(row.position),
                xmi=float(row.minutes),
                share_xg=float(row.share_xG),
                share_xa=float(row.share_xA),
                exp_defcon_hit=float(row.exp_defcon_hit),
                pot=quotes[0],
            )
        out[int(gw)] = scores
    return out


def scores_for_horizon(
    line_scores: Mapping[int, Mapping[str, float]],
    horizon: Sequence[int],
) -> tuple[dict[int, dict[str, float]], list[tuple[int, int]]]:
    """Fill a horizon week that has no line with the previous line's scores.

    The copy is the last priced step, not a new pot. The pairs are
    ``(week, source)``.
    """
    if not horizon:
        raise ScorerError("no club week to price")
    out: dict[int, dict[str, float]] = {}
    copies: list[tuple[int, int]] = []
    last: int | None = None
    for gw in horizon:
        week = int(gw)
        if week in line_scores:
            out[week] = {str(pid): float(value) for pid, value in line_scores[week].items()}
            last = week
            continue
        if last is None:
            raise ScorerError(f"GW{week} has no earlier line to copy")
        out[week] = {str(pid): float(value) for pid, value in line_scores[last].items()}
        copies.append((week, last))
    return out, copies


def copy_note(copies: Sequence[tuple[int, int]]) -> str:
    """One sentence per copied week. Empty when every horizon week has a line."""
    if not copies:
        return ""
    parts = [
        f"GW{int(gw)} repeats GW{int(src)} and has no 1X2 of its own"
        for gw, src in copies
    ]
    return ". ".join(parts) + "."


def clubs_from_fixtures(
    fixtures: Sequence[Mapping[str, Any]],
    names: Mapping[int, str],
    start: int,
    end: int,
) -> dict[int, set[str]]:
    """Clubs with a fixture. A week with none is left out, and it is not copied."""
    out: dict[int, set[str]] = {}
    for gw in range(int(start), int(end) + 1):
        clubs = _clubs_in_week(fixtures, names, gw)
        if clubs:
            out[gw] = clubs
    return out


def _capture_dir(gw: int, folder: Path | None) -> Path:
    if folder is not None:
        return folder
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "predictions"
        / "2026-27"
        / f"gw{int(gw):02d}"
    )


def decision_capture_file(gw: int, folder: Path | None = None) -> Path:
    """The T−1h official file. A T−24h file is not a fallback."""
    base = _capture_dir(gw, folder)
    marker = base / "slot_t1.json"
    if not marker.is_file():
        raise ScorerError(
            f"GW{gw} has no T-1h decision capture. An earlier file is not a fallback."
        )
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if str(meta.get("slot")) != "t1":
        raise ScorerError(f"GW{gw} decision slot is not t1")
    name = Path(str(meta.get("official") or "")).name
    official = base / name
    if not name or not official.is_file():
        raise ScorerError(f"GW{gw} T-1h capture is missing its official file")
    return official


def load_ep_next(
    gw: int,
    folder: Path | None = None,
    *,
    path: Path | None = None,
) -> dict[str, float]:
    """The pre-deadline ``ep_next`` capture. The decision file is the T−1h slot.

    ``path`` is only for a named dry run. It does not become the decision.
    There is no engine fill, and an all-zero column is refused.
    """
    csv_path = path if path is not None else decision_capture_file(gw, folder)
    if not csv_path.is_file():
        raise ScorerError(f"GW{gw} has no ep_next capture")
    frame = pd.read_csv(csv_path)
    if "source_field" not in frame.columns or not (frame["source_field"] == "ep_next").all():
        raise ScorerError(f"GW{gw} capture is not ep_next")
    chosen: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        if not _finite_score(row.official_xp):
            continue
        chosen[str(row.player_id)] = float(row.official_xp)
    if not chosen:
        raise ScorerError(f"GW{gw} ep_next capture is empty")
    if all(value == 0.0 for value in chosen.values()):
        raise ScorerError(f"GW{gw} ep_next capture is all zeros")
    return chosen


def _one_stamp(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise ScorerError(f"the capture has no {column}")
    values = {str(value) for value in frame[column].dropna().unique()}
    if len(values) != 1:
        raise ScorerError(f"{column} is not one capture time")
    return next(iter(values))


def write_shadow_log(
    dest: Path,
    engine_path: Path,
    official_path: Path,
    *,
    require_same_stamp: bool = True,
) -> pd.DataFrame:
    """Pair the engine score with ``ep_next``. Neither source file is overwritten.

    A decision pair requires both files to carry the same capture time.
    A missing score is not filled from the other column.
    """
    if dest.resolve() in {engine_path.resolve(), official_path.resolve()}:
        raise ScorerError("refusing to overwrite a source capture")
    engine = pd.read_csv(engine_path)
    official = pd.read_csv(official_path)
    if require_same_stamp:
        engine_stamp = _one_stamp(engine, "created_at")
        official_stamp = _one_stamp(official, "captured_at")
        if engine_stamp != official_stamp:
            raise ScorerError(
                f"score_xp {engine_stamp} and ep_next {official_stamp} are not the same capture"
            )
    left = engine.loc[:, ["player_id", "gw", "score"]].rename(columns={"score": "score_xp"})
    right = official.loc[:, ["player_id", "gw", "official_xp"]].rename(columns={"official_xp": "ep_next"})
    paired = left.merge(right, on=["player_id", "gw"], how="outer")
    paired["choice_field"] = "ep_next"
    if require_same_stamp:
        paired["captured_at"] = _one_stamp(engine, "created_at")
    paired.to_csv(dest, index=False)
    return paired


def paired_live_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Both scores, for every player who has both.

    ``choice_field`` records which score drives the transfer. It does not
    select these rows, and a missing score is not filled from the other.
    """
    needed = ("player_id", "gw", "score_xp", "ep_next")
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise ScorerError("the paired live log is missing " + ", ".join(missing))
    work = frame.loc[:, list(needed)].copy()
    work["score_xp"] = pd.to_numeric(work["score_xp"], errors="coerce")
    work["ep_next"] = pd.to_numeric(work["ep_next"], errors="coerce")
    both = work["score_xp"].notna() & work["ep_next"].notna()
    out = work.loc[both].copy()
    out["score_xp_minus_ep_next"] = out["score_xp"] - out["ep_next"]
    return out.reset_index(drop=True)


def live_choice(
    owned: set[str],
    pool_ids: set[str],
    ep_next: Mapping[str, float],
) -> dict[str, float]:
    """Decision-week scores from the capture. The engine is not a fallback."""
    missing = sorted(
        pid
        for pid in owned
        if pid not in ep_next or not _finite_score(ep_next[pid])
    )
    if missing:
        raise ScorerError(
            "owned players have no ep_next capture: " + ", ".join(missing)
        )
    chosen: dict[str, float] = {}
    for pid in pool_ids:
        if pid in ep_next and _finite_score(ep_next[pid]):
            chosen[pid] = float(ep_next[pid])
    return chosen


def _finite_score(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def plan_deadline(
    current_gw: int,
    state: SquadState,
    pool: pd.DataFrame,
    step_scores: Mapping[int, Mapping[str, float]],
    clubs: Mapping[int, set[str]],
    played: Mapping[int, str] | None = None,
    priced: set[int] | None = None,
    choice: Mapping[str, float] | None = None,
) -> tuple[HalfPlan, list[WeekInputs]]:
    """One rebuild on the decision-week scores, then the half plan.

    Later weeks keep that fifteen. They do not solve the squad again.
    ``priced`` is the weeks with their own opening line. Omitting it counts
    every week in the table. ``choice`` is the captured ``ep_next`` map.
    When it is passed, later weeks add nothing and are not filled from the engine.
    """
    if int(current_gw) not in step_scores:
        raise ScorerError(f"GW{int(current_gw)} has no outlook scores")
    frame = pool.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    pool_ids = set(frame["player_id"])
    if choice is None:
        decision = {
            str(pid): float(value) for pid, value in step_scores[int(current_gw)].items()
        }
        frame[SCORE_COL] = [decision.get(pid, 0.0) for pid in frame["player_id"]]
        prepared = {
            int(gw): {str(pid): float(value) for pid, value in scores.items()}
            for gw, scores in step_scores.items()
        }
    else:
        decision = live_choice({str(pid) for pid in state.ids()}, pool_ids, choice)
        frame[SCORE_COL] = [decision.get(pid, float("nan")) for pid in frame["player_id"]]
        frame = frame.loc[pd.to_numeric(frame[SCORE_COL], errors="coerce").notna()].copy()
        owned = {str(pid) for pid in state.ids()}
        if not owned <= set(frame["player_id"]):
            raise ScorerError("an owned player was left out of the ep_next choice")
        prepared = {}
        for gw in step_scores:
            if int(gw) == int(current_gw):
                prepared[int(gw)] = dict(decision)
            else:
                prepared[int(gw)] = {pid: 0.0 for pid in pool_ids}
        priced = {int(current_gw)}
    frame["total_points"] = frame[SCORE_COL]
    weeks = week_inputs(int(current_gw), state, frame, dict(clubs), prepared, SCORE_COL)
    plan = plan_half(int(current_gw), weeks, played=played, priced=priced)
    return plan, weeks


def price_half(
    *,
    gw: int,
    logs: pd.DataFrame,
    odds: pd.DataFrame,
    fixtures: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    state: SquadState,
    minutes: Mapping[str, float],
    played: Mapping[int, str] | None = None,
    choice: Mapping[str, float] | None = None,
) -> ScorerResult:
    """Price the half from the opening line and call ``plan_half`` once.

    ``choice`` is the captured ``ep_next`` map. The engine scores stay on
    ``step_scores`` and do not choose the squad when ``choice`` is passed.
    """
    names = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    shares = deadline_shares(logs, int(gw))
    roster = roster_from_bootstrap(bootstrap)
    pool = build_pool(roster, shares, minutes, set(state.ids()))
    pots = opening_pots_by_team_gw(odds, list(fixtures), names)
    end = half_end(int(gw))
    line_weeks = priced_gameweeks(fixtures, pots, names, int(gw), end, limit=3)
    if not line_weeks or int(line_weeks[0]) != int(gw):
        raise ScorerError(f"GW{int(gw)} is not fully priced")
    line_scores = score_steps(pool, pots, line_weeks)
    clubs = clubs_from_fixtures(fixtures, names, int(gw), end)
    horizon = club_steps(int(gw), clubs)
    step_scores, copies = scores_for_horizon(line_scores, horizon)
    choice_prices = None if choice is None else {str(pid): float(value) for pid, value in choice.items()}
    plan, weeks = plan_deadline(
        int(gw),
        state,
        pool,
        step_scores,
        clubs,
        played,
        priced=set(int(week) for week in line_weeks) if choice is None else {int(gw)},
        choice=choice_prices,
    )
    return ScorerResult(
        plan=plan,
        weeks=tuple(weeks),
        line_weeks=tuple(int(week) for week in line_weeks),
        horizon_weeks=tuple(int(week) for week in horizon),
        copy_note=copy_note(copies),
        step_scores=step_scores,
    )


def _key_from_log(value: object) -> str:
    text = str(value)
    if ":" in text:
        return text
    return player_key(int(text))


def _defcon_hit(frame: pd.DataFrame) -> pd.Series:
    threshold = frame["position"].map(DEFCON_THRESH)
    hit = (
        frame["position"].isin(list(DEFCON_THRESH))
        & (frame["minutes"] >= MIN_MINUTES)
        & (frame["defcon_raw"] >= threshold)
    )
    return hit.astype(float)


def _clubs_in_week(
    fixtures: Sequence[Mapping[str, Any]],
    names: Mapping[int, str],
    gw: int,
) -> set[str]:
    clubs: set[str] = set()
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None or int(event) != int(gw):
            continue
        for side in ("team_h", "team_a"):
            raw = names.get(int(fixture[side]))
            if raw:
                clubs.add(norm_team(raw))
    return clubs
