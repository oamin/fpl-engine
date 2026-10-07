"""Decision-layer replay on the closed seasons.

The scoring column is a parameter. This batch passes score_xp. ep_next is
logged on live captures and is not a historical column. No winner is
declared between them. The fitted hit threshold is reported and is not
applied. Chips stay an empty map.

Running this module writes the gated report. The formulas were locked
before the totals were read.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.decision_spec import (
    CAPTAIN_BASELINE,
    HORIZON,
    LOGGED_ALONGSIDE,
    MIN_LIVE_WEEKS,
    SCORE_COLUMN,
    TEMPLATE_SLOTS,
)
from src.eval.provenance import OFFICIAL_XP_COLUMNS
from src.models.season_climb import FORMATIONS
from src.models.season_climb_budget import pick_squad
from src.rules.fpl_2026 import (
    BUDGET_TENTHS,
    HIT_COST,
    MAX_PER_CLUB,
    normalize_position,
    sell_price,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


@dataclass
class Squad:
    purchase: dict[str, int]
    position: dict[str, str]
    club: dict[str, str]
    bank: int

    def copy(self) -> Squad:
        return Squad(
            dict(self.purchase),
            dict(self.position),
            dict(self.club),
            int(self.bank),
        )


def _pid(value: object) -> tuple[int, int | str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def assert_open_score(n_live_weeks: int, winner: str | None) -> None:
    """A winner between score_xp and ep_next waits for 20 live weeks."""
    if winner and int(n_live_weeks) < MIN_LIVE_WEEKS:
        raise RuntimeError(
            "refusing a winner between score_xp and ep_next before 20 live weeks"
        )


def chips_fired() -> int:
    """This batch plays no chip. Margins are not searched."""
    return 0


def collapse_gameweek(block: pd.DataFrame, sum_columns: tuple[str, ...]) -> pd.DataFrame:
    """One row per player. Points and scores add. A double is not two steps."""
    work = block.copy()
    if "date" not in work.columns:
        work["date"] = ""
    if "fixture_id" not in work.columns:
        work["fixture_id"] = ""
    work["player_id"] = work["player_id"].astype(str)
    present = list(dict.fromkeys(column for column in sum_columns if column in work.columns))
    for column in present:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    work = work.sort_values(["date", "fixture_id", "player_id"], kind="mergesort")
    base = work.groupby("player_id", as_index=False).first()
    summed = work.groupby("player_id", as_index=False)[present].sum()
    keep = [column for column in base.columns if column not in present]
    out = base[keep].merge(summed, on="player_id", how="left")
    if "eligible" in work.columns:
        flags = work.groupby("player_id")["eligible"].any()
        out["eligible"] = out["player_id"].map(flags).astype(bool)
    return out


def _price(value: object) -> int | None:
    number = pd.to_numeric(value, errors="coerce")
    if number is None or not np.isfinite(number):
        return None
    return int(round(float(number)))


def score_lookup(week: pd.DataFrame, score_col: str) -> dict[str, float]:
    """Eligible finite scores. Anyone else, including a missing row, is 0."""
    found: dict[str, float] = {}
    if week.empty or score_col not in week.columns:
        return found
    for row in week.itertuples(index=False):
        pid = str(row.player_id)
        raw = getattr(row, score_col)
        eligible = bool(getattr(row, "eligible", False))
        if eligible and raw is not None and math.isfinite(float(raw)):
            found[pid] = float(raw)
        else:
            found[pid] = 0.0
    return found


def points_lookup(week: pd.DataFrame) -> dict[str, float]:
    if week.empty:
        return {}
    return {
        str(row.player_id): float(row.total_points)
        for row in week.itertuples(index=False)
    }


def value_lookup(week: pd.DataFrame) -> dict[str, int]:
    found: dict[str, int] = {}
    if week.empty or "value" not in week.columns:
        return found
    for row in week.itertuples(index=False):
        price = _price(row.value)
        if price is not None:
            found[str(row.player_id)] = price
    return found


def _squad_from_rows(rows: pd.DataFrame) -> Squad:
    purchase: dict[str, int] = {}
    position: dict[str, str] = {}
    club: dict[str, str] = {}
    spent = 0
    seen: set[str] = set()
    for row in rows.itertuples(index=False):
        pid = str(row.player_id)
        if pid in seen:
            raise RuntimeError("a squad named the same player twice")
        seen.add(pid)
        price = _price(row.value)
        if price is None:
            raise RuntimeError("a squad player has no price")
        purchase[pid] = price
        position[pid] = normalize_position(str(row.position))
        club[pid] = str(row.team_norm)
        spent += price
    if spent > BUDGET_TENTHS:
        raise RuntimeError("squad costs more than the budget")
    counts: dict[str, int] = {}
    for name in club.values():
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > MAX_PER_CLUB:
            raise RuntimeError("squad has more than three players from one club")
    return Squad(purchase, position, club, BUDGET_TENTHS - spent)


def opening_pool(week: pd.DataFrame, score_col: str) -> pd.DataFrame:
    pool = week.loc[week["eligible"].astype(bool)].copy()
    pool[score_col] = pd.to_numeric(pool[score_col], errors="coerce")
    pool["value"] = pd.to_numeric(pool["value"], errors="coerce")
    pool = pool.loc[np.isfinite(pool[score_col]) & np.isfinite(pool["value"])]
    pool["player_id"] = pool["player_id"].astype(str)
    pool["position"] = pool["position"].map(lambda value: normalize_position(str(value)))
    return pool.drop_duplicates("player_id", keep="first")


def opening_squad(week: pd.DataFrame, score_col: str) -> Squad:
    picked = pick_squad(opening_pool(week, score_col), score_col)
    return _squad_from_rows(picked)


def template_squad(week: pd.DataFrame, score_col: str) -> Squad:
    """One pass through the price targets. Later weeks do not transfer."""
    pool = opening_pool(week, score_col)
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    clubs: dict[str, int] = {}
    spent = 0
    for slot, target in TEMPLATE_SLOTS:
        slot_pos = normalize_position(slot)
        best: dict[str, Any] | None = None
        best_key: tuple[Any, ...] | None = None
        block = pool.loc[pool["position"] == slot_pos]
        for row in block.itertuples(index=False):
            pid = str(row.player_id)
            if pid in used:
                continue
            price = int(round(float(row.value)))
            if spent + price > BUDGET_TENTHS:
                continue
            club = str(row.team_norm)
            if clubs.get(club, 0) >= MAX_PER_CLUB:
                continue
            key = (abs(price - int(target)), -float(getattr(row, score_col)), _pid(pid))
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "player_id": pid,
                    "position": slot_pos,
                    "team_norm": club,
                    "value": price,
                    "slot_position": slot_pos,
                }
        if best is None:
            raise RuntimeError(f"template cannot fill {slot_pos} near {target}")
        chosen.append(best)
        used.add(str(best["player_id"]))
        clubs[str(best["team_norm"])] = clubs.get(str(best["team_norm"]), 0) + 1
        spent += int(best["value"])
    squad = _squad_from_rows(pd.DataFrame(chosen))
    for row, (slot, _target) in zip(chosen, TEMPLATE_SLOTS, strict=True):
        if normalize_position(str(row["position"])) != normalize_position(slot):
            raise RuntimeError("a template slot took the wrong position")
    return squad


def neutral_pool(week: pd.DataFrame) -> pd.DataFrame:
    """Eligible players with a finite price. A score does not enter the filter."""
    if "eligible" not in week.columns:
        raise RuntimeError("neutral squad has no eligibility column")
    pool = week.loc[week["eligible"].astype(bool)].copy()
    pool["value"] = pd.to_numeric(pool["value"], errors="coerce")
    pool = pool.loc[np.isfinite(pool["value"])]
    pool["player_id"] = pool["player_id"].astype(str)
    pool["position"] = pool["position"].map(lambda value: normalize_position(str(value)))
    return pool.drop_duplicates("player_id", keep="first")


def neutral_squad(week: pd.DataFrame) -> Squad:
    """Price-ladder fifteen. The tie-break is the lower player id. Score is unused."""
    pool = neutral_pool(week)
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    clubs: dict[str, int] = {}
    spent = 0
    for slot, target in TEMPLATE_SLOTS:
        slot_pos = normalize_position(slot)
        best: dict[str, Any] | None = None
        best_key: tuple[Any, ...] | None = None
        block = pool.loc[pool["position"] == slot_pos]
        for row in block.itertuples(index=False):
            pid = str(row.player_id)
            if pid in used:
                continue
            price = int(round(float(row.value)))
            if spent + price > BUDGET_TENTHS:
                continue
            club = str(row.team_norm)
            if clubs.get(club, 0) >= MAX_PER_CLUB:
                continue
            key = (abs(price - int(target)), _pid(pid))
            if best_key is None or key < best_key:
                best_key = key
                best = {
                    "player_id": pid,
                    "position": slot_pos,
                    "team_norm": club,
                    "value": price,
                }
        if best is None:
            raise RuntimeError(f"neutral squad cannot fill {slot_pos} near {target}")
        chosen.append(best)
        used.add(str(best["player_id"]))
        clubs[str(best["team_norm"])] = clubs.get(str(best["team_norm"]), 0) + 1
        spent += int(best["value"])
    return _squad_from_rows(pd.DataFrame(chosen))


def select_xi(frame: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Legal XI. Ties break toward the higher score, then the lower player id."""
    if score_col == "total_points":
        raise RuntimeError("the XI is not chosen by realised points")
    best_proxy: float | None = None
    best_rows: list[dict[str, Any]] | None = None
    for n_def, n_mid, n_fwd in FORMATIONS:
        picked: list[Any] = []
        short = False
        for pos, need in (("GKP", 1), ("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)):
            block = frame.loc[frame["position"] == pos]
            ordered = sorted(
                block.itertuples(index=False),
                key=lambda row: (-float(getattr(row, score_col)), _pid(row.player_id)),
            )
            if len(ordered) < need:
                short = True
                break
            picked.extend(ordered[:need])
        if short:
            continue
        proxy = sum(float(getattr(row, score_col)) for row in picked)
        if best_proxy is None or proxy > best_proxy:
            best_proxy = proxy
            best_rows = [row._asdict() for row in picked]
    if best_rows is None:
        raise RuntimeError("Could not form an XI")
    return pd.DataFrame(best_rows)


def armband(xi: pd.DataFrame, score_col: str) -> str:
    """Captain inside an XI. Realised points are not a captain rule."""
    if score_col == "total_points":
        raise RuntimeError("captain is not chosen by realised points")
    winner: str | None = None
    winner_key: tuple[Any, ...] | None = None
    for row in xi.itertuples(index=False):
        key = (-float(getattr(row, score_col)), _pid(row.player_id))
        if winner_key is None or key < winner_key:
            winner_key = key
            winner = str(row.player_id)
    if winner is None:
        raise RuntimeError("an XI has no captain")
    return winner


def _view(squad: Squad, week: pd.DataFrame, score_col: str) -> pd.DataFrame:
    scores = score_lookup(week, score_col)
    rows = []
    for pid, pos in squad.position.items():
        rows.append(
            {
                "player_id": pid,
                "position": pos,
                "team_norm": squad.club[pid],
                "value": squad.purchase[pid],
                score_col: scores.get(pid, 0.0),
            }
        )
    return pd.DataFrame(rows)


def week_points(squad: Squad, week: pd.DataFrame, score_col: str) -> float:
    xi = select_xi(_view(squad, week, score_col), score_col)
    captain = armband(xi, score_col)
    points = points_lookup(week)
    total = sum(points.get(str(pid), 0.0) for pid in xi["player_id"])
    return float(total + points.get(captain, 0.0))


def captain_gap(squad: Squad, week: pd.DataFrame, score_col: str, baseline_col: str) -> float:
    """Points of the score captain minus the baseline captain, inside one XI."""
    if baseline_col == "total_points" or score_col == "total_points":
        raise RuntimeError("captain is not compared with the oracle")
    view = _view(squad, week, score_col)
    base_scores = score_lookup(week, baseline_col)
    view[baseline_col] = [base_scores.get(str(pid), 0.0) for pid in view["player_id"]]
    xi = select_xi(view, score_col)
    xi[baseline_col] = [base_scores.get(str(pid), 0.0) for pid in xi["player_id"]]
    primary = armband(xi, score_col)
    other = armband(xi, baseline_col)
    points = points_lookup(week)
    return float(points.get(primary, 0.0) - points.get(other, 0.0))


def _club_ok(squad: Squad, out_id: str, in_club: str) -> bool:
    counts: dict[str, int] = {}
    for pid, club in squad.club.items():
        if pid == out_id:
            continue
        counts[club] = counts.get(club, 0) + 1
    counts[in_club] = counts.get(in_club, 0) + 1
    return all(count <= MAX_PER_CLUB for count in counts.values())


def legal_moves(squad: Squad, week: pd.DataFrame, score_col: str) -> list[dict[str, Any]]:
    """Same-position swaps that clear price, club cap, and this week's eligibility.

    Predicted gain may be negative. A fitted slope is not an argument.
    """
    scores = score_lookup(week, score_col)
    values = value_lookup(week)
    if week.empty or "eligible" not in week.columns:
        return []
    pool = week.loc[week["eligible"].astype(bool)]
    owned = set(squad.purchase)
    moves: list[dict[str, Any]] = []
    for incoming in pool.itertuples(index=False):
        in_id = str(incoming.player_id)
        if in_id in owned:
            continue
        in_pos = normalize_position(str(incoming.position))
        in_price = _price(incoming.value)
        if in_price is None:
            continue
        in_score = scores.get(in_id, 0.0)
        in_club = str(incoming.team_norm)
        for out_id, out_pos in squad.position.items():
            if out_pos != in_pos:
                continue
            if not _club_ok(squad, out_id, in_club):
                continue
            current = values.get(out_id, squad.purchase[out_id])
            proceeds = sell_price(squad.purchase[out_id], current)
            bank = squad.bank + proceeds - in_price
            if bank < 0:
                continue
            moves.append(
                {
                    "player_in": in_id,
                    "player_out": out_id,
                    "position": in_pos,
                    "predicted": float(in_score - scores.get(out_id, 0.0)),
                    "price": in_price,
                    "proceeds": int(proceeds),
                    "bank": int(bank),
                    "in_score": float(in_score),
                    "out_score": float(scores.get(out_id, 0.0)),
                    "club": in_club,
                }
            )
    return moves


def greedy_step(squad: Squad, week: pd.DataFrame, score_col: str) -> tuple[Squad, dict[str, Any] | None]:
    """At most one same-position free transfer. No hit and no saved transfer.

    The argument is this gameweek only. A fitted slope is not an argument.
    The input squad is not mutated.
    """
    best: dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for move in legal_moves(squad, week, score_col):
        gain = float(move["predicted"])
        if gain <= 0.0:
            continue
        key = (-gain, -float(move["in_score"]), _pid(move["player_in"]), _pid(move["player_out"]))
        if best_key is None or key < best_key:
            best_key = key
            best = move
    if best is None:
        return squad, None
    nxt = squad.copy()
    out_id = str(best["player_out"])
    in_id = str(best["player_in"])
    del nxt.purchase[out_id]
    del nxt.position[out_id]
    del nxt.club[out_id]
    nxt.purchase[in_id] = int(best["price"])
    nxt.position[in_id] = str(best["position"])
    nxt.club[in_id] = str(best["club"])
    nxt.bank = int(best["bank"])
    return nxt, best


def realised_over(
    points_by_gw: dict[int, dict[str, float]],
    gw: int,
    player_in: str,
    player_out: str,
    horizon: int = HORIZON,
) -> float:
    """Sum in-minus-out on t, t+1, t+2 when those gameweeks exist. Never t−1."""
    total = 0.0
    for step in range(int(horizon)):
        target = int(gw) + step
        week = points_by_gw.get(target)
        if week is None:
            continue
        total += float(week.get(player_in, 0.0)) - float(week.get(player_out, 0.0))
    return total


def _prepare_weeks(
    frame: pd.DataFrame, score_col: str, gw_start: int, gw_end: int
) -> dict[int, pd.DataFrame]:
    work = frame.copy()
    banned = [column for column in OFFICIAL_XP_COLUMNS if column in work.columns]
    if banned:
        work = work.drop(columns=banned)
    if score_col not in work.columns:
        raise RuntimeError(f"the replay has no {score_col}")
    if score_col in OFFICIAL_XP_COLUMNS:
        raise RuntimeError("the decision replay does not score scraped xP")
    work["gw"] = pd.to_numeric(work["gw"], errors="coerce")
    work = work.dropna(subset=["gw"])
    weeks: dict[int, pd.DataFrame] = {}
    sums = ("total_points", score_col, CAPTAIN_BASELINE)
    for gw, block in work.groupby(work["gw"].astype(int), sort=True):
        number = int(gw)
        if number < int(gw_start) or number > int(gw_end):
            continue
        weeks[number] = collapse_gameweek(block, sums)
    return weeks


def permute_week(
    week: pd.DataFrame,
    score_col: str,
    season_index: int,
    gw: int,
) -> pd.DataFrame:
    """Permute one score inside one gameweek, among eligible finite values only."""
    out = week.copy()
    values = pd.to_numeric(out[score_col], errors="coerce")
    mask = out["eligible"].astype(bool) & np.isfinite(values)
    rng = np.random.default_rng(np.random.SeedSequence([0, int(season_index), int(gw)]))
    shuffled = rng.permutation(values.loc[mask].to_numpy(float))
    out.loc[mask, score_col] = shuffled
    return out


def audit_move(before: Squad, move: dict[str, Any], week: pd.DataFrame, score_col: str) -> None:
    """Raise unless this transfer obeys budget, sell price, position, and this week only."""
    out_id = str(move["player_out"])
    in_id = str(move["player_in"])
    if out_id not in before.purchase or in_id in before.purchase:
        raise RuntimeError("transfer names a player the squad cannot buy or sell")
    if len(before.purchase) != 15:
        raise RuntimeError("transfer started from a squad that is not 15")
    incoming = week.loc[week["player_id"].astype(str) == in_id]
    if incoming.empty:
        raise RuntimeError("buy is not in this gameweek")
    row = incoming.iloc[0]
    if not bool(row["eligible"]):
        raise RuntimeError("buy is not eligible this week")
    in_pos = normalize_position(str(row["position"]))
    if in_pos != before.position[out_id] or str(move["position"]) != in_pos:
        raise RuntimeError("transfer is not same-position")
    current = value_lookup(week).get(out_id, before.purchase[out_id])
    proceeds = sell_price(before.purchase[out_id], current)
    price = _price(row["value"])
    if price is None or int(move["proceeds"]) != int(proceeds) or int(move["price"]) != int(price):
        raise RuntimeError("buy or sell price is not this week's price")
    bank = before.bank + int(proceeds) - int(price)
    if bank < 0 or int(move["bank"]) != int(bank):
        raise RuntimeError("bank does not match the sell and the buy")
    counts: dict[str, int] = {}
    for pid, club in before.club.items():
        if pid == out_id:
            continue
        counts[club] = counts.get(club, 0) + 1
    in_club = str(row["team_norm"])
    counts[in_club] = counts.get(in_club, 0) + 1
    if any(count > MAX_PER_CLUB for count in counts.values()):
        raise RuntimeError("transfer breaks the club cap")
    scores = score_lookup(week, score_col)
    predicted = float(scores.get(in_id, 0.0) - scores.get(out_id, 0.0))
    if predicted <= 0.0 or abs(predicted - float(move["predicted"])) > 1e-8:
        raise RuntimeError("predicted gain is not this week's score")


def replay_season(
    frame: pd.DataFrame,
    score_col: str = SCORE_COLUMN,
    *,
    gw_start: int = 5,
    gw_end: int = 38,
    permute_season_index: int | None = None,
) -> dict[str, Any]:
    """Hold, one-free-transfer greedy, and the template, from one legal start.

    ``permute_season_index`` shuffles the score inside each gameweek before
    the opening squad and every later transfer. Points stay on the player.
    """
    weeks = _prepare_weeks(frame, score_col, gw_start, gw_end)
    if permute_season_index is not None:
        weeks = {
            gw: permute_week(week, score_col, permute_season_index, gw)
            for gw, week in weeks.items()
        }
    start: int | None = None
    hold = template = None
    for gw in sorted(weeks):
        try:
            hold = opening_squad(weeks[gw], score_col)
            template = template_squad(weeks[gw], score_col)
        except RuntimeError:
            continue
        start = gw
        break
    if start is None or hold is None or template is None:
        return {
            "weeks": [],
            "transfers": [],
            "chips": chips_fired(),
            "start_gw": None,
            "score_column": score_col,
        }
    greedy = hold.copy()
    points_by_gw = {gw: points_lookup(week) for gw, week in weeks.items()}
    rows: list[dict[str, Any]] = []
    transfers: list[dict[str, Any]] = []
    for gw in sorted(weeks):
        if gw < start:
            continue
        week = weeks[gw]
        if gw != start:
            before = greedy
            greedy, move = greedy_step(before, week, score_col)
            if move is not None:
                audit_move(before, move, week, score_col)
                points = points_by_gw[gw]
                transfers.append(
                    {
                        "gw": gw,
                        "player_in": move["player_in"],
                        "player_out": move["player_out"],
                        "predicted": move["predicted"],
                        "realised": realised_over(
                            points_by_gw, gw, move["player_in"], move["player_out"]
                        ),
                        "realised_t": float(
                            points.get(str(move["player_in"]), 0.0)
                            - points.get(str(move["player_out"]), 0.0)
                        ),
                    }
                )
            if len(greedy.purchase) != 15:
                raise RuntimeError("a transfer changed the squad size")
        rows.append(
            {
                "gw": gw,
                "hold": week_points(hold, week, score_col),
                "greedy": week_points(greedy, week, score_col),
                "template": week_points(template, week, score_col),
                "captain_gap": captain_gap(hold, week, score_col, CAPTAIN_BASELINE),
            }
        )
    return {
        "weeks": rows,
        "transfers": transfers,
        "chips": chips_fired(),
        "start_gw": start,
        "score_column": score_col,
    }


def ols_line(predicted: np.ndarray, realised: np.ndarray) -> tuple[float, float] | None:
    """realised = a + b * predicted. None when the slope is not identified."""
    x = np.asarray(predicted, dtype=float)
    y = np.asarray(realised, dtype=float)
    if x.size < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if float(np.unique(x).size) < 2:
        return None
    design = np.column_stack([np.ones(x.size), x])
    coef, *_rest = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def calibrate(
    transfers: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Slope of realised horizon gain on predicted gain. Not fed back into greedy."""
    if transfers.empty:
        return {"claimed": False, "reason": "no transfers"}
    seasons = tuple(dict.fromkeys(transfers["season"].astype(str).tolist()))
    groups = {
        season: transfers.loc[transfers["season"] == season, ["predicted", "realised"]].to_numpy(
            float
        )
        for season in seasons
    }
    groups = {season: rows for season, rows in groups.items() if len(rows)}
    pooled = np.concatenate(list(groups.values()), axis=0)
    fit = ols_line(pooled[:, 0], pooled[:, 1])
    if fit is None:
        return {"claimed": False, "reason": "slope not identified", "n": int(len(pooled))}
    intercept, slope = fit
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    kept = 0
    for draw in range(n_boot):
        parts = []
        for rows in groups.values():
            index = rng.integers(0, len(rows), size=len(rows))
            parts.append(rows[index])
        sample = np.concatenate(parts, axis=0)
        refit = ols_line(sample[:, 0], sample[:, 1])
        if refit is None:
            continue
        boots[kept] = refit[1]
        kept += 1
    if kept < n_boot * 0.95:
        return {
            "claimed": False,
            "reason": "bootstrap draws were singular",
            "a": intercept,
            "b": slope,
            "n": int(len(pooled)),
        }
    lo, hi = np.quantile(boots[:kept], [0.025, 0.975])
    claimed = float(lo) > 0.0 and slope != 0.0
    out: dict[str, Any] = {
        "claimed": bool(claimed),
        "a": intercept,
        "b": slope,
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(pooled)),
        "n_gws": {season: int(len(rows)) for season, rows in groups.items()},
    }
    if claimed:
        out["hit_threshold"] = float((HIT_COST - intercept) / slope)
        out["shrinkage"] = float(slope / HORIZON)
    return out


def loso_thresholds(transfers: pd.DataFrame) -> list[dict[str, Any]]:
    """One-week hurdle fit on the other seasons. The result is not applied.

    ``transfers`` uses ``predicted`` and the decision-week point gap in
    ``realised``. A non-positive or unidentified slope has no threshold.
    """
    if transfers.empty:
        return []
    seasons = list(dict.fromkeys(transfers["season"].astype(str).tolist()))
    rows: list[dict[str, Any]] = []
    for held in seasons:
        train = transfers.loc[transfers["season"].astype(str) != held]
        fit = ols_line(
            train["predicted"].to_numpy(float),
            train["realised"].to_numpy(float),
        )
        row: dict[str, Any] = {"held_out": held, "n": int(len(train))}
        if fit is None or fit[1] <= 0.0:
            row["a"] = None if fit is None else fit[0]
            row["b"] = None if fit is None else fit[1]
            row["threshold"] = None
        else:
            intercept, slope = fit
            row["a"] = intercept
            row["b"] = slope
            row["threshold"] = float((HIT_COST - intercept) / slope)
        rows.append(row)
    return rows


def _complete(rows: pd.DataFrame, column: str, minimum: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    complete: dict[str, np.ndarray] = {}
    incomplete: dict[str, int] = {}
    for season, block in rows.groupby("season", sort=True):
        values = block[column].to_numpy(float)
        if int(values.size) >= minimum:
            complete[str(season)] = values
        else:
            incomplete[str(season)] = int(values.size)
    return complete, incomplete


def pool_columns(
    rows: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    minimum: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    from src.eval.gates import cluster_interval

    pooled: dict[str, Any] = {}
    for column in columns:
        complete, incomplete = _complete(rows, column, minimum)
        if not complete:
            pooled[column] = {"incomplete_seasons": incomplete, "mean": None, "lo": None, "hi": None}
            continue
        summary = cluster_interval(
            complete, seasons=tuple(complete), n_boot=n_boot, seed=seed
        )
        summary["incomplete_seasons"] = incomplete
        pooled[column] = summary
    return pooled


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "not identified"
    return f"{float(value):+.{digits}f}"


def _interval_text(row: dict[str, Any]) -> str:
    if row.get("mean") is None:
        return "no season reached 20 gameweeks"
    return f"{row['mean']:+.4f} [{row['lo']:+.4f}, {row['hi']:+.4f}]"


def run() -> None:
    from src.eval.alignment import save_scatter
    from src.eval.encompassing import live_encompassing
    from src.eval.gates import load_protocol, run_asof_audit, write_gated_report
    from src.eval.honest_pool import build_season
    from src.eval.power import minimum_detectable

    protocol = load_protocol()
    layer = protocol.get("decision_layer") or {}
    if layer.get("score_column") != SCORE_COLUMN:
        raise RuntimeError("the decision batch is not locked on score_xp")
    if layer.get("winner") not in (None, "none"):
        raise RuntimeError("the locked batch already names a winner")
    live = live_encompassing()
    assert_open_score(int(live.get("n_gws") or 0), None)
    codes = protocol["season_codes"]
    frames = {
        season: build_season(season, codes[season], protocol)
        for season in protocol["closed_seasons"]
    }
    focus = frames["2024-25"]
    plot = ROOT / "data" / "plots" / "alignment_2024_25_gw10.png"
    aligned = save_scatter(focus, int(layer["alignment_gw"]), plot)
    aligned["season"] = "2024-25"
    log_rows = pd.read_csv(PROCESSED / "player_gw_logscore.csv")
    key = "score_xp_minus_score_exp_points"
    deltas = log_rows.loc[log_rows["comparison"] == key, "delta"].to_numpy(float)
    power = minimum_detectable(deltas)
    week_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    for season, frame in frames.items():
        played = replay_season(
            frame,
            str(layer["score_column"]),
            gw_start=int(protocol["gw_start"]),
            gw_end=int(protocol["gw_end"]),
        )
        if played["chips"] != 0:
            raise RuntimeError("a chip fired in an empty-map batch")
        for row in played["weeks"]:
            week_rows.append({"season": season, **row})
        for row in played["transfers"]:
            transfer_rows.append({"season": season, **row})
    weeks = pd.DataFrame(week_rows)
    weeks["greedy_minus_hold"] = weeks["greedy"] - weeks["hold"]
    weeks["greedy_minus_template"] = weeks["greedy"] - weeks["template"]
    weeks["hold_minus_template"] = weeks["hold"] - weeks["template"]
    transfers = pd.DataFrame(transfer_rows)
    fit = calibrate(
        transfers,
        n_boot=int(protocol["bootstrap"]),
        seed=int(protocol["seed"]),
    )
    decision_intervals = pool_columns(
        weeks,
        (
            "greedy_minus_hold",
            "greedy_minus_template",
            "hold_minus_template",
            "captain_gap",
        ),
        minimum=int(protocol["min_gws_per_season"]),
        n_boot=int(protocol["bootstrap"]),
        seed=int(protocol["seed"]),
    )
    certified = json.loads((PROCESSED / "procedure_intervals.json").read_text(encoding="utf-8"))
    comparisons = {
        key: certified["comparisons"][key],
        **decision_intervals,
    }
    intervals = {
        "seasons": list(protocol["closed_seasons"]),
        "min_gws": int(protocol["min_gws_per_season"]),
        "comparisons": comparisons,
    }
    audit = run_asof_audit()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    weeks.to_csv(PROCESSED / "decision_weeks.csv", index=False)
    transfers.to_csv(PROCESSED / "decision_transfers.csv", index=False)
    (PROCESSED / "decision_power.json").write_text(
        json.dumps(power, indent=2) + "\n", encoding="utf-8"
    )
    (PROCESSED / "decision_alignment.json").write_text(
        json.dumps(aligned, indent=2) + "\n", encoding="utf-8"
    )
    lines = _report_lines(aligned, power, weeks, fit, decision_intervals, live, plot)
    write_gated_report(REPORTS / "decision_layer.md", audit, intervals, lines)


def _num(value: float | None) -> str:
    if value is None:
        return "undefined"
    return f"{value:+.4f}"


def _report_lines(
    aligned: dict[str, Any],
    power: dict[str, Any],
    weeks: pd.DataFrame,
    fit: dict[str, Any],
    intervals: dict[str, Any],
    live: dict[str, Any],
    plot: Path,
) -> list[str]:
    mde = power["mde"]
    mde_text = "not reached on the grid" if mde is None else f"{mde:.4f}"
    if fit.get("claimed"):
        threshold = (
            f"The implied one-week hit threshold is {fit['hit_threshold']:+.3f} points "
            f"and the per-week shrinkage is {fit['shrinkage']:+.4f}. "
            "Neither number was used to choose a transfer."
        )
    else:
        threshold = (
            "No hit threshold is claimed. "
            f"Reason: {fit.get('reason', 'the slope interval does not stay above zero')}."
        )
        if "b" in fit:
            threshold += (
                f" The fitted slope is {_fmt(fit.get('b'))} "
                f"[{_fmt(fit.get('lo'))}, {_fmt(fit.get('hi'))}]."
            )
    counts = ", ".join(
        f"{season} {int((weeks['season'] == season).sum())}"
        for season in weeks["season"].drop_duplicates()
    )
    return [
        "# Decision layer",
        "",
        "First batch, locked before these totals were read. The score column is "
        f"`{SCORE_COLUMN}`. `{LOGGED_ALONGSIDE}` is the live column captured beside it. "
        f"Live pre-deadline gameweeks so far: {int(live.get('n_gws') or 0)}. "
        "No winner is declared between the two. Chips did not fire. "
        "The captain baseline is the highest `score_exp_points` inside the hold eleven, "
        "not the highest realised points.",
        "",
        "## Alignment of the two forecasts",
        "",
        "One closed gameweek, 2024-25 gameweek "
        f"{aligned['gw']}. This is a diagnostic of the join, not a benchmark. "
        "The withdrawn −0.35 is Spearman(score_xp, points) minus Spearman(scraped xP, points). "
        "It is not the correlation of the two forecasts with each other.",
        "",
        f"Rows {aligned['n_rows']}, players {aligned['n_players']}, "
        f"paired {aligned['n_paired']}. "
        f"Duplicate player-fixture keys: {aligned['duplicate_player_fixture']}. "
        f"Max rows for one player: {aligned['max_rows_per_player']}. "
        f"Same-row join: {aligned['same_row']}. "
        f"Positions legal: {aligned['positions_ok']}. Gameweek matches: {aligned['gw_ok']}.",
        "",
        f"Spearman(score_xp, scraped xP) {_num(aligned['spearman_score_vs_official'])}. "
        f"Pearson {_num(aligned['pearson_score_vs_official'])}. "
        f"Spearman(−score_xp, scraped xP) {_num(aligned['spearman_negated_score_vs_official'])}. "
        f"A sign flip fits better: {aligned['sign_flip_fits_better']}.",
        "",
        f"Spearman(score_xp, points) {_num(aligned['spearman_score_vs_points'])}. "
        f"Spearman(scraped xP, points) {_num(aligned['spearman_official_vs_points'])}. "
        f"Gap {_num(aligned['rank_gap_score_minus_official'])}.",
        "",
        f"Scatter: `{plot.relative_to(ROOT)}`.",
        "",
        "## Power of 20 gameweeks",
        "",
        "Residuals are the certified score_xp minus expected-points gameweek deltas "
        "with the mean removed. Each draw takes 20 residuals, adds a candidate effect, "
        f"and reruns the gameweek cluster interval (B = {power['n_boot']}, seed {power['seed']}, "
        f"{power['n_sims']} draws). Power is the share of intervals that lie entirely above zero.",
        "",
        f"Smallest effect with power at least {power['power_target']:.0%} at 20 gameweeks: "
        f"{mde_text}. "
        f"Median half-width of the null intervals: {power['median_null_half_width']:+.4f}. "
        f"Residual sd of the closed-season deltas: {power['residual_sd']:+.4f} "
        f"({power['n_deltas']} gameweeks).",
        "",
        "The full-sample interval on about 130 gameweeks is not this resolution. "
        "A gap as small as the withdrawn −0.0016 log score is below what 20 weeks resolve.",
        "",
        "## Baselines",
        "",
        "Shared start: the budgeted fifteen that maximises score_xp, and a template "
        "built the same week from the locked price targets. Hold keeps that fifteen. "
        "Greedy may make one same-position free transfer when the score gain is positive. "
        "It does not bank a second transfer and it does not take a hit. "
        "That rule is not the published hold margin of 1.25. "
        "The template never transfers. A week that cannot fill both fifteens is skipped. "
        f"Gameweeks in the table: {counts}.",
        "",
        "A positive mean is points per gameweek for the first name.",
        "",
        _season_table(weeks),
        "",
        "| comparison | mean [95% interval] |",
        "|---|---:|",
        f"| greedy − hold | {_interval_text(intervals['greedy_minus_hold'])} |",
        f"| greedy − template | {_interval_text(intervals['greedy_minus_template'])} |",
        f"| hold − template | {_interval_text(intervals['hold_minus_template'])} |",
        f"| captain score_xp − highest score_exp_points | {_interval_text(intervals['captain_gap'])} |",
        "",
        "## Transfer calibration",
        "",
        "Realised gain is points of the player in, minus points of the player out, "
        "over the decision week and the next two gameweeks that exist. "
        f"Ordinary least squares of that sum on the predicted score gain: "
        f"a = {_fmt(fit.get('a'))}, b = {_fmt(fit.get('b'))}, "
        f"interval for b [{_fmt(fit.get('lo'))}, {_fmt(fit.get('hi'))}], "
        f"n = {fit.get('n', 0)}.",
        "",
        threshold,
        "",
        "## Chips",
        "",
        "The chip map is empty. Chips did not fire. Free Hit 12 and Wildcard 16 were not searched.",
        "",
        "## Review",
        "",
        "The formulas were locked before the run (bc-ffc0ced9). "
        "The diagnostics were reviewed after it (bc-3d00af39). "
        "The weekly gap is not a captain counted twice and not a double gameweek summed twice. "
        "The Gameweek 10 correlation shows the two forecasts move together. "
        "It does not explain the pooled −0.35 rank gap against points. "
        "0.010 is the effect 20 gameweeks detect at 80% power. "
        "The null half-width is about half of that, which is a weaker bar. "
        "The hit threshold is an observation. It was not used to choose a transfer.",
        "",
    ]


def _season_table(weeks: pd.DataFrame) -> str:
    lines = [
        "| season | weeks | hold | greedy | template |",
        "|---|---:|---:|---:|---:|",
    ]
    for season, block in weeks.groupby("season", sort=True):
        lines.append(
            f"| {season} | {len(block)} | {block['hold'].mean():.2f} | "
            f"{block['greedy'].mean():.2f} | {block['template'].mean():.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    run()
