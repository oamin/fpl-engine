"""Chip-week squad frontier, locked in reports/squad_frontier_plan.md.

Both fifteens are scored with the model's own ``xi_xp``. A week enters the
call only when the model's pre-chip bank and sell prices can buy the human
fifteen. Realised points are stored beside the call and do not make it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.live.benchmark import build_frames, player_key
from src.live.entry import load_entry
from src.models.blank_context import apply_fixture_tags
from src.models.cohort_carry import carry_entry, cohort_specs
from src.models.half_plan_scores import squad_outlook
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import PROCESSED, REPORTS, _early, bank_before, merged_clubs
from src.models.season_climb import pick_xi
from src.models.season_climb_ft import EARLY_SCORE_CAP, _gw_pool
from src.models.squad_gap import CHIP_SQUADS
from src.rules.fpl_2026 import sell_price, squad_legal
from src.teams import norm_team

NEAR = 1.0
FAR = 4.0
SHARE = 0.5
CHIPS = ("wildcard", "free_hit")
STATUSES = ("reachable", "unreachable_money", "pool", "rules", "rules_mismatch")
POSITIONS = ("GKP", "DEF", "MID", "FWD")
PLAN = REPORTS / "squad_frontier_plan.md"
WEEKS_CSV = PROCESSED / "squad_frontier_gw15.csv"
REPORT = REPORTS / "squad_frontier.md"


def week_band(gap_per: float) -> str:
    """Near at or below 1.0. Far above 4.0. The point between is middle."""
    if float(gap_per) <= NEAR:
        return "near"
    if float(gap_per) > FAR:
        return "far"
    return "middle"


def chip_call(bands: list[str]) -> str:
    """Half the reachable weeks. An empty list has no call."""
    n = len(bands)
    if n == 0:
        return "none"
    n_near = sum(band == "near" for band in bands)
    n_far = sum(band == "far" for band in bands)
    if n_near / n >= SHARE:
        return "near"
    if n_far / n >= SHARE:
        return "far"
    return "middle"


def objective_gap(chip: str, model_steps: list[float], human_steps: list[float]) -> tuple[float, int]:
    """Raw model-minus-human gap, and the number of steps that gap uses.

    A free hit uses the decision week. A wildcard uses every priced step.
    """
    if chip not in CHIPS:
        raise RuntimeError(f"{chip} is not a chip squad")
    if not model_steps or len(model_steps) != len(human_steps):
        raise RuntimeError("the two portfolios need the same priced steps")
    if chip == "free_hit":
        return float(model_steps[0]) - float(human_steps[0]), 1
    return (
        float(sum(model_steps)) - float(sum(human_steps)),
        len(model_steps),
    )


def per_week(raw: float, steps: int) -> float:
    if int(steps) < 1:
        raise RuntimeError("a priced window needs a step")
    return float(raw) / int(steps)


def acquisition_cost(
    ids: list[str],
    owned_sell: dict[str, int],
    market: dict[str, int],
) -> int | None:
    """Chip cost in tenths. An owned player costs his sell price. A missing price is None."""
    total = 0
    for pid in ids:
        if pid in owned_sell:
            total += int(owned_sell[pid])
        elif pid in market:
            total += int(market[pid])
        else:
            return None
    return int(total)


def classify_status(
    *,
    in_pool: bool,
    scored: bool,
    shape_ok: bool,
    human_over: bool | None,
    model_over: bool,
) -> str:
    """Reachability before a week enters the call.

    A missing pool row or a missing score stays out. Squad shape and the
    club cap stay out. The human's own budget, when it can be priced and
    it rejects the fifteen, stays out. Money from the model's state is the
    remaining exclusion.
    """
    if not in_pool or not scored:
        return "pool"
    if not shape_ok:
        return "rules"
    if human_over is True:
        return "rules_mismatch"
    if model_over:
        return "unreachable_money"
    return "reachable"


def _element(player_id: str) -> int:
    return int(str(player_id).split(":")[-1])


def _gw1_prices(roster: pd.DataFrame) -> dict[int, int]:
    gw1 = roster.loc[pd.to_numeric(roster["gw"], errors="coerce") == 1].drop_duplicates("player_id")
    prices: dict[int, int] = {}
    for pid, raw in zip(gw1["player_id"], gw1["value"], strict=True):
        price = _price(raw)
        if price is not None:
            prices[_element(str(pid))] = price
    return prices


def held_purchases(
    entry: dict[str, Any],
    gw1_prices: dict[int, int],
    gw: int,
    held_ids: list[str],
) -> dict[str, int] | None:
    """Purchase price of the fifteen he held before this deadline.

    A chip week lists its deals together, not in the order they must be
    applied. The price is the latest single buy before this deadline, or
    the Gameweek 1 price when he has held the player since the start.
    Two buys in the same week are left unpriced.
    """
    earlier = [row for row in entry.get("transfers") or [] if int(row["gw"]) < int(gw)]
    opening = {int(player["id"]) for player in entry.get("opening_squad") or []}
    found: dict[str, int] = {}
    for pid in held_ids:
        element = _element(pid)
        buys = [row for row in earlier if int(row["in_id"]) == element]
        if buys:
            last = max(int(row["gw"]) for row in buys)
            same = [row for row in buys if int(row["gw"]) == last]
            if len(same) != 1:
                return None
            found[pid] = int(same[0]["in_cost"])
            continue
        if element not in opening or element not in gw1_prices:
            return None
        found[pid] = int(gw1_prices[element])
    return found


def human_acquisition(
    ids: list[str],
    owned_sell: dict[str, int],
    entry: dict[str, Any],
    gw: int,
) -> int | None:
    """What his own deals paid. A new player uses that week's buy price."""
    buys: dict[str, list[int]] = {}
    for row in entry.get("transfers") or []:
        if int(row["gw"]) != int(gw):
            continue
        pid = player_key(row["in_id"])
        buys.setdefault(pid, []).append(int(row["in_cost"]))
    total = 0
    for pid in ids:
        if pid in owned_sell:
            total += int(owned_sell[pid])
            continue
        paid = buys.get(pid) or []
        if len(paid) != 1:
            return None
        total += int(paid[0])
    return int(total)


def _sell_map(purchase: dict[str, int], market: dict[str, int]) -> dict[str, int] | None:
    found: dict[str, int] = {}
    for pid, bought in purchase.items():
        if pid not in market:
            return None
        found[str(pid)] = sell_price(int(bought), int(market[pid]))
    return found


def _price(raw: object) -> int | None:
    value = pd.to_numeric(raw, errors="coerce")
    if pd.isna(value):
        return None
    return int(value)


def _finite(raw: object) -> bool:
    value = pd.to_numeric(raw, errors="coerce")
    return bool(pd.notna(value))


def _index(pool: pd.DataFrame) -> dict[str, Any]:
    frame = pool.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame = frame.drop_duplicates("player_id", keep="first")
    return {str(row.player_id): row for row in frame.itertuples(index=False)}


def _market(pool_by: dict[str, Any]) -> dict[str, int]:
    found: dict[str, int] = {}
    for pid, row in pool_by.items():
        price = _price(getattr(row, "value", None))
        if price is not None:
            found[pid] = price
    return found


def _roster_market(roster: pd.DataFrame, gw: int) -> dict[str, int]:
    block = roster.loc[pd.to_numeric(roster["gw"], errors="coerce") == int(gw)]
    found: dict[str, int] = {}
    for row in block.itertuples(index=False):
        price = _price(getattr(row, "value", None))
        if price is None:
            continue
        found[str(row.player_id)] = price
    return found


def _club(row: Any) -> str:
    club = getattr(row, "team_norm", None)
    if not isinstance(club, str) or club.strip() == "" or club.lower() == "nan":
        club = getattr(row, "team", None)
    text = "" if club is None else str(club).strip()
    if text == "" or text.lower() == "nan":
        raise RuntimeError(f"{getattr(row, 'player_id', '')} has no club")
    return text


def _position(row: Any) -> str:
    position = str(getattr(row, "position", ""))
    if position == "GK":
        position = "GKP"
    if position not in POSITIONS:
        raise RuntimeError(f"{getattr(row, 'player_id', '')} has no position")
    return position


def _ids(players: list[dict[str, Any]]) -> list[str]:
    ids = [player_key(player["id"]) for player in players]
    if len(ids) != len(set(ids)) or len(ids) != 15:
        raise RuntimeError("a chip fifteen does not have 15 players")
    return ids


def _step_values(
    pool: pd.DataFrame,
    ids: list[str],
    step_scores: dict[int, dict[str, float]],
    window: list[int],
) -> list[float]:
    values: list[float] = []
    for gw in window:
        scores = {str(pid): float(value) for pid, value in step_scores[int(gw)].items()}
        missing = [pid for pid in ids if pid not in scores or scores[pid] != scores[pid]]
        if missing:
            raise RuntimeError(f"GW{int(gw)} has no step score for {sorted(missing)}")
        view = squad_outlook(pool, set(ids), scores)
        values.append(float(view.xi_xp))
    return values


def _squad_points(roster: pd.DataFrame, gw: int, ids: list[str]) -> float | None:
    block = roster.loc[pd.to_numeric(roster["gw"], errors="coerce") == int(gw)].copy()
    block["player_id"] = block["player_id"].astype(str)
    block["total_points"] = pd.to_numeric(block["total_points"], errors="coerce")
    grouped = block.groupby("player_id", as_index=True)["total_points"].sum()
    if any(pid not in grouped.index or pd.isna(grouped.loc[pid]) for pid in ids):
        return None
    return float(sum(float(grouped.loc[pid]) for pid in ids))


def _position_spend(ids: list[str], pool_by: dict[str, Any], market: dict[str, int]) -> dict[str, int] | None:
    spend = {position: 0 for position in POSITIONS}
    for pid in ids:
        if pid not in pool_by or pid not in market:
            return None
        spend[_position(pool_by[pid])] += int(market[pid])
    return spend


def _max_club(ids: list[str], pool_by: dict[str, Any]) -> int | None:
    clubs: dict[str, int] = {}
    for pid in ids:
        if pid not in pool_by:
            return None
        club = _club(pool_by[pid])
        clubs[club] = clubs.get(club, 0) + 1
    if not clubs:
        return None
    return int(max(clubs.values()))


def _buy_pool(ids: list[str], pool_by: dict[str, Any]) -> int:
    return sum(1 for pid in ids if pid in pool_by and bool(getattr(pool_by[pid], "eligible", False)))


def _norm_pos(raw: object) -> str:
    position = str(raw)
    if position == "GK":
        return "GKP"
    return position


def _week_sheet(roster: pd.DataFrame, gw: int) -> pd.DataFrame:
    block = roster.loc[pd.to_numeric(roster["gw"], errors="coerce") == int(gw)].copy()
    if block.empty:
        return block
    block["player_id"] = block["player_id"].astype(str)
    block["minutes"] = pd.to_numeric(block["minutes"], errors="coerce").fillna(0.0)
    return block


def _minutes_by_player(roster: pd.DataFrame, gw: int) -> dict[str, float]:
    block = _week_sheet(roster, gw)
    if block.empty:
        return {}
    return {str(pid): float(total) for pid, total in block.groupby("player_id")["minutes"].sum().items()}


def _early_lookup(early: pd.DataFrame | None, gw: int) -> dict[str, float]:
    """Capped early scores for this deadline. A missing table is empty."""
    if early is None or early.empty or "score_xp" not in early.columns:
        return {}
    week = early.loc[pd.to_numeric(early["gw"], errors="coerce") == int(gw)]
    found: dict[str, float] = {}
    scores = pd.to_numeric(week["score_xp"], errors="coerce")
    for pid, score in zip(week["player_id"].astype(str), scores, strict=False):
        if pd.isna(score):
            continue
        found[str(pid)] = min(float(score), float(EARLY_SCORE_CAP))
    return found


def _roster_stub(
    roster: pd.DataFrame,
    gw: int,
    pid: str,
    score: float,
    columns: pd.Index,
) -> dict[str, Any] | None:
    block = _week_sheet(roster, gw)
    hit = block.loc[block["player_id"] == str(pid)]
    if hit.empty:
        return None
    source = hit.iloc[0]
    stub = {col: (source[col] if col in source.index else pd.NA) for col in columns}
    stub["player_id"] = str(pid)
    stub["score_xp"] = float(score)
    stub["eligible"] = False
    if _norm_pos(stub.get("position")) == "GKP":
        stub["position"] = "GKP"
    return stub


def attach_early_players(
    pool: pd.DataFrame,
    roster: pd.DataFrame,
    early: pd.DataFrame | None,
    gw: int,
    human_ids: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Score a played player the published frame has not kept yet.

    The number is capped at 6. Minutes of 0 stay out, even when an early
    score exists. The row is not eligible to buy.
    """
    lookup = _early_lookup(early, gw)
    minutes = _minutes_by_player(roster, gw)
    out = pool.copy()
    out["player_id"] = out["player_id"].astype(str)
    attached: list[str] = []
    stubs: list[dict[str, Any]] = []
    for pid in human_ids:
        if float(minutes.get(str(pid), 0.0)) <= 0.0:
            continue
        score = lookup.get(str(pid))
        if score is None:
            continue
        mask = out["player_id"] == str(pid)
        if bool(mask.any()):
            current = pd.to_numeric(out.loc[mask, "score_xp"], errors="coerce")
            if bool(current.notna().all()):
                continue
            out.loc[mask, "score_xp"] = float(score)
            out.loc[mask, "eligible"] = False
            attached.append(str(pid))
            continue
        stub = _roster_stub(roster, gw, str(pid), float(score), out.columns)
        if stub is None:
            continue
        stubs.append(stub)
        attached.append(str(pid))
    if stubs:
        out = pd.concat([out, pd.DataFrame(stubs)], ignore_index=True)
    out = out.drop_duplicates("player_id", keep="first").reset_index(drop=True)
    return out, attached


def blank_tag(
    roster: pd.DataFrame,
    clubs: dict[int, set[str]],
    gw: int,
    player_id: str,
) -> str:
    """Sheet tag for a player who did not play. A player who played has none.

    ``replaced`` is a teammate at the same club and position on at least 60
    minutes. ``benched`` is a club that played while nobody at that position
    did. ``no_fixture`` is a club with no game. A teammate on 1 to 59
    minutes is unresolved. The tag does not change a score.
    """
    block = _week_sheet(roster, gw)
    hit = block.loc[block["player_id"] == str(player_id)] if not block.empty else block
    played = 0.0 if hit.empty else float(hit["minutes"].sum())
    if played > 0.0:
        return ""
    if hit.empty:
        return "unresolved"
    row = hit.iloc[0]
    club_raw = row["team_norm"] if "team_norm" in hit.columns and pd.notna(row["team_norm"]) else row.get("team", "")
    club = norm_team("" if club_raw is None else str(club_raw))
    if club == "" or club.lower() == "nan":
        return "unresolved"
    playing = {norm_team(str(name)) for name in clubs.get(int(gw), set())}
    if club not in playing:
        return "no_fixture"
    position = _norm_pos(row["position"])
    others = block.loc[block["player_id"] != str(player_id)].copy()
    if others.empty:
        return "benched"
    club_col = "team_norm" if "team_norm" in others.columns else "team"
    others = others.copy()
    others["_club"] = others[club_col].map(lambda value: norm_team("" if value is None else str(value)))
    others["_pos"] = others["position"].map(_norm_pos)
    mates = others.loc[(others["_club"] == club) & (others["_pos"] == position)]
    if mates.empty:
        return "benched"
    highest = float(mates.groupby("player_id")["minutes"].sum().max())
    if highest >= 60.0:
        return "replaced"
    if highest <= 0.0:
        return "benched"
    return "unresolved"


def _shape_ok(ids: list[str], pool: pd.DataFrame, pool_by: dict[str, Any]) -> bool:
    if any(pid not in pool_by for pid in ids):
        return False
    positions = [_position(pool_by[pid]) for pid in ids]
    clubs = [_club(pool_by[pid]) for pid in ids]
    if not squad_legal(positions, clubs):
        return False
    frame = pool.loc[pool["player_id"].astype(str).isin(ids)].drop_duplicates("player_id")
    frame = frame.copy()
    frame["score_xp"] = 0.0
    try:
        pick_xi(frame, "score_xp")
    except RuntimeError:
        return False
    return True


def score_week(
    week: dict[str, Any],
    entry: dict[str, Any],
    spec: dict[str, Any],
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    clubs: dict[int, set[str]],
    horizon_scores: Any,
    early: pd.DataFrame | None,
    *,
    use_early: bool = False,
) -> dict[str, Any]:
    """One human chip week. The stored rebuild is the model's portfolio.

    ``use_early`` scores a played human player from the capped early table
    when the published frame has not kept him. It does not change the
    model's steps, and it does not make that player buyable.
    """
    chip = str(week["his_chip"])
    if chip not in CHIPS:
        raise RuntimeError(f"{chip} is not in this reading")
    gw = int(week["gw"])
    window = [int(step) for step in week["horizon"]]
    if not window or int(window[0]) != gw:
        raise RuntimeError(f"GW{gw} horizon does not start at the decision week")
    pre_ids = [str(pid) for pid in week["pre_ids"]]
    rebuilt = [str(pid) for pid in week["rebuilt_ids"]]
    if len(set(pre_ids)) != 15 or len(set(rebuilt)) != 15:
        raise RuntimeError(f"GW{gw} model fifteen is not 15 players")
    pool = _gw_pool(feat, roster, gw, set(pre_ids), early)
    pool = apply_fixture_tags(pool, gw, clubs)
    pool_by = _index(pool)
    market = _market(pool_by)
    market.update({pid: price for pid, price in _roster_market(roster, gw).items() if pid not in market})
    step_scores = horizon_scores(gw, pool, window)
    model_steps = _step_values(pool, rebuilt, step_scores, window)
    held_steps = _step_values(pool, pre_ids, step_scores, window)
    lead = float(sum(model_steps)) - float(sum(held_steps))
    if abs(lead - float(week["wc_sum"])) > 1e-4:
        raise RuntimeError(f"GW{gw} rebuild lead {lead} does not match the stored {week['wc_sum']}")
    decision = float(model_steps[0]) - float(held_steps[0])
    if abs(decision - float(week["fh_margin"])) > 1e-4:
        raise RuntimeError(f"GW{gw} free-hit lead {decision} does not match the stored {week['fh_margin']}")
    played = next(row for row in entry["gameweeks"] if int(row["gw"]) == gw)
    human_ids = _ids(list(played["xi"]) + list(played["bench"]))
    names = {
        player_key(player["id"]): str(player.get("name") or player["id"])
        for player in list(played["xi"]) + list(played["bench"])
    }
    attached: list[str] = []
    if use_early:
        pool, attached = attach_early_players(pool, roster, early, gw, human_ids)
        pool = apply_fixture_tags(pool, gw, clubs)
        pool_by = _index(pool)
        market = _market(pool_by)
        market.update({pid: price for pid, price in _roster_market(roster, gw).items() if pid not in market})
        step_scores = horizon_scores(gw, pool, window)
    in_pool = all(pid in pool_by for pid in human_ids)
    scored = in_pool and all(_finite(getattr(pool_by[pid], "score_xp", None)) for pid in human_ids)
    shape_ok = _shape_ok(human_ids, pool, pool_by) if in_pool else False
    model_sell = _sell_map({str(pid): int(price) for pid, price in week["purchase"].items()}, market)
    if model_sell is None:
        raise RuntimeError(f"GW{gw} is missing a sell price for the model's fifteen")
    model_budget = int(week["bank"]) + sum(model_sell.values())
    model_cost = acquisition_cost(rebuilt, model_sell, market)
    human_model_cost = acquisition_cost(human_ids, model_sell, market)
    if model_cost is None or (in_pool and human_model_cost is None):
        raise RuntimeError(f"GW{gw} cannot price a fifteen that is in the pool")
    held_ids = [str(pid) for pid in week["human_pre_ids"]]
    if len(set(held_ids)) != 15:
        raise RuntimeError(f"GW{gw} human pre-chip fifteen is not 15 players")
    held_prices = held_purchases(entry, _gw1_prices(roster), gw, held_ids)
    human_sell = None if held_prices is None else _sell_map(held_prices, market)
    human_budget = None if human_sell is None else int(bank_before(entry, gw)) + sum(human_sell.values())
    human_cost_own = None if human_sell is None else human_acquisition(human_ids, human_sell, entry, gw)
    if human_cost_own is None or human_budget is None:
        human_over = None
        human_account = "unpriced"
    elif int(human_cost_own) > int(human_budget):
        human_over = True
        human_account = "over"
    else:
        human_over = False
        human_account = "ok"
    status = classify_status(
        in_pool=in_pool,
        scored=scored,
        shape_ok=shape_ok,
        human_over=human_over,
        model_over=human_model_cost is not None and int(human_model_cost) > int(model_budget),
    )
    human_steps: list[float] | None = None
    if scored and shape_ok:
        human_steps = _step_values(pool, human_ids, step_scores, window)
    raw = None
    steps_used = None
    gap = None
    band = None
    if human_steps is not None and status in {"reachable", "unreachable_money", "rules_mismatch"}:
        raw, steps_used = objective_gap(chip, model_steps, human_steps)
        gap = per_week(raw, steps_used)
        band = week_band(gap)
    shortfall = None if human_model_cost is None else max(0, int(human_model_cost) - int(model_budget))
    model_spend_pos = _position_spend(rebuilt, pool_by, market)
    human_spend_pos = _position_spend(human_ids, pool_by, market)
    row: dict[str, Any] = {
        "entry_id": int(spec["entry_id"]),
        "label": str(spec["label"]),
        "group": str(spec["group"]),
        "gw": gw,
        "chip": chip,
        "status": status,
        "n_steps": int(steps_used if steps_used is not None else (1 if chip == "free_hit" else len(window))),
        "model_xp": None if human_steps is None else float(sum(model_steps) if chip == "wildcard" else model_steps[0]),
        "human_xp": None if human_steps is None else float(sum(human_steps) if chip == "wildcard" else human_steps[0]),
        "raw_gap": raw,
        "gap_per": gap,
        "band": band or "",
        "model_budget": int(model_budget),
        "model_cost": int(model_cost),
        "human_cost": human_model_cost,
        "shortfall": shortfall,
        "human_budget": human_budget,
        "human_cost_own": human_cost_own,
        "human_account": human_account,
        "price_model": None if model_spend_pos is None else int(sum(model_spend_pos.values())),
        "price_human": None if human_spend_pos is None else int(sum(human_spend_pos.values())),
        "max_club_model": _max_club(rebuilt, pool_by),
        "max_club_human": _max_club(human_ids, pool_by),
        "buy_pool_model": _buy_pool(rebuilt, pool_by),
        "buy_pool_human": _buy_pool(human_ids, pool_by),
        "overlap": len(set(human_ids) & set(rebuilt)),
        "same_fifteen": set(human_ids) == set(rebuilt),
        "points_model": _squad_points(roster, gw, rebuilt),
        "points_human": _squad_points(roster, gw, human_ids),
        "early_players": "; ".join(names.get(pid, pid) for pid in attached),
        "blank_tags": "; ".join(
            f"{names.get(pid, pid)} {tag}"
            for pid in human_ids
            for tag in [blank_tag(roster, clubs, gw, pid)]
            if tag
        ),
    }
    for position in POSITIONS:
        row[f"spend_{position.lower()}_model"] = None if model_spend_pos is None else model_spend_pos[position]
        row[f"spend_{position.lower()}_human"] = None if human_spend_pos is None else human_spend_pos[position]
    return row


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One call per chip, on the reachable weeks only."""
    if any(str(row["group"]) == "reference" for row in rows):
        raise RuntimeError("the reference manager is in the frontier")
    if any(str(row["chip"]) not in CHIPS for row in rows):
        raise RuntimeError("a week is not a wildcard or a free hit")
    chips: dict[str, Any] = {}
    for chip in CHIPS:
        part = [row for row in rows if row["chip"] == chip]
        reachable = [row for row in part if row["status"] == "reachable"]
        bands = [str(row["band"]) for row in reachable]
        if any(band not in {"near", "far", "middle"} for band in bands):
            raise RuntimeError(f"{chip} has a reachable week with no band")
        mean = None if not reachable else float(sum(float(row["gap_per"]) for row in reachable) / len(reachable))
        chips[chip] = {
            "n": len(part),
            "n_reachable": len(reachable),
            "n_money": sum(row["status"] == "unreachable_money" for row in part),
            "n_pool": sum(row["status"] == "pool" for row in part),
            "n_rules": sum(row["status"] == "rules" for row in part),
            "n_mismatch": sum(row["status"] == "rules_mismatch" for row in part),
            "n_near": sum(band == "near" for band in bands),
            "n_middle": sum(band == "middle" for band in bands),
            "n_far": sum(band == "far" for band in bands),
            "mean": mean,
            "call": chip_call(bands),
        }
    return {"n": len(rows), "chips": chips}


def _fmt_xp(value: float | None) -> str:
    if value is None or value != value:
        return ""
    return f"{float(value):.2f}"


def _fmt_share(numer: int, denom: int) -> str:
    if denom == 0:
        return ""
    share = numer / denom
    if abs(share) < 5e-3:
        return "0.00"
    return f"{share:.2f}"


def _fmt_pounds(tenths: int | None) -> str:
    if tenths is None or tenths != tenths:
        return ""
    return f"{int(tenths) / 10:.1f}"


def _meaning(call: str) -> str:
    if call == "near":
        return (
            "The human portfolio sits close to the model's own objective. "
            "The next suspicion is that objective or the state the rebuild started from."
        )
    if call == "far":
        return (
            "The human portfolio is expensive on the model's objective. "
            "The realised points sit beside this call. They are not the reason for it."
        )
    if call == "middle":
        return "The reachable weeks do not gather on one side of the bar. The question stays open."
    return "This chip has no reachable week, so it has no call."


def unscored_players(
    rows: list[dict[str, Any]],
    feat: pd.DataFrame,
    roster: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Human chip players with no decision-week row on the published frame."""
    frame = feat.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    sheet = roster.copy()
    sheet["player_id"] = sheet["player_id"].astype(str)
    found: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] != "pool":
            continue
        entry = load_entry(int(row["entry_id"]))
        played = next(item for item in entry["gameweeks"] if int(item["gw"]) == int(row["gw"]))
        names = {
            player_key(player["id"]): str(player.get("name") or player["id"])
            for player in list(played["xi"]) + list(played["bench"])
        }
        ids = list(names)
        present = set(frame.loc[pd.to_numeric(frame["gw"], errors="coerce") == int(row["gw"]), "player_id"])
        for pid in ids:
            if pid in present:
                continue
            played_row = sheet.loc[
                (sheet["player_id"] == pid) & (pd.to_numeric(sheet["gw"], errors="coerce") == int(row["gw"]))
            ]
            minutes = None if played_row.empty else float(pd.to_numeric(played_row["minutes"], errors="coerce").sum())
            found.append(
                {
                    "label": row["label"],
                    "gw": int(row["gw"]),
                    "chip": row["chip"],
                    "name": names[pid],
                    "minutes": minutes,
                }
            )
    return found


def write_report(
    path: Any,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    missing: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# Squad frontier",
        "",
        "Counted after the lock in `reports/squad_frontier_plan.md`. "
        "The gap is model `xi_xp` minus human `xi_xp`, divided by the priced steps that chip uses. "
        "A wildcard sums the window. A free hit uses the decision week. "
        "Realised points are the fifteen's points that week, and they do not choose the call. "
        "The score is not changed. The force-in of extra human players was not run.",
        "",
        (
            f"Chip weeks: {summary['n']}. "
            "The reference manager is absent. Bench Boost and Triple Captain are absent."
        ),
        "",
        "## Reachability",
        "",
        "A reachable week is a legal fifteen from the model's pre-chip bank and sell prices, with every player in that week's pool and a decision-week score already on the row. Money is a squad that passes shape and the club cap and costs more than that budget. Pool is a player missing from the pool or missing a score. Rules is squad shape, the club cap, or a fifteen that cannot form an eleven. A rules mismatch is the human's own budget rejecting the fifteen.",
        "",
        "| Chip | Weeks | Reachable | Money | Pool | Rules | Mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for chip in CHIPS:
        block = summary["chips"][chip]
        lines.append(
            f"| {chip} | {block['n']} | {block['n_reachable']} | {block['n_money']} | "
            f"{block['n_pool']} | {block['n_rules']} | {block['n_mismatch']} |"
        )
    lines.extend(["", "## The call", ""])
    lines.append("The call uses reachable weeks only. Each of those weeks has equal weight. Near is a per-week gap of at most 1.0. Far is above 4.0.")
    lines.append("")
    lines.append("| Chip | Reachable | Near | Middle | Far | Near share | Far share | Mean gap | Call |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for chip in CHIPS:
        block = summary["chips"][chip]
        lines.append(
            f"| {chip} | {block['n_reachable']} | {block['n_near']} | {block['n_middle']} | {block['n_far']} | "
            f"{_fmt_share(block['n_near'], block['n_reachable'])} | "
            f"{_fmt_share(block['n_far'], block['n_reachable'])} | "
            f"{_fmt_xp(block['mean'])} | {block['call']} |"
        )
    lines.append("")
    for chip in CHIPS:
        lines.append(f"{chip}: {_meaning(summary['chips'][chip]['call'])}")
        lines.append("")
    unpriced = sum(str(row.get("human_account")) == "unpriced" for row in rows)
    lines.append(
        f"The human's own budget could not be priced on {unpriced} weeks. Those weeks are not a rules mismatch."
    )
    lines.append("")
    held_out = [
        row for row in rows
        if row["gap_per"] is not None and row["gap_per"] == row["gap_per"] and row["status"] != "reachable"
    ]
    if held_out:
        lines.append("A gap on a week that is not reachable is printed here and stays out of the call.")
        lines.append("")
        for row in held_out:
            lines.append(
                f"{row['label']}, Gameweek {int(row['gw'])} {row['chip']}: "
                f"per-week gap {_fmt_xp(row['gap_per'])}, status {row['status']}, "
                f"shortfall {_fmt_pounds(row['shortfall'])}. "
                f"Human spend by position, in £m, is "
                f"goalkeeper {_fmt_pounds(row['spend_gkp_human'])}, "
                f"defence {_fmt_pounds(row['spend_def_human'])}, "
                f"midfield {_fmt_pounds(row['spend_mid_human'])}, "
                f"forward {_fmt_pounds(row['spend_fwd_human'])}. "
                f"The rebuild is "
                f"goalkeeper {_fmt_pounds(row['spend_gkp_model'])}, "
                f"defence {_fmt_pounds(row['spend_def_model'])}, "
                f"midfield {_fmt_pounds(row['spend_mid_model'])}, "
                f"forward {_fmt_pounds(row['spend_fwd_model'])}."
            )
            lines.append("")
    if missing:
        lines.append(
            "Each pool week has at least one player with no row on the published frame that week. "
            "The frame keeps a player once he has three prior appearances. No score is filled in for the weeks below."
        )
        lines.append("")
        lines.append("| Manager | GW | Chip | Player | Minutes |")
        lines.append("|---|---:|---|---|---:|")
        for item in missing:
            minutes = "" if item["minutes"] is None else f"{float(item['minutes']):.0f}"
            lines.append(
                f"| {item['label']} | {int(item['gw'])} | {item['chip']} | {item['name']} | {minutes} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Weeks",
            "",
            "Overlap is how many of the human fifteen are in the rebuild. Shortfall is the model's budget deficit, in £m. Points are descriptive.",
            "",
            "| Manager | GW | Chip | Status | Gap | Overlap | Buy pool | Shortfall | Human points | Model points |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered = sorted(rows, key=lambda row: (str(row["chip"]), str(row["label"]), int(row["gw"])))
    for row in ordered:
        lines.append(
            f"| {row['label']} | {int(row['gw'])} | {row['chip']} | {row['status']} | "
            f"{_fmt_xp(row['gap_per'])} | {int(row['overlap'])} | {int(row['buy_pool_human'])} | "
            f"{_fmt_pounds(row['shortfall'])} | {_fmt_xp(row['points_human'])} | {_fmt_xp(row['points_model'])} |"
        )
    lines.extend(
        [
            "",
            "Gemini kept the count ([squad frontier](bc-9194ff85-d0a7-5b7b-a9e9-12f9524f4cac)).",
            "",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    """Replay the carried cohort and score the human chip fifteens. Does not retune."""
    specs = [row for row in cohort_specs() if row["group"] != "reference"]
    if len(specs) != 14:
        raise RuntimeError("the frontier cohort is not 14 managers")
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    early = _early(feat)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"frontier {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        if not result["finished"]:
            raise RuntimeError(f"{spec['label']} did not finish: {result['error']}")
        entry = load_entry(int(spec["entry_id"]))
        for week in result["weeks"]:
            if week["his_chip"] not in CHIP_SQUADS:
                continue
            rows.append(
                score_week(week, entry, spec, feat, roster, clubs, horizon_scores, early)
            )
    if not rows:
        raise RuntimeError("the cohort has no chip week")
    summary = summarise(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WEEKS_CSV, index=False)
    missing = unscored_players(rows, feat, roster)
    write_report(REPORT, rows, summary, missing)
    for chip in CHIPS:
        block = summary["chips"][chip]
        print(
            f"{chip} call {block['call']} reachable {block['n_reachable']} "
            f"mean {block['mean']}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    run()
