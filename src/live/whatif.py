"""Two Gameweek 6 paths for the stored squad.

The wildcard path is one rebuild. The hold path is one free-transfer
search. Both use the priced weeks only. A later week that repeats the
last line is not in the search, and it is not added up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.live.deadline import (
    BOOTSTRAP_PATH,
    ENTRY_PATH,
    FIXTURES_PATH,
    LOG_PATH,
    apply_live_minutes,
    current_costs,
    file_hash,
    final_players,
    gameweek_values,
    holdings_state,
    last_observed_minutes,
    load_minutes,
    load_odds_frame,
    purchase_source,
    reconstruct_purchases,
    resolve_holdings,
    team_names,
)
from src.live.lines import LINES_PATH
from src.live.scorer import (
    SCORE_COL,
    build_pool,
    clubs_from_fixtures,
    deadline_shares,
    player_key,
    price_half,
    roster_from_bootstrap,
)
from src.models.half_plan_scores import squad_outlook
from src.models.season_climb import pick_xi
from src.models.season_climb_ft import SquadState, choose_transfers, rebuild_squad
from src.rules.fpl_2026 import sell_price
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "live_whatif_gw6.md"
MINUTES_PATH = ROOT / "data" / "live" / "xmi_gw6.csv"
DECISION_GW = 6
_POS = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


class WhatIfError(RuntimeError):
    """The two paths left the priced weeks or the priced rebuild."""


@dataclass(frozen=True)
class Person:
    name: str
    position: str
    club: str
    price: int
    score: float = 0.0
    starting: bool = False
    captain: bool = False


@dataclass(frozen=True)
class PathView:
    bank: int
    hits: int
    sells: tuple[Person, ...]
    buys: tuple[Person, ...]
    squad: tuple[Person, ...]
    outlook: tuple[tuple[int, float, float], ...]


@dataclass(frozen=True)
class WhatIf:
    team: str
    minutes_hash: str
    line_weeks: tuple[int, ...]
    wildcard: PathView
    hold: PathView


def visible_weeks(
    line_weeks: Sequence[int],
    step_scores: Mapping[int, Mapping[str, float]],
    clubs: Mapping[int, set[str]],
    pool_ids: set[str],
) -> dict[str, Any]:
    """The priced weeks only. A copied later week is left out."""
    weeks = [int(gw) for gw in line_weeks]
    if not weeks or weeks != sorted(weeks):
        raise WhatIfError("priced weeks are missing")
    missing = [gw for gw in weeks if int(gw) not in step_scores or int(gw) not in clubs]
    if missing:
        raise WhatIfError(f"priced weeks have no scores: {missing}")
    return {
        "future_gws": weeks,
        "score_by_gw": {gw: {str(pid): float(value) for pid, value in step_scores[gw].items()} for gw in weeks},
        "clubs": {gw: set(clubs[gw]) for gw in weeks},
        "roster_by_gw": {gw: set(pool_ids) for gw in weeks},
    }


def _directory(bootstrap: Mapping[str, Any], holdings) -> dict[str, dict[str, str]]:
    short = {
        norm_team(str(row["name"])): str(row.get("short_name") or row["name"])
        for row in bootstrap["teams"]
    }
    found: dict[str, dict[str, str]] = {}
    for element in bootstrap["elements"]:
        key = player_key(int(element["id"]))
        club = norm_team(team_names(bootstrap)[int(element["team"])])
        found[key] = {
            "name": str(element.get("web_name") or element["id"]),
            "club": short.get(club, club),
        }
    for row in holdings:
        found[row.key] = {"name": row.name, "club": row.team}
    return found


def _person(
    key: str,
    directory: Mapping[str, Mapping[str, str]],
    pool: pd.DataFrame,
    price: int,
    score: float = 0.0,
    starting: bool = False,
    captain: bool = False,
) -> Person:
    meta = directory[key]
    row = pool.loc[pool["player_id"].astype(str) == key].iloc[0]
    return Person(
        name=str(meta["name"]),
        position=str(row["position"]),
        club=str(meta["club"]),
        price=int(price),
        score=float(score),
        starting=starting,
        captain=captain,
    )


def _market(pool: pd.DataFrame) -> dict[str, int]:
    return {
        str(row.player_id): int(row.value)
        for row in pool.drop_duplicates("player_id").itertuples()
    }


def _score_frame(pool: pd.DataFrame, scores: Mapping[str, float]) -> pd.DataFrame:
    frame = pool.drop_duplicates("player_id", keep="first").copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame[SCORE_COL] = [float(scores.get(pid, 0.0)) for pid in frame["player_id"]]
    return frame


def _squad_rows(
    state: SquadState,
    pool: pd.DataFrame,
    scores: Mapping[str, float],
    directory: Mapping[str, Mapping[str, str]],
    market: Mapping[str, int],
) -> tuple[Person, ...]:
    frame = _score_frame(pool, scores)
    owned = frame.loc[frame["player_id"].isin(state.ids())].copy()
    xi, _form = pick_xi(owned, SCORE_COL)
    xi_ids = set(xi["player_id"].astype(str))
    xi_scores = pd.to_numeric(xi[SCORE_COL], errors="coerce").fillna(0.0)
    captain = str(xi.loc[xi_scores.idxmax(), "player_id"]) if len(xi) else ""
    rows = []
    for pid in state.ids():
        score = float(scores.get(pid, 0.0))
        price = int(state.purchase.get(pid, market[pid]))
        rows.append(
            _person(
                pid,
                directory,
                frame,
                price,
                score=score,
                starting=pid in xi_ids,
                captain=pid == captain,
            )
        )
    rows.sort(key=lambda row: (_POS.get(row.position, 9), not row.starting, row.name))
    return tuple(rows)


def _outlook(
    state: SquadState,
    pool: pd.DataFrame,
    step_scores: Mapping[int, Mapping[str, float]],
    weeks: Sequence[int],
) -> tuple[tuple[int, float, float], ...]:
    rows = []
    for gw in weeks:
        frame = _score_frame(pool, step_scores[int(gw)])
        view = squad_outlook(frame, state.ids(), dict(step_scores[int(gw)]), SCORE_COL)
        rows.append((int(gw), float(view.xi_xp), float(view.bench_xp)))
    return tuple(rows)


def _moves(
    before: SquadState,
    after: SquadState,
    pool: pd.DataFrame,
    directory: Mapping[str, Mapping[str, str]],
    market: Mapping[str, int],
) -> tuple[tuple[Person, ...], tuple[Person, ...]]:
    sells = []
    for pid in sorted(before.ids() - after.ids()):
        price = sell_price(int(before.purchase[pid]), int(market[pid]))
        sells.append(_person(pid, directory, pool, price))
    buys = []
    for pid in sorted(after.ids() - before.ids()):
        buys.append(_person(pid, directory, pool, int(market[pid])))
    sells.sort(key=lambda row: (_POS.get(row.position, 9), row.name))
    buys.sort(key=lambda row: (_POS.get(row.position, 9), row.name))
    return tuple(sells), tuple(buys)


def _check_rebuild(
    scored_weeks,
    outlook: Sequence[tuple[int, float, float]],
) -> None:
    by_gw = {int(row.gw): row for row in scored_weeks}
    for gw, xi_xp, bench_xp in outlook:
        row = by_gw[int(gw)]
        if row.rebuilt is None:
            raise WhatIfError(f"GW{int(gw)} has no priced rebuild")
        if abs(float(row.rebuilt.xi_xp) - float(xi_xp)) > 1e-6:
            raise WhatIfError(f"GW{int(gw)} wildcard eleven left the priced rebuild")
        if abs(float(row.rebuilt.bench_xp) - float(bench_xp)) > 1e-6:
            raise WhatIfError(f"GW{int(gw)} wildcard bench left the priced rebuild")


def build(
    *,
    entry_path: Path = ENTRY_PATH,
    log_path: Path = LOG_PATH,
    odds_path: Path | None = None,
    bootstrap_path: Path = BOOTSTRAP_PATH,
    fixtures_path: Path = FIXTURES_PATH,
    minutes_path: Path = MINUTES_PATH,
    live_path: Path | None = LINES_PATH,
    gw: int = DECISION_GW,
) -> WhatIf:
    """Price the two paths. The Odds API is not called."""
    from src.live.deadline import ODDS_PATH

    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    logs = pd.read_csv(log_path)
    odds = load_odds_frame(ODDS_PATH if odds_path is None else odds_path, live_path)
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    holdings = resolve_holdings(
        players, purchases, current_costs(bootstrap), None, sources
    )
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    if not minutes_path.is_file():
        raise WhatIfError("the minutes file is absent")
    supplied = load_minutes(minutes_path, int(gw))
    resolved = apply_live_minutes(
        [int(element["id"]) for element in bootstrap["elements"]],
        supplied,
        last_observed_minutes(logs),
    )
    minute_map = {player_key(int(row["player_id"])): float(row["xmi"]) for row in resolved}
    played = {
        int(row["gw"]): str(row["chip"]) for row in entry.get("chips_played") or []
    }
    scored = price_half(
        gw=int(gw),
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=bootstrap,
        state=state,
        minutes=minute_map,
        played=played,
    )
    line_weeks = tuple(int(week) for week in scored.line_weeks)
    if int(gw) not in line_weeks:
        raise WhatIfError(f"GW{int(gw)} is not a priced week")
    names = team_names(bootstrap)
    shares = deadline_shares(logs, int(gw))
    roster = roster_from_bootstrap(bootstrap)
    pool = build_pool(roster, shares, minute_map, set(state.ids()))
    clubs = clubs_from_fixtures(fixtures, names, int(gw), int(gw) + 1)
    search = visible_weeks(line_weeks, scored.step_scores, clubs, set(pool["player_id"].astype(str)))
    if any(week not in line_weeks for week in search["future_gws"]):
        raise WhatIfError("the search reached past the priced weeks")
    directory = _directory(bootstrap, holdings)
    market = _market(pool)
    decision = search["score_by_gw"][int(gw)]
    frame = _score_frame(pool, decision)
    rebuilt = rebuild_squad(state, frame, SCORE_COL)
    wildcard_outlook = _outlook(rebuilt, pool, search["score_by_gw"], line_weeks)
    _check_rebuild(scored.weeks, wildcard_outlook)
    sells, buys = _moves(state, rebuilt, frame, directory, market)
    hold_state, n_tx, hits = choose_transfers(
        state,
        frame,
        SCORE_COL,
        int(gw),
        list(search["future_gws"]),
        search["roster_by_gw"],
        score_by_gw=search["score_by_gw"],
        clubs=search["clubs"],
        bench_gw=None,
    )
    if int(n_tx) > int(state.ft) + 2:
        raise WhatIfError("the hold path took more than two hits")
    hold_sells, hold_buys = _moves(state, hold_state, frame, directory, market)
    return WhatIf(
        team=str(entry.get("team_name") or ""),
        minutes_hash=file_hash(minutes_path),
        line_weeks=line_weeks,
        wildcard=PathView(
            bank=int(rebuilt.bank),
            hits=0,
            sells=sells,
            buys=buys,
            squad=_squad_rows(rebuilt, pool, decision, directory, market),
            outlook=wildcard_outlook,
        ),
        hold=PathView(
            bank=int(hold_state.bank),
            hits=int(hits),
            sells=hold_sells,
            buys=hold_buys,
            squad=_squad_rows(hold_state, pool, decision, directory, market),
            outlook=_outlook(hold_state, pool, search["score_by_gw"], line_weeks),
        ),
    )


def _money(tenths: int) -> str:
    return f"{int(tenths) / 10:.1f}"


def _num(value: float) -> str:
    return f"{float(value):.2f}"


def _people(rows: Sequence[Person]) -> str:
    if not rows:
        return "none"
    return ", ".join(
        f"{row.name} ({row.position}, {row.club}, {_money(row.price)})" for row in rows
    )


def _xi_table(rows: Sequence[Person]) -> list[str]:
    lines = [
        "| Player | Pos | Club | Price | GW6 xP | XI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        role = "captain" if row.captain else ("XI" if row.starting else "bench")
        lines.append(
            f"| {row.name} | {row.position} | {row.club} | {_money(row.price)} | "
            f"{_num(row.score)} | {role} |"
        )
    return lines


def _outlook_row(label: str, view: PathView) -> str:
    cells = [label]
    for _gw, xi_xp, bench_xp in view.outlook:
        cells.append(_num(xi_xp))
        cells.append(_num(bench_xp))
    return "| " + " | ".join(cells) + " |"


def render(result: WhatIf) -> str:
    """The two priced weeks. A season total is not in this note."""
    weeks = result.line_weeks
    header = ["Path"]
    for gw in weeks:
        header.extend([f"GW{int(gw)} XI", f"GW{int(gw)} bench"])
    delta = ["Wildcard − hold"]
    for left, right in zip(result.wildcard.outlook, result.hold.outlook, strict=True):
        delta.append(_num(left[1] - right[1]))
        delta.append(_num(left[2] - right[2]))
    lines = [
        f"# {result.team} Gameweek {weeks[0]} paths",
        "",
        "Two paths from the squad already owned. Both use the minutes file and "
        "the priced weeks only. The wildcard path is one rebuild, paid from the "
        "bank and sales. The hold path is the free-transfer search on those same "
        "weeks, with the hold margin 1.25 and the switch penalty 1.0. "
        "A week after the last priced week is not in the search. "
        "The figures are this plan's expected points for the priced weeks.",
        "",
        f"Minutes file SHA-256 prefix `{result.minutes_hash}`. "
        "Prices are millions of pounds. A sale uses the rules-module formula.",
        "",
        "## Wildcard",
        "",
        f"Bank left £{_money(result.wildcard.bank)}m. "
        f"Sells: {_people(result.wildcard.sells)}. "
        f"Buys: {_people(result.wildcard.buys)}.",
        "",
        *_xi_table(result.wildcard.squad),
        "",
        "## Hold",
        "",
        f"Hits {result.hold.hits}. Bank left £{_money(result.hold.bank)}m. "
        f"Sells: {_people(result.hold.sells)}. "
        f"Buys: {_people(result.hold.buys)}.",
        "",
        *_xi_table(result.hold.squad),
        "",
        "## Priced weeks",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
        _outlook_row("Wildcard", result.wildcard),
        _outlook_row("Hold", result.hold),
        "| " + " | ".join(delta) + " |",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(result: WhatIf, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(result), encoding="utf-8")
    return path


if __name__ == "__main__":
    note = build()
    destination = write_report(note)
    print(destination)
