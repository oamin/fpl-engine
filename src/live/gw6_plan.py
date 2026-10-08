"""Gameweek 6 plans from the current squad.

Gameweek 6 uses ``score_xp`` with the minutes file, overwritten when the
current bootstrap says a player is ruled out or doubtful. Each later week
that has a 1X2 for every club uses the same shares on that week's line.
The first week without a full line stops the horizon. Later weeks are not
copied. The three plans are the held squad, one free transfer, and a wildcard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.live.deadline import (
    ENTRY_PATH,
    LOG_PATH,
    ODDS_PATH,
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
)
from src.live.fpl_snapshot import refresh as refresh_fpl
from src.live.lines import refresh_lines
from src.live.scorer import (
    SCORE_COL,
    deadline_shares,
    player_key,
    price_half,
    score_on_line,
)
from src.live.whatif import (
    DECISION_GW,
    MINUTES_PATH,
    WhatIfError,
    _directory,
    _market,
    _moves,
    _outlook,
    _people,
    _score_frame,
    _squad_rows,
    visible_weeks,
)
from src.models.open_horizon import opening_pots_by_team_gw
from src.models.season_climb_ft import choose_transfers, rebuild_squad
from src.rules.fpl_2026 import FIRST_HALF_END_GW

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "predictions" / "2026-27" / "gw06" / "live_20261008"
REPORT_PATH = ROOT / "reports" / "gw6_plan_20261008.md"
WATCH = (
    (8, "Calafiori"),
    (411, "Haaland"),
    (154, "Cole Palmer"),
    (427, "Mbeumo"),
)


def news_minutes(
    status: str,
    chance: object,
    file_minutes: float | None,
    last_minutes: float | None,
) -> tuple[float, str]:
    """Minutes for this week from the file, then the current availability flag.

    A firm starter keeps the file. A player ruled out is 0. A stated chance
    below 100 scales the file, or the last observed minutes when the file
    has no row.
    """
    try:
        chance_num = None if chance is None or chance == "" else float(chance)
    except (TypeError, ValueError):
        chance_num = None
    base = file_minutes if file_minutes is not None else last_minutes
    if status in {"s", "u", "i"} and (chance_num is None or chance_num <= 0.0):
        return 0.0, "ruled out"
    if status in {"s", "u"} or chance_num == 0.0:
        return 0.0, "ruled out"
    if chance_num is not None and chance_num < 100.0:
        src = 0.0 if base is None else float(base)
        return src * chance_num / 100.0, "doubtful"
    if file_minutes is not None:
        return float(file_minutes), "minutes file"
    if last_minutes is not None:
        return float(last_minutes), "last observed"
    return 0.0, "no history"


def _chance(element: Mapping[str, Any]) -> object:
    return element.get("chance_of_playing_next_round")


def tagged_minutes(
    bootstrap: Mapping[str, Any],
    supplied: Mapping[int, float],
    last: Mapping[int, float],
) -> tuple[dict[str, float], dict[int, str]]:
    """One minutes value and one tag per player in the current bootstrap."""
    values: dict[str, float] = {}
    tags: dict[int, str] = {}
    for element in bootstrap["elements"]:
        pid = int(element["id"])
        minutes, tag = news_minutes(
            str(element.get("status") or ""),
            _chance(element),
            supplied.get(pid),
            last.get(pid),
        )
        values[player_key(pid)] = minutes
        tags[pid] = tag
    return values, tags


def _history(logs: pd.DataFrame, pid: int) -> list[tuple[int, float, float]]:
    rows = logs.loc[pd.to_numeric(logs["player_id"], errors="coerce") == pid]
    rows = rows.loc[pd.to_numeric(rows["gw"], errors="coerce").between(1, 5)]
    found = []
    for row in rows.sort_values("gw").itertuples(index=False):
        found.append((int(row.gw), float(row.total_points), float(row.minutes)))
    return found


def _element(bootstrap: Mapping[str, Any], pid: int) -> Mapping[str, Any]:
    for element in bootstrap["elements"]:
        if int(element["id"]) == pid:
            return element
    raise WhatIfError(f"player {pid} is not in the bootstrap")


def _project(
    element: Mapping[str, Any],
    names: Mapping[int, str],
    shares: pd.DataFrame,
    minutes: float,
    pots: Mapping[tuple[int, str], list],
    weeks: list[int],
) -> list[tuple[int, float | None]]:
    key = player_key(int(element["id"]))
    from src.teams import norm_team

    club_name = names.get(int(element["team"]))
    club = norm_team(club_name or "")
    share_xg = share_xa = defcon = 0.0
    if not shares.empty:
        hit = shares.loc[shares["player_id"].astype(str) == key]
        if not hit.empty:
            src = hit.iloc[0]
            share_xg = float(src["share_xG"])
            share_xa = float(src["share_xA"])
            defcon = float(src["exp_defcon_hit"])
    from src.live.fpl_snapshot import ELEMENT

    position = ELEMENT[int(element["element_type"])]
    out = []
    for gw in weeks:
        quotes = pots.get((int(gw), club), [])
        if not quotes:
            out.append((int(gw), None))
            continue
        out.append(
            (
                int(gw),
                score_on_line(
                    position=position,
                    xmi=minutes,
                    share_xg=share_xg,
                    share_xa=share_xa,
                    exp_defcon_hit=defcon,
                    pot=quotes[0],
                ),
            )
        )
    return out


def _money(tenths: int) -> str:
    return f"{tenths / 10:.1f}"


def _num(value: float) -> str:
    return f"{value:.2f}"


def build_report(
    *,
    entry_path: Path = ENTRY_PATH,
    log_path: Path = LOG_PATH,
    odds_path: Path = ODDS_PATH,
    minutes_path: Path = MINUTES_PATH,
    live_dir: Path = OUT_DIR,
    gw: int = DECISION_GW,
) -> str:
    """Refresh the free FPL files and the odds, then write the three plans."""
    live_dir.mkdir(parents=True, exist_ok=True)
    snap = refresh_fpl(live_dir)
    bootstrap = snap["bootstrap"]
    fixtures = snap["fixtures"]
    names = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    odds_result = refresh_lines(
        fixtures=fixtures,
        team_names=names,
        lines_path=live_dir / "gw_lines.csv",
        raw_path=live_dir / "odds_api_trial.json",
        meta_path=live_dir / "odds_api_meta.json",
    )
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    logs = pd.read_csv(log_path)
    odds = load_odds_frame(odds_path, live_dir / "gw_lines.csv")
    supplied = load_minutes(minutes_path, int(gw)) if minutes_path.is_file() else {}
    last = last_observed_minutes(logs)
    minute_map, tags = tagged_minutes(bootstrap, supplied, last)
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    holdings = resolve_holdings(players, purchases, current_costs(bootstrap), None, sources)
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    played = {int(row["gw"]): str(row["chip"]) for row in entry.get("chips_played") or []}
    scored = price_half(
        gw=int(gw),
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=bootstrap,
        state=state,
        minutes=minute_map,
        played=played,
        week_limit=None,
    )
    line_weeks = [int(week) for week in scored.line_weeks]
    if int(gw) not in line_weeks:
        raise WhatIfError(f"GW{int(gw)} is not a priced week")
    shares = deadline_shares(logs, int(gw))
    from src.live.scorer import build_pool, roster_from_bootstrap

    pool = build_pool(
        roster_from_bootstrap(bootstrap),
        shares,
        minute_map,
        set(state.ids()),
    )
    from src.live.scorer import clubs_from_fixtures

    clubs = clubs_from_fixtures(fixtures, names, int(gw), FIRST_HALF_END_GW)
    search = visible_weeks(line_weeks, scored.step_scores, clubs, set(pool["player_id"].astype(str)))
    directory = _directory(bootstrap, holdings)
    market = _market(pool)
    decision = search["score_by_gw"][int(gw)]
    frame = _score_frame(pool, decision)
    held_outlook = _outlook(state, pool, search["score_by_gw"], line_weeks)
    import src.models.season_climb_ft as climb

    saved_hits = climb.MAX_HITS
    climb.MAX_HITS = 0
    try:
        ft_state, n_tx, hits = choose_transfers(
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
    finally:
        climb.MAX_HITS = saved_hits
    if int(hits) != 0:
        raise WhatIfError("the free-transfer plan took a hit")
    ft_outlook = _outlook(ft_state, pool, search["score_by_gw"], line_weeks)
    ft_sells, ft_buys = _moves(state, ft_state, frame, directory, market)
    rebuilt = rebuild_squad(state, frame, SCORE_COL)
    wc_outlook = _outlook(rebuilt, pool, search["score_by_gw"], line_weeks)
    wc_sells, wc_buys = _moves(state, rebuilt, frame, directory, market)
    pots = opening_pots_by_team_gw(odds, list(fixtures), names)
    lines = _render(
        team=str(entry.get("team_name") or ""),
        minutes_hash=file_hash(minutes_path) if minutes_path.is_file() else "absent",
        meta=odds_result["meta"],
        lines_path=live_dir / "gw_lines.csv",
        line_weeks=line_weeks,
        copy_note=scored.copy_note,
        held=_squad_rows(state, pool, decision, directory, market),
        held_outlook=held_outlook,
        ft=_squad_rows(ft_state, pool, decision, directory, market),
        ft_sells=ft_sells,
        ft_buys=ft_buys,
        ft_bank=int(ft_state.bank),
        ft_outlook=ft_outlook,
        n_tx=int(n_tx),
        wildcard=_squad_rows(rebuilt, pool, decision, directory, market),
        wc_sells=wc_sells,
        wc_buys=wc_buys,
        wc_bank=int(rebuilt.bank),
        wc_outlook=wc_outlook,
        bootstrap=bootstrap,
        names=names,
        logs=logs,
        shares=shares,
        minute_map=minute_map,
        tags=tags,
        pots=pots,
        through=FIRST_HALF_END_GW,
    )
    return lines


def _coverage(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    lines = ["| GW | Priced fixtures | Odds API | ESPN |", "| --- | ---: | ---: | ---: |"]
    if frame.empty:
        return lines
    for gw, block in frame.groupby(frame["gw"].astype(int)):
        source = block["source"].astype(str)
        lines.append(
            f"| {int(gw)} | {len(block)} | {int((source == 'odds_api').sum())} | "
            f"{int((source == 'espn').sum())} |"
        )
    return lines


def _outlook_table(weeks: list[int], rows: list[tuple[str, tuple]]) -> list[str]:
    header = ["Plan"]
    for gw in weeks:
        header.extend([f"GW{gw} XI", f"GW{gw} bench"])
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for label, outlook in rows:
        cells = [label]
        for _gw, xi_xp, bench_xp in outlook:
            cells.extend([_num(xi_xp), _num(bench_xp)])
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _xi(rows) -> list[str]:
    lines = [
        "| Player | Pos | Club | GW6 xP | XI |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        role = "captain" if row.captain else ("XI" if row.starting else "bench")
        lines.append(
            f"| {row.name} | {row.position} | {row.club} | {_num(row.score)} | {role} |"
        )
    return lines


def _render(**kw: Any) -> str:
    weeks: list[int] = kw["line_weeks"]
    meta = kw["meta"]
    held_xi = sum(row[1] for row in kw["held_outlook"])
    ft_xi = sum(row[1] for row in kw["ft_outlook"])
    wc_xi = sum(row[1] for row in kw["wc_outlook"])
    lines = [
        f"# {kw['team']} Gameweek {weeks[0]} plan",
        "",
        "The capture workflow is unchanged. This note is one pass of the existing "
        "score on the current fifteen. Gameweek 6 uses `score_xp` and the minutes "
        "file. A player the current bootstrap lists as suspended, unavailable, or "
        "zero chance is set to 0. A chance below 100 scales that file. Later weeks "
        "use the same shares on that week's opening line. The horizon stops at the "
        "first gameweek that is not fully priced. Those later weeks are left unknown. "
        "They are not copies of the last priced week, and they are not filled from "
        "a predicted league table.",
        "",
        f"Minutes file SHA-256 prefix `{kw['minutes_hash']}`. "
        f"Odds request regions `{meta.get('regions')}`, markets `{meta.get('markets')}`. "
        f"The bill was {meta.get('last') or '-'} credits, "
        f"{meta.get('remaining') or '-'} remaining, "
        f"{meta.get('n_events') or 0} events returned. "
        f"Reason `{meta.get('reason')}`.",
        "",
        "## Priced fixtures",
        "",
        *_coverage(kw["lines_path"]),
        "",
        f"Priced gameweeks in the plan: {', '.join(str(gw) for gw in weeks)}.",
    ]
    if kw["copy_note"]:
        lines.append(
            "The half-season scorer still records a copy for its own chip note: "
            f"{kw['copy_note']} Those copies are not in the three plans below."
        )
    lines.extend(
        [
            "",
            "## Three plans",
            "",
            "The free-transfer plan is allowed one move and no hit. "
            "The wildcard is one rebuild, paid from the bank and sales, "
            "using selling price rather than the current price.",
            "",
            *_outlook_table(
                weeks,
                [
                    ("Hold", kw["held_outlook"]),
                    ("Free transfer", kw["ft_outlook"]),
                    ("Wildcard", kw["wc_outlook"]),
                ],
            ),
            "",
            f"Sum of the priced XI columns: hold {_num(held_xi)}, "
            f"free transfer {_num(ft_xi)} ({int(kw['n_tx'])} move), "
            f"wildcard {_num(wc_xi)}.",
            "",
            "The free-transfer search scores both priced weeks. "
            "The wildcard squad is the one-week rebuild on the Gameweek 6 scores, "
            "and the Gameweek 7 column is that squad priced afterwards. "
            "The captain is the highest Gameweek 6 score in the eleven. "
            "Adding UK and European books did not post a fixture after "
            "19 October 2026. The returned slate is still Gameweeks 6 and 7.",
            "",
            f"Free transfer bank left £{_money(kw['ft_bank'])}m. "
            f"Sells: {_people(kw['ft_sells'])}. Buys: {_people(kw['ft_buys'])}.",
            "",
            *_xi(kw["ft"]),
            "",
            f"Wildcard bank left £{_money(kw['wc_bank'])}m. "
            f"Sells: {_people(kw['wc_sells'])}. Buys: {_people(kw['wc_buys'])}.",
            "",
            *_xi(kw["wildcard"]),
            "",
            "## Held squad",
            "",
            *_xi(kw["held"]),
            "",
            "## Four players",
            "",
            "Gameweeks 1–5 are the points they scored. From Gameweek 6 the number "
            "is this model's score on the priced line. A blank cell is a gameweek "
            "with no stored 1X2 for that club. The tag is the availability rule "
            "applied to the minutes file.",
            "",
        ]
    )
    for pid, label in WATCH:
        element = _element(kw["bootstrap"], pid)
        news = str(element.get("news") or "").strip() or "none"
        chance = element.get("chance_of_playing_next_round")
        chance_text = "none" if chance is None else str(chance)
        history = _history(kw["logs"], pid)
        points = ", ".join(f"GW{gw} {pts:.0f} ({mins:.0f} min)" for gw, pts, mins in history)
        total = sum(pts for _gw, pts, _mins in history)
        minutes = float(kw["minute_map"][player_key(pid)])
        projected = _project(
            element,
            kw["names"],
            kw["shares"],
            minutes,
            kw["pots"],
            list(range(int(weeks[0]), int(kw["through"]) + 1)),
        )
        priced = ", ".join(
            f"GW{gw} {value:.2f}" if value is not None else f"GW{gw} unknown"
            for gw, value in projected
            if gw in weeks or value is not None
        )
        unknown = [gw for gw, value in projected if value is None]
        lines.extend(
            [
                f"### {label}",
                "",
                f"Status `{element.get('status')}`, chance {chance_text}, "
                f"news: {news}. Tag `{kw['tags'][pid]}`. "
                f"Minutes used for Gameweek {weeks[0]}: {minutes:.0f}.",
                "",
                f"Gameweeks 1–5: {points or 'no rows'}. Total {total:.0f}.",
                "",
                f"Model: {priced or 'no priced week'}.",
                "",
                "Unpriced gameweeks: "
                + (", ".join(str(gw) for gw in unknown) if unknown else "none")
                + ".",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    note = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(note, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
