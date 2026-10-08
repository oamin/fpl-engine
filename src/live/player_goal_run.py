"""Apply Gameweek 6 anytime prices to the saved plan. One paid request per fixture."""

from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from src.live.deadline import (
    ENTRY_PATH,
    LOG_PATH,
    ODDS_PATH,
    current_costs,
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
from src.live.fpl_snapshot import ELEMENT
from src.live.gw6_plan import WATCH, tagged_minutes
from src.live.lines import load_key, loose_team
from src.live.player_goals import (
    LIVE_DIR,
    RAW_PATH,
    REPORT_PATH,
    fetch_events,
    gw6_events,
    index_players,
    match_player,
    poisson_mean,
    quotes_from_event,
    team_goal_rates,
)
from src.live.scorer import (
    SCORE_COL,
    build_pool,
    clubs_from_fixtures,
    deadline_shares,
    player_key,
    price_half,
    roster_from_bootstrap,
    score_on_line,
)
from src.live.whatif import (
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
from src.teams import norm_team


def _club_ids(home: str, away: str, teams: list[Mapping[str, Any]]) -> dict[str, int]:
    found: dict[str, int] = {}
    for team in teams:
        key = loose_team(str(team["name"]))
        found[key] = int(team["id"])
    return {
        "home": found.get(loose_team(home), -1),
        "away": found.get(loose_team(away), -1),
    }


def goal_rates(
    events: list[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    shares: pd.DataFrame,
    minute_map: Mapping[str, float],
    pots: Mapping[tuple[int, str], list],
    names: Mapping[int, str],
) -> tuple[dict[int, float], dict[str, int]]:
    """Element id to Gameweek 6 expected goals. Unmatched names are counted."""
    index = index_players(list(bootstrap["elements"]))
    by_id = {int(element["id"]): element for element in bootstrap["elements"]}
    share_of: dict[str, float] = {}
    if not shares.empty:
        for row in shares.itertuples(index=False):
            share_of[str(row.player_id)] = float(row.share_xG)
    rates: dict[int, float] = {}
    counts = {"quoted": 0, "matched": 0, "unmatched": 0}
    for event in events:
        home = str(event.get("home_team") or "")
        away = str(event.get("away_team") or "")
        clubs = _club_ids(home, away, list(bootstrap["teams"]))
        quoted = quotes_from_event(event)
        counts["quoted"] += len(quoted)
        matched: dict[int, float] = {}
        for quote in quoted:
            pid = match_player(str(quote["name"]), index)
            if pid is None or pid not in by_id:
                counts["unmatched"] += 1
                continue
            team = int(by_id[pid]["team"])
            if team not in {clubs["home"], clubs["away"]}:
                counts["unmatched"] += 1
                continue
            _probability, mean = poisson_mean(list(quote["prices"]))
            matched[pid] = mean
            counts["matched"] += 1
        for side, club_id in clubs.items():
            del side
            if club_id < 0:
                continue
            club_name = names.get(club_id)
            if not club_name:
                continue
            pot_rows = pots.get((6, norm_team(club_name)), [])
            if not pot_rows:
                continue
            lam = float(pot_rows[0]["lam_scored"])
            players = []
            for element in bootstrap["elements"]:
                if int(element["team"]) != club_id:
                    continue
                pid = int(element["id"])
                players.append(
                    {
                        "id": pid,
                        "share": share_of.get(player_key(pid), 0.0),
                        "xmi": float(minute_map.get(player_key(pid), 0.0)),
                        "mu_raw": matched.get(pid),
                    }
                )
            rates.update(team_goal_rates(players, lam))
    return rates, counts


def _score_map(
    pool: pd.DataFrame,
    pots: Mapping[tuple[int, str], list],
    rates: Mapping[int, float],
    gw: int,
) -> dict[str, float]:
    """Gameweek scores. A priced goal rate replaces that player's share."""
    scores: dict[str, float] = {}
    for row in pool.itertuples(index=False):
        quotes = pots.get((int(gw), str(row.team_norm)), [])
        if not quotes:
            scores[str(row.player_id)] = 0.0
            continue
        share = float(row.share_xG)
        pid = int(str(row.player_id).rsplit(":", 1)[-1])
        if pid in rates and float(quotes[0]["lam_scored"]) > 0:
            share = float(rates[pid]) / float(quotes[0]["lam_scored"])
        scores[str(row.player_id)] = score_on_line(
            position=str(row.position),
            xmi=float(row.minutes),
            share_xg=share,
            share_xa=float(row.share_xA),
            exp_defcon_hit=float(row.exp_defcon_hit),
            pot=quotes[0],
        )
    return scores


def _load_saved() -> tuple[dict[str, Any], list[Any], dict[int, str]]:
    bootstrap = json.loads((LIVE_DIR / "bootstrap.json").read_text(encoding="utf-8"))
    fixtures = json.loads((LIVE_DIR / "fixtures.json").read_text(encoding="utf-8"))
    names = {int(row["id"]): str(row["name"]) for row in bootstrap["teams"]}
    return bootstrap, fixtures, names


def run() -> str:
    bootstrap, fixtures, names = _load_saved()
    slate = json.loads((LIVE_DIR / "odds_api_trial.json").read_text(encoding="utf-8"))
    events = gw6_events(slate)
    secret = load_key()
    if not secret:
        raise RuntimeError("no odds key")
    payload, notes = fetch_events(events, secret)
    RAW_PATH.write_text(json.dumps(payload), encoding="utf-8")
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    logs = pd.read_csv(LOG_PATH)
    odds = load_odds_frame(ODDS_PATH, LIVE_DIR / "gw_lines.csv")
    from src.live.whatif import MINUTES_PATH

    supplied = load_minutes(MINUTES_PATH, 6)
    last = last_observed_minutes(logs)
    minute_map, tags = tagged_minutes(bootstrap, supplied, last)
    players = final_players(entry)
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    sources = {pid: purchase_source(entry, pid) for pid in purchases}
    holdings = resolve_holdings(players, purchases, current_costs(bootstrap), None, sources)
    state = holdings_state(holdings, int(entry["bank"]), int(entry["ft_for_next"]))
    played = {int(row["gw"]): str(row["chip"]) for row in entry.get("chips_played") or []}
    scored = price_half(
        gw=6,
        logs=logs,
        odds=odds,
        fixtures=fixtures,
        bootstrap=bootstrap,
        state=state,
        minutes=minute_map,
        played=played,
        week_limit=None,
    )
    shares = deadline_shares(logs, 6)
    pots = opening_pots_by_team_gw(odds, list(fixtures), names)
    rates, counts = goal_rates(payload, bootstrap, shares, minute_map, pots, names)
    pool = build_pool(roster_from_bootstrap(bootstrap), shares, minute_map, set(state.ids()))
    line_weeks = [int(week) for week in scored.line_weeks]
    clubs = clubs_from_fixtures(fixtures, names, 6, FIRST_HALF_END_GW)
    base_search = visible_weeks(
        line_weeks, scored.step_scores, clubs, set(pool["player_id"].astype(str))
    )
    overlaid = {gw: dict(scores) for gw, scores in base_search["score_by_gw"].items()}
    overlaid[6] = _score_map(pool, pots, rates, 6)
    directory = _directory(bootstrap, holdings)
    market = _market(pool)

    def plans(scores: dict[int, dict[str, float]]) -> dict[str, Any]:
        search = dict(base_search)
        search["score_by_gw"] = scores
        decision = scores[6]
        frame = _score_frame(pool, decision)
        import src.models.season_climb_ft as climb

        saved = climb.MAX_HITS
        climb.MAX_HITS = 0
        try:
            ft_state, n_tx, hits = choose_transfers(
                state,
                frame,
                SCORE_COL,
                6,
                list(search["future_gws"]),
                search["roster_by_gw"],
                score_by_gw=scores,
                clubs=search["clubs"],
                bench_gw=None,
            )
        finally:
            climb.MAX_HITS = saved
        if int(hits) != 0:
            raise RuntimeError("the free-transfer plan took a hit")
        rebuilt = rebuild_squad(state, frame, SCORE_COL)
        return {
            "held": _outlook(state, pool, scores, line_weeks),
            "ft": _outlook(ft_state, pool, scores, line_weeks),
            "wc": _outlook(rebuilt, pool, scores, line_weeks),
            "ft_moves": _moves(state, ft_state, frame, directory, market),
            "wc_moves": _moves(state, rebuilt, frame, directory, market),
            "n_tx": int(n_tx),
            "ft_squad": _squad_rows(ft_state, pool, decision, directory, market),
            "wc_squad": _squad_rows(rebuilt, pool, decision, directory, market),
        }

    before = plans(base_search["score_by_gw"])
    after = plans(overlaid)
    text = _render(
        notes=notes,
        counts=counts,
        rates=rates,
        tags=tags,
        minute_map=minute_map,
        before=before,
        after=after,
        bootstrap=bootstrap,
        names=names,
        pots=pots,
        shares=shares,
    )
    REPORT_PATH.write_text(text, encoding="utf-8")
    return text


def _xi_sum(outlook: tuple) -> float:
    return sum(float(row[1]) for row in outlook)


def _watch_lines(
    bootstrap: Mapping[str, Any],
    rates: Mapping[int, float],
    minute_map: Mapping[str, float],
    tags: Mapping[int, str],
    pots: Mapping[tuple[int, str], list],
    shares: pd.DataFrame,
    names: Mapping[int, str],
) -> list[str]:
    by_id = {int(element["id"]): element for element in bootstrap["elements"]}
    share_of: dict[str, float] = {}
    if not shares.empty:
        for row in shares.itertuples(index=False):
            share_of[str(row.player_id)] = float(row.share_xG)
    lines = [
        "| Player | Tag | Minutes | Old goals | Book goals | Old GW6 | Book GW6 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pid, _label in WATCH:
        element = by_id[pid]
        key = player_key(pid)
        club = norm_team(names[int(element["team"])])
        pot_rows = pots.get((6, club), [])
        lam = float(pot_rows[0]["lam_scored"]) if pot_rows else 0.0
        share = share_of.get(key, 0.0)
        old_goals = share * lam
        new_goals = float(rates[pid]) if pid in rates else old_goals
        minutes = float(minute_map.get(key, 0.0))
        old_xp = new_xp = 0.0
        if pot_rows:
            assist = 0.0
            defcon = 0.0
            if not shares.empty:
                hit = shares.loc[shares["player_id"].astype(str) == key]
                if not hit.empty:
                    assist = float(hit.iloc[0]["share_xA"])
                    defcon = float(hit.iloc[0]["exp_defcon_hit"])
            old_xp = score_on_line(
                position=ELEMENT[int(element["element_type"])],
                xmi=minutes,
                share_xg=share,
                share_xa=assist,
                exp_defcon_hit=defcon,
                pot=pot_rows[0],
            )
            used = new_goals / lam if lam > 0 else share
            new_xp = score_on_line(
                position=ELEMENT[int(element["element_type"])],
                xmi=minutes,
                share_xg=used,
                share_xa=assist,
                exp_defcon_hit=defcon,
                pot=pot_rows[0],
            )
        lines.append(
            f"| {element.get('web_name')} | {tags.get(pid)} | {minutes:.0f} | "
            f"{old_goals:.2f} | {new_goals:.2f} | {old_xp:.2f} | {new_xp:.2f} |"
        )
    return lines


def _plan_row(label: str, outlook: tuple) -> str:
    cells = [label]
    for _gw, xi_xp, bench_xp in outlook:
        cells.append(f"{float(xi_xp):.2f}")
        cells.append(f"{float(bench_xp):.2f}")
    return "| " + " | ".join(cells) + " |"


def _render(**kw: Any) -> str:
    before = kw["before"]
    after = kw["after"]
    lines = [
        "# Gameweek 6 anytime goalscorer trial",
        "",
        "Historical `score_xp` is unchanged. This note replaces Gameweek 6 goals "
        "for players with a US anytime price. The price is the mean of 1/decimal, "
        "clipped, then converted with -ln(1-p), multiplied by minutes/90, and "
        "reduced when the priced players on a club would exceed the match line "
        "left after unpriced shares. A missing price keeps share times team λ. "
        "Gameweek 7 is the match line only. Gameweek 8 onward stays unknown: "
        "the Odds API sports list has no Premier League winner, top-four, or "
        "relegation market, and a two-match attack rate is not used as a ranking.",
        "",
        "Requests, region `us`, market `player_goal_scorer_anytime`:",
        "",
        *[f"- {note}" for note in kw["notes"]],
        "",
        f"Quoted names {kw['counts']['quoted']}. "
        f"Matched {kw['counts']['matched']}. "
        f"Unmatched {kw['counts']['unmatched']}.",
        "",
        "## Four players",
        "",
        *_watch_lines(
            kw["bootstrap"],
            kw["rates"],
            kw["minute_map"],
            kw["tags"],
            kw["pots"],
            kw["shares"],
            kw["names"],
        ),
        "",
        "## Plans on the match line",
        "",
        "| Plan | GW6 XI | GW6 bench | GW7 XI | GW7 bench |",
        "| --- | ---: | ---: | ---: | ---: |",
        _plan_row("Hold", before["held"]),
        _plan_row("Free transfer", before["ft"]),
        _plan_row("Wildcard", before["wc"]),
        "",
        f"Free transfer: sells {_people(before['ft_moves'][0])}; "
        f"buys {_people(before['ft_moves'][1])}.",
        "",
        "## Plans with Gameweek 6 book goals",
        "",
        "| Plan | GW6 XI | GW6 bench | GW7 XI | GW7 bench |",
        "| --- | ---: | ---: | ---: | ---: |",
        _plan_row("Hold", after["held"]),
        _plan_row("Free transfer", after["ft"]),
        _plan_row("Wildcard", after["wc"]),
        "",
        f"Free transfer ({after['n_tx']} move): "
        f"sells {_people(after['ft_moves'][0])}; "
        f"buys {_people(after['ft_moves'][1])}.",
        "",
        f"Wildcard sells {_people(after['wc_moves'][0])}. "
        f"Buys {_people(after['wc_moves'][1])}.",
        "",
        "Two-week XI sums, match line then book goals: "
        f"hold {_xi_sum(before['held']):.2f} to {_xi_sum(after['held']):.2f}, "
        f"free transfer {_xi_sum(before['ft']):.2f} to {_xi_sum(after['ft']):.2f}, "
        f"wildcard {_xi_sum(before['wc']):.2f} to {_xi_sum(after['wc']):.2f}.",
        "",
        "This is a one-week overlay. It does not count toward a chip rule.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    print(REPORT_PATH)
    run()


if __name__ == "__main__":
    main()
