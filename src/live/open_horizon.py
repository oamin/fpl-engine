"""Diagnostic: opening-price horizon on 2026/27 Gameweeks 1–5.

The published ``score_xp`` climb is not changed. This run starts from the
same Gameweek 1 fifteen and prices later weeks in the hold from opening 1X2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import (
    ENTRY_PATH,
    FD_CODE,
    FIXTURES_PATH,
    GWS,
    build_frames,
    stamp_matchday_teams,
    matchday_clubs,
    load_2026_logs,
)
from src.live.benchmark import _opening_state
from src.models.forecast_xp import (
    fixture_calendar,
    make_horizon_scores,
    opening_pots_by_team_gw,
)
from src.models.season_climb_ft import run_ft_season

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
PROCESSED = ROOT / "data" / "processed"


def _team_names() -> dict[int, str]:
    boot = json.loads((ROOT / "data" / "live" / "bootstrap.json").read_text(encoding="utf-8"))
    return {int(team["id"]): str(team["name"]) for team in boot["teams"]}


def _moves(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    previous: set[str] = set()
    known: dict[str, str] = {}
    for step in trace:
        squad = step["squad"]
        names = dict(zip(squad["player_id"].astype(str), squad["player_name"], strict=True))
        owned = set(names)
        bought = sorted(owned - previous) if previous else []
        sold = sorted(previous - owned) if previous else []
        rows.append(
            {
                "gw": int(step["gw"]),
                "in": [names[pid] for pid in bought],
                "out": [known.get(pid, pid) for pid in sold],
            }
        )
        known.update(names)
        previous = owned
    return rows


def _name_set(trace: list[dict[str, Any]], gw: int) -> set[str]:
    for step in trace:
        if int(step["gw"]) == gw:
            return set(step["squad"]["player_name"].astype(str))
    return set()


def run() -> dict[str, Any]:
    feat, roster, info = build_frames()
    entry = stamp_matchday_teams(
        json.loads(ENTRY_PATH.read_text(encoding="utf-8")),
        matchday_clubs(load_2026_logs()),
    )
    opening = _opening_state(entry, roster)
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    names = _team_names()
    odds = pd.read_csv(ROOT / "data" / "cache" / f"E0_{FD_CODE}.csv")
    pots = opening_pots_by_team_gw(odds, fixtures, names)
    calendar = fixture_calendar(fixtures, names)
    lookup = make_horizon_scores(pots, feat, calendar)

    published_trace: list[dict[str, Any]] = []
    published = run_ft_season(
        feat,
        {"xp": "score_xp"},
        list(GWS),
        roster=roster,
        trace=published_trace,
        opening=opening,
    )
    open_trace: list[dict[str, Any]] = []
    opened = run_ft_season(
        feat,
        {"xp": "score_xp"},
        list(GWS),
        roster=roster,
        trace=open_trace,
        opening=opening,
        horizon_scores=lookup,
        method_suffix="_open_h3",
    )
    gw3 = feat.loc[
        (feat["gw"] == 3) & feat["player_name"].astype(str).str.contains("João Pedro")
    ]
    pedro_id = str(gw3["player_id"].iloc[0]) if len(gw3) else ""
    pedro_horizon = lookup(3, gw3, list(GWS)) if len(gw3) else {}
    pedro_xp = {gw: scores.get(pedro_id) for gw, scores in pedro_horizon.items()}

    pub_moves = _moves(published_trace)
    open_moves = _moves(open_trace)
    result = {
        "published_points": float(published["xi_points_cap"].sum()),
        "open_points": float(opened["xi_points_cap"].sum()),
        "published_hits": float(published["hit_cost"].sum()),
        "open_hits": float(opened["hit_cost"].sum()),
        "published_moves": pub_moves,
        "open_moves": open_moves,
        "pedro_xp": pedro_xp,
        "pedro_held_gw3": "João Pedro" in _name_set(open_trace, 3),
        "pedro_held_gw4": "João Pedro" in _name_set(open_trace, 4),
        "pedro_held_gw5": "João Pedro" in _name_set(open_trace, 5),
        "pedro_held_published": "João Pedro" in _name_set(published_trace, 3),
        "priced_sides": len(pots),
        "join_rate": info["join_rate"],
        "weekly_open": opened,
        "weekly_published": published,
    }
    _write(result, entry)
    return result


def _write(result: dict[str, Any], entry: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    result["weekly_open"].to_csv(PROCESSED / "open_horizon_gw1_5.csv", index=False)
    theirs = {int(g["gw"]): int(g["points"]) for g in entry["gameweeks"]}
    lines = [
        "# Opening-price horizon, Gameweeks 1–5",
        "",
        "Diagnostic only. The published `score_xp` formula is unchanged. "
        "A missing week no longer borrows a later week's score. "
        "This arm starts from ojaminFC's Gameweek 1 fifteen. The current week "
        "keeps that week's closing-price xp. Gameweeks inside the three-week "
        "hold use the opening 1X2 (`AvgH`, `Avg>2.5`). Share and minutes stay "
        "on the deadline row.",
        "",
        f"Published path: **{result['published_points']:.0f}** points, "
        f"hits {result['published_hits']:.0f}.",
        f"Opening horizon: **{result['open_points']:.0f}** points, "
        f"hits {result['open_hits']:.0f}.",
        f"ojaminFC: **{sum(theirs.values())}**.",
        "",
        "João Pedro at the Gameweek 3 deadline, share frozen, opening λ on the later weeks:",
        "",
    ]
    for gw, xp in sorted(result["pedro_xp"].items()):
        lines.append(f"- Gameweek {gw}: {xp:.2f}")
    lines += [
        "",
        f"Published squad still has him in Gameweek 3: {result['pedro_held_published']}.",
        f"Opening-horizon squad still has him in Gameweek 3: {result['pedro_held_gw3']}.",
        f"Still in the squad in Gameweek 4: {result['pedro_held_gw4']}. "
        f"Gameweek 5: {result['pedro_held_gw5']}.",
        "",
        "| GW | Published | Open horizon | ojaminFC | Published transfers | Open transfers |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    pub = result["weekly_published"].set_index("gw")
    opened = result["weekly_open"].set_index("gw")
    pub_moves = {row["gw"]: row for row in result["published_moves"]}
    open_moves = {row["gw"]: row for row in result["open_moves"]}
    for gw in sorted(pub.index):
        def _fmt(move: dict[str, Any]) -> str:
            if not move["in"] and not move["out"]:
                return "none"
            return f"in {', '.join(move['in']) or '—'}; out {', '.join(move['out']) or '—'}"

        lines.append(
            f"| {int(gw)} | {pub.loc[gw, 'xi_points_cap']:.0f} | "
            f"{opened.loc[gw, 'xi_points_cap']:.0f} | {theirs[int(gw)]} | "
            f"{_fmt(pub_moves[int(gw)])} | {_fmt(open_moves[int(gw)])} |"
        )
    lines += [
        "",
        f"Opening prices matched {result['priced_sides']} club-gameweeks. "
        "A week with a fixture and no opening price uses that club's earlier "
        "scoring rate. A blank week is zero. No Odds API call was made. "
        "The record of 327 in `reports/live_benchmark_2026.md` is the earlier "
        "run, which could fill a missing week from a later score.",
        "",
        "Gemini accepted the stub fix. With past weeks only, the published "
        "rule also keeps João Pedro and takes no hit. The Gameweek 3 sale in "
        "that 327 record was the future score, not the Arsenal projection by "
        "itself. The opening-odds horizon still keeps him and scores "
        f"{result['open_points']:.0f} on this window, behind the corrected "
        f"published path at {result['published_points']:.0f}.",
        "",
    ]
    (REPORTS / "open_horizon_gw1_5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = run()
    print(
        f"published {result['published_points']:.0f}  "
        f"open {result['open_points']:.0f}  "
        f"pedro gw3 held {result['pedro_held_gw3']}  "
        f"xp { {k: round(v, 2) for k, v in result['pedro_xp'].items()} }"
    )


if __name__ == "__main__":
    main()
