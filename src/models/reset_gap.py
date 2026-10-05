"""Gameweeks 1–5, reset to the squad ojaminFC held before each deadline.

The bar was locked before the total was read. A sum of model minus
official points above 0, and at least 3 of 5 weeks non-negative, is a
gain on these five decisions. Anything else is the model not beating
these five decisions. Five weeks are too few to call the rule reliable.
Gameweek 6 is not in the sum.

The search is the published rule. The model's squad is discarded. The
next week starts from the fifteen he fielded, not from the model's buys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import (
    BOOTSTRAP_PATH,
    FIXTURES_PATH,
    GWS,
    build_frames,
    player_key,
)
from src.live.entry import load_entry
from src.models.blank_context import apply_fixture_tags, clubs_by_gw
from src.models.open_horizon import attach_opening_horizon, fixture_calendar
from src.models.season_climb import bank_squad_gw
from src.models.season_climb_ft import (
    HORIZON,
    SquadState,
    _chip_additions,
    _fill_score,
    _gw_pool,
    choose_transfers,
)
from src.models.stage_40_gw15_gap import _their_final
from src.rules.fpl_2026 import HIT_COST, captain_extra_points

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ENTRY_ID = 2632584
LOOKAHEAD = 8


def beats_five(gaps: list[float]) -> bool:
    """The pre-registered bar. Five numbers, nothing else."""
    if len(gaps) != 5:
        raise RuntimeError("the bar is five weeks")
    cleaned = [0.0 if abs(g) < 1e-6 else float(g) for g in gaps]
    return sum(cleaned) > 0 and sum(g >= 0 for g in cleaned) >= 3


def horizon_gws(gw: int, clubs: dict[int, set[str]]) -> list[int]:
    """This week plus the next two weeks that have clubs.

    A blank is skipped. The list stops at ``HORIZON`` weeks.
    """
    later: list[int] = []
    nxt = int(gw) + 1
    while len(later) < HORIZON - 1 and nxt <= int(gw) + LOOKAHEAD:
        if clubs.get(nxt):
            later.append(nxt)
        nxt += 1
    return [int(gw), *later]


def bank_before(entry: dict[str, Any], gw: int) -> int:
    """Bank at the deadline, before this week's transfers.

    A later week uses the previous event's bank, which is already after
    that event. Gameweek 1 uses its own bank when that week had no
    transfers, and reverses ``in_cost`` and ``out_cost`` when it did.
    """
    weeks = {int(row["gw"]): row for row in entry["gameweeks"]}
    if int(gw) > 1:
        return int(weeks[int(gw) - 1]["bank"])
    week = weeks[1]
    made = [row for row in entry.get("transfers") or [] if int(row["gw"]) == 1]
    if not made:
        return int(week["bank"])
    spent = sum(int(row["in_cost"]) for row in made)
    raised = sum(int(row["out_cost"]) for row in made)
    return int(week["bank"]) - raised + spent


def _squad_elements(entry: dict[str, Any], gw: int) -> list[int]:
    if int(gw) == 1:
        return [int(player["id"]) for player in entry["opening_squad"]]
    weeks = {int(row["gw"]): row for row in entry["gameweeks"]}
    prev = weeks[int(gw) - 1]
    return [int(player["id"]) for player in list(prev["xi"]) + list(prev["bench"])]


def purchases_before(
    entry: dict[str, Any], gw1_prices: dict[int, int], gw: int
) -> dict[str, int]:
    """Purchase price of the fifteen owned before this deadline."""
    owned: dict[int, int] = {}
    for player in entry["opening_squad"]:
        element = int(player["id"])
        if element not in gw1_prices:
            raise RuntimeError(f"{player.get('name') or element} has no Gameweek 1 price")
        owned[element] = int(gw1_prices[element])
    for row in sorted(entry.get("transfers") or [], key=lambda item: int(item["gw"])):
        if int(row["gw"]) >= int(gw):
            continue
        out_id = int(row["out_id"])
        in_id = int(row["in_id"])
        if out_id not in owned:
            raise RuntimeError(f"GW{int(row['gw'])} sells {out_id} who is not owned")
        if in_id in owned:
            raise RuntimeError(f"GW{int(row['gw'])} buys {in_id} who is already owned")
        del owned[out_id]
        owned[in_id] = int(row["in_cost"])
    expected = set(_squad_elements(entry, gw))
    if set(owned) != expected:
        raise RuntimeError(
            f"GW{int(gw)} squad does not match the transfers: "
            f"extra {sorted(set(owned) - expected)} missing {sorted(expected - set(owned))}"
        )
    return {player_key(element): price for element, price in owned.items()}


def pre_deadline(
    entry: dict[str, Any], roster: pd.DataFrame, gw: int
) -> SquadState:
    """His fifteen, bank, and free transfers before this week's deals."""
    gw1 = roster.loc[roster["gw"] == 1].drop_duplicates("player_id", keep="first")
    prices = {
        int(str(pid).split(":")[-1]): int(value)
        for pid, value in zip(gw1["player_id"], gw1["value"], strict=True)
    }
    weeks = {int(row["gw"]): row for row in entry["gameweeks"]}
    week = weeks[int(gw)]
    return SquadState(
        purchase=purchases_before(entry, prices, gw),
        bank=bank_before(entry, gw),
        ft=int(week["ft_available"]),
        selling=None,
    )


def model_chip(his: str | None) -> str | None:
    """Mirror Triple Captain. Wildcard, Free Hit, and Bench Boost raise."""
    if his in (None, "", "triple_captain"):
        return his or None
    raise RuntimeError(f"{his} is not mirrored on these five decisions")


def classify_gap(
    pre: set[str],
    model_squad: set[str],
    their_squad: set[str],
    model_final: dict[str, float],
    their_final: dict[str, float],
) -> dict[str, Any]:
    """Split the final-eleven difference into transfers and lineup choices.

    A player in both fielded fifteens is a lineup choice. A player who
    was bought, or sold by one side, is a transfer. Shared starters cancel.
    """
    transfer_m: list[str] = []
    lineup_m: list[str] = []
    transfer_t: list[str] = []
    lineup_t: list[str] = []

    def take(pid: str, model_side: bool) -> None:
        both = pid in model_squad and pid in their_squad
        bought = pid not in pre
        sold = pid in pre and (pid not in model_squad or pid not in their_squad)
        if both:
            (lineup_m if model_side else lineup_t).append(pid)
            return
        if bought or sold:
            (transfer_m if model_side else transfer_t).append(pid)
            return
        raise RuntimeError(f"{pid} is in one final eleven and in neither squad")

    for pid in set(model_final) - set(their_final):
        take(pid, True)
    for pid in set(their_final) - set(model_final):
        take(pid, False)
    transfer_m.sort()
    lineup_m.sort()
    transfer_t.sort()
    lineup_t.sort()
    transfer = sum(model_final[pid] for pid in transfer_m) - sum(
        their_final[pid] for pid in transfer_t
    )
    lineup = sum(model_final[pid] for pid in lineup_m) - sum(
        their_final[pid] for pid in lineup_t
    )
    xi_gap = sum(model_final.values()) - sum(their_final.values())
    if abs((transfer + lineup) - xi_gap) > 1e-6:
        raise RuntimeError("the transfer and lineup pieces do not make the XI gap")
    return {
        "transfer_gap": float(transfer),
        "lineup_gap": float(lineup),
        "transfer_model": transfer_m,
        "transfer_their": transfer_t,
        "lineup_model": lineup_m,
        "lineup_their": lineup_t,
    }


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{value:+.0f}"
    return f"{value:+.2f}"


def _names(ids: list[str], names: dict[str, str]) -> str:
    if not ids:
        return ""
    return ", ".join(names.get(pid, pid) for pid in sorted(ids))


def _as_list(value: Any) -> list[str]:
    if value is None or isinstance(value, float):
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part) for part in value]


def phrases(row: dict[str, Any], names: dict[str, str] | None = None) -> list[str]:
    """One sentence per additive piece. The signed numbers sum to the gap.

    ``names`` maps an id to a display name. A row that already holds display
    names, as the CSV does, is used as it stands.
    """
    labels = names or {}

    def show(key: str) -> list[str]:
        return [labels.get(pid, pid) for pid in _as_list(row.get(key))]

    cap = float(row["captain_gap"])
    cap_word = "added" if cap >= 0 else "cost"
    lines = [
        f"Captaincy on {row['model_captain']} vs {row['their_captain']} {cap_word} {_fmt(cap)}."
    ]
    bought = [pid for pid in show("transfer_model") if pid in set(show("model_in"))]
    held = [pid for pid in show("transfer_model") if pid not in set(show("model_in"))]
    outgoing = show("transfer_their")
    tx = float(row["transfer_gap"])
    tx_word = "added" if tx >= 0 else "cost"
    chunks: list[str] = []
    if held:
        chunks.append(f"Holding {', '.join(sorted(held))}")
    if bought:
        verb = "transferring" if chunks else "Transferring"
        chunks.append(f"{verb} {', '.join(sorted(bought))} in")
    head = " and ".join(chunks) if chunks else "Holding"
    if outgoing:
        lines.append(
            f"{head} for {', '.join(sorted(outgoing))} {tx_word} {_fmt(tx)} net points."
        )
    else:
        lines.append(f"{head} {tx_word} {_fmt(tx)} net points.")
    started = show("lineup_model")
    benched = show("lineup_their")
    line = float(row["lineup_gap"])
    line_word = "added" if line >= 0 else "cost"
    if not started and not benched:
        lines.append(f"Starting the same eleven {line_word} {_fmt(line)}.")
    elif not started:
        lines.append(
            f"Starting the rest of the eleven over {', '.join(sorted(benched))} {line_word} {_fmt(line)}."
        )
    elif not benched:
        lines.append(
            f"Starting {', '.join(sorted(started))} over the rest of the eleven {line_word} {_fmt(line)}."
        )
    else:
        lines.append(
            f"Starting {', '.join(sorted(started))} over {', '.join(sorted(benched))} {line_word} {_fmt(line)}."
        )
    hits = float(row["hit_gap"])
    hit_word = "added" if hits >= 0 else "cost"
    lines.append(f"Hits {hit_word} {_fmt(hits)}.")
    residual = float(row["residual"])
    if abs(residual) >= 1e-6:
        lines.append(
            f"The rebuilt week differs from the official total by {_fmt(residual)}."
        )
    return lines


def _join(ids: list[str], names: dict[str, str]) -> str:
    return _names(ids, names)


def _points(frame: pd.DataFrame) -> dict[str, float]:
    return {
        str(row.player_id): float(row.total_points)
        for row in frame.itertuples(index=False)
    }


def _name_map(roster: pd.DataFrame, gw: int) -> dict[str, str]:
    sheet = roster.loc[roster["gw"] == int(gw)].drop_duplicates("player_id", keep="first")
    return dict(zip(sheet["player_id"].astype(str), sheet["player_name"].astype(str), strict=False))


def _pick_id(week: dict[str, Any], name: str) -> str | None:
    for player in list(week["xi"]) + list(week["bench"]):
        if player["name"] == name:
            return player_key(player["id"])
    return None


def _effective_name(
    cap_id: str,
    vice_id: str,
    minutes: dict[str, float],
    names: dict[str, str],
) -> str:
    if minutes.get(cap_id, 0.0) > 0:
        return names.get(cap_id, cap_id)
    if minutes.get(vice_id, 0.0) > 0:
        return names.get(vice_id, vice_id)
    return "nobody"


def _their_score(week: dict[str, Any], roster: pd.DataFrame, chip: str | None) -> dict[str, Any]:
    final = _their_final(week, roster)
    base = float(final["total_points"].sum())
    cap_row = final.loc[final["name"] == week["captain"]]
    vice_row = final.loc[final["name"] == week["vice"]]
    cap_pts = float(cap_row["total_points"].iloc[0]) if len(cap_row) else 0.0
    vice_pts = float(vice_row["total_points"].iloc[0]) if len(vice_row) else 0.0
    cap_played = bool(len(cap_row) and float(cap_row["minutes"].iloc[0]) > 0)
    vice_played = bool(len(vice_row) and float(vice_row["minutes"].iloc[0]) > 0)
    extra = captain_extra_points(
        cap_pts,
        vice_pts,
        captain_played=cap_played,
        vice_played=vice_played,
        chip=chip,
    )
    hits = float(week["transfer_cost"])
    if cap_played:
        captain = str(week["captain"])
    elif vice_played:
        captain = str(week["vice"])
    else:
        captain = "nobody"
    return {
        "final": final,
        "points_map": _points(final),
        "base": base,
        "extra": float(extra),
        "hits": hits,
        "built": base + float(extra) - hits,
        "captain": captain,
    }


def _early(feat: pd.DataFrame) -> pd.DataFrame | None:
    stored = feat.attrs.get("early_scores")
    if isinstance(stored, pd.DataFrame):
        return stored
    if stored:
        return pd.DataFrame(list(stored), columns=["player_id", "gw", "score_xp"])
    return None


def fixture_clubs() -> dict[int, set[str]]:
    """Clubs with a fixture in the stored live calendar. No network."""
    if not FIXTURES_PATH.exists() or not BOOTSTRAP_PATH.exists():
        return {}
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    boot = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
    names = {int(team["id"]): str(team["name"]) for team in boot["teams"]}
    found: dict[int, set[str]] = {}
    for (gw, club), count in fixture_calendar(fixtures, names).items():
        if int(count) <= 0:
            continue
        found.setdefault(int(gw), set()).add(str(club))
    return found


def merged_clubs(roster: pd.DataFrame) -> dict[int, set[str]]:
    """Played weeks keep the sheet. Later weeks use the fixture list."""
    clubs = clubs_by_gw(roster)
    for gw, names in fixture_clubs().items():
        if gw not in clubs:
            clubs[gw] = set(names)
    return clubs


def assert_not_carried(rows: list[dict[str, Any]]) -> None:
    """The next pre-deadline fifteen is his, not the model's."""
    for prev, nxt in zip(rows, rows[1:], strict=False):
        if set(nxt["pre_ids"]) != set(prev["their_ids"]):
            raise RuntimeError(
                f"GW{int(nxt['gw'])} did not start from the fifteen he fielded in GW{int(prev['gw'])}"
            )
        if set(prev["model_ids"]) != set(prev["their_ids"]) and set(nxt["pre_ids"]) == set(
            prev["model_ids"]
        ):
            raise RuntimeError(f"GW{int(nxt['gw'])} carried the model's squad")


def one_week(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    entry: dict[str, Any],
    gw: int,
    *,
    clubs: dict[int, set[str]],
    roster_by_gw: dict[int, set[str]],
    horizon_scores: Any,
) -> dict[str, Any]:
    """One deadline. The returned squad is not an input to the next week."""
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    if not clubs.get(int(gw)):
        raise RuntimeError(f"GW{int(gw)} has no clubs")
    state = pre_deadline(entry, roster, gw)
    pre_ids = state.ids()
    week = next(row for row in entry["gameweeks"] if int(row["gw"]) == int(gw))
    chip = model_chip(week.get("chip"))
    early = _early(feat)
    pool = _gw_pool(feat, roster, int(gw), pre_ids, early)
    pool = apply_fixture_tags(pool, int(gw), clubs)
    missing = pre_ids - set(pool["player_id"].astype(str))
    if missing:
        raise RuntimeError(f"GW{int(gw)} is missing owned players {sorted(missing)}")
    window = horizon_gws(int(gw), clubs)
    step_scores = horizon_scores(int(gw), pool, window)
    new_state, n_tx, hits = choose_transfers(
        state,
        pool,
        "score_xp",
        gw=int(gw),
        future_gws=window,
        roster_by_gw=roster_by_gw,
        score_by_gw=step_scores,
        clubs=clubs,
        bench_gw=None,
    )
    squad = pool.loc[pool["player_id"].astype(str).isin(new_state.ids())].copy()
    squad = squad.drop_duplicates("player_id", keep="first")
    if len(squad) != 15:
        raise RuntimeError(f"GW{int(gw)} squad has {len(squad)} players")
    for column in ("minutes", "total_points"):
        if column not in squad.columns:
            raise RuntimeError(f"GW{int(gw)} squad has no {column}")
    squad["score_xp"] = _fill_score(squad, "score_xp")
    banked = bank_squad_gw(squad, "score_xp", use_autosubs=True)
    if chip is None:
        cap_pts = float(banked["cap_extra"])
    else:
        cap_pts, bench_pts = _chip_additions(squad, banked, chip)
        if bench_pts != 0:
            raise RuntimeError("bench points were added without a bench boost")
    hit_pts = float(HIT_COST * int(hits))
    model_points = float(banked["xi_points"]) + float(cap_pts) - hit_pts
    theirs = _their_score(week, roster, chip)
    official = float(week["points"])
    residual = float(theirs["built"] - official)
    names = _name_map(roster, gw)
    minutes = {
        str(row.player_id): float(getattr(row, "minutes") or 0) for row in squad.itertuples()
    }
    model_captain = _effective_name(
        str(banked["captain_id"]), str(banked["vice_id"]), minutes, names
    )
    their_ids = {player_key(player["id"]) for player in list(week["xi"]) + list(week["bench"])}
    split = classify_gap(
        pre_ids,
        new_state.ids(),
        their_ids,
        _points(banked["xi"]),
        theirs["points_map"],
    )
    captain_gap = float(cap_pts) - float(theirs["extra"])
    hit_gap = float(theirs["hits"]) - hit_pts
    gap = model_points - official
    pieces = (
        captain_gap
        + float(split["transfer_gap"])
        + float(split["lineup_gap"])
        + hit_gap
        + residual
    )
    if abs(pieces - gap) > 1e-6:
        raise RuntimeError(f"GW{int(gw)} pieces {pieces} do not equal the gap {gap}")
    bought = sorted(new_state.ids() - pre_ids)
    sold = sorted(pre_ids - new_state.ids())
    their_bought = sorted(their_ids - pre_ids)
    their_sold = sorted(pre_ids - their_ids)
    return {
        "gw": int(gw),
        "model_points": model_points,
        "their_points": official,
        "gap": gap,
        "captain_gap": captain_gap,
        "transfer_gap": float(split["transfer_gap"]),
        "lineup_gap": float(split["lineup_gap"]),
        "hit_gap": hit_gap,
        "residual": residual,
        "model_captain": model_captain,
        "their_captain": theirs["captain"],
        "transfer_model": split["transfer_model"],
        "transfer_their": split["transfer_their"],
        "lineup_model": split["lineup_model"],
        "lineup_their": split["lineup_their"],
        "model_in": bought,
        "model_out": sold,
        "their_in": their_bought,
        "their_out": their_sold,
        "n_transfers": int(n_tx),
        "hits": int(hits),
        "ft": int(state.ft),
        "bank": int(state.bank),
        "horizon": window,
        "chip": chip,
        "pre_ids": sorted(pre_ids),
        "model_ids": sorted(new_state.ids()),
        "their_ids": sorted(their_ids),
        "names": names,
    }


def _roster_index(roster: pd.DataFrame) -> dict[int, set[str]]:
    return {
        int(gw): set(block["player_id"].astype(str))
        for gw, block in roster.groupby("gw")
    }


def run() -> dict[str, Any]:
    """Score the five resets and write the table. Does not retune."""
    feat, roster, _info = build_frames()
    entry = load_entry(ENTRY_ID)
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = _roster_index(roster)
    rows = [
        one_week(
            feat,
            roster,
            entry,
            gw,
            clubs=clubs,
            roster_by_gw=roster_by_gw,
            horizon_scores=horizon_scores,
        )
        for gw in GWS
    ]
    assert_not_carried(rows)
    if any(row["chip"] not in (None, "triple_captain") for row in rows):
        raise RuntimeError("a chip other than Triple Captain was scored")
    if rows[0]["chip"] != "triple_captain":
        raise RuntimeError("Gameweek 1 did not mirror Triple Captain")
    gaps = [float(row["gap"]) for row in rows]
    gained = beats_five(gaps)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_csv(PROCESSED / "reset_gap_gw15.csv", rows)
    _write_report(REPORTS / "reset_gap_gw15.md", rows, gained)
    return {
        "gaps": gaps,
        "total": float(sum(gaps)),
        "non_negative": int(sum(g >= -1e-6 for g in gaps)),
        "gain": gained,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        names = row["names"]
        flat.append(
            {
                "gw": row["gw"],
                "model_points": row["model_points"],
                "their_points": row["their_points"],
                "gap": row["gap"],
                "captain_gap": row["captain_gap"],
                "transfer_gap": row["transfer_gap"],
                "lineup_gap": row["lineup_gap"],
                "hit_gap": row["hit_gap"],
                "residual": row["residual"],
                "model_captain": row["model_captain"],
                "their_captain": row["their_captain"],
                "transfer_model": _join(row["transfer_model"], names),
                "transfer_their": _join(row["transfer_their"], names),
                "lineup_model": _join(row["lineup_model"], names),
                "lineup_their": _join(row["lineup_their"], names),
                "model_in": _join(row["model_in"], names),
                "model_out": _join(row["model_out"], names),
                "their_in": _join(row["their_in"], names),
                "their_out": _join(row["their_out"], names),
                "n_transfers": row["n_transfers"],
                "hits": row["hits"],
                "ft": row["ft"],
                "bank": row["bank"],
                "horizon": ",".join(str(gw) for gw in row["horizon"]),
                "chip": row["chip"] or "",
            }
        )
    pd.DataFrame(flat).to_csv(path, index=False)


def _write_report(path: Path, rows: list[dict[str, Any]], gained: bool) -> None:
    gaps = [float(row["gap"]) for row in rows]
    total = float(sum(gaps))
    n_ok = int(sum(g >= -1e-6 for g in gaps))
    if gained:
        verdict = (
            f"The sum is {_fmt(total)} and {n_ok} of 5 weeks are non-negative. "
            "That is a gain on these five decisions."
        )
    else:
        verdict = (
            f"The sum is {_fmt(total)} and {n_ok} of 5 weeks are non-negative. "
            "The model did not beat these five decisions."
        )
    lines = [
        "# Gameweeks 1–5, reset to the squad he held",
        "",
        "Each week starts from the fifteen ojaminFC owned before that deadline, with that week's bank and free transfers. The search is the published rule: hold margin 1.25, switch penalty 1.0, a three-week opening horizon, and the early score capped at 6. The model plays no wildcard, free hit, or bench boost. Gameweek 1 triples the model's captain, because that is the chip he played. The squad is then discarded. The next week starts from the fifteen he actually fielded.",
        "",
        "The search maximises a discounted three-week value. The score is one week of realised points, so a hit taken for a later week is charged here and those later points are not in the total. Gameweeks 4 and 5 look ahead to Gameweeks 6 and 7 on those fixtures' opening prices. Gameweek 6 is not in the sum.",
        "",
        "The bar was locked before this total was read. A sum above 0 and at least 3 of 5 weeks non-negative is a gain on these five decisions. Anything else is the model not beating these five decisions.",
        "",
        verdict + " Five weeks remain too few to call the rule reliable.",
        "",
        "| GW | Model | ojaminFC | Gap | Captain | Transfers | Lineup | Hits | Residual | Horizon |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        horizon = ",".join(str(item) for item in row["horizon"])
        lines.append(
            f"| {int(row['gw'])} | {row['model_points']:.0f} | {row['their_points']:.0f} | "
            f"{_fmt(row['gap'])} | {_fmt(row['captain_gap'])} | {_fmt(row['transfer_gap'])} | "
            f"{_fmt(row['lineup_gap'])} | {_fmt(row['hit_gap'])} | {_fmt(row['residual'])} | {horizon} |"
        )
    lines.append(
        f"| Total | {sum(row['model_points'] for row in rows):.0f} | "
        f"{sum(row['their_points'] for row in rows):.0f} | {_fmt(total)} | "
        f"{_fmt(sum(row['captain_gap'] for row in rows))} | "
        f"{_fmt(sum(row['transfer_gap'] for row in rows))} | "
        f"{_fmt(sum(row['lineup_gap'] for row in rows))} | "
        f"{_fmt(sum(row['hit_gap'] for row in rows))} | "
        f"{_fmt(sum(row['residual'] for row in rows))} | |"
    )
    lines += [
        "",
        "Captain is the extra copy only. Gameweek 1 adds two extra copies, because both sides play Triple Captain. A player who finished in one scoring eleven is a transfer when he was bought or sold, and a lineup choice when both fifteens contained him. Hits are his charge minus the model's. The residual is the rebuilt week minus the official total. The five pieces sum to the gap.",
        "",
    ]
    matched = all(abs(float(row["residual"])) < 1e-6 for row in rows)
    if matched:
        lines.append("The rebuilt week matches the official total in every week.")
        lines.append("")
    for row in rows:
        lines.append(f"## Gameweek {int(row['gw'])}")
        lines.append("")
        lines.extend(phrases(row, row["names"]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
