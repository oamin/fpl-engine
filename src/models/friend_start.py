"""Gameweeks 1–5 from one other manager's starting fifteen.

The squad carries. It does not reset to his later transfers, and it does
not copy his chips. The chip rule is the priced-horizon wallet. A small
realised bench is reported and does not change the margin.

The bar was locked before his total was read. A sum of model minus official
points above 0, and at least 3 of 5 weeks non-negative, is a gain on these
five weeks from his starting fifteen. Anything else is the model not
beating these five weeks.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import GWS, build_frames, player_key
from src.live.entry import load_entry
from src.models.blank_context import apply_fixture_tags
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_chips import choose_chip, chip_phrases, price_horizon
from src.models.reset_gap import (
    PROCESSED,
    REPORTS,
    _chip_additions,
    _early,
    _effective_name,
    _fill_score,
    _fmt,
    _join,
    _name_map,
    _points,
    beats_five,
    classify_gap,
    horizon_gws,
    merged_clubs,
    pre_deadline,
)
from src.models.season_climb import bank_squad_gw
from src.models.season_climb_ft import SquadState, _gw_pool, choose_transfers, rebuild_squad
from src.models.stage_40_gw15_gap import _their_final
from src.rules.fpl_2026 import (
    FREE_TRANSFER_CHIPS,
    HIT_COST,
    ChipWallet,
    advance_ft,
    captain_extra_points,
)

ENTRY_ID = 1078627
LAST_PRICED_GW = 7


def carried_state(
    before: SquadState, after: SquadState, chip: str | None, n_tx: int
) -> SquadState:
    """The squad and free transfers at the next deadline.

    A free hit returns to the squad from before the chip. A wildcard keeps
    the rebuild. Neither chip grants another free transfer.
    """
    if chip == "free_hit":
        out = deepcopy(before)
        spent = 0
    elif chip == "wildcard":
        out = after
        spent = 0
    else:
        out = after
        spent = int(n_tx)
    out.ft = advance_ft(
        int(before.ft),
        spent,
        chip if chip in FREE_TRANSFER_CHIPS else None,
    )
    return out


def his_score(week: dict[str, Any], roster: pd.DataFrame) -> dict[str, Any]:
    """His official week, rebuilt from his picks. Bench Boost adds the bench."""
    chip = week.get("chip") or None
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
    gw = int(week["gw"])
    sheet = roster.loc[roster["gw"] == gw].drop_duplicates("player_id", keep="first")
    by_id = {str(row.player_id): row for row in sheet.itertuples(index=False)}
    all_points = 0.0
    for player in list(week["xi"]) + list(week["bench"]):
        src = by_id.get(player_key(player["id"]))
        all_points += float(getattr(src, "total_points", 0) or 0) if src is not None else 0.0
    bench = all_points - base if chip == "bench_boost" else 0.0
    hits = float(week["transfer_cost"])
    if cap_played:
        captain = str(week["captain"])
    elif vice_played:
        captain = str(week["vice"])
    else:
        captain = "nobody"
    return {
        "final": final,
        "points_map": {
            str(row.player_id): float(row.total_points) for row in final.itertuples(index=False)
        },
        "base": base,
        "extra": float(extra),
        "bench": float(bench),
        "hits": hits,
        "built": base + float(extra) + float(bench) - hits,
        "captain": captain,
        "chip": chip,
    }


def assert_carried(rows: list[dict[str, Any]]) -> None:
    """The next fifteen is the model's, unless the week was a free hit."""
    for prev, nxt in zip(rows, rows[1:], strict=False):
        expected = prev["pre_ids"] if prev["chip"] == "free_hit" else prev["model_ids"]
        if set(nxt["pre_ids"]) != set(expected):
            raise RuntimeError(
                f"GW{int(nxt['gw'])} did not continue the squad from GW{int(prev['gw'])}"
            )


def one_week(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    entry: dict[str, Any],
    gw: int,
    state: SquadState,
    wallet: ChipWallet,
    *,
    clubs: dict[int, set[str]],
    roster_by_gw: dict[int, set[str]],
    horizon_scores: Any,
) -> tuple[dict[str, Any], SquadState]:
    """One carried week. Returns the row and the state for the next deadline."""
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    week = next(row for row in entry["gameweeks"] if int(row["gw"]) == int(gw))
    pre_ids = state.ids()
    if len(pre_ids) != 15:
        raise RuntimeError(f"GW{int(gw)} does not start from 15 players")
    early = _early(feat)
    pool = _gw_pool(feat, roster, int(gw), pre_ids, early)
    pool = apply_fixture_tags(pool, int(gw), clubs)
    missing = pre_ids - set(pool["player_id"].astype(str))
    if missing:
        raise RuntimeError(f"GW{int(gw)} is missing owned players {sorted(missing)}")
    window = horizon_gws(int(gw), clubs)
    if any(int(step) > LAST_PRICED_GW for step in window):
        raise RuntimeError("the horizon goes past Gameweek 7")
    step_scores = horizon_scores(int(gw), pool, window)
    prior = deepcopy(state)
    rebuilt = rebuild_squad(state, pool, "score_xp")
    steps = price_horizon(pool, pre_ids, rebuilt.ids(), step_scores, window)
    chip, gain = choose_chip(steps, wallet.available(int(gw)))
    if chip is not None:
        wallet.play(int(gw), chip)
    ft_before = int(state.ft)
    if chip in FREE_TRANSFER_CHIPS:
        new_state, n_tx, hits = rebuilt, len(rebuilt.ids() - pre_ids), 0
    else:
        new_state, n_tx, hits = choose_transfers(
            state,
            pool,
            "score_xp",
            gw=int(gw),
            future_gws=window,
            roster_by_gw=roster_by_gw,
            score_by_gw=step_scores,
            clubs=clubs,
            bench_gw=int(gw) if chip == "bench_boost" else None,
        )
    squad = pool.loc[pool["player_id"].astype(str).isin(new_state.ids())].copy()
    squad = squad.drop_duplicates("player_id", keep="first")
    if len(squad) != 15:
        raise RuntimeError(f"GW{int(gw)} squad has {len(squad)} players")
    squad["score_xp"] = _fill_score(squad, "score_xp")
    banked = bank_squad_gw(squad, "score_xp", use_autosubs=True)
    if chip is None:
        cap_pts, bench_pts = float(banked["cap_extra"]), 0.0
    else:
        cap_pts, bench_pts = _chip_additions(squad, banked, chip)
        if chip != "bench_boost" and float(bench_pts) != 0:
            raise RuntimeError("bench points were added without a bench boost")
    hit_pts = 0.0 if chip in FREE_TRANSFER_CHIPS else float(HIT_COST * int(hits))
    model_points = float(banked["xi_points"]) + float(cap_pts) + float(bench_pts) - hit_pts
    theirs = his_score(week, roster)
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
    bench_gap = float(bench_pts) - float(theirs["bench"])
    gap = model_points - official
    pieces = (
        captain_gap
        + float(split["transfer_gap"])
        + float(split["lineup_gap"])
        + hit_gap
        + bench_gap
        + residual
    )
    if abs(pieces - gap) > 1e-6:
        raise RuntimeError(f"GW{int(gw)} pieces {pieces} do not equal the gap {gap}")
    nxt = carried_state(prior, new_state, chip, int(n_tx))
    row = {
        "gw": int(gw),
        "model_points": model_points,
        "their_points": official,
        "gap": gap,
        "captain_gap": captain_gap,
        "transfer_gap": float(split["transfer_gap"]),
        "lineup_gap": float(split["lineup_gap"]),
        "hit_gap": hit_gap,
        "bench_gap": bench_gap,
        "residual": residual,
        "model_captain": model_captain,
        "their_captain": theirs["captain"],
        "transfer_model": split["transfer_model"],
        "transfer_their": split["transfer_their"],
        "lineup_model": split["lineup_model"],
        "lineup_their": split["lineup_their"],
        "model_in": sorted(new_state.ids() - pre_ids),
        "model_out": sorted(pre_ids - new_state.ids()),
        "n_transfers": int(n_tx),
        "hits": int(hits),
        "ft": ft_before,
        "bank": int(prior.bank),
        "horizon": window,
        "chip": chip,
        "his_chip": theirs["chip"],
        "chip_gain": float(gain),
        "wc_sum": float(sum(step.rebuilt_xi - step.held_xi for step in steps)),
        "fh_margin": float(steps[0].fh_xi - steps[0].held_xi),
        "cap_xp": float(steps[0].cap_xp),
        "bench_xp": float(steps[0].bench_xp),
        "bench_scored": float(bench_pts),
        "his_bench": float(theirs["bench"]),
        "pre_ids": sorted(pre_ids),
        "model_ids": sorted(new_state.ids()),
        "names": names,
    }
    return row, nxt


def run() -> dict[str, Any]:
    """Carry his Gameweek 1 fifteen for five weeks. Does not retune."""
    feat, roster, _info = build_frames()
    entry = load_entry(ENTRY_ID)
    if int(entry["entry_id"]) != ENTRY_ID:
        raise RuntimeError("the entry file is not 1078627")
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    state = pre_deadline(entry, roster, 1)
    if int(state.ft) != 0:
        raise RuntimeError("Gameweek 1 did not start with 0 free transfers")
    wallet = ChipWallet()
    rows: list[dict[str, Any]] = []
    for gw in GWS:
        row, state = one_week(
            feat,
            roster,
            entry,
            gw,
            state,
            wallet,
            clubs=clubs,
            roster_by_gw=roster_by_gw,
            horizon_scores=horizon_scores,
        )
        rows.append(row)
    assert_carried(rows)
    played = [row["chip"] for row in rows if row["chip"]]
    if len(played) != len(set(played)):
        raise RuntimeError("a chip was played twice")
    if rows[0]["chip"] in FREE_TRANSFER_CHIPS:
        raise RuntimeError("Gameweek 1 played a wildcard or a free hit")
    gaps = [float(row["gap"]) for row in rows]
    gained = beats_five(gaps)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_csv(PROCESSED / "friend_start_gw15.csv", rows)
    _write_report(
        REPORTS / "friend_start_gw15.md",
        rows,
        gained,
        str(entry.get("team_name") or ENTRY_ID),
    )
    return {
        "team": entry.get("team_name"),
        "gaps": gaps,
        "total": float(sum(row["model_points"] for row in rows)),
        "his": float(sum(row["their_points"] for row in rows)),
        "gap": float(sum(gaps)),
        "non_negative": int(sum(g >= -1e-6 for g in gaps)),
        "gain": gained,
        "chips": [(int(row["gw"]), row["chip"], float(row["bench_xp"]), float(row["bench_scored"])) for row in rows],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        names = row["names"]
        flat.append(
            {
                "gw": row["gw"],
                "chip": row["chip"] or "",
                "his_chip": row["his_chip"] or "",
                "chip_gain": row["chip_gain"],
                "wc_sum": row["wc_sum"],
                "fh_margin": row["fh_margin"],
                "cap_xp": row["cap_xp"],
                "bench_xp": row["bench_xp"],
                "bench_scored": row["bench_scored"],
                "his_bench": row["his_bench"],
                "model_points": row["model_points"],
                "their_points": row["their_points"],
                "gap": row["gap"],
                "captain_gap": row["captain_gap"],
                "transfer_gap": row["transfer_gap"],
                "lineup_gap": row["lineup_gap"],
                "hit_gap": row["hit_gap"],
                "bench_gap": row["bench_gap"],
                "residual": row["residual"],
                "model_captain": row["model_captain"],
                "their_captain": row["their_captain"],
                "transfer_model": _join(row["transfer_model"], names),
                "transfer_their": _join(row["transfer_their"], names),
                "model_in": _join(row["model_in"], names),
                "model_out": _join(row["model_out"], names),
                "n_transfers": row["n_transfers"],
                "hits": row["hits"],
                "ft": row["ft"],
                "horizon": ",".join(str(gw) for gw in row["horizon"]),
            }
        )
    pd.DataFrame(flat).to_csv(path, index=False)


def _write_report(path: Path, rows: list[dict[str, Any]], gained: bool, team: str) -> None:
    gaps = [float(row["gap"]) for row in rows]
    total = float(sum(row["model_points"] for row in rows))
    his = float(sum(row["their_points"] for row in rows))
    gap = float(sum(gaps))
    n_ok = int(sum(g >= -1e-6 for g in gaps))
    if gained:
        verdict = (
            f"The sum is {_fmt(gap)} and {n_ok} of 5 weeks are non-negative. "
            "That is a gain on these five weeks from his starting fifteen."
        )
    else:
        verdict = (
            f"The sum is {_fmt(gap)} and {n_ok} of 5 weeks are non-negative. "
            "The model did not beat these five weeks."
        )
    lines = [
        f"# Gameweeks 1–5 from {team}'s starting fifteen",
        "",
        f"Entry 1078627, {team}. The model starts from his Gameweek 1 fifteen, with that week's bank and with no free transfer. The squad then carries. His later transfers are not copied, and his chips are not spent in advance. The wallet starts full.",
        "",
        "The chip rule is the one from the free-wallet reset. The choice uses expected points on at most three priced weeks, and no week after Gameweek 7. Triple Captain and Bench Boost have to be strictly the best week on that horizon. Free Hit needs 12. Wildcard needs 16 on those priced weeks. Gameweek 1 cannot play either. A tie plays nothing. A bench that scores fewer points than its outlook is reported. The margin is not changed.",
        "",
        "The bar was locked before this total was read. A sum above 0 and at least 3 of 5 weeks non-negative is a gain on these five weeks from his starting fifteen. Anything else is the model not beating these five weeks.",
        "",
        verdict + " One squad and five weeks remain too few to call the rule reliable.",
        "",
        "| GW | Model chip | His chip | Model | His | Gap | Captain | Transfers | Lineup | Hits | Bench | Bench outlook | Bench scored | Residual |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['gw'])} | {row['chip'] or 'none'} | {row['his_chip'] or 'none'} | "
            f"{row['model_points']:.0f} | {row['their_points']:.0f} | {_fmt(row['gap'])} | "
            f"{_fmt(row['captain_gap'])} | {_fmt(row['transfer_gap'])} | {_fmt(row['lineup_gap'])} | "
            f"{_fmt(row['hit_gap'])} | {_fmt(row['bench_gap'])} | {row['bench_xp']:.2f} | "
            f"{row['bench_scored']:.0f} | {_fmt(row['residual'])} |"
        )
    lines.append(
        f"| Total |  |  | {total:.0f} | {his:.0f} | {_fmt(gap)} | "
        f"{_fmt(sum(row['captain_gap'] for row in rows))} | "
        f"{_fmt(sum(row['transfer_gap'] for row in rows))} | "
        f"{_fmt(sum(row['lineup_gap'] for row in rows))} | "
        f"{_fmt(sum(row['hit_gap'] for row in rows))} | "
        f"{_fmt(sum(row['bench_gap'] for row in rows))} |  |  | "
        f"{_fmt(sum(row['residual'] for row in rows))} |"
    )
    lines += [
        "",
        "Bench outlook is the expected bench on the squad before the transfers. Bench scored is the realised Bench Boost award, and it is 0 when that chip is not played. The bench column in the gap is his award subtracted from the model's. The six pieces sum to the gap.",
        "",
    ]
    if all(abs(float(row["residual"])) < 1e-6 for row in rows):
        lines.append("The rebuilt week matches his official total in every week.")
        lines.append("")
    for row in rows:
        his_chip = row["his_chip"] or "no chip"
        lines.append(
            f"## Gameweek {int(row['gw'])}, model {row['chip'] or 'no chip'}, he played {his_chip}"
        )
        lines.append("")
        lines.extend(chip_phrases(row, row["names"]))
        if row["chip"] == "bench_boost":
            lines.append(
                f"The bench outlook was {float(row['bench_xp']):.2f} and the bench scored {float(row['bench_scored']):.0f}."
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
