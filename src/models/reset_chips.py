"""Gameweeks 1–5 with the chip wallet free to choose.

The squad reset matches ``reset_gap``. The wallet does not copy his chips.
A chip is chosen from the priced opening horizon only. Realised points are
the score, not the choice. The wildcard sum does not copy a later week.

The bar was locked before the total was read. A sum of model minus official
points above 0, and at least 3 of 5 weeks non-negative, is a gain on these
five decisions with a free chip wallet. Anything else is the wallet not
beating these five decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import GWS, build_frames, player_key
from src.live.entry import load_entry
from src.live.policy import FH_MARGIN, WC_MARGIN
from src.models.blank_context import apply_fixture_tags
from src.models.half_plan_scores import squad_outlook
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import (
    ENTRY_ID,
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
    _their_score,
    assert_not_carried,
    beats_five,
    classify_gap,
    horizon_gws,
    merged_clubs,
    phrases,
    pre_deadline,
)
from src.models.season_climb import bank_squad_gw
from src.models.season_climb_ft import (
    _gw_pool,
    choose_transfers,
    rebuild_squad,
)
from src.rules.fpl_2026 import FREE_TRANSFER_CHIPS, HIT_COST, ChipWallet

# A week after this would be outside the two priced steps past Gameweek 5.
LAST_PRICED_GW = 7
BASELINE_CSV = PROCESSED / "reset_gap_gw15.csv"
_TIE = 1e-6


@dataclass(frozen=True)
class StepOutlook:
    """Expected points for one priced week. ``held_xi`` includes the captain double."""

    gw: int
    held_xi: float
    bench_xp: float
    cap_xp: float
    rebuilt_xi: float
    fh_xi: float


def choose_chip(
    steps: list[StepOutlook], available: tuple[str, ...]
) -> tuple[str | None, float]:
    """The unique legal chip that clears its hurdle, or nothing on a tie.

    Triple Captain and Bench Boost must be strictly the best week on this
    horizon. Free Hit and Wildcard use the locked margins and are refused
    in Gameweek 1 even if a caller lists them.
    """
    if not steps:
        raise RuntimeError("the chip choice has no priced week")
    now = steps[0]
    legal = set(available)
    if int(now.gw) == 1:
        legal -= set(FREE_TRANSFER_CHIPS)
    later_cap = max((step.cap_xp for step in steps[1:]), default=-1.0)
    later_bench = max((step.bench_xp for step in steps[1:]), default=-1.0)
    gains: dict[str, float] = {}
    if "triple_captain" in legal and now.cap_xp > 0 and now.cap_xp > later_cap:
        gains["triple_captain"] = float(now.cap_xp)
    if "bench_boost" in legal and now.bench_xp > 0 and now.bench_xp > later_bench:
        gains["bench_boost"] = float(now.bench_xp)
    if "free_hit" in legal:
        margin = float(now.fh_xi) - float(now.held_xi)
        if margin >= FH_MARGIN:
            gains["free_hit"] = margin
    if "wildcard" in legal:
        total = sum(step.rebuilt_xi - step.held_xi for step in steps)
        if total >= WC_MARGIN:
            gains["wildcard"] = float(total)
    if not gains:
        return None, 0.0
    ranked = sorted(gains.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) <= _TIE:
        return None, 0.0
    return ranked[0][0], float(ranked[0][1])


def price_horizon(
    pool: pd.DataFrame,
    held_ids: set[str],
    rebuilt_ids: set[str],
    step_scores: dict[int, dict[str, float]],
    window: list[int],
) -> list[StepOutlook]:
    """Outlook for the priced weeks. The decision-week free hit is the rebuild."""
    if len(window) > 3:
        raise RuntimeError("the horizon is longer than three weeks")
    if any(int(gw) > LAST_PRICED_GW for gw in window):
        raise RuntimeError("the wildcard sum includes a week after Gameweek 7")
    steps: list[StepOutlook] = []
    for gw in window:
        if int(gw) not in step_scores:
            raise RuntimeError(f"GW{int(gw)} has no outlook scores")
        scores = {str(pid): float(value) for pid, value in step_scores[int(gw)].items()}
        held = squad_outlook(pool, held_ids, scores)
        rebuilt = squad_outlook(pool, rebuilt_ids, scores)
        steps.append(
            StepOutlook(
                gw=int(gw),
                held_xi=float(held.xi_xp),
                bench_xp=float(held.bench_xp),
                cap_xp=float(held.cap_xp),
                rebuilt_xi=float(rebuilt.xi_xp),
                fh_xi=float(rebuilt.xi_xp),
            )
        )
    return steps


def chip_phrases(row: dict[str, Any], names: dict[str, str] | None = None) -> list[str]:
    """The reset sentences, plus the bench piece, plus a chip label on the captain."""
    lines = phrases(row, names)
    cap = float(row["captain_gap"])
    word = "added" if cap >= 0 else "cost"
    model_chip = row.get("chip") or None
    his_chip = row.get("his_chip") or None
    if model_chip == "triple_captain" and his_chip != "triple_captain":
        lines[0] = (
            f"Triple Captain on {row['model_captain']} versus {row['their_captain']} {word} {_fmt(cap)}."
        )
    elif his_chip == "triple_captain" and model_chip != "triple_captain":
        lines[0] = (
            f"Captaincy on {row['model_captain']} versus Triple Captain on {row['their_captain']} {word} {_fmt(cap)}."
        )
    bench = float(row["bench_gap"])
    bench_word = "added" if bench >= 0 else "cost"
    bench_line = f"Bench Boost {bench_word} {_fmt(bench)}."
    if lines[-1].startswith("The rebuilt"):
        lines.insert(-1, bench_line)
    else:
        lines.append(bench_line)
    return lines


def _reviewed_total() -> float:
    frame = pd.read_csv(BASELINE_CSV)
    return float(frame["model_points"].sum())


def _roster_index(roster: pd.DataFrame) -> dict[int, set[str]]:
    return {
        int(gw): set(block["player_id"].astype(str))
        for gw, block in roster.groupby("gw")
    }


def one_week(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    entry: dict[str, Any],
    gw: int,
    wallet: ChipWallet,
    *,
    clubs: dict[int, set[str]],
    roster_by_gw: dict[int, set[str]],
    horizon_scores: Any,
) -> dict[str, Any]:
    """One deadline. The chip is chosen before the realised score."""
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    if not clubs.get(int(gw)):
        raise RuntimeError(f"GW{int(gw)} has no clubs")
    state = pre_deadline(entry, roster, gw)
    pre_ids = state.ids()
    week = next(row for row in entry["gameweeks"] if int(row["gw"]) == int(gw))
    his_chip = week.get("chip") or None
    if his_chip not in (None, "triple_captain"):
        raise RuntimeError(f"GW{int(gw)} official chip {his_chip} is outside this comparison")
    early = _early(feat)
    pool = _gw_pool(feat, roster, int(gw), pre_ids, early)
    pool = apply_fixture_tags(pool, int(gw), clubs)
    missing = pre_ids - set(pool["player_id"].astype(str))
    if missing:
        raise RuntimeError(f"GW{int(gw)} is missing owned players {sorted(missing)}")
    window = horizon_gws(int(gw), clubs)
    step_scores = horizon_scores(int(gw), pool, window)
    if any(int(step) > LAST_PRICED_GW for step in step_scores):
        raise RuntimeError("an outlook score is past Gameweek 7")
    rebuilt = rebuild_squad(state, pool, "score_xp")
    steps = price_horizon(pool, pre_ids, rebuilt.ids(), step_scores, window)
    chip, gain = choose_chip(steps, wallet.available(int(gw)))
    if chip is not None:
        wallet.play(int(gw), chip)
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
        if chip != "bench_boost" and bench_pts != 0:
            raise RuntimeError("bench points were added without a bench boost")
    hit_pts = 0.0 if chip in FREE_TRANSFER_CHIPS else float(HIT_COST * int(hits))
    model_points = float(banked["xi_points"]) + float(cap_pts) + float(bench_pts) - hit_pts
    theirs = _their_score(week, roster, his_chip)
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
    bench_gap = float(bench_pts)
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
    return {
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
        "their_in": sorted(their_ids - pre_ids),
        "their_out": sorted(pre_ids - their_ids),
        "n_transfers": int(n_tx),
        "hits": int(hits),
        "ft": int(state.ft),
        "bank": int(state.bank),
        "horizon": window,
        "chip": chip,
        "his_chip": his_chip,
        "chip_gain": float(gain),
        "wc_sum": float(sum(step.rebuilt_xi - step.held_xi for step in steps)),
        "fh_margin": float(steps[0].fh_xi - steps[0].held_xi),
        "cap_xp": float(steps[0].cap_xp),
        "bench_xp": float(steps[0].bench_xp),
        "pre_ids": sorted(pre_ids),
        "model_ids": sorted(new_state.ids()),
        "their_ids": sorted(their_ids),
        "names": names,
    }


def run() -> dict[str, Any]:
    """Score the five resets with a free wallet. Does not retune."""
    feat, roster, _info = build_frames()
    entry = load_entry(ENTRY_ID)
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = _roster_index(roster)
    wallet = ChipWallet()
    rows = []
    for gw in GWS:
        rows.append(
            one_week(
                feat,
                roster,
                entry,
                gw,
                wallet,
                clubs=clubs,
                roster_by_gw=roster_by_gw,
                horizon_scores=horizon_scores,
            )
        )
    assert_not_carried(rows)
    played = [row["chip"] for row in rows if row["chip"]]
    if len(played) != len(set(played)):
        raise RuntimeError("a chip was played twice")
    if rows[0]["chip"] in FREE_TRANSFER_CHIPS:
        raise RuntimeError("Gameweek 1 played a wildcard or a free hit")
    gaps = [float(row["gap"]) for row in rows]
    gained = beats_five(gaps)
    baseline = _reviewed_total()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_csv(PROCESSED / "reset_chips_gw15.csv", rows)
    _write_report(REPORTS / "reset_chips_gw15.md", rows, gained, baseline)
    return {
        "gaps": gaps,
        "total": float(sum(row["model_points"] for row in rows)),
        "his": float(sum(row["their_points"] for row in rows)),
        "gap": float(sum(gaps)),
        "non_negative": int(sum(g >= -1e-6 for g in gaps)),
        "gain": gained,
        "versus_reset": float(sum(row["model_points"] for row in rows) - baseline),
        "chips": [(int(row["gw"]), row["chip"]) for row in rows],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat = []
    for row in rows:
        names = row["names"]
        flat.append(
            {
                "gw": row["gw"],
                "chip": row["chip"] or "",
                "chip_gain": row["chip_gain"],
                "wc_sum": row["wc_sum"],
                "fh_margin": row["fh_margin"],
                "cap_xp": row["cap_xp"],
                "bench_xp": row["bench_xp"],
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
                "lineup_model": _join(row["lineup_model"], names),
                "lineup_their": _join(row["lineup_their"], names),
                "model_in": _join(row["model_in"], names),
                "model_out": _join(row["model_out"], names),
                "n_transfers": row["n_transfers"],
                "hits": row["hits"],
                "horizon": ",".join(str(gw) for gw in row["horizon"]),
                "his_chip": row["his_chip"] or "",
            }
        )
    pd.DataFrame(flat).to_csv(path, index=False)


def _write_report(
    path: Path, rows: list[dict[str, Any]], gained: bool, baseline: float
) -> None:
    gaps = [float(row["gap"]) for row in rows]
    total = float(sum(row["model_points"] for row in rows))
    his = float(sum(row["their_points"] for row in rows))
    gap = float(sum(gaps))
    n_ok = int(sum(g >= -1e-6 for g in gaps))
    if gained:
        verdict = (
            f"The sum is {_fmt(gap)} and {n_ok} of 5 weeks are non-negative. "
            "That is a gain on these five decisions with a free chip wallet."
        )
    else:
        verdict = (
            f"The sum is {_fmt(gap)} and {n_ok} of 5 weeks are non-negative. "
            "The wallet did not beat these five decisions."
        )
    lines = [
        "# Gameweeks 1–5, free chip wallet",
        "",
        "Each week starts from the fifteen ojaminFC owned before that deadline, with that week's bank and free transfers. The search is the published rule. The wallet starts full: wildcard, free hit, bench boost, and triple captain. His Gameweek 1 Triple Captain is not copied and is not spent in advance. Gameweek 1 cannot play a wildcard or a free hit. One chip a week, one of each, and no free hit in consecutive weeks.",
        "",
        "The chip is chosen from expected points on the priced opening horizon, at most three weeks. A blank is skipped. No week after Gameweek 7 is priced, and no later week is copied. Triple Captain needs this week's extra captain copy to be strictly the best on that horizon. Bench Boost needs the same for the bench. Free Hit needs a lead of 12 on this week's rebuild. Wildcard needs the rebuilt eleven to lead by 16 across those priced weeks. A tie plays nothing. Realised points are applied after the choice.",
        "",
        "A wildcard or a free hit is scored for this week and the squad is then discarded. Bench Boost and Triple Captain only affect this week, so their points are complete. The transfer search still values three weeks, and a hit taken for a later week is charged here. Gameweek 6 is not in the sum.",
        "",
        "The bar was locked before this total was read. A sum above 0 and at least 3 of 5 weeks non-negative is a gain on these five decisions with a free chip wallet. Anything else is the wallet not beating these five decisions.",
        "",
        verdict + " Five weeks remain too few to call the rule reliable.",
        "",
        f"The reviewed reset, with his Triple Captain mirrored and no other chip, scored {baseline:.0f}. This wallet scores {total:.0f}, a difference of {_fmt(total - baseline)}.",
        "",
        "| GW | Chip | Model | ojaminFC | Gap | Captain | Transfers | Lineup | Hits | Bench | Residual | Horizon |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        horizon = ",".join(str(item) for item in row["horizon"])
        lines.append(
            f"| {int(row['gw'])} | {row['chip'] or 'none'} | {row['model_points']:.0f} | "
            f"{row['their_points']:.0f} | {_fmt(row['gap'])} | {_fmt(row['captain_gap'])} | "
            f"{_fmt(row['transfer_gap'])} | {_fmt(row['lineup_gap'])} | {_fmt(row['hit_gap'])} | "
            f"{_fmt(row['bench_gap'])} | {_fmt(row['residual'])} | {horizon} |"
        )
    lines.append(
        f"| Total |  | {total:.0f} | {his:.0f} | {_fmt(gap)} | "
        f"{_fmt(sum(row['captain_gap'] for row in rows))} | "
        f"{_fmt(sum(row['transfer_gap'] for row in rows))} | "
        f"{_fmt(sum(row['lineup_gap'] for row in rows))} | "
        f"{_fmt(sum(row['hit_gap'] for row in rows))} | "
        f"{_fmt(sum(row['bench_gap'] for row in rows))} | "
        f"{_fmt(sum(row['residual'] for row in rows))} | |"
    )
    lines += [
        "",
        "Captain is the extra copy, or two extra copies when that side plays Triple Captain. Bench is the Bench Boost award. The other pieces match the reset. The six pieces sum to the gap. The wildcard figure in the file is the priced-horizon outlook, not this realised gap.",
        "",
    ]
    if all(abs(float(row["residual"])) < 1e-6 for row in rows):
        lines.append("The rebuilt week matches the official total in every week.")
        lines.append("")
    for row in rows:
        label = row["chip"] or "no chip"
        lines.append(f"## Gameweek {int(row['gw'])}, {label}")
        lines.append("")
        lines.extend(chip_phrases(row, row["names"]))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
