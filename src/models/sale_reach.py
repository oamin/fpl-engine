"""Whether a dearer same-position player was inside the one-swap list.

The price tag on a ranked-lower pair is one sale: his price against the
sell price of the player he was paired with, plus the bank. The search
sells any owned player at that position. This measurement asks which of
those other sales were legal, and whether the best of them was in the
list of 35. It does not change the hold, the buy gate, or the score.

The 17 pairs are the cohort price tag from the wildcard-lead carry. The
reference manager is printed and left out of the reading. A pair that
names a player already owned makes the batch unverified.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.reset_gap import PROCESSED, REPORTS, _fmt
from src.models.season_climb_ft import (
    HOLD_EPS,
    SquadState,
    _as_int_value,
    _fill_score,
    _gw_pool,
    _one_swap_candidates,
    _squad_legal,
)
from src.rules.fpl_2026 import MAX_PER_CLUB, sell_price

PRICE_TARGET = -39.0
PRICE_TOLERANCE = 1.0
SHARE_CUT = 0.5
BEAM = 35
CLASSES = ("gate_block", "no_fund", "delta_le_125", "truncated", "seen")


def _within_club(clubs: list[str]) -> bool:
    return all(count <= MAX_PER_CLUB for count in Counter(clubs).values())


def _price(row: Any) -> int:
    """The same missing-price fill the one-swap list uses."""
    return _as_int_value(getattr(row, "value", None))


def _index(pool: pd.DataFrame) -> dict[str, Any]:
    frame = pool.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame = frame.drop_duplicates("player_id", keep="first")
    return {str(row.player_id): row for row in frame.itertuples()}


def _scores(pool: pd.DataFrame) -> dict[str, float]:
    frame = pool.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame = frame.drop_duplicates("player_id", keep="first")
    filled = _fill_score(frame, "score_xp")
    return {str(pid): float(score) for pid, score in zip(frame["player_id"], filled, strict=True)}


def _state(week: dict[str, Any]) -> SquadState:
    """The pre-deadline squad. Purchase prices stay the ones he paid."""
    return SquadState(
        purchase={str(pid): int(price) for pid, price in week["purchase"].items()},
        bank=int(week["bank"]),
        ft=int(week["ft"]),
    )


def availability(human_id: str, pool_by: dict[str, Any]) -> str:
    """Absent, ineligible, or eligible. Eligible is the only buy."""
    row = pool_by.get(str(human_id))
    if row is None:
        return "absent"
    if bool(getattr(row, "eligible", False)):
        return "eligible"
    return "ineligible"


def position_rank(
    human_id: str,
    pool_by: dict[str, Any],
    scores: dict[str, float],
    owned: set[str],
) -> int | None:
    """1 is the highest same-position score outside the squad. Ties break on id."""
    row = pool_by.get(str(human_id))
    if row is None or not bool(getattr(row, "eligible", False)):
        return None
    position = str(row.position)
    buys = [
        pid
        for pid, other in pool_by.items()
        if pid not in owned
        and bool(getattr(other, "eligible", False))
        and str(other.position) == position
    ]
    buys.sort(key=lambda pid: (-float(scores.get(pid, 0.0)), pid))
    try:
        return buys.index(str(human_id)) + 1
    except ValueError:
        return None


def funded_sales(
    state: SquadState,
    human_id: str,
    pool_by: dict[str, Any],
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Owned same-position sales that pay for him and keep the squad legal."""
    human = pool_by.get(str(human_id))
    if human is None or not bool(getattr(human, "eligible", False)):
        return []
    if str(human_id) in state.ids():
        return []
    position = str(human.position)
    price = _price(human)
    human_score = float(scores.get(str(human_id), 0.0))
    found: list[tuple[str, float]] = []
    owned = state.ids()
    for sale in owned:
        other = pool_by.get(sale)
        if other is None or str(other.position) != position:
            continue
        if sale not in state.purchase:
            continue
        proceeds = sell_price(int(state.purchase[sale]), _price(other))
        if proceeds + int(state.bank) < price:
            continue
        new_ids = (owned - {sale}) | {str(human_id)}
        clubs = [str(pool_by[pid].team_norm) for pid in new_ids]
        if not _within_club(clubs):
            continue
        if not _squad_legal(
            [str(pool_by[pid].position) for pid in new_ids],
            clubs,
        ):
            continue
        delta = human_score - float(scores.get(sale, 0.0))
        found.append((sale, delta))
    found.sort(key=lambda item: (-item[1], item[0]))
    return found


def in_beam(state: SquadState, pool: pd.DataFrame, sale: str, human_id: str) -> bool:
    """True when that exact swap is one of the 35 the search evaluates."""
    listed = _one_swap_candidates(state, pool, "score_xp", top_n=BEAM)
    return any(sold == sale and bought == str(human_id) for sold, bought, _squad, _delta in listed)


def _enablers(pool_by: dict[str, Any], owned: set[str], human_id: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for pid, row in pool_by.items():
        if pid in owned or pid == str(human_id):
            continue
        if not bool(getattr(row, "eligible", False)):
            continue
        grouped.setdefault(str(row.position), []).append(pid)
    for ids in grouped.values():
        ids.sort(key=lambda pid: (_price(pool_by[pid]), pid))
    return grouped


def two_transfer_funds(state: SquadState, human_id: str, pool: pd.DataFrame) -> bool:
    """A second sale plus the cheapest legal player at that position.

    Descriptive. A true result does not open a two-transfer replay.
    """
    pool_by = _index(pool)
    human = pool_by.get(str(human_id))
    if human is None or not bool(getattr(human, "eligible", False)):
        return False
    if str(human_id) in state.ids():
        return False
    position = str(human.position)
    human_price = _price(human)
    owned = state.ids()
    same = [
        pid
        for pid in owned
        if pid in pool_by and str(pool_by[pid].position) == position
    ]
    buys = _enablers(pool_by, owned, str(human_id))
    for sale in same:
        if sale not in pool_by:
            continue
        for other in owned:
            if other == sale or other not in pool_by:
                continue
            other_position = str(pool_by[other].position)
            sale_cash = sell_price(int(state.purchase[sale]), _price(pool_by[sale]))
            other_cash = sell_price(int(state.purchase[other]), _price(pool_by[other]))
            for enabler in buys.get(other_position, []):
                spend = human_price + _price(pool_by[enabler])
                if int(state.bank) + sale_cash + other_cash < spend:
                    break
                new_ids = (owned - {sale, other}) | {str(human_id), enabler}
                clubs = [str(pool_by[pid].team_norm) for pid in new_ids]
                if not _within_club(clubs):
                    continue
                if _squad_legal(
                    [str(pool_by[pid].position) for pid in new_ids],
                    clubs,
                ):
                    return True
    return False


def classify_pair(
    human_id: str,
    model_id: str,
    points: float,
    state: SquadState,
    pool: pd.DataFrame,
) -> dict[str, Any]:
    """One price pair. The class is the first rule that matches."""
    pool_by = _index(pool)
    scores = _scores(pool)
    status = availability(human_id, pool_by)
    defect = str(human_id) in state.ids()
    rank = position_rank(human_id, pool_by, scores, state.ids())
    sales = [] if defect else funded_sales(state, human_id, pool_by, scores)
    sale = ""
    delta = None
    if defect:
        klass = "defect"
    elif status != "eligible":
        klass = "gate_block"
    elif not sales:
        klass = "no_fund"
    else:
        sale, delta = sales[0]
        if float(delta) <= HOLD_EPS:
            klass = "delta_le_125"
        elif in_beam(state, pool, sale, human_id):
            klass = "seen"
        else:
            klass = "truncated"
    return {
        "human_id": str(human_id),
        "model_id": str(model_id),
        "points": float(points),
        "status": status,
        "rank": rank,
        "sale_id": sale,
        "delta": delta,
        "klass": klass,
        "defect": defect,
        "two_transfer": klass == "no_fund" and two_transfer_funds(state, human_id, pool),
    }


def decide(rows: list[dict[str, Any]]) -> str:
    """The locked reading. First share at or above a half wins."""
    if any(bool(row.get("defect")) or row.get("klass") == "defect" for row in rows):
        return "unverified"
    total = sum(float(row["points"]) for row in rows)
    if abs(total - PRICE_TARGET) > PRICE_TOLERANCE:
        return "inconclusive"
    if total == 0:
        return "inconclusive"
    grouped = {name: 0.0 for name in CLASSES}
    for row in rows:
        grouped[str(row["klass"])] += float(row["points"])
    gate = grouped["gate_block"] / total
    budget = (grouped["no_fund"] + grouped["delta_le_125"]) / total
    truncated = grouped["truncated"] / total
    if gate >= SHARE_CUT:
        return "gate"
    if budget >= SHARE_CUT:
        return "budget"
    if truncated >= SHARE_CUT:
        return "truncation"
    return "seen"


def _class_points(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = {name: 0.0 for name in CLASSES}
    for row in rows:
        name = str(row["klass"])
        if name in totals:
            totals[name] += float(row["points"])
    return totals


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASSES}
    for row in rows:
        name = str(row["klass"])
        if name in counts:
            counts[name] += 1
    return counts


def reading_sentence(call: str, rows: list[dict[str, Any]]) -> str:
    """One sentence for the report. The call picks it."""
    totals = _class_points(rows)
    counts = _counts(rows)
    two = sum(1 for row in rows if row.get("two_transfer"))
    if call == "unverified":
        return "A price pair named a player the squad already owned. The batch is unverified."
    if call == "inconclusive":
        total = sum(float(row["points"]) for row in rows)
        return (
            f"The price pairs sum to {_fmt(total)}, not -39. The reading is inconclusive."
        )
    detail = (
        f"Gate {_fmt(totals['gate_block'])} from {counts['gate_block']}, "
        f"no sale {_fmt(totals['no_fund'])} from {counts['no_fund']}, "
        f"within 1.25 {_fmt(totals['delta_le_125'])} from {counts['delta_le_125']}, "
        f"truncated {_fmt(totals['truncated'])} from {counts['truncated']}, "
        f"seen {_fmt(totals['seen'])} from {counts['seen']}. "
        f"A second sale funds {two} of the unsold pairs."
    )
    if call == "gate":
        head = "The higher-scored players were outside the buy pool. Ordinary transfers stay on three appearances."
    elif call == "budget":
        head = "The fifteen could not pay for them, or the best funded sale sat within 1.25. The hold stays 1.25."
    elif call == "truncation":
        head = (
            "A funded sale cleared 1.25 and the list of 35 did not include it. "
            "A wider list would be a later historical screen. This batch stops."
        )
    else:
        head = (
            "The list of 35 included a funded sale above 1.25. "
            "The horizon value is what declined it. This batch stops."
        )
    return f"{head} {detail}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    """The measurement, then the reading. Gemini's line is added after review."""
    cohort = result["cohort"]
    lines = [
        "# Could another sale have paid for the dearer player",
        "",
        "The price tag compared his cost with one paired sale. This asks about every owned player at that position. A funded sale keeps the club cap and the squad shape. The best of those sales is the largest gap on this week's score. Seen means that swap is in the 35 the search evaluates. The hold and the score stay as they are.",
        "",
        reading_sentence(str(result["call"]), cohort),
        "",
        f"Price pairs {len(cohort)}. Points {_fmt(sum(float(row['points']) for row in cohort))}.",
        "",
    ]
    reference = result["reference"]
    if reference:
        ref_points = sum(float(row["points"]) for row in reference)
        lines.append(
            f"ojaminFC, left out of the reading: {len(reference)} price pairs, {_fmt(ref_points)}."
        )
        lines.append("")
    lines.append(
        "A second sale is counted and does not change the reading. "
        f"It funds {sum(1 for row in cohort if row.get('two_transfer'))} pairs that one sale does not."
    )
    lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    columns = [
        "entry_id",
        "label",
        "group",
        "gw",
        "human_id",
        "model_id",
        "points",
        "status",
        "rank",
        "sale_id",
        "delta",
        "klass",
        "two_transfer",
        "defect",
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    frame[columns].to_csv(path, index=False)


def run() -> dict[str, Any]:
    """Read the price pairs on the stored carry. Does not retune."""
    from src.live.benchmark import build_frames
    from src.models.blank_context import apply_fixture_tags
    from src.models.chip_lead import explain_week
    from src.models.cohort_carry import carry_entry, cohort_specs
    from src.models.open_horizon import attach_opening_horizon
    from src.models.reset_gap import _early, merged_clubs

    specs = cohort_specs()
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    early = _early(feat)
    classified: list[dict[str, Any]] = []
    for spec in specs:
        print(f"sale {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        if not result["finished"]:
            raise RuntimeError(f"{spec['label']} did not finish: {result['error']}")
        for week in result["weeks"]:
            pool = apply_fixture_tags(
                _gw_pool(feat, roster, int(week["gw"]), set(week["pre_ids"]), early),
                int(week["gw"]),
                clubs,
            )
            state = _state(week)
            if state.ids() != set(week["pre_ids"]):
                raise RuntimeError(f"{spec['label']} GW{int(week['gw'])} purchase does not match the squad")
            for row in explain_week(week, pool):
                if row["kind"] != "ranked_lower" or row["tag"] != "price":
                    continue
                labelled = classify_pair(
                    str(row["human_id"]),
                    str(row["model_id"]),
                    float(row["points"]),
                    state,
                    pool,
                )
                classified.append(
                    {
                        **labelled,
                        "entry_id": result["entry_id"],
                        "label": result["label"],
                        "group": result["group"],
                        "gw": int(week["gw"]),
                    }
                )
    cohort = [row for row in classified if row["group"] != "reference"]
    reference = [row for row in classified if row["group"] == "reference"]
    call = decide(cohort)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_csv(PROCESSED / "sale_reach_gw15.csv", classified)
    outcome = {
        "call": call,
        "cohort": cohort,
        "reference": reference,
        "two_transfer": int(sum(1 for row in cohort if row.get("two_transfer"))),
        "points": float(sum(float(row["points"]) for row in cohort)),
    }
    write_report(REPORTS / "sale_reach_gw15.md", outcome)
    summary = {
        "call": call,
        "points": outcome["points"],
        "two_transfer": outcome["two_transfer"],
        "n": len(cohort),
    }
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    run()
