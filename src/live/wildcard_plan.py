"""Logged wildcard judgement for one live deadline.

The solver maximises the discounted sum of the fifteen. The decision number
is the discounted XI plus captain on those same scores. ``score_xp`` is not
read. A missing odds week stays at zero and its weight is not moved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.live.deadline import (
    BOOTSTRAP_PATH,
    ENTRY_PATH,
    FIXTURES_PATH,
    LOG_PATH,
    ODDS_PATH,
    current_costs,
    final_players,
    fixture_calendar,
    gameweek_values,
    load_odds_frame,
    player_key,
    reconstruct_purchases,
    team_names,
)
from src.live.fpl_snapshot import ELEMENT
from src.live.scorer import ScorerError, load_ep_next
from src.models.forecast_xp import opening_pots_by_team_gw
from src.models.season_climb import pick_xi
from src.models.season_climb_ft import GAMMA
from src.rules.fpl_2026 import (
    CHIPS,
    FIRST_HALF_END_GW,
    MAX_PER_CLUB,
    N_GAMEWEEKS,
    SQUAD_QUOTA,
    ChipWallet,
    sell_price,
    squad_legal,
)
from src.rules.fpl_2026 import OFFICIAL_FORMATIONS
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
LOG_DEST = ROOT / "data" / "live" / "chip_decisions.jsonl"
HORIZON_STEPS = 4
# Lower element id wins a tie. The shift is far below a hundredth of a point.
TIE_BREAK = 1e-9


def horizon_weights(steps: int = HORIZON_STEPS) -> list[float]:
    """``GAMMA**h``. A missing week does not change these weights."""
    return [float(GAMMA) ** h for h in range(int(steps))]


def minutes_ok(started_weeks: int, status: str, chance: object) -> bool:
    """Two starts in the last three, available, and no doubt chance."""
    if int(started_weeks) < 2 or str(status) != "a":
        return False
    if chance is None:
        return True
    if isinstance(chance, float) and math.isnan(chance):
        return True
    try:
        number = float(chance)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number == 100.0


def later_week_score(ep: float, lam_decision: float | None, lam_week: float | None) -> float:
    """``ep_next`` times the fixture ratio. A missing or non-positive pot is zero."""
    if lam_decision is None or lam_week is None:
        return 0.0
    if lam_decision <= 0.0 or lam_week <= 0.0:
        return 0.0
    return float(ep) * float(lam_week) / float(lam_decision)


def _first_lam(pots: Mapping[tuple[int, str], Sequence[Mapping[str, float]]], gw: int, club: str) -> float | None:
    rows = pots.get((int(gw), club)) or []
    if not rows:
        return None
    try:
        lam = float(rows[0]["lam_scored"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(lam) or lam <= 0.0:
        return None
    return lam


def _outlook(chosen: Sequence[Mapping[str, Any]], week: int) -> dict[str, Any]:
    frame = pd.DataFrame(
        [
            {
                "player_id": str(row["pid"]),
                "position": row["position"],
                "score": float(row["weeks"][week]),
                "name": row["name"],
            }
            for row in chosen
        ]
    )
    xi, form = pick_xi(frame, "score", formations=list(OFFICIAL_FORMATIONS))
    values = pd.to_numeric(xi["score"], errors="coerce").fillna(0.0)
    order = values.sort_values(ascending=False, kind="mergesort")
    captain = str(xi.loc[order.index[0], "name"])
    return {
        "formation": f"1-{form[0]}-{form[1]}-{form[2]}",
        "xi": [str(name) for name in xi["name"].tolist()],
        "captain": captain,
        "xi_captain": float(values.sum() + values.max()),
    }


def _decision_number(chosen: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> float:
    total = 0.0
    for week, weight in enumerate(weights):
        total += float(weight) * float(_outlook(chosen, week)["xi_captain"])
    return total


def _squad_sum(chosen: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> float:
    total = 0.0
    for row in chosen:
        total += sum(float(weight) * float(score) for weight, score in zip(weights, row["weeks"], strict=True))
    return total


def _sheet(chosen: Sequence[Mapping[str, Any]], weights: Sequence[float], *, move: str) -> dict[str, Any]:
    week0 = _outlook(chosen, 0)
    return {
        "move": move,
        "players": [str(row["name"]) for row in chosen],
        "squad_sum": _squad_sum(chosen, weights),
        "decision_number": _decision_number(chosen, weights),
        "week0_xi_captain": week0["xi_captain"],
        "formation": week0["formation"],
        "xi": week0["xi"],
        "captain": week0["captain"],
        "feasible": True,
    }


def best_free_transfer(
    players: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    bank: int,
) -> dict[str, Any]:
    """One legal swap, or the hold when nothing improves the decision number."""
    held = [row for row in players if row["owned"]]
    base = _sheet(held, weights, move="hold")
    best: dict[str, Any] | None = None
    best_key: tuple[float, int, int] | None = None
    for outgoing in held:
        for incoming in players:
            if incoming["owned"] or not incoming["buyable"]:
                continue
            if int(bank) + int(outgoing["sell"]) < int(incoming["now_cost"]):
                continue
            squad = [row for row in held if row["pid"] != outgoing["pid"]] + [incoming]
            positions = [str(row["position"]) for row in squad]
            clubs = [str(row["club"]) for row in squad]
            if not squad_legal(positions, clubs):
                continue
            number = _decision_number(squad, weights)
            gain = number - float(base["decision_number"])
            if gain <= 0.0:
                continue
            key = (gain, -int(incoming["pid"]), int(outgoing["pid"]))
            if best_key is None or key > best_key:
                best_key = key
                best = _sheet(squad, weights, move=f"sell {outgoing['name']} buy {incoming['name']}")
                best["in"] = str(incoming["name"])
                best["out"] = str(outgoing["name"])
                best["gain_vs_held"] = gain
    if best is None:
        base["in"] = ""
        base["out"] = ""
        base["gain_vs_held"] = 0.0
        return base
    return best


def solve_wildcard(
    players: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    budget: int,
) -> dict[str, Any]:
    """Linear fifteen. Incoming players already carry the minutes flag."""
    candidates = [row for row in players if row["owned"] or row["buyable"]]
    counts: dict[str, int] = {pos: 0 for pos in SQUAD_QUOTA}
    for row in candidates:
        counts[str(row["position"])] = counts.get(str(row["position"]), 0) + 1
    short = [pos for pos, need in SQUAD_QUOTA.items() if counts.get(pos, 0) < need]
    if short:
        return {"feasible": False, "reason": "minutes filter leaves " + ", ".join(short)}
    ordered = list(candidates)
    size = len(ordered)
    objective = np.array(
        [
            -(
                sum(float(weight) * float(score) for weight, score in zip(weights, row["weeks"], strict=True))
                - TIE_BREAK * int(row["pid"])
            )
            for row in ordered
        ]
    )
    rows: list[np.ndarray] = [np.ones(size)]
    lower = [float(sum(SQUAD_QUOTA.values()))]
    upper = [float(sum(SQUAD_QUOTA.values()))]
    for pos, need in SQUAD_QUOTA.items():
        row = np.zeros(size)
        for index, candidate in enumerate(ordered):
            if candidate["position"] == pos:
                row[index] = 1.0
        rows.append(row)
        lower.append(float(need))
        upper.append(float(need))
    for club in sorted({str(row["club"]) for row in ordered}):
        row = np.zeros(size)
        for index, candidate in enumerate(ordered):
            if candidate["club"] == club:
                row[index] = 1.0
        rows.append(row)
        lower.append(0.0)
        upper.append(float(MAX_PER_CLUB))
    cost = np.array([float(row["now_cost"]) for row in ordered])
    rows.append(cost)
    lower.append(-np.inf)
    upper.append(float(budget))
    result = milp(
        c=objective,
        integrality=np.ones(size),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(np.vstack(rows), lower, upper),
    )
    if not result.success or result.x is None:
        return {"feasible": False, "reason": f"solver status {result.message}"}
    picked = [row for row, bit in zip(ordered, result.x, strict=True) if float(bit) > 0.5]
    if len(picked) != sum(SQUAD_QUOTA.values()):
        return {"feasible": False, "reason": "solver returned a partial squad"}
    sheet = _sheet(picked, weights, move="wildcard")
    sheet["cost"] = sum(int(row["now_cost"]) for row in picked)
    return sheet


def compare_plans(
    players: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    budget: int,
    bank: int,
) -> dict[str, Any]:
    """Held, one free transfer, and wildcard on one score table."""
    held_rows = [row for row in players if row["owned"]]
    held = _sheet(held_rows, weights, move="hold")
    held["gain_vs_held"] = 0.0
    free = best_free_transfer(players, weights, bank)
    wild = solve_wildcard(players, weights, budget)
    gain = None
    given_up = None
    if wild.get("feasible"):
        gain = float(wild["decision_number"]) - float(free["decision_number"])
        given_up = float(wild["week0_xi_captain"]) - float(free["week0_xi_captain"])
    return {
        "held": held,
        "free_transfer": free,
        "wildcard": wild,
        "wildcard_minus_free_transfer": gain,
        "decision_week_xi_given_up": given_up,
    }


def _started_weeks(logs: pd.DataFrame, gws: Sequence[int]) -> dict[int, int]:
    wanted = {int(gw) for gw in gws}
    counts: dict[int, set[int]] = {}
    for row in logs.itertuples(index=False):
        gw = int(row.gw)
        if gw not in wanted:
            continue
        try:
            started = int(row.starts) >= 1
        except (TypeError, ValueError):
            started = False
        if not started:
            continue
        counts.setdefault(int(row.player_id), set()).add(gw)
    return {pid: len(weeks) for pid, weeks in counts.items()}


def _bootstrap_for(capture: Path) -> tuple[dict[str, Any], str]:
    stamp = ""
    name = capture.name
    if name.startswith("official_") and name.endswith(".csv"):
        stamp = name[len("official_") : -len(".csv")]
        sibling = capture.parent / f"bootstrap_{stamp}.json"
        if sibling.is_file():
            return json.loads(sibling.read_text(encoding="utf-8")), str(sibling)
    return json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8")), str(BOOTSTRAP_PATH)


def _capture_kind(capture: Path) -> str:
    marker = capture.parent / "slot_t1.json"
    if not marker.is_file():
        return "dry_run"
    meta = json.loads(marker.read_text(encoding="utf-8"))
    official = Path(str(meta.get("official") or "")).name
    if official == capture.name and str(meta.get("slot")) == "t1":
        return "decision"
    return "dry_run"


def build_plan(capture: Path, *, gw: int = 6) -> dict[str, Any]:
    """One judgement from a named ``ep_next`` file. Odds are the stored file."""
    capture = Path(capture)
    ep = load_ep_next(int(gw), path=capture)
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    logs = pd.read_csv(LOG_PATH)
    bootstrap, bootstrap_source = _bootstrap_for(capture)
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    from src.live.betfair_props import betfair_gw_lines

    odds = load_odds_frame(ODDS_PATH, betfair_gw_lines(int(gw)))
    names = team_names(bootstrap)
    pots = opening_pots_by_team_gw(odds, list(fixtures), names)
    owned_ids = [int(player["id"]) for player in final_players(entry)]
    missing = [pid for pid in owned_ids if player_key(pid) not in ep]
    if missing:
        raise ScorerError("owned players have no ep_next capture: " + ", ".join(str(pid) for pid in missing))
    purchases = reconstruct_purchases(entry, gameweek_values(logs, 1))
    costs = current_costs(bootstrap)
    prior = [int(gw) - 3, int(gw) - 2, int(gw) - 1]
    started = _started_weeks(logs, prior)
    gws = [int(gw) + step for step in range(HORIZON_STEPS)]
    weights = horizon_weights()
    priced = {int(key[0]) for key in pots}
    unpriced = [week for week in gws[1:] if week not in priced]
    elements = {int(row["id"]): row for row in bootstrap["elements"]}
    players: list[dict[str, Any]] = []
    owned = set(owned_ids)
    for pid, element in elements.items():
        key = player_key(pid)
        if pid not in owned and key not in ep:
            continue
        if pid in owned and key not in ep:
            continue
        club_name = names.get(int(element["team"]), "")
        club = norm_team(club_name)
        base = _first_lam(pots, gws[0], club)
        weeks: list[float] = []
        for step, week in enumerate(gws):
            if step == 0:
                weeks.append(float(ep[key]))
            else:
                weeks.append(later_week_score(float(ep[key]), base, _first_lam(pots, week, club)))
        chance = element.get("chance_of_playing_next_round")
        secure = minutes_ok(started.get(pid, 0), str(element.get("status") or ""), chance)
        now = int(costs[pid])
        sell = sell_price(int(purchases[pid]), now) if pid in owned else 0
        players.append(
            {
                "pid": pid,
                "name": str(element.get("web_name") or pid),
                "position": ELEMENT[int(element["element_type"])],
                "club": club_name,
                "now_cost": now,
                "sell": sell,
                "owned": pid in owned,
                "buyable": secure,
                "weeks": weeks,
            }
        )
    found = {int(row["pid"]) for row in players if row["owned"]}
    if found != owned:
        raise ScorerError("the owned fifteen are not all on the capture bootstrap")
    bank = int(entry["bank"])
    budget = bank + sum(int(row["sell"]) for row in players if row["owned"])
    plans = compare_plans(players, weights, budget=budget, bank=bank)
    flagged = [str(row["name"]) for row in players if row["owned"] and not minutes_ok(
        started.get(int(row["pid"]), 0),
        str(elements[int(row["pid"])].get("status") or ""),
        elements[int(row["pid"])].get("chance_of_playing_next_round"),
    )]
    stamp = ""
    frame = pd.read_csv(capture)
    if "captured_at" in frame.columns and len(frame):
        stamp = str(frame["captured_at"].iloc[0])
    kind = _capture_kind(capture)
    return {
        "decision_id": f"gw{int(gw)}-{kind}-{stamp or capture.stem}",
        "kind": kind,
        "gw": int(gw),
        "capture": str(capture),
        "captured_at": stamp,
        "bootstrap": bootstrap_source,
        "chip_intent": "wildcard",
        "counts_toward_chip_rule": False,
        "gamma": float(GAMMA),
        "weights": weights,
        "horizon_gws": gws,
        "unpriced_gws": unpriced,
        "budget": budget,
        "bank": bank,
        "owned_failing_filter": flagged,
        "plans": plans,
        "h1_wildcard_expires_gw": FIRST_HALF_END_GW,
        "realised_outcome": None,
        "requirement": False,
        "note": (
            "A requirement was not pre-registered. The gap is a logged preference. "
            "It does not count toward a chip rule."
        ),
    }


def chip_calendar(
    fixtures: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    played: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Use-by gameweek for each chip, and the fixture count in between."""
    wallet = ChipWallet()
    for gw, chip in sorted((int(week), str(name)) for week, name in played.items()):
        wallet.play(gw, chip)
    deadlines = {int(event["id"]): str(event.get("deadline_time") or "") for event in events}
    rows: list[dict[str, Any]] = []
    for chip in CHIPS:
        for half, end in (("H1", FIRST_HALF_END_GW), ("H2", N_GAMEWEEKS)):
            if (half, chip) in wallet.used:
                rows.append(
                    {
                        "chip": chip,
                        "half": half,
                        "status": "played",
                        "use_by_gw": end,
                        "first_gw": None,
                        "deadline": deadlines.get(end, ""),
                    }
                )
                continue
            start = 1 if half == "H1" else FIRST_HALF_END_GW + 1
            first = None
            for gw in range(start, end + 1):
                if chip in wallet.available(gw):
                    first = gw
                    break
            rows.append(
                {
                    "chip": chip,
                    "half": half,
                    "status": "open" if first is not None else "closed",
                    "use_by_gw": end,
                    "first_gw": first,
                    "deadline": deadlines.get(end, ""),
                }
            )
    return rows


def render_plan(record: Mapping[str, Any], calendar: Sequence[Mapping[str, Any]], fixtures: Sequence[Mapping[str, Any]]) -> str:
    """The judgement note. A dry run says so."""
    plans = record["plans"]
    lines = [
        f"# Gameweek {record['gw']} wildcard judgement",
        "",
    ]
    if record["kind"] != "decision":
        lines.append(
            "This is a dry run. It is not the T-1h decision. "
            "The decision file is slot_t1.json."
        )
        lines.append("")
    lines.append(
        f"Capture `{record['captured_at'] or record['capture']}`. "
        "The wildcard intent preceded this file. "
        "The gap does not count toward a chip rule, and it is not a measure of skill."
    )
    lines.append("")
    lines.append(
        f"Weights {', '.join(f'{float(weight):.3f}' for weight in record['weights'])} "
        f"from gamma {record['gamma']}. "
        f"Unpriced gameweeks: {', '.join(str(gw) for gw in record['unpriced_gws']) or 'none'}. "
        "Their weight is not moved onto the priced weeks."
    )
    lines.append("")
    lines.append(
        f"Budget {record['budget']} tenths, bank {record['bank']} tenths. "
        "Selling prices are half the rise, rounded down, and the full fall."
    )
    if record["owned_failing_filter"]:
        lines.append("")
        lines.append(
            "Owned players who fail the minutes filter, and can be kept but not bought back: "
            + ", ".join(record["owned_failing_filter"])
            + "."
        )
    lines.append("")
    lines.append("| Plan | Decision number | Squad sum | GW6 XI+captain | Move |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key, label in (
        ("held", "Held"),
        ("free_transfer", "Free transfer"),
        ("wildcard", "Wildcard"),
    ):
        plan = plans[key]
        if not plan.get("feasible", True):
            lines.append(f"| {label} |  |  |  | {plan.get('reason', 'infeasible')} |")
            continue
        lines.append(
            f"| {label} | {float(plan['decision_number']):.2f} | {float(plan['squad_sum']):.2f} | "
            f"{float(plan['week0_xi_captain']):.2f} | {plan['move']} |"
        )
    lines.append("")
    wild = plans["wildcard"]
    if wild.get("feasible"):
        lines.append(
            f"Wildcard minus the free-transfer plan: {float(plans['wildcard_minus_free_transfer']):.2f} "
            "on the discounted XI plus captain. "
            f"The Gameweek {record['gw']} XI plus captain given up by waiting is "
            f"{float(plans['decision_week_xi_given_up']):.2f}. "
            f"The first-half wildcard remains available through gameweek {record['h1_wildcard_expires_gw']}."
        )
        lines.append("")
        priced = [int(gw) for gw in record["horizon_gws"] if int(gw) not in set(record["unpriced_gws"])]
        priced_bits = [
            f"gameweek {gw}" if int(gw) == int(record["gw"]) else f"fixture-scaled gameweek {gw}"
            for gw in priced
        ]
        priced_text = " and ".join(priced_bits)
        unpriced_bits = [f"gameweek {gw}" for gw in record["unpriced_gws"]]
        unpriced_text = " and ".join(unpriced_bits)
        lines.append(
            f"Gameweeks 1 to {int(record['gw']) - 1} are finished, so the next legal week is "
            f"gameweek {record['gw']}. Unpriced weeks add nothing, so a priced horizon shorter "
            "than four steps is the one in the table. The one-week gap is the waiting figure. "
            "The discounted gap also includes later weeks whose scores are the decision-week "
            "ep_next scaled by the fixture ratio."
        )
        lines.append("")
        zero = f", with {unpriced_text} contributing zero" if unpriced_bits else ""
        lines.append(
            f"The {float(plans['wildcard_minus_free_transfer']):.2f} discounted margin reflects "
            f"expected gains across only {len(priced)} priced gameweeks ({priced_text}{zero}). "
            "It is a descriptive decision preference for this squad, not a four-week sum "
            "and not a certified evaluation of an autonomous chip rule."
        )
        lines.append("")
        lines.append(
            f"Wildcard XI ({wild['formation']}), captain {wild['captain']}: "
            + ", ".join(wild["xi"])
            + "."
        )
        lines.append("")
        lines.append("Wildcard fifteen: " + ", ".join(wild["players"]) + ".")
    else:
        lines.append(f"Wildcard is infeasible: {wild.get('reason', '')}.")
    lines.append("")
    lines.append(str(record["note"]))
    lines.append("")
    lines.append("Realised outcome: blank until the matches are played.")
    lines.append("")
    lines.append("## Chip calendar")
    lines.append("")
    lines.append("| Chip | Half | Status | First gameweek | Use by | Deadline |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in calendar:
        first = "" if row["first_gw"] is None else str(row["first_gw"])
        lines.append(
            f"| {row['chip']} | {row['half']} | {row['status']} | {first} | "
            f"{row['use_by_gw']} | {row['deadline']} |"
        )
    lines.append("")
    lines.append(
        "A free hit cannot be played in the gameweek next to another free hit. One chip per week. "
        "The first-gameweek column is the earliest week the rules allow, including weeks already played. "
        f"From gameweek {record['gw']} onward the remaining first-half chips run to their use-by week."
    )
    lines.append("")
    lines.append(f"## Fixtures through gameweek {FIRST_HALF_END_GW}")
    lines.append("")
    lines.append("| GW | Matches | Clubs | Doubles |")
    lines.append("| --- | --- | --- | --- |")
    for row in fixture_calendar(fixtures, int(record["gw"]), FIRST_HALF_END_GW):
        lines.append(f"| {row['gw']} | {row['matches']} | {row['clubs']} | {row['doubles']} |")
    lines.append("")
    return "\n".join(lines)


def append_log(record: Mapping[str, Any], dest: Path = LOG_DEST) -> None:
    """One row per decision id. A second write of the same id replaces that row."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    if dest.is_file():
        for line in dest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            stored = json.loads(line)
            if stored.get("decision_id") == record["decision_id"]:
                continue
            rows.append(line)
    rows.append(json.dumps(record, sort_keys=True))
    dest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildcard judgement from one ep_next capture.")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--gw", type=int, default=6)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    record = build_plan(args.capture, gw=args.gw)
    entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
    bootstrap, _source = _bootstrap_for(args.capture)
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    played = {int(row["gw"]): str(row["chip"]) for row in entry.get("chips_played") or []}
    calendar = chip_calendar(fixtures, bootstrap.get("events") or [], played)
    text = render_plan(record, calendar, fixtures)
    report = args.report
    if report is None:
        suffix = "dry_run" if record["kind"] != "decision" else "decision"
        report = ROOT / "reports" / f"wildcard_plan_gw{int(args.gw)}_{suffix}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text + "\n", encoding="utf-8")
    calendar_path = ROOT / "reports" / f"chip_calendar_gw{int(args.gw)}.md"
    calendar_path.write_text(text[text.index("## Chip calendar"):] + "\n", encoding="utf-8")
    append_log(record)
    gap = record["plans"]["wildcard_minus_free_transfer"]
    print(f"{record['kind']} {record['decision_id']} gap {gap}")


if __name__ == "__main__":
    main()
