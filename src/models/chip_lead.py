"""Why the carry never played a wildcard.

The lead is the one the carry already computes: the rebuilt eleven minus
the held eleven, summed over the priced weeks. A chip signing is tagged
from the pool that week. The reading was locked before these totals were
read. A pool replay or a hurdle of 12 is a side call. Neither number is
written into the published rule.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.friend_start import _early
from src.models.reset_chips import StepOutlook, choose_chip
from src.models.reset_gap import PROCESSED, REPORTS, _fmt
from src.models.season_climb_ft import _gw_pool
from src.models.squad_gap import label_transfers
from src.rules.fpl_2026 import MAX_PER_CLUB, ChipWallet, sell_price
from src.live.policy import FH_MARGIN, WC_MARGIN

CHIP_BASE = -329.0
OPEN_CUT = -40.0
POOL_SHARE = 0.5
LATE_LOW = 12.0
LATE_HIGH = 16.0
LATE_N = 8
FINISHED_MIN = 12
HOLD_GAP = 1.25
TIE = 1e-6
WINDOW_CSV = PROCESSED / "chip_lead_gw15.csv"
REPORT = REPORTS / "chip_lead_gw15.md"


def reconstruct_choice(row: dict[str, Any], available: tuple[str, ...]) -> str | None:
    """The chip `choose_chip` would play from the stored outlook."""
    now = StepOutlook(
        gw=int(row["gw"]),
        held_xi=0.0,
        bench_xp=float(row["bench_xp"]),
        cap_xp=float(row["cap_xp"]),
        rebuilt_xi=float(row["wc_sum"]),
        fh_xi=float(row["fh_margin"]),
    )
    later = StepOutlook(
        gw=int(row["gw"]) + 1,
        held_xi=0.0,
        bench_xp=float(row["later_bench"]),
        cap_xp=float(row["later_cap"]),
        rebuilt_xi=0.0,
        fh_xi=0.0,
    )
    chip, _gain = choose_chip([now, later], available)
    return chip


def bug_status(row: dict[str, Any], available: tuple[str, ...]) -> str:
    """bug, outranked, tie, or clear. A tie that plays nothing is allowed."""
    expected = reconstruct_choice(row, available)
    played = row["chip"] or None
    if expected != played:
        return "bug"
    name = "wildcard"
    lead = float(row["wc_sum"])
    hurdle = WC_MARGIN
    if name in available and lead + TIE >= hurdle and played != name:
        return _cleared_but_unplayed(row, available, name, lead)
    name = "free_hit"
    lead = float(row["fh_margin"])
    hurdle = FH_MARGIN
    if name in available and lead + TIE >= hurdle and played != name:
        return _cleared_but_unplayed(row, available, name, lead)
    return "clear"


def _cleared_but_unplayed(
    row: dict[str, Any], available: tuple[str, ...], name: str, lead: float
) -> str:
    """Outranked when another legal chip's gain is strictly larger."""
    gains = _cleared_gains(row, available)
    larger = [
        chip for chip, gain in gains.items() if chip != name and gain > lead + TIE
    ]
    if larger:
        return "outranked"
    near = [
        chip
        for chip, gain in gains.items()
        if chip != name and abs(gain - lead) <= TIE
    ]
    if near and row["chip"] is None:
        return "tie"
    return "bug"


def _cleared_gains(row: dict[str, Any], available: tuple[str, ...]) -> dict[str, float]:
    gains: dict[str, float] = {}
    legal = set(available)
    cap = float(row["cap_xp"])
    bench = float(row["bench_xp"])
    if "triple_captain" in legal and cap > 0 and cap > float(row["later_cap"]):
        gains["triple_captain"] = cap
    if "bench_boost" in legal and bench > 0 and bench > float(row["later_bench"]):
        gains["bench_boost"] = bench
    if "free_hit" in legal and float(row["fh_margin"]) >= FH_MARGIN:
        gains["free_hit"] = float(row["fh_margin"])
    if "wildcard" in legal and float(row["wc_sum"]) >= WC_MARGIN:
        gains["wildcard"] = float(row["wc_sum"])
    return gains


def _pool_index(pool: pd.DataFrame) -> dict[str, Any]:
    frame = pool.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame = frame.drop_duplicates("player_id", keep="first")
    return {str(row.player_id): row for row in frame.itertuples()}


def _prior(row: Any) -> float | None:
    raw = getattr(row, "n_prior", None)
    if raw is None or raw != raw:
        return None
    return float(raw)


def buy_tag(player_id: str, pool_by: dict[str, Any]) -> str:
    """The first buy-pool reason. Eligible means the search could buy him."""
    src = pool_by.get(str(player_id))
    if src is None:
        return "absent"
    if bool(getattr(src, "eligible", False)):
        return "eligible"
    prior = _prior(src)
    if prior is None or prior < 3:
        return "appearances"
    return "minutes"


def pair_tag(
    human_id: str,
    model_id: str,
    human_xp: float,
    model_xp: float,
    pool_by: dict[str, Any],
    pre_ids: set[str],
    purchase: dict[str, int],
    bank: int,
) -> str:
    """Why the higher-scored human player was not the owned starter."""
    tag = buy_tag(human_id, pool_by)
    if tag != "eligible":
        return tag
    if model_id not in pre_ids:
        return "arrived"
    human = pool_by[str(human_id)]
    model = pool_by.get(str(model_id))
    if model is None or str(model_id) not in purchase:
        return "arrived"
    price = int(getattr(human, "value"))
    funded = sell_price(int(purchase[str(model_id)]), int(getattr(model, "value"))) + int(bank)
    if price > funded:
        return "price"
    clubs = [
        str(getattr(pool_by[pid], "team_norm"))
        for pid in pre_ids
        if pid in pool_by
    ]
    if len(clubs) != len(pre_ids):
        raise RuntimeError("a squad player has no club")
    human_club = str(getattr(human, "team_norm"))
    model_club = str(getattr(model, "team_norm"))
    count = sum(1 for club in clubs if club == human_club)
    if model_club == human_club:
        count -= 1
    if count + 1 > MAX_PER_CLUB:
        return "club"
    lead = float(human_xp) - float(model_xp)
    if lead <= HOLD_GAP + TIE:
        return "inside 1.25"
    return "open"


def explain_week(week: dict[str, Any], pool: pd.DataFrame) -> list[dict[str, Any]]:
    """One row per chip signing and per ranked-lower pair."""
    labelled = label_transfers(
        week["transfer_model_detail"],
        week["transfer_their_detail"],
        set(week["human_pre_ids"]),
        week["his_chip"],
    )
    pool_by = _pool_index(pool)
    pre_ids = set(week["pre_ids"])
    purchase = {str(pid): int(price) for pid, price in week["purchase"].items()}
    bank = int(week["bank"])
    rows: list[dict[str, Any]] = []
    for item in labelled:
        if item["tag"] == "chip_squad":
            rows.append(
                {
                    "kind": "chip_squad",
                    "tag": buy_tag(str(item["id"]), pool_by),
                    "points": float(item["points"]),
                    "human_id": str(item["id"]),
                    "model_id": "",
                    "position": item.get("position") or "",
                }
            )
        elif item["tag"] == "ranked_lower":
            model_id, human_id = str(item["id"]).split("|", 1)
            rows.append(
                {
                    "kind": "ranked_lower",
                    "tag": pair_tag(
                        human_id,
                        model_id,
                        float(item["human_xp"]),
                        float(item["model_xp"]),
                        pool_by,
                        pre_ids,
                        purchase,
                        bank,
                    ),
                    "points": float(item["points"]),
                    "human_id": human_id,
                    "model_id": model_id,
                    "position": item.get("position") or "",
                }
            )
    return rows


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def blocked_share(chip_rows: list[dict[str, Any]]) -> float | None:
    """Share of the published −329 that is absent or short of three appearances."""
    if not chip_rows:
        return None
    blocked = sum(
        float(row["points"])
        for row in chip_rows
        if row["tag"] in {"absent", "appearances"}
    )
    return float(blocked / CHIP_BASE)


def decide(
    *,
    n_finished: int,
    bugs: int,
    share: float | None,
    n_late: int,
    late_median: float | None,
    open_points: float,
) -> str:
    """One reading. The early median is not an input."""
    if n_finished < FINISHED_MIN:
        return "inconclusive"
    if bugs:
        return "bug"
    if share is not None and share >= POOL_SHARE:
        return "pool"
    if (
        n_late >= LATE_N
        and late_median is not None
        and late_median + TIE >= LATE_LOW
        and late_median < LATE_HIGH
    ):
        return "margin"
    if open_points < OPEN_CUT:
        return "open"
    return "stop"


def week_status(weeks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legal wildcard weeks, in played order, plus the bug flag."""
    wallet = ChipWallet()
    out = []
    for week in weeks:
        gw = int(week["gw"])
        available = wallet.available(gw)
        status = bug_status(week, available)
        legal = "wildcard" in available
        out.append(
            {
                "gw": gw,
                "chip": week["chip"] or "",
                "legal": legal,
                "early": legal and gw in {2, 3},
                "late": legal and gw in {4, 5},
                "wc_sum": float(week["wc_sum"]),
                "fh_margin": float(week["fh_margin"]),
                "status": status,
            }
        )
        played = week["chip"] or None
        if played:
            wallet.play(gw, played)
    return out


def analyse(
    managers: list[dict[str, Any]], explained: list[dict[str, Any]]
) -> dict[str, Any]:
    """The locked reading from one carry. The reference line stays out."""
    cohort = [row for row in managers if row["group"] != "reference" and row["finished"]]
    leads = []
    for manager in cohort:
        for row in week_status(manager["weeks"]):
            leads.append({**row, "entry_id": manager["entry_id"], "group": manager["group"]})
    bugs = sum(1 for row in leads if row["status"] == "bug")
    early = [float(row["wc_sum"]) for row in leads if row["early"]]
    late = [float(row["wc_sum"]) for row in leads if row["late"]]
    chip_rows = [
        row for row in explained if row["kind"] == "chip_squad" and row["group"] != "reference"
    ]
    ranked = [
        row for row in explained if row["kind"] == "ranked_lower" and row["group"] != "reference"
    ]
    chip_total = sum(float(row["points"]) for row in chip_rows)
    share = blocked_share(chip_rows)
    open_points = sum(float(row["points"]) for row in ranked if row["tag"] == "open")
    drifted = abs(chip_total - CHIP_BASE) > 1.0
    if len(cohort) < FINISHED_MIN:
        call = "inconclusive"
    elif bugs:
        call = "bug"
    elif drifted:
        call = "inconclusive"
    else:
        call = decide(
            n_finished=len(cohort),
            bugs=0,
            share=share,
            n_late=len(late),
            late_median=_median(late),
            open_points=open_points,
        )
    return {
        "call": call,
        "n_finished": len(cohort),
        "bugs": bugs,
        "chip_total": chip_total,
        "share": share,
        "n_early": len(early),
        "n_late": len(late),
        "early_median": _median(early),
        "late_median": _median(late),
        "open_points": open_points,
        "leads": leads,
        "chip_rows": chip_rows,
        "ranked": ranked,
        "drifted": drifted,
    }


def _tag_line(rows: list[dict[str, Any]]) -> str:
    tags: dict[str, list[float]] = {}
    for row in rows:
        tags.setdefault(str(row["tag"]), []).append(float(row["points"]))
    parts = []
    for tag in (
        "absent",
        "appearances",
        "minutes",
        "eligible",
        "price",
        "club",
        "inside 1.25",
        "open",
        "arrived",
    ):
        if tag not in tags:
            continue
        points = tags[tag]
        parts.append(f"{tag} {_fmt(sum(points))} from {len(points)}")
    return ", ".join(parts) if parts else "none"


def write_report(path: Path, result: dict[str, Any]) -> None:
    call = result["call"]
    sentences = {
        "bug": "A legal chip cleared its hurdle and was not played, and nothing outranked it. No replay.",
        "pool": "At least half of the chip-signing points were players the buy pool could not see. The pool replay is next. The hurdle stays 16.",
        "margin": "The late wildcard lead sits between 12 and 16. The side replay uses 12 for that call only. The published hurdle stays 16.",
        "open": "Neither chip lever fired. The open pairs, a score lead above 1.25 that was still affordable, are the next gap.",
        "stop": "Neither chip lever fired, and the open pairs are not past 40 points. This batch stops.",
        "inconclusive": "The carry did not reproduce the chip-signing total, or too few managers finished. No replay.",
    }
    early = result["early_median"]
    late = result["late_median"]
    share = result["share"]
    lines = [
        "# Wildcard lead on the carried fifteens",
        "",
        "The lead is the rebuilt eleven minus the held eleven, summed over the priced weeks. That is the number already compared with 16. Gameweek 1 cannot play a wildcard. A week after a wildcard has been used is left out of the median. The reference manager is left out of the reading.",
        "",
        "A chip signing is absent when he is not in that week's pool, or short of appearances when he is in it and has fewer than three. The share is those points divided by the published −329. Half or more sends the next replay to the chip buy pool: one or two appearances, the early score capped at 6, and only inside a wildcard or a free hit. Ordinary transfers stay on three appearances.",
        "",
        "If that share is under half, and the Gameweek 4–5 median lead is at least 12 and under 16, the next replay passes 12 as the wildcard hurdle for that call. The published hurdle stays 16. No other hurdle is tried. The Gameweek 2–3 median does not choose this.",
        "",
        "A cleared chip that loses to a strictly larger chip is outranked. A tie that plays nothing is allowed. Anything else that clears and is not played stops the batch.",
        "",
        sentences[call],
        "",
        (
            f"Finished {result['n_finished']} of the 14. "
            f"Bugs {result['bugs']}. "
            f"Chip signings {_fmt(result['chip_total'])}. "
            f"Blocked share {'' if share is None else f'{share:.2f}'}. "
            f"Early median {'' if early is None else f'{early:.2f}'} on {result['n_early']} weeks. "
            f"Late median {'' if late is None else f'{late:.2f}'} on {result['n_late']} weeks. "
            f"Open pairs {_fmt(result['open_points'])}."
        ),
        "",
        f"Chip signings: {_tag_line(result['chip_rows'])}.",
        f"Ranked lower: {_tag_line(result['ranked'])}.",
        "",
    ]
    if call == "open":
        lines.append("Open pairs:")
        lines.append("")
        for row in result["ranked"]:
            if row["tag"] != "open":
                continue
            lines.append(
                f"- {row['label']} GW{int(row['gw'])} {row['position']}: "
                f"{row['model_id']} against {row['human_id']}, {_fmt(float(row['points']))}."
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    """Replay the locked carry and read the leads. Does not retune."""
    from src.live.benchmark import build_frames
    from src.models.cohort_carry import carry_entry, cohort_specs
    from src.models.open_horizon import attach_opening_horizon
    from src.models.reset_gap import merged_clubs

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
    managers: list[dict[str, Any]] = []
    explained: list[dict[str, Any]] = []
    for spec in specs:
        print(f"lead {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        managers.append(result)
        if not result["finished"]:
            continue
        for week in result["weeks"]:
            pool = _gw_pool(feat, roster, int(week["gw"]), set(week["pre_ids"]), early)
            for row in explain_week(week, pool):
                explained.append(
                    {
                        **row,
                        "entry_id": result["entry_id"],
                        "label": result["label"],
                        "group": result["group"],
                        "gw": int(week["gw"]),
                    }
                )
    outcome = analyse(managers, explained)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outcome["leads"]).to_csv(WINDOW_CSV, index=False)
    detail = PROCESSED / "chip_lead_players_gw15.csv"
    pd.DataFrame(explained).to_csv(detail, index=False)
    write_report(REPORT, outcome)
    return {
        "call": outcome["call"],
        "share": outcome["share"],
        "late_median": outcome["late_median"],
        "early_median": outcome["early_median"],
        "chip_total": outcome["chip_total"],
        "open_points": outcome["open_points"],
        "bugs": outcome["bugs"],
    }


if __name__ == "__main__":
    print(run())
