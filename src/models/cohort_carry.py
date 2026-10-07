"""The free-wallet carry on the locked live-season cohort.

Each manager starts from his own Gameweek 1 fifteen. The squad carries.
His chips are not copied. The chip rule is the priced-horizon wallet.
Entry 1078627 stays out. ojaminFC is a reference row and stays out of the
mean.

The bar was locked before these totals were read. A manager gains when
his five-week gap is above 0, at least 3 weeks are non-negative, and the
residual is 0. The batch is ahead when at least 8 of the 14 finishers
gain and the mean gap is above 0. Anything else is the batch not showing
the model ahead of these starting fifteens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import GWS, build_frames
from src.live.entry import load_entry
from src.models.blank_context import apply_fixture_tags
from src.models.friend_start import assert_carried, one_week
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import PROCESSED, REPORTS, _fmt, beats_five, merged_clubs, pre_deadline
from src.models.stage_46_transfer_gap import RANK_SLOTS, REFERENCE, VETERANS
from src.rules.fpl_2026 import ChipWallet

COHORT_N = 14
GAINS_NEEDED = 8
FRIEND_ID = 1078627


def cohort_specs() -> list[dict[str, Any]]:
    """The 14 locked managers, then the reference row. The friend is absent."""
    specs = [dict(row) for row in (*VETERANS, *RANK_SLOTS, REFERENCE)]
    ids = [int(row["entry_id"]) for row in specs]
    if FRIEND_ID in ids:
        raise RuntimeError("entry 1078627 is already reviewed and stays out")
    if ids.count(int(REFERENCE["entry_id"])) != 1:
        raise RuntimeError("ojaminFC is missing from the reference row")
    if len([row for row in specs if row["group"] != "reference"]) != COHORT_N:
        raise RuntimeError("the cohort is not 14 managers")
    return specs


def manager_result(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One carried manager. A non-zero residual is not a gain."""
    if len(rows) != 5:
        raise RuntimeError("a carried manager needs five weeks")
    gaps = [float(row["gap"]) for row in rows]
    residual = float(sum(row["residual"] for row in rows))
    residual_ok = all(abs(float(row["residual"])) < 1e-6 for row in rows)
    point_gain = beats_five(gaps)
    chips = [f"GW{int(row['gw'])} {row['chip']}" for row in rows if row["chip"]]
    his_chips = [f"GW{int(row['gw'])} {row['his_chip']}" for row in rows if row["his_chip"]]
    boosts = [row for row in rows if row["chip"] == "bench_boost"]
    late_wildcard = [
        int(row["gw"]) for row in rows if row["chip"] == "wildcard" and int(row["gw"]) >= 4
    ]
    return {
        "entry_id": int(spec["entry_id"]),
        "label": str(spec["label"]),
        "group": str(spec["group"]),
        "finished": True,
        "identity_ok": True,
        "residual_ok": residual_ok,
        "gain": bool(point_gain and residual_ok),
        "model": float(sum(row["model_points"] for row in rows)),
        "his": float(sum(row["their_points"] for row in rows)),
        "gap": float(sum(gaps)),
        "non_negative": int(sum(g >= -1e-6 for g in gaps)),
        "captain_gap": float(sum(row["captain_gap"] for row in rows)),
        "transfer_gap": float(sum(row["transfer_gap"] for row in rows)),
        "lineup_gap": float(sum(row["lineup_gap"] for row in rows)),
        "hit_gap": float(sum(row["hit_gap"] for row in rows)),
        "bench_gap": float(sum(row["bench_gap"] for row in rows)),
        "residual": residual,
        "model_chips": chips,
        "his_chips": his_chips,
        "bench_outlook": None if not boosts else float(boosts[0]["bench_xp"]),
        "bench_scored": None if not boosts else float(boosts[0]["bench_scored"]),
        "bench_gw": None if not boosts else int(boosts[0]["gw"]),
        "late_wildcard": late_wildcard,
        "error": "",
        "weeks": rows,
    }


def failed_result(spec: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "entry_id": int(spec["entry_id"]),
        "label": str(spec["label"]),
        "group": str(spec["group"]),
        "finished": False,
        "identity_ok": False,
        "residual_ok": False,
        "gain": False,
        "model": None,
        "his": None,
        "gap": None,
        "non_negative": None,
        "captain_gap": None,
        "transfer_gap": None,
        "lineup_gap": None,
        "hit_gap": None,
        "bench_gap": None,
        "residual": None,
        "model_chips": [],
        "his_chips": [],
        "bench_outlook": None,
        "bench_scored": None,
        "bench_gw": None,
        "late_wildcard": [],
        "error": error,
        "weeks": [],
    }


def batch_ahead(results: list[dict[str, Any]]) -> bool:
    """At least 8 gains and a positive mean. A failure blocks the call."""
    cohort = [row for row in results if row["group"] != "reference"]
    if len(cohort) != COHORT_N:
        return False
    if any(not row["finished"] or not row["identity_ok"] for row in cohort):
        return False
    gains = sum(1 for row in cohort if row["gain"])
    mean = sum(float(row["gap"]) for row in cohort) / COHORT_N
    return gains >= GAINS_NEEDED and mean > 0


def _mean(rows: list[dict[str, Any]]) -> float | None:
    finished = [row for row in rows if row["finished"] and row["gap"] is not None]
    if len(finished) != len(rows) or not rows:
        return None
    return float(sum(float(row["gap"]) for row in finished) / len(finished))


def carry_entry(
    spec: dict[str, Any],
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    clubs: dict[int, set[str]],
    roster_by_gw: dict[int, set[str]],
    horizon_scores: Any,
) -> dict[str, Any]:
    """Carry one entry. A broken week is a failure row, not a dropped id."""
    try:
        entry = load_entry(int(spec["entry_id"]))
        state = pre_deadline(entry, roster, 1)
        if int(state.ft) != 0:
            raise RuntimeError("Gameweek 1 did not start with 0 free transfers")
        wallet = ChipWallet()
        weeks: list[dict[str, Any]] = []
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
            weeks.append(row)
        assert_carried(weeks)
        played = [row["chip"] for row in weeks if row["chip"]]
        if len(played) != len(set(played)):
            raise RuntimeError("a chip was played twice")
        if weeks[0]["chip"] in {"wildcard", "free_hit"}:
            raise RuntimeError("Gameweek 1 played a wildcard or a free hit")
        return manager_result(spec, weeks)
    except Exception as exc:
        return failed_result(spec, f"{type(exc).__name__}: {exc}")


def run() -> dict[str, Any]:
    """Carry the locked cohort. Does not retune."""
    specs = cohort_specs()
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    results: list[dict[str, Any]] = []
    for spec in specs:
        print(f"carry {spec['entry_id']} {spec['label']}", flush=True)
        results.append(carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores))
    ahead = batch_ahead(results)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write_csv(PROCESSED / "cohort_carry_gw15.csv", results)
    _write_report(REPORTS / "cohort_carry_gw15.md", results, ahead)
    cohort = [row for row in results if row["group"] != "reference"]
    return {
        "ahead": ahead,
        "gains": int(sum(1 for row in cohort if row["gain"])),
        "finished": int(sum(1 for row in cohort if row["finished"])),
        "mean": _mean(cohort),
        "reference_gap": next(row["gap"] for row in results if row["group"] == "reference"),
    }


def _chip_list(names: list[str]) -> str:
    return ", ".join(names) if names else "none"


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    flat = []
    for manager in results:
        for row in manager["weeks"]:
            flat.append(
                {
                    "entry_id": manager["entry_id"],
                    "label": manager["label"],
                    "group": manager["group"],
                    "gw": row["gw"],
                    "chip": row["chip"] or "",
                    "his_chip": row["his_chip"] or "",
                    "chip_gain": row["chip_gain"],
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
                    "n_transfers": row["n_transfers"],
                    "hits": row["hits"],
                }
            )
        if not manager["weeks"]:
            flat.append(
                {
                    "entry_id": manager["entry_id"],
                    "label": manager["label"],
                    "group": manager["group"],
                    "gw": "",
                    "error": manager["error"],
                }
            )
    pd.DataFrame(flat).to_csv(path, index=False)


def _group_line(title: str, rows: list[dict[str, Any]]) -> str:
    finished = [row for row in rows if row["finished"]]
    gains = sum(1 for row in finished if row["gain"])
    mean = _mean(finished)
    mean_text = "unfinished" if mean is None else _fmt(mean)
    return (
        f"{title}: {gains} gains out of {len(rows)}, mean gap {mean_text}."
    )


def _manager_line(row: dict[str, Any]) -> str:
    if not row["finished"]:
        return (
            f"| {row['label']} | {row['group']} |  |  |  |  | no |  |  |  |  |  | {row['error']} |"
        )
    boost = ""
    if row["bench_gw"] is not None:
        boost = f"GW{row['bench_gw']} {row['bench_outlook']:.2f} to {row['bench_scored']:.0f}"
    note = "unchecked split" if not row["residual_ok"] else ""
    if row["late_wildcard"]:
        weeks = ", ".join(f"GW{gw}" for gw in row["late_wildcard"])
        truncated = f"wildcard in {weeks} is cut off at Gameweek 5"
        note = f"{note}; {truncated}".strip("; ")
    return (
        f"| {row['label']} | {row['group']} | {row['model']:.0f} | {row['his']:.0f} | "
        f"{_fmt(row['gap'])} | {row['non_negative']} | {'yes' if row['gain'] else 'no'} | "
        f"{_fmt(row['captain_gap'])} | {_fmt(row['transfer_gap'])} | {_fmt(row['lineup_gap'])} | "
        f"{_fmt(row['hit_gap'])} | {_fmt(row['bench_gap'])} | {note} |"
    )


def _chip_line(row: dict[str, Any]) -> str:
    if not row["finished"]:
        return f"| {row['label']} | {row['group']} |  |  |  |"
    boost = ""
    if row["bench_gw"] is not None:
        boost = f"{row['bench_outlook']:.2f} / {row['bench_scored']:.0f}"
    return (
        f"| {row['label']} | {row['group']} | {_chip_list(row['model_chips'])} | "
        f"{_chip_list(row['his_chips'])} | {boost} |"
    )


def _write_report(path: Path, results: list[dict[str, Any]], ahead: bool) -> None:
    cohort = [row for row in results if row["group"] != "reference"]
    veterans = [row for row in cohort if row["group"] == "veteran"]
    ranks = [row for row in cohort if row["group"] == "rank"]
    gains = sum(1 for row in cohort if row["gain"])
    mean = _mean([row for row in cohort if row["finished"]])
    if ahead:
        verdict = (
            f"{gains} of 14 gained and the mean gap is {_fmt(mean)}. "
            "The batch is ahead on these starting fifteens."
        )
    else:
        mean_text = "unfinished" if mean is None else _fmt(mean)
        verdict = (
            f"{gains} of 14 gained and the mean gap is {mean_text}. "
            "The batch did not show the model ahead of these starting fifteens."
        )
    lines = [
        "# Gameweeks 1–5 from the locked cohort's starting fifteens",
        "",
        "The same carry as entry 1078627. Each manager starts from his own Gameweek 1 fifteen, with that week's bank and with no free transfer. The squad then carries. His chips are not copied. The wallet starts full. The chip rule is the priced horizon: at most three weeks, nothing after Gameweek 7, Bench Boost and Triple Captain only when that week is strictly best, Free Hit at 12, Wildcard at 16 on the priced weeks. Gameweek 1 cannot play Wildcard or Free Hit.",
        "",
        "The 14 were locked on 4 October, before those squads were opened. Seven are the veterans. Seven are the rank slots. ojaminFC is carried on the same rule and left out of the count and the mean. Entry 1078627 is already reviewed and is not in this batch. Their official points from the earlier empty-chip note were not used to choose this list or this bar.",
        "",
        "A manager gains when the five-week gap is above 0, at least 3 of 5 weeks are non-negative, and every residual is 0. The batch is ahead when at least 8 of the 14 gain and the mean gap is above 0. A manager who does not finish blocks that call. A Bench Boost is reported with its outlook and the points the bench scored. A wildcard in Gameweek 4 or 5 is marked, because the five-week window cuts the horizon short. The margin is not changed.",
        "",
        verdict + " Five weeks remain too few to call the rule reliable.",
        "",
        _group_line("Veterans", veterans),
        _group_line("Rank slots", ranks),
        _group_line("The 14", cohort),
        "",
        "| Manager | Group | Model | His | Gap | Weeks ≥ 0 | Gain | Captain | Transfers | Lineup | Hits | Bench | Note |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(_manager_line(row))
    lines += [
        "",
        "The six piece columns sum to the gap when the residual is 0. A non-zero residual marks the split unchecked, and that manager is not a gain.",
        "",
        "| Manager | Group | Model chips | His chips | Bench outlook / scored |",
        "|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(_chip_line(row))
    lines += [
        "",
        "Bench outlook is the expected bench on the squad before the transfers in the week the chip was played. Bench scored is the realised award.",
        "",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
