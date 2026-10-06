"""Discounted wildcard sum on the carried Gameweeks 1–5.

The published choice stays undiscounted. A flip is recorded and the later
weeks are not re-solved. The crowd climbs stay out.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.live.benchmark import build_frames
from src.live.policy import WC_MARGIN
from src.models.cohort_carry import carry_entry, cohort_specs
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_chips import StepOutlook, choose_chip, discounted_gap
from src.models.reset_gap import PROCESSED, REPORTS, merged_clubs
from src.models.season_climb_ft import GAMMA

WEEKS_CSV = PROCESSED / "chip_horizon_gw15.csv"
REPORT = REPORTS / "chip_horizon.md"
LIVE_LEADS = (12.56, 5.11)
LIVE_RAW = 17.67


def live_row() -> dict[str, float]:
    """The stored Gameweek 6 leads. No new search."""
    raw = float(LIVE_LEADS[0]) + float(LIVE_LEADS[1])
    if abs(raw - float(LIVE_RAW)) > 1e-9:
        raise RuntimeError(f"the stored live sum is {raw}, not {LIVE_RAW}")
    discounted = discounted_gap(list(LIVE_LEADS))
    return {"raw": raw, "discounted": discounted, "clears": discounted >= float(WC_MARGIN)}


def _steps(row: dict[str, Any]) -> list[StepOutlook]:
    gaps = [float(value) for value in row["step_gaps"]]
    caps = [float(value) for value in row["step_caps"]]
    benches = [float(value) for value in row["step_bench"]]
    window = [int(value) for value in row["horizon"]]
    if not gaps or len(gaps) != len(caps) or len(gaps) != len(benches) or len(gaps) != len(window):
        raise RuntimeError(f"GW{int(row['gw'])} steps do not line up")
    built: list[StepOutlook] = []
    for gw, gap, cap, bench in zip(window, gaps, caps, benches, strict=True):
        built.append(
            StepOutlook(
                gw=int(gw),
                held_xi=0.0,
                bench_xp=float(bench),
                cap_xp=float(cap),
                rebuilt_xi=float(gap),
                fh_xi=float(gap),
            )
        )
    return built


def _chip_name(chip: str | None) -> str:
    return "" if chip is None else str(chip)


def score_deadline(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """One carried deadline. The stored chip must match the undiscounted call."""
    steps = _steps(row)
    available = tuple(str(chip) for chip in row["open_chips"])
    raw_chip, raw_gain = choose_chip(steps, available)
    disc_chip, disc_gain = choose_chip(steps, available, gamma=GAMMA)
    stored = None if not row["chip"] else str(row["chip"])
    if raw_chip != stored:
        raise RuntimeError(
            f"{spec['label']} GW{int(row['gw'])} stored {stored} and the undiscounted call is {raw_chip}"
        )
    gaps = [float(value) for value in row["step_gaps"]]
    raw_sum = float(sum(gaps))
    if abs(raw_sum - float(row["wc_sum"])) > 1e-4:
        raise RuntimeError(f"{spec['label']} GW{int(row['gw'])} step gaps do not match the stored sum")
    disc_sum = discounted_gap(gaps)
    return {
        "entry_id": int(spec["entry_id"]),
        "label": str(spec["label"]),
        "group": str(spec["group"]),
        "gw": int(row["gw"]),
        "n_steps": len(gaps),
        "raw_sum": raw_sum,
        "disc_sum": disc_sum,
        "raw_clears": raw_sum >= float(WC_MARGIN),
        "disc_clears": disc_sum >= float(WC_MARGIN),
        "raw_chip": _chip_name(raw_chip),
        "disc_chip": _chip_name(disc_chip),
        "raw_gain": float(raw_gain),
        "disc_gain": float(disc_gain),
        "flip": raw_chip != disc_chip,
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if any(str(row["group"]) == "reference" for row in rows):
        raise RuntimeError("the reference manager is in the horizon count")
    if len(rows) != 70:
        raise RuntimeError(f"the horizon count has {len(rows)} weeks, not 70")
    flips = [row for row in rows if row["flip"]]
    crosses = [row for row in rows if bool(row["raw_clears"]) != bool(row["disc_clears"])]
    return {"n": len(rows), "n_flips": len(flips), "n_crosses": len(crosses)}


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"


def write_report(
    path: Any,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    live: dict[str, float],
) -> None:
    lines = [
        "# Chip horizon",
        "",
        "Counted after the lock in `reports/chip_horizon_plan.md`. "
        "The wildcard sum is weighted by 0.9 for each step ahead. "
        "The hurdle stays 16. Free hit, bench boost, and triple captain are unchanged. "
        "A flip is a different chip on that deadline. The weeks after it were not re-solved. "
        "The crowd climbs were not replayed.",
        "",
        (
            f"Gameweek 6, from the stored leads, is {_fmt(LIVE_LEADS[0])} this week and "
            f"{_fmt(LIVE_LEADS[1])} the next. The raw sum is {_fmt(live['raw'])}. "
            f"The discounted sum is {_fmt(live['discounted'])}. "
            f"It {'still clears' if live['clears'] else 'misses'} 16. No new search was run."
        ),
        "",
        (
            f"Deadlines: {summary['n']}. Flips: {summary['n_flips']}. "
            f"Weeks where the raw sum and the discounted sum disagree on 16: {summary['n_crosses']}."
        ),
        "",
    ]
    if summary["n_flips"] == 0:
        lines.append("The discount does not change a chip on these five weeks.")
    else:
        lines.append("A flip is listed here. The path after that deadline was left as it was.")
    lines.append("")
    interesting = [row for row in rows if row["flip"] or bool(row["raw_clears"]) != bool(row["disc_clears"])]
    if interesting:
        lines.extend(
            [
                "| Manager | GW | Steps | Raw sum | Discounted | Raw chip | Discounted chip |",
                "|---|---:|---:|---:|---:|---|---|",
            ]
        )
        ordered = sorted(interesting, key=lambda row: (str(row["label"]), int(row["gw"])))
        for row in ordered:
            lines.append(
                f"| {row['label']} | {int(row['gw'])} | {int(row['n_steps'])} | "
                f"{_fmt(row['raw_sum'])} | {_fmt(row['disc_sum'])} | "
                f"{row['raw_chip'] or 'none'} | {row['disc_chip'] or 'none'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    """Carry the 14 managers and compare the two wildcard sums. Does not retune."""
    live = live_row()
    specs = [row for row in cohort_specs() if row["group"] != "reference"]
    if len(specs) != 14:
        raise RuntimeError("the horizon cohort is not 14 managers")
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"horizon {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        if not result["finished"]:
            raise RuntimeError(f"{spec['label']} did not finish: {result['error']}")
        for week in result["weeks"]:
            rows.append(score_deadline(week, spec))
    summary = summarise(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WEEKS_CSV, index=False)
    write_report(REPORT, rows, summary, live)
    print(
        f"flips {summary['n_flips']} crosses {summary['n_crosses']} "
        f"live {live['discounted']:.3f}",
        flush=True,
    )
    return {"summary": summary, "live": live}


if __name__ == "__main__":
    run()
