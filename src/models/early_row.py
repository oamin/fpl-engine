"""Reading D. The nine pool chip weeks, scored with the capped early number.

The buy gate stays at three appearances. A 0-minute player with no row
still blocks the week. Mark Brookes in Gameweek 5 is not reopened.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.live.benchmark import build_frames
from src.live.entry import load_entry
from src.models.cohort_carry import carry_entry, cohort_specs
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import PROCESSED, REPORTS, _early, merged_clubs
from src.models.squad_frontier import (
    CHIPS,
    _fmt_pounds,
    _fmt_share,
    _fmt_xp,
    _meaning,
    score_week,
    summarise,
)
from src.models.squad_gap import CHIP_SQUADS

WEEKS_CSV = PROCESSED / "early_row_gw15.csv"
REPORT = REPORTS / "early_row.md"
FRONTIER_CSV = PROCESSED / "squad_frontier_gw15.csv"
BROOKES_GW5 = (616, 5, "wildcard")


def pool_keys(frame: pd.DataFrame) -> set[tuple[int, int, str]]:
    """The nine pool weeks. The Gameweek 5 money week stays out."""
    part = frame.loc[frame["status"].astype(str) == "pool"].copy()
    if len(part) != 9:
        raise RuntimeError(f"the pool is {len(part)} weeks, not 9")
    keys: set[tuple[int, int, str]] = set()
    for row in part.itertuples(index=False):
        key = (int(row.entry_id), int(row.gw), str(row.chip))
        if key == BROOKES_GW5:
            raise RuntimeError("Mark Brookes in Gameweek 5 is not in this reading")
        keys.add(key)
    if len(keys) != 9:
        raise RuntimeError("a pool week is duplicated")
    return keys


def write_report(path: Any, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Early row",
        "",
        "Counted after the lock in `reports/early_row_plan.md`. "
        "A player with one or two prior appearances is scored with that early number, capped at 6, "
        "including when the model does not own him. "
        "The number evaluates the human fifteen. It does not enter the buy pool. "
        "A player with 0 minutes and no score row stays unscored, and that week stays out of the call. "
        "Mark Brookes in Gameweek 5 is the money week already counted, and it is not in this table. "
        "The chip sum is still the undiscounted `xi_xp`. This reading does not add a decay.",
        "",
        (
            "A later priced week repeats the deadline score when the shot share is missing. "
            "That is the existing step for a player with no share."
        ),
        "",
        f"Pool weeks: {summary['n']}. The reference manager is absent.",
        "",
        "## Reachability",
        "",
        "The bars are unchanged. Near is a per-week gap of at most 1.0. Far is above 4.0. The call needs half the reachable weeks of that chip.",
        "",
        "| Chip | Weeks | Reachable | Money | Pool | Rules | Mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for chip in CHIPS:
        block = summary["chips"][chip]
        lines.append(
            f"| {chip} | {block['n']} | {block['n_reachable']} | {block['n_money']} | "
            f"{block['n_pool']} | {block['n_rules']} | {block['n_mismatch']} |"
        )
    lines.extend(
        [
            "",
            "## The call",
            "",
            "| Chip | Reachable | Near | Middle | Far | Near share | Far share | Mean gap | Call |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for chip in CHIPS:
        block = summary["chips"][chip]
        lines.append(
            f"| {chip} | {block['n_reachable']} | {block['n_near']} | {block['n_middle']} | {block['n_far']} | "
            f"{_fmt_share(block['n_near'], block['n_reachable'])} | "
            f"{_fmt_share(block['n_far'], block['n_reachable'])} | "
            f"{_fmt_xp(block['mean'])} | {block['call']} |"
        )
    lines.append("")
    for chip in CHIPS:
        lines.append(f"{chip}: {_meaning(summary['chips'][chip]['call'])}")
        lines.append("")
    held_out = [
        row
        for row in rows
        if row["gap_per"] is not None and row["gap_per"] == row["gap_per"] and row["status"] != "reachable"
    ]
    if held_out:
        lines.append("A gap on a week that is not reachable is printed here and stays out of the call.")
        lines.append("")
        for row in held_out:
            lines.append(
                f"{row['label']}, Gameweek {int(row['gw'])} {row['chip']}: "
                f"per-week gap {_fmt_xp(row['gap_per'])}, status {row['status']}, "
                f"shortfall {_fmt_pounds(row['shortfall'])}."
            )
            lines.append("")
    lines.extend(
        [
            "## Weeks",
            "",
            "Early players are the ones scored from the capped table. Blank tags are the 0-minute slots. Points are descriptive.",
            "",
            "| Manager | GW | Chip | Status | Gap | Early players | Blanks | Human points | Model points |",
            "|---|---:|---|---|---:|---|---|---:|---:|",
        ]
    )
    ordered = sorted(rows, key=lambda row: (str(row["chip"]), str(row["label"]), int(row["gw"])))
    for row in ordered:
        lines.append(
            f"| {row['label']} | {int(row['gw'])} | {row['chip']} | {row['status']} | "
            f"{_fmt_xp(row['gap_per'])} | {row['early_players']} | {row['blank_tags']} | "
            f"{_fmt_xp(row['points_human'])} | {_fmt_xp(row['points_model'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    """Score the nine locked pool weeks. Does not retune, and does not reopen the money week."""
    if not FRONTIER_CSV.exists():
        raise RuntimeError("the squad frontier file is missing")
    keys = pool_keys(pd.read_csv(FRONTIER_CSV))
    wanted = {entry for entry, _gw, _chip in keys}
    specs = [
        row
        for row in cohort_specs()
        if row["group"] != "reference" and int(row["entry_id"]) in wanted
    ]
    if {int(row["entry_id"]) for row in specs} != wanted:
        raise RuntimeError("a pool manager is missing from the cohort")
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")}
    early = _early(feat)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"early-row {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        if not result["finished"]:
            raise RuntimeError(f"{spec['label']} did not finish: {result['error']}")
        entry = load_entry(int(spec["entry_id"]))
        for week in result["weeks"]:
            if week["his_chip"] not in CHIP_SQUADS:
                continue
            key = (int(spec["entry_id"]), int(week["gw"]), str(week["his_chip"]))
            if key not in keys:
                continue
            rows.append(
                score_week(
                    week,
                    entry,
                    spec,
                    feat,
                    roster,
                    clubs,
                    horizon_scores,
                    early,
                    use_early=True,
                )
            )
    if len(rows) != 9:
        raise RuntimeError(f"the reading scored {len(rows)} weeks, not 9")
    if any((int(row["entry_id"]), int(row["gw"]), str(row["chip"])) == BROOKES_GW5 for row in rows):
        raise RuntimeError("Mark Brookes in Gameweek 5 was reopened")
    summary = summarise(rows)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(WEEKS_CSV, index=False)
    write_report(REPORT, rows, summary)
    for chip in CHIPS:
        block = summary["chips"][chip]
        print(
            f"{chip} call {block['call']} reachable {block['n_reachable']} mean {block['mean']}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    run()
