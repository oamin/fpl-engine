"""Split the unmatched shared starters into formation and displacement.

Formation is an extra player at that position in the eleven named before
the deadline. Displacement is a shared starter left over when that extra
count does not cover him: the other side used the place for someone who
is not in both fifteens. Realised points do not choose the label. The two
labels sum to the shape tag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import build_frames
from src.models.cohort_carry import carry_entry, cohort_specs
from src.models.lineup_cause import tag_week
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import PROCESSED, REPORTS, _fmt, merged_clubs

POSITIONS = ("GKP", "DEF", "MID", "FWD")
REASONS = ("formation", "displacement")


def label_shape(
    shape_rows: list[dict[str, Any]],
    model_counts: dict[str, int],
    human_counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Label each unmatched shared starter. The order is score, then id."""
    labelled: list[dict[str, Any]] = []
    for side in ("model", "human"):
        own = model_counts if side == "model" else human_counts
        other = human_counts if side == "model" else model_counts
        for position in POSITIONS:
            group = [
                row
                for row in shape_rows
                if row["side"] == side and row["position"] == position
            ]
            group.sort(key=lambda row: (-float(row["score_xp"]), str(row["id"])))
            extra = max(0, int(own[position]) - int(other[position]))
            for index, row in enumerate(group):
                reason = "formation" if index < extra else "displacement"
                labelled.append({**row, "reason": reason})
    return labelled


def _shape_rows(week: dict[str, Any]) -> list[dict[str, Any]]:
    tagged = tag_week(week["lineup_model_detail"], week["lineup_their_detail"])
    return [row for row in tagged if row["tag"] == "shape"]


def _sum_reason(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = {reason: 0.0 for reason in REASONS}
    for row in rows:
        totals[str(row["reason"])] += float(row["points"])
    return totals


def _line(title: str, rows: list[dict[str, Any]]) -> str:
    totals = _sum_reason(rows)
    counts = {reason: sum(1 for row in rows if row["reason"] == reason) for reason in REASONS}
    return (
        f"{title}: formation {_fmt(totals['formation'])} from {counts['formation']} players, "
        f"displacement {_fmt(totals['displacement'])} from {counts['displacement']} players."
    )


def _by_position(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for position in POSITIONS:
        block = [row for row in rows if row["position"] == position]
        if not block:
            continue
        lines.append(_line(position, block))
    return lines


def run() -> dict[str, Any]:
    """Label the shape tag on the locked cohort. Does not retune."""
    specs = cohort_specs()
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    records: list[dict[str, Any]] = []
    for spec in specs:
        print(f"shape {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        if not result["finished"]:
            raise RuntimeError(f"{spec['label']} did not finish: {result['error']}")
        for week in result["weeks"]:
            labelled = label_shape(
                _shape_rows(week),
                week["model_intended_counts"],
                week["their_intended_counts"],
            )
            shape_points = sum(float(row["points"]) for row in labelled)
            tagged = tag_week(week["lineup_model_detail"], week["lineup_their_detail"])
            expected = sum(float(row["points"]) for row in tagged if row["tag"] == "shape")
            if abs(shape_points - expected) > 1e-6:
                raise RuntimeError(
                    f"{spec['label']} GW{int(week['gw'])} labels {shape_points} "
                    f"do not equal shape {expected}"
                )
            for row in labelled:
                records.append(
                    {
                        "entry_id": result["entry_id"],
                        "label": result["label"],
                        "group": result["group"],
                        "gw": int(week["gw"]),
                        "reason": row["reason"],
                        "position": row["position"],
                        "side": row["side"],
                        "points": float(row["points"]),
                        "score_xp": float(row["score_xp"]),
                        "model_def": int(week["model_intended_counts"]["DEF"]),
                        "model_mid": int(week["model_intended_counts"]["MID"]),
                        "model_fwd": int(week["model_intended_counts"]["FWD"]),
                        "their_def": int(week["their_intended_counts"]["DEF"]),
                        "their_mid": int(week["their_intended_counts"]["MID"]),
                        "their_fwd": int(week["their_intended_counts"]["FWD"]),
                    }
                )
    cohort = [row for row in records if row["group"] != "reference"]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(PROCESSED / "shape_split_gw15.csv", index=False)
    _write_report(REPORTS / "shape_split_gw15.md", records)
    totals = _sum_reason(cohort)
    return {"formation": totals["formation"], "displacement": totals["displacement"], "players": len(cohort)}


def _write_report(path: Path, records: list[dict[str, Any]]) -> None:
    cohort = [row for row in records if row["group"] != "reference"]
    lines = [
        "# Formation or a player the other side does not own",
        "",
        "These are the unmatched shared starters from the lineup tags, the shape figure of −196. A player both sides own is in one starting eleven only.",
        "",
        "Formation means that side named more players at that position before the deadline. A 5-defender eleven against a 3-defender eleven has two extra defender places. Displacement means the counts at that position do not cover the extra shared starter. The other side used a place at that position for someone who is not in both fifteens, so the shared player stayed unmatched.",
        "",
        "The label uses the eleven named before the deadline, including a starter who later played 0 minutes. It does not use the eleven after automatic substitutes. Where several shared starters are unmatched, the higher score is labelled first. The points scored do not choose the label. The two labels sum to the shape figure.",
        "",
        "The 14 are the cohort locked on 4 October. ojaminFC is apart from that total. Entry 1078627 is not in this run. The split is not a new score.",
        "",
        _line("The 14", cohort),
        _line("Veterans", [row for row in cohort if row["group"] == "veteran"]),
        _line("Rank slots", [row for row in cohort if row["group"] == "rank"]),
        _line("Gameweeks 1–3", [row for row in cohort if int(row["gw"]) <= 3]),
        _line("Gameweeks 4–5", [row for row in cohort if int(row["gw"]) >= 4]),
        _line("ojaminFC", [row for row in records if row["group"] == "reference"]),
        "",
        "By position, for the 14:",
        "",
        *_by_position(cohort),
        "",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
