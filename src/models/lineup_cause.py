"""What the lineup gap is made of.

A lineup player is in both fifteens and in exactly one final eleven.
The tags below are assigned from the intended eleven and from score_xp.
Realised points are the size of the tag, not the choice of tag. The tags
sum to the lineup gap. Nothing here changes the score or the chip rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import build_frames
from src.models.cohort_carry import carry_entry, cohort_specs
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import PROCESSED, REPORTS, _fmt, merged_clubs

TAGS = ("autosub", "blank", "ranked_ahead", "ranked_behind", "tie", "shape")


def _single(player: dict[str, Any], side: str, tag: str) -> dict[str, Any]:
    sign = 1.0 if side == "model" else -1.0
    return {
        "tag": tag,
        "side": side,
        "id": str(player["id"]),
        "points": sign * float(player["points"]),
        "inversion": False,
        "position": str(player["position"]),
        "score_xp": float(player["score_xp"]),
        "model_xp": float(player["score_xp"]) if side == "model" else None,
        "human_xp": float(player["score_xp"]) if side == "human" else None,
        "model_points": float(player["points"]) if side == "model" else None,
        "human_points": float(player["points"]) if side == "human" else None,
    }


def tag_week(model: list[dict[str, Any]], human: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One tag per autosub, blank, pair, or unpaired shape slot."""
    records: list[dict[str, Any]] = []
    selected_model: list[dict[str, Any]] = []
    selected_human: list[dict[str, Any]] = []
    for player in model:
        if not player["intended"]:
            records.append(_single(player, "model", "autosub"))
        elif float(player["minutes"]) <= 0:
            records.append(_single(player, "model", "blank"))
        else:
            selected_model.append(player)
    for player in human:
        if not player["intended"]:
            records.append(_single(player, "human", "autosub"))
        elif float(player["minutes"]) <= 0:
            records.append(_single(player, "human", "blank"))
        else:
            selected_human.append(player)
    positions = sorted({str(player["position"]) for player in selected_model + selected_human})
    for position in positions:
        ours = sorted(
            (player for player in selected_model if player["position"] == position),
            key=lambda player: (-float(player["score_xp"]), str(player["id"])),
        )
        theirs = sorted(
            (player for player in selected_human if player["position"] == position),
            key=lambda player: (-float(player["score_xp"]), str(player["id"])),
        )
        paired = min(len(ours), len(theirs))
        for index in range(paired):
            model_xp = float(ours[index]["score_xp"])
            human_xp = float(theirs[index]["score_xp"])
            if model_xp > human_xp:
                tag = "ranked_ahead"
            elif model_xp < human_xp:
                tag = "ranked_behind"
            else:
                tag = "tie"
            model_points = float(ours[index]["points"])
            human_points = float(theirs[index]["points"])
            records.append(
                {
                    "tag": tag,
                    "points": model_points - human_points,
                    "inversion": tag == "ranked_ahead" and model_points < human_points,
                    "position": position,
                    "model_xp": model_xp,
                    "human_xp": human_xp,
                    "model_points": model_points,
                    "human_points": human_points,
                }
            )
        for player in ours[paired:]:
            records.append(_single(player, "model", "shape"))
        for player in theirs[paired:]:
            records.append(_single(player, "human", "shape"))
    return records


def _sum_tags(records: list[dict[str, Any]]) -> dict[str, float]:
    totals = {tag: 0.0 for tag in TAGS}
    for record in records:
        totals[str(record["tag"])] += float(record["points"])
    return totals


def _records_for(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not result["finished"]:
        return []
    flat: list[dict[str, Any]] = []
    for week in result["weeks"]:
        tagged = tag_week(week["lineup_model_detail"], week["lineup_their_detail"])
        total = sum(float(record["points"]) for record in tagged)
        if abs(total - float(week["lineup_gap"])) > 1e-6:
            raise RuntimeError(
                f"{result['label']} GW{int(week['gw'])} tags {total} "
                f"do not equal the lineup gap {week['lineup_gap']}"
            )
        for record in tagged:
            flat.append(
                {
                    "entry_id": result["entry_id"],
                    "label": result["label"],
                    "group": result["group"],
                    "gw": int(week["gw"]),
                    "residual": float(week["residual"]),
                    **record,
                }
            )
    return flat


def _slice_line(title: str, records: list[dict[str, Any]]) -> str:
    totals = _sum_tags(records)
    ahead = [row for row in records if row["tag"] == "ranked_ahead"]
    lost = sum(float(row["points"]) for row in ahead if row["inversion"])
    kept = sum(float(row["points"]) for row in ahead if not row["inversion"])
    pieces = ", ".join(f"{tag} {_fmt(totals[tag])}" for tag in TAGS)
    return (
        f"{title}: {pieces}. "
        f"Of the ranked-ahead points, {_fmt(lost)} are weeks the higher score scored fewer, "
        f"and {_fmt(kept)} are weeks it scored at least as many."
    )


def run() -> dict[str, Any]:
    """Tag the lineup gap on the locked cohort. Does not retune."""
    specs = cohort_specs()
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    results = []
    records: list[dict[str, Any]] = []
    for spec in specs:
        print(f"lineup {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        results.append(result)
        records.extend(_records_for(result))
    cohort = [row for row in records if row["group"] != "reference"]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(PROCESSED / "lineup_cause_gw15.csv", index=False)
    _write_report(REPORTS / "lineup_cause_gw15.md", records, results)
    totals = _sum_tags(cohort)
    return {"tags": totals, "rows": len(cohort)}


def _write_report(
    path: Path, records: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    cohort = [row for row in records if row["group"] != "reference"]
    footnotes = []
    for result in results:
        for week in result["weeks"]:
            if abs(float(week["residual"])) >= 1e-6:
                footnotes.append(
                    f"{result['label']} Gameweek {int(week['gw'])} misses the official total "
                    f"by {_fmt(week['residual'])}. Its lineup points stay in the tags."
                )
    lines = [
        "# What the lineup gap is made of",
        "",
        "A lineup player is in both fifteens and in exactly one final eleven. A player in only one fifteen stays in the transfer column and is not tagged here. The tag comes from the eleven named before the deadline and from that week's score. The points are what that player scored. The tags sum to the lineup gap.",
        "",
        "Autosub is someone who entered the final eleven because a starter played 0 minutes. Blank is an intended starter who played 0 minutes and was not replaced. The rest are paired within position, highest score with highest score. Ranked ahead means the model starter had the higher score. Ranked behind means the human starter had the higher score. A tie is an equal score. Shape is the starters left over when the two elevens do not have the same number at that position. Realised points do not choose the tag.",
        "",
        "This is the same carry as the locked cohort. ojaminFC is shown apart from the 14. Entry 1078627 is not in this run. The shares are not a new score.",
        "",
        _slice_line("The 14", cohort),
        _slice_line("Veterans", [row for row in cohort if row["group"] == "veteran"]),
        _slice_line("Rank slots", [row for row in cohort if row["group"] == "rank"]),
        _slice_line("Gameweeks 1–3", [row for row in cohort if int(row["gw"]) <= 3]),
        _slice_line("Gameweeks 4–5", [row for row in cohort if int(row["gw"]) >= 4]),
        _slice_line(
            "ojaminFC",
            [row for row in records if row["group"] == "reference"],
        ),
        "",
    ]
    if footnotes:
        lines.extend(footnotes)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
