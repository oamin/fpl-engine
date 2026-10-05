"""Where the transfer gap comes from.

A transfer player is in one final eleven and not in both fifteens. Chip
squad is a player the human acquired with a wildcard or a free hit this
week. Ranked lower is a same-position pair where his player had the higher
score. Everyone else is an unmatched squad player. Realised points do not
choose the tag. The tags sum to the transfer gap.

The bar was locked before these totals were read. Chip-squad points more
negative than -90 mean the squad gap on this window is the chip reset.
Ranked-lower points more negative than -90 mean the search kept the lower
score. Anything else does not isolate one lever. No rule changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import build_frames
from src.models.cohort_carry import carry_entry, cohort_specs
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import PROCESSED, REPORTS, _fmt, merged_clubs

TAGS = ("chip_squad", "ranked_lower", "unmatched_squad")
CHIP_CUT = -90.0
RANK_CUT = -90.0
CHIP_SQUADS = {"wildcard", "free_hit"}


def _signed(player: dict[str, Any], side: str) -> float:
    sign = 1.0 if side == "model" else -1.0
    return sign * float(player["points"])


def label_transfers(
    model: list[dict[str, Any]],
    human: list[dict[str, Any]],
    human_pre: set[str],
    chip: str | None,
) -> list[dict[str, Any]]:
    """Tag the transfer players. Chip acquisitions come out before any pair."""
    records: list[dict[str, Any]] = []
    remaining_model = list(model)
    remaining_human: list[dict[str, Any]] = []
    for player in human:
        acquired = chip in CHIP_SQUADS and player["id"] not in human_pre
        if acquired:
            records.append({**player, "side": "human", "tag": "chip_squad", "points": _signed(player, "human")})
        else:
            remaining_human.append(player)
    positions = sorted(
        {
            str(player["position"])
            for player in remaining_model + remaining_human
            if player.get("position") and player.get("score_xp") is not None
        }
    )
    paired_model: set[str] = set()
    paired_human: set[str] = set()
    for position in positions:
        ours = sorted(
            (
                player
                for player in remaining_model
                if player.get("position") == position and player.get("score_xp") is not None
            ),
            key=lambda player: (-float(player["score_xp"]), str(player["id"])),
        )
        theirs = sorted(
            (
                player
                for player in remaining_human
                if player.get("position") == position and player.get("score_xp") is not None
            ),
            key=lambda player: (-float(player["score_xp"]), str(player["id"])),
        )
        for ours_player, their_player in zip(ours, theirs, strict=False):
            paired_model.add(ours_player["id"])
            paired_human.add(their_player["id"])
            points = float(ours_player["points"]) - float(their_player["points"])
            higher = float(their_player["score_xp"]) > float(ours_player["score_xp"])
            records.append(
                {
                    "id": f"{ours_player['id']}|{their_player['id']}",
                    "side": "pair",
                    "position": position,
                    "score_xp": None,
                    "points": points,
                    "tag": "ranked_lower" if higher else "unmatched_squad",
                    "model_xp": float(ours_player["score_xp"]),
                    "human_xp": float(their_player["score_xp"]),
                }
            )
    for player in remaining_model:
        if player["id"] in paired_model:
            continue
        records.append({**player, "side": "model", "tag": "unmatched_squad", "points": _signed(player, "model")})
    for player in remaining_human:
        if player["id"] in paired_human:
            continue
        records.append({**player, "side": "human", "tag": "unmatched_squad", "points": _signed(player, "human")})
    return records


def _totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = {tag: 0.0 for tag in TAGS}
    for row in rows:
        totals[str(row["tag"])] += float(row["points"])
    return totals


def call_gap(chip_points: float, ranked_points: float) -> str:
    """The locked reading. More negative than -90 isolates a lever."""
    if chip_points < CHIP_CUT:
        return "chip reset"
    if ranked_points < RANK_CUT:
        return "ranked lower"
    return "no single lever"


def _line(title: str, rows: list[dict[str, Any]]) -> str:
    totals = _totals(rows)
    counts = {tag: sum(1 for row in rows if row["tag"] == tag) for tag in TAGS}
    return (
        f"{title}: chip squad {_fmt(totals['chip_squad'])} from {counts['chip_squad']} players, "
        f"ranked lower {_fmt(totals['ranked_lower'])} from {counts['ranked_lower']} pairs, "
        f"unmatched squad {_fmt(totals['unmatched_squad'])} from {counts['unmatched_squad']} rows."
    )


def run() -> dict[str, Any]:
    """Tag the transfer gap on the locked cohort. Does not retune."""
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
        print(f"squad {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        if not result["finished"]:
            raise RuntimeError(f"{spec['label']} did not finish: {result['error']}")
        for week in result["weeks"]:
            labelled = label_transfers(
                week["transfer_model_detail"],
                week["transfer_their_detail"],
                set(week["human_pre_ids"]),
                week["his_chip"],
            )
            total = sum(float(row["points"]) for row in labelled)
            if abs(total - float(week["transfer_gap"])) > 1e-6:
                raise RuntimeError(
                    f"{spec['label']} GW{int(week['gw'])} tags {total} "
                    f"do not equal the transfer gap {week['transfer_gap']}"
                )
            for row in labelled:
                records.append(
                    {
                        "entry_id": result["entry_id"],
                        "label": result["label"],
                        "group": result["group"],
                        "gw": int(week["gw"]),
                        "tag": row["tag"],
                        "points": float(row["points"]),
                        "position": row.get("position") or "",
                        "side": row.get("side") or "",
                    }
                )
    cohort = [row for row in records if row["group"] != "reference"]
    totals = _totals(cohort)
    reading = call_gap(totals["chip_squad"], totals["ranked_lower"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(PROCESSED / "squad_gap_gw15.csv", index=False)
    _write_report(REPORTS / "squad_gap_gw15.md", records, reading)
    return {"reading": reading, **totals}


def _write_report(path: Path, records: list[dict[str, Any]], reading: str) -> None:
    cohort = [row for row in records if row["group"] != "reference"]
    if reading == "chip reset":
        verdict = (
            "Chip-squad points are more negative than -90. On this window the squad gap "
            "is the chip reset. No rule changes."
        )
    elif reading == "ranked lower":
        verdict = (
            "Ranked-lower points are more negative than -90. On this window the search "
            "kept the lower score. No rule changes."
        )
    else:
        verdict = "This window does not isolate a single lever. No rule changes."
    lines = [
        "# Where the players only one side owns come from",
        "",
        "A transfer player is in one final eleven and not in both fifteens. Chip squad means the human played a wildcard or a free hit this week and the player was not in the fifteen he held before that deadline. Ranked lower means a same-position pair where his player had the higher score. Everyone else stays an unmatched squad player. The points scored do not choose the tag. The tags sum to the transfer gap of −180.",
        "",
        "The bar was locked before these totals were read. Chip-squad points more negative than −90 mean the squad gap is the chip reset. Ranked-lower points more negative than −90 mean the search kept the lower score. Anything else does not isolate one lever. The formation figure of −163 is not treated as points a different shape would have scored. The score and the formation list stay as they are.",
        "",
        verdict,
        "",
        _line("The 14", cohort),
        _line("Veterans", [row for row in cohort if row["group"] == "veteran"]),
        _line("Rank slots", [row for row in cohort if row["group"] == "rank"]),
        _line("Gameweeks 1–3", [row for row in cohort if int(row["gw"]) <= 3]),
        _line("Gameweeks 4–5", [row for row in cohort if int(row["gw"]) >= 4]),
        _line("ojaminFC", [row for row in records if row["group"] == "reference"]),
        "",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(run())
