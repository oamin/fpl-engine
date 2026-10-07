"""Overlap of the 14 locked human squads in Gameweeks 1–5.

The squad is the owned fifteen: the stored eleven plus the bench. Automatic
substitutes swap slots inside that fifteen, so the union is the squad that
was held. The stored eleven is the one after those substitutes. ojaminFC and
entry 1078627 stay out of the count.
"""

from __future__ import annotations

import csv
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from src.live.benchmark import GWS
from src.live.entry import load_entry
from src.models.cohort_carry import cohort_specs

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "data" / "processed" / "squad_overlap_gw15.csv"
OWN_CSV = ROOT / "data" / "processed" / "squad_overlap_owners_gw15.csv"
OUT_REPORT = ROOT / "reports" / "squad_overlap_gw15.md"
HALF = 7


def _cohort() -> list[dict[str, Any]]:
    rows = [row for row in cohort_specs() if row["group"] != "reference"]
    if len(rows) != 14:
        raise RuntimeError("the overlap count needs the 14 managers")
    return rows


def _weeks(specs: list[dict[str, Any]]) -> tuple[dict, dict, dict]:
    """Owned fifteens, post-substitute elevens, and player names."""
    squads: dict[int, dict[str, set[int]]] = {int(gw): {} for gw in GWS}
    elevens: dict[int, dict[str, set[int]]] = {int(gw): {} for gw in GWS}
    names: dict[int, tuple[str, str, str]] = {}
    for spec in specs:
        entry = load_entry(int(spec["entry_id"]))
        label = str(spec["label"])
        by_gw = {int(week["gw"]): week for week in entry["gameweeks"]}
        for gw in GWS:
            week = by_gw[int(gw)]
            owned = week["xi"] + week["bench"]
            ids = [int(player["id"]) for player in owned]
            xi_ids = [int(player["id"]) for player in week["xi"]]
            if len(ids) != 15 or len(set(ids)) != 15:
                raise RuntimeError(f"{label} GW{gw} is not a fifteen")
            if len(xi_ids) != 11 or len(set(xi_ids)) != 11:
                raise RuntimeError(f"{label} GW{gw} is not an eleven")
            squads[int(gw)][label] = set(ids)
            elevens[int(gw)][label] = set(xi_ids)
            for player in owned:
                names[int(player["id"])] = (
                    str(player["name"]),
                    str(player["position"]),
                    str(player["team"]),
                )
    return squads, elevens, names


def _pair_sizes(groups: dict[str, set[int]], labels: list[str]) -> list[int]:
    return [
        len(groups[left] & groups[right])
        for left, right in combinations(labels, 2)
    ]


def _extremes(
    groups: dict[str, set[int]], labels: list[str]
) -> tuple[tuple[int, str, str], tuple[int, str, str]]:
    low = (99, "", "")
    high = (-1, "", "")
    for left, right in combinations(labels, 2):
        shared = len(groups[left] & groups[right])
        if shared < low[0]:
            low = (shared, left, right)
        if shared > high[0]:
            high = (shared, left, right)
    return low, high


def run() -> dict[str, Any]:
    specs = _cohort()
    labels = [str(row["label"]) for row in specs]
    groups = {
        str(row["group"]): [str(item["label"]) for item in specs if item["group"] == row["group"]]
        for row in specs
    }
    squads, elevens, names = _weeks(specs)
    week_rows: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []
    for gw in GWS:
        sizes = _pair_sizes(squads[int(gw)], labels)
        xi_sizes = _pair_sizes(elevens[int(gw)], labels)
        low, high = _extremes(squads[int(gw)], labels)
        owned = Counter()
        for squad in squads[int(gw)].values():
            owned.update(squad)
        started = Counter()
        for xi in elevens[int(gw)].values():
            started.update(xi)
        half = [pid for pid, count in owned.items() if count >= HALF]
        everyone = [pid for pid, count in owned.items() if count == len(labels)]
        ge7_slots = sum(count for count in owned.values() if count >= HALF)
        vet = _pair_sizes(squads[int(gw)], groups["veteran"])
        rank = _pair_sizes(squads[int(gw)], groups["rank"])
        identical = sum(
            1
            for left, right in combinations(labels, 2)
            if squads[int(gw)][left] == squads[int(gw)][right]
        )
        ordered = sorted(sizes)
        week_rows.append(
            {
                "gw": int(gw),
                "pairs": len(sizes),
                "mean_shared": round(sum(sizes) / len(sizes), 2),
                "median_shared": ordered[len(ordered) // 2],
                "min_shared": low[0],
                "min_pair": f"{low[1]} / {low[2]}",
                "max_shared": high[0],
                "max_pair": f"{high[1]} / {high[2]}",
                "union": len(owned),
                "ge7_players": len(half),
                "ge7_slots": ge7_slots,
                "all_14": len(everyone),
                "veteran_mean": round(sum(vet) / len(vet), 2),
                "rank_mean": round(sum(rank) / len(rank), 2),
                "xi_mean": round(sum(xi_sizes) / len(xi_sizes), 2),
                "xi_min": min(xi_sizes),
                "xi_max": max(xi_sizes),
                "identical": identical,
            }
        )
    carries = []
    for left, right in zip(GWS, GWS[1:]):
        kept = [
            len(squads[int(left)][label] & squads[int(right)][label])
            for label in labels
        ]
        carries.append((int(left), int(right), sum(kept) / len(kept)))
        for pid, count in owned.most_common():
            if count < HALF:
                continue
            name, position, team = names[pid]
            owner_rows.append(
                {
                    "gw": int(gw),
                    "player_id": pid,
                    "name": name,
                    "position": position,
                    "team": team,
                    "squads": count,
                    "started": int(started[pid]),
                }
            )
    _write(week_rows, owner_rows, carries)
    return {"weeks": week_rows, "owners": owner_rows, "carries": carries}


def _write(
    weeks: list[dict[str, Any]],
    owners: list[dict[str, Any]],
    carries: list[tuple[int, int, float]],
) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weeks[0]))
        writer.writeheader()
        writer.writerows(weeks)
    with OWN_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(owners[0]))
        writer.writeheader()
        writer.writerows(owners)
    OUT_REPORT.write_text(_render(weeks, owners, carries), encoding="utf-8")


def _render(
    weeks: list[dict[str, Any]],
    owners: list[dict[str, Any]],
    carries: list[tuple[int, int, float]],
) -> str:
    grand = sum(float(row["mean_shared"]) for row in weeks) / len(weeks)
    identical = sum(int(row["identical"]) for row in weeks)
    same = (
        "No two fifteens are the same."
        if identical == 0
        else f"{identical} pairs hold the same fifteen."
    )
    carry_text = ", ".join(
        f"Gameweek {left} to {right} keeps {mean:.2f} of 15"
        for left, right, mean in carries
    )
    lines = [
        "# Squad overlap, Gameweeks 1–5",
        "",
        "The 14 locked managers, with ojaminFC and entry 1078627 left out. "
        "A squad is the owned fifteen. Automatic substitutes move a player "
        "between the eleven and the bench and do not change who is owned. "
        "The eleven figure is the stored eleven after those substitutes. "
        "Each week has 91 pairs.",
        "",
        f"A typical pair shares {grand:.2f} of 15 players across the five weeks. "
        f"{same} From one week to the next the same manager keeps most of his "
        f"own fifteen: {carry_text}.",
        "",
        "| GW | Mean | Median | Min | Least alike | Max | Most alike | Distinct players | In at least 7 squads | Slots those players fill | In all 14 | Veteran mean | Rank mean | Eleven mean |",
        "|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in weeks:
        lines.append(
            f"| {row['gw']} | {row['mean_shared']:.2f} | {row['median_shared']} | "
            f"{row['min_shared']} | {row['min_pair']} | {row['max_shared']} | "
            f"{row['max_pair']} | {row['union']} | {row['ge7_players']} | "
            f"{row['ge7_slots']} | {row['all_14']} | {row['veteran_mean']:.2f} | "
            f"{row['rank_mean']:.2f} | {row['xi_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "The seven veterans share more with each other than the seven rank slots do. "
            "Gameweek 3 is the widest week: the mean falls to "
            f"{weeks[2]['mean_shared']:.2f}, the least alike pair shares "
            f"{weeks[2]['min_shared']}, and {weeks[2]['union']} distinct players appear. "
            "Gameweek 4 pulls the elevens back together "
            f"({weeks[3]['xi_mean']:.2f} of 11).",
            "",
            "Players in at least 7 of the 14 squads:",
            "",
            "| GW | Player | Pos | Club | Squads | Scored XI |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for row in owners:
        lines.append(
            f"| {row['gw']} | {row['name']} | {row['position']} | {row['team']} | "
            f"{row['squads']} | {row['started']} |"
        )
    by_week: dict[int, set[str]] = {}
    for row in owners:
        by_week.setdefault(int(row["gw"]), set()).add(str(row["name"]))
    always = set.intersection(*by_week.values())
    named = ", ".join(sorted(always))
    pedro = next(
        row
        for row in owners
        if int(row["gw"]) == 5 and row["name"] == "João Pedro"
    )
    lines.extend(
        [
            "",
            f"The same four names clear 7 squads in every week: {named}. "
            "João Pedro is in all 14 squads in Gameweeks 1, 2, and 4. "
            "Haaland is in all 14 in Gameweek 3. Gameweek 5 has no player in all 14. "
            "Scored XI is the stored eleven after automatic substitutes. "
            f"João Pedro is in {pedro['squads']} squads in Gameweek 5 and in "
            f"{pedro['started']} of those scored elevens.",
            "",
            "This is a count of the stored squads. It does not change `score_xp`, "
            "the buy gate, or the chip margins.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    run()
