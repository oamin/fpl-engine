"""Gameweeks 1–5 with the opening horizon and the zero fill.

Starts from ojaminFC's Gameweek 1 fifteen. Does not rewrite the leaked
327 record. Five weeks are a diagnosis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import GWS, _opening_state, build_frames
from src.live.entry import load_entry
from src.models.season_climb_ft import run_ft_season

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
ENTRY_ID = 2632584
SANGARE = "2026-27:565"


def _captain(step: dict[str, Any], roster: pd.DataFrame) -> str:
    gw = int(step["gw"])
    pid = str(step["captain_id"])
    sheet = roster.loc[roster["gw"] == gw]
    names = dict(zip(sheet["player_id"].astype(str), sheet["player_name"], strict=False))
    return names.get(pid, pid)


def _sold(trace: list[dict[str, Any]], pid: str) -> int | None:
    owned: set[str] | None = None
    for step in trace:
        squad = set(step["squad"]["player_id"].astype(str))
        if owned is not None and pid in owned and pid not in squad:
            return int(step["gw"])
        owned = squad
    return None


def _arm(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    opening: Any,
    **kwargs: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    weekly = run_ft_season(
        feat,
        {"xp": "score_xp"},
        list(GWS),
        roster=roster,
        trace=trace,
        opening=opening,
        **kwargs,
    )
    return weekly, trace


def run() -> dict[str, Any]:
    feat, roster, _info = build_frames()
    entry = load_entry(ENTRY_ID)
    opening = _opening_state(entry, roster)
    theirs = {int(g["gw"]): float(g["points"]) for g in entry["gameweeks"]}
    arms = {
        "freeze": dict(freeze_horizon=True, use_early_scores=False),
        "horizon": dict(use_early_scores=False),
        "horizon_early": dict(),
    }
    ran: dict[str, tuple[pd.DataFrame, list[dict[str, Any]]]] = {}
    for name, kwargs in arms.items():
        ran[name] = _arm(feat, roster, opening, **kwargs)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (weekly, trace) in ran.items():
        by_gw = weekly.set_index("gw")
        for gw in GWS:
            step = next(s for s in trace if int(s["gw"]) == gw)
            rows.append(
                {
                    "arm": name,
                    "gw": gw,
                    "points": float(by_gw.loc[gw, "xi_points_cap"]),
                    "hits": float(by_gw.loc[gw, "hit_cost"]),
                    "captain": _captain(step, roster),
                    "theirs": theirs[gw],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(PROCESSED / "stage_42_gw15_horizon.csv", index=False)
    summary = {
        name: {
            "points": float(weekly["xi_points_cap"].sum()),
            "hits": float(weekly["hit_cost"].sum()),
            "sangare_sold": _sold(trace, SANGARE),
        }
        for name, (weekly, trace) in ran.items()
    }
    _write(table, summary, sum(theirs.values()))
    return summary


def _write(table: pd.DataFrame, summary: dict[str, dict[str, Any]], theirs: float) -> None:
    lines = [
        "# Gameweeks 1–5, opening horizon",
        "",
        "The same opening fifteen as the 336 path. The freeze arm carries this week's score forward. The horizon arms price later weeks from that fixture's opening line, and a double is still one fixture. The early arm also gives an owned player with one or two prior appearances his past-only score, capped at 6. A missing score with no such row stays 0.",
        "",
        f"ojaminFC **{theirs:.0f}**.",
        "",
        "| arm | points | hits | Sangaré sold |",
        "|---|---:|---:|---|",
    ]
    for name, row in summary.items():
        sold = "held" if row["sangare_sold"] is None else f"GW{row['sangare_sold']}"
        lines.append(
            f"| {name} | {row['points']:.0f} | {row['hits']:.0f} | {sold} |"
        )
    lines += [
        "",
        "| GW | Freeze | Horizon | Horizon + early | Theirs | Freeze captain | Horizon captain | Early captain |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    freeze = table.loc[table["arm"] == "freeze"].set_index("gw")
    horizon = table.loc[table["arm"] == "horizon"].set_index("gw")
    early = table.loc[table["arm"] == "horizon_early"].set_index("gw")
    for gw in GWS:
        lines.append(
            f"| {gw} | {freeze.loc[gw, 'points']:.0f} | {horizon.loc[gw, 'points']:.0f} | "
            f"{early.loc[gw, 'points']:.0f} | {freeze.loc[gw, 'theirs']:.0f} | "
            f"{freeze.loc[gw, 'captain']} | {horizon.loc[gw, 'captain']} | {early.loc[gw, 'captain']} |"
        )
    lines.append("")
    (REPORTS / "stage_42_gw15_horizon.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    print(run())
