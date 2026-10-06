"""The squad the published climber picks for itself across Gameweeks 1–5.

No opening fifteen is passed in. Gameweek 1 is solved from the buy pool.
Later weeks are free transfers on ``score_xp``, with the opening horizon and
an empty chip map. The week total is the scoring eleven, captain doubled,
hits removed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import GWS, build_frames
from src.models.season_climb_ft import run_ft_season

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "data" / "processed" / "own_squad_gw15.csv"
OUT_REPORT = ROOT / "reports" / "own_squad_gw15.md"
_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def run() -> dict[str, Any]:
    feat, roster, _info = build_frames()
    trace: list[dict[str, Any]] = []
    weekly = run_ft_season(feat, {"xp": "score_xp"}, list(GWS), roster=roster, trace=trace)
    if set(weekly["gw"].astype(int)) != set(GWS):
        raise RuntimeError("the own-squad climb did not finish five weeks")
    rows = _rows(trace)
    weeks = weekly.to_dict("records")
    _write(rows, weeks)
    return {"rows": rows, "weeks": weeks}


def _rows(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: set[str] = set()
    for step in trace:
        gw = int(step["gw"])
        named = set(step["xi"]["player_id"].astype(str))
        final = set(step["final_xi"]["player_id"].astype(str))
        captain = str(step["captain_id"])
        vice = str(step["vice_id"])
        owned = {str(row.player_id) for row in step["squad"].itertuples(index=False)}
        arrived = sorted(owned - previous) if previous else []
        left = sorted(previous - owned) if previous else []
        for row in step["squad"].itertuples(index=False):
            pid = str(row.player_id)
            in_xi = pid in named
            scored = pid in final
            rows.append(
                {
                    "gw": gw,
                    "player_id": pid,
                    "name": str(row.player_name),
                    "position": str(row.position),
                    "team": str(row.team),
                    "lineup": "XI" if in_xi else "bench",
                    "scored": int(scored),
                    "captain": int(pid == captain),
                    "vice": int(pid == vice),
                    "xp": round(float(row.score_xp), 2),
                    "points": float(row.total_points),
                    "minutes": float(row.minutes),
                    "in": ",".join(arrived),
                    "out": ",".join(left),
                }
            )
        previous = owned
    return rows


def _write(rows: list[dict[str, Any]], weeks: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT_REPORT.write_text(_render(rows, weeks), encoding="utf-8")


def _render(rows: list[dict[str, Any]], weeks: list[dict[str, Any]]) -> str:
    by_gw = {int(row["gw"]): row for row in weeks}
    total = sum(float(row["xi_points_cap"]) for row in weeks)
    lines = [
        "# The squad the model picks, Gameweeks 1–5",
        "",
        "The published climber on `score_xp`. Gameweek 1 is solved from the "
        "buy pool. Later weeks are free transfers. The chip map is empty. "
        "The horizon is the opening line. xp is that week's score. Points "
        "are the official return. The week total is the scoring eleven after "
        "automatic substitutes, with the captain doubled and hits removed.",
        "",
        f"Five weeks sum to {total:.0f}.",
        "",
    ]
    frame = pd.DataFrame(rows)
    names = dict(zip(frame["player_id"].astype(str), frame["name"].astype(str), strict=False))
    for gw in GWS:
        week = by_gw[int(gw)]
        block = frame.loc[frame["gw"] == int(gw)].copy()
        block["_ord"] = block["position"].map(_ORDER).fillna(9)
        block["_xi"] = (block["lineup"] != "XI").astype(int)
        block = block.sort_values(["_xi", "_ord", "name"], kind="mergesort")
        if int(gw) == 1:
            move = "Solved from the buy pool."
        else:
            inn = _names(names, str(block["in"].iloc[0]))
            out = _names(names, str(block["out"].iloc[0]))
            move = "No transfers." if not inn and not out else f"In {inn}. Out {out}."
        lines.append(
            f"## Gameweek {gw}"
        )
        lines.append("")
        lines.append(
            f"{week['formation']}. {move} "
            f"Automatic substitutes {int(week['n_autosubs'])}. "
            f"Hits {float(week['hit_cost']):.0f}. "
            f"Scoring total {float(week['xi_points_cap']):.0f}."
        )
        lines.append("")
        lines.append("| Player | Pos | Club | Lineup | xp | Points | Minutes |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for row in block.itertuples(index=False):
            mark = str(row.name)
            if int(row.captain):
                mark += " (C)"
            elif int(row.vice):
                mark += " (V)"
            if row.lineup == "XI" and int(row.scored) == 0:
                mark += " — subbed"
            elif row.lineup == "bench" and int(row.scored) == 1:
                mark += " — on"
            lines.append(
                f"| {mark} | {row.position} | {row.team} | {row.lineup} | "
                f"{float(row.xp):.2f} | {float(row.points):.0f} | {float(row.minutes):.0f} |"
            )
        lines.append("")
    lines.extend(
        [
            "A player marked subbed was in the named eleven and played 0 minutes. "
            "A player marked on came off the bench. The captain's extra points "
            "are in the week total and are not added again on his row.",
            "",
        ]
    )
    return "\n".join(lines)


def _names(names: dict[str, str], packed: str) -> str:
    if not packed or packed == "nan":
        return ""
    return ", ".join(names.get(pid, pid) for pid in packed.split(",") if pid)


if __name__ == "__main__":
    run()
