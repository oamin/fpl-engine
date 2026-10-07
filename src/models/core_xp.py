"""Published score against points for the four shared names, Gameweeks 1–5.

``score_xp`` is the climb's pre-deadline score: earlier appearances and that
week's opening line. A week with no minutes is absent from that frame. The
one blank among these four, João Pedro in Gameweek 5, is scored with the same
formula on a single added row and is labelled as such.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import (
    FD_CODE,
    GWS,
    MIN_HISTORY,
    _prior_frame,
    _scoring_columns,
    load_2026_logs,
    load_football_data,
    player_key,
    stamp_unmatched_priors,
    build_frames,
    join_players_to_fixtures,
)
from src.models.xp_engine import add_market_pots, add_player_priors, compute_xp

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "data" / "processed" / "core_xp_gw15.csv"
OUT_REPORT = ROOT / "reports" / "core_xp_gw15.md"

CORE = (
    ("2026-27:124", "Groß"),
    ("2026-27:411", "Haaland"),
    ("2026-27:165", "João Pedro"),
    ("2026-27:368", "Szoboszlai"),
)
BLANK = ("2026-27:165", 5)


def _scored(played: pd.DataFrame) -> pd.DataFrame:
    """The published Gameweek 1–5 frame for one set of appearance rows."""
    prior, _ = _prior_frame()
    fixtures = load_football_data(code=FD_CODE)
    joined, _, _ = join_players_to_fixtures(played, fixtures)
    joined = _scoring_columns(joined)
    joined["player_id"] = [player_key(element) for element in joined["player_id"]]
    joined["gw"] = pd.to_numeric(joined["gw"], errors="coerce").astype(int)
    if "value" not in joined.columns:
        joined["value"] = float("nan")
    joined["value"] = pd.to_numeric(joined["value"], errors="coerce")
    joined["value"] = joined["value"].fillna(
        joined.groupby("position")["value"].transform("median")
    )
    combined = pd.concat([prior, joined], ignore_index=True, sort=False)
    scored = add_market_pots(combined)
    scored = add_player_priors(scored, fill_from=prior)
    scored, _ = stamp_unmatched_priors(scored)
    scored = compute_xp(scored)
    feat = scored.loc[scored["gw"].isin(GWS)].copy()
    feat = feat.loc[feat["n_prior"] >= MIN_HISTORY].copy()
    feat["player_id"] = feat["player_id"].astype(str)
    feat["score_xp"] = feat["xp"]
    return feat


def _blank_score(published: pd.DataFrame) -> float:
    """Same formula on the one dropped row. Shared rows must match."""
    logs = load_2026_logs()
    played = logs.loc[pd.to_numeric(logs["minutes"], errors="coerce") > 0].copy()
    element, gw = BLANK
    extra = logs.loc[
        (logs["player_id"].astype(str) == element.split(":")[-1])
        & (pd.to_numeric(logs["gw"], errors="coerce").astype(int) == gw)
    ]
    if len(extra) != 1:
        raise RuntimeError("João Pedro's Gameweek 5 row is missing from the log")
    plus = _scored(pd.concat([played, extra], ignore_index=True))
    shared = published.merge(plus, on=["player_id", "gw"], suffixes=("_pub", "_plus"))
    # One later forward in the same week sees this row in the expanding
    # forward correction. Every earlier score has to stay put.
    blank_element = int(element.split(":")[-1])
    earlier = shared.loc[
        (shared["gw"] < gw)
        | (shared["player_id"].astype(str).str.split(":").str[-1].astype(int) < blank_element)
    ]
    gap = (earlier["score_xp_pub"] - earlier["score_xp_plus"]).abs()
    if float(gap.max()) > 1e-9:
        raise RuntimeError("adding the blank row moved an earlier score")
    row = plus.loc[(plus["player_id"] == element) & (plus["gw"] == gw)]
    if len(row) != 1:
        raise RuntimeError("the blank row did not score")
    return float(row["score_xp"].iloc[0])


def run() -> list[dict[str, Any]]:
    published, _, _ = build_frames()
    names = dict(CORE)
    wanted = set(names)
    frame = published.loc[published["player_id"].isin(wanted)].copy()
    blank_xp = _blank_score(frame)
    rows: list[dict[str, Any]] = []
    for pid, name in CORE:
        player = frame.loc[frame["player_id"] == pid]
        by_gw = {int(row.gw): row for row in player.itertuples(index=False)}
        for gw in GWS:
            src = by_gw.get(int(gw))
            if src is None:
                if (pid, int(gw)) != BLANK:
                    raise RuntimeError(f"{name} has no score in Gameweek {gw}")
                carried = float(by_gw[gw - 1].score_xp)
                points = 0.0
                minutes = 0.0
                rows.append(
                    {
                        "player": name,
                        "player_id": pid,
                        "gw": int(gw),
                        "score_xp": None,
                        "carried_xp": round(carried, 6),
                        "fixture_xp": round(blank_xp, 6),
                        "points": points,
                        "minutes": minutes,
                        "gap": None,
                    }
                )
                continue
            xp = float(src.score_xp)
            points = float(src.total_points)
            rows.append(
                {
                    "player": name,
                    "player_id": pid,
                    "gw": int(gw),
                    "score_xp": round(xp, 6),
                    "carried_xp": None,
                    "fixture_xp": None,
                    "points": points,
                    "minutes": float(src.minutes),
                    "gap": round(points - xp, 6),
                }
            )
    _write(rows)
    return rows


def _write(rows: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT_REPORT.write_text(_render(rows), encoding="utf-8")


def _num(value: Any) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.2f}"


def _render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Core names, score against points",
        "",
        "Groß, Haaland, João Pedro, and Szoboszlai, the four names in at least "
        "7 of the 14 squads in every week of Gameweeks 1–5. The score is "
        "`score_xp`. It uses appearances before that deadline and that week's "
        "opening line. Points are the official return that week.",
        "",
        "| Player | GW | xp | Points | Points − xp | Minutes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    totals: dict[str, list[float]] = {}
    for row in rows:
        totals.setdefault(row["player"], [0.0, 0.0])
        if row["score_xp"] is None:
            lines.append(
                f"| {row['player']} | {row['gw']} |  | {row['points']:.0f} |  | "
                f"{row['minutes']:.0f} |"
            )
            continue
        shown = round(float(row["score_xp"]), 2)
        totals[row["player"]][0] += shown
        totals[row["player"]][1] += float(row["points"])
        lines.append(
            f"| {row['player']} | {row['gw']} | {shown:.2f} | "
            f"{row['points']:.0f} | {float(row['points']) - shown:.2f} | "
            f"{row['minutes']:.0f} |"
        )
    lines.extend(["", "| Player | xp | Points | Points − xp |", "|---|---:|---:|---:|"])
    for name, (xp, points) in totals.items():
        lines.append(f"| {name} | {xp:.2f} | {points:.0f} | {points - xp:.2f} |")
    blank = next(row for row in rows if row["score_xp"] is None)
    lines.extend(
        [
            "",
            "The sums leave out João Pedro in Gameweek 5. He played 0 minutes, "
            "so that week is not in the published frame. An owned squad copies "
            f"his Gameweek 4 score of {_num(blank['carried_xp'])}. "
            "The same formula on the Gameweek 5 fixture, which the climb does "
            f"not store, is {_num(blank['fixture_xp'])}. He scored 0.",
            "",
            "The score and the margins stay.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    run()
