"""Would a five-midfield or three-forward shape have scored more?

The fifteen stays the one the carry already chose. Only the eleven changes.
The named eleven is the shape with the highest score. Sacrifice is how far
the alternate sits below that score. Realised gain is the alternate's
points minus the named eleven's points, after each shape's own substitutes.
No captain, no hit, and no chip.

The bar was locked before these totals were read. A level shift needs a
median sacrifice of at least 2 and a mean realised gain of at least 2, on
the 14 and on the veterans. Noisy needs a median sacrifice of at most 1 and
the same realised gain, on both. A mean realised gain of 0 or below on the
14 means the alternate does not score more. Anything else leaves the picker
as it is. This run does not change the score.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import build_frames
from src.models.cohort_carry import carry_entry, cohort_specs
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import PROCESSED, REPORTS, _fmt, merged_clubs
from src.models.season_climb import FORMATIONS, apply_autosubs, ordered_bench, pick_xi

FIVE_MID = [(3, 5, 2), (4, 5, 1)]
THREE_FWD = [(3, 4, 3), (4, 3, 3)]
ALTERNATES = (("five midfielders", FIVE_MID), ("three forwards", THREE_FWD))
EXCLUDED_LIMIT = 10


def frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).rename(columns={"id": "player_id"})
    return frame


def play_shape(frame: pd.DataFrame, formations: list[tuple[int, int, int]]) -> dict[str, Any] | None:
    """Score one shape. Substitutes come from the players this shape left out."""
    try:
        xi, form = pick_xi(
            frame,
            "score_xp",
            formations=formations,
            priority_col="xi_priority",
        )
    except RuntimeError:
        return None
    bench = ordered_bench(frame, xi, "score_xp")
    final, _n = apply_autosubs(xi, bench)
    return {
        "form": tuple(int(part) for part in form),
        "xp": float(pd.to_numeric(xi["score_xp"], errors="coerce").fillna(0.0).sum()),
        "points": float(pd.to_numeric(final["total_points"], errors="coerce").fillna(0.0).sum()),
        "ids": sorted(str(pid) for pid in xi["player_id"]),
    }


def week_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per alternate. A named eleven that does not match the carry fails."""
    flat = []
    for week in result["weeks"]:
        frame = frame_from_rows(week["squad_rows"])
        named = play_shape(frame, FORMATIONS)
        if named is None or named["ids"] != list(week["model_intended_ids"]):
            raise RuntimeError(
                f"{result['label']} GW{int(week['gw'])} did not rebuild the named eleven"
            )
        for label, formations in ALTERNATES:
            alternate = play_shape(frame, formations)
            feasible = alternate is not None
            sacrifice = None if not feasible else float(named["xp"] - alternate["xp"])
            gain = None if not feasible else float(alternate["points"] - named["points"])
            if feasible and sacrifice < -1e-6:
                raise RuntimeError(
                    f"{result['label']} GW{int(week['gw'])} alternate outscored the maximum"
                )
            flat.append(
                {
                    "entry_id": result["entry_id"],
                    "label": result["label"],
                    "group": result["group"],
                    "gw": int(week["gw"]),
                    "alternate": label,
                    "feasible": feasible,
                    "named_form": "-".join(str(part) for part in named["form"]),
                    "alt_form": "" if not feasible else "-".join(str(part) for part in alternate["form"]),
                    "named_xp": float(named["xp"]),
                    "alt_xp": None if not feasible else float(alternate["xp"]),
                    "named_points": float(named["points"]),
                    "alt_points": None if not feasible else float(alternate["points"]),
                    "sacrifice": sacrifice,
                    "gain": gain,
                }
            )
    return flat


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["feasible"]]
    excluded = len(rows) - len(usable)
    if not usable:
        return {"n": 0, "excluded": excluded, "median": None, "mean": None, "call": "inconclusive"}
    median = _median([float(row["sacrifice"]) for row in usable])
    mean = _mean([float(row["gain"]) for row in usable])
    return {"n": len(usable), "excluded": excluded, "median": median, "mean": mean}


def decide(cohort: dict[str, Any], veterans: dict[str, Any], *, cohort_rows: int) -> str:
    """The locked call. Rank slots cannot make it on their own."""
    if cohort["excluded"] > EXCLUDED_LIMIT or cohort["n"] == 0 or veterans["n"] == 0:
        return "inconclusive"
    if cohort_rows != 70:
        return "inconclusive"
    if float(cohort["mean"]) <= 0:
        return "stays"
    cohort_shift = float(cohort["median"]) >= 2 and float(cohort["mean"]) >= 2
    veteran_shift = float(veterans["median"]) >= 2 and float(veterans["mean"]) >= 2
    if cohort_shift and veteran_shift:
        return "level shift"
    cohort_noisy = float(cohort["median"]) <= 1 and float(cohort["mean"]) >= 2
    veteran_noisy = float(veterans["median"]) <= 1 and float(veterans["mean"]) >= 2
    if cohort_noisy and veteran_noisy:
        return "noisy"
    return "inconclusive"


def _verdict(label: str, call: str) -> str:
    if call == "level shift":
        return (
            f"{label}: the named shape is ahead on score by a wide margin and the alternate "
            "still scores more. That sends the position levels to a four-season screen. "
            "The score is not changed here."
        )
    if call == "noisy":
        return (
            f"{label}: the alternate is close on score and ahead on points. "
            "These five weeks are the noisy case. The picker stays."
        )
    if call == "stays":
        return f"{label}: the alternate does not score more. The picker stays."
    return f"{label}: the result is inconclusive. The picker stays."


def _stat_line(title: str, stats: dict[str, Any]) -> str:
    if stats["median"] is None:
        return f"{title}: no feasible week. Excluded {stats['excluded']}."
    return (
        f"{title}: median sacrifice {stats['median']:.2f}, "
        f"mean realised gain {_fmt(stats['mean'])}, "
        f"{stats['n']} weeks, {stats['excluded']} excluded."
    )


def run() -> dict[str, Any]:
    """Compare the shapes on the carried fifteens. Does not retune."""
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
        print(f"trial {spec['entry_id']} {spec['label']}", flush=True)
        result = carry_entry(spec, feat, roster, clubs, roster_by_gw, horizon_scores)
        if not result["finished"]:
            raise RuntimeError(f"{spec['label']} did not finish: {result['error']}")
        records.extend(week_rows(result))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(PROCESSED / "shape_trial_gw15.csv", index=False)
    calls = _write_report(REPORTS / "shape_trial_gw15.md", records)
    return calls


def _write_report(path: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    calls = {}
    lines = [
        "# Five midfielders or three forwards, on the same fifteen",
        "",
        "The fifteen is the one the carry already chose. Only the eleven changes. Five midfielders means the higher score of 3-5-2 and 4-5-1. Three forwards means the higher score of 3-4-3 and 4-3-3. The named eleven is the shape with the highest score, and it has to match the eleven the carry banked.",
        "",
        "Sacrifice is the named score minus the alternate score. Realised gain is the alternate's points minus the named points, after each shape brings on its own substitutes. Captain points, hits, and chips are left out.",
        "",
        "The bar was locked before these totals were read. A level shift needs a median sacrifice of at least 2 and a mean realised gain of at least 2, on the 14 and on the veterans. Noisy needs a median sacrifice of at most 1 and that same realised gain, on both. A mean realised gain of 0 or below on the 14 means the alternate does not score more. Any other result, including the 14 and the veterans disagreeing, leaves the picker as it is. More than 10 weeks that cannot form the shape makes that alternate inconclusive. The score is not changed in this run.",
        "",
    ]
    for label, _formations in ALTERNATES:
        block = [row for row in records if row["alternate"] == label and row["group"] != "reference"]
        veterans = [row for row in block if row["group"] == "veteran"]
        ranks = [row for row in block if row["group"] == "rank"]
        reference = [row for row in records if row["alternate"] == label and row["group"] == "reference"]
        cohort_stats = summarise(block)
        veteran_stats = summarise(veterans)
        call = decide(cohort_stats, veteran_stats, cohort_rows=len(block))
        calls[label] = call
        lines.append(f"## {label}")
        lines.append("")
        lines.append(_verdict(label.capitalize(), call))
        lines.append("")
        lines.append(_stat_line("The 14", cohort_stats))
        lines.append(_stat_line("Veterans", veteran_stats))
        lines.append(_stat_line("Rank slots", summarise(ranks)))
        lines.append(_stat_line("ojaminFC", summarise(reference)))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return calls


if __name__ == "__main__":
    print(run())
