"""How often the higher score wins when two players are close.

One pair per position per week: the eligible leader and the closest
other eligible player, kept only when the gap is above 0 and at most a
half point. Both need five earlier weeks in that season. The shape is
the share of those weeks at 2 points or fewer, and the share at 8 or
more. No standard deviation. The score is not changed.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.reset_gap import PROCESSED, REPORTS

GAP = 0.5
PRIOR_WEEKS = 5
BLANK_AT = 2
HAUL_AT = 8
MIN_DECISIVE = 80
MARGIN = Fraction(1, 20)
MEAN_LOW = Fraction(45, 100)
MEAN_HIGH = Fraction(55, 100)
POSITIONS = ("GKP", "DEF", "MID", "FWD")
FIRST_GW = 5
LAST_GW = 38


def _slash(season: str) -> str:
    return season.replace("-", "/")


def week_points(frame: pd.DataFrame) -> dict[str, dict[int, float]]:
    """One total per player per week. A 0-minute row stays in the sum."""
    gw = pd.to_numeric(frame["gw"], errors="coerce")
    points = pd.to_numeric(frame["total_points"], errors="coerce").fillna(0.0)
    table = pd.DataFrame(
        {
            "element": frame["element"].astype(str),
            "gw": gw,
            "total_points": points,
        }
    ).dropna(subset=["gw"])
    table["gw"] = table["gw"].astype(int)
    grouped = table.groupby(["element", "gw"], sort=False)["total_points"].sum()
    history: dict[str, dict[int, float]] = {}
    for (element, week), total in grouped.items():
        history.setdefault(str(element), {})[int(week)] = float(total)
    return history


def prior_points(history: dict[int, float], gw: int) -> list[float]:
    """Weeks strictly before this deadline, in week order."""
    return [float(history[week]) for week in sorted(history) if int(week) < int(gw)]


def shape_rates(points: list[float]) -> tuple[Fraction, Fraction]:
    """Blank share at 2 or fewer, haul share at 8 or more."""
    if len(points) < PRIOR_WEEKS:
        raise ValueError("a rate needs five earlier weeks")
    blank = sum(1 for value in points if float(value) <= BLANK_AT)
    haul = sum(1 for value in points if float(value) >= HAUL_AT)
    width = len(points)
    return Fraction(blank, width), Fraction(haul, width)


def choose_pair(players: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Leader against the closest score. An equal score forms no pair."""
    if len(players) < 2:
        return None
    ordered = sorted(
        players,
        key=lambda player: (-float(player["score_xp"]), str(player["player_id"])),
    )
    leader, other = ordered[0], ordered[1]
    gap = float(leader["score_xp"]) - float(other["score_xp"])
    if gap <= 0 or gap > GAP:
        return None
    return {"leader": leader, "other": other, "gap": gap}


def _outcome(left: float, right: float) -> str:
    if left > right:
        return "win"
    if left < right:
        return "loss"
    return "tie"


def _shape_outcome(left_rate: Fraction, right_rate: Fraction, left_points: float, right_points: float, *, higher: bool) -> str | None:
    if left_rate == right_rate:
        return None
    left_picked = left_rate > right_rate if higher else left_rate < right_rate
    picked = left_points if left_picked else right_points
    other = right_points if left_picked else left_points
    return _outcome(picked, other)


def build_pair(
    players: list[dict[str, Any]],
    history: dict[str, dict[int, float]],
    gw: int,
) -> dict[str, Any] | None:
    """Attach rates from earlier weeks. A short history drops the pair."""
    chosen = choose_pair(players)
    if chosen is None:
        return None
    for side in ("leader", "other"):
        element = str(chosen[side]["element"])
        earlier = prior_points(history.get(element, {}), gw)
        if len(earlier) < PRIOR_WEEKS:
            return None
        blank, haul = shape_rates(earlier)
        chosen[side] = {
            **chosen[side],
            "blank": blank,
            "haul": haul,
            "n_prior_weeks": len(earlier),
        }
    leader_points = float(chosen["leader"]["points"])
    other_points = float(chosen["other"]["points"])
    return {
        **chosen,
        "mean": _outcome(leader_points, other_points),
        "blank": _shape_outcome(
            chosen["leader"]["blank"],
            chosen["other"]["blank"],
            leader_points,
            other_points,
            higher=False,
        ),
        "haul": _shape_outcome(
            chosen["leader"]["haul"],
            chosen["other"]["haul"],
            leader_points,
            other_points,
            higher=True,
        ),
    }


def _tally(results: list[str | None]) -> dict[str, Any]:
    kept = [result for result in results if result in {"win", "loss", "tie"}]
    wins = sum(result == "win" for result in kept)
    losses = sum(result == "loss" for result in kept)
    ties = sum(result == "tie" for result in kept)
    decisive = wins + losses
    fraction = None if decisive == 0 else Fraction(wins, decisive)
    return {"wins": wins, "losses": losses, "ties": ties, "n": decisive, "fraction": fraction}


def summarise(rows: list[dict[str, Any]], seasons: list[str]) -> dict[str, Any]:
    """Season fractions and the locked reading. Position is not a gate."""
    built = []
    for season in seasons:
        block = [row for row in rows if row["season"] == season]
        mean = _tally([row["mean"] for row in block])
        blank = _tally([row["blank"] for row in block])
        haul = _tally([row["haul"] for row in block])
        positions = {}
        for position in POSITIONS:
            part = [row for row in block if row["position"] == position]
            positions[position] = _tally([row["mean"] for row in part])
        built.append(
            {
                "season": season,
                "n_pairs": len(block),
                "mean": mean,
                "blank": blank,
                "haul": haul,
                "positions": positions,
            }
        )
    return {"seasons": built, **_reading(built)}


def _ahead(block: list[dict[str, Any]], arm: str) -> str:
    for season in block:
        if int(season[arm]["n"]) < MIN_DECISIVE or season[arm]["fraction"] is None:
            return "parked"
        if season["mean"]["fraction"] is None:
            return "parked"
        if season[arm]["fraction"] < season["mean"]["fraction"] + MARGIN:
            return "parked"
    return "ahead"


def _reading(block: list[dict[str, Any]]) -> dict[str, Any]:
    neutral = all(
        season["mean"]["fraction"] is not None
        and MEAN_LOW <= season["mean"]["fraction"] <= MEAN_HIGH
        for season in block
    )
    return {
        "mean_neutral": neutral and len(block) == 4,
        "blank_call": _ahead(block, "blank"),
        "haul_call": _ahead(block, "haul"),
    }


def _pct(fraction: Fraction | None) -> str:
    if fraction is None:
        return "n/a"
    return f"{float(fraction) * 100:.1f}%"


def _arm_line(title: str, arm: dict[str, Any]) -> str:
    ties = int(arm["ties"])
    tie_word = "tie" if ties == 1 else "ties"
    return (
        f"{title}: {_pct(arm['fraction'])} on {arm['n']} decisive pairs, "
        f"{ties} {tie_word}."
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    """The counts, then the reading. A review line is added after Gemini."""
    lines = [
        "# Close players and the shape of their earlier weeks",
        "",
        "Each week, within one position, the pair is the eligible leader and the closest other eligible player. The pair is kept when the leader is strictly ahead by at most half a point, and both have at least five earlier weeks in that season. Eligible is three prior appearances and expected minutes of at least 45. A tie on the score forms no pair. Realised points are that week's total. A tie on points counts for neither side. Blank is the share of earlier weeks at 2 points or fewer. Haul is the share at 8 or more. Those shares use the full sheet, including a week on 0 minutes. The published score table, which supplies the leader, is built from weeks with minutes above 0.",
        "",
        "A shape is ahead of the mean only when every season has at least 80 decisive pairs and the shape beats the mean by at least 5 percentage points in every season. Ahead does not change the score. It would allow a later screen. Anything else parks the shape.",
        "",
    ]
    if result["mean_neutral"]:
        lines.append(
            "The mean does not separate these pairs. In every season the leader wins between 45% and 55% of the decisive pairs."
        )
    else:
        lines.append(
            "The leader's win rate is outside 45% to 55% in at least one season. The mean arm still does not change the score."
        )
    lines.append("")
    for season in result["seasons"]:
        lines.append(f"## {_slash(season['season'])}")
        lines.append("")
        lines.append(f"Pairs {season['n_pairs']}.")
        lines.append(_arm_line("Leader", season["mean"]))
        lines.append(_arm_line("Lower blank rate", season["blank"]))
        lines.append(_arm_line("Higher haul rate", season["haul"]))
        lines.append("")
        for position in POSITIONS:
            arm = season["positions"][position]
            lines.append(f"{position}: {_arm_line('leader', arm)}")
        lines.append("")
    blank = "ahead of the mean" if result["blank_call"] == "ahead" else "parked"
    haul = "ahead of the mean" if result["haul_call"] == "ahead" else "parked"
    lines.extend(
        [
            "## Reading",
            "",
            f"The lower blank rate is {blank}. The higher haul rate is {haul}. The score stays as it is.",
            "",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sheet_points(season: str) -> dict[str, dict[int, float]]:
    from src.models.ridge_multiseason import CACHE

    path = CACHE / f"merged_gw_{season.replace('-', '_')}.csv"
    raw = pd.read_csv(path)
    gw_col = "GW" if "GW" in raw.columns else "round"
    frame = pd.DataFrame(
        {
            "element": raw["element"].astype(str),
            "gw": raw[gw_col],
            "total_points": raw["total_points"],
        }
    )
    return week_points(frame)


def _candidates(feat: pd.DataFrame) -> pd.DataFrame:
    frame = feat.copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame["element"] = frame["player_id"].str.split(":").str[-1]
    frame["position"] = frame["position"].astype(str).replace({"GK": "GKP"})
    frame["score_xp"] = pd.to_numeric(frame["score_xp"], errors="coerce")
    frame["xmi"] = pd.to_numeric(frame["xmi"], errors="coerce")
    frame["n_prior"] = pd.to_numeric(frame["n_prior"], errors="coerce")
    frame["gw"] = pd.to_numeric(frame["gw"], errors="coerce")
    frame = frame.dropna(subset=["gw", "score_xp", "xmi", "n_prior"])
    frame["gw"] = frame["gw"].astype(int)
    frame = frame.loc[
        (frame["n_prior"] >= 3)
        & (frame["xmi"] >= 45)
        & frame["position"].isin(POSITIONS)
        & frame["gw"].between(FIRST_GW, LAST_GW)
    ]
    if "date" in frame.columns:
        frame = frame.sort_values(["gw", "player_id", "date"], kind="mergesort")
    else:
        frame = frame.sort_values(["gw", "player_id"], kind="mergesort")
    return frame.drop_duplicates(["gw", "player_id"], keep="first")


def season_pairs(
    season: str,
    feat: pd.DataFrame,
    history: dict[str, dict[int, float]],
) -> list[dict[str, Any]]:
    """Every kept pair in one season. Gameweek 5 has none: five earlier weeks do not exist."""
    rows: list[dict[str, Any]] = []
    candidates = _candidates(feat)
    for (gw, position), block in candidates.groupby(["gw", "position"], sort=False):
        players = []
        for row in block.itertuples(index=False):
            element = str(row.element)
            played = history.get(element, {}).get(int(gw))
            if played is None:
                continue
            players.append(
                {
                    "player_id": str(row.player_id),
                    "element": element,
                    "score_xp": float(row.score_xp),
                    "points": float(played),
                }
            )
        chosen = build_pair(players, history, int(gw))
        if chosen is None:
            continue
        rows.append(
            {
                "season": season,
                "gw": int(gw),
                "position": str(position),
                "leader_id": chosen["leader"]["player_id"],
                "other_id": chosen["other"]["player_id"],
                "gap": float(chosen["gap"]),
                "leader_points": float(chosen["leader"]["points"]),
                "other_points": float(chosen["other"]["points"]),
                "leader_blank": float(chosen["leader"]["blank"]),
                "other_blank": float(chosen["other"]["blank"]),
                "leader_haul": float(chosen["leader"]["haul"]),
                "other_haul": float(chosen["other"]["haul"]),
                "mean": chosen["mean"],
                "blank": chosen["blank"] if chosen["blank"] is not None else "same",
                "haul": chosen["haul"] if chosen["haul"] is not None else "same",
            }
        )
    return rows


def _flat(rows: list[dict[str, Any]]) -> None:
    path = PROCESSED / "close_pair_shape.csv"
    pd.DataFrame(rows).to_csv(path, index=False)


def run() -> dict[str, Any]:
    """Count the four seasons. Does not change the score."""
    from src.models.ridge_multiseason import SEASONS, build_one_season

    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    names = [season for season, _code in SEASONS]
    rows: list[dict[str, Any]] = []
    for season, code in SEASONS:
        print(f"shape {season}", flush=True)
        feat = build_one_season(season, code)
        history = _sheet_points(season)
        found = season_pairs(season, feat, history)
        print(f"  pairs {len(found)}", flush=True)
        rows.extend(found)
    result = summarise(rows, names)
    _flat(rows)
    write_report(REPORTS / "close_pair_shape.md", result)
    summary = {
        "mean_neutral": result["mean_neutral"],
        "blank": result["blank_call"],
        "haul": result["haul_call"],
        "n": len(rows),
        "seasons": [
            {
                "season": season["season"],
                "n": season["n_pairs"],
                "mean": None if season["mean"]["fraction"] is None else float(season["mean"]["fraction"]),
                "blank_n": season["blank"]["n"],
                "haul_n": season["haul"]["n"],
            }
            for season in result["seasons"]
        ],
    }
    print(summary, flush=True)
    return summary


if __name__ == "__main__":
    run()
