"""Hold and climb the four crowd Gameweek 1 fifteens.

The decision score is ``score_xp``. The chip map is empty. The default
opening-price horizon is the published one. A hold sets the margin high
enough that a legal fifteen never transfers. A hold that still transfers
is a failure, not a baseline. Totals are written only to this side report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.blank_context import clubs_by_gw, playable_gws
from src.models.crowd_openings import SQUAD_LABEL, SQUAD_NAMES
from src.models.ridge_multiseason import SEASONS, build_one_season
from src.models.season_climb_ft import (
    HOLD_EPS,
    SquadState,
    load_vaastav_roster,
    run_ft_season,
)
from src.rules.fpl_2026 import BUDGET_TENTHS

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OPENINGS = PROCESSED / "crowd_openings.csv"
SCORE_CSV = PROCESSED / "crowd_opening_scores.csv"
WEEK_CSV = PROCESSED / "crowd_opening_score_weeks.csv"
REPORT_PATH = REPORTS / "crowd_opening_scores.md"

GWS = list(range(1, 39))
# A legal squad cannot clear this. An illegal squad is allowed to move,
# and that move fails the hold.
HOLD_NEVER = 1.0e9


def feature_season(crowd_season: str) -> str:
    """``2022_23`` in the openings file is ``2022-23`` on the feature table."""
    return str(crowd_season).replace("_", "-")


def opening_state(season: str, squad: pd.DataFrame) -> SquadState:
    """Purchase prices are the Gameweek 1 values. Bank is the leftover budget."""
    if len(squad) != 15:
        raise ValueError(f"{season} opening has {len(squad)} players")
    elements = [int(element) for element in squad["element"]]
    if len(set(elements)) != 15:
        raise ValueError(f"{season} opening repeats a player")
    purchase: dict[str, int] = {}
    for element, value in zip(elements, squad["value"], strict=True):
        price = int(value)
        if price < 0:
            raise ValueError(f"{season}:{element} has a negative price")
        purchase[f"{season}:{element}"] = price
    spent = sum(purchase.values())
    if spent > BUDGET_TENTHS:
        raise ValueError(f"{season} opening costs {spent} tenths")
    return SquadState(purchase=purchase, bank=int(BUDGET_TENTHS - spent), ft=1)


def load_openings(path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(OPENINGS if path is None else path)
    frame["element"] = pd.to_numeric(frame["element"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["element", "value", "season", "squad"])
    frame["element"] = frame["element"].astype(int)
    frame["value"] = frame["value"].astype(int)
    return frame


def _empty_path(error: str) -> dict[str, object]:
    return {
        "points": float("nan"),
        "transfers": 0,
        "hits": 0,
        "hit_cost": 0.0,
        "weeks": [],
        "ok": False,
        "error": error,
    }


def path_result(weekly: pd.DataFrame, expected: list[int], *, hold: bool) -> dict[str, object]:
    """Sum the published week score. A short run, a chip, or a moving hold fails."""
    if weekly is None or weekly.empty or "method" not in weekly.columns:
        return _empty_path("no weeks scored")
    part = weekly.loc[weekly["method"] == "xp_ft"].copy()
    if part.empty:
        return _empty_path("no weeks scored")
    weeks = sorted(int(gw) for gw in part["gw"].unique())
    points = float(part["xi_points_cap"].sum())
    transfers = int(part["n_transfers"].sum())
    hits = int(part["hits"].sum())
    hit_cost = float(part["hit_cost"].sum())
    reasons: list[str] = []
    if weeks != sorted(int(gw) for gw in expected):
        reasons.append(
            f"scored {len(weeks)} weeks, expected {len(expected)}"
        )
    if "chip" in part.columns and part["chip"].notna().any():
        reasons.append("a chip was played")
    if hold and (transfers != 0 or hits != 0):
        reasons.append(f"hold transferred {transfers} times, {hits} hits")
    return {
        "points": points,
        "transfers": transfers,
        "hits": hits,
        "hit_cost": hit_cost,
        "weeks": weeks,
        "ok": not reasons,
        "error": "; ".join(reasons),
    }


def summarise_pair(
    season: str,
    squad: str,
    cost: int,
    hold: dict[str, object],
    climb: dict[str, object],
    expected: list[int],
) -> dict[str, object]:
    both = bool(hold["ok"]) and bool(climb["ok"])
    hold_points = float(hold["points"])
    climb_points = float(climb["points"])
    gap = climb_points - hold_points if both else float("nan")
    return {
        "season": season,
        "squad": squad,
        "cost": int(cost),
        "bank": int(BUDGET_TENTHS - cost),
        "hold_points": hold_points,
        "climb_points": climb_points,
        "gap": gap,
        "hold_transfers": int(hold["transfers"]),
        "climb_transfers": int(climb["transfers"]),
        "hold_hits": int(hold["hits"]),
        "climb_hits": int(climb["hits"]),
        "hold_hit_cost": float(hold["hit_cost"]),
        "climb_hit_cost": float(climb["hit_cost"]),
        "hold_weeks": len(hold["weeks"]),
        "climb_weeks": len(climb["weeks"]),
        "expected_weeks": len(expected),
        "hold_ok": bool(hold["ok"]),
        "climb_ok": bool(climb["ok"]),
        "hold_error": str(hold["error"]),
        "climb_error": str(climb["error"]),
    }


def best_climbs(table: pd.DataFrame) -> dict[str, list[str]]:
    """Highest completed climb inside each season. A tie keeps every name."""
    winners: dict[str, list[str]] = {}
    if table.empty:
        return winners
    for season, block in table.groupby("season", sort=False):
        done = block.loc[block["climb_ok"].astype(bool)]
        if done.empty:
            winners[str(season)] = []
            continue
        top = float(done["climb_points"].max())
        names = done.loc[done["climb_points"] == top, "squad"].astype(str).tolist()
        winners[str(season)] = names
    return winners


def prepare_season(season: str, code: str) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    feat = build_one_season(season, code)
    feat["eligible"] = pd.to_numeric(feat["xmi"], errors="coerce").fillna(0) >= 45.0
    roster = load_vaastav_roster(season)
    expected = playable_gws(GWS, clubs_by_gw(roster))
    return feat, roster, expected


def _run_path(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    opening: SquadState,
    expected: list[int],
    *,
    hold: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    try:
        weekly = run_ft_season(
            feat,
            {"xp": "score_xp"},
            GWS,
            roster=roster,
            hold_eps=HOLD_NEVER if hold else None,
            chips=None,
            opening=opening,
            freeze_horizon=False,
        )
    except Exception as exc:
        return _empty_path(f"{type(exc).__name__}: {exc}"), pd.DataFrame()
    return path_result(weekly, expected, hold=hold), weekly


def _tag(weekly: pd.DataFrame, season: str, squad: str, path: str) -> pd.DataFrame:
    if weekly.empty:
        return weekly
    out = weekly.copy()
    out.insert(0, "path", path)
    out.insert(0, "squad", squad)
    out.insert(0, "season", season)
    return out


def score_squad(
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    season: str,
    squad: str,
    rows: pd.DataFrame,
    expected: list[int],
) -> tuple[dict[str, object], pd.DataFrame]:
    """One fifteen, held and then climbed. The feature frame is shared."""
    known = set(roster["player_id"].astype(str))
    try:
        opening = opening_state(season, rows)
    except ValueError as exc:
        failed = _empty_path(str(exc))
        return summarise_pair(season, squad, 0, failed, failed, expected), pd.DataFrame()
    missing = sorted(opening.ids() - known)
    if missing:
        failed = _empty_path("missing from the roster: " + ", ".join(missing))
        cost = sum(opening.purchase.values())
        return summarise_pair(season, squad, cost, failed, failed, expected), pd.DataFrame()
    cost = sum(opening.purchase.values())
    hold, hold_weeks = _run_path(feat, roster, opening, expected, hold=True)
    climb, climb_weeks = _run_path(feat, roster, opening, expected, hold=False)
    summary = summarise_pair(season, squad, cost, hold, climb, expected)
    weeks = pd.concat(
        [
            _tag(hold_weeks, season, squad, "hold"),
            _tag(climb_weeks, season, squad, "climb"),
        ],
        ignore_index=True,
    )
    return summary, weeks


def score_all(openings: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Four seasons, four squads, hold and climb. One season is built once."""
    openings = load_openings() if openings is None else openings
    summaries: list[dict[str, object]] = []
    weeks: list[pd.DataFrame] = []
    for season, code in SEASONS:
        print(f"building {season}", flush=True)
        slug = season.replace("-", "_")
        try:
            feat, roster, expected = prepare_season(season, code)
        except Exception as exc:
            failed = _empty_path(f"{type(exc).__name__}: {exc}")
            for squad in SQUAD_NAMES:
                summaries.append(summarise_pair(season, squad, 0, failed, failed, []))
            continue
        for squad in SQUAD_NAMES:
            print(f"  {squad}", flush=True)
            block = openings.loc[
                (openings["season"] == slug) & (openings["squad"] == squad)
            ]
            summary, detail = score_squad(feat, roster, season, squad, block, expected)
            summaries.append(summary)
            if not detail.empty:
                weeks.append(detail)
            pd.DataFrame(summaries).to_csv(SCORE_CSV, index=False)
    table = pd.DataFrame(summaries)
    detail = pd.concat(weeks, ignore_index=True) if weeks else pd.DataFrame()
    return table, detail


def _points(value: object) -> str:
    number = float(value)
    if number != number:
        return "failure"
    return f"{number:.0f}"


def _signed(value: object) -> str:
    number = float(value)
    if number != number:
        return "failure"
    return f"{number:+.0f}"


def _squad_name(squad: str) -> str:
    return SQUAD_LABEL.get(squad, squad)


def render_report(table: pd.DataFrame, *, review: str = "") -> str:
    """Side report. The best climb is named inside its own season."""
    lines = [
        "# Crowd opening scores",
        "",
        "Built from the Gameweek 1 fifteens in `reports/crowd_openings.md`. "
        "Each fifteen is scored twice on the published rule. "
        "The hold sets the margin high enough that a legal squad never transfers. "
        "The climb uses the published hold margin "
        f"({HOLD_EPS}). The chip map is empty. The decision score is `score_xp`. "
        "Later weeks in the three-week window use that fixture's opening price. "
        "A double is one fixture on the decision score. "
        "Actual points are Vaastav `total_points` for the chosen eleven, "
        "with the captain extra, automatic substitutes, and hit deductions.",
        "",
        "2022/23, 2023/24, 2024/25, and 2025/26 are all in this run. "
        "The chip map is empty, so 2024/25 does not need an Assistant Manager. "
        "Purchase price is the Gameweek 1 value. The bank is £100.0m minus that cost. "
        "A hold that transfers is a failure, and its sum is not a baseline. "
        "The best climb is named inside that season.",
        "",
        "These are crowd templates from Gameweek 1 ownership. "
        "The ownership file is scraped after the gameweek. "
        "Nothing here replaces a published climb total.",
        "",
    ]
    if review:
        lines.extend([review.strip(), ""])
    else:
        lines.extend(["The diagnostic review of these totals is still open.", ""])
    if table.empty:
        lines.append("No squads were scored.")
        return "\n".join(lines).rstrip() + "\n"
    winners = best_climbs(table)
    for season, block in table.groupby("season", sort=False):
        label = str(season).replace("-", "/")
        lines.append(f"## {label}")
        lines.append("")
        lines.append(
            "| Squad | Cost | Hold | Climb | Climb − hold | "
            "Climb transfers | Climb hits |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in block.iterrows():
            hold_cell = _points(row["hold_points"]) if bool(row["hold_ok"]) else "failure"
            climb_cell = _points(row["climb_points"]) if bool(row["climb_ok"]) else "failure"
            gap_cell = _signed(row["gap"]) if bool(row["hold_ok"]) and bool(row["climb_ok"]) else "failure"
            transfers = int(row["climb_transfers"]) if bool(row["climb_ok"]) else "failure"
            hits = int(row["climb_hits"]) if bool(row["climb_ok"]) else "failure"
            cost = int(row["cost"])
            cost_cell = f"£{cost / 10:.1f}m" if cost else "failure"
            lines.append(
                f"| {_squad_name(str(row['squad']))} | {cost_cell} | {hold_cell} | "
                f"{climb_cell} | {gap_cell} | {transfers} | {hits} |"
            )
        lines.append("")
        for _, row in block.iterrows():
            notes = []
            if not bool(row["hold_ok"]) and str(row["hold_error"]):
                notes.append(f"hold: {row['hold_error']}")
            if not bool(row["climb_ok"]) and str(row["climb_error"]):
                notes.append(f"climb: {row['climb_error']}")
            if notes:
                lines.append(f"{_squad_name(str(row['squad']))} — {'; '.join(notes)}.")
        if any(
            (not bool(row["hold_ok"]) and str(row["hold_error"]))
            or (not bool(row["climb_ok"]) and str(row["climb_error"]))
            for _, row in block.iterrows()
        ):
            lines.append("")
        names = winners.get(str(season), [])
        if not names:
            lines.append(f"Best climb in {label}: none. Every climb failed.")
        elif len(names) == 1:
            winner = block.loc[block["squad"] == names[0]].iloc[0]
            lines.append(
                f"Best climb in {label}: {_squad_name(names[0])} "
                f"at {_points(winner['climb_points'])}."
            )
        else:
            top = float(block.loc[block["squad"].isin(names), "climb_points"].iloc[0])
            joined = " and ".join(_squad_name(name) for name in names)
            lines.append(f"Best climb in {label}: {joined}, tied at {_points(top)}.")
        lines.append("")
    lines.extend(
        [
            "Week rows are in `data/processed/crowd_opening_score_weeks.csv`. "
            "Season totals are in `data/processed/crowd_opening_scores.csv`.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    table: pd.DataFrame | None = None,
    weeks: pd.DataFrame | None = None,
    *,
    review: str = "",
) -> pd.DataFrame:
    if table is None or weeks is None:
        table, weeks = score_all()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(SCORE_CSV, index=False)
    weeks.to_csv(WEEK_CSV, index=False)
    REPORT_PATH.write_text(render_report(table, review=review), encoding="utf-8")
    return table


if __name__ == "__main__":
    scored = write_outputs()
    print(f"wrote {len(scored)} squads")
