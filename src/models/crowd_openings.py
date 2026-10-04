"""Gameweek 1 fifteens drawn from high ownership.

These are crowd templates. They are not hall-of-fame squads. The picker
uses ownership, price, and the squad law. It does not read points.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, MAX_PER_CLUB, SQUAD_QUOTA

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SEASONS = ("2022_23", "2023_24", "2024_25", "2025_26")
SEASON_LABEL = {
    "2022_23": "2022/23",
    "2023_24": "2023/24",
    "2024_25": "2024/25",
    "2025_26": "2025/26",
}
POOL_DEPTH = {"GKP": 8, "DEF": 16, "MID": 16, "FWD": 10}
SQUAD_NAMES = ("A", "B", "C", "D")
_POSITION = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def gameweek_one(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per player from gameweek 1, earliest kickoff."""
    need = {"name", "position", "team", "element", "selected", "value", "GW", "kickoff_time"}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"gameweek file is missing {sorted(missing)}")
    out = frame.loc[frame["GW"].astype(int) == 1, list(need)].copy()
    out["element"] = pd.to_numeric(out["element"], errors="coerce")
    out["selected"] = pd.to_numeric(out["selected"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["position"] = out["position"].map(_POSITION)
    out["team"] = out["team"].astype(str)
    out["name"] = out["name"].astype(str)
    out = out.dropna(subset=["element", "selected", "value", "position"])
    out = out.loc[out["team"].ne("nan")]
    out["element"] = out["element"].astype(int)
    out["value"] = out["value"].astype(int)
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True, errors="coerce")
    out = out.sort_values(
        ["kickoff_time", "element"], ascending=[True, True], na_position="last"
    )
    out = out.drop_duplicates("element", keep="first")
    total = float(out["selected"].sum())
    if total <= 0:
        raise ValueError("gameweek 1 has no ownership")
    # Share of managers. selected / (sum/15) leaves the rank order unchanged.
    out["own"] = 15.0 * out["selected"] / total
    return out.reset_index(drop=True)


def ownership_pool(gw1: pd.DataFrame) -> pd.DataFrame:
    """Top owned players in each position. Ties go to the smaller element id."""
    parts = []
    for position, depth in POOL_DEPTH.items():
        block = gw1.loc[gw1["position"] == position].sort_values(
            ["own", "element"], ascending=[False, True]
        )
        parts.append(block.head(depth))
    if not parts:
        return gw1.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def exclusions(gw1: pd.DataFrame) -> dict[str, set[int]]:
    """Who each squad must leave out, from the gameweek ranking."""
    ranked = gw1.sort_values(["own", "element"], ascending=[False, True])
    top = int(ranked.iloc[0]["element"])
    per_position: dict[str, list[int]] = {}
    for position in SQUAD_QUOTA:
        ids = (
            ranked.loc[ranked["position"] == position, "element"]
            .astype(int)
            .head(2)
            .tolist()
        )
        per_position[position] = ids
    return {
        "A": set(),
        "B": {top},
        "C": {ids[0] for ids in per_position.values() if ids},
        "D": {element for ids in per_position.values() for element in ids},
    }


def _can_finish(
    pool: pd.DataFrame,
    picked: set[int],
    quota: dict[str, int],
    clubs: dict[str, int],
    budget: int,
) -> bool:
    """True when the remaining slots can be filled from the pool on price alone."""
    need = dict(quota)
    used = set(picked)
    club_count = dict(clubs)
    spent = 0
    while sum(need.values()):
        best: pd.Series | None = None
        for position, left in need.items():
            if left <= 0:
                continue
            block = pool.loc[
                (pool["position"] == position) & (~pool["element"].isin(used))
            ].sort_values(["value", "element"], ascending=[True, True])
            for _, row in block.iterrows():
                if club_count.get(str(row["team"]), 0) >= MAX_PER_CLUB:
                    continue
                if best is None or (int(row["value"]), int(row["element"])) < (
                    int(best["value"]),
                    int(best["element"]),
                ):
                    best = row
                break
        if best is None or spent + int(best["value"]) > budget:
            return False
        spent += int(best["value"])
        used.add(int(best["element"]))
        need[str(best["position"])] -= 1
        club = str(best["team"])
        club_count[club] = club_count.get(club, 0) + 1
    return True


def build_squad(pool: pd.DataFrame, banned: set[int]) -> pd.DataFrame | None:
    """Highest-ownership legal fifteen inside the pool, skipping ``banned``."""
    quota = {position: int(count) for position, count in SQUAD_QUOTA.items()}
    picked: list[pd.Series] = []
    clubs: dict[str, int] = {}
    budget = int(BUDGET_TENTHS)
    order = pool.sort_values(["own", "element"], ascending=[False, True])
    while sum(quota.values()):
        chosen: pd.Series | None = None
        picked_ids = {int(row["element"]) for row in picked}
        for _, row in order.iterrows():
            element = int(row["element"])
            position = str(row["position"])
            club = str(row["team"])
            price = int(row["value"])
            if element in banned or element in picked_ids:
                continue
            if quota[position] <= 0 or clubs.get(club, 0) >= MAX_PER_CLUB:
                continue
            if price > budget:
                continue
            quota_after = dict(quota)
            quota_after[position] -= 1
            clubs_after = dict(clubs)
            clubs_after[club] = clubs_after.get(club, 0) + 1
            if _can_finish(pool, picked_ids | {element}, quota_after, clubs_after, budget - price):
                chosen = row
                break
        if chosen is None:
            return None
        picked.append(chosen)
        quota[str(chosen["position"])] -= 1
        club = str(chosen["team"])
        clubs[club] = clubs.get(club, 0) + 1
        budget -= int(chosen["value"])
    out = pd.DataFrame(picked)
    return out.reset_index(drop=True)


def build_season(frame: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    """The four cross-section fifteens for one season file."""
    gw1 = gameweek_one(frame)
    pool = ownership_pool(gw1)
    built = {}
    for name, banned in exclusions(gw1).items():
        built[name] = build_squad(pool, banned)
    return built


def squad_frame(season: str, squads: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """One row per player, with the season and the squad letter."""
    rows = []
    for name in SQUAD_NAMES:
        squad = squads[name]
        if squad is None:
            continue
        block = squad.copy()
        block.insert(0, "squad", name)
        block.insert(0, "season", season)
        rows.append(block)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = ["season", "squad", "element", "name", "position", "team", "value", "own", "selected"]
    return out[keep]


def load_season(season: str) -> pd.DataFrame:
    path = CACHE / f"merged_gw_{season}.csv"
    return pd.read_csv(path)


def build_all(seasons: tuple[str, ...] = SEASONS) -> pd.DataFrame:
    frames = []
    for season in seasons:
        frames.append(squad_frame(season, build_season(load_season(season))))
    return pd.concat(frames, ignore_index=True)


def _pounds(tenths: int) -> str:
    return f"£{tenths / 10:.1f}m"


def _percent(own: float) -> str:
    return f"{100.0 * own:.1f}%"


def render_report(table: pd.DataFrame) -> str:
    """The fifteens, one season at a time."""
    lines = [
        "# Crowd Gameweek 1 fifteens",
        "",
        "Built on 2026-10-04. Gemini kept the rule before the squads were read "
        "([crowd openings](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). "
        "These are templates from Gameweek 1 ownership. They are not hall-of-fame squads, "
        "and no season climb was run.",
        "",
        "Ownership is the share of managers, `15 × selected / sum(selected)`, on the "
        "deduped Gameweek 1 list. The file is scraped after the gameweek, so a Gameweek 2 "
        "transfer made before the scrape can move a rank. Later weeks are not read. "
        "Points are not read.",
        "",
        "The pool is the top 8 goalkeepers, top 16 defenders, top 16 midfielders, and "
        "top 10 forwards. Squad A takes the highest-owned legal fifteen inside that pool. "
        "Squad B leaves out the single most-owned player. Squad C leaves out the most-owned "
        "player in each position. Squad D leaves out the two most-owned in each position. "
        "A fifteen that cannot be finished inside the pool is left missing.",
        "",
    ]
    position_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
    for season in SEASONS:
        lines.append(f"## {SEASON_LABEL[season]}")
        lines.append("")
        block = table.loc[table["season"] == season]
        for name in SQUAD_NAMES:
            squad = block.loc[block["squad"] == name].copy()
            lines.append(f"### Squad {name}")
            lines.append("")
            if squad.empty:
                lines.append("Missing. The pool could not fill a legal fifteen.")
                lines.append("")
                continue
            spent = int(squad["value"].sum())
            lines.append(f"Cost {_pounds(spent)}.")
            lines.append("")
            lines.append("| Player | Position | Club | Price | Owned |")
            lines.append("| --- | --- | --- | --- | --- |")
            squad["_ord"] = squad["position"].map(position_order)
            squad = squad.sort_values(["_ord", "own", "element"], ascending=[True, False, True])
            for _, row in squad.iterrows():
                lines.append(
                    f"| {row['name']} | {row['position']} | {row['team']} | "
                    f"{_pounds(int(row['value']))} | {_percent(float(row['own']))} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(table: pd.DataFrame | None = None) -> pd.DataFrame:
    table = build_all() if table is None else table
    PROCESSED.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED / "crowd_openings.csv", index=False)
    (REPORTS / "crowd_openings.md").write_text(render_report(table), encoding="utf-8")
    return table


if __name__ == "__main__":
    built = write_outputs()
    print(f"wrote {len(built)} rows")
