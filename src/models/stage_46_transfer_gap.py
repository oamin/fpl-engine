"""Gameweeks 1–5 transfer gap against a locked set of managers.

The cohort was fixed before any of these squads were opened. Veterans are
the first seven names on the FPL Form multi-season table with at least two
top-10,000 finishes in 2022/23–2025/26. Rank slots are overall classic
league ``rank_sort`` 100, 500, 1000, 5000, 10000, 25000, and 50000 on
2026-10-04. ojaminFC is a reference row and stays out of both means.

Each climb starts from that manager's Gameweek 1 fifteen on the published
path: opening horizon, early score, empty chip map. A sale is someone in
one week's squad and absent from the next. Pairs stay inside a position,
highest decision-week ``score_xp`` with highest. A move that crosses
position is counted on its own.

The window is the sale week and the next two, the same length as the
published horizon, and it stops at Gameweek 5. A shorter window is flagged
and left short. Gross is points scored by the player sold minus points
scored by the player bought. The week's hit is added once, on the week.
Positive means the player sold scored more.

The same window is counted on the managers' own transfers, using the pairs
the public entry already recorded. Their season points include chips and
are a yardstick beside the transfer count. A sign is consistent when, among
managers with a paired sale, one sign is held by at least four and by at
least twice the other sign. The same sign on the model's sales, with the
managers' own sales failing to share it, is the systematic reading. Five
weeks diagnose the transfers. The score stays ``score_xp``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from src.live.benchmark import GWS, _opening_state, build_frames, player_key
from src.live.entry import ENTRY_DIR, ENTRY_URL, EntryError, build_entry, load_entry, save_entry
from src.models.season_climb_ft import HORIZON, run_ft_season

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
LOCK_DATE = "2026-10-04"
LAST_GW = max(GWS)
CONSISTENT_MIN = 4
CONSISTENT_RATIO = 2
PUBLISHED_REFERENCE_POINTS = 333.0
REFERENCE_ID = 2632584

# Prior ranks are 2025/26, 2024/25, 2023/24, 2022/23, in that order.
# A rank above 10000 is outside the top 10,000. The entry id is the lock.
VETERANS: tuple[dict[str, Any], ...] = (
    {
        "entry_id": 2076855,
        "label": "elevenify.com",
        "group": "veteran",
        "prior_ranks": (210, 1309, 939, 1891),
    },
    {
        "entry_id": 9267,
        "label": "Cameron Scott",
        "group": "veteran",
        "prior_ranks": (135, 2063, 62, 31639),
    },
    {
        "entry_id": 31365,
        "label": "Ashley Marsh",
        "group": "veteran",
        "prior_ranks": (2778, 11475, 3, 24541),
    },
    {
        "entry_id": 616,
        "label": "Mark Brookes",
        "group": "veteran",
        "prior_ranks": (309, 31902, 24, 20293),
    },
    {
        "entry_id": 4109079,
        "label": "Will Morrison",
        "group": "veteran",
        "prior_ranks": (8205, 2140, 332, 20293),
    },
    {
        "entry_id": 2116184,
        "label": "Paul Mitchell",
        "group": "veteran",
        "prior_ranks": (12632, 66, 2072, 52952),
    },
    {
        "entry_id": 32058,
        "label": "Thomas O'Brien",
        "group": "veteran",
        "prior_ranks": (1720, 797, 3433, 38905),
    },
)

# Displayed rank skips on ties. These rows were selected by rank_sort.
RANK_SLOTS: tuple[dict[str, Any], ...] = (
    {
        "entry_id": 5874076,
        "label": "Jess Bernstein",
        "group": "rank",
        "rank_sort": 100,
        "displayed_rank": 95,
        "points_at_lock": 429,
    },
    {
        "entry_id": 3669095,
        "label": "Adam Whitting",
        "group": "rank",
        "rank_sort": 500,
        "displayed_rank": 485,
        "points_at_lock": 419,
    },
    {
        "entry_id": 169164,
        "label": "Filip Stripaj",
        "group": "rank",
        "rank_sort": 1000,
        "displayed_rank": 994,
        "points_at_lock": 414,
    },
    {
        "entry_id": 217044,
        "label": "daniel bowes",
        "group": "rank",
        "rank_sort": 5000,
        "displayed_rank": 4923,
        "points_at_lock": 403,
    },
    {
        "entry_id": 534464,
        "label": "Sion Jones",
        "group": "rank",
        "rank_sort": 10000,
        "displayed_rank": 9881,
        "points_at_lock": 398,
    },
    {
        "entry_id": 3503484,
        "label": "Viniii Denie",
        "group": "rank",
        "rank_sort": 25000,
        "displayed_rank": 24786,
        "points_at_lock": 391,
    },
    {
        "entry_id": 1585157,
        "label": "Tom Anderson",
        "group": "rank",
        "rank_sort": 50000,
        "displayed_rank": 49790,
        "points_at_lock": 384,
    },
)

REFERENCE: dict[str, Any] = {
    "entry_id": REFERENCE_ID,
    "label": "ojaminFC",
    "group": "reference",
}

COHORT: tuple[dict[str, Any], ...] = VETERANS + RANK_SLOTS + (REFERENCE,)
POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def prior_inside(ranks: tuple[int, ...], cutoff: int = 10000) -> int:
    return sum(1 for rank in ranks if int(rank) <= cutoff)


def window_gws(sale_gw: int, last_gw: int, horizon: int = HORIZON) -> list[int]:
    """Sale week through the next ``horizon - 1`` weeks, clipped at ``last_gw``."""
    return [gw for gw in range(int(sale_gw), int(sale_gw) + int(horizon)) if gw <= int(last_gw)]


def window_points(
    player_id: str, gws: list[int], points: dict[tuple[str, int], float]
) -> float:
    """Sum of actual points. A week with no roster row scores 0."""
    total = 0.0
    for gw in gws:
        total += float(points.get((str(player_id), int(gw)), 0.0))
    return total


def pair_gross(sold_points: float, bought_points: float) -> float:
    return float(sold_points) - float(bought_points)


def week_net(sold_points: list[float], bought_points: list[float], hit_cost: float) -> float:
    """Points of everyone sold, minus everyone bought, plus the week's hit once."""
    return float(sum(sold_points) - sum(bought_points) + float(hit_cost))


def _score_sort_key(player: dict[str, Any]) -> tuple[float, str]:
    score = player.get("score_xp")
    if score is None or (isinstance(score, float) and math.isnan(score)):
        value = float("-inf")
    else:
        value = float(score)
    return (-value, str(player["player_id"]))


def pair_within_position(
    sold: list[dict[str, Any]], bought: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Highest decision-week score with highest, inside one position.

    A player left over, or a move into a different position, stays unpaired.
    """
    positions = {str(player["position"]) for player in sold}
    positions.update(str(player["position"]) for player in bought)
    ordered = sorted(positions, key=lambda pos: (POSITION_ORDER.get(pos, 9), pos))
    rows: list[dict[str, Any]] = []
    for position in ordered:
        outs = sorted(
            (player for player in sold if str(player["position"]) == position),
            key=_score_sort_key,
        )
        ins = sorted(
            (player for player in bought if str(player["position"]) == position),
            key=_score_sort_key,
        )
        for index in range(max(len(outs), len(ins))):
            rows.append(
                {
                    "sold": outs[index] if index < len(outs) else None,
                    "bought": ins[index] if index < len(ins) else None,
                }
            )
    return rows


def pairs_from_entry_transfers(transfers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep each public-entry transfer as the pair it was recorded as."""
    pairs = []
    for row in transfers:
        pairs.append(
            {
                "gw": int(row["gw"]),
                "sold": {
                    "player_id": str(row["sold_id"]),
                    "position": str(row.get("sold_position") or ""),
                    "name": str(row.get("sold_name") or ""),
                    "team": str(row.get("sold_team") or ""),
                    "score_xp": None,
                },
                "bought": {
                    "player_id": str(row["bought_id"]),
                    "position": str(row.get("bought_position") or ""),
                    "name": str(row.get("bought_name") or ""),
                    "team": str(row.get("bought_team") or ""),
                    "score_xp": None,
                },
            }
        )
    return pairs


def decision_score(
    player_id: str, gw: int, scores: dict[tuple[str, int], float]
) -> float:
    """This week's score, else the latest earlier week, else 0."""
    key = (str(player_id), int(gw))
    if key in scores:
        value = float(scores[key])
        if math.isnan(value):
            return 0.0
        return value
    earlier = [week for pid, week in scores if pid == str(player_id) and week < int(gw)]
    if not earlier:
        return 0.0
    value = float(scores[(str(player_id), max(earlier))])
    if math.isnan(value):
        return 0.0
    return value


def lookup_from_frame(feat: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Early scores first, then the buy-gate ``score_xp`` overwrites that week."""
    scores: dict[tuple[str, int], float] = {}
    stored = feat.attrs.get("early_scores") or ()
    for player_id, gw, score in stored:
        scores[(str(player_id), int(gw))] = float(score)
    if feat.empty:
        return scores
    seen: set[tuple[str, int]] = set()
    for row in feat.itertuples(index=False):
        key = (str(row.player_id), int(row.gw))
        if key in seen:
            continue
        scores[key] = float(row.score_xp)
        seen.add(key)
    return scores


def points_from_roster(roster: pd.DataFrame) -> dict[tuple[str, int], float]:
    points: dict[tuple[str, int], float] = {}
    for row in roster.itertuples(index=False):
        value = row.total_points
        if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
            number = 0.0
        else:
            number = float(value)
        points[(str(row.player_id), int(row.gw))] = number
    return points


def bought_held(player_id: str, later_gws: list[int], squads: dict[int, set[str]]) -> bool:
    """True when the bought player is still owned on every later week of the window.

    A window that ends on the sale week has no later week, so the buy was held.
    """
    for gw in later_gws:
        if str(player_id) not in squads.get(int(gw), set()):
            return False
    return True


def adjacent(prev_gw: int, gw: int) -> bool:
    return int(gw) == int(prev_gw) + 1


def iter_squad_diffs(steps: list[tuple[int, set[str]]]) -> list[dict[str, Any]]:
    """Sales between consecutive gameweeks.

    A hole larger than one week is recorded and is not treated as a transfer.
    """
    ordered = sorted(steps, key=lambda item: int(item[0]))
    diffs: list[dict[str, Any]] = []
    for (prev_gw, prev_ids), (gw, curr_ids) in zip(ordered, ordered[1:]):
        if not adjacent(prev_gw, gw):
            diffs.append({"gw": int(gw), "gap": True, "sold": set(), "bought": set()})
            continue
        sold = set(prev_ids) - set(curr_ids)
        bought = set(curr_ids) - set(prev_ids)
        if sold or bought:
            diffs.append({"gw": int(gw), "gap": False, "sold": sold, "bought": bought})
    return diffs


def score_paired_move(
    *,
    gw: int,
    last_gw: int,
    sold_id: str | None,
    bought_id: str | None,
    points: dict[tuple[str, int], float],
    later_squads: dict[int, set[str]],
    manager_squad: set[str] | None,
) -> dict[str, Any]:
    gws = window_gws(gw, last_gw, HORIZON)
    later = [week for week in gws if week > int(gw)]
    sold_points = None if sold_id is None else window_points(sold_id, gws, points)
    bought_points = None if bought_id is None else window_points(bought_id, gws, points)
    paired = sold_id is not None and bought_id is not None
    gross = pair_gross(sold_points, bought_points) if paired else None
    kept = None
    if sold_id is not None and manager_squad is not None:
        kept = str(sold_id) in manager_squad
    held = None if bought_id is None else bought_held(bought_id, later, later_squads)
    return {
        "gw": int(gw),
        "window_len": len(gws),
        "short": len(gws) < HORIZON,
        "sold_id": sold_id,
        "bought_id": bought_id,
        "sold_points": sold_points,
        "bought_points": bought_points,
        "gross": gross,
        "paired": paired,
        "manager_kept": kept,
        "bought_held": held,
    }


def assign_week_net(
    rows: list[dict[str, Any]],
    sold_points: list[float],
    bought_points: list[float],
    hit_cost: float,
) -> float:
    """Store the week net on the first row so a later sum counts the hit once."""
    net = week_net(sold_points, bought_points, hit_cost)
    for index, row in enumerate(rows):
        row["week_net"] = net if index == 0 else None
        row["hit_on_row"] = float(hit_cost) if index == 0 else 0.0
    return net


def rows_for_week(
    *,
    gw: int,
    last_gw: int,
    pairs: list[dict[str, Any]],
    sold_ids: list[str],
    bought_ids: list[str],
    points: dict[tuple[str, int], float],
    later_squads: dict[int, set[str]],
    manager_squad: set[str] | None,
    hit_cost: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gws = window_gws(gw, last_gw, HORIZON)
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        sold = pair["sold"]
        bought = pair["bought"]
        row = score_paired_move(
            gw=gw,
            last_gw=last_gw,
            sold_id=None if sold is None else str(sold["player_id"]),
            bought_id=None if bought is None else str(bought["player_id"]),
            points=points,
            later_squads=later_squads,
            manager_squad=manager_squad,
        )
        if sold is not None:
            row["sold_name"] = sold.get("name")
            row["sold_position"] = sold.get("position")
            row["sold_score"] = sold.get("score_xp")
            row["sold_team"] = sold.get("team")
        if bought is not None:
            row["bought_name"] = bought.get("name")
            row["bought_position"] = bought.get("position")
            row["bought_score"] = bought.get("score_xp")
            row["bought_team"] = bought.get("team")
        rows.append(row)
    sold_points = [window_points(player_id, gws, points) for player_id in sold_ids]
    bought_points = [window_points(player_id, gws, points) for player_id in bought_ids]
    net = assign_week_net(rows, sold_points, bought_points, hit_cost)
    week = {
        "gw": int(gw),
        "week_net": net,
        "hit_cost": float(hit_cost),
        "window_len": len(gws),
        "short": len(gws) < HORIZON,
        "n_sold": len(sold_ids),
        "n_bought": len(bought_ids),
    }
    return rows, week


def pooled_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarise_sales(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired = [row for row in rows if row.get("paired") and row.get("gross") is not None]
    kept = [row for row in paired if row.get("manager_kept")]
    return {
        "n_sales": len(paired),
        "n_unpaired": sum(1 for row in rows if not row.get("paired")),
        "mean_gross": pooled_mean([float(row["gross"]) for row in paired]),
        "total_gross": float(sum(float(row["gross"]) for row in paired)) if paired else None,
        "n_kept": len(kept),
        "mean_gross_kept": pooled_mean([float(row["gross"]) for row in kept]),
        "n_short": sum(1 for row in paired if row.get("short")),
        "n_churned": sum(1 for row in paired if row.get("bought_held") is False),
    }


def summarise_weeks(weeks: list[dict[str, Any]]) -> dict[str, Any]:
    nets = [float(week["week_net"]) for week in weeks]
    return {
        "n_weeks": len(weeks),
        "mean_week_net": pooled_mean(nets),
        "total_hit": float(sum(float(week["hit_cost"]) for week in weeks)),
    }


def means_by_manager(rows: list[dict[str, Any]]) -> list[float]:
    """One mean per manager who has a paired sale, in first-seen order."""
    order: list[Any] = []
    for row in rows:
        if row.get("paired") and row.get("gross") is not None and row["entry_id"] not in order:
            order.append(row["entry_id"])
    means = []
    for entry_id in order:
        values = [
            float(row["gross"])
            for row in rows
            if row["entry_id"] == entry_id and row.get("paired") and row.get("gross") is not None
        ]
        if values:
            means.append(float(sum(values) / len(values)))
    return means


def consistency(means: list[float]) -> str:
    """Label a group of per-manager means.

    Flat means sit out of both sides. Consistent requires at least
    ``CONSISTENT_MIN`` managers on one side and at least
    ``CONSISTENT_RATIO`` times the other side.
    """
    positive = sum(1 for value in means if value > 0)
    negative = sum(1 for value in means if value < 0)
    larger = max(positive, negative)
    smaller = min(positive, negative)
    if larger < CONSISTENT_MIN:
        return "too small"
    if larger >= CONSISTENT_RATIO * smaller:
        return "consistent sold ahead" if positive > negative else "consistent bought ahead"
    return "mixed"


def select_group(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("group") == group]


REBUILD_CHIPS = frozenset({"wildcard", "free_hit"})


def chip_sale_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wildcard and free hit rebuild a squad. The other weeks spend the transfer bank."""
    paired = [row for row in rows if row.get("paired") and row.get("gross") is not None]
    rebuild = [row for row in paired if row.get("chip") in REBUILD_CHIPS]
    other = [row for row in paired if row.get("chip") not in REBUILD_CHIPS]
    return {
        "n": len(paired),
        "n_rebuild": len(rebuild),
        "mean_rebuild": pooled_mean([float(row["gross"]) for row in rebuild]),
        "n_other": len(other),
        "mean_other": pooled_mean([float(row["gross"]) for row in other]),
    }


def repeated_pairs(rows: list[dict[str, Any]]) -> list[tuple[str, str, int]]:
    """Model sales of the same two names from more than one opening fifteen."""
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        if row.get("group") == "reference" or not row.get("paired"):
            continue
        if row.get("side") not in (None, "model"):
            continue
        key = (str(row.get("sold_name")), str(row.get("bought_name")))
        counts[key] = counts.get(key, 0) + 1
    found = [(sold, bought, count) for (sold, bought), count in counts.items() if count >= 2]
    return sorted(found, key=lambda item: (-item[2], item[0], item[1]))


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "none"
    return f"{value:.{digits}f}"


def _squad_index(squad: pd.DataFrame) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for row in squad.itertuples(index=False):
        player_id = str(row.player_id)
        found[player_id] = {
            "player_id": player_id,
            "position": str(row.position),
            "name": str(row.player_name),
            "team": str(getattr(row, "team", "") or ""),
            "score_xp": float(getattr(row, "score_xp", float("nan"))),
        }
    return found


def _meta(
    player_id: str,
    source: dict[str, dict[str, Any]],
    gw: int,
    scores: dict[tuple[str, int], float],
    teams: dict[tuple[str, int], str],
) -> dict[str, Any]:
    meta = dict(source.get(player_id) or {"player_id": player_id, "position": "UNK", "name": player_id})
    meta["player_id"] = player_id
    meta["position"] = str(meta.get("position") or "UNK")
    meta["score_xp"] = decision_score(player_id, gw, scores)
    meta["team"] = teams.get((player_id, int(gw)), meta.get("team") or "")
    return meta


def _manager_people(entry: dict[str, Any]) -> dict[str, dict[str, str]]:
    people: dict[str, dict[str, str]] = {}
    for week in entry.get("gameweeks") or []:
        for person in list(week.get("xi") or []) + list(week.get("bench") or []):
            people[player_key(person["id"])] = {
                "name": str(person.get("name") or ""),
                "position": str(person.get("position") or ""),
                "team": str(person.get("team") or ""),
            }
    return people


def _manager_squads(entry: dict[str, Any]) -> dict[int, set[str]]:
    squads: dict[int, set[str]] = {}
    for week in entry.get("gameweeks") or []:
        ids = {
            player_key(person["id"])
            for person in list(week.get("xi") or []) + list(week.get("bench") or [])
        }
        squads[int(week["gw"])] = ids
    return squads


def _get_json(client: httpx.Client, url: str) -> Any:
    response = client.get(url)
    if response.status_code == 429:
        time.sleep(2.0)
        response = client.get(url)
    response.raise_for_status()
    return response.json()


def _bootstrap_names(client: httpx.Client) -> dict[int, dict[str, str]]:
    bootstrap = _get_json(client, "https://fantasy.premierleague.com/api/bootstrap-static/")
    teams = {int(row["id"]): row["short_name"] for row in bootstrap["teams"]}
    positions = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
    return {
        int(element["id"]): {
            "web_name": element["web_name"],
            "position": positions[int(element["element_type"])],
            "team": teams[int(element["team"])],
        }
        for element in bootstrap["elements"]
    }


def _fetch_entry(entry_id: int, client: httpx.Client, names: dict[int, dict[str, str]]) -> dict[str, Any]:
    base = ENTRY_URL.format(entry_id=int(entry_id))
    entry = _get_json(client, base)
    history = _get_json(client, base + "history/")
    transfers = _get_json(client, base + "transfers/")
    picks_by_gw = {}
    for row in history.get("current") or []:
        gw = int(row["event"])
        picks_by_gw[gw] = _get_json(client, base + f"event/{gw}/picks/")
    return build_entry(entry, history, transfers, picks_by_gw, names)


def _ensure_entry(
    entry_id: int, client: httpx.Client, names: dict[int, dict[str, str]] | None
) -> dict[str, Any]:
    path = ENTRY_DIR / f"{int(entry_id)}.json"
    if path.exists():
        return load_entry(entry_id)
    if names is None:
        raise RuntimeError("bootstrap names are required for a new entry")
    payload = _fetch_entry(entry_id, client, names)
    save_entry(payload)
    time.sleep(0.15)
    return payload


def _climb_one(
    spec: dict[str, Any],
    feat: pd.DataFrame,
    roster: pd.DataFrame,
    scores: dict[tuple[str, int], float],
    points: dict[tuple[str, int], float],
    teams: dict[tuple[str, int], str],
    client: httpx.Client,
    names: dict[int, dict[str, str]] | None,
) -> dict[str, Any]:
    entry_id = int(spec["entry_id"])
    base = {
        "entry_id": entry_id,
        "label": spec["label"],
        "group": spec["group"],
        "rank_sort": spec.get("rank_sort"),
        "prior_ranks": spec.get("prior_ranks"),
        "points_at_lock": spec.get("points_at_lock"),
    }
    try:
        entry = _ensure_entry(entry_id, client, names)
        opening = _opening_state(entry, roster)
    except (EntryError, RuntimeError, ValueError, KeyError, httpx.HTTPError) as exc:
        return {**base, "error": str(exc), "model_rows": [], "own_rows": [], "model_weeks": [], "own_weeks": []}

    last_gw = min(LAST_GW, int(entry.get("played_through") or LAST_GW))
    gws = [int(gw) for gw in GWS if int(gw) <= last_gw]
    trace: list[dict[str, Any]] = []
    weekly = run_ft_season(
        feat,
        {"xp": "score_xp"},
        gws,
        roster=roster,
        trace=trace,
        opening=opening,
    )
    if weekly.empty or not trace:
        return {
            **base,
            "error": "climb returned no weeks",
            "team_name": entry.get("team_name"),
            "season_points": entry.get("points"),
            "model_rows": [],
            "own_rows": [],
            "model_weeks": [],
            "own_weeks": [],
        }

    by_gw = weekly.set_index("gw")
    steps = [(int(step["gw"]), set(step["squad"]["player_id"].astype(str))) for step in trace]
    squads = {int(step["gw"]): step["squad"] for step in trace}
    model_squads = {gw: ids for gw, ids in steps}
    manager_squads = _manager_squads(entry)
    people = _manager_people(entry)
    model_rows: list[dict[str, Any]] = []
    model_weeks: list[dict[str, Any]] = []
    n_gaps = 0
    for diff in iter_squad_diffs(steps):
        if diff["gap"]:
            n_gaps += 1
            continue
        if not diff["sold"] and not diff["bought"]:
            continue
        gw = int(diff["gw"])
        if gw not in by_gw.index:
            continue
        hit_cell = by_gw.loc[gw, "hit_cost"]
        transfer_cell = by_gw.loc[gw, "n_transfers"]
        if isinstance(hit_cell, pd.Series):
            hit_cell = hit_cell.iloc[0]
            transfer_cell = transfer_cell.iloc[0]
        prev_gw = gw - 1
        sold_meta = [
            _meta(player_id, _squad_index(squads[prev_gw]), gw, scores, teams)
            for player_id in sorted(diff["sold"])
        ]
        bought_meta = [
            _meta(player_id, _squad_index(squads[gw]), gw, scores, teams)
            for player_id in sorted(diff["bought"])
        ]
        pairs = pair_within_position(sold_meta, bought_meta)
        later = {week: model_squads.get(week, set()) for week in window_gws(gw, last_gw)}
        rows, week = rows_for_week(
            gw=gw,
            last_gw=last_gw,
            pairs=pairs,
            sold_ids=[player["player_id"] for player in sold_meta],
            bought_ids=[player["player_id"] for player in bought_meta],
            points=points,
            later_squads=later,
            manager_squad=manager_squads.get(gw, set()),
            hit_cost=float(hit_cell),
        )
        n_transfers = int(transfer_cell)
        for row in rows:
            row.update(base)
            row["side"] = "model"
            row["n_transfers"] = n_transfers
        week.update(
            {
                "entry_id": entry_id,
                "group": spec["group"],
                "side": "model",
                "n_transfers": n_transfers,
                "transfer_mismatch": n_transfers != len(sold_meta),
            }
        )
        model_rows.extend(rows)
        model_weeks.append(week)

    prepared = []
    for transfer in entry.get("transfers") or []:
        gw = int(transfer["gw"])
        if gw < 1 or gw > last_gw:
            continue
        sold_key = player_key(transfer["out_id"])
        bought_key = player_key(transfer["in_id"])
        prepared.append(
            {
                "gw": gw,
                "sold_id": sold_key,
                "bought_id": bought_key,
                "sold_name": transfer.get("out"),
                "bought_name": transfer.get("in"),
                "sold_position": (people.get(sold_key) or {}).get("position", ""),
                "bought_position": (people.get(bought_key) or {}).get("position", ""),
                "sold_team": teams.get((sold_key, gw), (people.get(sold_key) or {}).get("team", "")),
                "bought_team": teams.get((bought_key, gw), (people.get(bought_key) or {}).get("team", "")),
            }
        )
    hit_by_gw = {int(week["gw"]): float(week["transfer_cost"]) for week in entry.get("gameweeks") or []}
    chip_by_gw = {int(week["gw"]): week.get("chip") for week in entry.get("gameweeks") or []}
    own_rows: list[dict[str, Any]] = []
    own_weeks: list[dict[str, Any]] = []
    by_sale_gw: dict[int, list[dict[str, Any]]] = {}
    for pair in pairs_from_entry_transfers(prepared):
        by_sale_gw.setdefault(int(pair["gw"]), []).append(pair)
    for gw in sorted(by_sale_gw):
        pairs = by_sale_gw[gw]
        later = {week: manager_squads.get(week, set()) for week in window_gws(gw, last_gw)}
        rows, week = rows_for_week(
            gw=gw,
            last_gw=last_gw,
            pairs=pairs,
            sold_ids=[pair["sold"]["player_id"] for pair in pairs],
            bought_ids=[pair["bought"]["player_id"] for pair in pairs],
            points=points,
            later_squads=later,
            manager_squad=None,
            hit_cost=hit_by_gw.get(gw, 0.0),
        )
        for row in rows:
            row.update(base)
            row["side"] = "manager"
            row["chip"] = chip_by_gw.get(gw)
        week.update(
            {
                "entry_id": entry_id,
                "group": spec["group"],
                "side": "manager",
                "chip": chip_by_gw.get(gw),
            }
        )
        own_rows.extend(rows)
        own_weeks.append(week)

    model_points = float(weekly["xi_points_cap"].sum())
    model_hits = float(weekly["hit_cost"].sum())
    return {
        **base,
        "error": None,
        "team_name": entry.get("team_name"),
        "season_points": float(entry.get("points") or 0),
        "overall_rank": entry.get("overall_rank"),
        "model_points": model_points,
        "model_hits": model_hits,
        "n_gaps": n_gaps,
        "chips": [row.get("chip") for row in entry.get("chips_played") or []],
        "model_rows": model_rows,
        "own_rows": own_rows,
        "model_weeks": model_weeks,
        "own_weeks": own_weeks,
    }


def _group_block(
    group: str,
    model_rows: list[dict[str, Any]],
    own_rows: list[dict[str, Any]],
    model_weeks: list[dict[str, Any]],
    own_weeks: list[dict[str, Any]],
) -> dict[str, Any]:
    model = summarise_sales(select_group(model_rows, group))
    own = summarise_sales(select_group(own_rows, group))
    model.update(summarise_weeks(select_group(model_weeks, group)))
    own_week = summarise_weeks(select_group(own_weeks, group))
    model["own_n_sales"] = own["n_sales"]
    model["own_mean_gross"] = own["mean_gross"]
    model["own_n_weeks"] = own_week["n_weeks"]
    model["own_mean_week_net"] = own_week["mean_week_net"]
    model["own_total_hit"] = own_week["total_hit"]
    model["model_sign"] = consistency(means_by_manager(select_group(model_rows, group)))
    model["own_sign"] = consistency(means_by_manager(select_group(own_rows, group)))
    return model


def write_report(
    managers: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    own_rows: list[dict[str, Any]],
    model_weeks: list[dict[str, Any]],
    blocks: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Transfer gaps, Gameweeks 1–5",
        "",
        "The cohort was locked before any of these squads were opened. "
        "Veterans are the first seven names on the FPL Form multi-season ranking "
        "with at least two finishes inside the top 10,000 across 2022/23–2025/26. "
        f"Rank slots are overall classic-league rank_sort 100, 500, 1000, 5000, 10000, 25000, and 50000, taken on {LOCK_DATE}. "
        "ojaminFC is a reference row and is left out of both means.",
        "",
        "Each climb starts from that manager's Gameweek 1 fifteen, with an empty chip map, "
        "on the published path (opening horizon and the early score). "
        "A sale is a player in one week's squad and absent from the next. "
        "Pairs stay inside a position, highest decision-week score with highest. "
        "A move that crosses position is counted and stays unpaired.",
        "",
        "The window is the sale week and the next two, and it stops at Gameweek 5. "
        "A shorter window is flagged and left short. "
        "Gross is the sold player's points minus the bought player's points. "
        "The week's hit is added once, on the week. "
        "Positive means the player sold scored more than the player bought.",
        "",
        "The same window is counted on the managers' own transfers. "
        "Their season points include chips and sit beside the transfer count as a yardstick. "
        f"A sign is consistent when at least {CONSISTENT_MIN} managers with a paired sale share it "
        f"and that side is at least {CONSISTENT_RATIO} times the other. "
        "The same sign on the model's sales, with the managers' own sales failing to share it, "
        "is the systematic reading. A split, or the same sign on their own sales, "
        "is the variance of a three-week window. Five weeks leave the score as it is.",
        "",
        "| group | ran | skipped | model sales | mean gross | kept | mean kept | mean week net | own sales | mean own gross | model sign | own sign |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for group in ("veteran", "rank"):
        block = blocks[group]
        ran = sum(1 for row in managers if row["group"] == group and not row.get("error"))
        skipped = sum(1 for row in managers if row["group"] == group and row.get("error"))
        lines.append(
            f"| {group} | {ran} | {skipped} | {block['n_sales']} | {_fmt(block['mean_gross'])} "
            f"| {block['n_kept']} | {_fmt(block['mean_gross_kept'])} | {_fmt(block['mean_week_net'])} "
            f"| {block['own_n_sales']} | {_fmt(block['own_mean_gross'])} | {block['model_sign']} | {block['own_sign']} |"
        )
    lines.extend(
        [
            "",
            "Mean gross gives every paired sale the same weight. "
            "The sign gives every manager with a paired sale the same weight. "
            "Kept means the sold player was still in that manager's fifteen in the sale week. "
            "Week net adds that week's hit once.",
            "",
        ]
    )
    for group in ("veteran", "rank"):
        split = chip_sale_split(select_group(own_rows, group))
        lines.append(
            f"{group.capitalize()}: {split['n_rebuild']} of {split['n']} own sales are a wildcard or a free hit "
            f"(mean gross {_fmt(split['mean_rebuild'])}). "
            f"The other {split['n_other']} have mean gross {_fmt(split['mean_other'])}. "
            "The group mean above keeps both."
        )
    repeats = repeated_pairs(model_rows)
    if repeats:
        listed = ", ".join(f"{sold} to {bought} ({count})" for sold, bought, count in repeats)
        lines.append(
            "The same model sale appears from more than one opening fifteen: " + listed + "."
        )
    lines.extend(
        [
            "",
            "| label | group | season points | model points | model hits | sales | mean gross | kept | mean kept | own sales | mean own | chips |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for manager in managers:
        if manager.get("error"):
            continue
        model = [row for row in model_rows if row["entry_id"] == manager["entry_id"]]
        own = [row for row in own_rows if row["entry_id"] == manager["entry_id"]]
        model_sum = summarise_sales(model)
        own_sum = summarise_sales(own)
        chips = ", ".join(chip for chip in (manager.get("chips") or []) if chip) or "none"
        lines.append(
            f"| {manager['label']} | {manager['group']} | {_fmt(manager.get('season_points'), 0)} "
            f"| {_fmt(manager.get('model_points'), 0)} | {_fmt(manager.get('model_hits'), 0)} "
            f"| {model_sum['n_sales']} | {_fmt(model_sum['mean_gross'])} | {model_sum['n_kept']} "
            f"| {_fmt(model_sum['mean_gross_kept'])} | {own_sum['n_sales']} | {_fmt(own_sum['mean_gross'])} | {chips} |"
        )
    gaps = [manager for manager in managers if manager.get("n_gaps")]
    if gaps:
        listed = ", ".join(f"{manager['label']} ({manager['n_gaps']})" for manager in gaps)
        lines.extend(["", f"A hole in the climb, left out of the sales: {listed}."])
    mismatches = [week for week in model_weeks if week.get("transfer_mismatch")]
    if mismatches:
        listed = ", ".join(f"{week['entry_id']} GW{week['gw']}" for week in mismatches)
        lines.extend(["", f"Squad change and transfer count disagree: {listed}."])
    skipped = [manager for manager in managers if manager.get("error")]
    if skipped:
        lines.extend(["", "Skipped:", ""])
        for manager in skipped:
            lines.append(f"- {manager['label']} ({manager['entry_id']}): {manager['error']}")
    reference = next(manager for manager in managers if manager["group"] == "reference")
    if reference.get("error"):
        lines.extend(["", f"ojaminFC was skipped: {reference['error']}"])
    else:
        lines.extend(
            [
                "",
                f"ojaminFC season points {_fmt(reference.get('season_points'), 0)}, "
                f"model {_fmt(reference.get('model_points'), 0)} "
                f"with {_fmt(reference.get('model_hits'), 0)} hit points. "
                f"The published path from this opening fifteen scored {PUBLISHED_REFERENCE_POINTS:.0f} on the last run.",
            ]
        )
    kept_rows = [
        row
        for row in model_rows
        if row.get("paired") and row.get("manager_kept") and row.get("group") != "reference"
    ]
    lines.extend(["", "## Sales the manager kept", ""])
    if not kept_rows:
        lines.append("No paired model sale was still in the manager's fifteen.")
    else:
        lines.extend(
            [
                "| group | label | gw | window | sold | bought | gross | buy still owned |",
                "|---|---|---:|---:|---|---|---:|---|",
            ]
        )
        for row in kept_rows:
            held = "yes" if row.get("bought_held") else "no"
            lines.append(
                f"| {row['group']} | {row['label']} | {row['gw']} | {row['window_len']} "
                f"| {row.get('sold_name')} | {row.get('bought_name')} | {_fmt(row.get('gross'), 0)} | {held} |"
            )
    lines.extend(
        [
            "",
            "Each pair, including unpaired moves and the reference row, is in "
            "`data/processed/stage_46_transfer_gap.csv`. "
            "One row per sale week, with the hit counted once, is in "
            "`data/processed/stage_46_transfer_weeks.csv`.",
            "",
            "Gemini kept this count "
            "([transfer gap](bc-b57f0f87-87e5-5037-b65f-a697e55bf2a9)). "
            "Both groups are bought ahead on the model's sales and on the managers' own sales, "
            "which is the variance reading under the rule above. "
            "The reference row is the other sign and stays out of the means. "
            "The score stays `score_xp`.",
            "",
        ]
    )
    return "\n".join(lines)


def _flatten(rows: list[dict[str, Any]]) -> pd.DataFrame:
    fields = [
        "group",
        "entry_id",
        "label",
        "side",
        "gw",
        "window_len",
        "short",
        "paired",
        "sold_id",
        "sold_name",
        "sold_position",
        "sold_team",
        "sold_score",
        "sold_points",
        "bought_id",
        "bought_name",
        "bought_position",
        "bought_team",
        "bought_score",
        "bought_points",
        "gross",
        "manager_kept",
        "bought_held",
        "hit_on_row",
        "week_net",
        "chip",
    ]
    return pd.DataFrame(rows, columns=fields)


def run() -> dict[str, Any]:
    feat, roster, _info = build_frames()
    scores = lookup_from_frame(feat)
    points = points_from_roster(roster)
    teams = {
        (str(row.player_id), int(row.gw)): str(row.team)
        for row in roster.itertuples(index=False)
    }
    needs_fetch = any(not (ENTRY_DIR / f"{int(spec['entry_id'])}.json").exists() for spec in COHORT)
    names = None
    with httpx.Client(timeout=40.0, headers={"User-Agent": "fpl-engine-diagnostic"}) as client:
        if needs_fetch:
            names = _bootstrap_names(client)
        results = []
        for spec in COHORT:
            print(f"climb {spec['group']} {spec['entry_id']} {spec['label']}", flush=True)
            results.append(_climb_one(spec, feat, roster, scores, points, teams, client, names))

    model_rows: list[dict[str, Any]] = []
    own_rows: list[dict[str, Any]] = []
    model_weeks: list[dict[str, Any]] = []
    own_weeks: list[dict[str, Any]] = []
    for result in results:
        model_rows.extend(result.get("model_rows") or [])
        own_rows.extend(result.get("own_rows") or [])
        model_weeks.extend(result.get("model_weeks") or [])
        own_weeks.extend(result.get("own_weeks") or [])
    blocks = {
        group: _group_block(group, model_rows, own_rows, model_weeks, own_weeks)
        for group in ("veteran", "rank")
    }
    text = write_report(results, model_rows, own_rows, model_weeks, blocks)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "stage_46_transfer_gap.md").write_text(text, encoding="utf-8")
    _flatten(model_rows + own_rows).to_csv(PROCESSED / "stage_46_transfer_gap.csv", index=False)
    pd.DataFrame(model_weeks + own_weeks).to_csv(PROCESSED / "stage_46_transfer_weeks.csv", index=False)
    print(text, flush=True)
    return {"managers": results, "blocks": blocks}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
