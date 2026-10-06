"""Were Groß and João Pedro offered to the Gameweek 3 wildcard rebuild?

The four managers who were forced onto a Gameweek 3 wildcard are replayed
through Gameweek 4. The score, the hold, and the chip margins stay as they
are. His squad is not copied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.live.benchmark import GWS, build_frames, player_key
from src.live.entry import load_entry
from src.models.chip_lead import _lowest, _pool_index, buy_tag, place_signing
from src.models.cohort_carry import cohort_specs
from src.models.forced_chips import chips_from_entry, legal_calendar
from src.models.friend_start import one_week
from src.models.open_horizon import attach_opening_horizon
from src.models.reset_gap import _early, merged_clubs, pre_deadline
from src.models.season_climb_ft import _gw_pool
from src.rules.fpl_2026 import ChipWallet

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "data" / "processed" / "gw4_stack.csv"
OUT_REPORT = ROOT / "reports" / "gw4_stack.md"
WEEKS = (1, 2, 3, 4)
WILDCARD_GW = 3
HAUL_GW = 4
# The shared Gameweek 4 haul. Element ids from the 2026/27 sheet.
TARGETS = (
    (124, "Groß"),
    (165, "João Pedro"),
    (115, "De Cuyper"),
)
WANTED = {2076855, 31365, 32058, 534464}


def _names(roster: pd.DataFrame) -> dict[str, str]:
    frame = roster.drop_duplicates("player_id")
    if "player_name" not in frame.columns:
        return {}
    return {
        str(row.player_id): str(row.player_name)
        for row in frame.itertuples(index=False)
    }


def _number(row: Any, *names: str) -> float | None:
    for name in names:
        raw = getattr(row, name, None)
        if raw is None or raw != raw:
            continue
        return float(raw)
    return None


def inspect_week(
    label: str,
    entry_id: int,
    gw: int,
    week: dict[str, Any],
    pool: pd.DataFrame,
    names: dict[str, str],
    points: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    """One row per haul player. The rebuild is the squad the chip bought."""
    pool_by = _pool_index(pool)
    rebuilt = [str(pid) for pid in week["rebuilt_ids"]]
    purchase = {str(pid): int(price) for pid, price in week["purchase"].items()}
    bank = int(week["rebuilt_bank"])
    xi = set(week["model_intended_ids"])
    owned = set(week["model_ids"])
    rows: list[dict[str, Any]] = []
    for element, fallback in TARGETS:
        pid = player_key(element)
        src = pool_by.get(pid)
        tag = buy_tag(pid, pool_by)
        if src is None or tag != "eligible":
            place, gap, block, lowest = "out", None, tag, ""
        else:
            signed = place_signing(pid, pool_by, rebuilt, bank, purchase)
            place = str(signed["place"])
            gap = signed["gap"]
            block = str(signed["block"])
            lowest = ""
            if place == "out":
                low = _lowest(rebuilt, str(getattr(src, "position")), pool_by)
                lowest = names.get(low, low)
        rows.append(
            {
                "entry_id": entry_id,
                "label": label,
                "gw": gw,
                "player": fallback if pid not in names else names.get(pid, fallback),
                "player_id": pid,
                "tag": tag,
                "place": place if gw == WILDCARD_GW else ("in" if pid in owned else "out"),
                "block": block if gw == WILDCARD_GW else "",
                "gap": gap if gw == WILDCARD_GW else None,
                "score_xp": None if src is None else _number(src, "score_xp"),
                "n_prior": None if src is None else _number(src, "n_prior"),
                "xmi": None if src is None else _number(src, "xmi", "exp_minutes"),
                "value": None if src is None else _number(src, "value"),
                "lowest": lowest,
                "in_xi": pid in xi,
                "points": points.get((pid, gw)),
            }
        )
    return rows


def run() -> list[dict[str, Any]]:
    """Replay the four wildcard carries and record the three haul players."""
    specs = [row for row in cohort_specs() if int(row["entry_id"]) in WANTED]
    if len(specs) != 4:
        raise RuntimeError("the four wildcard managers are not in the cohort")
    feat, roster, _info = build_frames()
    clubs = merged_clubs(roster)
    horizon_scores = attach_opening_horizon(feat)
    if horizon_scores is None:
        raise RuntimeError("the opening horizon is missing")
    early = _early(feat)
    names = _names(roster)
    roster_by_gw = {
        int(gw): set(block["player_id"].astype(str)) for gw, block in roster.groupby("gw")
    }
    points = {
        (str(row.player_id), int(row.gw)): float(row.total_points)
        for row in roster.itertuples(index=False)
        if int(row.gw) in WEEKS
    }
    found: list[dict[str, Any]] = []
    for spec in specs:
        entry = load_entry(int(spec["entry_id"]))
        kept, _dropped = legal_calendar(chips_from_entry(entry, GWS))
        if kept.get(WILDCARD_GW) != "wildcard":
            raise RuntimeError(f"{spec['label']} did not wildcard in Gameweek 3")
        state = pre_deadline(entry, roster, 1)
        wallet = ChipWallet()
        print(f"stack {spec['entry_id']} {spec['label']}", flush=True)
        for gw in WEEKS:
            owned = state.ids()
            pool = _gw_pool(feat, roster, gw, owned, early)
            week, state = one_week(
                feat,
                roster,
                entry,
                gw,
                state,
                wallet,
                clubs=clubs,
                roster_by_gw=roster_by_gw,
                horizon_scores=horizon_scores,
                forced_chip=kept.get(gw),
            )
            if gw in (WILDCARD_GW, HAUL_GW):
                found.extend(
                    inspect_week(
                        str(spec["label"]),
                        int(spec["entry_id"]),
                        gw,
                        week,
                        pool,
                        names,
                        points,
                    )
                )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(found).to_csv(OUT_CSV, index=False)
    text = render(found)
    OUT_REPORT.write_text(text, encoding="utf-8")
    print(text, flush=True)
    return found


def _num(value: Any) -> str:
    if value is None or value != value:
        return ""
    text = f"{float(value):.2f}"
    return text[:-3] if text.endswith(".00") else text.rstrip("0").rstrip(".")


def render(rows: list[dict[str, Any]]) -> str:
    """The decision week and the haul week. No constant moves."""
    lines = [
        "# Gameweek 4 stack",
        "",
        "Groß, João Pedro, and De Cuyper are the shared Gameweek 4 haul. The four managers who wildcarded in Gameweek 3 under the forced calendar are replayed. The rebuild is the model's. Eligible means three prior appearances and expected minutes of at least 45. A negative gap means the cheapest player the rebuild kept at that position had the higher score. Price means selling that player plus the bank still could not buy him. Neither means the score left him out with the money and the club slot available.",
        "",
        "| Manager | Week | Player | Pool | Place | Block | Score gap | Score | Appearances | Minutes | Price | Instead | Started | Points |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['gw']} | {row['player']} | {row['tag']} | "
            f"{row['place']} | {row['block']} | {_num(row['gap'])} | {_num(row['score_xp'])} | "
            f"{_num(row['n_prior'])} | {_num(row['xmi'])} | {_num(row['value'])} | "
            f"{row['lowest']} | {'yes' if row['in_xi'] else 'no'} | {_num(row['points'])} |"
        )
    lines += [
        "",
        "The margins stay 12 and 16. `score_xp` is unchanged.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    run()
