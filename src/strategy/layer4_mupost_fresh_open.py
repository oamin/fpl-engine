"""Fresh opening 15 for L3/L4 climbs — crowd template, not μ_post wildcard.

Locked comparison: same transfer policies as ``strategy_layer4_mupost_priced``,
but both arms start from the Gameweek-1 crowd ``template`` fifteen instead of
a shared L2 wildcard rebuild. Isolates whether L4−L3 is an artifact of opening
already on the μ_post ridge.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, normalize_position
from src.strategy.layer3_week import HeldState

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OPENINGS = PROCESSED / "crowd_openings.csv"

SQUAD_LABEL = "template"


def _crowd_season(season: str) -> str:
    return season.replace("-", "_")


def _pid(season: str, element: int) -> str:
    return f"{season}:{int(element)}"


def load_template_opening(season: str, *, squad: str = SQUAD_LABEL) -> pd.DataFrame:
    """Fifteen rows from ``crowd_openings.csv`` for one season/squad."""
    frame = pd.read_csv(OPENINGS)
    key = _crowd_season(season)
    block = frame.loc[(frame["season"] == key) & (frame["squad"] == squad)].copy()
    if len(block) != 15:
        raise ValueError(f"{season}/{squad}: expected 15 rows, got {len(block)}")
    block["element"] = pd.to_numeric(block["element"], errors="coerce").astype(int)
    block["value"] = pd.to_numeric(block["value"], errors="coerce").astype(int)
    block["player_id"] = block["element"].map(lambda e: _pid(season, int(e)))
    block["position"] = block["position"].map(normalize_position)
    block["purchase"] = block["value"].astype(int)
    spent = int(block["purchase"].sum())
    if spent > BUDGET_TENTHS:
        raise ValueError(f"{season}/{squad} costs {spent} > {BUDGET_TENTHS}")
    return block.reset_index(drop=True)


def _last_price(store_prices: pd.DataFrame, element: int, gw: int) -> int | None:
    hist = store_prices.loc[
        (store_prices["element"] == int(element)) & (store_prices["gw"] <= int(gw))
    ]
    if hist.empty:
        return None
    return int(hist.sort_values("gw").iloc[-1]["value"])


def inject_template_into_pool(
    pool: pd.DataFrame,
    opening: pd.DataFrame,
    *,
    season: str,
    gw: int,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure every template player is in the decision pool (stub if absent)."""
    work = pool.copy()
    have = set(work["player_id"].astype(str))
    stubs: list[dict] = []
    for _, row in opening.iterrows():
        pid = str(row["player_id"])
        if pid in have:
            continue
        el = int(row["element"])
        val = _last_price(prices, el, gw)
        if val is None:
            val = int(row["purchase"])
        stubs.append(
            {
                "element": el,
                "player_id": pid,
                "name": str(row.get("name", pid)),
                "position": normalize_position(str(row["position"])),
                "club": str(row.get("team", "UNK")),
                "team": str(row.get("team", "UNK")),
                "value": int(val),
                "score_xp_v2": 0.0,
                "xp_se": 0.0,
                "total_points": 0.0,
                "mu": 0.0,
                "mu_post": 0.0,
            }
        )
    if stubs:
        work = pd.concat([work, pd.DataFrame(stubs)], ignore_index=True)
    return work


def held_from_template(opening: pd.DataFrame, *, ft: int = 0) -> HeldState:
    """HeldState with GW1 purchase prices and leftover bank. Default ft=0 forces hold."""
    ids = tuple(str(p) for p in opening["player_id"].tolist())
    purchase = {str(r["player_id"]): int(r["purchase"]) for _, r in opening.iterrows()}
    bank = int(BUDGET_TENTHS - sum(purchase.values()))
    return HeldState(ids, purchase, bank=max(0, bank), ft=int(ft))
