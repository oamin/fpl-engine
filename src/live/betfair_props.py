"""Store and apply valuable Betfair props for live ``score_xp`` / ``forecast_xp``.

Gemini 2026-10-08 (bc-e75c8209):
- TO_SCORE (anytime) enters imminent-week ``score_xp`` under the team-λ cap.
- BTTS / clean sheet: diagnostic JSON only.
- FIRST_GOAL_SCORER, correct score, half-time: not used.
- Season outrights → shrunk strength pots for unpriced ``forecast_xp`` weeks.
"""

from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from src.live.betfair import (
    EPL_COMPETITION_ID,
    EVENT_TYPE_SOCCER,
    BetfairClient,
    classify_outright,
    expected_rank,
    mid_implied,
    best_prices,
    runner_mids,
    simplex,
    strength_from_rank,
)
from src.models.forecast_xp import strength_match_pots
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "data" / "scratch" / "betfair"
PREDICTIONS = ROOT / "data" / "predictions" / "2026-27"
LIVE_DIR = ROOT / "data" / "live"
MIN_RUNNER_MATCHED = 250.0
MAX_RUNNER_SPREAD = 0.35
# Club absent from outrights (often a thin promoted side): bottom table, not
# mid-table 0.0 (Gemini 2026-10-08 forecast-horizon review).
MISSING_OUTRIGHT_RANK = 18.5
MISSING_OUTRIGHT_STRENGTH = strength_from_rank(MISSING_OUTRIGHT_RANK)
# Require a Betfair-specific file so a plain historical gw_lines.csv is not
# mistaken for an Exchange pull (data/live always has a lines file).
ARTIFACT_MARKERS = (
    "betfair_to_score.json",
    "outrights_ranks.json",
    "betfair_meta.json",
)


def discover_betfair_artifacts(gw: int) -> Path | None:
    """Newest directory that holds Betfair derived files for this gameweek.

    Order: ``BETFAIR_ARTIFACTS_DIR``, then ``gwNN/betfair_t1`` when that
    folder has derived files, then the newest other ``betfair_*`` directory,
    then ``data/live`` if it already contains Betfair artifacts.
    """
    import os

    forced = os.environ.get("BETFAIR_ARTIFACTS_DIR", "").strip()
    if forced:
        path = Path(forced)
        if path.is_dir() and _has_artifacts(path):
            return path
    folder = PREDICTIONS / f"gw{int(gw):02d}"
    if folder.is_dir():
        t1 = folder / "betfair_t1"
        if t1.is_dir() and _has_artifacts(t1):
            return t1
        candidates = sorted(
            [p for p in folder.glob("betfair_*") if p.is_dir() and _has_artifacts(p)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    if _has_artifacts(LIVE_DIR):
        return LIVE_DIR
    return None


def _has_artifacts(path: Path) -> bool:
    return any((path / name).is_file() for name in ARTIFACT_MARKERS)


def _frozen_lines() -> Path:
    return (LIVE_DIR / "gw_lines.csv").resolve()


def betfair_gw_lines(gw: int) -> Path | None:
    """Exchange ``gw_lines.csv`` for this gameweek.

    A path that resolves to the frozen holdout file is not a live book.
    """
    artifacts = discover_betfair_artifacts(int(gw))
    if artifacts is None:
        return None
    candidate = artifacts / "gw_lines.csv"
    if not candidate.is_file():
        return None
    if candidate.resolve() == _frozen_lines():
        return None
    return candidate


def resolve_live_book(gw: int, live_path: Path | None) -> Path | None:
    """Use a caller file when it exists and is not the frozen slate.

    The default, and any path that is the frozen file, is the T−1 Exchange
    sheet when that file exists, otherwise the newest Betfair slate. A
    missing caller file stays missing.
    """
    if live_path is not None:
        candidate = Path(live_path)
        if candidate.resolve() != _frozen_lines():
            return candidate if candidate.is_file() else None
    from src.live.t1_inputs import exchange_sheet

    sheet = exchange_sheet(int(gw))
    if sheet.is_file() and sheet.resolve() != _frozen_lines():
        return sheet
    return betfair_gw_lines(gw)


def poisson_mean(prices: list[float]) -> tuple[float, float]:
    """Mean of 1/price → p, then μ = -ln(1-p). Clips p to [0.01, 0.85]."""
    implied = [1.0 / p for p in prices if p and p > 1.0]
    if not implied:
        return 0.0, 0.0
    p = sum(implied) / len(implied)
    p = min(max(p, 0.01), 0.85)
    return p, float(-math.log(1.0 - p))


def team_goal_rates(
    rates: Mapping[str, float],
    *,
    lam: float,
    minutes: Mapping[str, float],
) -> dict[str, float]:
    """Minutes-scale each anytime rate. Cap only when those rates exceed team λ.

    A priced player's own rate is his allocation. Other players' historical
    shares do not reduce it. Unpriced players stay on ``share_xG × λ``.
    """
    team = float(lam)
    if not math.isfinite(team) or team <= 0.0:
        return {}
    priced: dict[str, float] = {}
    for pid, mu_raw in rates.items():
        mu = float(mu_raw)
        if not math.isfinite(mu) or mu < 0.0:
            continue
        xmi = float(minutes.get(pid, 90.0))
        if not math.isfinite(xmi):
            xmi = 0.0
        priced[str(pid)] = mu * max(xmi, 0.0) / 90.0
    if not priced:
        return {}
    total = sum(priced.values())
    scale = 1.0 if total <= team else team / total
    return {pid: mu * scale for pid, mu in priced.items()}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fetch_to_score(
    client: BetfairClient,
    *,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Pull TO_SCORE markets; write raw under scratch and a derived summary."""
    catalogue = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "competitionIds": [EPL_COMPETITION_ID],
            "marketTypeCodes": ["TO_SCORE"],
        },
        max_results=50,
    )
    books = client.list_market_book([str(m["marketId"]) for m in catalogue])
    book_by = {str(b["marketId"]): b for b in books}
    _write(SCRATCH / "raw_to_score_catalogue.json", catalogue)
    _write(SCRATCH / "raw_to_score_books.json", books)

    rows: list[dict[str, Any]] = []
    for market in catalogue:
        book = book_by.get(str(market["marketId"]))
        if book is None:
            continue
        event = market.get("event") or {}
        names = {
            int(r["selectionId"]): str(r.get("runnerName") or r["selectionId"])
            for r in market.get("runners") or []
        }
        for runner in book.get("runners") or []:
            sid = int(runner["selectionId"])
            back, lay = best_prices(runner)
            mid = mid_implied(back, lay, max_rel_spread=MAX_RUNNER_SPREAD)
            matched = float(runner.get("totalMatched") or 0.0)
            if mid is None or matched < MIN_RUNNER_MATCHED:
                continue
            p = min(max(mid, 0.01), 0.85)
            mu = float(-math.log(1.0 - p))
            rows.append(
                {
                    "event": event.get("name"),
                    "start": market.get("marketStartTime") or event.get("openDate"),
                    "runner": names.get(sid, str(sid)),
                    "selectionId": sid,
                    "p_mid": p,
                    "mu_raw": mu,
                    "matched": matched,
                    "back": back,
                    "lay": lay,
                    "marketId": market["marketId"],
                }
            )
    _write(out_dir / "betfair_to_score.json", rows)
    return rows


def fetch_btts_diagnostics(
    client: BetfairClient,
    *,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """BTTS mids for diagnostic JSON only (not player scoring)."""
    catalogue = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "competitionIds": [EPL_COMPETITION_ID],
            "marketTypeCodes": ["BOTH_TEAMS_TO_SCORE"],
        },
        max_results=50,
    )
    books = client.list_market_book([str(m["marketId"]) for m in catalogue])
    book_by = {str(b["marketId"]): b for b in books}
    _write(SCRATCH / "raw_diagnostics_btts.json", {"catalogue": catalogue, "books": books})
    rows = []
    for market in catalogue:
        book = book_by.get(str(market["marketId"]))
        if book is None:
            continue
        mids = runner_mids(book, market)
        probs = simplex(mids, mass=1.0)
        event = market.get("event") or {}
        rows.append(
            {
                "event": event.get("name"),
                "start": market.get("marketStartTime") or event.get("openDate"),
                "p_yes": probs.get("Yes") or probs.get("yes"),
                "p_no": probs.get("No") or probs.get("no"),
                "matched": float(book.get("totalMatched") or 0.0),
            }
        )
    _write(out_dir / "diagnostics_btts_cs.json", rows)
    return rows


def fetch_outrights(
    client: BetfairClient,
    *,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Winner / top-6 / relegation → expected rank + strength for forecast_xp."""
    catalogue = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "competitionIds": [EPL_COMPETITION_ID],
            "marketTypeCodes": ["WINNER", "SPECIAL", "OTHER_PLACE"],
        },
        max_results=200,
    )
    extra = client.list_market_catalogue(
        {
            "eventTypeIds": [EVENT_TYPE_SOCCER],
            "textQuery": "Premier League",
            "marketBettingTypes": ["ODDS"],
        },
        max_results=200,
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in list(catalogue) + list(extra):
        mid = str(row.get("marketId") or "")
        kind = classify_outright(str(row.get("marketName") or ""))
        if not mid or kind is None:
            continue
        row = dict(row)
        row["_kind"] = kind
        by_id[mid] = row
    markets = list(by_id.values())
    books = client.list_market_book([str(m["marketId"]) for m in markets])
    book_by = {str(b["marketId"]): b for b in books}
    _write(SCRATCH / "raw_outrights.json", {"catalogue": markets, "books": books})

    by_kind: dict[str, dict[str, float]] = {}
    for market in markets:
        kind = str(market["_kind"])
        book = book_by.get(str(market["marketId"]))
        if book is None:
            continue
        mids = runner_mids(book, market)
        mass = {"winner": 1.0, "top6": 6.0, "relegation": 3.0}[kind]
        normalised = simplex(mids, mass=mass)
        matched = float(book.get("totalMatched") or 0.0)
        prev = by_kind.get(kind)
        # Keep the most-matched market per kind (store matched on a side map).
        if prev is not None and float(prev.get("__matched__", 0.0)) >= matched:
            continue
        normalised["__matched__"] = matched
        by_kind[kind] = normalised

    clubs = set()
    for kind in ("winner", "top6", "relegation"):
        clubs.update(k for k in (by_kind.get(kind) or {}) if k != "__matched__")
    table = []
    for club in sorted(clubs):
        p_win = float((by_kind.get("winner") or {}).get(club, 0.0))
        p_top6 = float((by_kind.get("top6") or {}).get(club, 0.0))
        p_rel = float((by_kind.get("relegation") or {}).get(club, 0.0))
        rank = expected_rank(p_win, p_top6, p_rel)
        table.append(
            {
                "club": club,
                "club_norm": norm_team(club),
                "p_win": p_win,
                "p_top6": p_top6,
                "p_rel": p_rel,
                "E_rank": rank,
                "strength": strength_from_rank(rank),
            }
        )
    table.sort(key=lambda r: r["E_rank"])
    _write(out_dir / "outrights_ranks.json", table)
    return table


def strength_index(table: list[Mapping[str, Any]]) -> dict[str, float]:
    """club_norm → strength."""
    return {str(r["club_norm"]): float(r["strength"]) for r in table if r.get("club_norm")}


def club_strength(strengths: Mapping[str, float], club_norm: str) -> float:
    """Strength for a club; missing sides use the bottom-tier outright default."""
    if club_norm in strengths:
        return float(strengths[club_norm])
    return float(MISSING_OUTRIGHT_STRENGTH)


def forecast_pots_from_outrights(
    fixtures: list[Mapping[str, Any]],
    team_names: Mapping[int, str],
    strengths: Mapping[str, float],
    *,
    start: int,
    end: int,
    priced_weeks: set[int],
) -> dict[tuple[int, str], list[dict[str, float]]]:
    """Shrunk strength pots for unpriced fixture weeks only.

    Only scheduled fixtures emit pots (blanks stay zero). Clubs absent from
    the outright table get ``MISSING_OUTRIGHT_STRENGTH``, not mid-table 0.
    """
    out: dict[tuple[int, str], list[dict[str, float]]] = {}
    for fixture in fixtures:
        event = fixture.get("event")
        if event is None:
            continue
        gw = int(event)
        if gw < int(start) or gw > int(end) or gw in priced_weeks:
            continue
        home = team_names.get(int(fixture["team_h"]))
        away = team_names.get(int(fixture["team_a"]))
        if not home or not away:
            continue
        h = norm_team(home)
        a = norm_team(away)
        s_h = club_strength(strengths, h)
        s_a = club_strength(strengths, a)
        pot_h, pot_a = strength_match_pots(s_h, s_a)
        out.setdefault((gw, h), []).append(pot_h)
        out.setdefault((gw, a), []).append(pot_a)
    return out


def apply_to_score_overlay(
    pool_row: Mapping[str, Any],
    pot: Mapping[str, float],
    *,
    goal_rate: float | None,
) -> float:
    """One-match score with optional Betfair goal rate replacing share×λ goals."""
    from src.models.forecast_xp import xp_on_pot

    if goal_rate is None:
        return xp_on_pot(
            position=str(pool_row.get("position") or "MID"),
            xmi=float(pool_row.get("minutes") or 0.0),
            share_xg=float(pool_row.get("share_xG") or 0.0),
            share_xa=float(pool_row.get("share_xA") or 0.0),
            exp_defcon_hit=float(pool_row.get("exp_defcon_hit") or 0.0),
            fwd_goal_scale=1.0,
            pot=dict(pot),
        )
    # Rebuild with share_xG such that share * lam = goal_rate.
    lam = float(pot.get("lam_scored") or 0.0)
    share = 0.0 if lam <= 0 else float(goal_rate) / lam
    return xp_on_pot(
        position=str(pool_row.get("position") or "MID"),
        xmi=float(pool_row.get("minutes") or 0.0),
        share_xg=share,
        share_xa=float(pool_row.get("share_xA") or 0.0),
        exp_defcon_hit=float(pool_row.get("exp_defcon_hit") or 0.0),
        fwd_goal_scale=1.0,
        pot=dict(pot),
    )


def _fold_name(text: str) -> str:
    """Case-fold and strip accents so Gyökeres and Ødegaard match plain spellings."""
    raw = str(text)
    for src, dst in (("ø", "o"), ("Ø", "o"), ("æ", "ae"), ("Æ", "ae"), ("å", "a"), ("Å", "a")):
        raw = raw.replace(src, dst)
    raw = unicodedata.normalize("NFKD", raw)
    return "".join(ch for ch in raw if not unicodedata.combining(ch)).casefold().strip()


def _name_hits(
    runner_name: str,
    by_label: Mapping[str, set[int]],
    players: list[dict[str, Any]],
) -> set[int]:
    """One FPL id for a Betfair runner, or empty when the name is ambiguous."""
    key = _fold_name(runner_name)
    exact = by_label.get(key) or set()
    if len(exact) == 1:
        return set(exact)
    if len(exact) > 1 or not key:
        return set()
    parts = [part for part in key.split() if part]
    if len(parts) < 2:
        return set()
    last = parts[-1]
    found: set[int] = set()
    for player in players:
        tokens = player["tokens"]
        if not tokens or tokens[-1] != last:
            continue
        if all(
            any(len(part) >= 3 and len(token) >= 3 and token.startswith(part) for token in tokens)
            for part in parts[:-1]
        ):
            found.add(int(player["pid"]))
    return found if len(found) == 1 else set()


def match_to_score_runners(
    rows: list[Mapping[str, Any]],
    bootstrap_players: list[Mapping[str, Any]],
) -> dict[str, float]:
    """Map FPL player_id → μ_raw from Betfair anytime rows.

    Labels are web name, surname, and full name, with accents folded. A
    shortened first name still matches when the surname token is unique
    (Ben White → Benjamin White, Gabriel Magalhaes → Gabriel dos Santos
    Magalhães). ``mu_raw`` is the populated goal rate. ``matched`` only
    breaks a tie.
    """
    by_label: dict[str, set[int]] = {}
    players: list[dict[str, Any]] = []
    for player in bootstrap_players:
        pid = int(player["id"])
        full = f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
        tokens = [part for part in _fold_name(full).split() if part]
        players.append({"pid": pid, "tokens": tokens})
        for label in (
            str(player.get("web_name") or ""),
            str(player.get("second_name") or ""),
            full,
        ):
            key = _fold_name(label)
            if not key:
                continue
            by_label.setdefault(key, set()).add(pid)
    out: dict[str, float] = {}
    matched_vol: dict[str, float] = {}
    for row in rows:
        hits = _name_hits(str(row.get("runner") or ""), by_label, players)
        if len(hits) != 1:
            continue
        pid = f"2026-27:{next(iter(hits))}"
        vol = float(row.get("matched") or 0.0)
        if pid in out and vol < matched_vol.get(pid, 0.0):
            continue
        out[pid] = float(row["mu_raw"])
        matched_vol[pid] = vol
    return out
