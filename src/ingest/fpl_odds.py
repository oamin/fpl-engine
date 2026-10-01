"""Stage 1 — ingest Vaastav FPL logs + football-data odds (1X2, OU, AH)."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import numpy as np
import pandas as pd

from src.markets.shin import shin_1x2, shin_pair
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FPL_GW_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/"
    "data/{season}/gws/merged_gw.csv"
)

POS_MAP = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
SEASON = "2025-26"
FD_CODE = "2526"


def _https(url: str) -> str:
    p = urlparse(url)
    return urlunparse(p._replace(scheme="https")) if p.scheme == "http" else url


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def parse_fd_date(value: str) -> str:
    return datetime.strptime(value.strip(), "%d/%m/%Y").date().isoformat()


def parse_fpl_date(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def _first_float(row: dict[str, str], *keys: str) -> float | None:
    for k in keys:
        v = to_float(row.get(k))
        if v is not None:
            return v
    return None


def load_football_data(
    client: httpx.Client | None = None, code: str = FD_CODE
) -> pd.DataFrame:
    cache = DATA / "cache" / f"E0_{code}.csv"
    if cache.exists():
        text = cache.read_text(encoding="utf-8-sig")
    else:
        if client is None:
            raise FileNotFoundError(f"Missing {cache} and no httpx client")
        text = client.get(_https(FD_URL.format(code=code)), timeout=60.0).text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    rows = list(csv.DictReader(StringIO(text)))
    out: list[dict[str, Any]] = []
    for row in rows:
        home = row.get("HomeTeam") or ""
        away = row.get("AwayTeam") or ""
        if not home or not away:
            continue
        h = _first_float(row, "AvgCH", "B365CH", "AvgH", "B365H", "PSH")
        d = _first_float(row, "AvgCD", "B365CD", "AvgD", "B365D", "PSD")
        a = _first_float(row, "AvgCA", "B365CA", "AvgA", "B365A", "PSA")
        over = _first_float(row, "AvgC>2.5", "B365C>2.5", "Avg>2.5", "B365>2.5")
        under = _first_float(row, "AvgC<2.5", "B365C<2.5", "Avg<2.5", "B365<2.5")
        ah_line = _first_float(row, "AHh")
        ah_h = _first_float(row, "AvgCAHH", "B365CAHH", "AvgAHH", "B365AHH")
        ah_a = _first_float(row, "AvgCAHA", "B365CAHA", "AvgAHA", "B365AHA")
        if None in (h, d, a, over, under):
            continue
        assert h is not None and d is not None and a is not None
        assert over is not None and under is not None
        p_h, p_d, p_a = shin_1x2(h, d, a)
        p_over, p_under = shin_pair(over, under)
        p_ah_home = p_ah_away = None
        if ah_h is not None and ah_a is not None and ah_h > 1 and ah_a > 1:
            p_ah_home, p_ah_away = shin_pair(ah_h, ah_a)
        out.append(
            {
                "date": parse_fd_date(row["Date"]),
                "home": home,
                "away": away,
                "home_norm": norm_team(home),
                "away_norm": norm_team(away),
                "home_goals": to_float(row.get("FTHG")),
                "away_goals": to_float(row.get("FTAG")),
                "home_odds": h,
                "draw_odds": d,
                "away_odds": a,
                "over25_odds": over,
                "under25_odds": under,
                "ah_line": ah_line,
                "ah_home_odds": ah_h,
                "ah_away_odds": ah_a,
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "p_over25": p_over,
                "p_under25": p_under,
                "p_ah_home": p_ah_home,
                "p_ah_away": p_ah_away,
            }
        )
    return pd.DataFrame(out)


def load_player_logs(
    client: httpx.Client | None = None, season: str = SEASON
) -> pd.DataFrame:
    cache = DATA / "cache" / f"merged_gw_{season.replace('-', '_')}.csv"
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        if client is None:
            raise FileNotFoundError(f"Missing {cache} and no httpx client")
        text = client.get(_https(FPL_GW_URL.format(season=season)), timeout=60.0).text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    rows = list(csv.DictReader(StringIO(text)))
    out: list[dict[str, Any]] = []
    for row in rows:
        minutes = to_float(row.get("minutes")) or 0.0
        if minutes <= 0:
            continue
        pos = POS_MAP.get((row.get("position") or "").upper())
        if pos is None:
            continue
        team = row.get("team") or ""
        kickoff = row.get("kickoff_time") or ""
        if not team or not kickoff:
            continue
        out.append(
            {
                "date": parse_fpl_date(kickoff),
                "gw": to_float(row.get("GW") or row.get("round")),
                "player_id": str(row.get("element") or ""),
                "player_name": str(row.get("name") or ""),
                "team": team,
                "team_norm": norm_team(team),
                "position": pos,
                "is_home": 1 if str(row.get("was_home", "")).lower() in {"true", "1"} else 0,
                "minutes": minutes,
                "total_points": to_float(row.get("total_points")) or 0.0,
                "goals": to_float(row.get("goals_scored")) or 0.0,
                "assists": to_float(row.get("assists")) or 0.0,
                "clean_sheets": to_float(row.get("clean_sheets")) or 0.0,
                "goals_conceded": to_float(row.get("goals_conceded")) or 0.0,
                "saves": to_float(row.get("saves")) or 0.0,
                "bonus": to_float(row.get("bonus")) or 0.0,
                "bps": to_float(row.get("bps")) or 0.0,
                "xG": to_float(row.get("expected_goals")) or 0.0,
                "xA": to_float(row.get("expected_assists")) or 0.0,
                "defcon": to_float(row.get("defensive_contribution")) or 0.0,
                "yellow_cards": to_float(row.get("yellow_cards")) or 0.0,
                "red_cards": to_float(row.get("red_cards")) or 0.0,
                "starts": to_float(row.get("starts")) or 0.0,
            }
        )
    return pd.DataFrame(out)


def join_players_to_fixtures(
    players: pd.DataFrame, fixtures: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach fixture odds/probs to each player-match; build fixture table with pots later."""
    fix = fixtures.copy()
    fix["fixture_id"] = (
        fix["date"].astype(str)
        + ":"
        + fix["home_norm"]
        + ":"
        + fix["away_norm"]
    )

    # Map (date, team_norm) → fixture side.
    home_map = fix.set_index(["date", "home_norm"])["fixture_id"].to_dict()
    away_map = fix.set_index(["date", "away_norm"])["fixture_id"].to_dict()

    fixture_ids: list[str | None] = []
    for _, r in players.iterrows():
        key = (r["date"], r["team_norm"])
        if r["is_home"] == 1:
            fixture_ids.append(home_map.get(key) or away_map.get(key))
        else:
            fixture_ids.append(away_map.get(key) or home_map.get(key))
    players = players.copy()
    players["fixture_id"] = fixture_ids
    joined = players.dropna(subset=["fixture_id"]).merge(
        fix,
        on="fixture_id",
        how="inner",
        suffixes=("", "_fix"),
    )
    # Side-perspective market features.
    is_home = joined["is_home"].astype(int) == 1
    import numpy as np

    joined["p_win"] = np.where(is_home, joined["p_home"], joined["p_away"])
    joined["p_lose"] = np.where(is_home, joined["p_away"], joined["p_home"])
    joined["p_draw_m"] = joined["p_draw"]
    joined["p_over"] = joined["p_over25"]
    ah = pd.to_numeric(joined["ah_line"], errors="coerce")
    joined["ah_line_own"] = np.where(is_home, ah, -ah)
    joined["p_ah_cover"] = np.where(
        is_home,
        pd.to_numeric(joined["p_ah_home"], errors="coerce"),
        pd.to_numeric(joined["p_ah_away"], errors="coerce"),
    )
    joined["attack_strength"] = joined["p_win"] + 0.5 * joined["p_draw"]
    joined["defend_threat"] = joined["p_lose"] + 0.5 * joined["p_draw"]

    stats = {
        "n_fixtures_odds": int(len(fix)),
        "n_player_appearances": int(len(players)),
        "n_player_joined": int(len(joined)),
        "join_rate": float(len(joined) / max(len(players), 1)),
        "date_min": str(joined["date"].min()) if len(joined) else "",
        "date_max": str(joined["date"].max()) if len(joined) else "",
        "ah_coverage": float(joined["p_ah_cover"].notna().mean()) if len(joined) else 0.0,
        "n_by_position": joined.groupby("position").size().to_dict(),
    }
    return joined, fix, stats


def write_stage1_report(path: Path, stats: dict[str, Any]) -> None:
    pos = stats.get("n_by_position") or {}
    lines = [
        "# Stage 1 — Ingest",
        "",
        f"- Odds fixtures (football-data {FD_CODE}): **{stats['n_fixtures_odds']}**",
        f"- Player appearances (Vaastav {SEASON}): **{stats['n_player_appearances']}**",
        f"- Joined player-matches: **{stats['n_player_joined']}** (rate {stats['join_rate']:.3f})",
        f"- Date range: {stats['date_min']} → {stats['date_max']}",
        f"- AH coverage (p_ah_cover non-null): **{stats['ah_coverage']:.3f}**",
        "",
        "## Positions",
        "",
    ]
    for p in ("GKP", "DEF", "MID", "FWD"):
        lines.append(f"- {p}: {pos.get(p, 0)}")
    lines += [
        "",
        "## Outputs",
        "",
        "- `data/processed/player_matches.csv`",
        "- `data/processed/fixtures_odds.csv`",
        "",
        "Markets: Shin-devig 1X2, OU 2.5, AH (when present). No Odds API calls.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ingest(
    *,
    season: str = SEASON,
    fd_code: str = FD_CODE,
) -> dict[str, Any]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cache_fd = DATA / "cache" / f"E0_{fd_code}.csv"
    cache_gw = DATA / "cache" / f"merged_gw_{season.replace('-', '_')}.csv"
    need_net = not cache_fd.exists() or not cache_gw.exists()
    if need_net:
        with httpx.Client(
            timeout=60.0,
            follow_redirects=True,
            headers={"User-Agent": "fpl-engine/1.0"},
        ) as client:
            fixtures = load_football_data(client, fd_code)
            players = load_player_logs(client, season)
    else:
        fixtures = load_football_data(None, fd_code)
        players = load_player_logs(None, season)
    joined, fix, stats = join_players_to_fixtures(players, fixtures)
    joined.to_csv(PROCESSED / "player_matches.csv", index=False)
    fix.to_csv(PROCESSED / "fixtures_odds.csv", index=False)
    write_stage1_report(REPORTS / "stage_1_ingest.md", stats)
    return stats
