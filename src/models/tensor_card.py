"""Typed tensor card — source of truth for flashcard event painting.

A club-season is one tensor. Fields are typed (not Baudot bit planes).
Opacity is a display view of P(count ≥ 1), not a storage format.

  header     [T, …]   club/opp ids, home, 1X2 floats
  who        [P]      player id, role
  minutes    [P, T]   raw minutes (float, not /90)
  goals      [P, T]   uint counts
  assists    [P, T]   uint counts
  present    [P, T]   bool

Training spine: sliced player–GW observations (as-of).
Full-card FOV (stepped / horizon) is the same tensor, examined jointly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.fpl_odds import POS_MAP, load_football_data
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
REPORTS = ROOT / "reports"
ART = Path("/opt/cursor/artifacts")

TRAIN_SEASONS = ("2022-23", "2023-24", "2024-25")
TEST_SEASON = "2025-26"
MAX_PLAYERS = 40
MAX_GW = 40
ROLE = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
SEASON_FD = {
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}

SHEET_COLUMNS = (
    "name",
    "position",
    "team",
    "element",
    "fixture",
    "goals_scored",
    "assists",
    "kickoff_time",
    "minutes",
    "was_home",
    "GW",
)


@dataclass(frozen=True)
class TensorCard:
    """One club-season as typed arrays."""

    season: str
    team: str
    names: tuple[str, ...]
    player_id: np.ndarray  # (P,) FPL element ids; 0 = empty
    position: np.ndarray  # (P,) 0..3 or -1
    gw: np.ndarray  # (T,)
    home: np.ndarray  # (T,)
    opponent: tuple[str, ...]
    opp_id: np.ndarray  # (T,)
    club_id: int
    odds_1x2: np.ndarray  # (T, 3) assessed win, draw, opp win; nan if missing
    minutes: np.ndarray  # (P, T) float raw minutes
    goals: np.ndarray  # (P, T) uint
    assists: np.ndarray  # (P, T) uint
    present: np.ndarray  # (P, T) bool
    n_players: int
    n_fixtures: int


def load_season(season: str) -> pd.DataFrame:
    """Player-fixture rows for one closed season."""
    path = CACHE / f"merged_gw_{season.replace('-', '_')}.csv"
    frame = pd.read_csv(path, usecols=lambda name: name in SHEET_COLUMNS)
    frame["position"] = frame["position"].astype(str).str.upper().map(POS_MAP)
    frame = frame.dropna(subset=["position", "team", "kickoff_time", "element"]).copy()
    for column in ("minutes", "goals_scored", "assists", "GW"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    home = frame["was_home"].astype(str).str.lower().isin({"true", "1"})
    frame["is_home"] = home.astype(np.int64)
    frame["kickoff"] = pd.to_datetime(frame["kickoff_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["kickoff"]).copy()
    frame["season"] = season
    frame["element"] = frame["element"].astype(int)
    frame["name"] = frame["name"].astype(str)
    frame["team"] = frame["team"].astype(str)
    frame["opponent"] = _opponent_names(frame).to_numpy()
    return frame


def _opponent_names(frame: pd.DataFrame) -> pd.Series:
    rows: list[tuple[object, object, str]] = []
    for fixture, names in frame.groupby("fixture")["team"].agg(lambda s: list(pd.unique(s))).items():
        clubs = [str(name) for name in names if str(name)]
        if len(clubs) == 2:
            rows.append((fixture, clubs[0], clubs[1]))
            rows.append((fixture, clubs[1], clubs[0]))
        elif len(clubs) == 1:
            rows.append((fixture, clubs[0], "UNK"))
    if not rows:
        return pd.Series("UNK", index=frame.index)
    paired = pd.DataFrame(rows, columns=["fixture", "team", "opponent"])
    merged = frame[["fixture", "team"]].merge(paired, on=["fixture", "team"], how="left")
    return merged["opponent"].fillna("UNK")


def fit_clubs(cards: list[TensorCard] | list[str]) -> dict[str, int]:
    if cards and isinstance(cards[0], TensorCard):
        names = {c.team for c in cards}
        names.update(opp for c in cards for opp in c.opponent if opp)
    else:
        names = set(str(x) for x in cards)
    names.discard("")
    names.discard("UNK")
    return {name: i + 1 for i, name in enumerate(sorted(names))}


def load_odds_table(seasons: tuple[str, ...]) -> dict[tuple[str, str, str], np.ndarray]:
    """Map (season, home_norm, away_norm) → shin 1X2 probs (home, draw, away)."""
    out: dict[tuple[str, str, str], np.ndarray] = {}
    for season in seasons:
        code = SEASON_FD.get(season)
        if not code:
            continue
        try:
            frame = load_football_data(code=code)
        except FileNotFoundError:
            continue
        for row in frame.itertuples(index=False):
            key = (season, str(row.home_norm), str(row.away_norm))
            out[key] = np.array([row.p_home, row.p_draw, row.p_away], dtype=np.float32)
    return out


def _assessed_1x2(odds_home_draw_away: np.ndarray, home: bool) -> np.ndarray:
    """Return (club win, draw, opponent win) from match 1X2."""
    h, d, a = odds_home_draw_away
    if home:
        return np.array([h, d, a], dtype=np.float32)
    return np.array([a, d, h], dtype=np.float32)


def build_tensor_cards(
    seasons: tuple[str, ...],
    clubs: dict[str, int] | None = None,
    odds_table: dict | None = None,
) -> list[TensorCard]:
    """Build typed club-season cards. Goal/assist counts are not clipped to 3."""
    if odds_table is None:
        odds_table = load_odds_table(seasons)
    frames = {season: load_season(season) for season in seasons}
    if clubs is None:
        names: set[str] = set()
        for frame in frames.values():
            names.update(frame["team"].astype(str))
            names.update(frame["opponent"].astype(str))
        clubs = fit_clubs(sorted(names))

    cards: list[TensorCard] = []
    for season, frame in frames.items():
        frame = frame.sort_values(["element", "fixture", "kickoff"]).drop_duplicates(
            ["element", "fixture", "team"], keep="first"
        )
        for team, club in frame.groupby("team", sort=False):
            team = str(team)
            played = club[club["minutes"] >= 1]
            if played.empty:
                continue
            meta = (
                played.sort_values("kickoff")
                .groupby("element", as_index=False)
                .agg(name=("name", "last"), position=("position", "last"), minutes=("minutes", "sum"))
            )
            meta["pos"] = meta["position"].map(ROLE)
            meta = meta.dropna(subset=["pos"]).sort_values(["pos", "element"])
            if len(meta) > MAX_PLAYERS:
                meta = meta.nlargest(MAX_PLAYERS, "minutes").sort_values(["pos", "element"])
            elements = [int(i) for i in meta["element"]]
            names = ["spare"] * MAX_PLAYERS
            position = np.full(MAX_PLAYERS, -1, dtype=np.int64)
            player_id = np.zeros(MAX_PLAYERS, dtype=np.int64)
            for row, item in enumerate(meta.itertuples(index=False)):
                names[row] = str(item.name)
                position[row] = int(item.pos)
                player_id[row] = int(item.element)

            ordered = (
                club.groupby("fixture", as_index=False)
                .agg(
                    kickoff=("kickoff", "min"),
                    gw=("GW", "min"),
                    home=("is_home", "max"),
                    opponent=("opponent", "first"),
                )
                .sort_values(["kickoff", "fixture"])
                .reset_index(drop=True)
            )
            n_fix = min(len(ordered), MAX_GW)
            ordered = ordered.iloc[:n_fix]
            gw = np.zeros(MAX_GW, dtype=np.int64)
            home = np.zeros(MAX_GW, dtype=np.int64)
            opp_id = np.zeros(MAX_GW, dtype=np.int64)
            opponents = [""] * MAX_GW
            odds_1x2 = np.full((MAX_GW, 3), np.nan, dtype=np.float32)
            fixture_id = np.zeros(MAX_GW, dtype=np.int64)
            for col, item in enumerate(ordered.itertuples(index=False)):
                gw[col] = int(item.gw)
                home[col] = int(item.home)
                fixture_id[col] = int(item.fixture)
                opp = str(item.opponent) if item.opponent else ""
                opponents[col] = opp
                opp_id[col] = clubs.get(opp, 0)
                team_norm = norm_team(team)
                opp_norm = norm_team(opp) if opp else ""
                if home[col]:
                    key = (season, team_norm, opp_norm)
                else:
                    key = (season, opp_norm, team_norm)
                raw = odds_table.get(key)
                if raw is not None:
                    odds_1x2[col] = _assessed_1x2(raw, bool(home[col]))

            index = {int(fx): col for col, fx in enumerate(fixture_id) if col < n_fix and gw[col] > 0}
            minutes = np.zeros((MAX_PLAYERS, MAX_GW), dtype=np.float32)
            goals = np.zeros((MAX_PLAYERS, MAX_GW), dtype=np.uint8)
            assists = np.zeros((MAX_PLAYERS, MAX_GW), dtype=np.uint8)
            present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
            row_of = {element: row for row, element in enumerate(elements)}
            for item in club.itertuples(index=False):
                row = row_of.get(int(item.element))
                col = index.get(int(item.fixture))
                if row is None or col is None:
                    continue
                present[row, col] = True
                minutes[row, col] = float(item.minutes)
                goals[row, col] = np.uint8(min(int(item.goals_scored), 255))
                assists[row, col] = np.uint8(min(int(item.assists), 255))

            cards.append(
                TensorCard(
                    season=season,
                    team=team,
                    names=tuple(names),
                    player_id=player_id,
                    position=position,
                    gw=gw,
                    home=home,
                    opponent=tuple(opponents),
                    opp_id=opp_id,
                    club_id=clubs.get(team, 0),
                    odds_1x2=odds_1x2,
                    minutes=minutes,
                    goals=goals,
                    assists=assists,
                    present=present,
                    n_players=len(elements),
                    n_fixtures=n_fix,
                )
            )
    return cards


def cut_at_fraction(card: TensorCard, frac: float) -> int:
    n = int(card.n_fixtures)
    cut = int(round(float(frac) * n))
    return int(max(1, min(n - 1, cut)))


def role_minutes_prior(cards: list[TensorCard]) -> dict[int, float]:
    """Mean minutes by role over present cells (for lag-1 cold start)."""
    sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for card in cards:
        for p in range(card.n_players):
            role = int(card.position[p])
            if role < 0:
                continue
            for t in range(card.n_fixtures):
                if not card.present[p, t]:
                    continue
                sums[role] += float(card.minutes[p, t])
                counts[role] += 1
    return {role: (sums[role] / counts[role] if counts[role] else 0.0) for role in sums}


def lag1_minutes(
    card: TensorCard,
    player: int,
    t: int,
    role_prior: dict[int, float] | None = None,
) -> float:
    """Predicted minutes for week t: last week's minutes, else last present, else role prior."""
    role = int(card.position[player])
    prior = 0.0 if role_prior is None else float(role_prior.get(role, 0.0))
    if t <= 0:
        return prior
    if card.present[player, t - 1]:
        return float(card.minutes[player, t - 1])
    hist = np.flatnonzero(card.present[player, :t])
    if len(hist):
        return float(card.minutes[player, hist[-1]])
    return prior


def asof_inputs(
    card: TensorCard,
    player: int,
    t: int,
    *,
    minutes_mode: str = "actual",
    role_prior: dict[int, float] | None = None,
    include_mates: bool | None = None,
) -> dict[str, float]:
    """Features available before scoring week t. History is strictly before t.

    minutes_mode:
      actual — scored-week realised minutes (conditional-on-minutes spine)
      lag1 — ̂m_t = m_{t-1} (else last present, else role prior); no same-week mate leak
    """
    hist = np.flatnonzero(card.present[player, :t])
    minutes_hist = card.minutes[player, hist] if len(hist) else np.array([], np.float32)
    goals_hist = card.goals[player, hist].astype(np.float32) if len(hist) else np.array([], np.float32)
    assists_hist = card.assists[player, hist].astype(np.float32) if len(hist) else np.array([], np.float32)
    n = float(len(hist))
    role = int(card.position[player])
    odds = card.odds_1x2[t]
    if minutes_mode == "lag1":
        minutes = lag1_minutes(card, player, t, role_prior=role_prior)
        use_mates = False if include_mates is None else include_mates
    elif minutes_mode == "actual":
        minutes = float(card.minutes[player, t])
        use_mates = True if include_mates is None else include_mates
    else:
        raise ValueError(f"unknown minutes_mode={minutes_mode!r}")
    mate_minutes = 0.0
    mate_n = 0.0
    if use_mates:
        on = np.flatnonzero((card.minutes[:, t] > 0) & (card.position >= 0))
        on = on[on < card.n_players]
        mate_minutes = float(card.minutes[on, t].sum()) - float(card.minutes[player, t])
        mate_n = float(max(len(on) - (1 if player in on else 0), 0))
    return {
        "minutes": float(minutes),
        "home": float(card.home[t]),
        "role_gkp": 1.0 if role == 0 else 0.0,
        "role_def": 1.0 if role == 1 else 0.0,
        "role_mid": 1.0 if role == 2 else 0.0,
        "role_fwd": 1.0 if role == 3 else 0.0,
        "odds_win": float(odds[0]) if np.isfinite(odds[0]) else 1.0 / 3.0,
        "odds_draw": float(odds[1]) if np.isfinite(odds[1]) else 1.0 / 3.0,
        "odds_lose": float(odds[2]) if np.isfinite(odds[2]) else 1.0 / 3.0,
        "games_played": n,
        "mean_minutes": float(minutes_hist.mean()) if n else 0.0,
        "mean_goals": float(goals_hist.mean()) if n else 0.0,
        "mean_assists": float(assists_hist.mean()) if n else 0.0,
        "last_goals": float(goals_hist[-1]) if n else 0.0,
        "last_assists": float(assists_hist[-1]) if n else 0.0,
        "last5_goals": float(goals_hist[-5:].mean()) if n else 0.0,
        "last5_assists": float(assists_hist[-5:].mean()) if n else 0.0,
        "mate_minutes": mate_minutes,
        "mate_n": mate_n,
    }


FEATURE_NAMES = (
    "minutes",
    "home",
    "role_gkp",
    "role_def",
    "role_mid",
    "role_fwd",
    "odds_win",
    "odds_draw",
    "odds_lose",
    "games_played",
    "mean_minutes",
    "mean_goals",
    "mean_assists",
    "last_goals",
    "last_assists",
    "last5_goals",
    "last5_assists",
    "mate_minutes",
    "mate_n",
)

# Lag-1 arm drops same-week mate leak so the contrast is minutes policy only.
FEATURE_NAMES_LAG1 = tuple(name for name in FEATURE_NAMES if name not in ("mate_minutes", "mate_n"))

# Same-week pitch-mates: who shared the pitch (not collinear total mate minutes).
# Conditional-on-XI spine — parallel to conditioning on own realised minutes.
COPRESENCE_FEATURE_NAMES = (
    "mate_n_on",
    "mate_def_on",
    "mate_mid_on",
    "mate_fwd_on",
    "mate_sum_mean_goals",
    "mate_sum_mean_assists",
    "mate_max_mean_goals",
)


def mate_copresence_features(card: TensorCard, player: int, t: int) -> dict[str, float]:
    """Teammate co-presence at week t, with as-of history for those who played.

    Uses the scored week's who-played mask (like own minutes on the conditional
    spine). Mate history means use only weeks strictly before t.
    """
    roles = card.position
    mins = card.minutes[:, t]
    seats = np.arange(card.n_players)
    on = seats[(mins[: card.n_players] > 0) & (roles[: card.n_players] >= 0)]
    on = on[on != player]
    mate_n_on = float(len(on))
    mate_def_on = float(np.sum(roles[on] == 1))
    mate_mid_on = float(np.sum(roles[on] == 2))
    mate_fwd_on = float(np.sum(roles[on] == 3))
    sum_mg = 0.0
    sum_ma = 0.0
    max_mg = 0.0
    for mate in on:
        hist = np.flatnonzero(card.present[mate, :t])
        if len(hist) == 0:
            continue
        mg = float(card.goals[mate, hist].mean())
        ma = float(card.assists[mate, hist].mean())
        sum_mg += mg
        sum_ma += ma
        if mg > max_mg:
            max_mg = mg
    return {
        "mate_n_on": mate_n_on,
        "mate_def_on": mate_def_on,
        "mate_mid_on": mate_mid_on,
        "mate_fwd_on": mate_fwd_on,
        "mate_sum_mean_goals": sum_mg,
        "mate_sum_mean_assists": sum_ma,
        "mate_max_mean_goals": max_mg,
    }


def iter_slices(
    card: TensorCard,
    min_history: int = 1,
    *,
    minutes_mode: str = "actual",
    role_prior: dict[int, float] | None = None,
    include_mates: bool | None = None,
):
    """Yield sliced player–GW observations with as-of features."""
    for p in range(card.n_players):
        if int(card.position[p]) < 0 or int(card.player_id[p]) <= 0:
            continue
        for t in range(min_history, card.n_fixtures):
            if not card.present[p, t]:
                continue
            yield {
                "season": card.season,
                "team": card.team,
                "player_id": int(card.player_id[p]),
                "name": card.names[p],
                "gw": int(card.gw[t]),
                "t": t,
                "position": int(card.position[p]),
                "goals": int(card.goals[p, t]),
                "assists": int(card.assists[p, t]),
                "y_goal": 1 if int(card.goals[p, t]) >= 1 else 0,
                "y_assist": 1 if int(card.assists[p, t]) >= 1 else 0,
                "features": asof_inputs(
                    card,
                    p,
                    t,
                    minutes_mode=minutes_mode,
                    role_prior=role_prior,
                    include_mates=include_mates,
                ),
            }


def blank_future_events(card: TensorCard, cut: int) -> TensorCard:
    """Full-card FOV: keep header/who/minutes; blank G/A from cut onward."""
    goals = card.goals.copy()
    assists = card.assists.copy()
    goals[:, cut:] = 0
    assists[:, cut:] = 0
    return TensorCard(
        season=card.season,
        team=card.team,
        names=card.names,
        player_id=card.player_id.copy(),
        position=card.position.copy(),
        gw=card.gw.copy(),
        home=card.home.copy(),
        opponent=card.opponent,
        opp_id=card.opp_id.copy(),
        club_id=card.club_id,
        odds_1x2=card.odds_1x2.copy(),
        minutes=card.minutes.copy(),
        goals=goals,
        assists=assists,
        present=card.present.copy(),
        n_players=card.n_players,
        n_fixtures=card.n_fixtures,
    )
