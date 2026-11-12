"""Squad-card trial: one club grid, two readers, the hidden weeks filled in one pass.

Each card is a club-season. Rows are teammates. Columns are the club's fixtures.
Minutes stay visible. Goals and assists after the cut are hidden. The original
squad trial couples each fixture to the other squad. The locked card does not.
The locked card is the
square card. Each column shows the club symbol, the opponent symbol, home or
away, the Asian handicap, and the 2.5-goal total. Minutes are six squares of
fifteen minutes.
One goal or assist is one
square, and those rows are three squares tall. Concede, regulars, own
history, opponent history, and the team scoring rate are absent. The locked
vision reader gets the square card and nothing else: no minutes number, no
position id, no club id, no opponent id, no match-up, and no other squad.
Clean sheets are not a target.

The sequence reader is a separate numeric model. It is not the published
painter. Both readers attend across the squad and across the whole segment
at once. Neither walks column by column.

Train on 2022-23, 2023-24, and 2024-25. Each card is cut after 20%, 40%, 60%,
or 80% of its fixtures, starting from gameweek 1. The test scores each of
those four cuts. 2026-27 is not read.

This is the locked squad-card reader: squad vision on the square card, without
the other squad. The test score is the painted card against the actual card,
counted across every row. See the card comparison in the goal-hit log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.flashcard_trial import TRAIN_SEASONS, TEST_SEASON, load_season

ROOT_PLOTS = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "plots"
ROOT_REPORTS = __import__("pathlib").Path(__file__).resolve().parents[2] / "reports"
ROOT_PROCESSED = __import__("pathlib").Path(__file__).resolve().parents[2] / "data" / "processed"
ARTIFACTS = __import__("pathlib").Path("/opt/cursor/artifacts")

ROWS = 40
COLS = 40
BAND_H = 8
CELL_W = 4
HEADER_H = BAND_H * 5
MINUTE_H = BAND_H * 2
EVENT_H = BAND_H * 2
POS_H = BAND_H
PLAYER_H = MINUTE_H + EVENT_H + EVENT_H + POS_H
IMG_H = HEADER_H + ROWS * PLAYER_H
IMG_W = COLS * CELL_W
MASK_ID = 4
D = 32
HEADS = 4
EPOCHS = 30
BATCH = 15
LR = 1e-3
CUT_GW = 19
PRIOR_STRENGTH = 8
SEED = 0
POS = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}

MINUTES_C = (0, 114, 178)
GOALS_C = (213, 94, 0)
ASSISTS_C = (204, 121, 167)
HOME_C = (0, 158, 115)
AWAY_C = (230, 159, 0)
CONCEDE_C = (86, 180, 233)
REG_C = (30, 30, 30)
OWN_HIST_C = (140, 70, 20)
OPP_HIST_C = (20, 70, 120)
MASK_C = (176, 176, 176)
BASE_C = (226, 226, 226)
HATCH_BG = (214, 214, 214)
HATCH_FG = (168, 168, 168)
POS_C = (
    (180, 0, 80),
    (0, 90, 50),
    (90, 0, 160),
    (150, 90, 0),
)


@dataclass
class SquadCard:
    """One club-season on the fixed 40 by 40 grid."""

    season: str
    team: str
    names: tuple[str, ...]
    position: np.ndarray
    gw: np.ndarray
    kickoff: np.ndarray
    home: np.ndarray
    opponent: tuple[str, ...]
    fixture_id: np.ndarray
    minutes: np.ndarray
    goals: np.ndarray
    assists: np.ndarray
    present: np.ndarray
    n_players: int
    n_fixtures: int


def build_league(seasons: tuple[str, ...]) -> dict:
    """Per club, the fixtures needed for an opponent header. No player targets."""
    league: dict = {}
    for season in seasons:
        frame = load_season(season)
        frame = frame.sort_values(["element", "fixture", "kickoff"]).drop_duplicates(
            ["element", "fixture", "team"], keep="first"
        )
        by_team: dict = {}
        for team, club in frame.groupby("team", sort=False):
            fixtures = []
            ordered = (
                club.groupby("fixture", as_index=False)
                .agg(kickoff=("kickoff", "min"), gw=("GW", "min"))
                .sort_values(["kickoff", "fixture"])
            )
            lookup = {}
            for row in ordered.itertuples(index=False):
                block = club[club["fixture"] == row.fixture]
                long = block[block["minutes"] >= 60]
                gc = None if long.empty else float(long["goals_conceded"].max())
                item = {
                    "fixture": int(row.fixture),
                    "gw": int(row.gw),
                    "kickoff": int(row.kickoff.value),
                    "gc": gc,
                    "gf": float(block["goals_scored"].sum()),
                    "played60": set(int(i) for i in long["element"]),
                    "played1": set(int(i) for i in block.loc[block["minutes"] >= 1, "element"]),
                }
                fixtures.append(item)
                lookup[int(row.fixture)] = item
            by_team[str(team)] = {"fixtures": fixtures, "by_id": lookup}
        league[season] = by_team
    return league


def build_cards(seasons: tuple[str, ...], league: dict) -> list[SquadCard]:
    """One card per club-season. Players with minutes keep a stable row."""
    cards: list[SquadCard] = []
    for season in seasons:
        frame = load_season(season)
        frame = frame.sort_values(["element", "fixture", "kickoff"]).drop_duplicates(
            ["element", "fixture", "team"], keep="first"
        )
        other: dict[tuple[int, str], str] = {}
        for fixture, group in frame.groupby("fixture"):
            teams = sorted(set(group["team"].astype(str)))
            if len(teams) == 2:
                other[(int(fixture), teams[0])] = teams[1]
                other[(int(fixture), teams[1])] = teams[0]
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
            meta["pos"] = meta["position"].map(POS)
            meta = meta.dropna(subset=["pos"]).sort_values(["pos", "element"])
            if len(meta) > ROWS:
                meta = meta.nlargest(ROWS, "minutes").sort_values(["pos", "element"])
            elements = [int(i) for i in meta["element"]]
            names = ["spare"] * ROWS
            position = np.full(ROWS, -1, dtype=np.int64)
            for row, item in enumerate(meta.itertuples(index=False)):
                names[row] = str(item.name)
                position[row] = int(item.pos)
            ordered = (
                club.groupby("fixture", as_index=False)
                .agg(kickoff=("kickoff", "min"), gw=("GW", "min"), home=("is_home", "max"))
                .sort_values(["kickoff", "fixture"])
                .reset_index(drop=True)
            )
            n_fix = min(len(ordered), COLS)
            ordered = ordered.iloc[:n_fix]
            gw = np.zeros(COLS, dtype=np.int64)
            kickoff = np.zeros(COLS, dtype=np.int64)
            home = np.zeros(COLS, dtype=np.int64)
            fixture_id = np.zeros(COLS, dtype=np.int64)
            opponents = [""] * COLS
            for col, item in enumerate(ordered.itertuples(index=False)):
                gw[col] = int(item.gw)
                kickoff[col] = int(item.kickoff.value)
                home[col] = int(item.home)
                fixture_id[col] = int(item.fixture)
                opponents[col] = other.get((int(item.fixture), team), "")
            index = {int(fx): col for col, fx in enumerate(fixture_id) if col < n_fix and gw[col] > 0}
            minutes = np.zeros((ROWS, COLS), dtype=np.float32)
            goals = np.zeros((ROWS, COLS), dtype=np.int64)
            assists = np.zeros((ROWS, COLS), dtype=np.int64)
            present = np.zeros((ROWS, COLS), dtype=bool)
            row_of = {element: row for row, element in enumerate(elements)}
            for item in club.itertuples(index=False):
                row = row_of.get(int(item.element))
                col = index.get(int(item.fixture))
                if row is None or col is None:
                    continue
                present[row, col] = True
                minutes[row, col] = float(item.minutes)
                goals[row, col] = min(int(item.goals_scored), 3)
                assists[row, col] = min(int(item.assists), 3)
            cards.append(
                SquadCard(
                    season=season,
                    team=team,
                    names=tuple(names),
                    position=position,
                    gw=gw,
                    kickoff=kickoff,
                    home=home,
                    opponent=tuple(opponents),
                    fixture_id=fixture_id,
                    minutes=minutes,
                    goals=goals,
                    assists=assists,
                    present=present,
                    n_players=len(elements),
                    n_fixtures=n_fix,
                )
            )
    return cards


def _regulars(league: dict, season: str, team: str, cut_gw: int) -> set[int]:
    fixtures = [fx for fx in league[season][team]["fixtures"] if fx["gw"] <= cut_gw]
    if not fixtures:
        return set()
    need = max(1, int(np.ceil(0.5 * len(fixtures))))
    counts: dict[int, int] = {}
    for fx in fixtures:
        for element in fx["played60"]:
            counts[element] = counts.get(element, 0) + 1
    return {element for element, count in counts.items() if count >= need}


def fit_clubs(cards: list[SquadCard]) -> dict[str, int]:
    """Club ids from the training cards. Index 0 is a club this run has not trained on."""
    names = {card.team for card in cards}
    names.update(opp for card in cards for opp in card.opponent if opp)
    return {name: index + 1 for index, name in enumerate(sorted(names))}


_PRIOR_CACHE: dict[int, dict] = {}


def prior_table(league: dict) -> dict[tuple[str, str], tuple[float, float, int, int]]:
    """Goals and concessions per game from seasons strictly before the card's season.

    The value is (goals for, goals conceded, games scored, games conceded). A club
    missing from the table has no earlier season in this league.
    """
    cached = _PRIOR_CACHE.get(id(league))
    if cached is not None:
        return cached
    per_season: dict[tuple[str, str], tuple[float, float, int, int]] = {}
    for season, clubs in league.items():
        for team, info in clubs.items():
            scored = [fx["gf"] for fx in info["fixtures"] if fx.get("gf") is not None]
            conceded = [fx["gc"] for fx in info["fixtures"] if fx.get("gc") is not None]
            if not scored and not conceded:
                continue
            gf = float(np.mean(scored)) if scored else 0.0
            gc = float(np.mean(conceded)) if conceded else 0.0
            per_season[(season, team)] = (gf, gc, len(scored), len(conceded))
    prior: dict[tuple[str, str], tuple[float, float, int, int]] = {}
    for season in league:
        earlier = [name for name in league if name < season]
        teams = {team for name in earlier for team in league[name]}
        for team in teams:
            gf_sum = gc_sum = 0.0
            gf_n = gc_n = 0
            for name in earlier:
                item = per_season.get((name, team))
                if item is None:
                    continue
                gf, gc, n_for, n_against = item
                gf_sum += gf * n_for
                gc_sum += gc * n_against
                gf_n += n_for
                gc_n += n_against
            if gf_n or gc_n:
                prior[(season, team)] = (
                    gf_sum / gf_n if gf_n else 0.0,
                    gc_sum / gc_n if gc_n else 0.0,
                    gf_n,
                    gc_n,
                )
    _PRIOR_CACHE[id(league)] = prior
    return prior


def _rate_before(
    league: dict, season: str, team: str, kickoff: int, cut_gw: int, hidden: bool, key: str
) -> float | None:
    """Mean of an earlier fixture's team total. A hidden column cannot see past the cut."""
    info = league.get(season, {}).get(team)
    if info is None:
        return None
    vals = []
    for fx in info["fixtures"]:
        if fx["kickoff"] >= kickoff or fx.get(key) is None:
            continue
        if hidden and fx["gw"] > cut_gw:
            continue
        vals.append(fx[key])
    if not vals:
        return None
    return float(np.mean(vals))


def _concede_rate(league: dict, season: str, team: str, kickoff: int, cut_gw: int, hidden: bool) -> float | None:
    vals = []
    for fx in league[season][team]["fixtures"]:
        if fx["kickoff"] >= kickoff or fx["gc"] is None:
            continue
        if hidden and fx["gw"] > cut_gw:
            continue
        vals.append(fx["gc"])
    if not vals:
        return None
    return float(np.mean(vals))


def squad_index(cards: list[SquadCard]) -> dict[tuple[str, str], SquadCard]:
    """One card per club-season, so a fixture can find the other squad."""
    return {(card.season, card.team): card for card in cards}


def _opponent_squad(
    card: SquadCard, cut_gw: int, squads: dict[tuple[str, str], SquadCard] | None
) -> dict[str, np.ndarray]:
    """The other squad at each fixture. Hidden goals and assists stay masked.

    Rows are that opponent's players, so they are not the same people from one
    column to the next. Spare rows are padded. A named player who did not play
    keeps a zero once the week is visible.
    """
    position = np.full((ROWS, COLS), -1, dtype=np.int64)
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    present = np.zeros((ROWS, COLS), dtype=np.float32)
    goal_token = np.full((ROWS, COLS), MASK_ID, dtype=np.int64)
    assist_token = np.full((ROWS, COLS), MASK_ID, dtype=np.int64)
    pad = np.ones((ROWS, COLS), dtype=bool)
    if squads:
        for col in range(COLS):
            if card.gw[col] <= 0 or not card.opponent[col]:
                continue
            other = squads.get((card.season, card.opponent[col]))
            if other is None:
                continue
            matched = np.where(other.fixture_id == card.fixture_id[col])[0]
            if len(matched) == 0:
                continue
            src = int(matched[0])
            position[:, col] = other.position
            minutes[:, col] = other.minutes[:, src]
            present[:, col] = other.present[:, src].astype(np.float32)
            pad[:, col] = other.position < 0
            if int(other.gw[src]) <= cut_gw:
                named = other.position >= 0
                played = other.present[:, src]
                goal_token[named & played, col] = other.goals[named & played, src]
                assist_token[named & played, col] = other.assists[named & played, src]
                goal_token[named & ~played, col] = 0
                assist_token[named & ~played, col] = 0
    return {
        "opp_position": position,
        "opp_minutes": minutes,
        "opp_present": present,
        "opp_goal_token": goal_token,
        "opp_assist_token": assist_token,
        "opp_pad": pad,
    }


def apply_cut(
    card: SquadCard,
    league: dict,
    cut_gw: int,
    clubs: dict[str, int] | None = None,
    squads: dict[tuple[str, str], SquadCard] | None = None,
) -> dict[str, np.ndarray]:
    """Numbers the readers see. Hidden bars are the mask id, not zero.

    Club history comes from earlier seasons only. A club with no earlier season,
    in a league where other clubs have one, is marked missing: that is a promoted
    side. The first season in the file has no earlier season for anyone, so its
    history features stay zero and do not pretend every club is new.
    """
    hidden = (card.gw > cut_gw) & (card.gw > 0)
    goal_token = np.full((ROWS, COLS), MASK_ID, dtype=np.int64)
    assist_token = np.full((ROWS, COLS), MASK_ID, dtype=np.int64)
    visible = card.present & ~hidden.reshape(1, COLS)
    goal_token[visible] = card.goals[visible]
    assist_token[visible] = card.assists[visible]
    concede = np.zeros(COLS, dtype=np.float32)
    concede_missing = np.ones(COLS, dtype=np.float32)
    regulars = np.zeros(COLS, dtype=np.float32)
    regulars_missing = np.ones(COLS, dtype=np.float32)
    own_hist = np.zeros(COLS, dtype=np.float32)
    own_hist_missing = np.zeros(COLS, dtype=np.float32)
    opp_hist = np.zeros(COLS, dtype=np.float32)
    opp_hist_missing = np.zeros(COLS, dtype=np.float32)
    own_form = np.zeros(COLS, dtype=np.float32)
    own_form_missing = np.ones(COLS, dtype=np.float32)
    opponent_id = np.zeros(COLS, dtype=np.int64)
    column_on = (card.gw > 0).astype(np.float32)
    clubs = clubs or {}
    prior = prior_table(league)
    earlier = any(name < card.season for name in league)
    own = prior.get((card.season, card.team))
    if earlier and (own is None or own[2] == 0):
        own_hist_missing[:] = 1.0
    elif own is not None and own[2] > 0:
        own_hist[:] = own[0]
    cache: dict[str, set[int]] = {}
    for col in range(COLS):
        if card.gw[col] <= 0:
            continue
        is_hidden = bool(hidden[col])
        kickoff = int(card.kickoff[col])
        form = _rate_before(league, card.season, card.team, kickoff, cut_gw, is_hidden, "gf")
        if form is not None:
            own_form[col] = form
            own_form_missing[col] = 0.0
        opponent = card.opponent[col]
        opponent_id[col] = int(clubs.get(opponent, 0)) if opponent else 0
        if earlier:
            opp = prior.get((card.season, opponent)) if opponent else None
            if opp is None or opp[3] == 0:
                opp_hist_missing[col] = 1.0
            else:
                opp_hist[col] = opp[1]
        if not opponent or opponent not in league.get(card.season, {}):
            continue
        rate = _concede_rate(league, card.season, opponent, kickoff, cut_gw, is_hidden)
        if rate is not None:
            concede[col] = rate
            concede_missing[col] = 0.0
        if opponent not in cache:
            cache[opponent] = _regulars(league, card.season, opponent, cut_gw)
        regs = cache[opponent]
        if not regs:
            continue
        current = league[card.season][opponent]["by_id"].get(int(card.fixture_id[col]))
        if current is None:
            continue
        regulars[col] = len(regs & current["played1"]) / len(regs)
        regulars_missing[col] = 0.0
    loss = card.present & hidden.reshape(1, COLS)
    matchup = np.stack(
        [
            own_hist / 3.0,
            own_hist_missing,
            opp_hist / 3.0,
            opp_hist_missing,
            own_form / 3.0,
            own_form_missing,
            concede / 3.0,
            concede_missing,
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "minutes": card.minutes.astype(np.float32),
        "present": card.present.astype(np.float32),
        "goal_token": goal_token,
        "assist_token": assist_token,
        "position": card.position.astype(np.int64),
        "home": card.home.astype(np.float32),
        "concede": concede,
        "concede_missing": concede_missing,
        "regulars": regulars,
        "regulars_missing": regulars_missing,
        "own_hist": own_hist,
        "own_hist_missing": own_hist_missing,
        "opp_hist": opp_hist,
        "opp_hist_missing": opp_hist_missing,
        "own_form": own_form,
        "own_form_missing": own_form_missing,
        "matchup": matchup,
        "club": np.int64(clubs.get(card.team, 0)),
        "opponent": opponent_id,
        "column_on": column_on,
        "goal_label": card.goals.astype(np.int64),
        "assist_label": card.assists.astype(np.int64),
        "loss": loss.astype(np.float32),
        "row_pad": (card.position < 0),
        **_opponent_squad(card, cut_gw, squads),
    }


def _hatch(img: np.ndarray, y0: int, x0: int, height: int) -> None:
    ys = np.arange(height)[:, None] + y0
    xs = np.arange(CELL_W)[None, :] + x0
    mark = ((xs + ys) % 2) == 0
    img[y0 : y0 + height, x0 : x0 + CELL_W] = np.where(mark[..., None], HATCH_FG, HATCH_BG)


def _minutes_bar(img: np.ndarray, y0: int, x0: int, minutes: float) -> None:
    """Paint the minute count across two bands. Zero stays empty. No 60-minute step."""
    height = BAND_H * 2
    img[y0 : y0 + height, x0 : x0 + CELL_W] = 255
    img[y0 + height - 1, x0 : x0 + CELL_W] = BASE_C
    if minutes <= 0:
        return
    pixels = min(height, max(1, int(round(minutes / 90 * height))))
    img[y0 + height - pixels : y0 + height, x0 : x0 + CELL_W] = MINUTES_C


def event_pixels(count: int) -> int:
    """Map a goal or assist count onto the event band. Zero stays empty."""
    if count <= 0:
        return 0
    return min(EVENT_H, max(1, int(round(count / 3 * EVENT_H))))


def _bar(
    img: np.ndarray,
    y0: int,
    x0: int,
    pixels: int,
    color: tuple[int, int, int],
    height: int = BAND_H,
) -> None:
    img[y0 : y0 + height, x0 : x0 + CELL_W] = 255
    img[y0 + height - 1, x0 : x0 + CELL_W] = BASE_C
    if pixels > 0:
        pixels = min(int(pixels), height)
        img[y0 + height - pixels : y0 + height, x0 : x0 + CELL_W] = color


def render_grid(view: dict[str, np.ndarray]) -> np.ndarray:
    """Paint one card. Labels and names are not in the picture.

    Each player cell carries the minute count across sixteen pixels. Goals and
    assists each have their own sixteen-pixel band, and the count fills it.
    A position color sits under those bands. The conceded header uses the full
    eight-pixel band, so rates that shared a height before are separate.
    """
    img = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)
    for col in range(COLS):
        x0 = col * CELL_W
        if view["column_on"][col] < 0.5:
            _hatch(img, 0, x0, IMG_H)
            continue
        _bar(img, 0, x0, BAND_H, HOME_C if view["home"][col] > 0.5 else AWAY_C)
        if view["concede_missing"][col] > 0.5:
            _hatch(img, BAND_H, x0, BAND_H)
        else:
            pixels = int(round(min(float(view["concede"][col]), 3.0) / 3.0 * BAND_H))
            _bar(img, BAND_H, x0, pixels, CONCEDE_C)
        if view["regulars_missing"][col] > 0.5:
            _hatch(img, BAND_H * 2, x0, BAND_H)
        else:
            pixels = int(round(float(view["regulars"][col]) * BAND_H))
            _bar(img, BAND_H * 2, x0, pixels, REG_C)
        if view["own_hist_missing"][col] > 0.5:
            _hatch(img, BAND_H * 3, x0, BAND_H)
        else:
            pixels = int(round(min(float(view["own_hist"][col]), 3.0) / 3.0 * BAND_H))
            _bar(img, BAND_H * 3, x0, pixels, OWN_HIST_C)
        if view["opp_hist_missing"][col] > 0.5:
            _hatch(img, BAND_H * 4, x0, BAND_H)
        else:
            pixels = int(round(min(float(view["opp_hist"][col]), 3.0) / 3.0 * BAND_H))
            _bar(img, BAND_H * 4, x0, pixels, OPP_HIST_C)
        for row in range(ROWS):
            y = HEADER_H + row * PLAYER_H
            if view["position"][row] < 0:
                _hatch(img, y, x0, PLAYER_H)
                continue
            pos_y = y + MINUTE_H + 2 * EVENT_H
            img[pos_y : pos_y + POS_H, x0 : x0 + CELL_W] = POS_C[int(view["position"][row])]
            if view["present"][row, col] < 0.5:
                _hatch(img, y, x0, PLAYER_H - POS_H)
                continue
            _minutes_bar(img, y, x0, float(view["minutes"][row, col]))
            for offset, token, color in (
                (MINUTE_H, int(view["goal_token"][row, col]), GOALS_C),
                (MINUTE_H + EVENT_H, int(view["assist_token"][row, col]), ASSISTS_C),
            ):
                if token == MASK_ID:
                    img[y + offset : y + offset + EVENT_H, x0 : x0 + CELL_W] = MASK_C
                else:
                    _bar(img, y + offset, x0, event_pixels(token), color, EVENT_H)
    return img


class Coupler(nn.Module):
    """Attend across the squad, then across the weeks. The tokens are the picture."""

    def __init__(self, dropout: float, n_clubs: int = 1) -> None:
        super().__init__()
        del n_clubs
        self.row = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.col = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_row = nn.LayerNorm(D)
        self.norm_col = nn.LayerNorm(D)
        self.goal = nn.Linear(D, 4)
        self.assist = nn.Linear(D, 4)
        self.skip_goal = nn.Linear(D, 4)
        self.skip_assist = nn.Linear(D, 4)
        self.col_pos = nn.Parameter(torch.zeros(1, 1, COLS, D))
        nn.init.normal_(self.col_pos, std=0.02)
        for layer in (self.goal, self.assist, self.skip_goal, self.skip_assist):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        cells: torch.Tensor,
        row_pad: torch.Tensor,
        col_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _rows, cols, _dim = cells.shape
        hidden = self.norm_row(cells + self.col_pos[:, :, :cols])
        row_view = hidden.permute(0, 2, 1, 3).reshape(batch * cols, ROWS, D)
        row_mask = row_pad[:, None, :].expand(batch, cols, ROWS).reshape(batch * cols, ROWS)
        mixed, _ = self.row(row_view, row_view, row_view, key_padding_mask=row_mask, need_weights=False)
        row_view = self.norm_row(row_view + mixed)
        col_view = row_view.reshape(batch, cols, ROWS, D).permute(0, 2, 1, 3).reshape(batch * ROWS, cols, D)
        col_mask = col_pad[:, None, :].expand(batch, ROWS, cols).reshape(batch * ROWS, cols)
        mixed, _ = self.col(col_view, col_view, col_view, key_padding_mask=col_mask, need_weights=False)
        col_view = self.norm_col(col_view + mixed)
        out = col_view.reshape(batch, ROWS, cols, D)
        return self.goal(out) + self.skip_goal(cells), self.assist(out) + self.skip_assist(cells)


class SquadSequence(nn.Module):
    """Numeric reader. Hidden bars are an empty token."""

    def __init__(self, dropout: float = 0.1, n_clubs: int = 1) -> None:
        super().__init__()
        self.cell = nn.Linear(2, D)
        self.position = nn.Embedding(5, D)
        self.goal_in = nn.Embedding(5, D)
        self.assist_in = nn.Embedding(5, D)
        self.header = nn.Linear(4, D)
        self.couple = Coupler(dropout, n_clubs)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        stats = torch.stack([batch["minutes"] / 90.0, batch["present"]], dim=-1)
        cells = self.cell(stats)
        pos = torch.where(batch["row_pad"], torch.full_like(batch["position"], 4), batch["position"])
        cells = cells + self.position(pos)[:, :, None, :]
        cells = cells + self.goal_in(batch["goal_token"]) + self.assist_in(batch["assist_token"])
        header = self.header(
            torch.stack(
                [batch["home"], batch["concede"] / 3.0, batch["concede_missing"], batch["regulars"]],
                dim=-1,
            )
        )
        cells = cells + header[:, None, :, :]
        return self.couple(cells, batch["row_pad"], batch["column_on"] < 0.5)


class SquadVision(nn.Module):
    """Pixel reader. Each cell is a patch. The grid is the only input."""

    def __init__(self, dropout: float = 0.1, n_clubs: int = 1) -> None:
        super().__init__()
        self.header = nn.Linear(3 * HEADER_H * CELL_W, D)
        self.patch = nn.Linear(3 * PLAYER_H * CELL_W, D)
        self.couple = Coupler(dropout, n_clubs)

    def forward(
        self,
        images: torch.Tensor,
        row_pad: torch.Tensor,
        column_on: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        header = images[:, :, :HEADER_H, :].reshape(batch, 3, HEADER_H, COLS, CELL_W)
        header = header.permute(0, 3, 1, 2, 4).reshape(batch, COLS, -1)
        header_tok = self.header(header)
        body = images[:, :, HEADER_H:, :].reshape(batch, 3, ROWS, PLAYER_H, COLS, CELL_W)
        body = body.permute(0, 4, 2, 1, 3, 5).reshape(batch, COLS, ROWS, -1)
        cells = self.patch(body).permute(0, 2, 1, 3) + header_tok[:, None, :, :]
        return self.couple(cells, row_pad, column_on < 0.5)


def grey_except_minutes(img: np.ndarray, columns: np.ndarray) -> None:
    """Hatch a column except the minute bars and the venue band."""
    for col in np.asarray(columns, dtype=np.int64):
        x0 = int(col) * CELL_W
        _hatch(img, BAND_H, x0, HEADER_H - BAND_H)
        for row in range(ROWS):
            y = HEADER_H + row * PLAYER_H
            _hatch(img, y + MINUTE_H, x0, PLAYER_H - MINUTE_H)


def _stack(
    views: list[dict[str, np.ndarray]], grey_columns: list[np.ndarray] | None = None
) -> dict[str, torch.Tensor]:
    long_keys = {"goal_token", "assist_token", "goal_label", "assist_label", "position"}
    batch: dict[str, torch.Tensor] = {}
    for key in (
        "minutes", "present", "goal_token", "assist_token", "position", "home",
        "concede", "concede_missing", "regulars", "regulars_missing", "column_on",
        "goal_label", "assist_label", "loss",
    ):
        array = np.stack([view[key] for view in views])
        if key in long_keys:
            batch[key] = torch.from_numpy(array.astype(np.int64))
        else:
            batch[key] = torch.from_numpy(array.astype(np.float32))
    batch["row_pad"] = torch.from_numpy(np.stack([view["row_pad"] for view in views]))
    batch["club"] = torch.from_numpy(np.array([int(view["club"]) for view in views], dtype=np.int64))
    batch["opponent"] = torch.from_numpy(np.stack([view["opponent"] for view in views]).astype(np.int64))
    batch["matchup"] = torch.from_numpy(np.stack([view["matchup"] for view in views]).astype(np.float32))
    for key in ("opp_position", "opp_goal_token", "opp_assist_token"):
        batch[key] = torch.from_numpy(np.stack([view[key] for view in views]).astype(np.int64))
    for key in ("opp_minutes", "opp_present"):
        batch[key] = torch.from_numpy(np.stack([view[key] for view in views]).astype(np.float32))
    batch["opp_pad"] = torch.from_numpy(np.stack([view["opp_pad"] for view in views]))
    images = np.stack([render_grid(view) for view in views])
    if grey_columns is not None:
        for index, columns in enumerate(grey_columns):
            grey_except_minutes(images[index], columns)
    batch["image"] = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.0)
    return batch


def sample_cut(card: SquadCard, rng: np.random.Generator) -> int:
    """Reveal a prefix from gameweek 1: 20%, 40%, 60%, or 80% of the fixtures."""
    return cut_at_fraction(card, float(rng.choice((0.2, 0.4, 0.6, 0.8))))


def cut_at_fraction(card: SquadCard, revealed: float) -> int:
    """Gameweek of the last visible fixture when the first `revealed` share is shown."""
    n = card.n_fixtures
    visible = max(1, min(n - 1, int(round(revealed * n))))
    return int(card.gw[visible - 1])


def _nll(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1), reduction="none").reshape(labels.shape)
    weight = mask.sum().clamp_min(1.0)
    return (flat * mask).sum() / weight


def train_model(
    model: nn.Module,
    cards: list[SquadCard],
    league: dict,
    rng: np.random.Generator,
    epochs: int,
    kind: str,
    lr: float = LR,
    batch_size: int = BATCH,
    clubs: dict[str, int] | None = None,
) -> list[float]:
    fast_name = ("pos_", "position", "club_", "opp_", "form_")
    fast = [
        p
        for name, p in model.named_parameters()
        if any(token in name or name.startswith(token) for token in fast_name)
    ]
    fast_ids = {id(p) for p in fast}
    rest = [p for p in model.parameters() if id(p) not in fast_ids]
    groups = [{"params": rest, "lr": lr, "weight_decay": 1e-4}]
    if fast:
        groups.append({"params": fast, "lr": lr * 10, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    history = []
    model.train()
    squads = squad_index(cards)
    for epoch in range(epochs):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), batch_size):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [sample_cut(card, rng) for card in pick]
            batch = _stack(
                [
                    apply_cut(card, league, cut, clubs, squads)
                    for card, cut in zip(pick, cuts, strict=True)
                ]
            )
            opt.zero_grad()
            if kind == "vision":
                goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
            else:
                goal_logits, assist_logits = model(batch)
            mask = batch["loss"]
            loss = 0.5 * (_nll(goal_logits, batch["goal_label"], mask) + _nll(assist_logits, batch["assist_label"], mask))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(mask.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        epoch_nll = total / max(seen, 1.0)
        history.append(epoch_nll)
        print(f"{kind} epoch {epoch + 1} train nll {epoch_nll:.3f}", flush=True)
    return history


@torch.no_grad()
def predict(
    model: nn.Module,
    cards: list[SquadCard],
    league: dict,
    cut_gw: int,
    kind: str,
    clubs: dict[str, int] | None = None,
    squads: dict[tuple[str, str], SquadCard] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Goal and assist probabilities, shape (cards, rows, cols, 4)."""
    model.eval()
    squads = squads if squads is not None else squad_index(cards)
    goals = np.zeros((len(cards), ROWS, COLS, 4), dtype=np.float64)
    assists = np.zeros_like(goals)
    for start in range(0, len(cards), BATCH):
        pick = cards[start : start + BATCH]
        batch = _stack([apply_cut(card, league, cut_gw, clubs, squads) for card in pick])
        if kind == "vision":
            goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
        else:
            goal_logits, assist_logits = model(batch)
        goals[start : start + len(pick)] = F.softmax(goal_logits, dim=-1).numpy()
        assists[start : start + len(pick)] = F.softmax(assist_logits, dim=-1).numpy()
    return goals, assists


class GroupRate:
    """Position by minutes-bin frequencies from the training cards. No player id."""

    def __init__(self, cards: list[SquadCard]) -> None:
        self.counts = np.ones((2, 4, 3, 4), dtype=np.float64)
        for card in cards:
            self._add(card, card.gw > 0)

    def _add(self, card: SquadCard, columns: np.ndarray) -> None:
        bins = _minute_bin(card.minutes)
        for channel, labels in enumerate((card.goals, card.assists)):
            for row in range(card.n_players):
                pos = int(card.position[row])
                for col in np.where(columns & card.present[row])[0]:
                    self.counts[channel, pos, int(bins[row, col]), int(labels[row, col])] += 1

    def probability(self, card: SquadCard, channel: int) -> np.ndarray:
        out = np.zeros((ROWS, COLS, 4), dtype=np.float64)
        bins = _minute_bin(card.minutes)
        for row in range(ROWS):
            pos = int(card.position[row])
            if pos < 0:
                continue
            for col in range(COLS):
                dist = self.counts[channel, pos, int(bins[row, col])]
                out[row, col] = dist / dist.sum()
        return out


def _minute_bin(minutes: np.ndarray) -> np.ndarray:
    bins = np.zeros(minutes.shape, dtype=np.int64)
    bins[minutes >= 1] = 1
    bins[minutes >= 60] = 2
    return bins


def prefix_probability(card: SquadCard, cut_gw: int, group: GroupRate, channel: int) -> np.ndarray:
    """This player's visible bars, pulled toward the group rate with a fixed strength."""
    labels = card.goals if channel == 0 else card.assists
    visible = card.present & (card.gw.reshape(1, COLS) > 0) & (card.gw.reshape(1, COLS) <= cut_gw)
    bins = _minute_bin(card.minutes)
    out = np.zeros((ROWS, COLS, 4), dtype=np.float64)
    for row in range(ROWS):
        pos = int(card.position[row])
        if pos < 0:
            continue
        local = np.zeros((3, 4), dtype=np.float64)
        for col in np.where(visible[row])[0]:
            local[int(bins[row, col]), int(labels[row, col])] += 1
        for col in range(COLS):
            bin_id = int(bins[row, col])
            prior = group.counts[channel, pos, bin_id]
            prior = prior / prior.sum()
            mixed = local[bin_id] + PRIOR_STRENGTH * prior
            out[row, col] = mixed / mixed.sum()
    return out


def _channel_scores(
    probs: np.ndarray,
    labels: np.ndarray,
    cards: list[SquadCard],
    cut_gw: int,
) -> dict[str, float]:
    nll = 0.0
    weight = 0.0
    expect = 0.0
    actual = 0.0
    argmax = 0.0
    positive_p = 0.0
    positive_y = 0.0
    played_n = 0.0
    classes = np.arange(4)
    for index, card in enumerate(cards):
        hidden = (card.gw > cut_gw) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, COLS)
        if not mask.any():
            continue
        p = probs[index]
        picked = p[mask, labels[index][mask]]
        nll += float(-np.log(np.clip(picked, 1e-12, 1)).sum())
        weight += float(mask.sum())
        expect += float((p * classes).sum(axis=-1)[mask].sum())
        actual += float(labels[index][mask].sum())
        argmax += float(p.argmax(axis=-1)[mask].sum())
        positive_p += float((1.0 - p[:, :, 0])[mask].sum())
        positive_y += float((labels[index][mask] > 0).sum())
        played_n += float(mask.sum())
    return {
        "nll": nll / max(weight, 1.0),
        "expected_sum": expect / max(len(cards), 1),
        "actual_sum": actual / max(len(cards), 1),
        "argmax_sum": argmax / max(len(cards), 1),
        "positive_p": positive_p / max(played_n, 1.0),
        "positive_rate": positive_y / max(played_n, 1.0),
    }


def _capture(score: np.ndarray, hit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(-(score + rng.random(len(score)) * 1e-6))
    caught = []
    for frac in (0.1, 0.2, 0.5):
        take = max(1, int(round(frac * len(score))))
        caught.append(hit[order[:take]].sum() / hit.sum())
    return np.array(caught)


def capture_table(
    cards: list[SquadCard],
    scores: dict[str, np.ndarray],
    cut_gw: int,
    seeds: int = 20,
) -> dict[str, object]:
    """Share of a player's hidden goal weeks caught by ranking his appearances."""
    rows = []
    for index, card in enumerate(cards):
        hidden = (card.gw > cut_gw) & (card.gw > 0)
        for row in range(card.n_players):
            cols = np.where(hidden & card.present[row])[0]
            if len(cols) < 8:
                continue
            hit = card.goals[row, cols] > 0
            if int(hit.sum()) < 1:
                continue
            item = {"hit": hit.astype(np.float64), "minutes": card.minutes[row, cols].astype(np.float64)}
            for name, grid in scores.items():
                item[name] = grid[index, row, cols]
            rows.append(item)
    names = list(scores) + ["minutes"]
    totals = {name: np.zeros(3) for name in names}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for item in rows:
            for name in names:
                totals[name] += _capture(item[name], item["hit"], rng)
    scale = max(len(rows) * seeds, 1)
    return {"n_players": len(rows), **{name: (totals[name] / scale).tolist() for name in names}}


def appearance_hit_rate(
    cards: list[SquadCard],
    score: np.ndarray,
    cut_gw: int,
    seeds: int = 20,
) -> dict[str, float]:
    """Share of highlighted appearances that contain a goal. Same players as the capture table."""
    rows = []
    for index, card in enumerate(cards):
        hidden = (card.gw > cut_gw) & (card.gw > 0)
        for row in range(card.n_players):
            cols = np.where(hidden & card.present[row])[0]
            if len(cols) < 8:
                continue
            hit = card.goals[row, cols] > 0
            if int(hit.sum()) < 1:
                continue
            rows.append((hit.astype(np.float64), score[index, row, cols], card.minutes[row, cols]))
    base = float(np.mean([hit.mean() for hit, _, _ in rows])) if rows else 0.0
    chosen = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for hit, sc, _minutes in rows:
            order = np.argsort(-(sc + rng.random(len(sc)) * 1e-6))
            take = max(1, int(round(0.1 * len(hit))))
            chosen.append(float(hit[order[:take]].mean()))
    return {"players": len(rows), "base": base, "top10": float(np.mean(chosen)) if chosen else 0.0}


def positive_by_position(
    probs: np.ndarray,
    cards: list[SquadCard],
    cut_gw: int,
    cuts: list[int] | None = None,
) -> list[float]:
    """Mean probability of a bar, by position, on hidden appearances."""
    totals = np.zeros(4)
    counts = np.zeros(4)
    for index, card in enumerate(cards):
        cut = cuts[index] if cuts is not None else cut_gw
        hidden = (card.gw > cut) & (card.gw > 0)
        for row in range(card.n_players):
            pos = int(card.position[row])
            if pos < 0:
                continue
            cols = np.where(card.present[row] & hidden)[0]
            if len(cols) == 0:
                continue
            totals[pos] += float((1.0 - probs[index, row, cols, 0]).sum())
            counts[pos] += len(cols)
    return (totals / np.maximum(counts, 1.0)).tolist()


def forward_goal_rate(
    probs: np.ndarray, card: SquadCard, cut: int, columns: np.ndarray | None = None
) -> float:
    """Mean probability of a goal bar for forwards on the hidden appearances."""
    hidden = (card.gw > cut) & (card.gw > 0)
    if columns is not None:
        hidden = hidden & columns
    total = 0.0
    count = 0
    for row in range(card.n_players):
        if int(card.position[row]) != POS["FWD"]:
            continue
        cols = np.where(card.present[row] & hidden)[0]
        if len(cols) == 0:
            continue
        total += float((1.0 - probs[row, cols, 0]).sum())
        count += len(cols)
    if count == 0:
        return float("nan")
    return total / count


def club_remainder(
    cards: list[SquadCard],
    probs: np.ndarray,
    labels: np.ndarray,
    cuts: list[int],
    prior: dict,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Goal bars and the forward goal rate for one group of clubs."""
    if not cards:
        return {"prior_gf": 0.0, "actual_bars": 0, "drawn_bars": 0.0, "fwd": float("nan")}
    bars = bar_correspondence(cards, probs, labels, 0, rng, cuts=cuts)
    total = 0.0
    count = 0
    gf_sum = 0.0
    gf_n = 0
    for card, prob, cut in zip(cards, probs, cuts, strict=True):
        hidden = (card.gw > cut) & (card.gw > 0)
        for row in range(card.n_players):
            if int(card.position[row]) != POS["FWD"]:
                continue
            cols = np.where(card.present[row] & hidden)[0]
            if len(cols) == 0:
                continue
            total += float((1.0 - prob[row, cols, 0]).sum())
            count += len(cols)
        own = prior.get((card.season, card.team))
        if own is not None and own[2] > 0:
            gf_sum += own[0] * own[2]
            gf_n += own[2]
    return {
        "prior_gf": gf_sum / gf_n if gf_n else 0.0,
        "actual_bars": bars["actual_bars"],
        "drawn_bars": bars["drawn_bars"],
        "fwd": total / count if count else float("nan"),
    }


def sample_bars(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one bar height per cell from the class probabilities."""
    flat = np.asarray(probs, dtype=np.float64).reshape(-1, probs.shape[-1])
    cdf = np.cumsum(flat, axis=1)
    draw = rng.random(flat.shape[0])[:, None]
    classes = np.minimum((draw > cdf).sum(axis=1), probs.shape[-1] - 1)
    return classes.reshape(probs.shape[:-1]).astype(np.int64)


def _hidden_mask(cards: list[SquadCard], cut_gw: int, cuts: list[int] | None = None) -> np.ndarray:
    masks = []
    for index, card in enumerate(cards):
        cut = cuts[index] if cuts is not None else cut_gw
        hidden = (card.gw > cut) & (card.gw > 0)
        masks.append(card.present & hidden.reshape(1, COLS))
    return np.stack(masks)


def bar_correspondence(
    cards: list[SquadCard],
    probs: np.ndarray,
    labels: np.ndarray,
    cut_gw: int,
    rng: np.random.Generator,
    draws: int = 20,
    cuts: list[int] | None = None,
) -> dict[str, float]:
    """Count and week overlap between sampled bars and the actual bars.

    A bar is an appearance whose height is at least 1. Totals cover every
    hidden appearance of every club, which is the run of gameweeks after the cut.
    """
    mask = _hidden_mask(cards, cut_gw, cuts)
    actual = np.where(mask, labels, 0)
    actual_bars = int((actual > 0).sum())
    actual_height = int(actual.sum())
    drawn_bars = 0.0
    drawn_height = 0.0
    covered = 0.0
    on_actual = 0.0
    shared_height = 0.0
    for _ in range(draws):
        drawn = np.where(mask, sample_bars(probs, rng), 0)
        both = (actual > 0) & (drawn > 0)
        drawn_bars += float((drawn > 0).sum())
        drawn_height += float(drawn.sum())
        covered += float(both.sum())
        on_actual += float(both.sum())
        shared_height += float(np.minimum(actual, drawn).sum())
    return {
        "actual_bars": actual_bars,
        "drawn_bars": drawn_bars / draws,
        "actual_height": actual_height,
        "drawn_height": drawn_height / draws,
        "actual_covered": covered / draws / max(actual_bars, 1),
        "drawn_on_actual": on_actual / draws / max(drawn_bars / draws, 1.0),
        "shared_height": shared_height / draws / max(actual_height, 1),
    }


def random_same_count(
    cards: list[SquadCard],
    labels: np.ndarray,
    cut_gw: int,
    rng: np.random.Generator,
    draws: int = 20,
    cuts: list[int] | None = None,
) -> dict[str, float]:
    """Place each player's actual bars on random appearances of that player."""
    mask = _hidden_mask(cards, cut_gw, cuts)
    actual = np.where(mask, labels > 0, False)
    actual_bars = int(actual.sum())
    covered = 0.0
    for _ in range(draws):
        both = 0
        for index, card in enumerate(cards):
            for row in range(card.n_players):
                cols = np.where(mask[index, row])[0]
                hits = cols[actual[index, row, cols]]
                if len(hits) == 0 or len(cols) == 0:
                    continue
                picked = rng.choice(cols, size=len(hits), replace=False)
                both += int(np.isin(picked, hits).sum())
        covered += both
    rate = covered / draws / max(actual_bars, 1)
    return {
        "actual_bars": actual_bars,
        "drawn_bars": float(actual_bars),
        "actual_height": int(np.where(mask, labels, 0).sum()),
        "drawn_height": float(np.where(mask, labels, 0).sum()),
        "actual_covered": rate,
        "drawn_on_actual": rate,
        "shared_height": rate,
    }


def previous_name(payload: dict) -> str:
    return str(payload.get("previous_season", "the previous season"))


def _blank_nan(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.1%}"


def write_report(payload: dict[str, object], path) -> None:
    goals = payload["goals"]
    capture = payload["capture"]
    lines = [
        "# Squad-card trial",
        "",
        "Exploratory. One held-out season. Not a gated score comparison.",
        "2026-27 was not read.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        f"Cards: train {payload['n_train']}, test {payload['n_test']}. One card is a club-season.",
        f"Grid: {ROWS} players by {COLS} fixtures. Cut after gameweek {CUT_GW} for every club.",
        "Minutes, venue, opponent goals conceded, and regulars-on-the-pitch stay visible.",
        "Goals and assists after the cut are hidden. The segment is predicted in one pass.",
        f"Wall clock: {payload['seconds']:.0f} seconds on CPU.",
        "",
        "## Readers",
        "",
        "Position, the player's own minutes and visible bars, and the match-up",
        "each have a learned effect on the goal and assist heads.",
        "A club's earlier-season goals per game, and the opponent's earlier-season",
        "goals conceded per game, are shared across every card of that club.",
        "A club with no earlier season, once other clubs have one, is the promoted case.",
        "This season's own scoring rate and the opponent's conceding rate use only",
        "fixtures already played, and a hidden column stops at the cut.",
        "Each fixture is coupled to the other squad's players: position, minutes,",
        "and the goals and assists already shown. The same cut masks both sides.",
        (
            "Learned vision probability of a goal bar on a hidden appearance: "
            f"goalkeeper {payload['role_rate']['goal'][0]:.1%}, "
            f"defender {payload['role_rate']['goal'][1]:.1%}, "
            f"midfielder {payload['role_rate']['goal'][2]:.1%}, "
            f"forward {payload['role_rate']['goal'][3]:.1%}."
        ),
        "Sequence: the minute count, presence, position, and the visible bars as tokens.",
        "Vision: the same grid painted in pixels. Minutes use sixteen pixels, with no",
        "60-minute step, plus a position color and an eight-pixel opponent conceded bar.",
        "Both then attend across the 40 rows, across the other squad in that fixture,",
        "and across all 40 columns.",
        "The personal baseline is the player's visible bars pulled toward the position-and-minutes rate",
        f"with strength {PRIOR_STRENGTH}. The group baseline is that rate alone.",
        "",
        "## Hidden appearances",
        "",
        "Every recorded appearance is included, at whatever minutes. Lower negative log-likelihood is better.",
        "",
        "| reader | goal nll | assist nll | goal P | goal rate | expected goals | actual goals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("sequence", "vision", "prefix", "group"):
        goal = goals["goals"][name]
        assist = goals["assists"][name]
        lines.append(
            f"| {name} | {goal['nll']:.3f} | {assist['nll']:.3f} | {goal['positive_p']:.3f} | "
            f"{goal['positive_rate']:.3f} | {goal['expected_sum']:.2f} | {goal['actual_sum']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Where the goals land",
            "",
            f"Players with a goal and at least eight appearances after the cut: {capture['n_players']}.",
            "An appearance is any recorded minutes, including a zero-minute bench row.",
            f"A random appearance in that group is a goal {payload['hit_rate']['vision']['base']:.1%} of the time.",
            f"The vision model's top 10% of appearances are goals {payload['hit_rate']['vision']['top10']:.1%} of the time.",
            "",
            "| rank by | top 10% | top 20% | top 50% |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("sequence", "vision", "prefix", "group", "minutes"):
        vals = capture[name]
        lines.append(f"| {name} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |")
    lines.extend(
        [
            "| chance | 0.100 | 0.200 | 0.500 |",
            "",
            "Plot: `data/plots/flashcard_squad_trial.png`.",
            "",
            "## Bars across the hidden gameweeks",
            "",
            "A bar is an appearance with height at least 1. Each reader draws one by sampling",
            "its probabilities. The totals are every club, every appearance after the cut,",
            "averaged over 20 draws. Covered is the share of actual bars that sit on the same",
            "appearance as a drawn bar. On a real week is the share of drawn bars that sit on",
            "an actual bar. Same count, random weeks keeps each player's bar count and shuffles",
            "the appearances.",
            "",
            "| reader | channel | actual bars | drawn bars | covered | on a real week |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for channel in ("goals", "assists"):
        for name in ("vision", "sequence", "prefix", "group", "random"):
            item = payload["bars"][channel][name]
            lines.append(
                f"| {name} | {channel} | {item['actual_bars']:.0f} | {item['drawn_bars']:.1f} | "
                f"{item['actual_covered']:.1%} | {item['drawn_on_actual']:.1%} |"
            )
    example = payload["bar_example"]
    lines.extend(
        [
            "",
            f"One vision draw for {example['team']}, gameweeks {example['gw_from']}–{example['gw_to']}: "
            "`data/plots/flashcard_squad_bars.png`.",
            "",
            "## Remainders from gameweek 1",
            "",
            "Each card is shown from gameweek 1 through 80%, 60%, 40%, or 20% of its fixtures.",
            "The reader then draws the hidden remainder. Covered is the share of actual remainder",
            "bars that sit on the same appearance as a drawn bar. On a real week is the share of",
            "drawn bars that sit on an actual bar. The goalkeeper and forward columns are the",
            "model's probability of a goal bar on the hidden appearances.",
            "",
            "| revealed | reader | channel | actual bars | drawn bars | covered | on a real week | GK | FWD |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["reveals"]:
        for reader in ("vision", "sequence", "prefix"):
            for channel in ("goals", "assists"):
                bar = item[reader][channel]
                lines.append(
                    f"| {item['revealed']:.0%} | {reader} | {channel} | {bar['actual_bars']:.0f} | "
                    f"{bar['drawn_bars']:.1f} | {bar['actual_covered']:.1%} | {bar['drawn_on_actual']:.1%} | "
                    f"{item[reader]['gk']:.1%} | {item[reader]['fwd']:.1%} |"
                )
    lines.extend(
        [
            "",
            f"Four vision draws for {payload['reveal_card']}: `data/plots/flashcard_squad_reveals.png`.",
            "The title on each panel is that one draw for the whole club.",
            "",
            "## Club and match-up",
            "",
            "Vision's goal-bar rate for forwards on the hidden remainder. The club and",
            "match-up shift is learned separately for each position. Promoted clubs",
            f"are the sides missing from {previous_name(payload)}.",
            "A forward at a club that has been scoring, against a side that has been conceding,",
            "is where that shift should show.",
            "",
            "| revealed | group | prior goals/game | actual bars | drawn bars | forward goal rate |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in payload["club_balance"]:
        for key, label in (
            ("man_utd", payload["reveal_card"]),
            ("established", "established"),
            ("promoted", "promoted"),
        ):
            row = item[key]
            lines.append(
                f"| {item['revealed']:.0%} | {label} | {row['prior_gf']:.2f} | {row['actual_bars']:.0f} | "
                f"{row['drawn_bars']:.1f} | {_blank_nan(row['fwd'])} |"
            )
        lines.append(
            f"| {item['revealed']:.0%} | {payload['reveal_card']} vs promoted |  |  |  | "
            f"{_blank_nan(item['man_utd_vs_promoted'])} |"
        )
        lines.append(
            f"| {item['revealed']:.0%} | {payload['reveal_card']} vs established |  |  |  | "
            f"{_blank_nan(item['man_utd_vs_established'])} |"
        )
    names = ", ".join(payload["promoted_teams"]) or "none"
    lines.extend(
        [
            "",
            f"Promoted sides in {TEST_SEASON}, absent from the previous season: {names}.",
            "",
            "Forwards at 40% revealed, one club at a time.",
            "",
            "| club | prior goals/game | actual bars | drawn bars | forward goal rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["named_clubs"]:
        lines.append(
            f"| {row['team']} | {row['prior_gf']:.2f} | {row['actual_bars']:.0f} | "
            f"{row['drawn_bars']:.1f} | {_blank_nan(row['fwd'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(payload: dict[str, object], path) -> None:
    capture = payload["capture"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    names = ["sequence", "vision", "prefix", "group"]
    goal_nll = [payload["goals"]["goals"][name]["nll"] for name in names]
    axes[0].bar(names, goal_nll, color=["#e45756", "#4c78a8", "#54a24b", "#b0b0b0"])
    axes[0].set_ylabel("Appearance goal NLL")
    axes[0].set_title("Lower is better")
    fracs = [10, 20, 50]
    for name, color in (
        ("chance", "#888888"),
        ("minutes", "#4c78a8"),
        ("prefix", "#54a24b"),
        ("sequence", "#e45756"),
        ("vision", "#f58518"),
    ):
        vals = [10, 20, 50] if name == "chance" else [v * 100 for v in capture[name]]
        axes[1].plot(fracs, vals, marker="o", label=name, color=color)
    axes[1].set_xlabel("Appearances kept (%)")
    axes[1].set_ylabel("Goal weeks caught (%)")
    axes[1].set_title(f"{capture['n_players']} players, 2025-26 after GW19")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_bar_card(
    card: SquadCard,
    goal_actual: np.ndarray,
    goal_drawn: np.ndarray,
    assist_actual: np.ndarray,
    assist_drawn: np.ndarray,
    path,
) -> None:
    """One club, hidden gameweeks. Green is a bar on the same appearance."""
    hidden = np.where((card.gw > CUT_GW) & (card.gw > 0))[0]
    rows = [
        row
        for row in range(card.n_players)
        if (goal_actual[row, hidden] > 0).any()
        or (goal_drawn[row, hidden] > 0).any()
        or (assist_actual[row, hidden] > 0).any()
        or (assist_drawn[row, hidden] > 0).any()
    ]
    colors = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.96, 0.96, 0.96],
            [0.89, 0.43, 0.13],
            [0.30, 0.47, 0.66],
            [0.18, 0.62, 0.45],
        ]
    )

    def panel(actual: np.ndarray, drawn: np.ndarray) -> np.ndarray:
        grid = np.zeros((len(rows), len(hidden)), dtype=np.int64)
        for i, row in enumerate(rows):
            for j, col in enumerate(hidden):
                if not card.present[row, col]:
                    grid[i, j] = 1
                    continue
                has_actual = actual[row, col] > 0
                has_drawn = drawn[row, col] > 0
                grid[i, j] = 4 if has_actual and has_drawn else 2 if has_actual else 3 if has_drawn else 0
        return colors[grid]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, max(3.2, 0.28 * len(rows) + 1.4)), sharex=True)
    labels = [card.names[row].split()[-1][:16] for row in rows]
    gws = [str(int(card.gw[col])) for col in hidden]
    for axis, title, actual, drawn in (
        (axes[0], "Goals", goal_actual, goal_drawn),
        (axes[1], "Assists", assist_actual, assist_drawn),
    ):
        axis.imshow(panel(actual, drawn), aspect="auto", interpolation="nearest")
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_title(f"{card.team} {card.season}  {title}")
    axes[1].set_xticks(range(len(hidden)))
    axes[1].set_xticklabels(gws, fontsize=7)
    axes[1].set_xlabel("Gameweek. Orange is an actual bar, blue is a drawn bar, green is both.")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_reveals(card: SquadCard, draws: list[dict], path) -> None:
    """Four prefixes from gameweek 1. The line is where the revealed card ends.

    A row stays when that player scored, or when this draw gave them a goal bar.
    The title counts the whole club, including players who are not drawn as rows.
    """
    played = np.where(card.gw > 0)[0]
    drawn_any = np.zeros(card.n_players, dtype=bool)
    for item in draws:
        drawn_any |= (np.asarray(item["goals"])[: card.n_players, played] > 0).any(axis=1)
    rows = [
        row
        for row in range(card.n_players)
        if (card.goals[row, played] > 0).any() or drawn_any[row]
    ]
    colors = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.93, 0.93, 0.93],
            [0.89, 0.43, 0.13],
            [0.30, 0.47, 0.66],
            [0.18, 0.62, 0.45],
        ]
    )
    fig, axes = plt.subplots(len(draws), 1, figsize=(10.5, max(6.5, 0.32 * len(rows) * len(draws) * 0.45)), sharex=True)
    labels = [card.names[row].split()[-1][:16] for row in rows]
    gws = [str(int(card.gw[col])) for col in played]
    if len(draws) == 1:
        axes = [axes]
    for axis, item in zip(axes, draws, strict=True):
        grid = np.ones((len(rows), len(played)), dtype=np.int64)
        cut = item["cut"]
        for i, row in enumerate(rows):
            for j, col in enumerate(played):
                if card.gw[col] <= cut:
                    grid[i, j] = 1
                    continue
                has_actual = card.goals[row, col] > 0
                has_drawn = item["goals"][row, col] > 0
                grid[i, j] = 4 if has_actual and has_drawn else 2 if has_actual else 3 if has_drawn else 0
        axis.imshow(colors[grid], aspect="auto", interpolation="nearest")
        axis.axvline(int(np.sum(card.gw[played] <= cut)) - 0.5, color="black", linewidth=1)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels(labels, fontsize=6)
        hidden_cols = card.gw[played] > cut
        whole_actual = int(((card.goals[: card.n_players, played] > 0) & hidden_cols).sum())
        whole_drawn = int(((np.asarray(item["goals"])[: card.n_players, played] > 0) & hidden_cols).sum())
        axis.set_title(
            f"{card.team} goals, first {item['revealed']:.0%} revealed. "
            f"Whole club: {whole_actual} real, {whole_drawn} drawn"
        )
    axes[-1].set_xticks(range(len(played)))
    axes[-1].set_xticklabels(gws, fontsize=6)
    axes[-1].set_xlabel("Gameweek from 1. Left of the line is shown. Right is the drawn remainder.")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    print("loading squad cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    print(f"train {len(train)} test {len(test)}", flush=True)
    clubs = fit_clubs(train)
    test_squads = squad_index(test)
    n_clubs = max(clubs.values(), default=0) + 1
    torch.manual_seed(SEED)
    sequence = SquadSequence(dropout=0.1, n_clubs=n_clubs)
    vision = SquadVision(dropout=0.1, n_clubs=n_clubs)
    seq_hist = train_model(
        sequence, train, league, np.random.default_rng(SEED), EPOCHS, "sequence", clubs=clubs
    )
    vis_hist = train_model(
        vision, train, league, np.random.default_rng(SEED), EPOCHS, "vision", clubs=clubs
    )
    print("scoring", flush=True)
    seq_g, seq_a = predict(sequence, test, league, CUT_GW, "sequence", clubs)
    vis_g, vis_a = predict(vision, test, league, CUT_GW, "vision", clubs)
    group = GroupRate(train)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    prefix_g = np.stack([prefix_probability(card, CUT_GW, group, 0) for card in test])
    prefix_a = np.stack([prefix_probability(card, CUT_GW, group, 1) for card in test])
    group_g = np.stack([group.probability(card, 0) for card in test])
    group_a = np.stack([group.probability(card, 1) for card in test])
    goal_probs = {"sequence": seq_g, "vision": vis_g, "prefix": prefix_g, "group": group_g}
    assist_probs = {"sequence": seq_a, "vision": vis_a, "prefix": prefix_a, "group": group_a}
    goals = {"goals": {}, "assists": {}}
    for name in goal_probs:
        goals["goals"][name] = _channel_scores(goal_probs[name], labels, test, CUT_GW)
        goals["assists"][name] = _channel_scores(assist_probs[name], assist_labels, test, CUT_GW)
    capture_scores = {name: 1.0 - goal_probs[name][:, :, :, 0] for name in goal_probs}
    capture = capture_table(test, capture_scores, CUT_GW)
    hit_rate = {
        "vision": appearance_hit_rate(test, capture_scores["vision"], CUT_GW),
        "minutes": appearance_hit_rate(test, np.stack([card.minutes for card in test]), CUT_GW),
    }
    bar_rng = np.random.default_rng(SEED)
    bars = {"goals": {}, "assists": {}}
    for channel, probs, channel_labels in (
        ("goals", goal_probs, labels),
        ("assists", assist_probs, assist_labels),
    ):
        for name in ("vision", "sequence", "prefix", "group"):
            bars[channel][name] = bar_correspondence(
                test, probs[name], channel_labels, CUT_GW, bar_rng
            )
        bars[channel]["random"] = random_same_count(test, channel_labels, CUT_GW, bar_rng)
    example_index = int(
        np.argmax([(card.goals * ((card.gw > CUT_GW) & card.present)).sum() for card in test])
    )
    example_card = test[example_index]
    example_hidden = (example_card.gw > CUT_GW) & (example_card.gw > 0)
    role_rate = {
        "goal": positive_by_position(vis_g, test, CUT_GW),
        "assist": positive_by_position(vis_a, test, CUT_GW),
    }
    reveals = []
    reveal_draws = []
    club_balance = []
    prior = prior_table(league)
    previous = max(season for season in league if season < TEST_SEASON)
    previous_teams = set(league[previous])
    promoted_index = [index for index, card in enumerate(test) if card.team not in previous_teams]
    established_index = [index for index in range(len(test)) if index not in set(promoted_index)]
    show = next((card for card in test if card.team == "Man Utd"), test[example_index])
    show_at = test.index(show)
    named_clubs: list[dict] = []
    show_vs_promoted = np.array(
        [
            (opp_prior := prior.get((show.season, opp))) is None or opp_prior[3] == 0
            for opp in show.opponent
        ]
    )
    for revealed in (0.8, 0.6, 0.4, 0.2):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        entry = {"revealed": revealed}
        drawn_goal = None
        vision_probs = None
        for reader, model in (("vision", vision), ("sequence", sequence)):
            probs_g = np.zeros((len(test), ROWS, COLS, 4), dtype=np.float64)
            probs_a = np.zeros_like(probs_g)
            for index, card in enumerate(test):
                one_g, one_a = predict(model, [card], league, cuts[index], reader, clubs, test_squads)
                probs_g[index] = one_g[0]
                probs_a[index] = one_a[0]
            goal_bars = bar_correspondence(test, probs_g, labels, 0, bar_rng, cuts=cuts)
            assist_bars = bar_correspondence(test, probs_a, assist_labels, 0, bar_rng, cuts=cuts)
            rates = positive_by_position(probs_g, test, 0, cuts)
            entry[reader] = {"goals": goal_bars, "assists": assist_bars, "gk": rates[0], "fwd": rates[3]}
            if reader == "vision":
                vision_probs = probs_g
                drawn_goal = sample_bars(probs_g[show_at], np.random.default_rng(SEED))
        prefix_g = np.stack([prefix_probability(card, cut, group, 0) for card, cut in zip(test, cuts, strict=True)])
        prefix_a = np.stack([prefix_probability(card, cut, group, 1) for card, cut in zip(test, cuts, strict=True)])
        rates = positive_by_position(prefix_g, test, 0, cuts)
        entry["prefix"] = {
            "goals": bar_correspondence(test, prefix_g, labels, 0, bar_rng, cuts=cuts),
            "assists": bar_correspondence(test, prefix_a, assist_labels, 0, bar_rng, cuts=cuts),
            "gk": rates[0],
            "fwd": rates[3],
        }
        reveals.append(entry)
        reveal_draws.append({"revealed": revealed, "cut": cuts[show_at], "goals": drawn_goal})
        balance_rng = np.random.default_rng(SEED)
        man_row = club_remainder(
            [show],
            vision_probs[show_at : show_at + 1],
            labels[show_at : show_at + 1],
            [cuts[show_at]],
            prior,
            balance_rng,
        )
        promoted_row = club_remainder(
            [test[index] for index in promoted_index],
            vision_probs[promoted_index],
            labels[promoted_index],
            [cuts[index] for index in promoted_index],
            prior,
            balance_rng,
        )
        established_row = club_remainder(
            [test[index] for index in established_index],
            vision_probs[established_index],
            labels[established_index],
            [cuts[index] for index in established_index],
            prior,
            balance_rng,
        )
        balance = {
            "revealed": revealed,
            "man_utd": man_row,
            "promoted": promoted_row,
            "established": established_row,
            "man_utd_vs_promoted": forward_goal_rate(
                vision_probs[show_at], show, cuts[show_at], show_vs_promoted
            ),
            "man_utd_vs_established": forward_goal_rate(
                vision_probs[show_at], show, cuts[show_at], ~show_vs_promoted & (show.gw > 0)
            ),
        }
        club_balance.append(balance)
        if abs(revealed - 0.4) < 1e-9:
            for team in ("Man City", "Arsenal", "Liverpool", "Man Utd", "Leeds", "Burnley", "Sunderland"):
                found = next((index for index, card in enumerate(test) if card.team == team), None)
                if found is None:
                    continue
                row = club_remainder(
                    [test[found]],
                    vision_probs[found : found + 1],
                    labels[found : found + 1],
                    [cuts[found]],
                    prior,
                    balance_rng,
                )
                row["team"] = team
                named_clubs.append(row)
        vision_goals = entry["vision"]["goals"]
        print(
            f"reveal {revealed:.0%} vision goals actual {vision_goals['actual_bars']} "
            f"drawn {vision_goals['drawn_bars']:.1f} covered {vision_goals['actual_covered']:.3f} "
            f"gk {entry['vision']['gk']:.3f} fwd {entry['vision']['fwd']:.3f} "
            f"manutd {man_row['actual_bars']:.0f}/{man_row['drawn_bars']:.1f} "
            f"fwd {man_row['fwd']:.3f} promoted fwd {promoted_row['fwd']:.3f}",
            flush=True,
        )
    example_draw = np.random.default_rng(SEED)
    goal_drawn = sample_bars(vis_g[example_index], example_draw)
    assist_drawn = sample_bars(vis_a[example_index], example_draw)
    bar_example = {
        "team": example_card.team,
        "season": example_card.season,
        "gw_from": int(example_card.gw[example_hidden].min()),
        "gw_to": int(example_card.gw[example_hidden].max()),
    }
    elapsed = time.perf_counter() - started
    payload = {
        "n_train": len(train),
        "n_test": len(test),
        "seconds": elapsed,
        "sequence_history": seq_hist,
        "vision_history": vis_hist,
        "goals": goals,
        "capture": capture,
        "hit_rate": hit_rate,
        "bars": bars,
        "bar_example": bar_example,
        "role_rate": role_rate,
        "reveals": reveals,
        "reveal_card": show.team,
        "club_balance": club_balance,
        "promoted_teams": [test[index].team for index in promoted_index],
        "named_clubs": named_clubs,
        "previous_season": previous,
    }
    ROOT_PROCESSED.mkdir(parents=True, exist_ok=True)
    ROOT_REPORTS.mkdir(parents=True, exist_ok=True)
    (ROOT_PROCESSED / "flashcard_squad_trial.json").write_text(json.dumps(payload), encoding="utf-8")
    write_report(payload, ROOT_REPORTS / "flashcard_squad_trial.md")
    plot_results(payload, ROOT_PLOTS / "flashcard_squad_trial.png")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    plot_results(payload, ARTIFACTS / "squad_trial.png")
    plot_reveals(show, reveal_draws, ROOT_PLOTS / "flashcard_squad_reveals.png")
    plot_reveals(show, reveal_draws, ARTIFACTS / "squad_reveals.png")
    plot_bar_card(
        example_card,
        example_card.goals,
        goal_drawn,
        example_card.assists,
        assist_drawn,
        ROOT_PLOTS / "flashcard_squad_bars.png",
    )
    plot_bar_card(
        example_card,
        example_card.goals,
        goal_drawn,
        example_card.assists,
        assist_drawn,
        ARTIFACTS / "squad_bars.png",
    )
    print(f"done in {elapsed:.0f}s", flush=True)
    for name in ("sequence", "vision", "prefix", "group"):
        scored = goals["goals"][name]
        print(
            f"{name} goal nll {scored['nll']:.3f} expected {scored['expected_sum']:.2f} "
            f"actual {scored['actual_sum']:.2f} capture10 {capture[name][0]:.3f}",
            flush=True,
        )
    for channel in ("goals", "assists"):
        item = bars[channel]["vision"]
        print(
            f"vision {channel} bars actual {item['actual_bars']} drawn {item['drawn_bars']:.1f} "
            f"covered {item['actual_covered']:.3f} on_actual {item['drawn_on_actual']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
