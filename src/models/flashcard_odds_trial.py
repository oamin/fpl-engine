"""Train the next-gameweek task, then paint opening match odds.

The stepped card shows the true past, greys the next gameweek except minutes
and the venue, and chops every later column. Its odds bar is on the known
weeks and that next week.

The full painter still fills the whole remainder in one pass. Its odds bar
is on every known week and on every grey week still on the card.

Odds are opening 1X2 prices, turned into probabilities with the Shin de-vig.
The bar runs from the top: blue home, yellow draw, purple away. Those three
stay off the venue green and amber, and off the hatch grey. A missing price
or a week outside the horizon is hatched.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ingest.fpl_odds import load_football_data
from src.models.flashcard_calibrated_trial import place_expected
from src.models.flashcard_card_score import _paint, tally_cards
from src.models.flashcard_step_trial import chop_future, paint_steps
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    BAND_H,
    BATCH,
    CELL_W,
    COLS,
    EPOCHS,
    HEADER_H,
    IMG_W,
    LR,
    PLAYER_H,
    ROWS,
    SEED,
    Coupler,
    SquadVision,
    _hatch,
    _nll,
    _stack,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    grey_except_minutes,
    render_grid,
    sample_cut,
    squad_index,
)
from src.teams import norm_team

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
ODDS_H = HEADER_H + BAND_H
HOME_ODDS = (32, 84, 220)
DRAW_ODDS = (255, 214, 64)
AWAY_ODDS = (156, 40, 186)
SEASON_CODE = {
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}


def split_heights(probs: np.ndarray, height: int) -> np.ndarray:
    """Pixel counts that sum to `height`, in probability order."""
    raw = np.asarray(probs, dtype=np.float64)
    raw = raw / raw.sum()
    exact = raw * height
    floors = np.floor(exact + 1e-9).astype(np.int64)
    left = height - int(floors.sum())
    if left > 0:
        for index in np.argsort(-(exact - floors))[:left]:
            floors[index] += 1
    return floors


def paint_odds_bar(img: np.ndarray, x0: int, probs: np.ndarray) -> None:
    """Blue, yellow, and purple from the top of an 8-pixel bar."""
    y = 0
    for color, count in zip((HOME_ODDS, DRAW_ODDS, AWAY_ODDS), split_heights(probs, BAND_H), strict=True):
        if count <= 0:
            continue
        img[y : y + count, x0 : x0 + CELL_W] = color
        y += int(count)


def visible_odds(gameweeks: np.ndarray, kind: str, cut: int | None = None, target: int | None = None) -> np.ndarray:
    """Columns whose match odds may be painted."""
    show = np.zeros(len(gameweeks), dtype=bool)
    if kind == "step":
        show[(gameweeks > 0) & (gameweeks <= int(target))] = True
        return show
    show[gameweeks > 0] = True
    return show


def load_odds() -> dict[tuple[str, str, str, str], tuple[float, float, float]]:
    """Opening Shin probabilities keyed by season, date, home, away."""
    table: dict[tuple[str, str, str, str], tuple[float, float, float]] = {}
    for season, code in SEASON_CODE.items():
        frame = load_football_data(code=code)
        for row in frame.itertuples(index=False):
            table[(season, str(row.date), str(row.home_norm), str(row.away_norm))] = (
                float(row.p_home),
                float(row.p_draw),
                float(row.p_away),
            )
    return table


def column_odds(card, table: dict) -> np.ndarray:
    """Home, draw, away probabilities. Missing prices are NaN."""
    probs = np.full((COLS, 3), np.nan, dtype=np.float64)
    for col in range(card.n_fixtures):
        if int(card.gw[col]) <= 0 or not card.opponent[col]:
            continue
        date = pd.to_datetime(int(card.kickoff[col]), utc=True).strftime("%Y-%m-%d")
        team = norm_team(card.team)
        opponent = norm_team(card.opponent[col])
        if int(card.home[col]) == 1:
            key = (card.season, date, team, opponent)
        else:
            key = (card.season, date, opponent, team)
        found = table.get(key)
        if found is not None:
            probs[col] = found
    return probs


def paint_odds_strip(column_on: np.ndarray, probs: np.ndarray, visible: np.ndarray) -> np.ndarray:
    strip = np.full((BAND_H, IMG_W, 3), 255, dtype=np.uint8)
    for col in range(COLS):
        x0 = col * CELL_W
        if column_on[col] < 0.5 or not visible[col] or not np.isfinite(probs[col, 0]):
            _hatch(strip, 0, x0, BAND_H)
            continue
        paint_odds_bar(strip, x0, probs[col])
    return strip


def compose_image(view: dict, grey_columns: np.ndarray | None, probs: np.ndarray | None, visible: np.ndarray | None) -> np.ndarray:
    image = render_grid(view)
    if grey_columns is not None:
        grey_except_minutes(image, grey_columns)
    if probs is None or visible is None:
        return image
    strip = paint_odds_strip(view["column_on"], probs, visible)
    return np.concatenate([strip, image], axis=0)


class OddsVision(nn.Module):
    """Pixel reader whose header patch includes the odds row."""

    def __init__(self, dropout: float = 0.1, n_clubs: int = 1) -> None:
        super().__init__()
        del n_clubs
        self.header = nn.Linear(3 * ODDS_H * CELL_W, 32)
        self.patch = nn.Linear(3 * PLAYER_H * CELL_W, 32)
        self.couple = Coupler(dropout)

    def forward(
        self,
        images: torch.Tensor,
        row_pad: torch.Tensor,
        column_on: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        header = images[:, :, :ODDS_H, :].reshape(batch, 3, ODDS_H, COLS, CELL_W)
        header = header.permute(0, 3, 1, 2, 4).reshape(batch, COLS, -1)
        header_tok = self.header(header)
        body = images[:, :, ODDS_H:, :].reshape(batch, 3, ROWS, PLAYER_H, COLS, CELL_W)
        body = body.permute(0, 4, 2, 1, 3, 5).reshape(batch, COLS, ROWS, -1)
        cells = self.patch(body).permute(0, 2, 1, 3) + header_tok[:, None, :, :]
        return self.couple(cells, row_pad, column_on < 0.5)


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    fast_name = ("pos_", "position", "club_", "opp_", "form_")
    fast = [p for name, p in model.named_parameters() if any(token in name or name.startswith(token) for token in fast_name)]
    fast_ids = {id(p) for p in fast}
    rest = [p for p in model.parameters() if id(p) not in fast_ids]
    groups = [{"params": rest, "lr": LR, "weight_decay": 1e-4}]
    if fast:
        groups.append({"params": fast, "lr": LR * 10, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def _target_gw(card, rng: np.random.Generator) -> int:
    seen: list[int] = []
    for gw in card.gw:
        number = int(gw)
        if number > 0 and number not in seen:
            seen.append(number)
    return int(rng.choice(seen))


def _step_view(card, league, clubs, squads, target: int) -> dict:
    view = apply_cut(card, league, target - 1, clubs, squads)
    chop_future(view, card.gw, target)
    view["loss"] = np.zeros_like(view["loss"])
    view["loss"][card.present & (card.gw == target).reshape(1, -1)] = 1.0
    return view


def _run_batch(model: nn.Module, views: list[dict], images: np.ndarray | None, grey: list[np.ndarray] | None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    batch = _stack(views, grey)
    if images is not None:
        batch["image"] = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.0)
    goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
    return goal_logits, assist_logits, batch


def train_loop(model: nn.Module, cards: list, league: dict, rng: np.random.Generator, clubs: dict, odds: dict | None, kind: str) -> None:
    opt = _optimizer(model)
    squads = squad_index(cards)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            views = []
            greys = []
            pictures = []
            use_odds = odds is not None
            for card in pick:
                if kind == "step":
                    target = _target_gw(card, rng)
                    view = _step_view(card, league, clubs, squads, target)
                    grey_cols = np.flatnonzero(card.gw == target)
                    visible = visible_odds(card.gw, "step", target=target)
                else:
                    cut = sample_cut(card, rng)
                    view = apply_cut(card, league, cut, clubs, squads)
                    grey_cols = None
                    visible = visible_odds(card.gw, "painter", cut=cut)
                views.append(view)
                greys.append(grey_cols if grey_cols is not None else np.zeros(0, dtype=np.int64))
                if use_odds:
                    pictures.append(compose_image(view, grey_cols, column_odds(card, odds), visible))
            grey_arg = greys if kind == "step" else None
            images = np.stack(pictures) if use_odds else None
            opt.zero_grad()
            goal_logits, assist_logits, batch = _run_batch(model, views, images, grey_arg)
            mask = batch["loss"]
            loss = 0.5 * (
                _nll(goal_logits, batch["goal_label"], mask) + _nll(assist_logits, batch["assist_label"], mask)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(mask.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        print(f"{kind}{' odds' if odds is not None else ''} epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f}", flush=True)


def _forward_probs(model: nn.Module, views: list[dict], images: np.ndarray | None, grey: list[np.ndarray] | None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        goal_logits, assist_logits, _batch = _run_batch(model, views, images, grey)
    return F.softmax(goal_logits, dim=-1).numpy(), F.softmax(assist_logits, dim=-1).numpy()


def paint_odds_steps(model: nn.Module, cards: list, league: dict, clubs: dict, squads: dict, odds: dict) -> tuple[np.ndarray, np.ndarray]:
    goals = np.zeros((len(cards),) + cards[0].goals.shape, dtype=np.int64)
    assists = np.zeros_like(goals)
    by_gw: dict[int, list[int]] = {}
    for index, card in enumerate(cards):
        seen: set[int] = set()
        for gw in card.gw:
            number = int(gw)
            if number > 0 and number not in seen:
                seen.add(number)
                by_gw.setdefault(number, []).append(index)
    for gw, indexes in sorted(by_gw.items()):
        views = []
        greys = []
        pictures = []
        for index in indexes:
            card = cards[index]
            view = _step_view(card, league, clubs, squads, gw)
            grey_cols = np.flatnonzero(card.gw == gw)
            views.append(view)
            greys.append(grey_cols)
            pictures.append(
                compose_image(view, grey_cols, column_odds(card, odds), visible_odds(card.gw, "step", target=gw))
            )
        goal_p, assist_p = _forward_probs(model, views, np.stack(pictures), greys)
        for slot, index in enumerate(indexes):
            card = cards[index]
            mask = card.present & (card.gw == gw).reshape(1, -1)
            goals[index] += place_expected(goal_p[slot], card.minutes, mask)
            assists[index] += place_expected(assist_p[slot], card.minutes, mask)
    return goals, assists


def paint_odds_remainder(model: nn.Module, cards: list, league: dict, clubs: dict, squads: dict, odds: dict, cuts: list[int]) -> tuple[np.ndarray, np.ndarray]:
    views = []
    pictures = []
    for card, cut in zip(cards, cuts, strict=True):
        view = apply_cut(card, league, cut, clubs, squads)
        views.append(view)
        pictures.append(compose_image(view, None, column_odds(card, odds), visible_odds(card.gw, "painter", cut=cut)))
    return _forward_probs(model, views, np.stack(pictures), None)


def _row(revealed: float, name: str, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | {name} | {channel} | {totals['actual']} | {totals['painted']} | "
        f"{totals['exact']} | {totals['actual_within']} | {totals['painted_within']} |"
    )


def _score_steps(name: str, goals: np.ndarray, assists: np.ndarray, test: list, labels: np.ndarray, assist_labels: np.ndarray, lines: list[str]) -> None:
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        for channel, painted, actual in (("goals", goals, labels), ("assists", assists, assist_labels)):
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, name, channel, totals))
            print(
                f"{name} {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )


def _report(lines: list[str]) -> None:
    path = ROOT_REPORTS / "flashcard_odds_trial.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    print("loading odds cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(train + test)
    odds = load_odds()
    n_clubs = max(clubs.values(), default=0) + 1
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    lines = [
        "# Stepped training and match odds",
        "",
        "The gate paints the whole remainder at once and has no odds bar.",
        "Stepped training shows the true past and asks only for the next gameweek,",
        "with every later column removed. That week keeps minutes and the venue.",
        "The odds bar is opening 1X2 prices after",
        "the Shin de-vig: blue home, yellow draw, purple away, from the top.",
        "The stepped card shows that bar on known weeks and the step week.",
        "The full painter shows it on every known week and every grey week.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    print("training stepped reader", flush=True)
    torch.manual_seed(SEED)
    stepped = SquadVision(dropout=0.1, n_clubs=n_clubs)
    train_loop(stepped, train, league, np.random.default_rng(SEED), clubs, None, "step")
    step_g, step_a = paint_steps(stepped, test, league, clubs, squads)
    _score_steps("stepped", step_g, step_a, test, labels, assist_labels, lines)
    print("training stepped reader with odds", flush=True)
    torch.manual_seed(SEED)
    stepped_odds = OddsVision(dropout=0.1, n_clubs=n_clubs)
    train_loop(stepped_odds, train, league, np.random.default_rng(SEED), clubs, odds, "step")
    odds_g, odds_a = paint_odds_steps(stepped_odds, test, league, clubs, squads, odds)
    _score_steps("stepped odds", odds_g, odds_a, test, labels, assist_labels, lines)
    print("training full painter with odds", flush=True)
    torch.manual_seed(SEED)
    painter = OddsVision(dropout=0.1, n_clubs=n_clubs)
    train_loop(painter, train, league, np.random.default_rng(SEED), clubs, odds, "painter")
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p, assist_p = paint_odds_remainder(painter, test, league, clubs, squads, odds, cuts)
        for channel, probs, actual in (("goals", goal_p, labels), ("assists", assist_p, assist_labels)):
            painted = _paint(probs, minutes, test, cuts)
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, "painter odds", channel, totals))
            print(
                f"painter odds {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "The gate is the square card's full painter, and the other squad is",
            "not an input. At 20% revealed it puts 192 goal bars on the same",
            "appearance and 256 within one gameweek.",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    _report(lines)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
