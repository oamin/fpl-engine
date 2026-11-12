"""The locked card is the square card. The picture is the only feature input.

Concede, regulars, own history, opponent history, and the team scoring rate
are absent from the picture and from every tensor the reader sees. Minutes,
position, club, and opponent are on the card as squares. They are not passed
again as numbers. The odds row is sixteen squares: blue home, yellow draw,
purple away, from the Shin de-vig. The venue mark is one square. The opponent
is one square, a club symbol. Minutes are six squares of fifteen minutes. One
goal or assist is one square, and those rows are three squares tall. The
one-shot card paints the odds on every fixture, including a grey week. The
stepped reader uses this same card: true past, the odds, the venue, the
opponent, and the minutes on the next week, and every later column removed.
The other squad is not an input.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flashcard_card_score import _paint, paint_row, tally_cards
from src.models.flashcard_step_trial import chop_future
from src.models.flashcard_odds_trial import load_odds, paint_odds_strip
from src.models.flashcard_squad_trial import (
    BAND_H,
    BATCH,
    CELL_W,
    COLS,
    D,
    EPOCHS,
    HEADER_H,
    LR,
    PLAYER_H,
    ROWS,
    SEED,
    Coupler,
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
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
FEATURE_KEYS = (
    "concede",
    "concede_missing",
    "regulars",
    "regulars_missing",
    "own_hist",
    "own_hist_missing",
    "opp_hist",
    "opp_hist_missing",
)


def strip_features(view: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Zero the removed header rows and the whole numeric match-up, including the rate."""
    for key in FEATURE_KEYS:
        view[key] = np.zeros_like(view[key])
    if "own_form" in view:
        view["own_form"] = np.zeros_like(view["own_form"])
        view["own_form_missing"] = np.zeros_like(view["own_form_missing"])
    matchup = np.array(view["matchup"], copy=True)
    matchup[:, :] = 0
    view["matchup"] = matchup
    return view


def compose_stripped(
    view: dict[str, np.ndarray],
    grey_columns: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    visible: np.ndarray | None = None,
) -> np.ndarray:
    """Venue, then the squad. The odds bar, when given, sits above the venue."""
    image = render_grid(view)
    if grey_columns is not None:
        grey_except_minutes(image, grey_columns)
    image = np.concatenate([image[:BAND_H], image[HEADER_H:]], axis=0)
    if probs is None or visible is None:
        return image
    strip = paint_odds_strip(view["column_on"], probs, visible)
    return np.concatenate([strip, image], axis=0)


def compose_card(
    view: dict[str, np.ndarray],
    card,
    odds: dict,
    grey_columns: np.ndarray | None = None,
    target: int | None = None,
) -> np.ndarray:
    """The default picture: the square card.

    A one-shot card paints the odds on every fixture. A stepped card, given
    `target`, paints them on the true past and on that next week.
    """
    from src.models.flashcard_square_trial import compose_square

    return compose_square(view, card, odds, grey_columns, target)


class StripVision(nn.Module):
    """Pixel reader for a header that is venue, or odds plus venue."""

    def __init__(
        self,
        header_h: int,
        dropout: float = 0.1,
        n_clubs: int = 1,
        player_h: int = PLAYER_H,
    ) -> None:
        super().__init__()
        self.header_h = int(header_h)
        self.player_h = int(player_h)
        del n_clubs
        self.header = nn.Linear(3 * self.header_h * CELL_W, D)
        self.patch = nn.Linear(3 * self.player_h * CELL_W, D)
        self.couple = Coupler(dropout)

    def forward(
        self,
        images: torch.Tensor,
        row_pad: torch.Tensor,
        column_on: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        header = images[:, :, : self.header_h, :].reshape(batch, 3, self.header_h, COLS, CELL_W)
        header = header.permute(0, 3, 1, 2, 4).reshape(batch, COLS, -1)
        header_tok = self.header(header)
        body = images[:, :, self.header_h :, :].reshape(batch, 3, ROWS, self.player_h, COLS, CELL_W)
        body = body.permute(0, 4, 2, 1, 3, 5).reshape(batch, COLS, ROWS, -1)
        cells = self.patch(body).permute(0, 2, 1, 3) + header_tok[:, None, :, :]
        return self.couple(cells, row_pad, column_on < 0.5)


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    fast_name = ("pos_", "position", "club_", "opp_", "form_")
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    fast = [param for name, param in named if any(token in name or name.startswith(token) for token in fast_name)]
    fast_ids = {id(param) for param in fast}
    rest = [param for _, param in named if id(param) not in fast_ids]
    groups = [{"params": rest, "lr": LR, "weight_decay": 1e-4}]
    if fast:
        groups.append({"params": fast, "lr": LR * 10, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def _paint_one(
    view: dict,
    card,
    odds: dict | None,
    grey_columns: np.ndarray | None,
    target: int | None,
    compose,
) -> np.ndarray:
    """The locked card, or a painter passed in for one trial."""
    if compose is not None:
        return compose(view, card, odds, grey_columns, target)
    if odds is None:
        return compose_stripped(view, grey_columns)
    return compose_card(view, card, odds, grey_columns, target)


def _batch_images(views: list[dict], cards: list, odds: dict | None, compose=None) -> np.ndarray:
    pictures = [
        _paint_one(view, card, odds, None, None, compose)
        for view, card in zip(views, cards, strict=True)
    ]
    return np.stack(pictures)


def train_stripped(
    model: nn.Module,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    odds: dict | None,
    name: str,
    compose=None,
) -> None:
    opt = _optimizer(model)
    squads = squad_index(cards)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [sample_cut(card, rng) for card in pick]
            views = [strip_features(apply_cut(card, league, cut, clubs, squads)) for card, cut in zip(pick, cuts, strict=True)]
            batch = _stack(views)
            batch["image"] = torch.from_numpy(_batch_images(views, pick, odds, compose)).permute(0, 3, 1, 2).float().div(255.0)
            opt.zero_grad()
            goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
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
        print(f"{name} epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f}", flush=True)


def paint_stripped(
    model: nn.Module,
    cards: list,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict | None,
    cuts: list[int],
    compose=None,
) -> tuple[np.ndarray, np.ndarray]:
    views = [strip_features(apply_cut(card, league, cut, clubs, squads)) for card, cut in zip(cards, cuts, strict=True)]
    batch = _stack(views)
    batch["image"] = torch.from_numpy(_batch_images(views, cards, odds, compose)).permute(0, 3, 1, 2).float().div(255.0)
    model.eval()
    with torch.no_grad():
        goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
    return F.softmax(goal_logits, dim=-1).numpy(), F.softmax(assist_logits, dim=-1).numpy()


def _step_view(card, league, clubs, squads, target: int) -> dict:
    """True past, then one grey week. The four header rows are already gone."""
    view = apply_cut(card, league, target - 1, clubs, squads)
    chop_future(view, card.gw, target)
    strip_features(view)
    view["loss"] = np.zeros_like(view["loss"])
    view["loss"][card.present & (card.gw == target).reshape(1, -1)] = 1.0
    return view


def _step_image(view: dict, card, odds: dict, target: int, compose=None) -> np.ndarray:
    """The same card, cut to one next week, with the odds bar through that week."""
    return _paint_one(view, card, odds, np.flatnonzero(card.gw == target), target, compose)


def _gameweeks(cards: list) -> dict[int, list[int]]:
    by_gw: dict[int, list[int]] = {}
    for index, card in enumerate(cards):
        seen: set[int] = set()
        for gw in card.gw:
            number = int(gw)
            if number > 0 and number not in seen:
                seen.add(number)
                by_gw.setdefault(number, []).append(index)
    return by_gw


def _target_gw(card, rng: np.random.Generator) -> int:
    return int(rng.choice(sorted(_gameweeks([card]))))


def train_stripped_steps(
    model: nn.Module,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    odds: dict,
    compose=None,
) -> None:
    """One random gameweek per card. The loss is only that week."""
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
            pictures = []
            for card in pick:
                target = _target_gw(card, rng)
                view = _step_view(card, league, clubs, squads, target)
                views.append(view)
                pictures.append(_step_image(view, card, odds, target, compose))
            batch = _stack(views)
            batch["image"] = torch.from_numpy(np.stack(pictures)).permute(0, 3, 1, 2).float().div(255.0)
            opt.zero_grad()
            goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
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
        print(f"stripped step epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f}", flush=True)


def paint_stripped_steps(
    model: nn.Module,
    cards: list,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict,
    compose=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk every gameweek with the true past and keep that week's bars."""
    goals = np.zeros((len(cards),) + cards[0].goals.shape, dtype=np.int64)
    assists = np.zeros_like(goals)
    model.eval()
    for gw, indexes in sorted(_gameweeks(cards).items()):
        views = []
        pictures = []
        for index in indexes:
            card = cards[index]
            view = _step_view(card, league, clubs, squads, gw)
            views.append(view)
            pictures.append(_step_image(view, card, odds, gw, compose))
        batch = _stack(views)
        batch["image"] = torch.from_numpy(np.stack(pictures)).permute(0, 3, 1, 2).float().div(255.0)
        with torch.no_grad():
            goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
        goal_p = F.softmax(goal_logits, dim=-1).numpy()
        assist_p = F.softmax(assist_logits, dim=-1).numpy()
        for slot, index in enumerate(indexes):
            card = cards[index]
            mask = card.present & (card.gw == gw).reshape(1, -1)
            goals[index] += paint_row(goal_p[slot], mask)
            assists[index] += paint_row(assist_p[slot], mask)
    return goals, assists


def _row(revealed: float, name: str, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | {name} | {channel} | {totals['actual']} | {totals['painted']} | "
        f"{totals['exact']} | {totals['actual_within']} | {totals['painted_within']} |"
    )


def main() -> None:
    started = time.perf_counter()
    print("loading stripped cards", flush=True)
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
        "# Header rows removed",
        "",
        "The default card is the square card. The odds row is sixteen squares,",
        "the venue is one square, and minutes are six squares of fifteen",
        "minutes. One event is one square. It drops concede, regulars, own",
        "history, opponent history, and the team scoring rate. The reader",
        "sees the picture only. The no-odds reader is the",
        "comparison. The stepped reader uses the same square card, asks for",
        "one gameweek, and removes every later column.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    from src.models.flashcard_square_trial import HEADER_H as SQUARE_HEADER
    from src.models.flashcard_square_trial import PLAYER_H as SQUARE_PLAYER

    readers = (
        ("stripped", StripVision(BAND_H, dropout=0.1, n_clubs=n_clubs), None),
        ("square", StripVision(SQUARE_HEADER, dropout=0.1, n_clubs=n_clubs, player_h=SQUARE_PLAYER), odds),
    )
    for name, model, prices in readers:
        print(f"training {name}", flush=True)
        torch.manual_seed(SEED)
        train_stripped(model, train, league, np.random.default_rng(SEED), clubs, prices, name)
        for revealed in (0.2, 0.4, 0.6, 0.8):
            cuts = [cut_at_fraction(card, revealed) for card in test]
            goal_p, assist_p = paint_stripped(model, test, league, clubs, squads, prices, cuts)
            for channel, probs, actual in (("goals", goal_p, labels), ("assists", assist_p, assist_labels)):
                painted = _paint(probs, minutes, test, cuts)
                totals = tally_cards(test, actual, painted, 0, cuts)
                lines.append(_row(revealed, name, channel, totals))
                print(
                    f"{name} {revealed:.0%} {channel} painted {totals['painted']} "
                    f"exact {totals['exact']} within {totals['actual_within']}",
                    flush=True,
                )
    print("training stripped step", flush=True)
    torch.manual_seed(SEED)
    stepper = StripVision(SQUARE_HEADER, dropout=0.1, n_clubs=n_clubs, player_h=SQUARE_PLAYER)
    train_stripped_steps(stepper, train, league, np.random.default_rng(SEED), clubs, odds)
    step_g, step_a = paint_stripped_steps(stepper, test, league, clubs, squads, odds)
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        for channel, painted, actual in (("goals", step_g, labels), ("assists", step_a, assist_labels)):
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, "stripped step", channel, totals))
            print(
                f"stripped step {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "The default picture is the square card. The other squad is not an input.",
            "The gate is the full painter in reports/flashcard_square_trial.md.",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    path = ROOT_REPORTS / "flashcard_strip_trial.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
