"""The locked card is the square card. The picture is the only feature input.

Concede, regulars, own history, opponent history, and the team scoring rate
are absent from the picture and from every tensor the reader sees. Minutes,
position, club, and opponent are on the card as squares. They are not passed
again as numbers. The match rows, from the top, are the assessed club's symbol, the opponent's
symbol, home or away, then sixteen odds squares: blue home, yellow draw,
purple away, from the Shin de-vig. The club symbol repeats on every column.
Minutes are six squares of fifteen minutes. One
goal or assist is one square, and those rows are three squares tall. The
one-shot card paints the odds on every fixture, including a grey week. The
stepped reader uses this same card: true past, the club, the opponent, the
venue, the odds, and the minutes on the next week, and every later column removed.
The other squad is not an input. Minutes, goals, assists, and the position
square each have their own embedding. One vector for the whole cell lets
the minute block, which is most of the pixels, own the forecast. The locked goal head is fixture intensity: the match header owns the expected
bar total for that week, then the squad shares it. Presence is separate from
bar height. Assists keep a four-way head. The horizon reader and the column
reader stay separate. Absolute rates pin the volume. The paint keeps each
row's horizon count and moves the bars toward the weeks the column reader
scores highest. A row whose horizon count is zero stays empty.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flashcard_card_score import (
    _paint,
    column_count,
    paint_column,
    paint_horizon,
    paint_row,
    tally_cards,
    window_count,
)
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


def _square_bands(player_h: int) -> tuple[int, int, int, int] | None:
    """Minute, goal, assist, and position heights of the square card.

    Each band is embedded on its own and normalized before the bands are
    added. A raw sum lets the minute embedding grow until it drowns the
    goal squares and the position square.
    """
    from src.models.flashcard_square_trial import EVENT_H, MINUTE_H, PLAYER_H, POS_H

    if player_h != PLAYER_H or MINUTE_H + 2 * EVENT_H + POS_H != PLAYER_H:
        return None
    return (MINUTE_H, EVENT_H, EVENT_H, POS_H)


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
        self.band_h = _square_bands(self.player_h)
        if self.band_h is None:
            self.bands = nn.ModuleList()
            self.band_norm = nn.ModuleList()
            self.patch = nn.Linear(3 * self.player_h * CELL_W, D)
        else:
            self.bands = nn.ModuleList(nn.Linear(3 * height * CELL_W, D) for height in self.band_h)
            self.band_norm = nn.ModuleList(nn.LayerNorm(D) for _ in self.band_h)
            self.patch = None
            self.header_norm = nn.LayerNorm(D)
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
        flat = body.permute(0, 4, 2, 1, 3, 5)
        if self.band_h is None:
            cells = self.patch(flat.reshape(batch, COLS, ROWS, -1))
        else:
            header_tok = self.header_norm(header_tok)
            cells = flat.new_zeros(batch, COLS, ROWS, D)
            y0 = 0
            for layer, norm, height in zip(self.bands, self.band_norm, self.band_h, strict=True):
                part = flat[:, :, :, :, y0 : y0 + height, :].reshape(batch, COLS, ROWS, -1)
                cells = cells + norm(layer(part))
                y0 += height
        cells = cells.permute(0, 2, 1, 3) + header_tok[:, None, :, :]
        return self.couple(cells, row_pad, column_on < 0.5)


def _batch_loss(goal_logits: torch.Tensor, assist_logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cell cross-entropy plus each row's hidden bar count."""
    mask = batch["loss"]
    cell = 0.5 * (
        _nll(goal_logits, batch["goal_label"], mask) + _nll(assist_logits, batch["assist_label"], mask)
    )
    horizon = 0.5 * (
        window_count(goal_logits.softmax(-1), batch["goal_label"], mask)
        + window_count(assist_logits.softmax(-1), batch["assist_label"], mask)
    )
    return cell + horizon, cell.detach(), horizon.detach()


def _is_fixture(model: nn.Module) -> bool:
    return type(model).__name__ == "IntensityReader"


def _fixture_batch_loss(pack: dict, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixture-intensity goals plus assist NLL. Matches the locked trial."""
    from src.models.flashcard_intensity_trial import intensity_goal_loss

    mask = batch["loss"]
    goal_loss, count_loss = intensity_goal_loss(pack, batch["goal_label"], mask, ste=False)
    assist_loss = _nll(pack["assist_logits"], batch["assist_label"], mask)
    loss = 0.5 * (goal_loss + assist_loss)
    return loss, goal_loss.detach(), count_loss


def default_reader(header_h: int, dropout: float = 0.1, n_clubs: int = 1, player_h: int = PLAYER_H) -> nn.Module:
    """The locked square-card reader: fixture intensity for goals."""
    del n_clubs
    from src.models.flashcard_intensity_trial import IntensityReader

    return IntensityReader(header_h, mode="fixture", dropout=dropout, player_h=player_h)


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
        horizon_total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [sample_cut(card, rng) for card in pick]
            views = [strip_features(apply_cut(card, league, cut, clubs, squads)) for card, cut in zip(pick, cuts, strict=True)]
            batch = _stack(views)
            batch["image"] = torch.from_numpy(_batch_images(views, pick, odds, compose)).permute(0, 3, 1, 2).float().div(255.0)
            opt.zero_grad()
            if _is_fixture(model):
                pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
                loss, cell, horizon = _fixture_batch_loss(pack, batch)
            else:
                goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
                loss, cell, horizon = _batch_loss(goal_logits, assist_logits, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(batch["loss"].sum().detach())
            total += float(cell) * weight
            seen += weight
            horizon_total += float(horizon) * weight
        print(
            f"{name} epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f} "
            f"horizon {horizon_total / max(seen, 1.0):.3f}",
            flush=True,
        )


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
        if _is_fixture(model):
            pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
            return pack["goal_probs"].cpu().numpy(), F.softmax(pack["assist_logits"], dim=-1).cpu().numpy()
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
        horizon_total = 0.0
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
            if _is_fixture(model):
                pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
                loss, cell, horizon = _fixture_batch_loss(pack, batch)
            else:
                goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
                loss, cell, horizon = _batch_loss(goal_logits, assist_logits, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(batch["loss"].sum().detach())
            total += float(cell) * weight
            seen += weight
            horizon_total += float(horizon) * weight
        print(
            f"stripped step epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f} "
            f"horizon {horizon_total / max(seen, 1.0):.3f}",
            flush=True,
        )


def train_column_steps(
    model: nn.Module,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    odds: dict,
    name: str = "column step",
    compose=None,
) -> None:
    """One next week per card. The week is scored as one count down the squad.

    The picture keeps the true past and shows only that next column. Horizon
    row totals are not in this loss. Fixture intensity owns the week's rate;
    the column Poisson pins how many bars that week holds.
    """
    opt = _optimizer(model)
    squads = squad_index(cards)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        week_total = 0.0
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
            if _is_fixture(model):
                pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
                goal_loss, cell, _count = _fixture_batch_loss(pack, batch)
                weeks = 0.5 * (
                    column_count(pack["goal_probs"], batch["goal_label"], batch["loss"])
                    + column_count(pack["assist_logits"].softmax(-1), batch["assist_label"], batch["loss"])
                )
                loss = goal_loss + weeks
            else:
                goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
                loss, cell, weeks = _column_batch_loss(goal_logits, assist_logits, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(batch["loss"].sum().detach())
            total += float(cell) * weight
            week_total += float(weeks.detach() if torch.is_tensor(weeks) else weeks) * weight
            seen += weight
        print(
            f"{name} epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f} "
            f"column {week_total / max(seen, 1.0):.3f}",
            flush=True,
        )


def _column_batch_loss(
    goal_logits: torch.Tensor, assist_logits: torch.Tensor, batch: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cell cross-entropy plus Poisson rates on rows and on weeks.

    The week rate alone can put the bars on the wrong rows. Scoring each
    row's count as well keeps the position order while the week rate still
    pins how many events the fixture holds.
    """
    mask = batch["loss"]
    goal_p = goal_logits.softmax(-1)
    assist_p = assist_logits.softmax(-1)
    cell = 0.5 * (
        _nll(goal_logits, batch["goal_label"], mask) + _nll(assist_logits, batch["assist_label"], mask)
    )
    rows = 0.5 * (
        window_count(goal_p, batch["goal_label"], mask) + window_count(assist_p, batch["assist_label"], mask)
    )
    weeks = 0.5 * (
        column_count(goal_p, batch["goal_label"], mask) + column_count(assist_p, batch["assist_label"], mask)
    )
    return cell + rows + weeks, cell.detach(), weeks.detach()


def train_columns(
    model: nn.Module,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    odds: dict | None,
    name: str,
    compose=None,
) -> None:
    """Learn each hidden week's bar count. The weights stay off the horizon reader.

    The picture is the same cut card. Each square is scored, and each week is
    scored as a Poisson rate down the squad, so the reader cannot dump the
    week's bars on one position.
    """
    opt = _optimizer(model)
    squads = squad_index(cards)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        week_total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [sample_cut(card, rng) for card in pick]
            views = [
                strip_features(apply_cut(card, league, cut, clubs, squads))
                for card, cut in zip(pick, cuts, strict=True)
            ]
            batch = _stack(views)
            batch["image"] = torch.from_numpy(_batch_images(views, pick, odds, compose)).permute(0, 3, 1, 2).float().div(255.0)
            opt.zero_grad()
            if _is_fixture(model):
                pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
                goal_loss, cell, _count = _fixture_batch_loss(pack, batch)
                weeks = 0.5 * (
                    column_count(pack["goal_probs"], batch["goal_label"], batch["loss"])
                    + column_count(pack["assist_logits"].softmax(-1), batch["assist_label"], batch["loss"])
                )
                loss = goal_loss + weeks
            else:
                goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
                loss, cell, weeks = _column_batch_loss(goal_logits, assist_logits, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(batch["loss"].sum().detach())
            total += float(cell) * weight
            week_total += float(weeks.detach() if torch.is_tensor(weeks) else weeks) * weight
            seen += weight
        print(
            f"{name} epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f} "
            f"column {week_total / max(seen, 1.0):.3f}",
            flush=True,
        )


def train_coupled(
    model: nn.Module,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    odds: dict,
    name: str,
    compose=None,
) -> None:
    """One reader, two views, one step.

    The horizon view reveals the masked remainder in one shot. The column
    view keeps the true past, shows one next week, and scores only that week.
    The two losses are added and update the same weights.
    """
    opt = _optimizer(model)
    squads = squad_index(cards)
    model.train()
    for epoch in range(EPOCHS):
        horizon_cell = 0.0
        horizon_window = 0.0
        horizon_seen = 0.0
        column_cell = 0.0
        column_window = 0.0
        column_seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [sample_cut(card, rng) for card in pick]
            horizon_views = [
                strip_features(apply_cut(card, league, cut, clubs, squads))
                for card, cut in zip(pick, cuts, strict=True)
            ]
            column_views = []
            column_pictures = []
            for card in pick:
                target = _target_gw(card, rng)
                view = _step_view(card, league, clubs, squads, target)
                column_views.append(view)
                column_pictures.append(_step_image(view, card, odds, target, compose))
            horizon_batch = _stack(horizon_views)
            horizon_batch["image"] = torch.from_numpy(_batch_images(horizon_views, pick, odds, compose)).permute(0, 3, 1, 2).float().div(255.0)
            column_batch = _stack(column_views)
            column_batch["image"] = torch.from_numpy(np.stack(column_pictures)).permute(0, 3, 1, 2).float().div(255.0)
            opt.zero_grad()
            horizon_goals, horizon_assists = model(
                horizon_batch["image"], horizon_batch["row_pad"], horizon_batch["column_on"]
            )
            column_goals, column_assists = model(
                column_batch["image"], column_batch["row_pad"], column_batch["column_on"]
            )
            horizon_loss, horizon_nll, horizon_count = _batch_loss(horizon_goals, horizon_assists, horizon_batch)
            column_loss, column_nll, column_count = _batch_loss(column_goals, column_assists, column_batch)
            (horizon_loss + column_loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            horizon_weight = float(horizon_batch["loss"].sum().detach())
            column_weight = float(column_batch["loss"].sum().detach())
            horizon_cell += float(horizon_nll) * horizon_weight
            horizon_window += float(horizon_count) * horizon_weight
            horizon_seen += horizon_weight
            column_cell += float(column_nll) * column_weight
            column_window += float(column_count) * column_weight
            column_seen += column_weight
        print(
            f"{name} epoch {epoch + 1} "
            f"horizon nll {horizon_cell / max(horizon_seen, 1.0):.3f} "
            f"window {horizon_window / max(horizon_seen, 1.0):.3f} "
            f"column nll {column_cell / max(column_seen, 1.0):.3f} "
            f"window {column_window / max(column_seen, 1.0):.3f}",
            flush=True,
        )


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
            if _is_fixture(model):
                pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
                goal_p = pack["goal_probs"].cpu().numpy()
                assist_p = F.softmax(pack["assist_logits"], dim=-1).cpu().numpy()
            else:
                goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
                goal_p = F.softmax(goal_logits, dim=-1).numpy()
                assist_p = F.softmax(assist_logits, dim=-1).numpy()
        for slot, index in enumerate(indexes):
            card = cards[index]
            mask = card.present & (card.gw == gw).reshape(1, -1)
            goals[index] += paint_row(goal_p[slot], mask)
            assists[index] += paint_row(assist_p[slot], mask)
    return goals, assists


def paint_stripped_columns(
    model: nn.Module,
    cards: list,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict,
    compose=None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Walk every gameweek and place that week's bars as one squad count.

    The week is scored as a whole: the expected bar count is the sum of
    chances down the squad, then that many bars land on the most likely
    squares. A single cell that rounds to zero does not empty a week that
    still holds mass. The returned rates are the mean chance on a real bar
    and on a blank.
    """
    goals = np.zeros((len(cards),) + cards[0].goals.shape, dtype=np.int64)
    assists = np.zeros_like(goals)
    on = np.zeros(2)
    off = np.zeros(2)
    on_n = np.zeros(2)
    off_n = np.zeros(2)
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
            if _is_fixture(model):
                pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
                goal_p = pack["goal_probs"].cpu().numpy()
                assist_p = F.softmax(pack["assist_logits"], dim=-1).cpu().numpy()
            else:
                goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
                goal_p = F.softmax(goal_logits, dim=-1).numpy()
                assist_p = F.softmax(assist_logits, dim=-1).numpy()
        for slot, index in enumerate(indexes):
            card = cards[index]
            mask = card.present & (card.gw == gw).reshape(1, -1)
            goals[index] += paint_column(goal_p[slot], mask)
            assists[index] += paint_column(assist_p[slot], mask)
            for channel, probs, labels in (
                (0, goal_p, card.goals),
                (1, assist_p, card.assists),
            ):
                cols = np.flatnonzero(mask.any(axis=0))
                for col in cols:
                    rows = np.flatnonzero(mask[:, col])
                    if len(rows) == 0:
                        continue
                    chance = 1.0 - probs[slot, rows, col, 0]
                    hit = labels[rows, col] > 0
                    on[channel] += float(chance[hit].sum())
                    off[channel] += float(chance[~hit].sum())
                    on_n[channel] += int(hit.sum())
                    off_n[channel] += int((~hit).sum())
    rates = {
        "goal_on": on[0] / max(on_n[0], 1.0),
        "goal_off": off[0] / max(off_n[0], 1.0),
        "assist_on": on[1] / max(on_n[1], 1.0),
        "assist_off": off[1] / max(off_n[1], 1.0),
    }
    return goals, assists, rates


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
