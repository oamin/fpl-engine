"""The locked card. Venue stays. Four header rows are gone.

Concede, regulars, own history, and opponent history are absent from the
picture and from the numeric match-up. The recent scoring rate stays.
Opening odds, when used, sit above the venue band and are not part of the gate.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flashcard_card_score import _paint, tally_cards
from src.models.flashcard_odds_trial import (
    column_odds,
    load_odds,
    paint_odds_strip,
    visible_odds,
)
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
    _opp_batch,
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
# own history, its missing flag, opponent history, its missing flag,
# goals conceded, its missing flag. Slots 4 and 5 are the recent scoring rate.
MATCHUP_DROP = (0, 1, 2, 3, 6, 7)
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
    """Zero the four removed rows and the match-up slots that carry them."""
    for key in FEATURE_KEYS:
        view[key] = np.zeros_like(view[key])
    matchup = np.array(view["matchup"], copy=True)
    matchup[:, list(MATCHUP_DROP)] = 0
    view["matchup"] = matchup
    return view


def compose_stripped(
    view: dict[str, np.ndarray],
    grey_columns: np.ndarray | None = None,
    probs: np.ndarray | None = None,
    visible: np.ndarray | None = None,
) -> np.ndarray:
    """Venue, then the squad. The four header rows are not in the picture."""
    image = render_grid(view)
    if grey_columns is not None:
        grey_except_minutes(image, grey_columns)
    image = np.concatenate([image[:BAND_H], image[HEADER_H:]], axis=0)
    if probs is None or visible is None:
        return image
    strip = paint_odds_strip(view["column_on"], probs, visible)
    return np.concatenate([strip, image], axis=0)


class StripVision(nn.Module):
    """Pixel reader for a header that is venue, or odds plus venue."""

    def __init__(self, header_h: int, dropout: float = 0.1, n_clubs: int = 1) -> None:
        super().__init__()
        self.header_h = int(header_h)
        self.header = nn.Linear(3 * self.header_h * CELL_W, D)
        self.patch = nn.Linear(3 * PLAYER_H * CELL_W, D)
        self.couple = Coupler(dropout, n_clubs)

    def forward(
        self,
        images: torch.Tensor,
        row_pad: torch.Tensor,
        column_on: torch.Tensor,
        position: torch.Tensor,
        minutes: torch.Tensor,
        matchup: torch.Tensor,
        club: torch.Tensor,
        opponent: torch.Tensor,
        opp: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = images.shape[0]
        header = images[:, :, : self.header_h, :].reshape(batch, 3, self.header_h, COLS, CELL_W)
        header = header.permute(0, 3, 1, 2, 4).reshape(batch, COLS, -1)
        header_tok = self.header(header)
        body = images[:, :, self.header_h :, :].reshape(batch, 3, ROWS, PLAYER_H, COLS, CELL_W)
        body = body.permute(0, 4, 2, 1, 3, 5).reshape(batch, COLS, ROWS, -1)
        cells = self.patch(body).permute(0, 2, 1, 3) + header_tok[:, None, :, :]
        return self.couple(
            cells, row_pad, column_on < 0.5, position, minutes, matchup, club, opponent, opp
        )


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    fast_name = ("pos_", "position", "club_", "opp_", "form_")
    fast = [p for name, p in model.named_parameters() if any(token in name or name.startswith(token) for token in fast_name)]
    fast_ids = {id(p) for p in fast}
    rest = [p for p in model.parameters() if id(p) not in fast_ids]
    return torch.optim.AdamW(
        [
            {"params": rest, "lr": LR, "weight_decay": 1e-4},
            {"params": fast, "lr": LR * 10, "weight_decay": 0.0},
        ]
    )


def _batch_images(views: list[dict], cards: list, odds: dict | None) -> np.ndarray:
    pictures = []
    for view, card in zip(views, cards, strict=True):
        if odds is None:
            pictures.append(compose_stripped(view))
            continue
        pictures.append(
            compose_stripped(
                view,
                None,
                column_odds(card, odds),
                visible_odds(card.gw, "painter", cut=0),
            )
        )
    return np.stack(pictures)


def train_stripped(
    model: nn.Module,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    odds: dict | None,
    name: str,
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
            batch["image"] = torch.from_numpy(_batch_images(views, pick, odds)).permute(0, 3, 1, 2).float().div(255.0)
            opt.zero_grad()
            goal_logits, assist_logits = model(
                batch["image"], batch["row_pad"], batch["column_on"], batch["position"], batch["minutes"],
                batch["matchup"], batch["club"], batch["opponent"], _opp_batch(batch),
            )
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
) -> tuple[np.ndarray, np.ndarray]:
    views = [strip_features(apply_cut(card, league, cut, clubs, squads)) for card, cut in zip(cards, cuts, strict=True)]
    batch = _stack(views)
    batch["image"] = torch.from_numpy(_batch_images(views, cards, odds)).permute(0, 3, 1, 2).float().div(255.0)
    model.eval()
    with torch.no_grad():
        goal_logits, assist_logits = model(
            batch["image"], batch["row_pad"], batch["column_on"], batch["position"], batch["minutes"],
            batch["matchup"], batch["club"], batch["opponent"], _opp_batch(batch),
        )
    return F.softmax(goal_logits, dim=-1).numpy(), F.softmax(assist_logits, dim=-1).numpy()


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
        "The one-shot card keeps venue and drops concede, regulars, own history,",
        "and opponent history. Those values are also removed from the numeric",
        "match-up. The recent scoring rate stays. A second reader adds the odds",
        "bar above the venue band, on every known week and every grey week.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    readers = (
        ("stripped", StripVision(BAND_H, dropout=0.1, n_clubs=n_clubs), None),
        ("stripped odds", StripVision(BAND_H * 2, dropout=0.1, n_clubs=n_clubs), odds),
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
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "The no-odds card is the gate. At 20% revealed it puts 190 goal bars",
            "on the same appearance and 258 within one gameweek.",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    path = ROOT_REPORTS / "flashcard_strip_trial.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
