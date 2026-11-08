"""Budget trial on the locked vision card.

Goals and assists are painted as one flat color for the whole band:
white, red, green, or blue for 0, 1, 2, or 3. A hidden event stays gray.
The opponent's conceded-rate band is hatched on a hidden fixture.

A match head predicts the club's goal total and assist total for each
future fixture. Those totals are placed onto players with minutes, in
order of expected count. The cell loss is unchanged. The locked card and
its 24.7% score stay the gate.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    BAND_H,
    BATCH,
    CELL_W,
    COLS,
    CUT_GW,
    EPOCHS,
    HEADER_H,
    LR,
    PLAYER_H,
    MASK_C,
    MASK_ID,
    POS,
    ROWS,
    SEED,
    SquadCard,
    SquadVision,
    _hatch,
    _hidden_mask,
    _nll,
    _opp_batch,
    _stack,
    apply_cut,
    bar_correspondence,
    build_cards,
    build_league,
    capture_table,
    cut_at_fraction,
    fit_clubs,
    positive_by_position,
    random_same_count,
    render_grid,
    sample_cut,
    squad_index,
)

TEAM_MAX = 8
COUNT_C = (
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
)
ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"


def render_mark(view: dict[str, np.ndarray]) -> np.ndarray:
    """Locked card, with a flat event color and a hidden concede band."""
    img = render_grid(view).copy()
    shown = (view["present"] > 0.5) & (view["goal_token"] != MASK_ID)
    hidden_col = (view["column_on"] > 0.5) & ~shown.any(axis=0)
    for col in range(COLS):
        x0 = col * CELL_W
        if hidden_col[col]:
            _hatch(img, BAND_H, x0, BAND_H)
        for row in range(ROWS):
            if int(view["position"][row]) < 0 or view["present"][row, col] < 0.5:
                continue
            y = HEADER_H + row * PLAYER_H
            _paint_count(img, y + 2 * BAND_H, x0, int(view["goal_token"][row, col]))
            _paint_count(img, y + 3 * BAND_H, x0, int(view["assist_token"][row, col]))
    return img


def _paint_count(img: np.ndarray, y0: int, x0: int, token: int) -> None:
    if token == MASK_ID:
        img[y0 : y0 + BAND_H, x0 : x0 + CELL_W] = MASK_C
        return
    img[y0 : y0 + BAND_H, x0 : x0 + CELL_W] = COUNT_C[min(max(token, 0), 3)]


def _mark_batch(views: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    batch = _stack(views)
    images = np.stack([render_mark(view) for view in views])
    batch["image"] = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.0)
    return batch


class MarkVision(nn.Module):
    """Locked vision reader, plus a total for each future fixture."""

    def __init__(self, dropout: float = 0.1, n_clubs: int = 1) -> None:
        super().__init__()
        self.vision = SquadVision(dropout=dropout, n_clubs=n_clubs)
        self.match_goal = nn.Linear(ROWS * 4, TEAM_MAX + 1)
        self.match_assist = nn.Linear(ROWS * 4, TEAM_MAX + 1)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        goal, assist = self.vision(
            images, row_pad, column_on, position, minutes, matchup, club, opponent, opp
        )
        return goal, assist, self._total(self.match_goal, goal), self._total(self.match_assist, assist)

    @staticmethod
    def _total(head: nn.Linear, logits: torch.Tensor) -> torch.Tensor:
        pooled = logits.permute(0, 2, 1, 3).reshape(logits.shape[0], logits.shape[2], ROWS * 4)
        return head(pooled)


def _match_nll(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    clipped = labels.sum(dim=1).clamp(0, TEAM_MAX)
    flat = F.cross_entropy(logits.reshape(-1, TEAM_MAX + 1), clipped.reshape(-1), reduction="none")
    flat = flat.reshape(clipped.shape)
    weight = mask.sum().clamp_min(1.0)
    return (flat * mask).sum() / weight


def train_budget(
    model: MarkVision,
    cards: list[SquadCard],
    league: dict,
    rng: np.random.Generator,
    clubs: dict[str, int],
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH,
) -> list[float]:
    fast_name = ("pos_", "position", "club_", "opp_", "form_")
    fast = [p for name, p in model.named_parameters() if any(token in name for token in fast_name)]
    fast_ids = {id(p) for p in fast}
    rest = [p for p in model.parameters() if id(p) not in fast_ids]
    opt = torch.optim.AdamW(
        [
            {"params": rest, "lr": lr, "weight_decay": 1e-4},
            {"params": fast, "lr": lr * 10, "weight_decay": 0.0},
        ]
    )
    squads = squad_index(cards)
    history = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), batch_size):
            pick = [cards[int(i)] for i in order[start : start + batch_size]]
            views = [apply_cut(card, league, sample_cut(card, rng), clubs, squads) for card in pick]
            batch = _mark_batch(views)
            opt.zero_grad()
            goal, assist, match_g, match_a = model(
                batch["image"], batch["row_pad"], batch["column_on"], batch["position"], batch["minutes"],
                batch["matchup"], batch["club"], batch["opponent"], _opp_batch(batch),
            )
            cell = batch["loss"]
            column = cell.sum(dim=1) > 0
            loss = 0.5 * (
                _nll(goal, batch["goal_label"], cell) + _nll(assist, batch["assist_label"], cell)
            )
            loss = loss + 0.5 * (
                _match_nll(match_g, batch["goal_label"], column)
                + _match_nll(match_a, batch["assist_label"], column)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(cell.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        epoch_nll = total / max(seen, 1.0)
        history.append(epoch_nll)
        print(f"budget epoch {epoch + 1} train nll {epoch_nll:.3f}", flush=True)
    return history


@torch.no_grad()
def predict_budget(
    model: MarkVision,
    cards: list[SquadCard],
    league: dict,
    clubs: dict[str, int],
    squads: dict[tuple[str, str], SquadCard],
    cuts: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    goals = np.zeros((len(cards), ROWS, COLS, 4), dtype=np.float64)
    assists = np.zeros_like(goals)
    match_g = np.zeros((len(cards), COLS, TEAM_MAX + 1), dtype=np.float64)
    match_a = np.zeros_like(match_g)
    for start in range(0, len(cards), BATCH):
        stop = min(start + BATCH, len(cards))
        views = [
            apply_cut(cards[i], league, cuts[i], clubs, squads) for i in range(start, stop)
        ]
        batch = _mark_batch(views)
        g, a, mg, ma = model(
            batch["image"], batch["row_pad"], batch["column_on"], batch["position"], batch["minutes"],
            batch["matchup"], batch["club"], batch["opponent"], _opp_batch(batch),
        )
        goals[start:stop] = F.softmax(g, dim=-1).numpy()
        assists[start:stop] = F.softmax(a, dim=-1).numpy()
        match_g[start:stop] = F.softmax(mg, dim=-1).numpy()
        match_a[start:stop] = F.softmax(ma, dim=-1).numpy()
    return goals, assists, match_g, match_a


def place_budget(
    cell_probs: np.ndarray,
    match_probs: np.ndarray,
    minutes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place one sampled fixture total onto players with minutes, by expected count."""
    expected = (cell_probs * np.arange(4)).sum(axis=-1)
    drawn = np.zeros(cell_probs.shape[:-1], dtype=np.int64)
    for col in range(cell_probs.shape[-2]):
        cdf = np.cumsum(match_probs[col])
        total = int(min(np.searchsorted(cdf, rng.random()), TEAM_MAX))
        rank = expected[:, col] + minutes[:, col] / 90.0 * 1e-3
        order = np.argsort(-rank)
        remaining = total
        for row in order:
            if remaining <= 0:
                break
            if minutes[row, col] <= 0:
                continue
            give = 1
            if expected[row, col] >= 1.5 and remaining >= 2:
                give = int(min(3, remaining, max(2, round(float(expected[row, col])))))
            drawn[row, col] = give
            remaining -= give
    return drawn


def budget_correspondence(
    cards: list[SquadCard],
    cell_probs: np.ndarray,
    match_probs: np.ndarray,
    labels: np.ndarray,
    minutes: np.ndarray,
    rng: np.random.Generator,
    cuts: list[int],
    draws: int = 20,
) -> dict[str, float]:
    mask = _hidden_mask(cards, 0, cuts)
    actual = np.where(mask, labels, 0)
    actual_bars = int((actual > 0).sum())
    drawn_bars = 0.0
    covered = 0.0
    on_actual = 0.0
    drawn_height = 0.0
    for _ in range(draws):
        drawn = np.zeros_like(labels)
        for index in range(len(cards)):
            drawn[index] = place_budget(cell_probs[index], match_probs[index], minutes[index], rng)
        drawn = np.where(mask, drawn, 0)
        both = (actual > 0) & (drawn > 0)
        bars = float((drawn > 0).sum())
        drawn_bars += bars
        drawn_height += float(drawn.sum())
        covered += float(both.sum())
        on_actual += float(both.sum())
    return {
        "actual_bars": actual_bars,
        "drawn_bars": drawn_bars / draws,
        "drawn_height": drawn_height / draws,
        "actual_covered": covered / draws / max(actual_bars, 1),
        "drawn_on_actual": on_actual / draws / max(drawn_bars / draws, 1.0),
    }


def _report(lines: list[str]) -> None:
    path = ROOT_REPORTS / "flashcard_budget_trial.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    print("loading budget cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(train + test)
    n_clubs = max(clubs.values(), default=0) + 1
    print(f"train {len(train)} test {len(test)}", flush=True)
    torch.manual_seed(SEED)
    model = MarkVision(dropout=0.1, n_clubs=n_clubs)
    train_budget(model, train, league, np.random.default_rng(SEED), clubs)
    print("scoring", flush=True)
    fixed = [CUT_GW] * len(test)
    goal_p, assist_p, match_g, match_a = predict_budget(model, test, league, clubs, squads, fixed)
    capture = capture_table(test, {"budget": 1.0 - goal_p[:, :, :, 0]}, CUT_GW)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    bar_rng = np.random.default_rng(SEED)
    cell_bars = bar_correspondence(test, goal_p, labels, CUT_GW, bar_rng)
    budget_bars = budget_correspondence(test, goal_p, match_g, labels, minutes, bar_rng, fixed)
    chance = random_same_count(test, labels, CUT_GW, bar_rng)
    lines = [
        "# Budget trial",
        "",
        "Gated against squad vision with squad coupling: 24.7% top 10% on the",
        "all-minutes exam. A higher number on that exam would replace the default.",
        "",
        "The picture keeps the locked header and player rows. A goal or assist",
        "band is one flat color: white, red, green, or blue for 0, 1, 2, or 3.",
        "Hidden events stay gray. The conceded-rate band is hatched after the cut.",
        "A match head predicts the fixture total, and that total is placed onto",
        "players with minutes in order of expected count.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        f"Gameweek-{CUT_GW} top 10/20/50: "
        + ", ".join(f"{value:.1%}" for value in capture["budget"]),
        f"Players: {capture['n_players']}. Locked default: 24.7%, 41.5%, 80.1%.",
        "",
        f"Cell samples, goals, gameweeks 20-38: actual {cell_bars['actual_bars']}, "
        f"drawn {cell_bars['drawn_bars']:.1f}, covered {cell_bars['actual_covered']:.1%}, "
        f"on a real week {cell_bars['drawn_on_actual']:.1%}.",
        f"Budget placement, goals: drawn {budget_bars['drawn_bars']:.1f}, "
        f"height {budget_bars['drawn_height']:.1f}, covered {budget_bars['actual_covered']:.1%}, "
        f"on a real week {budget_bars['drawn_on_actual']:.1%}.",
        f"Same count on random weeks covers {chance['actual_covered']:.1%}.",
        "",
        "## Remainders from gameweek 1",
        "",
        "| revealed | decoder | actual bars | drawn bars | covered | on a real week | GK | FWD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for revealed in (0.8, 0.2):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        g, a, mg, ma = predict_budget(model, test, league, clubs, squads, cuts)
        rates = positive_by_position(g, test, 0, cuts)
        for channel, probs, totals, channel_labels in (
            ("goals", g, mg, labels),
            ("assists", a, ma, assist_labels),
        ):
            cell = bar_correspondence(test, probs, channel_labels, 0, bar_rng, cuts=cuts)
            placed = budget_correspondence(
                test, probs, totals, channel_labels, minutes, bar_rng, cuts
            )
            for decoder, bars in (("cell", cell), ("budget", placed)):
                lines.append(
                    f"| {revealed:.0%} | {decoder} {channel} | {bars['actual_bars']} | "
                    f"{bars['drawn_bars']:.1f} | {bars['actual_covered']:.1%} | "
                    f"{bars['drawn_on_actual']:.1%} | {rates[POS['GKP']]:.1%} | {rates[POS['FWD']]:.1%} |"
                )
            print(
                f"reveal {revealed:.0%} {channel} cell drawn {cell['drawn_bars']:.1f} "
                f"on_week {cell['drawn_on_actual']:.3f} budget drawn {placed['drawn_bars']:.1f} "
                f"on_week {placed['drawn_on_actual']:.3f}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    held = capture["budget"][0] <= 0.247
    lines.extend(
        [
            "",
            "The default holds." if held else "This run beats the locked default on the gate.",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    _report(lines)
    print(f"done in {seconds:.0f}s capture10 {capture['budget'][0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
