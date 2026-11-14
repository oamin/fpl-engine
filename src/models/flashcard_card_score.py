"""Compare the painted test card with the actual card, as totals.

Every club in 2025-26 is cut after 20%, 40%, 60%, and 80% of its fixtures,
starting from gameweek 1. Every row is included. A bar on one row does not
match a goal on another row. The totals are sums, not an average of per-player
rates.

The painted card comes from the locked vision reader. The picture is the
only feature input. It is the square card: the club symbol, the opponent
symbol, home or away, sixteen odds squares, six minute squares of fifteen
minutes, one
position square, and one square per goal or assist. Concede, regulars, own
history, opponent history, and the team scoring rate are absent. Minutes,
position, club, and opponent are not passed again as numbers. Goals use
fixture intensity: the match header owns the week's expected bar total,
then the squad shares it. Assists keep a four-way head. Two readers see
the card, and their weights stay separate. The paint keeps each row's
horizon count on that row and moves the bars toward the weeks the column
reader scores highest. A row whose horizon count is zero stays empty. A
row holds three squares, and that is the only cap.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    SEED,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    squad_index,
)

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
KEYS = ("actual", "painted", "exact", "actual_within", "painted_within")

# Published mixer table — Baudot row mixer + learnable Tversky, locked 2026-11-13.
# Method: `src/models/flashcard_baudot_row_trial.py` (full past + one-shot remainder).
# Card: GW-as-rows binary tape; odds win ON / draw BLANK / loss ON; ID_BITS=11.
# Loss: learnable Tversky α=0.5−δ, β=0.5+δ (β≥α). Seed 0. Report: reports/baudot_gate.md.
# Tallies use tally_row appearances (exact = hits).
# The default card/method from 2026-11-14 is the opponent cross-read below.
# This table stays the published 2026-11-13 mixer score. Do not rewrite it.
GATE = {
    ("goals", 0.2): {"actual": 730, "painted": 701, "exact": 194, "actual_within": 268, "painted_within": 377},
    ("assists", 0.2): {"actual": 707, "painted": 693, "exact": 114, "actual_within": 153, "painted_within": 232},
    ("goals", 0.4): {"actual": 544, "painted": 557, "exact": 151, "actual_within": 214, "painted_within": 299},
    ("assists", 0.4): {"actual": 531, "painted": 551, "exact": 89, "actual_within": 123, "painted_within": 174},
    ("goals", 0.6): {"actual": 360, "painted": 383, "exact": 105, "actual_within": 145, "painted_within": 205},
    ("assists", 0.6): {"actual": 351, "painted": 377, "exact": 66, "actual_within": 90, "painted_within": 126},
    ("goals", 0.8): {"actual": 199, "painted": 195, "exact": 60, "actual_within": 82, "painted_within": 101},
    ("assists", 0.8): {"actual": 192, "painted": 191, "exact": 32, "actual_within": 45, "painted_within": 59},
}

# Default flashcard card/method, locked 2026-11-14.
# Module: `src/models/flashcard_baudot_opp_train.py`.
# Card: club8 · opp8 · odds12 · gc6 · player blocks. Cross-read of the opponent card:
# earlier rows true, this fixture's minutes on the opponent card only, G/A and gc
# blank on both cards, later kickoffs hidden. Cell Tversky on the assessed remainder.
# Seed 0, early-stopped epoch 20 (best epoch 12). Report: reports/baudot_opp_train.md.
# Mean balanced_score 0.192. Screens that change this method compare here and to GATE.
GATE_OPP_2026_11_14 = {
    ("goals", 0.2): {"actual": 730, "painted": 770, "exact": 197, "actual_within": 271, "painted_within": 396},
    ("assists", 0.2): {"actual": 707, "painted": 767, "exact": 131, "actual_within": 172, "painted_within": 269},
    ("goals", 0.4): {"actual": 544, "painted": 644, "exact": 155, "actual_within": 218, "painted_within": 326},
    ("assists", 0.4): {"actual": 531, "painted": 640, "exact": 108, "actual_within": 142, "painted_within": 209},
    ("goals", 0.6): {"actual": 360, "painted": 451, "exact": 106, "actual_within": 150, "painted_within": 228},
    ("assists", 0.6): {"actual": 351, "painted": 449, "exact": 84, "actual_within": 108, "painted_within": 157},
    ("goals", 0.8): {"actual": 199, "painted": 238, "exact": 57, "actual_within": 79, "painted_within": 114},
    ("assists", 0.8): {"actual": 192, "painted": 233, "exact": 41, "actual_within": 56, "painted_within": 76},
}
DEFAULT_FLASHCARD = "src.models.flashcard_baudot_opp_train"

# Prior square-card full painter (12 Nov 2026). Archaeology only — not the gate.
GATE_SQUARE_2026_11_12 = {
    ("goals", 0.2): {"actual": 730, "painted": 798, "exact": 169, "actual_within": 322, "painted_within": 323},
    ("assists", 0.2): {"actual": 707, "painted": 664, "exact": 107, "actual_within": 215, "painted_within": 203},
    ("goals", 0.4): {"actual": 544, "painted": 594, "exact": 129, "actual_within": 241, "painted_within": 248},
    ("assists", 0.4): {"actual": 531, "painted": 498, "exact": 80, "actual_within": 159, "painted_within": 147},
    ("goals", 0.6): {"actual": 360, "painted": 368, "exact": 88, "actual_within": 153, "painted_within": 149},
    ("assists", 0.6): {"actual": 351, "painted": 302, "exact": 53, "actual_within": 100, "painted_within": 96},
    ("goals", 0.8): {"actual": 199, "painted": 184, "exact": 43, "actual_within": 84, "painted_within": 74},
    ("assists", 0.8): {"actual": 192, "painted": 140, "exact": 30, "actual_within": 48, "painted_within": 43},
}


def tally_row(gameweeks: np.ndarray, actual: np.ndarray, painted: np.ndarray) -> dict[str, int]:
    """Counts for one row. Within one gameweek includes the same appearance."""
    got = np.asarray(actual) > 0
    drew = np.asarray(painted) > 0
    goal_gw = np.asarray(gameweeks)[got]
    bar_gw = np.asarray(gameweeks)[drew]
    return {
        "actual": int(got.sum()),
        "painted": int(drew.sum()),
        "exact": int((got & drew).sum()),
        "actual_within": _close(goal_gw, bar_gw),
        "painted_within": _close(bar_gw, goal_gw),
    }


def _close(left: np.ndarray, right: np.ndarray) -> int:
    if len(left) == 0 or len(right) == 0:
        return 0
    return int(np.any(np.abs(left.astype(np.int64)[:, None] - right.astype(np.int64)[None, :]) <= 1, axis=1).sum())


def balanced_score(totals: dict[str, int]) -> dict[str, float]:
    """Rank a painted card by hits and by how close the bar counts are.

    The hit term is the harmonic mean of precision and recall. A painter that
    doubles the bars raises the denominator by every extra bar, and a painter
    that paints half the bars lowers the recall. The volume term is the
    smaller bar count divided by the larger one, so twice as many bars, or
    half as many, scores half before the hits are even considered.
    """
    actual = int(totals["actual"])
    painted = int(totals["painted"])
    exact = int(totals["exact"])
    if actual + painted == 0:
        hits = 1.0
        volume = 1.0
    else:
        hits = (2 * exact) / (actual + painted)
        volume = 0.0 if actual == 0 or painted == 0 else min(actual, painted) / max(actual, painted)
    within_hits = _harmonic(
        _rate(int(totals["actual_within"]), actual),
        _rate(int(totals["painted_within"]), painted),
    )
    return {"hits": hits, "volume": volume, "score": hits * volume, "within": within_hits * volume}


def _rate(hit: int, total: int) -> float:
    if total == 0:
        return 1.0 if hit == 0 else 0.0
    return hit / total


def _harmonic(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tally_cards(
    cards: list,
    labels: np.ndarray,
    painted: np.ndarray,
    cut_gw: int,
    cuts: list[int] | None = None,
) -> dict[str, int]:
    """Sum one channel over every hidden appearance of every club."""
    totals = {key: 0 for key in KEYS}
    for index, card in enumerate(cards):
        cut = cuts[index] if cuts is not None else cut_gw
        hidden = (card.gw > cut) & (card.gw > 0)
        for row in range(card.n_players):
            cols = np.where(hidden & card.present[row])[0]
            if len(cols) == 0:
                continue
            part = tally_row(card.gw[cols], labels[index, row, cols], painted[index, row, cols])
            for key in KEYS:
                totals[key] += part[key]
    return totals


def _poisson_count(predicted: torch.Tensor, actual: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Mean Poisson NLL of absolute bar counts.

    The rate is the expected number of bars. Empty rows with a light rate
    still pay for that mass, so the card's volume is pinned by the sum of
    the row rates. A relative scale left that total free.
    """
    rate = predicted.clamp_min(1e-8)
    nll = rate - actual * rate.log()
    return (nll * present).sum() / present.sum().clamp_min(1.0)


def window_count(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Poisson NLL of each row's hidden bar count.

    The card's total can match while every bar sits on an empty row. Each
    row is scored on its own, so that arrangement is a large error. The
    rate is absolute, so a cold reader cannot hide a short season total
    behind a matched average relative error.
    """
    predicted = ((1.0 - probs[..., 0]) * mask).sum(dim=-1)
    actual = ((labels > 0).to(probs.dtype) * mask).sum(dim=-1)
    present = mask.sum(dim=-1) > 0
    return _poisson_count(predicted, actual, present)


def column_count(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Poisson NLL of each column's hidden bar count.

    The sum runs down the squad, so one week is one count. A column with no
    hidden appearance is left out. Absolute rates pin the week's volume.
    """
    predicted = ((1.0 - probs[..., 0]) * mask).sum(dim=1)
    actual = ((labels > 0).to(probs.dtype) * mask).sum(dim=1)
    present = mask.sum(dim=1) > 0
    return _poisson_count(predicted, actual, present)


def fit_margins(horizon: np.ndarray, column: np.ndarray, mask: np.ndarray, rounds: int = 25) -> np.ndarray:
    """Keep each row's horizon count, and move the bars toward the column totals.

    `horizon` and `column` are one card, shaped rows by columns by classes.
    A row whose horizon count rounds to zero stays empty. The column reader
    sets the relative size of the weeks, and the horizon reader keeps the
    total and the row order.
    """
    horizon = np.asarray(horizon, dtype=np.float64)
    column = np.asarray(column, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    chance_h = np.where(mask, np.clip(1.0 - horizon[..., 0], 0.0, None), 0.0)
    chance_c = np.where(mask, np.clip(1.0 - column[..., 0], 0.0, None), 0.0)
    available = mask.sum(axis=1)
    row_target = np.clip(np.rint(chance_h.sum(axis=1)), 0, available)
    painted = np.zeros(mask.shape, dtype=np.int64)
    total = float(row_target.sum())
    if total <= 0:
        return painted
    raw = chance_c.sum(axis=0)
    if float(raw.sum()) <= 0:
        raw = chance_h.sum(axis=0)
    if float(raw.sum()) <= 0:
        raw = mask.sum(axis=0).astype(np.float64)
    week = raw * (total / float(raw.sum()))
    mass = chance_h.copy()
    mass[row_target <= 0] = 0.0
    for row in range(mass.shape[0]):
        if row_target[row] > 0 and float(mass[row].sum()) <= 0 and bool(mask[row].any()):
            mass[row] = mask[row].astype(np.float64)
    for _ in range(rounds):
        row_sum = mass.sum(axis=1, keepdims=True)
        row_scale = np.ones_like(row_sum)
        live = (row_target > 0) & (row_sum[:, 0] > 0)
        row_scale[live] = row_target[live, None] / row_sum[live]
        mass *= row_scale
        col_sum = mass.sum(axis=0, keepdims=True)
        col_scale = np.ones_like(col_sum)
        live_col = col_sum[0] > 0
        col_scale[0, live_col] = week[live_col] / col_sum[0, live_col]
        mass *= col_scale
    height = 1 + np.argmax(horizon[..., 1:], axis=-1)
    for row in range(mass.shape[0]):
        count = int(row_target[row])
        if count <= 0:
            continue
        score = np.where(mask[row], mass[row], -np.inf)
        chosen = np.argsort(-score, kind="stable")[:count]
        painted[row, chosen] = height[row, chosen]
    return np.where(mask, painted, 0)


def paint_horizon(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Place one row's expected bars on its most likely squares.

    The expected count is the sum of that row's chance of a bar. Each chosen
    square is painted with its most likely positive count. A row is three
    squares tall, and that is the only cap.
    """
    probs = np.asarray(probs, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    chance = np.where(mask, np.clip(1.0 - probs[..., 0], 0.0, None), -np.inf)
    available = int(mask.sum())
    expected = float(chance[mask].sum()) if available else 0.0
    count = int(np.clip(np.rint(expected), 0, available))
    painted = np.zeros(mask.shape, dtype=np.int64)
    if count == 0:
        return painted
    chosen = np.argsort(-chance, axis=None, kind="stable")[:count]
    height = 1 + np.argmax(probs[..., 1:], axis=-1)
    painted.ravel()[chosen] = height.ravel()[chosen]
    return np.where(mask, painted, 0)


def paint_column(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Place one week's expected bars on its most likely squares down the squad.

    The expected count is the sum of that column's chance of a bar. The week
    is scored as a whole, so a column that holds one event is not painted
    from single-cell chances that each round to zero.
    """
    probs = np.asarray(probs, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    chance = np.where(mask, np.clip(1.0 - probs[..., 0], 0.0, None), -np.inf)
    available = int(mask.sum())
    expected = float(chance[mask].sum()) if available else 0.0
    count = int(np.clip(np.rint(expected), 0, available))
    painted = np.zeros(mask.shape, dtype=np.int64)
    if count == 0:
        return painted
    chosen = np.argsort(-chance, axis=None, kind="stable")[:count]
    height = 1 + np.argmax(probs[..., 1:], axis=-1)
    painted.ravel()[chosen] = height.ravel()[chosen]
    return np.where(mask, painted, 0)


def paint_row(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paint the count with the highest probability.

    The event row is three squares tall, and the reader has one class for
    zero and one class for each square. Nothing else limits the count.
    """
    count = np.argmax(np.asarray(probs), axis=-1)
    return np.where(mask, count, 0).astype(np.int64)


def _paint_fitted(
    horizon: np.ndarray,
    column: np.ndarray,
    cards: list,
    cuts: list[int],
) -> np.ndarray:
    """Fit row counts to week counts on every test card."""
    painted = np.zeros(horizon.shape[:-1], dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        painted[index] = fit_margins(horizon[index], column[index], mask)
    return painted


def _paint(probs: np.ndarray, minutes: np.ndarray, cards: list, cuts: list[int]) -> np.ndarray:
    del minutes
    painted = np.zeros(probs.shape[:-1], dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        for row in range(mask.shape[0]):
            painted[index, row] = paint_horizon(probs[index, row], mask[row])
    return painted


def _paint_columns(probs: np.ndarray, cards: list, cuts: list[int]) -> np.ndarray:
    """Paint each hidden week as one count down the squad."""
    painted = np.zeros(probs.shape[:-1], dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        for col in range(mask.shape[1]):
            painted[index, :, col] = paint_column(probs[index, :, col], mask[:, col])
    return painted


def _chance_line(cards: list, probs: np.ndarray, cuts: list[int], channel: str) -> str:
    """Mean chance of a bar on a hidden appearance, by position."""
    names = ("GKP", "DEF", "MID", "FWD")
    total = np.zeros(4)
    count = np.zeros(4)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        for row in range(card.n_players):
            pos = int(card.position[row])
            if pos < 0 or pos > 3:
                continue
            cols = np.where(card.present[row] & hidden)[0]
            total[pos] += float((1.0 - probs[index, row, cols, 0]).sum())
            count[pos] += len(cols)
    rates = total / np.maximum(count, 1.0)
    body = ", ".join(f"{name} {rates[pos]:.3f}" for pos, name in enumerate(names))
    print(f"chance {channel} {body}", flush=True)
    return f"{channel} chance: {body}."


def _position_lines(
    cards: list,
    goals: np.ndarray,
    goal_paint: np.ndarray,
    assists: np.ndarray,
    assist_paint: np.ndarray,
    cuts: list[int],
    heading: str = "Goal and assist bars by position at the 20% cut.",
) -> list[str]:
    """Actual and painted bars by position at one cut. Rows are GKP, DEF, MID, FWD."""
    names = ("GKP", "DEF", "MID", "FWD")
    actual = np.zeros((2, 4), dtype=np.int64)
    painted = np.zeros((2, 4), dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        for row in range(card.n_players):
            pos = int(card.position[row])
            if pos < 0 or pos > 3:
                continue
            cols = np.where(card.present[row] & hidden)[0]
            actual[0, pos] += int((goals[index, row, cols] > 0).sum())
            actual[1, pos] += int((assists[index, row, cols] > 0).sum())
            painted[0, pos] += int((goal_paint[index, row, cols] > 0).sum())
            painted[1, pos] += int((assist_paint[index, row, cols] > 0).sum())
    lines = ["", heading, ""]
    for channel, index in (("goals", 0), ("assists", 1)):
        body = ", ".join(
            f"{name} {int(actual[index, pos])}/{int(painted[index, pos])}"
            for pos, name in enumerate(names)
        )
        lines.append(f"{channel}: actual/painted {body}.")
        print(f"position {channel} {body}", flush=True)
    return lines


def _column_note(
    cards: list,
    goals: np.ndarray,
    assists: np.ndarray,
    goal_probs: np.ndarray,
    assist_probs: np.ndarray,
    cuts: list[int],
) -> str:
    """Mean predicted and actual column counts, and their correlation."""

    def stats(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
        predicted = []
        actual = []
        for index, card in enumerate(cards):
            hidden = (card.gw > cuts[index]) & (card.gw > 0)
            mask = card.present & hidden.reshape(1, -1)
            predicted.append(((1.0 - probs[index, ..., 0]) * mask).sum(axis=0)[hidden])
            actual.append(((labels[index] > 0) & mask).sum(axis=0)[hidden])
        pred = np.concatenate(predicted)
        seen = np.concatenate(actual).astype(np.float64)
        if len(pred) < 2 or float(pred.std()) == 0.0 or float(seen.std()) == 0.0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(pred, seen)[0, 1])
        return float(pred.mean()), float(seen.mean()), corr

    goal = stats(goals, goal_probs)
    assist = stats(assists, assist_probs)
    text = (
        "Column reader at the 20% cut, mean predicted/actual count and correlation: "
        f"goals {goal[0]:.2f}/{goal[1]:.2f} ({goal[2]:.2f}), "
        f"assists {assist[0]:.2f}/{assist[1]:.2f} ({assist[2]:.2f})."
    )
    print(text, flush=True)
    return text


def _line(revealed: float, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | {channel} | {totals['actual']} | {totals['painted']} | {totals['exact']} | "
        f"{totals['actual_within']} | {totals['painted_within']} |"
    )


def _report(lines: list[str]) -> None:
    path = ROOT_REPORTS / "card_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    print("loading card comparison", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    test_squads = squad_index(train + test)
    n_clubs = max(clubs.values(), default=0) + 1
    print(f"train {len(train)} test {len(test)}", flush=True)
    from src.models.flashcard_card_store import CURRENT_DIR, load_reader, save_png, save_reader, write_layout
    from src.models.flashcard_odds_trial import load_odds
    from src.models.flashcard_square_trial import CELL_W, CLUB_H, HEADER_H, ODDS_H, ODDS_Y, OPP_H, PLAYER_H, VENUE_H
    from src.models.flashcard_square_trial import compose_square
    from src.models.flashcard_squad_trial import EPOCHS, apply_cut
    from src.models.flashcard_strip_trial import (
        default_reader,
        paint_stripped,
        strip_features,
        train_columns,
        train_stripped,
    )

    odds = load_odds()
    torch.manual_seed(SEED)
    horizon = default_reader(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped(horizon, train, league, np.random.default_rng(SEED), clubs, odds, "horizon")
    torch.manual_seed(SEED + 1)
    weeks = default_reader(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_columns(weeks, train, league, np.random.default_rng(SEED + 1), clubs, odds, "column")
    layout = {
        "id": "club-opponent-venue-1x2",
        "rows": ["club", "opponent", "venue", "odds-1x2"],
        "header_h": HEADER_H,
        "club_h": CLUB_H,
        "opponent_h": OPP_H,
        "venue_h": VENUE_H,
        "odds_y": ODDS_Y,
        "odds_h": ODDS_H,
        "seed": SEED,
        "column_seed": SEED + 1,
        "epochs": EPOCHS,
        "dropout": 0.1,
        "arch": "fixture_intensity",
        "mode": "fixture",
    }
    write_layout(CURRENT_DIR, layout)
    reader_meta = {
        "arch": "fixture_intensity",
        "mode": "fixture",
        "layout": layout["id"],
        "dropout": 0.1,
    }
    save_reader(CURRENT_DIR, "horizon", horizon, {**reader_meta, "role": "horizon", "seed": SEED})
    save_reader(
        CURRENT_DIR,
        "column",
        weeks,
        {**reader_meta, "role": "column", "seed": SEED + 1},
    )
    horizon = load_reader(CURRENT_DIR, "horizon")
    weeks = load_reader(CURRENT_DIR, "column")
    shown = test[0]
    shown_cut = cut_at_fraction(shown, 0.2)
    picture = compose_square(
        strip_features(apply_cut(shown, league, shown_cut, clubs, test_squads)),
        shown,
        odds,
    )
    save_png(CURRENT_DIR / "example.png", picture, 2)
    save_png(CURRENT_DIR / "header.png", picture[:HEADER_H, : 8 * CELL_W], 8)
    print("scoring stored weights", flush=True)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    lines = [
        "# Card comparison",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "Each training card is cut after 20%, 40%, 60%, or 80% of its fixtures,",
        "chosen at random, starting from gameweek 1. The test scores all four cuts.",
        "Every club and every row is included.",
        "",
        "The reader is squad vision on the square card. The flashcard is the",
        "only feature input. Each column shows the club symbol, the opponent",
        "symbol, home or away, then sixteen 1X2 squares. The",
        "other squad is not an input. Concede, regulars, own history,",
        "opponent history, and the team scoring rate are absent. Minutes,",
        "position, club, and opponent are not passed again as numbers.",
        "Two readers see that card, and their weights stay separate. Goals use",
        "fixture intensity: the match header owns that week's expected bar",
        "total, then the squad shares it. Assists keep a four-way head. The",
        "paint keeps each row's horizon count on that row and moves the bars",
        "toward the weeks the column reader scores highest. A row whose",
        "horizon count is zero stays empty. A row holds three squares, and",
        "that is the only cap.",
        "",
        "A bar matches on the same row only. Within one gameweek includes the same",
        "appearance and either neighboring gameweek. The numbers are totals.",
        "",
        "Horizon rows. This paint uses the horizon reader alone.",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    horizon_position: list[str] = []
    column_position: list[str] = []
    fitted_position: list[str] = []
    column_note = ""
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p, assist_p = paint_stripped(horizon, test, league, clubs, test_squads, odds, cuts)
        goal_paint = _paint(goal_p, minutes, test, cuts)
        assist_paint = _paint(assist_p, minutes, test, cuts)
        goals = tally_cards(test, labels, goal_paint, 0, cuts)
        assists = tally_cards(test, assist_labels, assist_paint, 0, cuts)
        lines.append(_line(revealed, "goals", goals))
        lines.append(_line(revealed, "assists", assists))
        if revealed == 0.2:
            horizon_position = _position_lines(
                test,
                labels,
                goal_paint,
                assist_labels,
                assist_paint,
                cuts,
                "Horizon rows, by position at the 20% cut.",
            )
            horizon_position.append(_chance_line(test, goal_p, cuts, "goal"))
            horizon_position.append(_chance_line(test, assist_p, cuts, "assist"))
        print(
            f"horizon {revealed:.0%} goals {goals['actual']} painted {goals['painted']} "
            f"exact {goals['exact']} within {goals['actual_within']} "
            f"assists exact {assists['exact']} within {assists['actual_within']}",
            flush=True,
        )
    lines.extend(horizon_position)
    lines.extend(
        [
            "",
            "Column weeks. This paint uses the column reader alone. Each row",
            "keeps that reader's own count, and each week is one count down the",
            "squad.",
            "",
            "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_c, assist_c = paint_stripped(weeks, test, league, clubs, test_squads, odds, cuts)
        goal_paint = _paint_fitted(goal_c, goal_c, test, cuts)
        assist_paint = _paint_fitted(assist_c, assist_c, test, cuts)
        goals = tally_cards(test, labels, goal_paint, 0, cuts)
        assists = tally_cards(test, assist_labels, assist_paint, 0, cuts)
        lines.append(_line(revealed, "goals", goals))
        lines.append(_line(revealed, "assists", assists))
        if revealed == 0.2:
            column_position = _position_lines(
                test,
                labels,
                goal_paint,
                assist_labels,
                assist_paint,
                cuts,
                "Column weeks, by position at the 20% cut.",
            )
            column_note = _column_note(test, labels, assist_labels, goal_c, assist_c, cuts)
        print(
            f"column {revealed:.0%} goals {goals['actual']} painted {goals['painted']} "
            f"exact {goals['exact']} within {goals['actual_within']} "
            f"assists exact {assists['exact']} within {assists['actual_within']}",
            flush=True,
        )
    lines.extend(column_position)
    lines.extend(
        [
            "",
            "Fitted weeks. Each row keeps its horizon count. The bars move",
            "toward the weeks the column reader scores highest.",
            "",
            "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_h, assist_h = paint_stripped(horizon, test, league, clubs, test_squads, odds, cuts)
        goal_c, assist_c = paint_stripped(weeks, test, league, clubs, test_squads, odds, cuts)
        goal_paint = _paint_fitted(goal_h, goal_c, test, cuts)
        assist_paint = _paint_fitted(assist_h, assist_c, test, cuts)
        goals = tally_cards(test, labels, goal_paint, 0, cuts)
        assists = tally_cards(test, assist_labels, assist_paint, 0, cuts)
        lines.append(_line(revealed, "goals", goals))
        lines.append(_line(revealed, "assists", assists))
        if revealed == 0.2:
            fitted_position = _position_lines(
                test,
                labels,
                goal_paint,
                assist_labels,
                assist_paint,
                cuts,
                "Fitted weeks, by position at the 20% cut.",
            )
        print(
            f"fitted {revealed:.0%} goals {goals['actual']} painted {goals['painted']} "
            f"exact {goals['exact']} within {goals['actual_within']} "
            f"assists exact {assists['exact']} within {assists['actual_within']}",
            flush=True,
        )
    lines.extend(fitted_position)
    lines.extend(["", column_note])
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "The two readers keep separate weights. Row and week counts are",
            "Poisson rates, so the volume is absolute. A row whose horizon",
            "count is zero stays empty. A matching card total with the bars",
            "on the empty rows is an error. The flashcard is the only feature",
            "input. The picture, the layout, and both state dicts are stored",
            "in `reports/cards/current/`. The table above is scored from those",
            "files.",
            "",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    _report(lines)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
