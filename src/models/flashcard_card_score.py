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
position, club, and opponent are not passed again as numbers. The loss
scores each hidden square and each row's own hidden bar count. A card
whose total is right still fails when the bars sit on the empty rows.
The paint keeps each row's count on that row. A row holds three squares,
and that is the only cap.
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

# Flashcard painter, 12 Nov 2026. Each row's horizon is scored, and each band
# of the cell is normalized before the bands are added. The flashcard is the only input.
GATE = {
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


def window_count(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Squared relative error of each row's hidden bar count.

    The card's total can match while every bar sits on an empty row. Each
    row is scored on its own, so that arrangement is a large error.
    """
    predicted = ((1.0 - probs[..., 0]) * mask).sum(dim=-1)
    actual = ((labels > 0).to(probs.dtype) * mask).sum(dim=-1)
    scale = torch.maximum(predicted, actual).clamp_min(1.0)
    error = ((predicted - actual) / scale).square()
    present = mask.sum(dim=-1) > 0
    return (error * present).sum() / present.sum().clamp_min(1.0)


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


def paint_row(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Paint the count with the highest probability.

    The event row is three squares tall, and the reader has one class for
    zero and one class for each square. Nothing else limits the count.
    """
    count = np.argmax(np.asarray(probs), axis=-1)
    return np.where(mask, count, 0).astype(np.int64)


def _paint(probs: np.ndarray, minutes: np.ndarray, cards: list, cuts: list[int]) -> np.ndarray:
    del minutes
    painted = np.zeros(probs.shape[:-1], dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        for row in range(mask.shape[0]):
            painted[index, row] = paint_horizon(probs[index, row], mask[row])
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
    lines = ["", "Goal and assist bars by position at the 20% cut.", ""]
    for channel, index in (("goals", 0), ("assists", 1)):
        body = ", ".join(
            f"{name} {int(actual[index, pos])}/{int(painted[index, pos])}"
            for pos, name in enumerate(names)
        )
        lines.append(f"{channel}: actual/painted {body}.")
        print(f"position {channel} {body}", flush=True)
    return lines


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
    from src.models.flashcard_odds_trial import load_odds
    from src.models.flashcard_square_trial import HEADER_H, PLAYER_H
    from src.models.flashcard_strip_trial import StripVision, paint_stripped, train_stripped

    odds = load_odds()
    torch.manual_seed(SEED)
    vision = StripVision(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped(vision, train, league, np.random.default_rng(SEED), clubs, odds, "square card")
    print("scoring", flush=True)
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
        "symbol, home or away, then the odds. The",
        "other squad is not an input. Concede, regulars, own history,",
        "opponent history, and the team scoring rate are absent. Minutes,",
        "position, club, and opponent are not passed again as numbers. Each",
        "loss scores each hidden square and each row's own hidden bar count.",
        "The paint keeps that count on the same row. A row holds three",
        "squares, and that is the only cap.",
        "",
        "A bar matches on the same row only. Within one gameweek includes the same",
        "appearance and either neighboring gameweek. The numbers are totals.",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    position_lines: list[str] = []
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p, assist_p = paint_stripped(vision, test, league, clubs, test_squads, odds, cuts)
        goal_paint = _paint(goal_p, minutes, test, cuts)
        assist_paint = _paint(assist_p, minutes, test, cuts)
        goals = tally_cards(test, labels, goal_paint, 0, cuts)
        assists = tally_cards(test, assist_labels, assist_paint, 0, cuts)
        lines.append(_line(revealed, "goals", goals))
        lines.append(_line(revealed, "assists", assists))
        if revealed == 0.2:
            position_lines = _position_lines(test, labels, goal_paint, assist_labels, assist_paint, cuts)
            position_lines.append(_chance_line(test, goal_p, cuts, "goal"))
            position_lines.append(_chance_line(test, assist_p, cuts, "assist"))
        print(
            f"reveal {revealed:.0%} goals {goals['actual']} painted {goals['painted']} "
            f"exact {goals['exact']} within {goals['actual_within']} "
            f"assists exact {assists['exact']} within {assists['actual_within']}",
            flush=True,
        )
    seconds = time.perf_counter() - started
    lines.extend(position_lines)
    lines.extend(
        [
            "",
            "The loss scores each hidden square and each row's own hidden bar count.",
            "A matching card total with the bars on the empty rows is an error.",
            "The paint keeps each row's count on that row. The flashcard is the",
            "only feature input.",
            "",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    _report(lines)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
