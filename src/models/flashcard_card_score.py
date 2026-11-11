"""Compare the painted test card with the actual card, as totals.

Every club in 2025-26 is cut after 20%, 40%, 60%, and 80% of its fixtures,
starting from gameweek 1. Every row is included. A bar on one row does not
match a goal on another row. The totals are sums, not an average of per-player
rates.

The painted card comes from the locked vision reader. The picture is the
square card: sixteen odds squares, one venue square, six minute squares of
fifteen minutes, and one square per goal or assist. Concede, regulars, own
history, and opponent history are absent from the picture and from the
match-up. Each row paints the count with the highest probability. A row
holds three squares, and that is the only cap.
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

# The locked painter at 21:47 UK on 10 Nov 2026.
GATE = {
    ("goals", 0.2): {"actual": 730, "painted": 759, "exact": 192, "actual_within": 256, "painted_within": 394},
    ("assists", 0.2): {"actual": 707, "painted": 692, "exact": 100, "actual_within": 161, "painted_within": 218},
    ("goals", 0.4): {"actual": 544, "painted": 589, "exact": 147, "actual_within": 200, "painted_within": 305},
    ("assists", 0.4): {"actual": 531, "painted": 535, "exact": 78, "actual_within": 119, "painted_within": 163},
    ("goals", 0.6): {"actual": 360, "painted": 384, "exact": 94, "actual_within": 134, "painted_within": 200},
    ("assists", 0.6): {"actual": 351, "painted": 348, "exact": 55, "actual_within": 84, "painted_within": 110},
    ("goals", 0.8): {"actual": 199, "painted": 204, "exact": 55, "actual_within": 73, "painted_within": 102},
    ("assists", 0.8): {"actual": 192, "painted": 185, "exact": 36, "actual_within": 51, "painted_within": 62},
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
        painted[index] = paint_row(probs[index], mask)
    return painted


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
        "The reader is squad vision on the square card. The opponent is one",
        "square, a club symbol. The other squad is not an input. Concede,",
        "regulars, own history, and opponent history are",
        "absent from the picture and from the match-up. Each row paints the",
        "count with the highest probability. A row holds three squares, and",
        "that is the only cap.",
        "",
        "A bar matches on the same row only. Within one gameweek includes the same",
        "appearance and either neighboring gameweek. The numbers are totals.",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p, assist_p = paint_stripped(vision, test, league, clubs, test_squads, odds, cuts)
        goals = tally_cards(test, labels, _paint(goal_p, minutes, test, cuts), 0, cuts)
        assists = tally_cards(test, assist_labels, _paint(assist_p, minutes, test, cuts), 0, cuts)
        lines.append(_line(revealed, "goals", goals))
        lines.append(_line(revealed, "assists", assists))
        print(
            f"reveal {revealed:.0%} goals {goals['actual']} painted {goals['painted']} "
            f"exact {goals['exact']} within {goals['actual_within']} "
            f"assists exact {assists['exact']} within {assists['actual_within']}",
            flush=True,
        )
    seconds = time.perf_counter() - started
    lines.extend(["", f"Wall clock: {seconds:.0f} seconds on CPU.", ""])
    _report(lines)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
