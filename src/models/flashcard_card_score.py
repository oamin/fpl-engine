"""Compare the painted test card with the actual card, as totals.

Every club in 2025-26 is cut after 20%, 40%, 60%, and 80% of its fixtures,
starting from gameweek 1. Every row is included. A bar on one row does not
match a goal on another row. The totals are sums, not an average of per-player
rates.

The painted card comes from the locked vision reader. Each future fixture
gets a number of bars equal to the sum of its bar probabilities. Those bars
go to the rows with minutes and the highest probabilities.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_calibrated_trial import place_expected
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    EPOCHS,
    SEED,
    SquadSequence,
    SquadVision,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    predict,
    squad_index,
    train_model,
)

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
KEYS = ("actual", "painted", "exact", "actual_within", "painted_within")


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


def _paint(probs: np.ndarray, minutes: np.ndarray, cards: list, cuts: list[int]) -> np.ndarray:
    painted = np.zeros(minutes.shape, dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        painted[index] = place_expected(probs[index], minutes[index], mask)
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
    test_squads = squad_index(test)
    n_clubs = max(clubs.values(), default=0) + 1
    print(f"train {len(train)} test {len(test)}", flush=True)
    torch.manual_seed(SEED)
    sequence = SquadSequence(dropout=0.1, n_clubs=n_clubs)
    vision = SquadVision(dropout=0.1, n_clubs=n_clubs)
    train_model(sequence, train, league, np.random.default_rng(SEED), EPOCHS, "sequence", clubs=clubs)
    train_model(vision, train, league, np.random.default_rng(SEED), EPOCHS, "vision", clubs=clubs)
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
        "The reader is squad vision with squad coupling. Each future fixture is given",
        "a number of bars equal to the sum of its bar probabilities. Those bars go to",
        "the rows with minutes and the highest probabilities.",
        "",
        "A bar matches on the same row only. Within one gameweek includes the same",
        "appearance and either neighboring gameweek. The numbers are totals.",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p = np.zeros(minutes.shape + (4,), dtype=np.float64)
        assist_p = np.zeros_like(goal_p)
        for index, card in enumerate(test):
            one_g, one_a = predict(vision, [card], league, cuts[index], "vision", clubs, test_squads)
            goal_p[index] = one_g[0]
            assist_p[index] = one_a[0]
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
