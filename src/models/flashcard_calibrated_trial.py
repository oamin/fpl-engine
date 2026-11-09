"""Place each fixture's expected bars by rank, and score hits and misses.

The reader is the locked vision model. The fixture total is the sum of its
own bar probabilities, rounded so the season count stays put. Those bars are
given to the players in that fixture with the highest probabilities, and only
to players with minutes.

A true zero with no bar stays out of the score. A bar on a zero is a false
positive. A real event with no bar is a false negative. The summary is the
harmonic mean of precision and recall, so painting nothing and painting
everything both score zero.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    CUT_GW,
    EPOCHS,
    POS,
    SEED,
    SquadSequence,
    SquadVision,
    _hidden_mask,
    build_cards,
    build_league,
    capture_table,
    cut_at_fraction,
    fit_clubs,
    positive_by_position,
    predict,
    random_same_count,
    sample_bars,
    squad_index,
    train_model,
)

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"


def integer_budgets(weights: np.ndarray) -> np.ndarray:
    """Whole bars per column. The columns sum to the rounded total."""
    weights = np.asarray(weights, dtype=np.float64)
    total = int(round(float(weights.sum())))
    floors = np.floor(weights + 1e-9).astype(np.int64)
    leftover = total - int(floors.sum())
    if leftover > 0:
        fraction = weights - floors
        for index in np.argsort(-fraction)[:leftover]:
            floors[index] += 1
    return floors


def place_expected(
    probs: np.ndarray, minutes: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """One bar each for the top players in a fixture, up to that fixture's budget."""
    chance = np.where(mask & (minutes > 0), 1.0 - probs[..., 0], 0.0)
    budgets = integer_budgets(chance.sum(axis=0))
    drawn = np.zeros(minutes.shape, dtype=np.int64)
    for col in range(minutes.shape[1]):
        budget = int(budgets[col])
        if budget <= 0:
            continue
        rows = np.flatnonzero(chance[:, col] > 0)
        if len(rows) == 0:
            continue
        order = rows[np.argsort(-chance[rows, col])]
        drawn[order[: min(budget, len(order))], col] = 1
    return drawn


def rates_from_counts(true_positive: float, false_positive: float, false_negative: float) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    paired = precision + recall
    f1 = 2.0 * precision * recall / paired if paired else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "drawn": true_positive + false_positive,
        "actual": true_positive + false_negative,
    }


def event_rates(actual: np.ndarray, drawn: np.ndarray) -> dict[str, float]:
    """Precision, recall, and their harmonic mean. A blank zero does not count."""
    got = actual > 0
    painted = drawn > 0
    return rates_from_counts(
        float((got & painted).sum()),
        float((~got & painted).sum()),
        float((got & ~painted).sum()),
    )


def sampled_rates(
    probs: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    draws: int = 20,
) -> dict[str, float]:
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for _ in range(draws):
        drawn = np.where(mask, sample_bars(probs, rng), 0)
        actual = np.where(mask, labels, 0)
        got = actual > 0
        painted = drawn > 0
        true_positive += float((got & painted).sum())
        false_positive += float((~got & painted).sum())
        false_negative += float((got & ~painted).sum())
    return rates_from_counts(
        true_positive / draws, false_positive / draws, false_negative / draws
    )


def _placed_rates(
    probs: np.ndarray, labels: np.ndarray, minutes: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    drawn = np.zeros_like(labels)
    for index in range(len(labels)):
        drawn[index] = place_expected(probs[index], minutes[index], mask[index])
    return event_rates(np.where(mask, labels, 0), np.where(mask, drawn, 0))


def _line(revealed: str, decoder: str, rates: dict[str, float], gk: float, fwd: float) -> str:
    return (
        f"| {revealed} | {decoder} | {rates['actual']:.0f} | {rates['drawn']:.1f} | "
        f"{rates['precision']:.1%} | {rates['recall']:.1%} | {rates['f1']:.1%} | "
        f"{gk:.1%} | {fwd:.1%} |"
    )


def _report(lines: list[str]) -> None:
    path = ROOT_REPORTS / "flashcard_calibrated_trial.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    print("loading calibrated cards", flush=True)
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
    goal_p, _assist_p = predict(vision, test, league, CUT_GW, "vision", clubs, test_squads)
    capture = capture_table(test, {"vision": 1.0 - goal_p[:, :, :, 0]}, CUT_GW)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    bar_rng = np.random.default_rng(SEED)
    mask = _hidden_mask(test, CUT_GW)
    cell = sampled_rates(goal_p, labels, mask, bar_rng)
    placed = _placed_rates(goal_p, labels, minutes, mask)
    chance = random_same_count(test, labels, CUT_GW, bar_rng)
    chance_f1 = chance["actual_covered"]
    lines = [
        "# Calibrated placement",
        "",
        "The reader is the locked vision model. Each future fixture is given a",
        "number of bars equal to the sum of its bar probabilities, rounded so",
        "the fixtures add up to the season total. The bars go to the players",
        "in that fixture with the highest probabilities and with minutes.",
        "",
        "A blank zero is not scored. Precision is the share of painted bars that",
        "land on a real event. Recall is the share of real events that were painted.",
        "The harmonic mean is zero if every cell is left blank, and it falls when",
        "extra bars are painted on zeros.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        f"Gameweek-{CUT_GW} top 10/20/50: "
        + ", ".join(f"{value:.1%}" for value in capture["vision"]),
        f"Players: {capture['n_players']}. Locked default: 24.7%, 41.5%, 80.1%.",
        "",
        f"Gameweeks 20-38, cell samples: drawn {cell['drawn']:.1f}, "
        f"precision {cell['precision']:.1%}, recall {cell['recall']:.1%}, F1 {cell['f1']:.1%}.",
        f"Ranked placement: drawn {placed['drawn']:.1f}, "
        f"precision {placed['precision']:.1%}, recall {placed['recall']:.1%}, F1 {placed['f1']:.1%}.",
        f"Same count on random weeks: precision, recall, and F1 are {chance_f1:.1%}.",
        "",
        "## Remainders from gameweek 1",
        "",
        "| revealed | decoder | actual bars | drawn bars | precision | recall | F1 | GK | FWD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for revealed in (0.8, 0.2):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        g = np.zeros_like(goal_p)
        a = np.zeros_like(goal_p)
        for index, card in enumerate(test):
            one_g, one_a = predict(vision, [card], league, cuts[index], "vision", clubs, test_squads)
            g[index] = one_g[0]
            a[index] = one_a[0]
        hidden = _hidden_mask(test, 0, cuts)
        rates = positive_by_position(g, test, 0, cuts)
        for channel, probs, channel_labels in (
            ("goals", g, labels),
            ("assists", a, assist_labels),
        ):
            sampled = sampled_rates(probs, channel_labels, hidden, bar_rng)
            ranked = _placed_rates(probs, channel_labels, minutes, hidden)
            lines.append(_line(f"{revealed:.0%}", f"cell {channel}", sampled, rates[POS["GKP"]], rates[POS["FWD"]]))
            lines.append(_line(f"{revealed:.0%}", f"ranked {channel}", ranked, rates[POS["GKP"]], rates[POS["FWD"]]))
            print(
                f"reveal {revealed:.0%} {channel} cell f1 {sampled['f1']:.3f} "
                f"ranked f1 {ranked['f1']:.3f} drawn {ranked['drawn']:.1f}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    top10 = capture["vision"][0]
    gate = "The capture gate stays at 24.7%." if round(top10 * 1000) <= 247 else (
        "This reader is above the 24.7% capture gate."
    )
    lines.extend(["", gate, f"Wall clock: {seconds:.0f} seconds on CPU.", ""])
    _report(lines)
    print(f"done in {seconds:.0f}s capture10 {capture['vision'][0]:.3f} ranked_f1 {placed['f1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
