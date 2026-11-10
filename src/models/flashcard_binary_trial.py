"""Score the binary card. The stored gate stays the color full painter.

Odds are black, then a white draw band, then black. The band's height is
the draw share and its position splits home from away. Venue is black for
home and white for away. The position color is a two-pixel rule. The
opponent symbol stays.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.models.flashcard_card_score import _paint, tally_cards
from src.models.flashcard_odds_trial import load_odds
from src.models.flashcard_square_trial import HEADER_H, PLAYER_H, ROOT_REPORTS, compose_binary
from src.models.flashcard_strip_trial import (
    StripVision,
    _row,
    paint_stripped,
    paint_stripped_steps,
    train_stripped,
    train_stripped_steps,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

SEED = 0

# 21:47 UK on 10 Nov 2026. Full painter, before the opponent symbol row.
GATE = {
    ("goals", 0.2): (192, 256),
    ("assists", 0.2): (100, 161),
    ("goals", 0.4): (147, 200),
    ("assists", 0.4): (78, 119),
    ("goals", 0.6): (94, 134),
    ("assists", 0.6): (55, 84),
    ("goals", 0.8): (55, 73),
    ("assists", 0.8): (36, 51),
}


def _compare(channel: str, revealed: float, totals: dict[str, int]) -> str:
    exact, within = GATE[(channel, revealed)]
    return (
        f"| {revealed:.0%} | {channel} | {totals['exact']} | {exact} | "
        f"{totals['exact'] - exact:+d} | {totals['actual_within']} | {within} | "
        f"{totals['actual_within'] - within:+d} |"
    )


def main() -> None:
    from src.models.flashcard_squad_trial import build_cards, build_league, cut_at_fraction, fit_clubs, squad_index

    started = time.perf_counter()
    print("loading binary cards", flush=True)
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
        "# Binary card",
        "",
        "The picture is the square card in black and white. Odds are black,",
        "then a white draw band, then black. The band's height is the draw",
        "share. Home is the black above the band and away is the black below",
        "it. Venue is black for home and white for away. The opponent symbol",
        "stays. Minutes, goals, and assists are black squares filled from the",
        "bottom. A hidden event is a checker. The position color is a two-pixel",
        "rule between player blocks. The other squad is not an input.",
        "",
        "This trial does not replace the gate. The gate is the 21:47 UK full",
        "painter on 10 Nov 2026.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    compared = [
        "",
        "| revealed | channel | same appearance | gate | difference | within 1 GW | gate | difference |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    print("training binary full", flush=True)
    torch.manual_seed(SEED)
    full = StripVision(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped(full, train, league, np.random.default_rng(SEED), clubs, odds, "binary full", compose_binary)
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p, assist_p = paint_stripped(full, test, league, clubs, squads, odds, cuts, compose_binary)
        for channel, probs, actual in (("goals", goal_p, labels), ("assists", assist_p, assist_labels)):
            painted = _paint(probs, minutes, test, cuts)
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, "binary full", channel, totals))
            compared.append(_compare(channel, revealed, totals))
            print(
                f"binary full {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    print("training binary step", flush=True)
    torch.manual_seed(SEED)
    stepper = StripVision(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped_steps(stepper, train, league, np.random.default_rng(SEED), clubs, odds, compose_binary)
    step_g, step_a = paint_stripped_steps(stepper, test, league, clubs, squads, odds, compose_binary)
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        for channel, painted, actual in (("goals", step_g, labels), ("assists", step_a, assist_labels)):
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, "binary step", channel, totals))
            print(
                f"binary step {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "Full painter against the gate. A positive difference is ahead of the gate.",
            *compared,
            "",
            "The gate stays the 21:47 UK full painter.",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    path = ROOT_REPORTS / "flashcard_binary_trial.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
