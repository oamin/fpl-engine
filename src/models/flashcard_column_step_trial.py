"""Stepped column one-shot under the locked fixture-intensity reader.

True past, one next week, that week painted as one squad count. No horizon
reader and no fitted mix. The holdout table is the only report.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_card_score import _position_lines, tally_cards
from src.models.flashcard_odds_trial import load_odds
from src.models.flashcard_squad_trial import (
    SEED,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    squad_index,
)
from src.models.flashcard_square_trial import HEADER_H, PLAYER_H, compose_square
from src.models.flashcard_strip_trial import (
    default_reader,
    paint_stripped_columns,
    train_column_steps,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUT = ROOT_REPORTS / "stepped_column_oneshot.md"


def _line(revealed: float, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | {channel} | {totals['actual']} | {totals['painted']} | "
        f"{totals['exact']} | {totals['actual_within']} | {totals['painted_within']} |"
    )


def main() -> None:
    started = time.perf_counter()
    print("loading stepped column one-shot", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(train + test)
    odds = load_odds()
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    print(f"train {len(train)} test {len(test)}", flush=True)
    torch.manual_seed(SEED + 1)
    model = default_reader(HEADER_H, dropout=0.1, player_h=PLAYER_H)
    train_column_steps(
        model,
        train,
        league,
        np.random.default_rng(SEED + 1),
        clubs,
        odds,
        "column step",
        compose_square,
    )
    print("painting stepped columns", flush=True)
    goal_paint, assist_paint, rates = paint_stripped_columns(
        model, test, league, clubs, squads, odds, compose_square
    )
    lines = [
        "# Stepped column one-shot",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "The reader is the locked fixture-intensity head. Training shows the",
        "true past and one next week. That week is one object: a Poisson rate",
        "for how many bars, then a share over the squad for who. Per-cell",
        "Bernoulli is not used. Paint walks every gameweek the same way.",
        "There is no horizon reader and no fitted mix in this table.",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    position: list[str] = []
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goals = tally_cards(test, labels, goal_paint, 0, cuts)
        assists = tally_cards(test, assist_labels, assist_paint, 0, cuts)
        lines.append(_line(revealed, "goals", goals))
        lines.append(_line(revealed, "assists", assists))
        if revealed == 0.2:
            position = _position_lines(
                test,
                labels,
                goal_paint,
                assist_labels,
                assist_paint,
                cuts,
                "Stepped column one-shot, by position at the 20% cut.",
            )
            position.append(
                f"goal chance on bar {rates['goal_on']:.3f}, off bar {rates['goal_off']:.3f}."
            )
            position.append(
                f"assist chance on bar {rates['assist_on']:.3f}, off bar {rates['assist_off']:.3f}."
            )
        print(
            f"column step {revealed:.0%} goals {goals['actual']} painted {goals['painted']} "
            f"exact {goals['exact']} within {goals['actual_within']} "
            f"assists exact {assists['exact']} within {assists['actual_within']}",
            flush=True,
        )
    lines.extend(position)
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "Each column is scored as a whole. Rows compete for the week's",
            "mass, so a position that never scores cannot keep the bars. The",
            "flashcard is the only feature input.",
            "",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
