"""Paint one gameweek, with every later column removed.

The gate paints the whole remainder at once. This test steps forward one
gameweek at a time. Earlier gameweeks stay revealed. The next gameweek keeps
player minutes and the venue, and is grey everywhere else. Columns after that
gameweek are turned off, so their minutes, opponents, and bars are not on the card.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.flashcard_calibrated_trial import place_expected
from src.models.flashcard_card_score import _paint, tally_cards
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    EPOCHS,
    MASK_ID,
    SEED,
    SquadSequence,
    SquadVision,
    _stack,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    predict,
    squad_index,
    train_model,
)

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
BLANK_KEYS = (
    "concede",
    "regulars",
    "own_hist",
    "opp_hist",
    "own_form",
)


def chop_future(view: dict[str, np.ndarray], gameweeks: np.ndarray, target_gw: int) -> None:
    """Drop columns after the target gameweek. The target keeps minutes and venue."""
    off = (gameweeks > target_gw) | (gameweeks <= 0)
    target = gameweeks == target_gw
    view["column_on"][off] = 0
    view["minutes"][:, off] = 0
    view["present"][:, off] = 0
    view["goal_token"][:, off] = MASK_ID
    view["assist_token"][:, off] = MASK_ID
    view["home"][off] = 0
    view["opponent"][off] = 0
    view["goal_token"][:, target] = MASK_ID
    view["assist_token"][:, target] = MASK_ID
    view["opponent"][target] = 0
    for key in BLANK_KEYS:
        view[key][off] = 0
        view[key][target] = 0
        view[key + "_missing"][off] = 1
        view[key + "_missing"][target] = 1
    view["matchup"][off] = 0
    view["matchup"][target] = 0
    for columns in (off, target):
        view["opp_minutes"][:, columns] = 0
        view["opp_present"][:, columns] = 0
        view["opp_position"][:, columns] = -1
        view["opp_goal_token"][:, columns] = MASK_ID
        view["opp_assist_token"][:, columns] = MASK_ID
        view["opp_pad"][:, columns] = True


def paint_steps(model: torch.nn.Module, cards: list, league: dict, clubs: dict, squads: dict) -> tuple[np.ndarray, np.ndarray]:
    """One bar placement per gameweek. The future of the card is off."""
    goals = np.zeros((len(cards),) + cards[0].goals.shape, dtype=np.int64)
    assists = np.zeros_like(goals)
    by_gw: dict[int, list[int]] = {}
    for index, card in enumerate(cards):
        seen: set[int] = set()
        for gw in card.gw:
            number = int(gw)
            if number > 0 and number not in seen:
                seen.add(number)
                by_gw.setdefault(number, []).append(index)
    model.eval()
    for gw, indexes in sorted(by_gw.items()):
        views = []
        grey_columns = []
        for index in indexes:
            card = cards[index]
            view = apply_cut(card, league, gw - 1, clubs, squads)
            chop_future(view, card.gw, gw)
            views.append(view)
            grey_columns.append(np.flatnonzero(card.gw == gw))
        batch = _stack(views, grey_columns)
        with torch.no_grad():
            goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
        goal_p = F.softmax(goal_logits, dim=-1).numpy()
        assist_p = F.softmax(assist_logits, dim=-1).numpy()
        for slot, index in enumerate(indexes):
            card = cards[index]
            mask = card.present & (card.gw == gw).reshape(1, -1)
            goals[index] += place_expected(goal_p[slot], card.minutes, mask)
            assists[index] += place_expected(assist_p[slot], card.minutes, mask)
    return goals.astype(np.int64), assists.astype(np.int64)


def _report(lines: list[str]) -> None:
    path = ROOT_REPORTS / "flashcard_step_trial.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    print("loading stepped cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(test)
    n_clubs = max(clubs.values(), default=0) + 1
    print(f"train {len(train)} test {len(test)}", flush=True)
    torch.manual_seed(SEED)
    sequence = SquadSequence(dropout=0.1, n_clubs=n_clubs)
    vision = SquadVision(dropout=0.1, n_clubs=n_clubs)
    train_model(sequence, train, league, np.random.default_rng(SEED), EPOCHS, "sequence", clubs=clubs)
    train_model(vision, train, league, np.random.default_rng(SEED), EPOCHS, "vision", clubs=clubs)
    print("painting one gameweek at a time", flush=True)
    step_goals, step_assists = paint_steps(vision, test, league, clubs, squads)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    lines = [
        "# One-gameweek step",
        "",
        "The gate paints every remaining fixture at once. The step shows the true",
        "card up to the previous gameweek, greys the next gameweek except player",
        "minutes and the venue, and chops every column after that. The counts use the same",
        "hidden fixtures as the gate.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | procedure | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p = np.zeros(minutes.shape + (4,), dtype=np.float64)
        assist_p = np.zeros_like(goal_p)
        for index, card in enumerate(test):
            one_g, one_a = predict(vision, [card], league, cuts[index], "vision", clubs, squads)
            goal_p[index] = one_g[0]
            assist_p[index] = one_a[0]
        gate_g = tally_cards(test, labels, _paint(goal_p, minutes, test, cuts), 0, cuts)
        gate_a = tally_cards(test, assist_labels, _paint(assist_p, minutes, test, cuts), 0, cuts)
        step_g = tally_cards(test, labels, step_goals, 0, cuts)
        step_a = tally_cards(test, assist_labels, step_assists, 0, cuts)
        lines.append(
            f"| {revealed:.0%} | gate | goals | {gate_g['actual']} | {gate_g['painted']} | {gate_g['exact']} | "
            f"{gate_g['actual_within']} | {gate_g['painted_within']} |"
        )
        lines.append(
            f"| {revealed:.0%} | gate | assists | {gate_a['actual']} | {gate_a['painted']} | {gate_a['exact']} | "
            f"{gate_a['actual_within']} | {gate_a['painted_within']} |"
        )
        lines.append(
            f"| {revealed:.0%} | one GW step | goals | {step_g['actual']} | {step_g['painted']} | {step_g['exact']} | "
            f"{step_g['actual_within']} | {step_g['painted_within']} |"
        )
        lines.append(
            f"| {revealed:.0%} | one GW step | assists | {step_a['actual']} | {step_a['painted']} | {step_a['exact']} | "
            f"{step_a['actual_within']} | {step_a['painted_within']} |"
        )
        print(
            f"reveal {revealed:.0%} gate goals {gate_g['exact']}/{gate_g['actual']} "
            f"step {step_g['exact']}/{step_g['painted']} within {step_g['actual_within']}",
            flush=True,
        )
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "The gate is the one-shot card at these four cuts. The step is a comparison.",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    _report(lines)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
