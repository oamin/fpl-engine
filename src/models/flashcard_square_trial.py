"""Square card. One square is the column width.

The odds row is sixteen squares. Blue is the home win, yellow the draw,
purple the away win, from the top, using the Shin de-vig of the opening
prices. A missing price is hatched. The venue mark is one square. Minutes
are six squares and each square is fifteen minutes, filled from the bottom:
1 to 15 minutes is the first square, 16 to 30 is the second, and 76 minutes
or more fills all six. A goal and an assist are one square each. Those rows
are three squares tall. The position color is one square.

The full painter shows that card for the whole remainder. The stepped painter
shows the true past, keeps the odds, the venue, and the minutes on the next
week, and removes every later column. The viewing grid is not part of the
picture.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_card_score import _paint, tally_cards
from src.models.flashcard_odds_trial import (
    AWAY_ODDS,
    DRAW_ODDS,
    HOME_ODDS,
    column_odds,
    load_odds,
    split_heights,
    visible_odds,
)
from src.models.flashcard_squad_trial import (
    ASSISTS_C,
    AWAY_C,
    BASE_C,
    CELL_W,
    COLS,
    GOALS_C,
    HOME_C,
    MASK_C,
    MASK_ID,
    MINUTES_C,
    POS_C,
    ROWS,
    SEED,
    _hatch,
)
from src.models.flashcard_strip_trial import (
    StripVision,
    _row,
    paint_stripped,
    paint_stripped_steps,
    train_stripped,
    train_stripped_steps,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"

SQ = CELL_W
ODDS_N = 16
MINUTE_N = 6
EVENT_N = 3
ODDS_H = SQ * ODDS_N
VENUE_H = SQ
MINUTE_H = SQ * MINUTE_N
EVENT_H = SQ * EVENT_N
POS_H = SQ
PLAYER_H = MINUTE_H + EVENT_H + EVENT_H + POS_H
HEADER_H = ODDS_H + VENUE_H


def minute_squares(minutes: float) -> int:
    """Squares filled by a minute count. Zero stays empty. Ninety fills six."""
    if minutes <= 0:
        return 0
    return min(MINUTE_N, int(np.ceil(float(minutes) / 15.0)))


def _squares(img: np.ndarray, y0: int, x0: int, count: int, color: tuple[int, int, int], height: int) -> None:
    """Fill whole squares from the bottom of a row. An empty row keeps a baseline."""
    img[y0 : y0 + height, x0 : x0 + SQ] = 255
    img[y0 + height - 1, x0 : x0 + SQ] = BASE_C
    count = max(0, min(int(count), height // SQ))
    if count:
        img[y0 + height - count * SQ : y0 + height, x0 : x0 + SQ] = color


def _odds(img: np.ndarray, x0: int, probs: np.ndarray) -> None:
    y = 0
    for color, count in zip((HOME_ODDS, DRAW_ODDS, AWAY_ODDS), split_heights(probs, ODDS_N), strict=True):
        pixels = int(count) * SQ
        if pixels <= 0:
            continue
        img[y : y + pixels, x0 : x0 + SQ] = color
        y += pixels


def paint_square(
    view: dict[str, np.ndarray],
    probs: np.ndarray,
    visible: np.ndarray,
    grey_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Odds, venue, then each player. A grey column keeps minutes and venue."""
    img = np.full((HEADER_H + ROWS * PLAYER_H, COLS * SQ, 3), 255, dtype=np.uint8)
    grey = np.zeros(COLS, dtype=bool)
    if grey_columns is not None:
        grey[np.asarray(grey_columns, dtype=np.int64)] = True
    for col in range(COLS):
        x0 = col * SQ
        on = view["column_on"][col] >= 0.5
        if (not on) or (not visible[col]) or (not np.isfinite(probs[col, 0])):
            _hatch(img, 0, x0, ODDS_H)
        else:
            _odds(img, x0, probs[col])
        if not on:
            _hatch(img, ODDS_H, x0, VENUE_H + ROWS * PLAYER_H)
            continue
        img[ODDS_H : ODDS_H + VENUE_H, x0 : x0 + SQ] = HOME_C if view["home"][col] > 0.5 else AWAY_C
        for row in range(ROWS):
            y = HEADER_H + row * PLAYER_H
            if view["position"][row] < 0:
                _hatch(img, y, x0, PLAYER_H)
                continue
            img[y + MINUTE_H + 2 * EVENT_H : y + PLAYER_H, x0 : x0 + SQ] = POS_C[int(view["position"][row])]
            if view["present"][row, col] < 0.5:
                _hatch(img, y, x0, MINUTE_H + 2 * EVENT_H)
            else:
                _squares(img, y, x0, minute_squares(float(view["minutes"][row, col])), MINUTES_C, MINUTE_H)
                for offset, token, color in (
                    (MINUTE_H, int(view["goal_token"][row, col]), GOALS_C),
                    (MINUTE_H + EVENT_H, int(view["assist_token"][row, col]), ASSISTS_C),
                ):
                    if token == MASK_ID:
                        img[y + offset : y + offset + EVENT_H, x0 : x0 + SQ] = MASK_C
                    else:
                        _squares(img, y + offset, x0, min(token, EVENT_N), color, EVENT_H)
            if grey[col]:
                _hatch(img, y + MINUTE_H, x0, PLAYER_H - MINUTE_H)
    return img


def compose_square(
    view: dict[str, np.ndarray],
    card,
    odds: dict,
    grey_columns: np.ndarray | None = None,
    target: int | None = None,
) -> np.ndarray:
    """One-shot odds on every fixture. A step, given `target`, stops at that week."""
    if target is None:
        visible = visible_odds(card.gw, "painter")
    else:
        visible = visible_odds(card.gw, "step", target=target)
    return paint_square(view, column_odds(card, odds), visible, grey_columns)


def main() -> None:
    from src.models.flashcard_squad_trial import build_cards, build_league, cut_at_fraction, fit_clubs, squad_index

    started = time.perf_counter()
    print("loading square cards", flush=True)
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
        "# Square card",
        "",
        "The odds row is sixteen squares. The venue mark is one square. Minutes",
        "are six squares of fifteen minutes, filled from the bottom. One goal or",
        "assist is one square, and those rows are three squares tall. The position",
        "color is one square. The full painter shows this card across the remainder.",
        "The stepped painter keeps the odds, the venue, and the minutes on the next",
        "week and removes every later column.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    print("training square full", flush=True)
    torch.manual_seed(SEED)
    full = StripVision(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped(full, train, league, np.random.default_rng(SEED), clubs, odds, "square full", compose_square)
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p, assist_p = paint_stripped(full, test, league, clubs, squads, odds, cuts, compose_square)
        for channel, probs, actual in (("goals", goal_p, labels), ("assists", assist_p, assist_labels)):
            painted = _paint(probs, minutes, test, cuts)
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, "square full", channel, totals))
            print(
                f"square full {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    print("training square step", flush=True)
    torch.manual_seed(SEED)
    stepper = StripVision(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped_steps(stepper, train, league, np.random.default_rng(SEED), clubs, odds, compose_square)
    step_g, step_a = paint_stripped_steps(stepper, test, league, clubs, squads, odds, compose_square)
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        for channel, painted, actual in (("goals", step_g, labels), ("assists", step_a, assist_labels)):
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, "square step", channel, totals))
            print(
                f"square step {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "The gate stays the stripped card measured at 15:53 UK on 9 Nov 2026,",
            "190 goal bars on the same appearance and 258 within one gameweek at",
            "20% revealed, unless this full painter is ahead at every cut.",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    path = ROOT_REPORTS / "flashcard_square_trial.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
