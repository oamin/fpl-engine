"""The card that painted 798 goals. Odds, then home or away, then the opponent.

This is the header from the run recorded in commit f3ecae4. It has no club
row. The header is 72 pixels, and the header layer reads 864 values. The
current card is a different picture. Its weights cannot load into this one.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.models.flashcard_card_store import (
    LEGACY_DIR,
    load_reader,
    save_png,
    save_reader,
    write_layout,
)
from src.models.flashcard_odds_trial import (
    AWAY_ODDS,
    DRAW_ODDS,
    HOME_ODDS,
    column_odds,
    split_heights,
    visible_odds,
)
from src.models.flashcard_squad_trial import (
    ASSISTS_C,
    AWAY_C,
    EPOCHS,
    GOALS_C,
    HOME_C,
    MASK_C,
    MASK_ID,
    MINUTES_C,
    POS_C,
    ROWS,
    SEED,
    _hatch,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    squad_index,
)
from src.models.flashcard_square_trial import (
    CELL_W,
    COLS,
    EVENT_H,
    EVENT_N,
    MINUTE_H,
    PLAYER_H,
    SQ,
    _opponent_square,
    _squares,
    minute_squares,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ODDS_N = 16
ODDS_H = SQ * ODDS_N
VENUE_H = SQ
OPP_H = SQ
HEADER_H = ODDS_H + VENUE_H + OPP_H
LAYOUT = {
    "id": "odds-venue-opponent",
    "rows": ["odds-1x2", "venue", "opponent"],
    "header_h": HEADER_H,
    "odds_h": ODDS_H,
    "venue_h": VENUE_H,
    "opponent_h": OPP_H,
    "seed": SEED,
    "epochs": EPOCHS,
    "dropout": 0.1,
    "source": "f3ecae4",
}


def _odds(img: np.ndarray, x0: int, probs: np.ndarray) -> None:
    y = 0
    for color, count in zip(
        (HOME_ODDS, DRAW_ODDS, AWAY_ODDS), split_heights(probs, ODDS_N), strict=True
    ):
        pixels = int(count) * SQ
        if pixels <= 0:
            continue
        img[y : y + pixels, x0 : x0 + SQ] = color
        y += pixels


def paint_legacy(
    view: dict[str, np.ndarray],
    probs: np.ndarray,
    visible: np.ndarray,
    grey_columns: np.ndarray | None = None,
    opponents: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Odds, venue, the opponent symbol, then each player.

    A grey column keeps the odds, the venue, the opponent, and the minutes.
    There is no club row.
    """
    img = np.full((HEADER_H + ROWS * PLAYER_H, COLS * SQ, 3), 255, dtype=np.uint8)
    grey = np.zeros(COLS, dtype=bool)
    if grey_columns is not None:
        grey[np.asarray(grey_columns, dtype=np.int64)] = True
    names = opponents or ()
    for col in range(COLS):
        x0 = col * SQ
        on = view["column_on"][col] >= 0.5
        if (not on) or (not visible[col]) or (not np.isfinite(probs[col, 0])):
            _hatch(img, 0, x0, ODDS_H)
        else:
            _odds(img, x0, probs[col])
        if not on:
            _hatch(img, ODDS_H, x0, HEADER_H - ODDS_H + ROWS * PLAYER_H)
            continue
        img[ODDS_H : ODDS_H + VENUE_H, x0 : x0 + SQ] = HOME_C if view["home"][col] > 0.5 else AWAY_C
        _opponent_square(img, ODDS_H + VENUE_H, x0, names[col] if col < len(names) else "")
        for row in range(ROWS):
            y = HEADER_H + row * PLAYER_H
            if view["position"][row] < 0:
                _hatch(img, y, x0, PLAYER_H)
                continue
            img[y + MINUTE_H + 2 * EVENT_H : y + PLAYER_H, x0 : x0 + SQ] = POS_C[
                int(view["position"][row])
            ]
            if view["present"][row, col] < 0.5:
                _hatch(img, y, x0, MINUTE_H + 2 * EVENT_H)
            else:
                _squares(
                    img,
                    y,
                    x0,
                    minute_squares(float(view["minutes"][row, col])),
                    MINUTES_C,
                    MINUTE_H,
                )
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


def compose_legacy(
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
    return paint_legacy(view, column_odds(card, odds), visible, grey_columns, card.opponent)


def _example(card, league, clubs, squads, odds) -> np.ndarray:
    from src.models.flashcard_strip_trial import strip_features

    cut = cut_at_fraction(card, 0.2)
    view = strip_features(apply_cut(card, league, cut, clubs, squads))
    return compose_legacy(view, card, odds)


def main() -> None:
    """Train the old header again and store that reader next to its picture."""
    from src.models.flashcard_card_score import (
        _chance_line,
        _line,
        _paint,
        _position_lines,
        tally_cards,
    )
    from src.models.flashcard_odds_trial import load_odds
    from src.models.flashcard_strip_trial import StripVision, paint_stripped, train_stripped

    started = time.perf_counter()
    print("loading legacy card", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    test_squads = squad_index(train + test)
    n_clubs = max(clubs.values(), default=0) + 1
    odds = load_odds()
    print(f"train {len(train)} test {len(test)} header {HEADER_H}", flush=True)
    torch.manual_seed(SEED)
    model = StripVision(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped(
        model, train, league, np.random.default_rng(SEED), clubs, odds, "legacy", compose_legacy
    )
    write_layout(LEGACY_DIR, LAYOUT)
    save_reader(
        LEGACY_DIR, "horizon", model, {"role": "horizon", "layout": LAYOUT["id"], "dropout": 0.1}
    )
    stored = load_reader(LEGACY_DIR, "horizon")
    picture = _example(test[0], league, clubs, test_squads, odds)
    save_png(LEGACY_DIR / "example.png", picture, 2)
    save_png(LEGACY_DIR / "header.png", picture[:HEADER_H, : 8 * CELL_W], 8)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    lines = [
        "# Legacy card",
        "",
        "Header, from the top: sixteen 1X2 squares, home or away, the opponent symbol.",
        "There is no club row. The header is 72 pixels. The weights are `horizon.pt`.",
        "Poisson row rates. This run painted the holdout under the absolute count loss.",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    position: list[str] = []
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goal_p, assist_p = paint_stripped(
            stored, test, league, clubs, test_squads, odds, cuts, compose_legacy
        )
        goal_paint = _paint(goal_p, minutes, test, cuts)
        assist_paint = _paint(assist_p, minutes, test, cuts)
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
                "By position at the 20% cut.",
            )
            position.append(_chance_line(test, goal_p, cuts, "goal"))
            position.append(_chance_line(test, assist_p, cuts, "assist"))
        print(
            f"legacy {revealed:.0%} goals {goals['actual']} painted {goals['painted']} exact {goals['exact']}",
            flush=True,
        )
    lines.extend(
        ["", *position, "", f"Wall clock: {time.perf_counter() - started:.0f} seconds on CPU.", ""]
    )
    path = LEGACY_DIR / "score.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
