"""The picture painter and the noise restorer, on the binary card.

Odds, venue, minutes, goals, and assists are black and white. A hidden
event square is a checker. Both painters train on that picture and are
scored by reading a square as a bar when its color is closer to black
than to white or to the checker. The stored gate stays the 21:47 UK
full painter.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.models.flashcard_card_score import GATE, balanced_score, tally_cards
from src.models.flashcard_diffusion_trial import Denoiser, paint_diffusion, train_denoiser
from src.models.flashcard_image_trial import (
    COLS_SHOWN,
    EVENT_ROW,
    EVENT_SLOTS,
    WHITE,
    _crop_rows,
    _panel,
    _write,
    cells_of,
    hidden_events,
    image_of,
)
from src.models.flashcard_odds_trial import load_odds
from src.models.flashcard_picture_trial import PicturePainter, paint_picture, train_picture
from src.models.flashcard_remainder_trial import _row, minute_rates
from src.models.flashcard_squad_trial import (
    COLS,
    ROWS,
    SEED,
    build_cards,
    build_league,
    cut_at_fraction,
)
from src.models.flashcard_square_trial import INK, SQ, compose_binary
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUT = Path("/opt/cursor/artifacts")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
# Color-card runs of these same painters.
COLOR_PICTURE = 0.0
COLOR_RESTORED = {
    ("goals", 0.2): 0.168,
    ("assists", 0.2): 0.035,
    ("goals", 0.4): 0.177,
    ("assists", 0.4): 0.039,
    ("goals", 0.6): 0.188,
    ("assists", 0.6): 0.035,
    ("goals", 0.8): 0.221,
    ("assists", 0.8): 0.035,
}


def checker_patch() -> np.ndarray:
    """One hidden square. The phase matches the binary card's checker."""
    ys = np.arange(SQ)[:, None]
    xs = np.arange(SQ)[None, :]
    mark = ((xs + ys) % 2) == 0
    patch = np.empty((SQ, SQ, 3), dtype=np.uint8)
    patch[:] = np.where(mark[..., None], INK, WHITE)
    return patch


CHECKER = checker_patch()


def hide_checker(images: np.ndarray, where: np.ndarray) -> None:
    """Cover hidden event squares with the binary checker."""
    _write(images, where, [CHECKER] * EVENT_SLOTS)


def binary_ink(cells: np.ndarray, present: np.ndarray) -> np.ndarray:
    """True where an event square is closer to black than to white or a checker."""
    single = cells.ndim == 5
    if single:
        cells = cells[None]
        present = present[None]
    ink = np.zeros((cells.shape[0], ROWS, EVENT_SLOTS, COLS), dtype=bool)
    values = cells.astype(np.int16)
    black = np.zeros(3, dtype=np.int16)
    white = np.full(3, 255, dtype=np.int16)
    mid = CHECKER.astype(np.int16).mean(axis=(0, 1))
    for slot in range(EVENT_SLOTS):
        mean = values[:, EVENT_ROW[:, slot]].mean(axis=(3, 4))
        dist_black = np.abs(mean - black).sum(axis=-1)
        dist_white = np.abs(mean - white).sum(axis=-1)
        dist_mid = np.abs(mean - mid).sum(axis=-1)
        ink[:, :, slot] = (dist_black < dist_white) & (dist_black < dist_mid) & present
    return ink[0] if single else ink


def render_binary(card, odds: dict) -> np.ndarray:
    """The finished binary card, split into squares."""
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    assists = np.zeros((ROWS, COLS), dtype=np.int64)
    goals[card.present] = card.goals[card.present]
    assists[card.present] = card.assists[card.present]
    view = {
        "column_on": (card.gw > 0).astype(np.float32),
        "home": card.home.astype(np.float32),
        "position": card.position.astype(np.int64),
        "present": card.present.astype(np.float32),
        "minutes": card.minutes.astype(np.float32),
        "goal_token": goals,
        "assist_token": assists,
    }
    return cells_of(compose_binary(view, card, odds))


def _examples(card, cells: np.ndarray, painted: np.ndarray, cut: int, filename: str, title: str) -> None:
    col_count = min(COLS_SHOWN, int(card.n_fixtures))
    rows = _crop_rows(card, col_count)
    shown_cols = int(np.sum((card.gw[:col_count] > 0) & (card.gw[:col_count] <= cut)))
    hidden_cols = np.flatnonzero((card.gw[:col_count] > cut) & (card.gw[:col_count] > 0))
    shown = np.stack([cells])
    hide_checker(shown, hidden_events([card], [cut]))
    panels = []
    for name, picture in (
        ("Shown", image_of(shown[0])),
        ("Actual", image_of(cells)),
        (title, image_of(painted)),
    ):
        if name == "Shown":
            subtitle = "a hidden event is a checker"
        else:
            ink = binary_ink(cells_of(picture), np.ones((ROWS, COLS), dtype=bool))
            goals = int(ink[:, :3, hidden_cols].any(axis=1)[rows].sum()) if len(hidden_cols) else 0
            assists = int(ink[:, 3:, hidden_cols].any(axis=1)[rows].sum()) if len(hidden_cols) else 0
            subtitle = f"{goals} goal rows and {assists} assist rows in this crop"
        panels.append(_panel(picture, card, rows, col_count, name, subtitle, shown_cols))
    gap = 18
    sheet = Image.new(
        "RGB",
        (sum(panel.width for panel in panels) + gap * (len(panels) - 1), max(panel.height for panel in panels)),
        WHITE,
    )
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / filename)


def _score(name: str, test, test_cells, labels, assist_labels, minutes, paint, color_of) -> dict:
    compared: list[str] = []
    counted: list[str] = []
    faults: list[str] = []
    chances: dict[float, dict] = {}
    kept = None
    arsenal_cut = None
    for revealed in FRACTIONS:
        cuts = [cut_at_fraction(card, revealed) for card in test]
        painted_g, painted_a, image, ink, fault = paint(cuts, revealed)
        if image is not None:
            kept = image
            arsenal_cut = cut_at_fraction(test[next(i for i, card in enumerate(test) if card.team == "Arsenal")], revealed)
        present_hidden = hidden_events(test, cuts)[:, :, 0, :]
        actual_ink = binary_ink(test_cells, np.stack([card.present for card in test]))
        chances[revealed] = {
            "painted": minute_rates(ink, minutes, present_hidden),
            "actual": minute_rates(actual_ink & present_hidden[:, :, None, :], minutes, present_hidden),
        }
        faults.append(
            f"| {revealed:.0%} | {fault['goal_filled']} | {fault['goal_bad']} | "
            f"{fault['assist_filled']} | {fault['assist_bad']} |"
        )
        for channel, painted, actual in (("goals", painted_g, labels), ("assists", painted_a, assist_labels)):
            totals = tally_cards(test, actual, painted, 0, cuts)
            counted.append(_row(revealed, channel, totals))
            score = balanced_score(totals)
            gate = balanced_score(GATE[(channel, revealed)])
            color = color_of(channel, revealed)
            compared.append(
                f"| {revealed:.0%} | {channel} | {totals['actual']} | {totals['painted']} | {totals['exact']} | "
                f"{score['score']:.3f} | {gate['score']:.3f} | {color:.3f} |"
            )
            print(
                f"binary {name} {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} score {score['score']:.3f}",
                flush=True,
            )
    return {
        "counted": counted,
        "compared": compared,
        "faults": faults,
        "chances": chances,
        "kept": kept,
        "cut": arsenal_cut,
    }


def _minute_lines(chances: dict[float, dict]) -> list[str]:
    lines = []
    for revealed in FRACTIONS:
        for source in ("actual", "painted"):
            item = chances[revealed][source]
            for channel in ("goal", "assist"):
                for label in ("zero", "played"):
                    slots = item[f"{channel}_{label}_slots"]
                    rate = item[f"{channel}_{label}"]
                    lines.append(
                        f"| {revealed:.0%} {source} {channel}s | {label} | "
                        f"{slots[0]:.3f} | {slots[1]:.3f} | {slots[2]:.3f} | {rate:.3f} |"
                    )
    return lines


def execute() -> dict:
    started = time.perf_counter()
    device = torch.device("cpu")
    print("rendering binary cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    odds = load_odds()
    train_cells = np.stack([render_binary(card, odds) for card in train])
    test_cells = np.stack([render_binary(card, odds) for card in test])
    print(f"train {len(train)} test {len(test)} binary pictures", flush=True)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    arsenal = next(i for i, card in enumerate(test) if card.team == "Arsenal")
    torch.manual_seed(SEED)
    picture = PicturePainter().to(device)
    print("training binary picture", flush=True)
    train_picture(picture, train, train_cells, np.random.default_rng(SEED), device, epochs=40, hide=hide_checker)

    def paint_one(cuts, revealed):
        keep = arsenal if revealed == 0.2 else None
        goals, assists, image, ink, fault, _error = paint_picture(
            picture, test, test_cells, cuts, device, keep, hide=hide_checker, read=binary_ink
        )
        return goals, assists, image, ink, fault

    picture_score = _score(
        "picture",
        test,
        test_cells,
        labels,
        assist_labels,
        minutes,
        paint_one,
        lambda channel, revealed: COLOR_PICTURE,
    )
    torch.manual_seed(SEED)
    denoiser = Denoiser().to(device)
    print("training binary diffusion", flush=True)
    train_denoiser(denoiser, train, train_cells, np.random.default_rng(SEED), device, hide=hide_checker)

    def paint_restored(cuts, revealed):
        keep = arsenal if revealed == 0.2 else None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED)
        return paint_diffusion(
            denoiser, test, test_cells, cuts, device, generator, keep, hide=hide_checker, read=binary_ink
        )

    restored_score = _score(
        "restored",
        test,
        test_cells,
        labels,
        assist_labels,
        minutes,
        paint_restored,
        lambda channel, revealed: COLOR_RESTORED[(channel, revealed)],
    )
    lines = [
        "# Binary card painters",
        "",
        "The training set is the binary card. Odds, venue, minutes, goals,",
        "and assists are black and white. A hidden event square is a checker.",
        "The picture painter writes a color on every hidden square in one",
        "pass. The restored painter starts those squares as noise and rebuilds",
        "the picture in 32 steps. A bar is a square closer to black than to",
        "white or to the checker. The other squad is not an input.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "The color score is the same painter on the color card. The gate is",
        "the 21:47 UK full painter. This run does not replace it.",
        "",
        "## Picture painter",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
        *picture_score["counted"],
        "",
        "| revealed | channel | actual bars | binary bars | binary hits | binary score | gate score | color picture score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *picture_score["compared"],
        "",
        "## Restored painter",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
        *restored_score["counted"],
        "",
        "| revealed | channel | actual bars | binary bars | binary hits | binary score | gate score | color restored score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *restored_score["compared"],
        "",
        "Squares are listed from the bottom. Zero minutes is an appearance",
        "with no minutes. Played is an appearance with minutes.",
        "",
        "### Picture painter",
        "",
        "| source | minutes | 1st square | 2nd square | 3rd square | any square |",
        "|---|---|---:|---:|---:|---:|",
        *_minute_lines(picture_score["chances"]),
        "",
        "| revealed | goal stacks | skipping | assist stacks | skipping |",
        "|---|---:|---:|---:|---:|",
        *picture_score["faults"],
        "",
        "### Restored painter",
        "",
        "| source | minutes | 1st square | 2nd square | 3rd square | any square |",
        "|---|---|---:|---:|---:|---:|",
        *_minute_lines(restored_score["chances"]),
        "",
        "| revealed | goal stacks | skipping | assist stacks | skipping |",
        "|---|---:|---:|---:|---:|",
        *restored_score["faults"],
        "",
    ]
    path = ROOT_REPORTS / "flashcard_binary_painter_trial.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    _examples(test[arsenal], test_cells[arsenal], picture_score["kept"], picture_score["cut"], "binary-picture-arsenal.png", "Picture")
    _examples(test[arsenal], test_cells[arsenal], restored_score["kept"], restored_score["cut"], "binary-restored-arsenal.png", "Restored")
    print(f"binary painters finished in {time.perf_counter() - started:.0f}s", flush=True)
    return {"report": path}


def main() -> None:
    execute()


if __name__ == "__main__":
    main()
