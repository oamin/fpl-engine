"""The three painters, with each event square on or off.

A hidden square keeps the mask bit. A draw paints zero, one, two, or three
black squares. The stored gate stays the 21:47 UK full painter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models.flashcard_generator_trial import execute
from src.models.flashcard_squad_trial import apply_cut, cut_at_fraction, squad_index
from src.models.flashcard_square_trial import HEADER_H, PLAYER_H, SQ, compose_binary
from src.models.flashcard_strip_trial import strip_features

OUT = Path("/opt/cursor/artifacts")
SCALE = 5
COLS_SHOWN = 16
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
NAME_W = 156


def _rows(card, col_count: int) -> list[int]:
    groups: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for row in range(len(card.position)):
        pos = int(card.position[row])
        if pos < 0 or float(card.minutes[row, :col_count].sum()) <= 0:
            continue
        groups[pos].append(row)
    picked: list[int] = []
    for pos, keep in ((0, 1), (1, 2), (2, 3), (3, 2)):
        ranked = sorted(groups[pos], key=lambda row: -float(card.minutes[row, :col_count].sum()))
        picked.extend(ranked[:keep])
    return sorted(picked)


def _copy_view(view: dict) -> dict:
    return {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in view.items()}


def _panel(picture: np.ndarray, card, rows: list[int], col_count: int, title: str, subtitle: str, cut_cols: int) -> Image.Image:
    width = col_count * SQ
    parts = [picture[:HEADER_H, :width]]
    for row in rows:
        y = HEADER_H + row * PLAYER_H
        parts.append(picture[y : y + PLAYER_H, :width])
    crop = np.concatenate(parts, axis=0)
    zoom = Image.fromarray(crop).resize((crop.shape[1] * SCALE, crop.shape[0] * SCALE), Image.Resampling.NEAREST)
    header_h, gw_h, pad = 46, 18, 10
    sheet = Image.new("RGB", (NAME_W + zoom.width + pad, header_h + gw_h + zoom.height + pad), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 6), title, fill=(0, 0, 0), font=FONT_B)
    draw.text((pad, 26), subtitle, fill=(60, 60, 60), font=FONT)
    top = header_h + gw_h
    sheet.paste(zoom, (NAME_W, top))
    for index, row in enumerate(rows):
        y = top + (HEADER_H + index * PLAYER_H) * SCALE + 8
        draw.text((8, y), str(card.names[row])[:24], fill=(20, 20, 20), font=FONT)
    for col in range(col_count):
        draw.text((NAME_W + col * SQ * SCALE + 1, header_h), str(int(card.gw[col])), fill=(60, 60, 60), font=FONT_S)
    if 0 < cut_cols < col_count:
        x = NAME_W + cut_cols * SQ * SCALE
        draw.line((x, header_h, x, sheet.height - 4), fill=(0, 0, 0), width=2)
    return sheet


def _examples(result: dict) -> None:
    test = result["test"]
    index = next(i for i, card in enumerate(test) if card.team == "Arsenal")
    card = test[index]
    cut = cut_at_fraction(card, 0.2)
    col_count = min(COLS_SHOWN, int(card.n_fixtures))
    rows = _rows(card, col_count)
    shown_cols = int(np.sum((card.gw[:col_count] > 0) & (card.gw[:col_count] <= cut)))
    hidden_cols = np.flatnonzero((card.gw[:col_count] > cut) & (card.gw[:col_count] > 0))
    squads = squad_index(result["train"] + test)
    view = strip_features(apply_cut(card, result["league"], cut, result["clubs"], squads))
    hidden = (card.gw > cut) & (card.gw > 0)
    hidden = card.present & hidden.reshape(1, -1)
    panels = []
    order = (
        ("Shown", None),
        ("Actual", None),
        ("MaskGIT", "maskgit"),
        ("Column", "column"),
        ("VAR", "var"),
    )
    for title, key in order:
        painted = _copy_view(view)
        if title == "Actual":
            painted["goal_token"][card.present] = card.goals[card.present]
            painted["assist_token"][card.present] = card.assists[card.present]
            goals, assists = card.goals, card.assists
        elif key is None:
            goals = np.zeros_like(card.goals)
            assists = np.zeros_like(card.assists)
        else:
            goals, assists = result["paints"][key]
            goals, assists = goals[index], assists[index]
            painted["goal_token"][hidden] = goals[hidden]
            painted["assist_token"][hidden] = assists[hidden]
        if title == "Shown":
            subtitle = "a hidden event is a checker"
        else:
            g_n = int((goals[np.ix_(rows, hidden_cols)] > 0).sum())
            a_n = int((assists[np.ix_(rows, hidden_cols)] > 0).sum())
            subtitle = f"{g_n} goal rows and {a_n} assist rows in this crop"
        picture = compose_binary(painted, card, result["odds"])
        panel = _panel(picture, card, rows, col_count, title, subtitle, shown_cols)
        panel.save(OUT / f"generator-binary-{title.lower()}.png")
        panels.append(panel)
    gap = 18
    sheet = Image.new("RGB", (sum(panel.width for panel in panels) + gap * (len(panels) - 1), max(panel.height for panel in panels)), (255, 255, 255))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + gap
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "generator-binary-cards-arsenal.png")
    print("binary cards", sheet.size, flush=True)


def main() -> None:
    result = execute(True)
    if result is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        _examples(result)


if __name__ == "__main__":
    main()
