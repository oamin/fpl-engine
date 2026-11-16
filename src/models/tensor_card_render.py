"""Render typed-tensor FOV cards (truth / input FOV / opacity paint)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.models.tensor_card import ART, MAX_PLAYERS, REPORTS, TensorCard, cut_at_fraction
from src.models.tensor_card_stepped import TypedCardMixer, _event_ink, _predict_card

BG = (246, 244, 239)
INK = (28, 28, 28)
MUTED = (120, 118, 112)
GRID = (210, 206, 198)
PLATE = (255, 255, 255)
PAST = (32, 32, 32)
BLANK = (230, 228, 222)
HEADER = (70, 90, 120)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _regulars(card: TensorCard, n: int = 11) -> list[int]:
    minutes = card.minutes.sum(axis=1)
    order = np.argsort(-minutes)
    picked = [int(i) for i in order if i < card.n_players and int(card.position[i]) >= 0]
    picked = picked[:n]
    return sorted(picked, key=lambda i: (int(card.position[i]), -float(minutes[i])))


def _gray(p: float) -> tuple[int, int, int]:
    p = float(np.clip(p, 0.0, 1.0))
    v = int(round(255 * (1.0 - p)))
    return (v, v, v)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    card: TensorCard,
    players: list[int],
    mode: str,
    soft: np.ndarray | None,
    cut: int,
    title: str,
) -> None:
    """mode: truth | fov | paint_week | paint_remainder."""
    left, top = origin
    dot, gap = 12, 3
    step = dot + gap
    font = _font(11)
    small = _font(9)
    channels = (("G", 0), ("A", 1))
    draw.text((left, top - 22), title, fill=INK, font=font)
    # column headers: G/A under a short surname
    for j, row in enumerate(players):
        name = card.names[row].split()[-1][:6]
        x0 = left + j * 2 * step
        draw.text((x0, top - 2), name, fill=MUTED, font=small)
        for c, (lab, _) in enumerate(channels):
            x = left + (j * 2 + c) * step
            draw.text((x + 1, top + 10), lab, fill=MUTED, font=small)
    body_top = top + 26
    for gw in range(card.n_fixtures):
        y = body_top + gw * step
        # week label
        draw.text((left - 28, y), f"{gw+1:02d}", fill=MUTED, font=small)
        for j, row in enumerate(players):
            for c, (_lab, ch) in enumerate(channels):
                x = left + (j * 2 + c) * step
                box = (x, y, x + dot, y + dot)
                present = bool(card.present[row, gw])
                if not present:
                    draw.rectangle(box, outline=GRID, fill=PLATE)
                    continue
                truth = float(_event_ink(card)[row, gw, ch])
                if mode == "truth":
                    fill = PAST if truth >= 0.5 else BLANK
                elif mode == "fov":
                    if gw < cut:
                        fill = PAST if truth >= 0.5 else BLANK
                    else:
                        fill = (255, 255, 255)
                        draw.rectangle(box, outline=(180, 180, 180), fill=fill)
                        continue
                elif mode == "paint_week":
                    if gw < cut:
                        fill = PAST if truth >= 0.5 else BLANK
                    elif gw == cut:
                        p = float(soft[row, gw, ch]) if soft is not None else 0.0
                        fill = _gray(p)
                    else:
                        fill = (255, 255, 255)
                        draw.rectangle(box, outline=(180, 180, 180), fill=fill)
                        continue
                else:  # paint_remainder
                    if gw < cut:
                        fill = PAST if truth >= 0.5 else BLANK
                    else:
                        p = float(soft[row, gw, ch]) if soft is not None else 0.0
                        fill = _gray(p)
                draw.rectangle(box, outline=GRID, fill=fill)
        # cut marker
        if gw == cut:
            y0 = body_top + gw * step - 1
            width = len(players) * 2 * step
            draw.line((left, y0, left + width, y0), fill=HEADER, width=2)


def render_stepped_d1(
    card: TensorCard,
    model: TypedCardMixer,
    player_codes: dict[int, int],
    frac: float = 0.4,
    path: Path | None = None,
) -> Path:
    """Truth | stepped input FOV | opacity paint on the d1 week."""
    cut = cut_at_fraction(card, frac)
    device = next(model.parameters()).device
    soft = _predict_card(model, card, cut, player_codes, device)
    players = _regulars(card)
    panel_w = 28 + len(players) * 2 * 15 + 20
    panel_h = 40 + card.n_fixtures * 15 + 20
    gutter = 36
    width = 24 + 3 * panel_w + 2 * gutter
    height = 70 + panel_h + 48
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    title = _font(16)
    draw.text(
        (24, 16),
        f"Stepped d1 — {card.team} {card.season} · cut {frac:.0%} (week {cut+1})",
        fill=INK,
        font=title,
    )
    draw.text(
        (24, 40),
        "Past = true G/A solids. From the cut: blank on input. Painted week = opacity (darker = likelier).",
        fill=MUTED,
        font=_font(12),
    )
    origins = [
        (48, 88),
        (48 + panel_w + gutter, 88),
        (48 + 2 * (panel_w + gutter), 88),
    ]
    _draw_panel(draw, origins[0], card, players, "truth", None, cut, "1. Truth")
    _draw_panel(draw, origins[1], card, players, "fov", None, cut, "2. Stepped FOV input")
    _draw_panel(draw, origins[2], card, players, "paint_week", soft, cut, "3. Paint d1 week")
    if path is None:
        path = REPORTS / f"typed_tensor_stepped_d1_{card.team.lower()}_{card.season}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    ART.mkdir(parents=True, exist_ok=True)
    img.save(ART / path.name)
    return path


def render_horizon(
    card: TensorCard,
    model: TypedCardMixer,
    player_codes: dict[int, int],
    frac: float = 0.4,
    path: Path | None = None,
) -> Path:
    """Truth | horizon FOV input | one-shot remainder opacity."""
    from src.models.tensor_card_horizon import predict_horizon

    cut = cut_at_fraction(card, frac)
    device = next(model.parameters()).device
    soft = predict_horizon(model, card, cut, player_codes, device)
    players = _regulars(card)
    panel_w = 28 + len(players) * 2 * 15 + 20
    panel_h = 40 + card.n_fixtures * 15 + 20
    gutter = 36
    width = 24 + 3 * panel_w + 2 * gutter
    height = 70 + panel_h + 48
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.text(
        (24, 16),
        f"Horizon FOV — {card.team} {card.season} · cut {frac:.0%} (week {cut+1})",
        fill=INK,
        font=_font(16),
    )
    draw.text(
        (24, 40),
        "Past = true. Remainder G/A blank on input; one-shot opacity on every live remainder cell.",
        fill=MUTED,
        font=_font(12),
    )
    origins = [
        (48, 88),
        (48 + panel_w + gutter, 88),
        (48 + 2 * (panel_w + gutter), 88),
    ]
    _draw_panel(draw, origins[0], card, players, "truth", None, cut, "1. Truth")
    _draw_panel(draw, origins[1], card, players, "fov", None, cut, "2. Horizon FOV input")
    _draw_panel(draw, origins[2], card, players, "paint_remainder", soft, cut, "3. Paint remainder")
    if path is None:
        path = REPORTS / f"typed_tensor_horizon_{card.team.lower()}_{card.season}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    ART.mkdir(parents=True, exist_ok=True)
    img.save(ART / path.name)
    return path
