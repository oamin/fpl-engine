"""Draw example opacity cards from the stepped-overlap models.

Each picture is one club-season at the 20% cut: the true card, the gray the
model paints on the remainder, and truth minus that gray. The first row under
the line is the week the step was trained to paint. Later rows are the same
forward pass, with those weeks left blank in the input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models.flashcard_baudot_residual_card import blend, soft_events
from src.models.flashcard_baudot_row_trial import (
    ART,
    BG,
    GOAL_BITS,
    GRID,
    INK,
    MUTED,
    OFF,
    ON,
    PLATE,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    BlockTokenMixer,
    _font,
    _player_pad,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    event_target,
    fit_clubs,
    load_odds,
)
from src.models.flashcard_baudot_stepped_overlap import train_arm

MISS = (176, 48, 42)
FALSE = (36, 78, 168)
CLUBS = ("Arsenal", "Liverpool")


def _rot_name(text: str, size: int = 11) -> Image.Image:
    font = _font(size)
    tmp = Image.new("RGB", (140, size + 6), PLATE)
    d = ImageDraw.Draw(tmp)
    d.text((2, 0), text, fill=INK, font=font)
    return tmp.rotate(90, expand=True, fillcolor=PLATE)


def _regulars(card, n: int = 12) -> list[int]:
    pad = _player_pad(card)
    minutes = card.minutes.sum(axis=1)
    order = np.argsort(-minutes)
    picked = [int(i) for i in order if not pad[int(i)]]
    picked = picked[:n]
    return sorted(picked, key=lambda i: (int(card.position[i]), -float(minutes[i])))


def _remainder_overlap(card, soft: np.ndarray, cut: int, players: list[int]) -> float:
    inter = ink = solid = 0.0
    pad = _player_pad(card)
    for gw in range(cut, card.n_fixtures):
        target = event_target(card, gw)
        for row in players:
            if pad[row] or not card.present[row, gw]:
                continue
            for bit in (0, GOAL_BITS):
                y = float(target[row, bit])
                p = float(soft[row, gw, bit])
                inter += p * y
                ink += p
                solid += y
    denom = ink + solid
    return 1.0 if denom == 0 else (2 * inter) / denom


def render_card(card, soft: np.ndarray, cut: int, arm: str, overlap: float, path: Path) -> Path:
    n_g = card.n_fixtures
    players = _regulars(card)
    channels = (("G", 0), ("A", GOAL_BITS))
    dot, gap = 11, 3
    step = dot + gap
    cols = len(players) * len(channels)
    left, top = 44, 78
    panel_w = left + cols * step + 18
    names_h = 156
    panel_h = top + n_g * step + names_h
    gutter = 28
    width = 3 * panel_w + 2 * gutter + 32
    paste_y = 64
    height = paste_y + panel_h + 56
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.text((16, 12), f"{card.team} {card.season}  ·  {arm} step", fill=INK, font=_font(18))
    draw.text(
        (16, 38),
        f"G = at least one goal, A = at least one assist. Line is the 20% cut. "
        f"The first row under the line is the trained week. Overlap on these dots {overlap:.3f}.",
        fill=MUTED,
        font=_font(11),
    )

    titles = ("True card", "Model card  ·  opacity = probability", "Difference  ·  truth − model")
    subs = (
        "Black = event. Light = none.",
        "Darker gray = a higher probability.",
        "Red = missed event. Blue = extra ink. Light = the cards agree.",
    )

    def paint(panel, r, c, fill):
        x = left + c * step
        y = top + r * step
        ImageDraw.Draw(panel).rectangle((x, y, x + dot - 1, y + dot - 1), fill=fill)

    panels = []
    for title, sub in zip(titles, subs, strict=True):
        panel = Image.new("RGB", (panel_w, panel_h), BG)
        pen = ImageDraw.Draw(panel)
        pen.text((12, 8), title, fill=INK, font=_font(15))
        pen.text((12, 30), sub, fill=MUTED, font=_font(10))
        pen.rounded_rectangle(
            (left - 8, top - 8, left + cols * step + 4, top + n_g * step + 4),
            radius=6,
            fill=PLATE,
            outline=GRID,
        )
        for r in range(n_g):
            pen.text((6, top + r * step), str(int(card.gw[r])), fill=MUTED, font=_font(8))
        if 0 < cut < n_g:
            y = top + cut * step - 2
            pen.line((left - 6, y, left + cols * step + 2, y), fill=MISS, width=2)
        for i, row in enumerate(players):
            name = str(card.names[row]).split()[-1][:12]
            x = left + i * len(channels) * step
            panel.paste(_rot_name(name), (x, top + n_g * step + 8))
            for k, (label, _) in enumerate(channels):
                pen.text((x + k * step, top - 22), label, fill=MUTED, font=_font(9))
        panels.append(panel)

    truth_p, opacity_p, resid_p = panels
    pad = _player_pad(card)
    for r in range(n_g):
        target = event_target(card, r)
        for i, row in enumerate(players):
            for k, (_, bit) in enumerate(channels):
                c = i * len(channels) + k
                played = bool(card.present[row, r]) and not pad[row]
                y = float(target[row, bit]) if played else 0.0
                p = float(soft[row, r, bit]) if played and r >= cut else y
                paint(truth_p, r, c, ON if y >= 0.5 else OFF)
                if not played:
                    paint(opacity_p, r, c, OFF)
                    paint(resid_p, r, c, OFF)
                    continue
                if r < cut:
                    paint(opacity_p, r, c, ON if y >= 0.5 else OFF)
                    paint(resid_p, r, c, OFF)
                    continue
                paint(opacity_p, r, c, blend(OFF, ON, p))
                resid = y - p
                if resid >= 0:
                    paint(resid_p, r, c, blend(OFF, MISS, resid))
                else:
                    paint(resid_p, r, c, blend(OFF, FALSE, -resid))

    for i, panel in enumerate(panels):
        img.paste(panel, (16 + i * (panel_w + gutter), paste_y))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    league = build_league(seasons)
    odds = load_odds()
    codes = build_name_codes(seasons)
    squad = build_cards(seasons, league)
    clubs = fit_clubs([c for c in squad if c.season in TRAIN_SEASONS])
    for card in squad:
        clubs.setdefault(card.team, len(clubs) + 1)
        for opp in card.opponent:
            if opp:
                clubs.setdefault(opp, len(clubs) + 1)
    baudot = build_baudot_cards(squad, clubs, codes, odds)
    train = [c for c in baudot if c.season in TRAIN_SEASONS]
    test = [c for c in baudot if c.season == TEST_SEASON]
    return device, train, test


def main() -> None:
    device, train, test = _load()
    pictures = []
    for arm in ("row", "column"):
        torch.manual_seed(SEED)
        model = BlockTokenMixer(width=train[0].width)
        print(f"training {arm}", flush=True)
        train_arm(arm, model, train, test, device)
        for club in CLUBS:
            card = next(c for c in test if c.team == club)
            cut = cut_at_fraction(card, 0.2)
            soft = soft_events(model, card, cut, device)
            overlap = _remainder_overlap(card, soft, cut, _regulars(card))
            name = f"baudot_step_{arm}_{club.lower()}_2025-26.png"
            path = render_card(card, soft, cut, arm, overlap, ART / name)
            render_card(card, soft, cut, arm, overlap, REPORTS / name)
            pictures.append({"arm": arm, "club": club, "overlap": overlap, "path": str(path)})
            print(f"wrote {path} overlap {overlap:.3f}", flush=True)
    (ART / "baudot_stepped_cards.json").write_text(json.dumps(pictures, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
