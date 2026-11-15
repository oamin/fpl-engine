"""Draw stepped, horizon, product, and stretched product opacity cards.

Stepped paints one gameweek at a time. Horizon paints the remainder beyond the
cutoff in one shot. Both are rows. Header and minutes stay visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models.flashcard_baudot_residual_card import blend
from src.models.flashcard_baudot_stepped_overlap import _run, step_row_view
from src.models.flashcard_baudot_row_trial import (
    ART,
    BG,
    EVENT_BITS,
    GOAL_BITS,
    MAX_PLAYERS,
    GRID,
    INK,
    MUTED,
    OFF,
    ON,
    PLATE,
    REPORTS,
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
    remainder_view,
    fit_clubs,
    load_odds,
)

MISS = (176, 48, 42)
FALSE = (36, 78, 168)
CLUBS = ("Arsenal", "Liverpool")
ARMS = ("stepped", "horizon")


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


def _dot_levels(card, soft: np.ndarray, cut: int, players: list[int]) -> dict:
    solid, blank, gkp = [], [], []
    pad = _player_pad(card)
    for gw in range(cut, card.n_fixtures):
        target = event_target(card, gw)
        for row in players:
            if pad[row] or not card.present[row, gw]:
                continue
            for bit in (0, GOAL_BITS):
                y = float(target[row, bit])
                p = float(soft[row, gw, bit])
                (solid if y >= 0.5 else blank).append(p)
                if int(card.position[row]) == 0 and bit == 0:
                    gkp.append(p)
    spreads = []
    for row in players:
        if pad[row] or int(card.position[row]) < 2:
            continue
        series = [
            float(soft[row, gw, 0])
            for gw in range(cut, card.n_fixtures)
            if card.present[row, gw]
        ]
        if len(series) >= 4:
            spreads.append(float(np.std(series)))
    return {
        "on_solid": float(np.mean(solid)) if solid else 0.0,
        "on_blank": float(np.mean(blank)) if blank else 0.0,
        "gkp": float(np.mean(gkp)) if gkp else 0.0,
        "spread": float(np.mean(spreads)) if spreads else 0.0,
    }


def _caption(arm: str, levels: dict, stretch: dict | None = None) -> str:
    if arm == "stepped":
        lead = "One gameweek, stepped. Header and minutes stay. "
    elif arm == "horizon":
        lead = "One shot of the weeks beyond the cutoff. Header and minutes stay. "
    elif arm == "product":
        lead = "Product of the stepped and horizon opacities. "
    else:
        lead = (
            f"Product opacities remapped from {stretch['lo']:.3f}–{stretch['hi']:.3f} onto 0–1. "
            if stretch
            else "Contrast-stretched product. "
        )
    return (
        lead
        + f"Solid dots {levels['on_solid']:.2f} opaque, blank dots {levels['on_blank']:.2f}, "
        f"goalkeeper goals {levels['gkp']:.2f}. "
        f"Week-to-week spread {levels['spread']:.2f}."
    )


def render_card(
    card,
    soft: np.ndarray,
    cut: int,
    arm: str,
    levels: dict,
    path: Path,
    stretch: dict | None = None,
) -> Path:
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
    title = {
        "stepped": "stepped",
        "horizon": "horizon",
        "product": "stepped × horizon",
        "product_stretched": "stepped × horizon, stretched",
    }.get(arm, arm)
    draw.text((16, 12), f"{card.team} {card.season}  ·  {title}", fill=INK, font=_font(18))
    draw.text((16, 38), _caption(arm, levels, stretch), fill=MUTED, font=_font(11))

    if arm == "product":
        mid_title = "Product card  ·  opacity = stepped × horizon"
        mid_sub = "A dot stays dark only when both cards are dark."
        right_title = "Difference  ·  truth − product"
    elif arm == "product_stretched":
        mid_title = "Stretched product  ·  contrast-mapped stepped × horizon"
        mid_sub = "Darkest product week becomes black. Lightest becomes paper."
        right_title = "Difference  ·  truth − stretched"
    else:
        mid_title = "Model card  ·  opacity = probability"
        mid_sub = "Darker gray = a higher probability."
        right_title = "Difference  ·  truth − model"
    titles = ("True card", mid_title, right_title)
    subs = (
        "Black = event. Light = none.",
        mid_sub,
        "Red = missed event. Blue = extra ink. Light = the cards agree.",
    )

    def paint(panel, r, c, fill):
        x = left + c * step
        y = top + r * step
        ImageDraw.Draw(panel).rectangle((x, y, x + dot - 1, y + dot - 1), fill=fill)

    panels = []
    for title_text, sub in zip(titles, subs, strict=True):
        panel = Image.new("RGB", (panel_w, panel_h), BG)
        pen = ImageDraw.Draw(panel)
        pen.text((12, 8), title_text, fill=INK, font=_font(15))
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


def paint_arm(model, card, cut, arm: str, device) -> np.ndarray:
    pad = _player_pad(card)
    soft = np.zeros((MAX_PLAYERS, card.n_fixtures, EVENT_BITS), np.float32)
    if arm == "stepped":
        for gw in range(cut, card.n_fixtures):
            one = _run(model, step_row_view(card, gw), pad, device)
            soft[:, gw] = one[:, gw]
    else:
        soft = _run(model, remainder_view(card, cut), pad, device)[:, : card.n_fixtures]
    return soft


def stretch_product(card, prod: np.ndarray, cut: int) -> tuple[np.ndarray, dict]:
    pad = _player_pad(card)
    out = prod.copy()
    vals = []
    for gw in range(cut, card.n_fixtures):
        for p in range(card.n_players):
            if pad[p] or not card.present[p, gw]:
                continue
            vals.append(prod[p, gw])
    if not vals:
        return out, {"lo": 0.0, "hi": 1.0}
    stack = np.concatenate(vals)
    lo = float(stack.min())
    hi = float(stack.max())
    span = max(hi - lo, 1e-8)
    for gw in range(cut, card.n_fixtures):
        for p in range(card.n_players):
            if pad[p] or not card.present[p, gw]:
                out[p, gw] = 0.0
                continue
            out[p, gw] = (prod[p, gw] - lo) / span
    return out, {"lo": lo, "hi": hi}


def main() -> None:
    device, train, test = _load()
    pictures = []
    painted = {}
    models = {}
    for arm in ARMS:
        weights = ART / f"baudot_step_gap_{arm}.pt"
        model = BlockTokenMixer(width=train[0].width).to(device)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.eval()
        models[arm] = model
        for club in CLUBS:
            card = next(c for c in test if c.team == club)
            cut = cut_at_fraction(card, 0.2)
            soft = paint_arm(model, card, cut, arm, device)
            painted[(arm, club)] = soft
            levels = _dot_levels(card, soft, cut, _regulars(card))
            name = f"baudot_step_{arm}_{club.lower()}_2025-26.png"
            path = render_card(card, soft, cut, arm, levels, ART / name)
            render_card(card, soft, cut, arm, levels, REPORTS / name)
            pictures.append({"arm": arm, "club": club, **levels, "path": str(path)})
            print(f"wrote {path} {levels}", flush=True)

    for club in CLUBS:
        card = next(c for c in test if c.team == club)
        cut = cut_at_fraction(card, 0.2)
        prod = painted[("stepped", club)] * painted[("horizon", club)]
        stretched, stretch = stretch_product(card, prod, cut)
        for arm, soft, stretch_meta in (
            ("product", prod, None),
            ("product_stretched", stretched, stretch),
        ):
            levels = _dot_levels(card, soft, cut, _regulars(card))
            name = f"baudot_step_{arm}_{club.lower()}_2025-26.png"
            path = render_card(card, soft, cut, arm, levels, ART / name, stretch=stretch_meta)
            render_card(card, soft, cut, arm, levels, REPORTS / name, stretch=stretch_meta)
            pictures.append(
                {
                    "arm": arm,
                    "club": club,
                    **levels,
                    "path": str(path),
                    **({"stretch": stretch_meta} if stretch_meta else {}),
                }
            )
            print(f"wrote {path} {levels} stretch={stretch_meta}", flush=True)

        gap = float(np.mean(np.abs(painted[("stepped", club)] - painted[("horizon", club)])))
        print(f"{club} mean absolute gap stepped vs horizon {gap:.4f}", flush=True)

    (ART / "baudot_stepped_cards.json").write_text(json.dumps(pictures, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
