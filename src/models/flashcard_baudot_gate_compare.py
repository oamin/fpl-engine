"""Render actual Baudot club-season card beside the gate's one-shot paint.

Left: full-season truth.
Right: gate FOV at a cut (default 20%) — past rows true; remainder G/A from
BlockTokenMixer thr 0.5 paint (header / who / mins kept). Cut marked.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.models.flashcard_baudot_row_trial import (
    ART,
    ASSIST_BITS,
    BG,
    EVENT_BITS,
    GOAL_BITS,
    GRID,
    HEADER_BITS,
    INK,
    MAX_PLAYERS,
    MINS_BITS,
    MUTED,
    OFF,
    ON,
    PLAYER_BITS,
    PLATE,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    WHO_BITS,
    BaudotCard,
    BlockTokenMixer,
    _font,
    _player_pad,
    blank_events_in_row,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    fit_clubs,
    load_odds,
    remainder_view,
    train_mixer,
)

# Painted remainder ink (distinct from truth black).
PAINT_ON = (36, 72, 140)
MISS = (190, 70, 60)  # truth had event, gate blank
FALSE = (180, 120, 40)  # gate painted, truth blank
CUT_LINE = (180, 40, 40)


def _dot_xy(left: int, top: int, r: int, c: int, step: int, dot: int) -> tuple[int, int, int, int]:
    x = left + c * step
    y = top + r * step
    return x, y, x + dot - 1, y + dot - 1


def paint_remainder_bits(
    model: BlockTokenMixer,
    card: BaudotCard,
    cut: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (composite bits, soft event ink P,T,E) for one-shot remainder paint."""
    view = remainder_view(card, cut)
    pad = _player_pad(card)
    model.eval()
    with torch.no_grad():
        soft = model(
            torch.from_numpy(view["bits"][None]).to(device),
            torch.from_numpy(view["row_on"][None]).to(device),
            torch.from_numpy(pad[None]).to(device),
        )[0].cpu().numpy()
    bits = card.bits.copy()
    for gw in range(cut, card.n_fixtures):
        bits[gw] = blank_events_in_row(bits[gw])
        for row in range(card.n_players):
            if pad[row] or not card.present[row, gw]:
                continue
            base = HEADER_BITS + row * PLAYER_BITS + WHO_BITS + MINS_BITS
            hard = (soft[row, gw] >= 0.5).astype(np.float32)
            bits[gw, base : base + EVENT_BITS] = hard
    return bits, soft


def render_bits_panel(
    bits: np.ndarray,
    card: BaudotCard,
    *,
    title: str,
    subtitle: str,
    cut: int | None = None,
    soft: np.ndarray | None = None,
    compare_truth: bool = False,
    show_players: int = 16,
) -> Image.Image:
    n_g = card.n_fixtures
    show_players = min(show_players, card.n_players)
    show_w = HEADER_BITS + show_players * PLAYER_BITS
    DOT, GAP = 3, 1
    step = DOT + GAP
    LEFT, TOP = 52, 78
    W = LEFT + show_w * step + 24
    H = TOP + n_g * step + 48
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((12, 10), title, fill=INK, font=_font(15))
    d.text((12, 32), subtitle, fill=MUTED, font=_font(10))
    legend = "black=truth ON · light=OFF"
    if compare_truth:
        legend = "blue=hit · amber=false paint · red=miss · light=OFF"
    d.text((12, 48), legend, fill=MUTED, font=_font(9))
    d.rounded_rectangle(
        (LEFT - 6, TOP - 6, LEFT + show_w * step + 2, TOP + n_g * step + 2),
        radius=6,
        fill=PLATE,
        outline=GRID,
        width=1,
    )
    for r in range(n_g):
        for c in range(show_w):
            box = _dot_xy(LEFT, TOP, r, c, step, DOT)
            # Default: binary from bits
            on = bits[r, c] >= 0.5
            fill = ON if on else OFF
            if compare_truth and soft is not None and cut is not None and r >= cut:
                # Only colour event channels in the remainder.
                body_c = c - HEADER_BITS
                if body_c >= 0:
                    row = body_c // PLAYER_BITS
                    off = body_c % PLAYER_BITS
                    ev0 = WHO_BITS + MINS_BITS
                    if row < card.n_players and off >= ev0:
                        e = off - ev0
                        if e < EVENT_BITS:
                            truth = card.bits[r, c] >= 0.5
                            painted = bits[r, c] >= 0.5
                            if truth and painted:
                                fill = PAINT_ON
                            elif painted and not truth:
                                fill = FALSE
                            elif truth and not painted:
                                fill = MISS
                            else:
                                fill = OFF
            d.rectangle(box, fill=fill)
        d.text((8, TOP + r * step - 1), str(int(card.gw[r]) or r + 1), fill=MUTED, font=_font(7))
    if cut is not None and 0 < cut < n_g:
        y = TOP + cut * step - 1
        d.line((LEFT - 4, y, LEFT + show_w * step + 2, y), fill=CUT_LINE, width=2)
        d.text((LEFT + show_w * step + 4, y - 6), f"cut {cut}", fill=CUT_LINE, font=_font(9))
    # player name ticks (first few)
    for i in range(min(8, show_players)):
        x = LEFT + (HEADER_BITS + i * PLAYER_BITS) * step
        name = card.names[i][:10] if i < len(card.names) else ""
        d.text((x, TOP + n_g * step + 6), name, fill=MUTED, font=_font(7))
    return img


def side_by_side(left: Image.Image, right: Image.Image, heading: str) -> Image.Image:
    gap = 24
    pad = 20
    W = left.width + right.width + gap + 2 * pad
    H = max(left.height, right.height) + 56
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((pad, 14), heading, fill=INK, font=_font(18))
    img.paste(left, (pad, 48))
    img.paste(right, (pad + left.width + gap, 48))
    return img


def render_compare(
    model: BlockTokenMixer,
    card: BaudotCard,
    device: torch.device,
    out: Path,
    frac: float = 0.2,
) -> Path:
    cut = cut_at_fraction(card, frac)
    painted_bits, soft = paint_remainder_bits(model, card, cut, device)
    left = render_bits_panel(
        card.bits,
        card,
        title=f"{card.team} · {card.season} · actual",
        subtitle="Full-season truth Baudot card (first 16 squad slots)",
        cut=None,
        compare_truth=False,
    )
    right = render_bits_panel(
        painted_bits,
        card,
        title=f"{card.team} · {card.season} · gate paint @ {frac:.0%}",
        subtitle=f"Past rows true · remainder G/A = mixer thr 0.5 · cut GW index {cut}",
        cut=cut,
        soft=soft,
        compare_truth=True,
    )
    combo = side_by_side(
        left,
        right,
        f"Baudot gate vs actual — {card.team} {card.season} (reveal {frac:.0%})",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    combo.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Arsenal")
    ap.add_argument("--season", default=TEST_SEASON)
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--skip-train", action="store_true", help="random init (debug only)")
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    print("loading cards", flush=True)
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
    card = next((c for c in baudot if c.team == args.team and c.season == args.season), None)
    if card is None:
        raise SystemExit(f"no card for {args.team} {args.season}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = BlockTokenMixer(width=train[0].width).to(device)
    if not args.skip_train:
        print(f"training gate mixer on {len(train)} cards (holdout early-stop)", flush=True)
        train_mixer(model, train, rng, device, test=test)
    else:
        print("skip-train: random init", flush=True)

    stem = f"baudot_gate_vs_actual_{args.team.lower().replace(' ', '_')}_{args.season}_{int(args.frac*100)}"
    out_art = ART / f"{stem}.png"
    out_rep = REPORTS / f"{stem}.png"
    render_compare(model, card, device, out_art, frac=args.frac)
    render_compare(model, card, device, out_rep, frac=args.frac)
    # Also a familiar 2024-25 if different
    if args.season != "2024-25":
        card_b = next((c for c in baudot if c.team == args.team and c.season == "2024-25"), None)
        if card_b is not None:
            stem_b = f"baudot_gate_vs_actual_{args.team.lower().replace(' ', '_')}_2024-25_{int(args.frac*100)}"
            render_compare(model, card_b, device, ART / f"{stem_b}.png", frac=args.frac)
            render_compare(model, card_b, device, REPORTS / f"{stem_b}.png", frac=args.frac)
            print(f"also wrote {stem_b}.png", flush=True)
    print(f"wrote {out_art} and {out_rep} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
