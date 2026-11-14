"""Baudot row card with a goals-conceded strip in the GW header.

Locked gate header is club8 · opp8 · odds12. This card inserts a unary
goals-conceded strip immediately after the odds bits:

    club8 · opp8 · odds12 · gc6 · [role+id+mins+G+A] × roster

gc is the club's goals conceded that fixture (max `goals_conceded` among
players with 60+ minutes). Unary, left-filled, clipped at 6. Same-week gc
is the scoreline, so a training remainder must blank it with G/A. The
picture here is the truth card.

Does not change `HEADER_BITS` in the locked gate module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.models.flashcard_baudot_row_trial import (
    ART,
    BG,
    CLUB_BITS,
    GRID,
    HEADER_BITS,
    INK,
    MUTED,
    ODDS_BITS,
    OFF,
    ON,
    OPP_BITS,
    PLAYER_BITS,
    PLATE,
    REPORTS,
    TEST_SEASON,
    TRAIN_SEASONS,
    BaudotCard,
    _font,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    count_bits,
    fit_clubs,
    load_odds,
)

GC_BITS = 6
GC_ON = (36, 72, 140)
HEADER_WITH_GC = HEADER_BITS + GC_BITS


@dataclass
class GcCard:
    base: BaudotCard
    bits: np.ndarray  # (G, W) with gc inserted after odds
    gc: np.ndarray  # (G,) integer conceded, -1 if unknown
    width: int


def fixture_gc(league: dict, season: str, team: str, fixture_id: int) -> int | None:
    club = league.get(season, {}).get(team)
    if not club:
        return None
    info = club["by_id"].get(int(fixture_id))
    if not info or info.get("gc") is None:
        return None
    return int(info["gc"])


def with_gc_header(squad_card, baudot: BaudotCard, league: dict) -> GcCard:
    """Insert the unary gc strip after the odds bits. Player blocks shift right."""
    n_g = baudot.n_fixtures
    width = baudot.width + GC_BITS
    bits = np.zeros((baudot.bits.shape[0], width), dtype=np.float32)
    gc = np.full(baudot.bits.shape[0], -1, dtype=np.int64)
    for gw in range(n_g):
        src = baudot.bits[gw]
        bits[gw, :HEADER_BITS] = src[:HEADER_BITS]
        value = fixture_gc(league, squad_card.season, squad_card.team, int(squad_card.fixture_id[gw]))
        if value is not None:
            gc[gw] = value
            bits[gw, HEADER_BITS : HEADER_WITH_GC] = count_bits(value, GC_BITS)
        bits[gw, HEADER_WITH_GC : HEADER_WITH_GC + (baudot.width - HEADER_BITS)] = src[HEADER_BITS:]
    return GcCard(base=baudot, bits=bits, gc=gc, width=width)


def render_gc_card(card: GcCard, path: Path, show_players: int = 8) -> Path:
    """Truth card with the gc strip marked. Header zoom on the left, context on the right."""
    base = card.base
    n_g = base.n_fixtures
    show_players = min(show_players, base.n_players)
    DOT, GAP = 5, 2
    step = DOT + GAP
    # Left panel: header only (club, opp, odds, gc)
    left_w = HEADER_WITH_GC
    # Right panel: header + players
    right_w = HEADER_WITH_GC + show_players * PLAYER_BITS
    LEFT_L, LEFT_R = 118, 118
    TOP = 108
    panel_l = LEFT_L + left_w * step
    panel_r = LEFT_R + right_w * step
    gap = 36
    W = 24 + panel_l + gap + panel_r + 24
    H = TOP + n_g * step + 72
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((16, 12), f"{base.team} · {base.season} · goals conceded in the header", fill=INK, font=_font(16))
    d.text(
        (16, 36),
        f"Each row is one GW. Header is club{CLUB_BITS} · opp{OPP_BITS} · odds{ODDS_BITS} · gc{GC_BITS} (blue).",
        fill=MUTED,
        font=_font(11),
    )
    d.text(
        (16, 54),
        "gc is unary, left-filled, clipped at 6. This is the truth card — same-week gc is the scoreline.",
        fill=MUTED,
        font=_font(11),
    )
    d.text(
        (16, 72),
        "A training remainder blanks this strip with G/A. Black = other bits on. Light = off.",
        fill=MUTED,
        font=_font(11),
    )

    def paint(origin_x: int, label: str, cols: int) -> None:
        d.text((origin_x, TOP - 22), label, fill=INK, font=_font(12))
        d.rounded_rectangle(
            (origin_x - 6, TOP - 6, origin_x + cols * step + 2, TOP + n_g * step + 2),
            radius=6,
            fill=PLATE,
            outline=GRID,
            width=1,
        )
        # band behind the gc columns
        gx0 = origin_x + HEADER_BITS * step
        gx1 = origin_x + HEADER_WITH_GC * step - GAP
        d.rectangle((gx0 - 1, TOP - 4, gx1 + 1, TOP + n_g * step), fill=(226, 232, 240))
        for r in range(n_g):
            opp = (base.opponent[r] or "")[:10]
            gc_n = int(card.gc[r])
            gc_s = "—" if gc_n < 0 else str(gc_n)
            d.text((origin_x - 108, TOP + r * step - 1), f"{int(base.gw[r]):>2} {opp}", fill=MUTED, font=_font(8))
            d.text((origin_x - 28, TOP + r * step - 1), gc_s, fill=GC_ON, font=_font(8))
            for c in range(cols):
                x = origin_x + c * step
                y = TOP + r * step
                on = card.bits[r, c] >= 0.5
                if HEADER_BITS <= c < HEADER_WITH_GC:
                    fill = GC_ON if on else (236, 240, 244)
                else:
                    fill = ON if on else OFF
                d.rectangle((x, y, x + DOT - 1, y + DOT - 1), fill=fill)
        # column ticks
        ticks = [
            (0, "club"),
            (CLUB_BITS, "opp"),
            (CLUB_BITS + OPP_BITS, "odds"),
            (HEADER_BITS, "gc"),
        ]
        for c, name in ticks:
            d.text((origin_x + c * step, TOP + n_g * step + 6), name, fill=INK if name == "gc" else MUTED, font=_font(9))

    paint(16 + 108, "Header only", left_w)
    paint(16 + 108 + panel_l + gap, f"Header + first {show_players} players", right_w)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def load_pair(team: str, season: str):
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
    for s, b in zip(squad, baudot, strict=True):
        if s.team == team and s.season == season:
            return with_gc_header(s, b, league)
    raise SystemExit(f"no card for {team} {season}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="Arsenal")
    ap.add_argument("--season", default=TEST_SEASON)
    args = ap.parse_args()
    card = load_pair(args.team, args.season)
    known = card.gc[: card.base.n_fixtures]
    known = known[known >= 0]
    print(
        f"{card.base.team} {card.base.season} fixtures {card.base.n_fixtures} "
        f"gc min {int(known.min()) if known.size else '—'} "
        f"max {int(known.max()) if known.size else '—'} "
        f"mean {float(known.mean()) if known.size else float('nan'):.2f}",
        flush=True,
    )
    stem = f"baudot_gc_header_{args.team.lower().replace(' ', '_')}_{args.season}"
    out = render_gc_card(card, ART / f"{stem}.png")
    render_gc_card(card, REPORTS / f"{stem}.png")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
