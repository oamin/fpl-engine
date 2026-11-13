"""Baudot row-card + block-token mixer trial.

Final card format (locked this run):
- Each gameweek is one horizontal row of pure binary dots (black=1, light=0).
- A club-season card stacks those rows top→bottom (GW1 at the top).
- Row layout, left→right:
    club mark (8) · opponent mark (8) · odds (12) · then each squad slot:
    who = role(4) + player id · minutes (6) · goals (3) · assists (3).
- Odds are assessed-team win ON · draw BLANK · opponent win ON (no home/away row).
- Player IDs are unique by normalised name across the four closed seasons.
  Ten bits hold 1024 codes; 1670 distinct names need eleven bits, so ID_BITS=11.

The preferred model is a schema-aligned block-token mixer: slice known offsets,
mix across past gameweeks and across the squad, paint the next row's event bits
as one soft binary picture, teach with channel Soft-Dice. No μ/share/argmax.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from src.models.flashcard_odds_trial import column_odds, load_odds
from src.models.flashcard_squad_trial import (
    SquadCard,
    build_cards,
    build_league,
    fit_clubs,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS, load_season

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
ART = Path("/opt/cursor/artifacts")

# --- schema ---
CLUB_BITS = 8
OPP_BITS = 8
ODDS_BITS = 12
ROLE_BITS = 4
# 10 bits were requested; 1670 unique closed-season names need 11 for uniqueness.
ID_BITS = 11
MINS_BITS = 6
GOAL_BITS = 3
ASSIST_BITS = 3
WHO_BITS = ROLE_BITS + ID_BITS
PLAYER_BITS = WHO_BITS + MINS_BITS + GOAL_BITS + ASSIST_BITS
HEADER_BITS = CLUB_BITS + OPP_BITS + ODDS_BITS

MAX_PLAYERS = 40
MAX_GW = 40
CONTEXT = 5  # past gameweeks shown before the target row

ROLE_CODE = {
    0: (1, 1, 1, 0),  # GKP
    1: (1, 1, 0, 1),  # DEF
    2: (1, 0, 1, 1),  # MID
    3: (0, 1, 1, 1),  # FWD
}

D = 48
HEADS = 4
EPOCHS = 20
BATCH = 12
LR = 1e-3
SEED = 0
ON = (22, 22, 22)
OFF = (228, 224, 216)
BG = (235, 232, 224)
PLATE = (250, 248, 242)
GRID = (170, 165, 155)
INK = (20, 20, 20)
MUTED = (95, 95, 95)


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def split_counts(probs: np.ndarray, n: int = ODDS_BITS) -> np.ndarray:
    """Integer counts summing to n (largest remainder)."""
    raw = np.asarray(probs, dtype=np.float64)
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        return np.array([n // 3, n - 2 * (n // 3), n // 3], dtype=np.int64)
    raw = raw / raw.sum()
    exact = raw * n
    floors = np.floor(exact + 1e-9).astype(np.int64)
    left = int(n - floors.sum())
    if left > 0:
        for index in np.argsort(-(exact - floors))[:left]:
            floors[index] += 1
    return floors


def odds_bits_assessed(p_home: float, p_draw: float, p_away: float, is_home: bool) -> np.ndarray:
    """Assessed win ON · draw BLANK · opponent win ON."""
    if is_home:
        win, draw, loss = float(p_home), float(p_draw), float(p_away)
    else:
        win, draw, loss = float(p_away), float(p_draw), float(p_home)
    w, d, l = split_counts(np.array([win, draw, loss], dtype=np.float64), ODDS_BITS)
    out = np.zeros(ODDS_BITS, dtype=np.float32)
    out[:w] = 1.0
    # middle d bits stay 0
    out[w + d :] = 1.0
    return out


def int_bits(value: int, width: int) -> np.ndarray:
    value = int(value) & ((1 << width) - 1)
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.float32)


def mark_bits(index: int, width: int = CLUB_BITS) -> np.ndarray:
    """Non-zero club/opponent mark from a 1-based club index."""
    return int_bits(max(1, int(index)), width)


def minutes_bits(minutes: float) -> np.ndarray:
    n = int(np.clip(math.ceil(float(minutes) / 15.0), 0, MINS_BITS))
    out = np.zeros(MINS_BITS, dtype=np.float32)
    out[:n] = 1.0
    return out


def count_bits(count: int, width: int) -> np.ndarray:
    n = int(np.clip(count, 0, width))
    out = np.zeros(width, dtype=np.float32)
    out[:n] = 1.0
    return out


def build_name_codes(seasons: tuple[str, ...]) -> dict[str, int]:
    """Unique 1..N codes for every normalised player name. 0 = empty slot."""
    names: set[str] = set()
    for season in seasons:
        frame = load_season(season)
        names.update(frame["name"].astype(str).str.strip().str.lower())
    ordered = sorted(names)
    if len(ordered) > (1 << ID_BITS) - 1:
        raise ValueError(
            f"{len(ordered)} player names need more than {ID_BITS} id bits "
            f"(capacity {(1 << ID_BITS) - 1})"
        )
    return {name: i + 1 for i, name in enumerate(ordered)}


@dataclass
class BaudotCard:
    """One club-season as GW rows × bit columns."""

    season: str
    team: str
    names: tuple[str, ...]
    position: np.ndarray  # (P,)
    player_id: np.ndarray  # (P,) codes
    gw: np.ndarray  # (G,)
    home: np.ndarray
    opponent: tuple[str, ...]
    bits: np.ndarray  # (G, W) float 0/1
    goals: np.ndarray  # (P, G)
    assists: np.ndarray
    minutes: np.ndarray
    present: np.ndarray  # (P, G)
    n_players: int
    n_fixtures: int
    width: int


def _player_block(role: int, pid: int, minutes: float, goals: int, assists: int) -> np.ndarray:
    role_bits = np.array(ROLE_CODE.get(int(role), (0, 0, 0, 0)), dtype=np.float32)
    return np.concatenate(
        [
            role_bits,
            int_bits(pid, ID_BITS),
            minutes_bits(minutes),
            count_bits(goals, GOAL_BITS),
            count_bits(assists, ASSIST_BITS),
        ]
    )


def build_baudot_cards(
    squad_cards: list[SquadCard],
    clubs: dict[str, int],
    name_codes: dict[str, int],
    odds_table: dict,
) -> list[BaudotCard]:
    """Render SquadCard grids into Baudot row cards (GW = row)."""
    width = HEADER_BITS + MAX_PLAYERS * PLAYER_BITS
    out: list[BaudotCard] = []
    for card in squad_cards:
        club_i = clubs.get(card.team, 0)
        probs = column_odds(card, odds_table)
        n_g = min(card.n_fixtures, MAX_GW)
        bits = np.zeros((MAX_GW, width), dtype=np.float32)
        goals = np.zeros((MAX_PLAYERS, MAX_GW), dtype=np.int64)
        assists = np.zeros_like(goals)
        minutes = np.zeros((MAX_PLAYERS, MAX_GW), dtype=np.float32)
        present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
        names = list(card.names[:MAX_PLAYERS]) + ["spare"] * max(0, MAX_PLAYERS - len(card.names))
        names = tuple(names[:MAX_PLAYERS])
        position = np.full(MAX_PLAYERS, -1, dtype=np.int64)
        position[: min(MAX_PLAYERS, len(card.position))] = card.position[:MAX_PLAYERS]
        player_id = np.zeros(MAX_PLAYERS, dtype=np.int64)
        for row in range(min(card.n_players, MAX_PLAYERS)):
            key = str(names[row]).strip().lower()
            player_id[row] = int(name_codes.get(key, 0))
        gw = np.zeros(MAX_GW, dtype=np.int64)
        home = np.zeros(MAX_GW, dtype=np.int64)
        opponents = [""] * MAX_GW
        for col in range(n_g):
            gw[col] = int(card.gw[col])
            home[col] = int(card.home[col])
            opponents[col] = card.opponent[col]
            opp_i = clubs.get(card.opponent[col], 0)
            row_bits = []
            row_bits.append(mark_bits(club_i, CLUB_BITS))
            row_bits.append(mark_bits(opp_i, OPP_BITS))
            p = probs[col]
            if np.isfinite(p).all():
                row_bits.append(odds_bits_assessed(p[0], p[1], p[2], bool(home[col])))
            else:
                # unknown book: equal thirds → blank draw band in the middle
                row_bits.append(odds_bits_assessed(1 / 3, 1 / 3, 1 / 3, True))
            for row in range(MAX_PLAYERS):
                if row < card.n_players and card.present[row, col] and int(position[row]) >= 0:
                    present[row, col] = True
                    minutes[row, col] = float(card.minutes[row, col])
                    goals[row, col] = int(card.goals[row, col])
                    assists[row, col] = int(card.assists[row, col])
                    block = _player_block(
                        int(position[row]),
                        int(player_id[row]),
                        minutes[row, col],
                        goals[row, col],
                        assists[row, col],
                    )
                elif row < card.n_players and int(position[row]) >= 0 and int(player_id[row]) > 0:
                    # rostered but not present this week: who only
                    block = _player_block(int(position[row]), int(player_id[row]), 0.0, 0, 0)
                else:
                    block = np.zeros(PLAYER_BITS, dtype=np.float32)
                row_bits.append(block)
            bits[col] = np.concatenate(row_bits)[:width]
        out.append(
            BaudotCard(
                season=card.season,
                team=card.team,
                names=names,
                position=position,
                player_id=player_id,
                gw=gw,
                home=home,
                opponent=tuple(opponents),
                bits=bits,
                goals=goals,
                assists=assists,
                minutes=minutes,
                present=present,
                n_players=min(card.n_players, MAX_PLAYERS),
                n_fixtures=n_g,
                width=width,
            )
        )
    return out


def event_target(card: BaudotCard, gw_index: int) -> np.ndarray:
    """Soft binary event mask for one GW row: (P, GOAL_BITS+ASSIST_BITS)."""
    target = np.zeros((MAX_PLAYERS, GOAL_BITS + ASSIST_BITS), dtype=np.float32)
    for row in range(card.n_players):
        if not card.present[row, gw_index]:
            continue
        target[row, :GOAL_BITS] = count_bits(int(card.goals[row, gw_index]), GOAL_BITS)
        target[row, GOAL_BITS:] = count_bits(int(card.assists[row, gw_index]), ASSIST_BITS)
    return target


def blank_events_in_row(bits: np.ndarray, n_players: int) -> np.ndarray:
    """Copy a GW bit-row with G/A regions cleared for every player slot."""
    out = bits.copy()
    for row in range(MAX_PLAYERS):
        base = HEADER_BITS + row * PLAYER_BITS + WHO_BITS + MINS_BITS
        out[base : base + GOAL_BITS + ASSIST_BITS] = 0.0
    return out


def window_tensor(card: BaudotCard, target_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Past CONTEXT rows + blanked target row. Returns bits (T,W), row_on (T,), present (P,)."""
    t_len = CONTEXT + 1
    window = np.zeros((t_len, card.width), dtype=np.float32)
    row_on = np.zeros(t_len, dtype=np.float32)
    start = max(0, target_index - CONTEXT)
    past = list(range(start, target_index))
    # right-align past into the first CONTEXT slots
    offset = CONTEXT - len(past)
    for i, src in enumerate(past):
        window[offset + i] = card.bits[src]
        row_on[offset + i] = 1.0
    window[CONTEXT] = blank_events_in_row(card.bits[target_index], card.n_players)
    row_on[CONTEXT] = 1.0
    present = card.present[:, target_index].astype(bool)
    return window, row_on, present


class BlockTokenMixer(nn.Module):
    """Schema-aligned column/row encoder + short temporal/squad mix → event ink."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.width = width
        self.club = nn.Linear(CLUB_BITS, D)
        self.opp = nn.Linear(OPP_BITS, D)
        self.odds = nn.Linear(ODDS_BITS, D)
        self.who = nn.Linear(WHO_BITS, D)
        self.mins = nn.Linear(MINS_BITS, D)
        self.events = nn.Linear(GOAL_BITS + ASSIST_BITS, D)
        self.header_mix = nn.Linear(3 * D, D)
        self.player_mix = nn.Linear(3 * D, D)
        self.norm_h = nn.LayerNorm(D)
        self.norm_p = nn.LayerNorm(D)
        self.time = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.squad = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(D)
        self.norm_s = nn.LayerNorm(D)
        self.time_pos = nn.Parameter(torch.zeros(1, CONTEXT + 1, 1, D))
        nn.init.normal_(self.time_pos, std=0.02)
        self.paint = nn.Linear(D, GOAL_BITS + ASSIST_BITS)
        nn.init.zeros_(self.paint.weight)
        nn.init.constant_(self.paint.bias, -3.0)

    def _encode_rows(self, bits: torch.Tensor) -> torch.Tensor:
        """bits: B,T,W → player tokens B,T,P,D with header broadcast."""
        b, t, _ = bits.shape
        club = self.club(bits[..., :CLUB_BITS])
        opp = self.opp(bits[..., CLUB_BITS : CLUB_BITS + OPP_BITS])
        odds = self.odds(bits[..., CLUB_BITS + OPP_BITS : HEADER_BITS])
        header = self.norm_h(self.header_mix(torch.cat([club, opp, odds], dim=-1)))
        body = bits[..., HEADER_BITS:].reshape(b, t, MAX_PLAYERS, PLAYER_BITS)
        who = self.who(body[..., :WHO_BITS])
        mins = self.mins(body[..., WHO_BITS : WHO_BITS + MINS_BITS])
        ev = self.events(body[..., WHO_BITS + MINS_BITS :])
        players = self.norm_p(self.player_mix(torch.cat([who, mins, ev], dim=-1)))
        return players + header[:, :, None, :]

    def forward(self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
        """Return soft event ink B,P,GOAL+ASSIST for the target (last) row."""
        tokens = self._encode_rows(bits) + self.time_pos[:, : bits.shape[1]]
        b, t, p, d = tokens.shape
        # temporal mix per player
        time_view = tokens.permute(0, 2, 1, 3).reshape(b * p, t, d)
        time_mask = (row_on < 0.5)[:, None, :].expand(b, p, t).reshape(b * p, t)
        mixed, _ = self.time(time_view, time_view, time_view, key_padding_mask=time_mask, need_weights=False)
        time_view = self.norm_t(time_view + mixed)
        # take target timestep
        target = time_view[:, -1, :].reshape(b, p, d)
        # squad mix on target week
        squad_mask = player_pad
        mixed, _ = self.squad(target, target, target, key_padding_mask=squad_mask, need_weights=False)
        target = self.norm_s(target + mixed)
        return torch.sigmoid(self.paint(target))


def soft_dice(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Channel Soft-Dice on goal bits and assist bits."""
    live = live.to(pred.dtype)
    pred = pred * live
    target = target * live
    terms = []
    for sl in (slice(0, GOAL_BITS), slice(GOAL_BITS, GOAL_BITS + ASSIST_BITS)):
        p = pred[:, :, sl]
        y = target[:, :, sl]
        inter = (p * y).sum()
        denom = p.sum() + y.sum()
        terms.append(1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5))
    return 0.5 * (terms[0] + terms[1])


def _player_pad(card: BaudotCard) -> np.ndarray:
    pad = np.ones(MAX_PLAYERS, dtype=bool)
    pad[: card.n_players] = False
    for row in range(card.n_players):
        if int(card.position[row]) < 0 or int(card.player_id[row]) <= 0:
            pad[row] = True
    return pad


def _batch(model_cards: list[BaudotCard], indexes: list[int], device: torch.device):
    windows = []
    row_ons = []
    presents = []
    targets = []
    pads = []
    for card, idx in zip(model_cards, indexes, strict=True):
        window, row_on, present = window_tensor(card, idx)
        windows.append(window)
        row_ons.append(row_on)
        presents.append(present)
        targets.append(event_target(card, idx))
        pads.append(_player_pad(card))
    return (
        torch.from_numpy(np.stack(windows)).to(device),
        torch.from_numpy(np.stack(row_ons)).to(device),
        torch.from_numpy(np.stack(pads)).to(device),
        torch.from_numpy(np.stack(presents)).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
    )


def train_mixer(
    model: BlockTokenMixer,
    cards: list[BaudotCard],
    rng: np.random.Generator,
    device: torch.device,
) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    history: list[float] = []
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            indexes = [int(rng.integers(0, card.n_fixtures)) for card in pick]
            bits, row_on, pad, present, target = _batch(pick, indexes, device)
            live = present[:, :, None] & (~pad)[:, :, None]
            opt.zero_grad()
            pred = model(bits, row_on, pad)
            loss = soft_dice(pred, target, live)
            # tiny BCE inside event region so empty weeks still move
            bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
            bce = (bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0)
            loss = loss + 0.05 * bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        avg = total / max(seen, 1)
        history.append(avg)
        print(f"baudot mixer epoch {epoch + 1} dice+ {avg:.4f}", flush=True)
    return history


@torch.no_grad()
def evaluate_mixer(model: BlockTokenMixer, cards: list[BaudotCard], device: torch.device) -> dict:
    """Walk every GW; threshold soft ink at 0.5; tally bars."""
    model.eval()
    actual_g = actual_a = pred_g = pred_a = exact_g = exact_a = 0
    within_g = within_a = cells = 0
    gkp_pred_goals = 0
    for card in cards:
        pad = _player_pad(card)
        for idx in range(card.n_fixtures):
            bits, row_on, present = window_tensor(card, idx)
            target = event_target(card, idx)
            bt = torch.from_numpy(bits[None]).to(device)
            rt = torch.from_numpy(row_on[None]).to(device)
            pt = torch.from_numpy(pad[None]).to(device)
            pred = model(bt, rt, pt)[0].cpu().numpy()
            for row in range(card.n_players):
                if pad[row] or not present[row]:
                    continue
                cells += 1
                ag = int(target[row, :GOAL_BITS].sum())
                aa = int(target[row, GOAL_BITS:].sum())
                pg = int((pred[row, :GOAL_BITS] >= 0.5).sum())
                pa = int((pred[row, GOAL_BITS:] >= 0.5).sum())
                actual_g += ag
                actual_a += aa
                pred_g += pg
                pred_a += pa
                exact_g += int(ag == pg)
                exact_a += int(aa == pa)
                within_g += int(abs(ag - pg) <= 1)
                within_a += int(abs(aa - pa) <= 1)
                if int(card.position[row]) == 0:
                    gkp_pred_goals += pg
    return {
        "cells": cells,
        "actual_goals": actual_g,
        "painted_goals": pred_g,
        "exact_goals": exact_g,
        "within1_goals": within_g,
        "actual_assists": actual_a,
        "painted_assists": pred_a,
        "exact_assists": exact_a,
        "within1_assists": within_a,
        "gkp_painted_goals": gkp_pred_goals,
        "exact_goal_rate": exact_g / max(cells, 1),
        "exact_assist_rate": exact_a / max(cells, 1),
    }


def render_season_card(card: BaudotCard, path: Path, title: str | None = None) -> Path:
    """Full club-season Baudot card: GW rows, bits left→right."""
    n_g = card.n_fixtures
    # show header + first 16 players for readability
    show_players = min(16, card.n_players)
    show_w = HEADER_BITS + show_players * PLAYER_BITS
    DOT, GAP = 3, 1
    step = DOT + GAP
    LEFT, TOP = 48, 70
    W = LEFT + show_w * step + 20
    H = TOP + n_g * step + 36
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title = title or f"{card.team} · {card.season} · Baudot row card"
    d.text((12, 10), title, fill=INK, font=_font(16))
    d.text(
        (12, 32),
        f"Each row = one GW · club{CLUB_BITS} opp{OPP_BITS} odds{ODDS_BITS} "
        f"· player=role{ROLE_BITS}+id{ID_BITS}+mins{MINS_BITS}+G{GOAL_BITS}+A{ASSIST_BITS}",
        fill=MUTED,
        font=_font(10),
    )
    d.text(
        (12, 48),
        "Odds: assessed win ON · draw BLANK · opp win ON · black=1 light=0",
        fill=MUTED,
        font=_font(10),
    )
    d.rounded_rectangle(
        (LEFT - 6, TOP - 6, LEFT + show_w * step + 2, TOP + n_g * step + 2),
        radius=6,
        fill=PLATE,
        outline=GRID,
        width=1,
    )
    for r in range(n_g):
        for c in range(show_w):
            on = card.bits[r, c] >= 0.5
            x = LEFT + c * step
            y = TOP + r * step
            d.rectangle((x, y, x + DOT - 1, y + DOT - 1), fill=ON if on else OFF)
        d.text((8, TOP + r * step - 1), str(int(card.gw[r]) or r + 1), fill=MUTED, font=_font(7))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def render_teach_diagram(path: Path) -> Path:
    """Simple process diagram for this trial."""
    W, H = 900, 320
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 16), "Baudot row mixer — how the flashcard is taught", fill=INK, font=_font(20))
    boxes = [
        (30, 80, 200, 200, "1. Card\nGW rows of\nbinary dots"),
        (240, 80, 410, 200, "2. Slice blocks\nclub/opp/odds\nwho·mins·G/A"),
        (450, 80, 620, 200, "3. Mix\npast weeks\n+ squad"),
        (660, 80, 860, 200, "4. Paint next\nrow events\nSoft-Dice match"),
    ]
    for x0, y0, x1, y1, text in boxes:
        d.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=PLATE, outline=GRID, width=2)
        d.multiline_text((x0 + 16, y0 + 28), text, fill=INK, font=_font(14), spacing=4)
    for x in (210, 420, 630):
        d.polygon([(x, 135), (x + 20, 145), (x, 155)], fill=MUTED)
    d.text(
        (30, 230),
        "No cell shares · no argmax paint · events read from thresholded ink after.",
        fill=MUTED,
        font=_font(12),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def write_report(stats: dict, train_n: int, test_n: int, history: list[float], name_n: int) -> Path:
    path = REPORTS / "baudot_row_mixer.md"
    lines = [
        "# Baudot row-card block-token mixer",
        "",
        "Exploratory trial (not a gated score comparison).",
        "",
        "## Card format",
        "",
        f"- GW as **rows** (left→right bits).",
        f"- Header: club {CLUB_BITS} · opp {OPP_BITS} · odds {ODDS_BITS} "
        "(assessed win ON · draw BLANK · opp win ON).",
        f"- Player: role {ROLE_BITS} + id **{ID_BITS}** + mins {MINS_BITS} + G {GOAL_BITS} + A {ASSIST_BITS}.",
        f"- Player IDs: unique by normalised name across closed seasons "
        f"({name_n} names; 10 bits are not enough, so ID_BITS={ID_BITS}).",
        "",
        "## Model",
        "",
        "Block-token mixer: schema slices → temporal attention over past "
        f"{CONTEXT} GWs → squad attention on the target week → soft event ink. "
        "Loss = channel Soft-Dice + 0.05 BCE. No μ/share/argmax.",
        "",
        f"Train seasons: {', '.join(TRAIN_SEASONS)} ({train_n} club cards). "
        f"Test: {TEST_SEASON} ({test_n} cards).",
        "",
        "## Holdout tallies (threshold 0.5)",
        "",
        "| channel | actual bars | painted bars | exact row-matches | within 1 | exact rate |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| goals | {stats['actual_goals']} | {stats['painted_goals']} | "
            f"{stats['exact_goals']} | {stats['within1_goals']} | {stats['exact_goal_rate']:.3f} |"
        ),
        (
            f"| assists | {stats['actual_assists']} | {stats['painted_assists']} | "
            f"{stats['exact_assists']} | {stats['within1_assists']} | {stats['exact_assist_rate']:.3f} |"
        ),
        "",
        f"GKP painted goals (should stay near 0): **{stats['gkp_painted_goals']}**.",
        "",
        f"Final train loss: {history[-1]:.4f}." if history else "",
        "",
        "Artifacts: `/opt/cursor/artifacts/baudot_season_arsenal.png`, "
        "`baudot_teach_diagram.png`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    print("loading league + odds + name codes", flush=True)
    league = build_league(seasons)
    odds_table = load_odds()
    name_codes = build_name_codes(seasons)
    print(f"unique player names {len(name_codes)} id_bits {ID_BITS}", flush=True)

    squad_all = build_cards(seasons, league)
    clubs = fit_clubs([c for c in squad_all if c.season in TRAIN_SEASONS])
    # ensure every opponent/club on test still has a mark
    for card in squad_all:
        clubs.setdefault(card.team, len(clubs) + 1)
        for opp in card.opponent:
            if opp:
                clubs.setdefault(opp, len(clubs) + 1)

    baudot_all = build_baudot_cards(squad_all, clubs, name_codes, odds_table)
    train = [c for c in baudot_all if c.season in TRAIN_SEASONS]
    test = [c for c in baudot_all if c.season == TEST_SEASON]
    print(f"cards train {len(train)} test {len(test)} width {train[0].width}", flush=True)

    # season example — prefer Arsenal 2024-25
    example = next((c for c in baudot_all if c.team == "Arsenal" and c.season == "2024-25"), train[0])
    season_path = ART / "baudot_season_arsenal.png"
    render_season_card(example, season_path)
    # also copy under reports for the PR
    render_season_card(example, REPORTS / "baudot_season_arsenal.png")
    teach_path = ART / "baudot_teach_diagram.png"
    render_teach_diagram(teach_path)
    render_teach_diagram(REPORTS / "baudot_teach_diagram.png")
    print(f"wrote season card {season_path}", flush=True)

    # uniqueness audit
    assert len(name_codes) == len(set(name_codes.values()))
    assert max(name_codes.values()) < (1 << ID_BITS)
    assert min(name_codes.values()) >= 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = BlockTokenMixer(width=train[0].width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"training BlockTokenMixer params={n_params} device={device}", flush=True)
    history = train_mixer(model, train, rng, device)
    print("evaluating holdout", flush=True)
    stats = evaluate_mixer(model, test, device)
    stats["params"] = n_params
    stats["seconds"] = round(time.time() - t0, 1)
    report = write_report(stats, len(train), len(test), history, len(name_codes))
    state = {
        "trial": "baudot_row_mixer",
        "id_bits": ID_BITS,
        "unique_names": len(name_codes),
        "width": train[0].width,
        "params": n_params,
        "history": history,
        "holdout": stats,
        "report": str(report),
        "season_card": str(season_path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)
    print(f"report {report}", flush=True)


if __name__ == "__main__":
    main()
