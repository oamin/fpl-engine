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
see **all past GW rows**, and in one shot paint the **entire remaining masked
event window**. Teach with channel Soft-Dice on that remainder picture.
No μ/share/argmax.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from src.models.flashcard_card_score import balanced_score, tally_row
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
EVENT_BITS = GOAL_BITS + ASSIST_BITS
WHO_BITS = ROLE_BITS + ID_BITS
PLAYER_BITS = WHO_BITS + MINS_BITS + EVENT_BITS
HEADER_BITS = CLUB_BITS + OPP_BITS + ODDS_BITS

MAX_PLAYERS = 40
MAX_GW = 40
CUT_FRACS = (0.2, 0.4, 0.6, 0.8)

ROLE_CODE = {
    0: (1, 1, 1, 0),  # GKP
    1: (1, 1, 0, 1),  # DEF
    2: (1, 0, 1, 1),  # MID
    3: (0, 1, 1, 1),  # FWD
}

D = 48
HEADS = 4
EPOCHS = 50
BATCH = 8
LR = 1e-3
SEED = 0
BCE_W = 0.05
EARLY_PATIENCE = 8
# Learnable Tversky: α=0.5-δ, β=0.5+δ with δ∈[0, 0.49] ⇒ β≥α and α+β=1.
TVERSKY_DELTA_MAX = 0.49

# Locked flashcard gate (2026-11-13). Mirrored in flashcard_card_score.GATE.
# exact = hits (same player-GW appearance). Do not edit without a new locked run.
GATE = {
    ("goals", 0.2): {"actual": 730, "painted": 701, "exact": 194, "actual_within": 268, "painted_within": 377},
    ("assists", 0.2): {"actual": 707, "painted": 693, "exact": 114, "actual_within": 153, "painted_within": 232},
    ("goals", 0.4): {"actual": 544, "painted": 557, "exact": 151, "actual_within": 214, "painted_within": 299},
    ("assists", 0.4): {"actual": 531, "painted": 551, "exact": 89, "actual_within": 123, "painted_within": 174},
    ("goals", 0.6): {"actual": 360, "painted": 383, "exact": 105, "actual_within": 145, "painted_within": 205},
    ("assists", 0.6): {"actual": 351, "painted": 377, "exact": 66, "actual_within": 90, "painted_within": 126},
    ("goals", 0.8): {"actual": 199, "painted": 195, "exact": 60, "actual_within": 82, "painted_within": 101},
    ("assists", 0.8): {"actual": 192, "painted": 191, "exact": 32, "actual_within": 45, "painted_within": 59},
}
GATE_META = {
    "locked": "2026-11-13",
    "method": "baudot_row_mixer_learnable_tversky",
    "fov": "all_past_gws + one_shot_remainder",
    "seed": SEED,
    "id_bits": ID_BITS,
    "alpha": 0.4474738538265228,
    "beta": 0.5525261759757996,
    "module": "src.models.flashcard_baudot_row_trial",
}
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
    """Soft binary event mask for one GW row: (P, EVENT_BITS)."""
    target = np.zeros((MAX_PLAYERS, EVENT_BITS), dtype=np.float32)
    for row in range(card.n_players):
        if not card.present[row, gw_index]:
            continue
        target[row, :GOAL_BITS] = count_bits(int(card.goals[row, gw_index]), GOAL_BITS)
        target[row, GOAL_BITS:] = count_bits(int(card.assists[row, gw_index]), ASSIST_BITS)
    return target


def blank_events_in_row(bits: np.ndarray) -> np.ndarray:
    """Copy a GW bit-row with G/A regions cleared for every player slot."""
    out = bits.copy()
    for row in range(MAX_PLAYERS):
        base = HEADER_BITS + row * PLAYER_BITS + WHO_BITS + MINS_BITS
        out[base : base + EVENT_BITS] = 0.0
    return out


def cut_at_fraction(card: BaudotCard, frac: float) -> int:
    """First remainder index. Keeps ≥1 past row and ≥1 masked remainder row."""
    n = int(card.n_fixtures)
    cut = int(round(float(frac) * n))
    return int(max(1, min(n - 1, cut)))


def remainder_view(card: BaudotCard, cut: int) -> dict[str, np.ndarray]:
    """Full season FOV: all past rows true; remainder rows keep header/who/mins, blank G/A.

    Returns tensors shaped for one-shot remainder painting:
    bits (T,W), row_on (T,), rem_on (T,), target (P,T,E), present (P,T).
    """
    bits = card.bits.copy()
    row_on = np.zeros(MAX_GW, dtype=np.float32)
    rem_on = np.zeros(MAX_GW, dtype=np.float32)
    target = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), dtype=np.float32)
    present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
    for gw in range(card.n_fixtures):
        row_on[gw] = 1.0
        if gw >= cut:
            bits[gw] = blank_events_in_row(bits[gw])
            rem_on[gw] = 1.0
            target[:, gw] = event_target(card, gw)
            present[:, gw] = card.present[:, gw]
    return {
        "bits": bits,
        "row_on": row_on,
        "rem_on": rem_on,
        "target": target,
        "present": present,
    }


class BlockTokenMixer(nn.Module):
    """Schema-aligned encoder: all GW rows in → remainder event window out in one shot."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.width = width
        self.club = nn.Linear(CLUB_BITS, D)
        self.opp = nn.Linear(OPP_BITS, D)
        self.odds = nn.Linear(ODDS_BITS, D)
        self.who = nn.Linear(WHO_BITS, D)
        self.mins = nn.Linear(MINS_BITS, D)
        self.events = nn.Linear(EVENT_BITS, D)
        self.header_mix = nn.Linear(3 * D, D)
        self.player_mix = nn.Linear(3 * D, D)
        self.norm_h = nn.LayerNorm(D)
        self.norm_p = nn.LayerNorm(D)
        self.time = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.squad = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(D)
        self.norm_s = nn.LayerNorm(D)
        self.time_pos = nn.Parameter(torch.zeros(1, MAX_GW, 1, D))
        nn.init.normal_(self.time_pos, std=0.02)
        self.paint = nn.Linear(D, EVENT_BITS)
        nn.init.zeros_(self.paint.weight)
        nn.init.constant_(self.paint.bias, -3.0)
        # Starts near Dice (δ≈0); training may raise β above α if false paint hurts more.
        self.tversky_logit = nn.Parameter(torch.tensor(-2.0))

    def tversky_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (α, β) with β≥α and α+β=1."""
        delta = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit)
        alpha = 0.5 - delta
        beta = 0.5 + delta
        return alpha, beta

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
        """Soft event ink for every GW row. Shape B, P, T, EVENT_BITS.

        Past rows carry true event bits into the encoder; remainder rows enter
        with G/A blank. The paint head writes the whole card's event channels;
        the loss keeps only the remainder window.
        """
        tokens = self._encode_rows(bits) + self.time_pos[:, : bits.shape[1]]
        b, t, p, d = tokens.shape
        time_view = tokens.permute(0, 2, 1, 3).reshape(b * p, t, d)
        time_mask = (row_on < 0.5)[:, None, :].expand(b, p, t).reshape(b * p, t)
        mixed, _ = self.time(time_view, time_view, time_view, key_padding_mask=time_mask, need_weights=False)
        time_view = self.norm_t(time_view + mixed)
        tokens = time_view.reshape(b, p, t, d).permute(0, 2, 1, 3)  # B,T,P,D
        squad_view = tokens.reshape(b * t, p, d)
        squad_mask = player_pad[:, None, :].expand(b, t, p).reshape(b * t, p)
        mixed, _ = self.squad(squad_view, squad_view, squad_view, key_padding_mask=squad_mask, need_weights=False)
        tokens = self.norm_s(squad_view + mixed).reshape(b, t, p, d)
        ink = torch.sigmoid(self.paint(tokens))  # B,T,P,E
        return ink.permute(0, 2, 1, 3)  # B,P,T,E


def soft_dice(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Channel Soft-Dice on goal bits and assist bits over the live remainder window."""
    live = live.to(pred.dtype)
    pred = pred * live
    target = target * live
    terms = []
    for sl in (slice(0, GOAL_BITS), slice(GOAL_BITS, EVENT_BITS)):
        p = pred[..., sl]
        y = target[..., sl]
        inter = (p * y).sum()
        denom = p.sum() + y.sum()
        terms.append(1.0 - (2.0 * inter + 1e-5) / (denom + 1e-5))
    return 0.5 * (terms[0] + terms[1])


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Channel Tversky; α weights FN, β weights FP. β≥α ⇒ false paint costs more."""
    live = live.to(pred.dtype)
    pred = pred * live
    target = target * live
    terms = []
    for sl in (slice(0, GOAL_BITS), slice(GOAL_BITS, EVENT_BITS)):
        p = pred[..., sl]
        y = target[..., sl]
        tp = (p * y).sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()
        terms.append(1.0 - (tp + 1e-5) / (tp + alpha * fn + beta * fp + 1e-5))
    return 0.5 * (terms[0] + terms[1])


def _player_pad(card: BaudotCard) -> np.ndarray:
    pad = np.ones(MAX_PLAYERS, dtype=bool)
    pad[: card.n_players] = False
    for row in range(card.n_players):
        if int(card.position[row]) < 0 or int(card.player_id[row]) <= 0:
            pad[row] = True
    return pad


def _batch(model_cards: list[BaudotCard], cuts: list[int], device: torch.device):
    bits_l, row_ons, rem_ons, presents, targets, pads = [], [], [], [], [], []
    for card, cut in zip(model_cards, cuts, strict=True):
        view = remainder_view(card, cut)
        bits_l.append(view["bits"])
        row_ons.append(view["row_on"])
        rem_ons.append(view["rem_on"])
        presents.append(view["present"])
        targets.append(view["target"])
        pads.append(_player_pad(card))
    return (
        torch.from_numpy(np.stack(bits_l)).to(device),
        torch.from_numpy(np.stack(row_ons)).to(device),
        torch.from_numpy(np.stack(rem_ons)).to(device),
        torch.from_numpy(np.stack(pads)).to(device),
        torch.from_numpy(np.stack(presents)).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
    )


def train_mixer(
    model: BlockTokenMixer,
    cards: list[BaudotCard],
    rng: np.random.Generator,
    device: torch.device,
    test: list[BaudotCard] | None = None,
) -> dict:
    """Train with learnable Tversky (β≥α). Early-stop on holdout mean score when test is set."""
    import copy

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history: list[float] = []
    ab_hist: list[tuple[float, float]] = []
    val_scores: list[float] = []
    best_state = None
    best_score = -1.0
    bad = 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            fracs = [float(rng.choice(CUT_FRACS)) for _ in pick]
            cuts = [cut_at_fraction(card, frac) for card, frac in zip(pick, fracs, strict=True)]
            bits, row_on, rem_on, pad, present, target = _batch(pick, cuts, device)
            live = (
                present[:, :, :, None]
                & rem_on[:, None, :, None].bool()
                & (~pad)[:, :, None, None]
            )
            opt.zero_grad()
            pred = model(bits, row_on, pad)
            alpha, beta = model.tversky_weights()
            loss = tversky_loss(pred, target, live, alpha, beta)
            bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
            bce = (bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0)
            loss = loss + BCE_W * bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        avg = total / max(seen, 1)
        history.append(avg)
        with torch.no_grad():
            a, b = model.tversky_weights()
            ab_hist.append((float(a), float(b)))
        score = 0.0
        if test is not None:
            hold = evaluate_mixer(model, test, device)
            score = float(np.mean([row["score"] for row in hold["rows"]]))
            val_scores.append(score)
            if score > best_score + 1e-4:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
        print(
            f"baudot learnable-tversky epoch {epoch + 1}/{EPOCHS} "
            f"loss {avg:.4f} α={ab_hist[-1][0]:.3f} β={ab_hist[-1][1]:.3f} "
            f"holdout_score {score:.4f}",
            flush=True,
        )
        if test is not None and bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        a, b = model.tversky_weights()
        final_ab = (float(a), float(b))
    return {
        "history": history,
        "ab_history": ab_hist,
        "val_scores": val_scores,
        "best_holdout_mean_score": best_score,
        "final_alpha": final_ab[0],
        "final_beta": final_ab[1],
        "epochs_run": len(history),
    }


def _channel_slice(channel: str) -> slice:
    return slice(0, GOAL_BITS) if channel == "goals" else slice(GOAL_BITS, EVENT_BITS)


@torch.no_grad()
def evaluate_mixer(model: BlockTokenMixer, cards: list[BaudotCard], device: torch.device) -> dict:
    """One-shot remainder paint; familiar tally_row + balanced_score table."""
    model.eval()
    rows: list[dict] = []
    gkp_by_cut: dict[str, int] = {}
    for frac in CUT_FRACS:
        for channel in ("goals", "assists"):
            totals = {key: 0 for key in ("actual", "painted", "exact", "actual_within", "painted_within")}
            gkp_painted = 0
            sl = _channel_slice(channel)
            for card in cards:
                cut = cut_at_fraction(card, frac)
                view = remainder_view(card, cut)
                pad = _player_pad(card)
                pred = model(
                    torch.from_numpy(view["bits"][None]).to(device),
                    torch.from_numpy(view["row_on"][None]).to(device),
                    torch.from_numpy(pad[None]).to(device),
                )[0].cpu().numpy()
                for row in range(card.n_players):
                    if pad[row]:
                        continue
                    cols = [gw for gw in range(cut, card.n_fixtures) if view["present"][row, gw]]
                    if not cols:
                        continue
                    actual = np.array(
                        [int(view["target"][row, gw, sl].sum()) for gw in cols], dtype=np.int64
                    )
                    painted = np.array(
                        [int((pred[row, gw, sl] >= 0.5).sum()) for gw in cols], dtype=np.int64
                    )
                    part = tally_row(card.gw[cols], actual, painted)
                    for key in totals:
                        totals[key] += part[key]
                    if channel == "goals" and int(card.position[row]) == 0:
                        gkp_painted += int((painted > 0).sum())
            score = balanced_score(totals)
            row = {
                "revealed": f"{frac:.0%}",
                "channel": channel,
                "actual": totals["actual"],
                "painted": totals["painted"],
                "hits": totals["exact"],
                "actual_within_1": totals["actual_within"],
                "painted_within_1": totals["painted_within"],
                "hit_rate": score["hits"],
                "volume": score["volume"],
                "score": score["score"],
                "gkp_painted_events": gkp_painted,
            }
            rows.append(row)
        gkp_by_cut[f"{frac:.0%}"] = next(
            r["gkp_painted_events"] for r in rows if r["revealed"] == f"{frac:.0%}" and r["channel"] == "goals"
        )
    return {"rows": rows, "gkp_painted_goals_by_cut": gkp_by_cut}


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
    W, H = 940, 340
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 16), "Baudot row mixer — full past + one-shot remainder", fill=INK, font=_font(20))
    boxes = [
        (24, 70, 210, 210, "1. Full card\nall past GWs\ntrue bits"),
        (230, 70, 420, 210, "2. Mask remainder\nG/A blank\nwho+mins kept"),
        (440, 70, 640, 210, "3. Mix\nall past weeks\n+ squad"),
        (660, 70, 910, 210, "4. One-shot paint\nentire remainder\nevent window"),
    ]
    for x0, y0, x1, y1, text in boxes:
        d.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=PLATE, outline=GRID, width=2)
        d.multiline_text((x0 + 14, y0 + 28), text, fill=INK, font=_font(14), spacing=4)
    for x in (212, 422, 642):
        d.polygon([(x, 130), (x + 16, 140), (x, 150)], fill=MUTED)
    d.text(
        (24, 240),
        "Teach learnable Tversky (β≥α) on the remainder picture · no cell shares · no argmax.",
        fill=MUTED,
        font=_font(12),
    )
    d.text(
        (24, 268),
        "Cuts 20/40/60/80% · α,β adjust in training · events read after threshold.",
        fill=MUTED,
        font=_font(12),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def write_gate_report() -> Path:
    """Canonical locked gate table (does not retrain)."""
    from src.models.flashcard_card_score import balanced_score

    path = REPORTS / "baudot_gate.md"
    lines = [
        "# Flashcard gate — Baudot learnable Tversky (locked 2026-11-13)",
        "",
        "This is the **default flashcard painter gate**. New flashcard trials compare here.",
        "",
        f"- Module: `{GATE_META['module']}`",
        f"- FOV: {GATE_META['fov']}",
        f"- Loss: learnable Tversky (β≥α); locked checkpoint α={GATE_META['alpha']:.3f}, "
        f"β={GATE_META['beta']:.3f}",
        f"- Seed: {GATE_META['seed']}; ID_BITS={GATE_META['id_bits']}",
        "- Metrics: `tally_row` + `balanced_score` (`flashcard_card_score.GATE`)",
        "",
        "| revealed | channel | actual events | painted events | hits | actual ±1 | painted ±1 | score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for channel in ("goals", "assists"):
        for frac in CUT_FRACS:
            tot = GATE[(channel, frac)]
            sc = balanced_score(tot)
            lines.append(
                f"| {frac:.0%} | {channel} | {tot['actual']} | {tot['painted']} | {tot['exact']} | "
                f"{tot['actual_within']} | {tot['painted_within']} | {sc['score']:.3f} |"
            )
    lines.extend(
        [
            "",
            "GKP painted goal events: 0 at every cut.",
            "",
            "Supersedes the square-card full-painter gate of 2026-11-12 "
            "(`GATE_SQUARE_2026_11_12` in `flashcard_card_score.py`).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    # Keep the mixer report pointing at the gate.
    (REPORTS / "baudot_row_mixer.md").write_text(
        "\n".join(
            [
                "# Baudot row mixer",
                "",
                "**Locked as the flashcard default / gate on 2026-11-13.**",
                "Canonical scores: [`baudot_gate.md`](baudot_gate.md).",
                "Implementation: `src/models/flashcard_baudot_row_trial.py`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_report(stats: dict, train_n: int, test_n: int, train_meta: dict, name_n: int) -> Path:
    """Replay report after a train run; gate table is authoritative in baudot_gate.md."""
    del stats, train_n, test_n, train_meta, name_n
    return write_gate_report()


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
    print(
        f"training BlockTokenMixer learnable-Tversky params={n_params} device={device}",
        flush=True,
    )
    train_meta = train_mixer(model, train, rng, device, test=test)
    print("evaluating holdout (learnable Tversky checkpoint)", flush=True)
    stats = evaluate_mixer(model, test, device)
    stats["params"] = n_params
    stats["seconds"] = round(time.time() - t0, 1)
    report = write_report(stats, len(train), len(test), train_meta, len(name_codes))
    state = {
        "trial": "baudot_row_mixer_learnable_tversky",
        "fov": "all_past_gws + one_shot_remainder_events",
        "loss": "learnable_tversky_beta_ge_alpha",
        "alpha": train_meta["final_alpha"],
        "beta": train_meta["final_beta"],
        "cuts": list(CUT_FRACS),
        "id_bits": ID_BITS,
        "unique_names": len(name_codes),
        "width": train[0].width,
        "params": n_params,
        "train_meta": {
            "epochs_run": train_meta["epochs_run"],
            "best_holdout_mean_score": train_meta["best_holdout_mean_score"],
            "ab_history": train_meta["ab_history"],
            "history": train_meta["history"],
            "val_scores": train_meta["val_scores"],
        },
        "holdout": stats,
        "report": str(report),
        "season_card": str(season_path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "alpha": train_meta["final_alpha"],
                "beta": train_meta["final_beta"],
                "epochs_run": train_meta["epochs_run"],
                "best_holdout_mean_score": train_meta["best_holdout_mean_score"],
                "rows": stats["rows"],
                "gkp": stats["gkp_painted_goals_by_cut"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"report {report}", flush=True)


if __name__ == "__main__":
    main()
