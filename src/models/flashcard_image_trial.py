"""Three painters trained on the rendered card.

The training set is the picture: odds, venue, the opponent mark, the minute
stack, the goal squares, the assist squares, and the position color. A future
goal or assist square is gray. The painter fills that square. A square is one
cell of the picture, four pixels on a side. The stored gate stays the 21:47 UK
full painter.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.models.flashcard_card_score import tally_cards
from src.models.flashcard_generator_trial import MASKGIT_STEPS, still_masked
from src.models.flashcard_odds_trial import load_market
from src.models.flashcard_squad_trial import (
    ASSISTS_C,
    BASE_C,
    BATCH,
    COLS,
    D,
    EPOCHS,
    GOALS_C,
    HEADS,
    LR,
    MASK_C,
    ROWS,
    SEED,
    build_cards,
    build_league,
    cut_at_fraction,
    sample_cut,
)
from src.models.flashcard_square_trial import HEADER_H, PLAYER_H, SQ, compose_square
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUT = Path("/opt/cursor/artifacts")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
SCALES = (5, 10, 20, 40)
HEADER_CELLS = HEADER_H // SQ
PLAYER_CELLS = PLAYER_H // SQ
MINUTE_CELLS = 6
EVENT_SLOTS = 6
CELL_ROWS = HEADER_CELLS + ROWS * PLAYER_CELLS
BANDS = ROWS + 1
WHITE = (255, 255, 255)
_GATE = {
    ("goals", 0.2): (192, 256),
    ("assists", 0.2): (100, 161),
    ("goals", 0.4): (147, 200),
    ("assists", 0.4): (78, 119),
    ("goals", 0.6): (94, 134),
    ("assists", 0.6): (55, 84),
    ("goals", 0.8): (55, 73),
    ("assists", 0.8): (36, 51),
}
SCALE = 5
COLS_SHOWN = 16
NAME_W = 156
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

EVENT_ROW = np.array(
    [HEADER_CELLS + player * PLAYER_CELLS + MINUTE_CELLS + slot for player in range(ROWS) for slot in range(EVENT_SLOTS)],
    dtype=np.int64,
).reshape(ROWS, EVENT_SLOTS)
SLOT_COLOR = np.array([GOALS_C] * 3 + [ASSISTS_C] * 3, dtype=np.uint8)


def _solid(color: tuple[int, int, int]) -> np.ndarray:
    patch = np.empty((SQ, SQ, 3), dtype=np.uint8)
    patch[:] = color
    return patch


def _empty(baseline: bool) -> np.ndarray:
    patch = np.full((SQ, SQ, 3), 255, dtype=np.uint8)
    if baseline:
        patch[-1, :, :] = BASE_C
    return patch


MASK_PATCH = _solid(MASK_C)
INK_PATCH = [_solid(tuple(int(channel) for channel in color)) for color in SLOT_COLOR]
EMPTY_PATCH = [_empty(slot in (2, 5)) for slot in range(EVENT_SLOTS)]


def cells_of(image: np.ndarray) -> np.ndarray:
    """Split a rendered card into its squares. Shape is cells, columns, 4, 4, 3."""
    height, width, _ = image.shape
    if height != CELL_ROWS * SQ or width != COLS * SQ:
        raise ValueError(f"card image is {height} by {width}")
    folded = image.reshape(CELL_ROWS, SQ, COLS, SQ, 3)
    return np.ascontiguousarray(folded.transpose(0, 2, 1, 3, 4))


def image_of(cells: np.ndarray) -> np.ndarray:
    """Put the squares back into a picture."""
    folded = cells.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(folded.reshape(CELL_ROWS * SQ, COLS * SQ, 3))


def render_cells(card, odds: dict) -> np.ndarray:
    """The finished card. Every played square keeps its goals and assists."""
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
    return cells_of(compose_square(view, card, odds))


def ink_of(cells: np.ndarray, present: np.ndarray) -> np.ndarray:
    """True where an event square is filled with that band's color."""
    single = cells.ndim == 5
    if single:
        cells = cells[None]
        present = present[None]
    ink = np.zeros((cells.shape[0], ROWS, EVENT_SLOTS, COLS), dtype=bool)
    values = cells.astype(np.int16)
    for slot in range(EVENT_SLOTS):
        mean = values[:, EVENT_ROW[:, slot]].mean(axis=(3, 4))
        color = SLOT_COLOR[slot].astype(np.int16)
        dist_ink = np.abs(mean - color).sum(axis=-1)
        dist_white = np.abs(mean - 255).sum(axis=-1)
        dist_blank = np.abs(mean - np.array(MASK_C, dtype=np.int16)).sum(axis=-1)
        ink[:, :, slot] = (dist_ink < dist_white) & (dist_ink < dist_blank) & present
    return ink[0] if single else ink


def _write(images: np.ndarray, where: np.ndarray, patches: list[np.ndarray]) -> None:
    for slot in range(EVENT_SLOTS):
        rows = EVENT_ROW[:, slot]
        mask = where[:, :, slot, :, None, None, None]
        current = images[:, rows]
        images[:, rows] = np.where(mask, patches[slot], current)


def apply_blank(images: np.ndarray, where: np.ndarray) -> None:
    _write(images, where, [MASK_PATCH] * EVENT_SLOTS)


def apply_bits(images: np.ndarray, bits: np.ndarray, where: np.ndarray) -> None:
    """Paint event squares. A set bit fills the band color. A clear bit empties it."""
    _write(images, where & bits, INK_PATCH)
    _write(images, where & ~bits, EMPTY_PATCH)


def hidden_events(cards: list, cuts: list[int]) -> np.ndarray:
    """Future squares of players who are on the card for that fixture."""
    out = np.zeros((len(cards), ROWS, EVENT_SLOTS, COLS), dtype=bool)
    for index, (card, cut) in enumerate(zip(cards, cuts, strict=True)):
        cols = (card.gw > cut) & (card.gw > 0)
        chosen = card.present[:, cols]
        # A boolean index on the last axis comes out in front of the row axis.
        out[index, :, :, cols] = np.broadcast_to(chosen.T[:, :, None], (int(cols.sum()), ROWS, EVENT_SLOTS))
    return out


def pool_max(bits: np.ndarray, side: int) -> np.ndarray:
    block = ROWS // side
    folded = bits.reshape(bits.shape[0], side, block, EVENT_SLOTS, side, block)
    return folded.max(axis=2).max(axis=4)


def repeat_blocks(bits: np.ndarray, side: int) -> np.ndarray:
    block = ROWS // side
    return np.repeat(np.repeat(bits, block, axis=1), block, axis=3)


def band_picture(ink: np.ndarray, side: int) -> np.ndarray:
    """A coarser card. Only the full card carries the second and third squares."""
    pooled = pool_max(ink, side)
    if side < COLS:
        pooled = np.array(pooled, copy=True)
        for slot in (0, 1, 3, 4):
            pooled[:, :, slot] = False
    return repeat_blocks(pooled, side)


def patches_of(cells: np.ndarray) -> torch.Tensor:
    flat = cells.reshape(cells.shape[0], CELL_ROWS, COLS, SQ * SQ * 3)
    ordered = np.transpose(flat, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(ordered)).float().div_(255)


class CellPainter(nn.Module):
    """Read every square of the card and paint the six event squares of each row.

    A square keeps its place in the column. The vertical window sees the
    squares above and below it, so the bottom of a band is a different place
    from the top, and the minute stack sits above the goal squares. Column
    attention then looks across gameweeks.
    """

    def __init__(self, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.embed = nn.Conv2d(SQ * SQ * 3, D, kernel_size=1)
        self.vertical = nn.Conv2d(D, D, kernel_size=(PLAYER_CELLS, 1), padding=(PLAYER_CELLS // 2, 0))
        self.band = nn.MultiheadAttention(D, HEADS, dropout=0.1, batch_first=True)
        self.column = nn.MultiheadAttention(D, HEADS, dropout=0.1, batch_first=True)
        self.norm_band = nn.LayerNorm(D)
        self.norm_column = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)

    def _column_mix(self, local: torch.Tensor) -> torch.Tensor:
        batch, width, rows, columns = local.shape
        sequence = local.permute(0, 2, 3, 1).reshape(batch * rows, columns, width)
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(torch.ones(columns, columns, dtype=torch.bool, device=local.device), diagonal=1)
        mixed, _ = self.column(sequence, sequence, sequence, attn_mask=attn_mask, need_weights=False)
        sequence = self.norm_column(sequence + mixed)
        return sequence.reshape(batch, rows, columns, width)

    def _band_context(self, local: torch.Tensor) -> torch.Tensor:
        batch, _, columns, width = local.shape
        header = local[:, :HEADER_CELLS].mean(dim=1)
        body = local[:, HEADER_CELLS:].reshape(batch, ROWS, PLAYER_CELLS, columns, width)
        bands = torch.cat([header.unsqueeze(1), body.mean(dim=2)], dim=1)
        sequence = bands.permute(0, 2, 1, 3).reshape(batch * columns, BANDS, width)
        mixed, _ = self.band(sequence, sequence, sequence, need_weights=False)
        sequence = self.norm_band(sequence + mixed)
        bands = sequence.reshape(batch, columns, BANDS, width).permute(0, 2, 1, 3)
        context = torch.zeros_like(local)
        context[:, :HEADER_CELLS] = bands[:, 0].unsqueeze(1)
        spread = bands[:, 1:, None].expand(-1, -1, PLAYER_CELLS, -1, -1)
        context[:, HEADER_CELLS:] = spread.reshape(batch, ROWS * PLAYER_CELLS, columns, width)
        return context

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        local = F.gelu(self.vertical(self.embed(patches)))
        local = self._column_mix(local)
        logits = self.head(local + self._band_context(local)).squeeze(-1)
        body = logits[:, HEADER_CELLS:].reshape(logits.shape[0], ROWS, PLAYER_CELLS, logits.shape[-1])
        return body[:, :, MINUTE_CELLS : MINUTE_CELLS + EVENT_SLOTS]


def _bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per * mask).sum() / mask.sum().clamp_min(1)


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


def _batch(cells: np.ndarray, ink: np.ndarray, cards: list, indexes: np.ndarray, cuts: list[int], device: torch.device):
    images = np.stack([cells[int(i)] for i in indexes])
    target = np.stack([ink[int(i)] for i in indexes])
    hidden = hidden_events([cards[int(i)] for i in indexes], cuts)
    return images, torch.from_numpy(target).to(device), torch.from_numpy(hidden).to(device)


def train_maskgit(model: CellPainter, cards: list, cells: np.ndarray, ink: np.ndarray, rng: np.random.Generator, device: torch.device) -> None:
    opt = _optimizer(model)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            indexes = order[start : start + BATCH]
            cuts = [sample_cut(cards[int(i)], rng) for i in indexes]
            images, target, hidden = _batch(cells, ink, cards, indexes, cuts, device)
            blank, drop_np = mask_from_bottom(hidden.cpu().numpy(), rng)
            apply_blank(images, blank)
            opt.zero_grad()
            logits = model(patches_of(images).to(device))
            mask = torch.from_numpy(drop_np).to(device)
            loss = _bce(logits, target.float(), mask.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"maskgit epoch {epoch + 1} train bce {total / max(seen, 1):.3f}", flush=True)


def train_column(model: CellPainter, cards: list, cells: np.ndarray, ink: np.ndarray, rng: np.random.Generator, device: torch.device) -> None:
    opt = _optimizer(model)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            indexes = order[start : start + BATCH]
            cuts = [sample_cut(cards[int(i)], rng) for i in indexes]
            images, target, hidden = _batch(cells, ink, cards, indexes, cuts, device)
            blank, drop_np = mask_from_bottom(hidden.cpu().numpy(), rng)
            apply_blank(images, blank)
            opt.zero_grad()
            loss = _bce(model(patches_of(images).to(device)), target.float(), torch.from_numpy(drop_np).to(device).float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"column epoch {epoch + 1} train bce {total / max(seen, 1):.3f}", flush=True)


def train_var(model: CellPainter, cards: list, cells: np.ndarray, ink: np.ndarray, rng: np.random.Generator, device: torch.device) -> None:
    opt = _optimizer(model)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            indexes = order[start : start + BATCH]
            cuts = [sample_cut(cards[int(i)], rng) for i in indexes]
            images, _, hidden = _batch(cells, ink, cards, indexes, cuts, device)
            side = int(SCALES[int(rng.integers(len(SCALES)))])
            picked = np.stack([ink[int(i)] for i in indexes])
            hidden_np = hidden.cpu().numpy()
            blank, drop = mask_from_bottom(hidden_np, rng)
            if side != SCALES[0]:
                apply_bits(images, band_picture(picked, side // 2), hidden_np)
            apply_blank(images, blank)
            opt.zero_grad()
            logits = model(patches_of(images).to(device))
            target = torch.from_numpy(np.ascontiguousarray(picked)).to(device).float()
            loss = _bce(logits, target, torch.from_numpy(drop).to(device).float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"var epoch {epoch + 1} train bce {total / max(seen, 1):.3f}", flush=True)


BAND_LEVELS = ((2, 5), (1, 4), (0, 3))


def mask_from_bottom(hidden: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Hide a suffix of each band. The loss is the next square up.

    The squares below that one stay as they were painted. A second square is
    never a training target while the first square is hidden.
    """
    blank = np.zeros_like(hidden)
    drop = np.zeros_like(hidden)
    for bottom, middle, top in ((2, 1, 0), (5, 4, 3)):
        base = hidden[:, :, bottom]
        keep = rng.integers(0, 3, size=base.shape)
        blank[:, :, bottom] = base & (keep == 0)
        blank[:, :, middle] = base & (keep <= 1)
        blank[:, :, top] = base & (keep <= 2)
        drop[:, :, bottom] = base & (keep == 0)
        drop[:, :, middle] = base & (keep == 1)
        drop[:, :, top] = base & (keep == 2)
    return blank, drop


def stack_faults(ink: np.ndarray) -> dict[str, int]:
    """Stacks with a higher square filled and a lower square empty."""

    def part(slots: np.ndarray) -> tuple[int, int]:
        top, mid, bottom = slots[:, :, 0], slots[:, :, 1], slots[:, :, 2]
        bad = (top & ~mid) | (top & ~bottom) | (mid & ~bottom)
        return int(bad.sum()), int((top | mid | bottom).sum())

    goal_bad, goal_filled = part(ink[:, :, :3])
    assist_bad, assist_filled = part(ink[:, :, 3:])
    return {
        "goal_bad": goal_bad,
        "goal_filled": goal_filled,
        "assist_bad": assist_bad,
        "assist_filled": assist_filled,
    }


def _counts(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    goals = ink[:, :, :3].sum(axis=2).astype(np.int64)
    assists = ink[:, :, 3:].sum(axis=2).astype(np.int64)
    return goals, assists


def _sample_cells(logits: torch.Tensor, generator: torch.Generator) -> np.ndarray:
    probs = torch.sigmoid(logits)
    draw = torch.rand(probs.shape, generator=generator, device=probs.device) < probs
    return draw.cpu().numpy()


def _block_mean(logits: torch.Tensor, hidden: torch.Tensor, side: int) -> torch.Tensor:
    block = ROWS // side
    values = torch.where(hidden, logits, torch.zeros_like(logits))
    grouped = values.reshape(values.shape[0], side, block, EVENT_SLOTS, side, block)
    weights = hidden.reshape(hidden.shape[0], side, block, EVENT_SLOTS, side, block).float()
    return grouped.sum(dim=(2, 5)) / weights.sum(dim=(2, 5)).clamp_min(1)


@torch.no_grad()
def paint_maskgit(
    model: CellPainter,
    cards: list,
    cells: np.ndarray,
    cuts: list[int],
    device: torch.device,
    generator: torch.Generator,
    keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    model.eval()
    images = np.stack([cells[i] for i in range(len(cards))])
    hidden = hidden_events(cards, cuts)
    apply_blank(images, hidden)
    for level in BAND_LEVELS:
        active = np.zeros_like(hidden)
        for slot in level:
            active[:, :, slot] = hidden[:, :, slot]
        original = active.reshape(len(cards), -1).sum(axis=1)
        for step in range(MASKGIT_STEPS):
            if not bool(active.any()):
                break
            logits = model(patches_of(images).to(device))
            probs = torch.sigmoid(logits)
            draw = (torch.rand(probs.shape, generator=generator, device=probs.device) < probs).cpu().numpy()
            confidence = torch.where(probs > 0.5, probs, 1 - probs).cpu().numpy()
            commit = np.zeros_like(active)
            for index in range(len(cards)):
                nxt = still_masked(
                    confidence[index].ravel(),
                    active[index].ravel(),
                    step,
                    MASKGIT_STEPS,
                    int(original[index]),
                )
                nxt = nxt.reshape(active[index].shape)
                commit[index] = active[index] & ~nxt
                active[index] = nxt
            apply_bits(images, draw, commit)
    ink = ink_of(images, hidden[:, :, 0, :])
    goals, assists = _counts(ink)
    kept = images[keep] if keep is not None else None
    return goals, assists, kept, stack_faults(ink)


@torch.no_grad()
def paint_column(
    model: CellPainter,
    cards: list,
    cells: np.ndarray,
    cuts: list[int],
    device: torch.device,
    generator: torch.Generator,
    keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    model.eval()
    images = np.stack([cells[i] for i in range(len(cards))])
    hidden = hidden_events(cards, cuts)
    apply_blank(images, hidden)
    for col in range(COLS):
        if not bool(hidden[:, :, :, col].any()):
            continue
        for level in BAND_LEVELS:
            logits = model(patches_of(images).to(device))
            draw = _sample_cells(logits, generator)
            where = np.zeros_like(hidden)
            for slot in level:
                where[:, :, slot, col] = hidden[:, :, slot, col]
            apply_bits(images, draw, where)
    ink = ink_of(images, hidden[:, :, 0, :])
    goals, assists = _counts(ink)
    kept = images[keep] if keep is not None else None
    return goals, assists, kept, stack_faults(ink)


@torch.no_grad()
def paint_var(
    model: CellPainter,
    cards: list,
    cells: np.ndarray,
    cuts: list[int],
    device: torch.device,
    generator: torch.Generator,
    keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, int]]:
    model.eval()
    images = np.stack([cells[i] for i in range(len(cards))])
    hidden_np = hidden_events(cards, cuts)
    apply_blank(images, hidden_np)
    hidden = torch.from_numpy(hidden_np).to(device)
    for side in SCALES:
        for level in BAND_LEVELS:
            logits = model(patches_of(images).to(device))
            level_hidden = hidden.clone()
            for slot in range(EVENT_SLOTS):
                if slot not in level:
                    level_hidden[:, :, slot] = False
            probs = torch.sigmoid(_block_mean(logits, level_hidden, side))
            draw = torch.rand(probs.shape, generator=generator, device=probs.device) < probs
            block = ROWS // side
            up = draw.repeat_interleave(block, dim=1).repeat_interleave(block, dim=3).cpu().numpy()
            where = np.zeros_like(hidden_np)
            for slot in level:
                where[:, :, slot] = hidden_np[:, :, slot]
            apply_bits(images, up, where)
    ink = ink_of(images, hidden_np[:, :, 0, :])
    goals, assists = _counts(ink)
    kept = images[keep] if keep is not None else None
    return goals, assists, kept, stack_faults(ink)


@torch.no_grad()
def paint_chance(model: CellPainter, cards: list, cells: np.ndarray, cuts: list[int], device: torch.device) -> dict:
    """Chance a blank appearance gets any goal square, before the draw."""
    model.eval()
    images = np.stack([cells[i] for i in range(len(cards))])
    hidden = hidden_events(cards, cuts)
    apply_blank(images, hidden)
    probs = torch.sigmoid(model(patches_of(images).to(device))).cpu().numpy()
    any_goal = 1.0 - np.prod(1.0 - probs[:, :, :3], axis=2)
    minutes = np.stack([card.minutes for card in cards])
    present = hidden[:, :, 0, :]
    zero = present & (minutes <= 0)
    played = present & (minutes > 0)

    def mean(mask: np.ndarray) -> float:
        return float(any_goal[mask].mean()) if bool(mask.any()) else 0.0

    def slot_mean(mask: np.ndarray, slot: int) -> float:
        return float(probs[:, :, slot][mask].mean()) if bool(mask.any()) else 0.0

    out = {"zero": mean(zero), "played": mean(played)}
    for label, mask in (("zero", zero), ("played", played)):
        out[label + "_slots"] = [slot_mean(mask, slot) for slot in (2, 1, 0)]
    return out


def _row(revealed: float, name: str, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | {name} | {channel} | {totals['actual']} | {totals['painted']} | "
        f"{totals['exact']} | {totals['actual_within']} | {totals['painted_within']} |"
    )


def _compare(revealed: float, channel: str, totals: dict[str, int]) -> str:
    exact, within = _GATE[(channel, revealed)]
    return (
        f"| {revealed:.0%} | {channel} | {totals['actual']} | {totals['painted']} | {totals['exact']} | {exact} | "
        f"{totals['exact'] - exact:+d} | {totals['actual_within']} | {within} | "
        f"{totals['actual_within'] - within:+d} |"
    )


def _crop_rows(card, col_count: int) -> list[int]:
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
    for row, name in enumerate(card.names):
        if "Havertz" in name and row not in picked:
            picked.append(row)
    return sorted(picked)


def _panel(picture: np.ndarray, card, rows: list[int], col_count: int, title: str, subtitle: str, cut_cols: int) -> Image.Image:
    width = col_count * SQ
    parts = [picture[:HEADER_H, :width]]
    for row in rows:
        y = HEADER_H + row * PLAYER_H
        parts.append(picture[y : y + PLAYER_H, :width])
    crop = np.concatenate(parts, axis=0)
    zoom = Image.fromarray(crop).resize((crop.shape[1] * SCALE, crop.shape[0] * SCALE), Image.Resampling.NEAREST)
    header_h, gw_h, pad = 46, 18, 10
    sheet = Image.new("RGB", (NAME_W + zoom.width + pad, header_h + gw_h + zoom.height + pad), WHITE)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 6), title, fill=(0, 0, 0), font=ImageFont.truetype(FONT_B, 15))
    draw.text((pad, 26), subtitle, fill=(60, 60, 60), font=ImageFont.truetype(FONT, 12))
    top = header_h + gw_h
    sheet.paste(zoom, (NAME_W, top))
    label = ImageFont.truetype(FONT, 12)
    small = ImageFont.truetype(FONT, 10)
    for index, row in enumerate(rows):
        y = top + (HEADER_H + index * PLAYER_H) * SCALE + 8
        draw.text((8, y), str(card.names[row])[:24], fill=(20, 20, 20), font=label)
    for col in range(col_count):
        draw.text((NAME_W + col * SQ * SCALE + 1, header_h), str(int(card.gw[col])), fill=(60, 60, 60), font=small)
    if 0 < cut_cols < col_count:
        x = NAME_W + cut_cols * SQ * SCALE
        draw.line((x, header_h, x, sheet.height - 4), fill=(0, 0, 0), width=2)
    return sheet


def _examples(card, cells: np.ndarray, painted: dict[str, np.ndarray], cut: int) -> None:
    col_count = min(COLS_SHOWN, int(card.n_fixtures))
    rows = _crop_rows(card, col_count)
    shown_cols = int(np.sum((card.gw[:col_count] > 0) & (card.gw[:col_count] <= cut)))
    hidden_cols = np.flatnonzero((card.gw[:col_count] > cut) & (card.gw[:col_count] > 0))
    shown = np.stack([cells])
    hidden = hidden_events([card], [cut])
    apply_blank(shown, hidden)
    panels = []
    order = (
        ("Shown", image_of(shown[0])),
        ("Actual", image_of(cells)),
        ("MaskGIT", image_of(painted["maskgit"])),
        ("Column", image_of(painted["column"])),
        ("VAR", image_of(painted["var"])),
    )
    for title, picture in order:
        if title == "Shown":
            subtitle = "future event squares are gray"
        else:
            ink = ink_of(cells_of(picture), np.ones((ROWS, COLS), dtype=bool))
            goals = int(ink[:, :3, hidden_cols].any(axis=1)[rows].sum()) if len(hidden_cols) else 0
            assists = int(ink[:, 3:, hidden_cols].any(axis=1)[rows].sum()) if len(hidden_cols) else 0
            subtitle = f"{goals} goal rows and {assists} assist rows in this crop"
        panel = _panel(picture, card, rows, col_count, title, subtitle, shown_cols)
        panels.append(panel)
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
    sheet.save(OUT / "image-cell-cards-arsenal.png")


def execute() -> dict:
    started = time.perf_counter()
    device = torch.device("cpu")
    print("rendering card pictures", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    odds = load_market()
    train_cells = np.stack([render_cells(card, odds) for card in train])
    test_cells = np.stack([render_cells(card, odds) for card in test])
    train_ink = ink_of(train_cells, np.stack([card.present for card in train]))
    print(f"train {len(train)} test {len(test)} pictures", flush=True)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    lines = [
        "# Image cell painters",
        "",
        "The training set is the rendered card. Each square is one cell.",
        "A future goal or assist square is gray, and the painter fills it.",
        "Odds, venue, the opponent mark, the minute stack, and the position",
        "color stay on the picture, including on a future gameweek. Each",
        "finished card is one draw at temperature 1. The other squad is not",
        "an input.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    compared: dict[str, list[str]] = {}
    chances: dict[str, dict] = {}
    faults: dict[str, list[str]] = {}
    arsenal = next(i for i, card in enumerate(test) if card.team == "Arsenal")
    kept: dict[str, np.ndarray] = {}
    runners = (
        ("maskgit", False, train_maskgit, paint_maskgit),
        ("column", True, train_column, paint_column),
        ("var", False, train_var, paint_var),
    )
    for name, causal, train_fn, paint_fn in runners:
        print(f"training {name}", flush=True)
        torch.manual_seed(SEED)
        model = CellPainter(causal=causal).to(device)
        train_fn(model, train, train_cells, train_ink, np.random.default_rng(SEED), device)
        rows: list[str] = []
        for revealed in FRACTIONS:
            cuts = [cut_at_fraction(card, revealed) for card in test]
            generator = torch.Generator(device="cpu")
            generator.manual_seed(SEED)
            keep = arsenal if revealed == 0.2 else None
            painted_g, painted_a, image, fault = paint_fn(model, test, test_cells, cuts, device, generator, keep)
            faults.setdefault(name, []).append(
                f"| {revealed:.0%} | {fault['goal_filled']} | {fault['goal_bad']} | {fault['assist_filled']} | {fault['assist_bad']} |"
            )
            if image is not None:
                kept[name] = image
            for channel, painted, actual in (("goals", painted_g, labels), ("assists", painted_a, assist_labels)):
                totals = tally_cards(test, actual, painted, 0, cuts)
                lines.append(_row(revealed, name, channel, totals))
                rows.append(_compare(revealed, channel, totals))
                print(
                    f"{name} {revealed:.0%} {channel} painted {totals['painted']} "
                    f"exact {totals['exact']} within {totals['actual_within']}",
                    flush=True,
                )
        compared[name] = rows
        cuts = [cut_at_fraction(card, 0.2) for card in test]
        chances[name] = paint_chance(model, test, test_cells, cuts, device)
        print(
            f"{name} goal-square chance, zero minutes {chances[name]['zero']:.3f}, "
            f"played {chances[name]['played']:.3f}",
            flush=True,
        )
    lines.extend(
        [
            "",
            "MaskGIT fills the gray squares in eight passes, keeping the sure",
            "ones each time. The column raster paints the next gameweek from",
            "the picture on its left, then writes those squares before the next",
            "week. A band is drawn from the bottom square upward, and the square",
            "above is drawn after the square below it is on the card. VAR paints",
            "scales of 5, 10, 20, and 40. A coarse block carries the bottom",
            "square only, and the full scale paints the squares above it. The",
            "stored gate is the 21:47 UK full painter.",
            "These runs do not replace it.",
            "",
        ]
    )
    for name in ("maskgit", "column", "var"):
        lines.extend(
            [
                f"## {name}",
                "",
                "| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                *compared[name],
                "",
            ]
        )
    lines.extend(
        [
            "A square keeps its row in the column. The bottom square of a band",
            "sees a different window from the square above it, and the minute",
            "stack is in that window. With the future event squares left gray,",
            "the chance of painting each goal square, from the bottom, before",
            "the draw:",
            "",
            "| model | minutes | 1st square | 2nd square | 3rd square | any square |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("maskgit", "column", "var"):
        for label, title in (("zero", "zero"), ("played", "played")):
            slots = chances[name][label + "_slots"]
            lines.append(
                f"| {name} | {title} | {slots[0]:.3f} | {slots[1]:.3f} | {slots[2]:.3f} | {chances[name][label]:.3f} |"
            )
    lines.extend(
        [
            "",
            "A painted stack skips a lower square when a higher square is filled",
            "and a square below it is empty.",
        ]
    )
    for name in ("maskgit", "column", "var"):
        lines.extend(["", f"### {name}", ""])
        lines.append("| revealed | goal stacks | skipping | assist stacks | skipping |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.extend(faults[name])
    lines.append("")
    path = ROOT_REPORTS / "flashcard_image_trial.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    cut = cut_at_fraction(test[arsenal], 0.2)
    _examples(test[arsenal], test_cells[arsenal], kept, cut)
    print(f"image painters finished in {time.perf_counter() - started:.0f}s", flush=True)
    return {"report": path, "chances": chances, "picture": OUT / "image-cell-cards-arsenal.png"}


def main() -> None:
    execute()


if __name__ == "__main__":
    main()
