"""Visual flash-card trial: a column transformer reads the standard card.

The input is the 400×75 picture agreed for the trial. Minutes, home/away, and
opponent stay visible. Goal, assist, clean-sheet, and conceded bars after the
cut are flat gray. The model reads pixels and predicts those hidden bar heights.

Train on 2022-23, 2023-24, and 2024-25. Score 2025-26 at a 50% cut.
2026-27 is not read.

This is an exploratory trial, not a gated score comparison.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from src.models.flashcard_trial import (
    CHANNELS,
    MIN_LEN,
    NCLASS,
    PLAYED_MINUTES,
    TEST_FRAC,
    TEST_SEASON,
    TRAIN_SEASONS,
    FrequencyBaseline,
    Seq,
    baseline_nll,
    build_sequences,
    choose_cut,
    load_season,
    loss_mask,
    mean_nll,
)
from src.teams import norm_team

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
PLOTS = ROOT / "data" / "plots"
REPORTS = ROOT / "reports"
ARTIFACTS = Path("/opt/cursor/artifacts")

SLOTS = 40
COL_W = 8
GUTTER = 2
PITCH = COL_W + GUTTER
BANDS = (
    ("minutes", 12),
    ("venue", 6),
    ("opponent", 6),
    ("goals", 9),
    ("assists", 9),
    ("clean_sheet", 6),
    ("conceded", 9),
)
GAP = 3
HEIGHT = sum(height for _, height in BANDS) + GAP * (len(BANDS) - 1)
WIDTH = SLOTS * PITCH
EPOCHS = 10
BATCH = 64
LR = 1e-3
HIDDEN = 64
SEED = 0
CAPTURE_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5)

BG = (255, 255, 255)
BASE = (230, 230, 230)
MASK = (176, 176, 176)
UNUSED = (214, 214, 214)
UNUSED_MARK = (170, 170, 170)
UNKNOWN_CLUB = (90, 90, 90)
INK = {
    "minutes": (0, 114, 178),
    "home": (0, 158, 115),
    "away": (230, 159, 0),
    "goals": (213, 94, 0),
    "assists": (204, 121, 167),
    "clean_sheet": (0, 0, 0),
    "conceded": (86, 180, 233),
}
CLUBS = (
    "arsenal",
    "aston villa",
    "bournemouth",
    "brentford",
    "brighton",
    "burnley",
    "chelsea",
    "crystal palace",
    "everton",
    "fulham",
    "ipswich",
    "leeds",
    "leicester",
    "liverpool",
    "luton",
    "man city",
    "man utd",
    "newcastle",
    "nottm forest",
    "sheffield utd",
    "southampton",
    "spurs",
    "sunderland",
    "west ham",
    "wolves",
)
PALETTE = (
    (178, 34, 34),
    (148, 103, 189),
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (152, 78, 163),
    (214, 39, 40),
    (255, 187, 120),
    (174, 199, 232),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
    (82, 84, 163),
    (140, 109, 49),
    (57, 59, 121),
    (132, 60, 57),
)
CLUB_COLOR = dict(zip(CLUBS, PALETTE, strict=True))
EVENT_BANDS = ("goals", "assists", "clean_sheet", "conceded")


def _band_spans() -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    top = 0
    for name, height in BANDS:
        spans[name] = (top, top + height)
        top += height + GAP
    return spans


SPANS = _band_spans()


def collapse_repeated_rows(seq: Seq) -> Seq:
    """Drop a log line that repeats the previous fixture exactly.

    A double gameweek stays, because the opponent or the minutes differ.
    """
    keep = [0]
    for index in range(1, len(seq.minutes)):
        same = (
            int(seq.gw[index]) == int(seq.gw[index - 1])
            and float(seq.minutes[index]) == float(seq.minutes[index - 1])
            and int(seq.home[index]) == int(seq.home[index - 1])
            and str(seq.opponent[index]) == str(seq.opponent[index - 1])
            and np.array_equal(seq.events[index], seq.events[index - 1])
        )
        if not same:
            keep.append(index)
    if len(keep) == len(seq.minutes):
        return seq
    chosen = np.asarray(keep)
    return replace(
        seq,
        minutes=seq.minutes[chosen],
        home=seq.home[chosen],
        team=seq.team[chosen],
        opponent=seq.opponent[chosen],
        position=seq.position[chosen],
        gw=seq.gw[chosen],
        events=seq.events[chosen],
    )


def load_cards(seasons: tuple[str, ...]) -> list[Seq]:
    """Player-seasons with repeated log lines removed. Drops short cards."""
    cards: list[Seq] = []
    for season in seasons:
        for seq in build_sequences(load_season(season), min_len=1):
            collapsed = collapse_repeated_rows(seq)
            if len(collapsed.minutes) >= MIN_LEN:
                cards.append(collapsed)
    return cards


def _fill(img: np.ndarray, x0: int, y0: int, y1: int, height: int, color: tuple[int, int, int]) -> None:
    img[y1 - 1, x0 : x0 + COL_W] = BASE
    if height > 0:
        img[y1 - height : y1, x0 : x0 + COL_W] = color


def _cover(img: np.ndarray, x0: int, y0: int, y1: int, color: tuple[int, int, int]) -> None:
    img[y0:y1, x0 : x0 + COL_W] = color


def render_full(seq: Seq, events: np.ndarray | None = None) -> np.ndarray:
    """Paint every real fixture. No gray mask. Shape (75, 400, 3)."""
    bars = seq.events if events is None else events
    img = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    length = len(seq.minutes)
    if length > SLOTS:
        raise ValueError(f"{seq.name} has {length} fixtures, above the {SLOTS}-slot card")
    ys = np.arange(HEIGHT)[:, None]
    for slot in range(SLOTS):
        x0 = slot * PITCH
        if slot >= length:
            xs = np.arange(x0, x0 + COL_W)[None, :]
            mark = ((xs + ys) % 4) == 0
            img[:, x0 : x0 + COL_W] = np.where(mark[..., None], UNUSED_MARK, UNUSED)
            continue
        minutes = float(seq.minutes[slot])
        y0, y1 = SPANS["minutes"]
        band = y1 - y0
        minute_px = 0 if minutes <= 0 else min(band, max(1, int(round(minutes / 90 * band))))
        _fill(img, x0, y0, y1, minute_px, INK["minutes"])
        y0, y1 = SPANS["venue"]
        _cover(img, x0, y0, y1, INK["home"] if int(seq.home[slot]) == 1 else INK["away"])
        y0, y1 = SPANS["opponent"]
        club = CLUB_COLOR.get(norm_team(str(seq.opponent[slot])), UNKNOWN_CLUB)
        _cover(img, x0, y0, y1, club)
        clean_h = SPANS["clean_sheet"][1] - SPANS["clean_sheet"][0]
        heights = (
            int(bars[slot, 0]) * 3,
            int(bars[slot, 1]) * 3,
            clean_h if int(bars[slot, 2]) else 0,
            int(bars[slot, 3]) * 3,
        )
        for band, height in zip(EVENT_BANDS, heights, strict=True):
            y0, y1 = SPANS[band]
            _fill(img, x0, y0, y1, height, INK[band])
    return img


def mask_remainder(full: np.ndarray, cut: int, length: int) -> np.ndarray:
    """Gray out event bars on real columns from ``cut`` onward."""
    masked = full.copy()
    for slot in range(cut, min(length, SLOTS)):
        x0 = slot * PITCH
        for band in EVENT_BANDS:
            y0, y1 = SPANS[band]
            _cover(masked, x0, y0, y1, MASK)
    return masked


class ColumnVision(nn.Module):
    """Each fixture column is one visual token. Two transformer layers, four heads."""

    def __init__(self) -> None:
        super().__init__()
        pixels = 3 * HEIGHT * PITCH
        self.proj = nn.Linear(pixels, HIDDEN)
        self.pos = nn.Parameter(torch.zeros(1, SLOTS, HIDDEN))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=4,
            dim_feedforward=HIDDEN * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.heads = nn.ModuleList([nn.Linear(HIDDEN, n_class) for n_class in NCLASS])

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        """``images`` is float (batch, 3, 75, 400) in 0–1. Returns logits per column."""
        batch = images.shape[0]
        columns = images.reshape(batch, 3, HEIGHT, SLOTS, PITCH)
        columns = columns.permute(0, 3, 1, 2, 4).reshape(batch, SLOTS, -1)
        tokens = self.encoder(self.proj(columns) + self.pos)
        return [head(tokens) for head in self.heads]


def _batch_images(
    full: list[np.ndarray],
    indexes: np.ndarray,
    cuts: np.ndarray,
    lengths: np.ndarray,
) -> torch.Tensor:
    images = np.stack([mask_remainder(full[int(i)], int(cut), int(length)) for i, cut, length in zip(indexes, cuts, lengths, strict=True)])
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255.0)
    return tensor


def _pad_events(sequences: list[Seq]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    events = torch.zeros(len(sequences), SLOTS, 4, dtype=torch.long)
    minutes = torch.zeros(len(sequences), SLOTS)
    length = torch.zeros(len(sequences), dtype=torch.long)
    for row, seq in enumerate(sequences):
        width = len(seq.minutes)
        events[row, :width] = torch.from_numpy(seq.events.astype(np.int64))
        minutes[row, :width] = torch.from_numpy(seq.minutes.astype(np.float32))
        length[row] = width
    return events, minutes, length


def train_vision(
    model: ColumnVision,
    sequences: list[Seq],
    full: list[np.ndarray],
    rng: np.random.Generator,
    epochs: int = EPOCHS,
) -> list[float]:
    """Cross-entropy on hidden bars. The prefix fraction is uniform between 30% and 70%."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    events, _minutes, length = _pad_events(sequences)
    history: list[float] = []
    model.train()
    order_n = len(sequences)
    for epoch in range(epochs):
        total = 0.0
        seen = 0.0
        order = rng.permutation(order_n)
        for start in range(0, order_n, BATCH):
            pick = order[start : start + BATCH]
            fracs = rng.uniform(0.3, 0.7, size=len(pick))
            cuts = np.array(
                [
                    choose_cut(int(length[int(i)]), float(frac))
                    for i, frac in zip(pick, fracs, strict=True)
                ]
            )
            images = _batch_images(full, pick, cuts, length.numpy()[pick])
            target = events[pick]
            cut_t = torch.from_numpy(cuts.astype(np.int64))
            opt.zero_grad()
            logits = model(images)
            mask = loss_mask(length[pick], cut_t, SLOTS)
            loss, _ = mean_nll(logits, target, mask)
            (loss * len(CHANNELS)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(mask.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        epoch_nll = total / max(seen, 1.0)
        history.append(epoch_nll)
        print(f"epoch {epoch + 1} train nll {epoch_nll:.3f}", flush=True)
    return history


@torch.no_grad()
def column_probabilities(
    model: ColumnVision,
    sequences: list[Seq],
    full: list[np.ndarray],
    frac: float,
) -> list[np.ndarray]:
    """Softmax per channel. Each array is (cards, 40, n_class)."""
    model.eval()
    length = np.array([len(seq.minutes) for seq in sequences])
    cuts = np.array([choose_cut(int(n), frac) for n in length])
    stored = [np.zeros((len(sequences), SLOTS, n_class), dtype=np.float64) for n_class in NCLASS]
    for start in range(0, len(sequences), BATCH):
        pick = np.arange(start, min(start + BATCH, len(sequences)))
        images = _batch_images(full, pick, cuts[pick], length[pick])
        logits = model(images)
        for channel, scores in enumerate(logits):
            stored[channel][pick] = F.softmax(scores, dim=-1).numpy()
    return stored


def _capture(score: np.ndarray, hit: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    order = np.argsort(-(score + rng.random(len(score)) * 1e-9))
    caught = np.zeros(len(CAPTURE_FRACS))
    for index, frac in enumerate(CAPTURE_FRACS):
        take = max(1, int(round(frac * len(score))))
        caught[index] = hit[order[:take]].sum() / hit.sum()
    return caught


def goal_capture(
    sequences: list[Seq],
    vision: np.ndarray,
    baseline: FrequencyBaseline,
    frac: float,
    seeds: int = 20,
) -> dict[str, object]:
    """Share of goal weeks caught by keeping the top fraction of played weeks."""
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row, seq in enumerate(sequences):
        cut = choose_cut(len(seq.minutes), frac)
        probs = baseline.probabilities(seq, cut)
        slots = np.arange(cut, len(seq.minutes))
        played = seq.minutes[slots] >= PLAYED_MINUTES
        slots = slots[played]
        if len(slots) < 8:
            continue
        hit = seq.events[slots, 0] > 0
        if int(hit.sum()) < 1:
            continue
        vision_p = 1.0 - vision[row, slots, 0]
        base_p = 1.0 - probs[0][slots, 0]
        blocks.append((vision_p, base_p, seq.minutes[slots].astype(np.float64), hit.astype(np.float64)))
    totals = {name: np.zeros(len(CAPTURE_FRACS)) for name in ("vision", "baseline", "minutes")}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for vision_p, base_p, minutes, hit in blocks:
            totals["vision"] += _capture(vision_p, hit, rng)
            totals["baseline"] += _capture(base_p, hit, rng)
            totals["minutes"] += _capture(minutes, hit, rng)
    scale = max(len(blocks) * seeds, 1)
    return {
        "n_cards": len(blocks),
        "fracs": list(CAPTURE_FRACS),
        "vision": (totals["vision"] / scale).tolist(),
        "baseline": (totals["baseline"] / scale).tolist(),
        "minutes": (totals["minutes"] / scale).tolist(),
    }


@torch.no_grad()
def evaluate_vision(
    model: ColumnVision,
    sequences: list[Seq],
    full: list[np.ndarray],
    baseline: FrequencyBaseline,
    frac: float,
) -> dict[str, object]:
    """Played-column NLL, remainder sums, and within-card goal capture."""
    model.eval()
    probs = column_probabilities(model, sequences, full, frac)
    totals = {name: 0.0 for name in CHANNELS}
    played_totals = {name: 0.0 for name in CHANNELS}
    expected = {name: 0.0 for name in CHANNELS}
    actual = {name: 0.0 for name in CHANNELS}
    abs_err = {name: 0.0 for name in CHANNELS}
    argmax_sum = {name: 0.0 for name in CHANNELS}
    positive_p = {name: 0.0 for name in CHANNELS}
    positive_rate = {name: 0.0 for name in CHANNELS}
    n_all = 0
    n_played = 0
    for row, seq in enumerate(sequences):
        cut = choose_cut(len(seq.minutes), frac)
        slots = np.arange(cut, len(seq.minutes))
        played = seq.minutes[slots] >= PLAYED_MINUTES
        n_all += int(len(slots))
        n_played += int(played.sum())
        for channel, name in enumerate(CHANNELS):
            labels = seq.events[slots, channel]
            p = probs[channel][row, slots]
            pick = p[np.arange(len(labels)), labels]
            totals[name] += float(-np.log(np.clip(pick, 1e-12, 1.0)).sum())
            if played.any():
                played_totals[name] += float(-np.log(np.clip(pick[played], 1e-12, 1.0)).sum())
                positive_p[name] += float((1.0 - p[played, 0]).sum())
                positive_rate[name] += float((labels[played] > 0).sum())
            height = np.arange(NCLASS[channel])
            pred_sum = float((p * height).sum())
            true_sum = float(labels.sum())
            expected[name] += pred_sum
            actual[name] += true_sum
            abs_err[name] += abs(pred_sum - true_sum)
            argmax_sum[name] += float(p.argmax(axis=1).sum())
    n_cards = max(len(sequences), 1)
    return {
        "nll_all": {name: totals[name] / max(n_all, 1) for name in CHANNELS},
        "nll_played": {name: played_totals[name] / max(n_played, 1) for name in CHANNELS},
        "mean_remainder_sum": {name: actual[name] / n_cards for name in CHANNELS},
        "mean_expected_sum": {name: expected[name] / n_cards for name in CHANNELS},
        "mean_argmax_sum": {name: argmax_sum[name] / n_cards for name in CHANNELS},
        "mae_sum": {name: abs_err[name] / n_cards for name in CHANNELS},
        "played_positive_p": {name: positive_p[name] / max(n_played, 1) for name in CHANNELS},
        "played_positive_rate": {name: positive_rate[name] / max(n_played, 1) for name in CHANNELS},
        "n_columns": n_all,
        "n_played": n_played,
        "capture": goal_capture(sequences, probs[0], baseline, frac),
        "goal_probs": probs[0],
    }


def _example_card(
    sequences: list[Seq],
    full: list[np.ndarray],
    goal_probs: np.ndarray,
    frac: float,
) -> dict[str, object]:
    """Gibbs-White when he is in the set, otherwise the busiest remainder."""
    named = [i for i, seq in enumerate(sequences) if seq.name == "Morgan Gibbs-White"]
    if named:
        index = named[0]
    else:
        index = max(
            range(len(sequences)),
            key=lambda i: int(sequences[i].events[choose_cut(len(sequences[i].minutes), frac) :, 0].sum()),
        )
    seq = sequences[index]
    cut = choose_cut(len(seq.minutes), frac)
    probs = goal_probs[index, : len(seq.minutes)]
    argmax = probs.argmax(axis=1).astype(np.int64)
    completed = seq.events.copy()
    completed[cut:, 0] = argmax[cut:]
    return {
        "name": seq.name,
        "cut": cut,
        "length": len(seq.minutes),
        "goals": seq.events[:, 0].astype(int).tolist(),
        "argmax_goals": argmax.tolist(),
        "goal_p": (1.0 - probs[:, 0]).astype(float).tolist(),
        "input": mask_remainder(full[index], cut, len(seq.minutes)),
        "actual": full[index],
        "argmax_card": render_full(seq, completed),
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(path, size)


def write_card_figure(example: dict[str, object], path: Path) -> None:
    """Input, most-likely completion, and the actual card. Labels sit outside the pixels."""
    scale = 3
    labels = (
        ("Minutes 0–90", "minutes"),
        ("Home / away", "venue"),
        ("Opponent", "opponent"),
        ("Goals 0–3", "goals"),
        ("Assists 0–3", "assists"),
        ("Clean sheet", "clean_sheet"),
        ("Conceded 0–3", "conceded"),
    )
    panels = (
        ("Input. Gray cells are hidden.", example["input"]),
        ("Most likely bar in each hidden cell.", example["argmax_card"]),
        ("What happened.", example["actual"]),
    )
    font = _font(14)
    font_b = _font(16, bold=True)
    label_w = 150
    margin = 16
    card_w = WIDTH * scale
    card_h = HEIGHT * scale
    header = 72
    block = 26 + card_h
    height = header + block * len(panels) + 16
    canvas = Image.new("RGB", (margin + label_w + card_w + margin, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 12),
        f"{example['name']}  ·  2025-26  ·  cut after fixture {example['cut']}",
        fill=(20, 20, 20),
        font=font_b,
    )
    draw.text(
        (margin, 38),
        "The model reads the top picture and fills the gray cells. The middle picture is its most likely card.",
        fill=(50, 50, 50),
        font=font,
    )
    for panel_index, (title, image) in enumerate(panels):
        top = header + panel_index * block
        draw.text((margin, top), title, fill=(20, 20, 20), font=font_b)
        body = top + 24
        zoom = Image.fromarray(image).resize((card_w, card_h), Image.Resampling.NEAREST)
        for text, key in labels:
            y0, y1 = SPANS[key]
            draw.text((margin, body + int((y0 + y1) / 2 * scale) - 8), text, fill=(30, 30, 30), font=font)
        x = margin + label_w
        canvas.paste(zoom, (x, body))
        cut_x = x + int(example["cut"]) * PITCH * scale - scale
        draw.line([(cut_x, body), (cut_x, body + card_h)], fill=(0, 0, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_capture_figure(capture: dict[str, object], path: Path) -> None:
    fracs = [int(v * 100) for v in capture["fracs"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(fracs, fracs, color="#888888", marker="o", label="Chance")
    ax.plot(fracs, [v * 100 for v in capture["minutes"]], color="#4c78a8", marker="o", label="Longer minutes")
    ax.plot(fracs, [v * 100 for v in capture["baseline"]], color="#54a24b", marker="o", label="Prefix rate")
    ax.plot(fracs, [v * 100 for v in capture["vision"]], color="#e45756", marker="o", label="Vision model")
    ax.set_xlabel("Played weeks kept, preferred ones first (%)")
    ax.set_ylabel("Goal weeks caught (%)")
    ax.set_title(f"2025-26 hidden half, {capture['n_cards']} players with a goal")
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(values: dict[str, float]) -> str:
    return ", ".join(f"{name} {values[name]:.3f}" for name in CHANNELS)


def write_report(payload: dict[str, object], path: Path) -> None:
    vision = payload["vision"]
    base = payload["baseline"]
    capture = vision["capture"]
    fracs = capture["fracs"]
    lines = [
        "# Flash-card vision trial",
        "",
        "Exploratory. One held-out season. Not a gated score comparison.",
        "2026-27 was not read.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        f"Cards with at least {MIN_LEN} fixtures after dropping repeated log lines: "
        f"train {payload['n_train']}, test {payload['n_test']}.",
        f"Longest card: {payload['max_length']} fixtures. The grid holds {SLOTS}.",
        f"Test cut: first {int(TEST_FRAC * 100)}% of fixture columns visible.",
        f"Input picture: {WIDTH}×{HEIGHT}. Minutes, home/away, and opponent stay visible.",
        "Hidden event cells are flat gray. Gray is not a zero.",
        f"Wall clock: {payload['seconds']:.0f} seconds on CPU.",
        "",
        "## Fit",
        "",
        "Each of the 40 columns is one visual token, read from its own pixels.",
        "Two transformer layers, hidden 64. Four softmax heads.",
        "Cross-entropy on the hidden columns. Prefix fraction during training is uniform between 30% and 70%.",
        "Epoch mean NLL per hidden bar: " + ", ".join(f"{value:.3f}" for value in payload["history"]) + ".",
        "",
        "Baseline: the player's prefix histogram, shrunk toward a position × minutes-bin",
        "prior fit on the training seasons only. The baseline sees position. The picture does not.",
        "",
        "## Held-out negative log-likelihood",
        "",
        "Lower is better. Played columns are remainder fixtures with at least 60 minutes.",
        "",
        "| model | goals | assists | clean sheets | goals conceded |",
        "|---|---:|---:|---:|---:|",
        "| Vision, all columns | "
        + " | ".join(f"{vision['nll_all'][name]:.3f}" for name in CHANNELS)
        + " |",
        "| Vision, played | "
        + " | ".join(f"{vision['nll_played'][name]:.3f}" for name in CHANNELS)
        + " |",
        "| Baseline, all columns | "
        + " | ".join(f"{base['nll_all'][name]:.3f}" for name in CHANNELS)
        + " |",
        "| Baseline, played | "
        + " | ".join(f"{base['nll_played'][name]:.3f}" for name in CHANNELS)
        + " |",
        "",
        "## Remainder sums",
        "",
        f"Mean actual remainder sum: {_fmt(vision['mean_remainder_sum'])}.",
        f"Mean vision expected sum: {_fmt(vision['mean_expected_sum'])}.",
        f"Mean vision argmax sum: {_fmt(vision['mean_argmax_sum'])}.",
        f"Mean baseline expected sum: {_fmt(base['mean_expected_sum'])}.",
        f"MAE of the per-card remainder sum, vision: {_fmt(vision['mae_sum'])}.",
        f"MAE of the per-card remainder sum, baseline: {_fmt(base['mae_sum'])}.",
        "",
        "Played-week rate of a bar of at least 1, then the vision model's mean probability of that:",
        "",
        "| channel | actual rate | vision P |",
        "|---|---:|---:|",
    ]
    for name in CHANNELS:
        lines.append(
            f"| {name} | {vision['played_positive_rate'][name]:.3f} | {vision['played_positive_p'][name]:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Which played week gets the goal",
            "",
            f"Players with at least one goal week and eight played weeks in the remainder: {capture['n_cards']}.",
            "The vision score is the probability the goal bar is at least 1. Ties are broken at random, averaged over 20 draws.",
            "",
            "| rank by | " + " | ".join(f"top {int(v * 100)}%" for v in fracs) + " |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("vision", "baseline", "minutes"):
        lines.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in capture[name]) + " |")
    lines.extend(
        [
            "| chance | " + " | ".join(f"{v:.3f}" for v in fracs) + " |",
            "",
            "## One card",
            "",
            f"{payload['example_name']}. Cut after fixture column {payload['example_cut']}.",
            "The middle picture is the most likely height in each hidden cell, not a sample.",
            "",
            "Plot: `data/plots/flashcard_visual_card.png`.",
            "Capture curve: `data/plots/flashcard_visual_capture.png`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    print("loading cards", flush=True)
    train = load_cards(TRAIN_SEASONS)
    test = load_cards((TEST_SEASON,))
    max_length = max(len(seq.minutes) for seq in train + test)
    print(f"train {len(train)} test {len(test)} max length {max_length}", flush=True)
    print("painting cards", flush=True)
    train_full = [render_full(seq) for seq in train]
    test_full = [render_full(seq) for seq in test]
    model = ColumnVision()
    history = train_vision(model, train, train_full, rng, EPOCHS)
    print("scoring", flush=True)
    baseline = FrequencyBaseline(train)
    base_scores = baseline_nll(baseline, test, TEST_FRAC)
    vision_scores = evaluate_vision(model, test, test_full, baseline, TEST_FRAC)
    example = _example_card(test, test_full, vision_scores.pop("goal_probs"), TEST_FRAC)
    elapsed = time.perf_counter() - started
    payload = {
        "n_train": len(train),
        "n_test": len(test),
        "max_length": max_length,
        "seconds": elapsed,
        "history": history,
        "vision": vision_scores,
        "baseline": base_scores,
        "example_name": example["name"],
        "example_cut": example["cut"],
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "flashcard_visual_trial.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, REPORTS / "flashcard_visual_trial.md")
    write_card_figure(example, PLOTS / "flashcard_visual_card.png")
    write_capture_figure(vision_scores["capture"], PLOTS / "flashcard_visual_capture.png")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_card_figure(example, ARTIFACTS / "visual_trial_card.png")
    write_capture_figure(vision_scores["capture"], ARTIFACTS / "visual_trial_capture.png")
    print(f"done in {elapsed:.0f}s", flush=True)
    print(f"played goal nll vision {vision_scores['nll_played']['goals']:.3f} baseline {base_scores['nll_played']['goals']:.3f}", flush=True)
    print("capture vision", [round(v, 3) for v in vision_scores["capture"]["vision"]], flush=True)
    print("capture baseline", [round(v, 3) for v in vision_scores["capture"]["baseline"]], flush=True)


if __name__ == "__main__":
    main()
