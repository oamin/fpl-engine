"""Stepped column as one flashcard picture.

The model sees the stepped square card — true past, and the next week with
minutes and match header kept but event bands blank. It writes the whole next
column's event strip in one pass. Teaching is the color error on that strip
as one object. No per-cell class head, no Poisson rate, no share, no argmax
paint rule. Bars are read afterwards: a square counts when its color is
closer to the goal or assist ink than to white.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from src.models.flashcard_card_score import _position_lines, tally_cards
from src.models.flashcard_odds_trial import load_odds
from src.models.flashcard_squad_trial import (
    ASSISTS_C,
    BATCH,
    CELL_W,
    COLS,
    D,
    EPOCHS,
    GOALS_C,
    HEADS,
    LR,
    MASK_ID,
    ROWS,
    SEED,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    squad_index,
)
from src.models.flashcard_square_trial import (
    EVENT_H,
    EVENT_N,
    HEADER_H,
    MINUTE_H,
    PLAYER_H,
    compose_square,
)
from src.models.flashcard_step_trial import chop_future
from src.models.flashcard_strip_trial import (
    _square_bands,
    strip_features,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUT = ROOT_REPORTS / "column_picture_learner.md"
ART = Path("/opt/cursor/artifacts")
WHITE = np.array([255, 255, 255], dtype=np.float32)
GOAL_INK = np.array(GOALS_C, dtype=np.float32)
ASSIST_INK = np.array(ASSISTS_C, dtype=np.float32)


def _step_blank(card, league, clubs, squads, target: int) -> dict:
    """True past; next week keeps minutes and header; events stay blank."""
    view = apply_cut(card, league, target - 1, clubs, squads)
    chop_future(view, card.gw, target)
    strip_features(view)
    return view


def _step_truth(card, league, clubs, squads, target: int) -> dict:
    """Same stepped FOV, with the next week's real event squares filled."""
    view = _step_blank(card, league, clubs, squads, target)
    cols = np.flatnonzero(card.gw == target)
    view["goal_token"] = np.array(view["goal_token"], copy=True)
    view["assist_token"] = np.array(view["assist_token"], copy=True)
    for col in cols:
        for row in range(ROWS):
            if card.present[row, col]:
                view["goal_token"][row, col] = int(card.goals[row, col])
                view["assist_token"][row, col] = int(card.assists[row, col])
            else:
                view["goal_token"][row, col] = MASK_ID
                view["assist_token"][row, col] = MASK_ID
    return view


def _column_index(card, target: int) -> int:
    cols = np.flatnonzero(card.gw == target)
    if len(cols) == 0:
        raise ValueError(f"no column for gameweek {target}")
    return int(cols[0])


def _event_strip_from_counts(goals: np.ndarray, assists: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Clean event-strip picture for one column. Shape ROWS, 2, EVENT_H, SQ, 3."""
    strip = np.ones((ROWS, 2, EVENT_H, CELL_W, 3), dtype=np.float32)
    for row in range(ROWS):
        if not present[row]:
            continue
        for channel, count, ink in ((0, int(goals[row]), GOAL_INK / 255.0), (1, int(assists[row]), ASSIST_INK / 255.0)):
            count = max(0, min(count, EVENT_N))
            for slot in range(count):
                y0 = EVENT_H - (slot + 1) * CELL_W
                strip[row, channel, y0 : y0 + CELL_W] = ink
    return strip


def _present_rows(card, col: int) -> np.ndarray:
    return card.present[:, col].astype(bool)


def _ink_counts(strip: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read bars from painted colors. A square counts when nearer the event ink than white."""
    goals = np.zeros(ROWS, dtype=np.int64)
    assists = np.zeros(ROWS, dtype=np.int64)
    for row in range(ROWS):
        if not present[row]:
            continue
        for slot in range(EVENT_N):
            y0 = EVENT_H - (slot + 1) * CELL_W
            patch_g = strip[row, 0, y0 : y0 + CELL_W].mean(axis=(0, 1))
            patch_a = strip[row, 1, y0 : y0 + CELL_W].mean(axis=(0, 1))
            if np.linalg.norm(patch_g - GOAL_INK / 255.0) < np.linalg.norm(patch_g - WHITE / 255.0):
                goals[row] += 1
            if np.linalg.norm(patch_a - ASSIST_INK / 255.0) < np.linalg.norm(patch_a - WHITE / 255.0):
                assists[row] += 1
    return goals, assists


class ColumnFlashcard(nn.Module):
    """Whole card in; one column event strip out."""

    def __init__(self, header_h: int = HEADER_H, player_h: int = PLAYER_H, dropout: float = 0.1) -> None:
        super().__init__()
        self.header_h = int(header_h)
        self.player_h = int(player_h)
        self.header = nn.Linear(3 * self.header_h * CELL_W, D)
        self.header_norm = nn.LayerNorm(D)
        bands = _square_bands(self.player_h)
        assert bands is not None
        self.band_h = bands
        self.bands = nn.ModuleList(nn.Linear(3 * height * CELL_W, D) for height in self.band_h)
        self.band_norm = nn.ModuleList(nn.LayerNorm(D) for _ in self.band_h)
        self.row = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.col = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_row = nn.LayerNorm(D)
        self.norm_col = nn.LayerNorm(D)
        self.col_pos = nn.Parameter(torch.zeros(1, 1, COLS, D))
        nn.init.normal_(self.col_pos, std=0.02)
        self.column_pool = nn.Linear(D, D)
        # Ink amount on each event square (goals then assists). The strip picture
        # is those squares filled with event color or left white.
        self.row_paint = nn.Linear(D, 2 * EVENT_N)
        nn.init.zeros_(self.row_paint.weight)
        nn.init.constant_(self.row_paint.bias, -4.0)

    def encode(self, images: torch.Tensor, row_pad: torch.Tensor, column_on: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        header = images[:, :, : self.header_h, :].reshape(batch, 3, self.header_h, COLS, CELL_W)
        header = header.permute(0, 3, 1, 2, 4).reshape(batch, COLS, -1)
        header_tok = self.header_norm(self.header(header))
        body = images[:, :, self.header_h :, :].reshape(batch, 3, ROWS, self.player_h, COLS, CELL_W)
        flat = body.permute(0, 4, 2, 1, 3, 5)
        cells = flat.new_zeros(batch, COLS, ROWS, D)
        y0 = 0
        for layer, norm, height in zip(self.bands, self.band_norm, self.band_h, strict=True):
            part = flat[:, :, :, :, y0 : y0 + height, :].reshape(batch, COLS, ROWS, -1)
            cells = cells + norm(layer(part))
            y0 += height
        cells = cells.permute(0, 2, 1, 3) + header_tok[:, None, :, :]
        col_pad = column_on < 0.5
        hidden = self.norm_row(cells + self.col_pos[:, :, :COLS])
        row_view = hidden.permute(0, 2, 1, 3).reshape(batch * COLS, ROWS, D)
        row_mask = row_pad[:, None, :].expand(batch, COLS, ROWS).reshape(batch * COLS, ROWS)
        mixed, _ = self.row(row_view, row_view, row_view, key_padding_mask=row_mask, need_weights=False)
        row_view = self.norm_row(row_view + mixed)
        col_view = row_view.reshape(batch, COLS, ROWS, D).permute(0, 2, 1, 3).reshape(batch * ROWS, COLS, D)
        col_mask = col_pad[:, None, :].expand(batch, ROWS, COLS).reshape(batch * ROWS, COLS)
        mixed, _ = self.col(col_view, col_view, col_view, key_padding_mask=col_mask, need_weights=False)
        col_view = self.norm_col(col_view + mixed)
        return col_view.reshape(batch, ROWS, COLS, D)

    def forward(
        self,
        images: torch.Tensor,
        row_pad: torch.Tensor,
        column_on: torch.Tensor,
        columns: torch.Tensor,
    ) -> torch.Tensor:
        """Return the painted event strip for each named column. Shape B, ROWS, 2, EVENT_H, SQ, 3.

        Rows keep their tokens, but the next column is pooled into one context
        vector and added to every row before the strip is written — one column
        object, one strip. Ink on each square becomes event color or white.
        """
        tokens = self.encode(images, row_pad, column_on)
        batch = tokens.shape[0]
        index = columns.view(batch, 1, 1, 1).expand(batch, ROWS, 1, D)
        chosen = tokens.gather(2, index).squeeze(2)
        live = (~row_pad).unsqueeze(-1).to(chosen.dtype)
        pooled = (chosen * live).sum(dim=1, keepdim=True) / live.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.nn.functional.gelu(self.column_pool(pooled))
        painted = chosen + pooled
        ink = torch.sigmoid(self.row_paint(painted)).view(batch, ROWS, 2, EVENT_N)
        colors = torch.ones(batch, ROWS, 2, EVENT_H, CELL_W, 3, device=images.device, dtype=images.dtype)
        goal = torch.tensor(GOAL_INK / 255.0, device=images.device, dtype=images.dtype).view(1, 1, 1, 1, 1, 3)
        assist = torch.tensor(ASSIST_INK / 255.0, device=images.device, dtype=images.dtype).view(1, 1, 1, 1, 1, 3)
        for slot in range(EVENT_N):
            y0 = EVENT_H - (slot + 1) * CELL_W
            g = ink[:, :, 0, slot, None, None, None]
            a = ink[:, :, 1, slot, None, None, None]
            colors[:, :, 0, y0 : y0 + CELL_W] = (1.0 - g) * 1.0 + g * goal
            colors[:, :, 1, y0 : y0 + CELL_W] = (1.0 - a) * 1.0 + a * assist
        return colors, ink


def _gameweeks(card) -> list[int]:
    return sorted({int(g) for g in card.gw if int(g) > 0})


def _batch(
    model: ColumnFlashcard,
    cards: list,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict,
    targets: list[int],
    device: torch.device,
):
    images = []
    truths = []
    row_pad = []
    column_on = []
    columns = []
    for card, target in zip(cards, targets, strict=True):
        blank = _step_blank(card, league, clubs, squads, target)
        col = _column_index(card, target)
        # Blank events via MASK tokens only — keep position color on the week.
        shown = compose_square(blank, card, odds, grey_columns=None, target=target)
        present = _present_rows(card, col)
        images.append(shown)
        truths.append(_event_strip_from_counts(card.goals[:, col], card.assists[:, col], present))
        pad = np.ones(ROWS, dtype=bool)
        pad[: card.n_players] = False
        for row in range(card.n_players):
            if int(card.position[row]) < 0:
                pad[row] = True
        row_pad.append(pad)
        on = np.zeros(COLS, dtype=np.float32)
        on[blank["column_on"] > 0.5] = 1.0
        column_on.append(on)
        columns.append(col)
    image_t = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div(255.0).to(device)
    truth_t = torch.from_numpy(np.stack(truths)).to(device)
    pad_t = torch.from_numpy(np.stack(row_pad)).to(device)
    on_t = torch.from_numpy(np.stack(column_on)).to(device)
    col_t = torch.tensor(columns, device=device, dtype=torch.long)
    return image_t, truth_t, pad_t, on_t, col_t, columns


def _truth_ink(truth: torch.Tensor) -> torch.Tensor:
    """Soft ink on each event square from the strip picture. Shape B, ROWS, 2, EVENT_N."""
    squares = []
    for slot in range(EVENT_N):
        y0 = EVENT_H - (slot + 1) * CELL_W
        patch = truth[:, :, :, y0 : y0 + CELL_W].mean(dim=(-2, -3))
        squares.append((1.0 - patch).abs().mean(dim=-1))
    return torch.stack(squares, dim=-1)


def column_picture_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    present: torch.Tensor,
    ink: torch.Tensor,
) -> torch.Tensor:
    """One number for the column strip as a sparse picture object.

    Dice on the ink pattern treats the strip as one mask — missing ink and
    extra ink both cost — so plain white cannot win by pixel count.
    """
    live = present[:, :, None, None].to(ink.dtype)
    target = _truth_ink(truth) * live
    pred_ink = ink * live
    dice_terms = []
    for channel in range(2):
        intersection = (pred_ink[:, :, channel] * target[:, :, channel]).sum()
        denom = pred_ink[:, :, channel].sum() + target[:, :, channel].sum()
        dice_terms.append(1.0 - (2.0 * intersection + 1e-5) / (denom + 1e-5))
    dice = 0.5 * (dice_terms[0] + dice_terms[1])
    color = ((pred - truth).abs() * present[:, :, None, None, None, None].to(pred.dtype)).sum()
    color = color / (present.sum().clamp_min(1.0) * 2 * EVENT_H * CELL_W * 3)
    return dice + 0.05 * color


def train_column_picture(
    model: ColumnFlashcard,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    squads: dict,
    odds: dict,
    device: torch.device,
) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            targets = [int(rng.choice(_gameweeks(card))) for card in pick]
            image_t, truth_t, pad_t, on_t, col_t, _ = _batch(
                model, pick, league, clubs, squads, odds, targets, device
            )
            present = torch.stack(
                [
                    torch.from_numpy(_present_rows(card, col)).to(device)
                    for card, col in zip(pick, col_t.tolist(), strict=True)
                ]
            )
            present = present & (~pad_t)
            opt.zero_grad()
            pred, ink = model(image_t, pad_t, on_t, col_t)
            loss = column_picture_loss(pred, truth_t, present, ink)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"column picture epoch {epoch + 1} strip error {total / max(seen, 1):.4f}", flush=True)


@torch.no_grad()
def paint_column_picture(
    model: ColumnFlashcard,
    cards: list,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk every gameweek; stamp the predicted strip; read bars from color."""
    goals = np.zeros((len(cards),) + cards[0].goals.shape, dtype=np.int64)
    assists = np.zeros_like(goals)
    model.eval()
    by_gw: dict[int, list[int]] = {}
    for index, card in enumerate(cards):
        for gw in _gameweeks(card):
            by_gw.setdefault(gw, []).append(index)
    for gw, indexes in sorted(by_gw.items()):
        pick = [cards[i] for i in indexes]
        targets = [gw] * len(pick)
        image_t, _, pad_t, on_t, col_t, columns = _batch(
            model, pick, league, clubs, squads, odds, targets, device
        )
        pred, _ink = model(image_t, pad_t, on_t, col_t)
        pred = pred.cpu().numpy()
        for slot, index in enumerate(indexes):
            card = cards[index]
            col = columns[slot]
            present = _present_rows(card, col)
            g, a = _ink_counts(pred[slot], present)
            goals[index, :, col] = g
            assists[index, :, col] = a
    return goals, assists


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_teach_diagram(
    card,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict,
    model: ColumnFlashcard,
    device: torch.device,
    path: Path,
) -> Path:
    """Shown card → one column embedding → painted strip → read bars."""
    target = _gameweeks(card)[max(0, len(_gameweeks(card)) // 5)]
    blank = _step_blank(card, league, clubs, squads, target)
    truth = _step_truth(card, league, clubs, squads, target)
    col = _column_index(card, target)
    shown = compose_square(blank, card, odds, grey_columns=None, target=target)
    actual = compose_square(truth, card, odds, grey_columns=None, target=target)
    image_t, truth_t, pad_t, on_t, col_t, _ = _batch(
        model, [card], league, clubs, squads, odds, [target], device
    )
    with torch.no_grad():
        pred, _ink = model(image_t, pad_t, on_t, col_t)
        pred = pred[0].cpu().numpy()
    painted = np.array(shown, copy=True)
    x0 = col * CELL_W
    for row in range(ROWS):
        if int(card.position[row]) < 0 or not card.present[row, col]:
            continue
        y0 = HEADER_H + row * PLAYER_H + MINUTE_H
        painted[y0 : y0 + EVENT_H, x0 : x0 + CELL_W] = np.clip(pred[row, 0] * 255.0, 0, 255).astype(np.uint8)
        painted[y0 + EVENT_H : y0 + 2 * EVENT_H, x0 : x0 + CELL_W] = np.clip(
            pred[row, 1] * 255.0, 0, 255
        ).astype(np.uint8)

    def panel(title: str, subtitle: str, image: np.ndarray) -> Image.Image:
        # crop to first 12 columns and first 16 player rows for a readable strip
        cols = min(12, image.shape[1] // CELL_W)
        rows = min(16, ROWS)
        crop = image[: HEADER_H + rows * PLAYER_H, : cols * CELL_W]
        scale = 3
        big = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
        sheet = Image.fromarray(big)
        canvas = Image.new("RGB", (sheet.width, sheet.height + 70), (250, 248, 242))
        canvas.paste(sheet, (0, 70))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), title, fill=(20, 20, 20), font=_font(22))
        draw.text((12, 40), subtitle, fill=(80, 80, 80), font=_font(14))
        return canvas

    panels = [
        panel("1. Shown flashcard", "past events visible; next column events blank", shown),
        panel("2. Actual strip", "the real next-column event picture", actual),
        panel("3. Model paint", "one column embedding writes the whole strip", painted),
    ]
    gap = 24
    width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    height = max(p.height for p in panels) + 120
    board = Image.new("RGB", (width, height), (235, 232, 224))
    draw = ImageDraw.Draw(board)
    draw.text(
        (24, 20),
        "Column flashcard learner — how one training example is taught",
        fill=(20, 20, 20),
        font=_font(28),
    )
    draw.text(
        (24, 56),
        "Read the whole card → pool the next column into one object → paint its event strip → "
        "loss is the strip color error. Bars are read from ink afterwards.",
        fill=(60, 60, 60),
        font=_font(16),
    )
    x = 0
    for panel_img in panels:
        board.paste(panel_img, (x, 100))
        x += panel_img.width + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path)
    return path


def _flow_diagram(path: Path) -> Path:
    """Simple processing flow, saved next to the example teach sheet."""
    img = Image.new("RGB", (1100, 420), (235, 232, 224))
    draw = ImageDraw.Draw(img)
    font = _font(18)
    title = _font(26)
    draw.text((24, 20), "Flashcard processing (stepped column)", fill=(20, 20, 20), font=title)
    boxes = [
        (40, 90, 280, 200, "Square card\nclub / opp / venue / 1X2\nminutes + blank events"),
        (340, 90, 580, 200, "Encode whole card\nband tokens + attention"),
        (640, 90, 880, 200, "One column vector\npool the next week"),
        (40, 250, 280, 360, "Decode event strip\nall rows, goals+assists"),
        (340, 250, 580, 360, "Teach: strip Dice\none object, one loss"),
        (640, 250, 880, 360, "Read bars from ink\ncloser to event color\nthan to white"),
    ]
    for x0, y0, x1, y1, text in boxes:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=(250, 248, 242), outline=(40, 40, 40), width=2)
        draw.multiline_text((x0 + 16, y0 + 24), text, fill=(20, 20, 20), font=font, spacing=4)
    for a, b in (((280, 145), (340, 145)), ((580, 145), (640, 145)), ((160, 200), (160, 250)), ((460, 200), (460, 250)), ((760, 200), (760, 250))):
        draw.line([a, b], fill=(40, 40, 40), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cpu")
    print("loading column flashcard learner", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(train + test)
    odds = load_odds()
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    print(f"train {len(train)} test {len(test)}", flush=True)
    torch.manual_seed(SEED + 1)
    model = ColumnFlashcard()
    train_column_picture(
        model, train, league, np.random.default_rng(SEED + 1), clubs, squads, odds, device
    )
    flow = _flow_diagram(ART / "column_picture_flow.png")
    teach = write_teach_diagram(
        test[0], league, clubs, squads, odds, model, device, ART / "column_picture_teach.png"
    )
    print(f"wrote diagrams {flow} {teach}", flush=True)
    print("painting stepped columns from pictures", flush=True)
    goal_paint, assist_paint = paint_column_picture(model, test, league, clubs, squads, odds, device)
    lines = [
        "# Column flashcard learner",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "The stepped square card is the only input. The model pools the next",
        "column into one object and paints that column's event strip in one",
        "pass. The loss is the strip color error. There is no per-cell class",
        "head, no Poisson rate, no share, and no argmax paint rule. Bars are",
        "read from ink afterwards.",
        "",
        "Diagrams: `column_picture_flow.png` (processing),",
        "`column_picture_teach.png` (one training example).",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    position: list[str] = []
    for revealed in (0.2, 0.4, 0.6, 0.8):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        goals = tally_cards(test, labels, goal_paint, 0, cuts)
        assists = tally_cards(test, assist_labels, assist_paint, 0, cuts)
        lines.append(
            f"| {revealed:.0%} | goals | {goals['actual']} | {goals['painted']} | {goals['exact']} | "
            f"{goals['actual_within']} | {goals['painted_within']} |"
        )
        lines.append(
            f"| {revealed:.0%} | assists | {assists['actual']} | {assists['painted']} | {assists['exact']} | "
            f"{assists['actual_within']} | {assists['painted_within']} |"
        )
        if revealed == 0.2:
            position = _position_lines(
                test,
                labels,
                goal_paint,
                assist_labels,
                assist_paint,
                cuts,
                "Column flashcard learner, by position at the 20% cut.",
            )
        print(
            f"column picture {revealed:.0%} goals {goals['actual']} painted {goals['painted']} "
            f"exact {goals['exact']} within {goals['actual_within']} "
            f"assists exact {assists['exact']} within {assists['actual_within']}",
            flush=True,
        )
    lines.extend(position)
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "Core idea: the flashcard is one picture. The column is one object.",
            "Do not replace this with cell posteriors, volume caps, or share rules.",
            "",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", flush=True)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
