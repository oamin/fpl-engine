"""Paint the gray remainder as one picture.

The painter reads the rendered card and writes a color on every future
event square in one pass. Training is the average absolute difference
between that picture and the real card. A bar is read afterwards, where
the painted color is closer to the goal or assist color than to white
or to the gray blank. The stored gate stays the 21:47 UK full painter.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.flashcard_card_score import GATE, balanced_score, tally_cards
from src.models.flashcard_image_trial import (
    COLS_SHOWN,
    EVENT_ROW,
    EVENT_SLOTS,
    WHITE,
    _crop_rows,
    _panel,
    apply_blank,
    cells_of,
    hidden_events,
    image_of,
    ink_of,
    patches_of,
    render_cells,
    stack_faults,
)
from src.models.flashcard_odds_trial import load_odds
from src.models.flashcard_remainder_trial import (
    HEADER_CELLS,
    PLAYER_CELLS,
    _Mix,
    _rate,
    _row,
    assist_partners,
    form_runs,
    minute_rates,
    string_density,
)
from src.models.flashcard_squad_trial import (
    BATCH,
    COLS,
    D,
    EPOCHS,
    LR,
    ROWS,
    SEED,
    build_cards,
    build_league,
    cut_at_fraction,
    sample_cut,
)
from src.models.flashcard_square_trial import SQ
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUT = Path("/opt/cursor/artifacts")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
# The remainder painter's rank, from its published run.
REMAINDER_SCORE = {
    ("goals", 0.2): 0.134,
    ("assists", 0.2): 0.088,
    ("goals", 0.4): 0.127,
    ("assists", 0.4): 0.085,
    ("goals", 0.6): 0.131,
    ("assists", 0.6): 0.065,
    ("goals", 0.8): 0.092,
    ("assists", 0.8): 0.100,
}


def event_targets(cells: np.ndarray) -> np.ndarray:
    """Real event squares, as colors from 0 to 1. Shape keeps the 4 by 4 pixels."""
    return cells[:, EVENT_ROW].astype(np.float32) / 255.0


def picture_loss(colors: torch.Tensor, target: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Average absolute color error on the gray squares.

    A missing bar and an extra bar of the same color move the picture by
    the same amount. `colors` is one RGB color per square. `target` is the
    real 4 by 4 patch.
    """
    pred = colors[:, :, :, :, None, None, :]
    err = (pred - target).abs()
    weight = hidden.to(dtype=err.dtype)[:, :, :, :, None, None, None]
    pixels = err.shape[-2] * err.shape[-3] * err.shape[-1]
    return (err * weight).sum() / (weight.sum().clamp_min(1.0) * pixels)


def stamp_colors(images: np.ndarray, colors: np.ndarray, where: np.ndarray) -> np.ndarray:
    """Write the painted colors onto the gray squares. Other squares stay."""
    out = np.array(images, copy=True)
    solid = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    for slot in range(EVENT_SLOTS):
        rows = EVENT_ROW[:, slot]
        mask = where[:, :, slot, :, None, None, None]
        color = solid[:, :, slot, :, None, None, :]
        out[:, rows] = np.where(mask, color, out[:, rows])
    return out


class PicturePainter(nn.Module):
    """Read the whole card and paint every event square's color at once."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Conv2d(SQ * SQ * 3, D, kernel_size=1)
        self.collapse = nn.Linear(PLAYER_CELLS * D, D)
        self.header = nn.Linear(HEADER_CELLS * D, D)
        self.read = _Mix()
        self.place = _Mix()
        self.color = nn.Linear(D, EVENT_SLOTS * 3)
        self.row_mark = nn.Parameter(torch.randn(1, ROWS, 1, D) * 0.02)
        self.column_mark = nn.Parameter(torch.randn(1, 1, COLS, D) * 0.02)
        nn.init.zeros_(self.color.weight)
        nn.init.constant_(self.color.bias, 4.0)

    def tokens_of(self, patches: torch.Tensor) -> torch.Tensor:
        local = torch.nn.functional.gelu(self.embed(patches))
        batch, width, _, columns = local.shape
        body = local[:, :, HEADER_CELLS:].reshape(batch, width, ROWS, PLAYER_CELLS, columns)
        flat = body.permute(0, 2, 4, 3, 1).reshape(batch, ROWS, columns, PLAYER_CELLS * width)
        players = self.collapse(flat)
        head = local[:, :, :HEADER_CELLS].permute(0, 3, 2, 1).reshape(batch, columns, HEADER_CELLS * width)
        tokens = players + self.header(head)[:, None, :, :] + self.row_mark + self.column_mark
        return self.read(tokens)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        tokens = self.place(self.tokens_of(patches))
        batch = tokens.shape[0]
        raw = self.color(tokens).view(batch, ROWS, COLS, EVENT_SLOTS, 3)
        return torch.sigmoid(raw).permute(0, 1, 3, 2, 4)


def train_picture(
    model: PicturePainter,
    cards: list,
    cells: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
    epochs: int = EPOCHS,
) -> None:
    targets = event_targets(cells)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            indexes = order[start : start + BATCH]
            cuts = [sample_cut(cards[int(i)], rng) for i in indexes]
            batch_cards = [cards[int(i)] for i in indexes]
            hidden = hidden_events(batch_cards, cuts)
            blank = np.array(cells[indexes], copy=True)
            apply_blank(blank, hidden)
            opt.zero_grad()
            colors = model(patches_of(blank).to(device))
            target = torch.from_numpy(targets[indexes]).to(device)
            loss = picture_loss(colors, target, torch.from_numpy(hidden).to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"picture epoch {epoch + 1} color error {total / max(seen, 1):.4f}", flush=True)


@torch.no_grad()
def paint_picture(
    model: PicturePainter,
    cards: list,
    cells: np.ndarray,
    cuts: list[int],
    device: torch.device,
    keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict[str, int], float]:
    """One picture per card. Bars are read from the painted colors."""
    model.eval()
    images = np.stack([cells[i] for i in range(len(cards))])
    hidden = hidden_events(cards, cuts)
    apply_blank(images, hidden)
    colors = np.zeros((len(cards), ROWS, EVENT_SLOTS, COLS, 3), dtype=np.float32)
    for start in range(0, len(cards), BATCH):
        stop = min(start + BATCH, len(cards))
        pred = model(patches_of(images[start:stop]).to(device))
        colors[start:stop] = pred.cpu().numpy()
    stamped = stamp_colors(images, colors, hidden)
    present = hidden[:, :, 0, :]
    ink = ink_of(stamped, present) & hidden
    target = event_targets(cells)
    weight = hidden.astype(np.float32)
    per_square = np.abs(colors[:, :, :, :, None, None, :] - target).mean(axis=(-1, -2, -3))
    color_error = float((per_square * weight).sum() / max(float(weight.sum()), 1.0))
    goals = ink[:, :, :3].sum(axis=2).astype(np.int64)
    assists = ink[:, :, 3:].sum(axis=2).astype(np.int64)
    kept = stamped[keep] if keep is not None else None
    return goals, assists, kept, ink, stack_faults(ink), color_error


def _compare(revealed: float, channel: str, totals: dict[str, int]) -> str:
    gate = balanced_score(GATE[(channel, revealed)])
    painted = balanced_score(totals)
    return (
        f"| {revealed:.0%} | {channel} | {totals['actual']} | {GATE[(channel, revealed)]['painted']} | "
        f"{totals['painted']} | {GATE[(channel, revealed)]['exact']} | {totals['exact']} | "
        f"{gate['score']:.3f} | {REMAINDER_SCORE[(channel, revealed)]:.3f} | {painted['score']:.3f} |"
    )


def _examples(card, cells: np.ndarray, painted: np.ndarray, cut: int) -> None:
    col_count = min(COLS_SHOWN, int(card.n_fixtures))
    rows = _crop_rows(card, col_count)
    shown_cols = int(np.sum((card.gw[:col_count] > 0) & (card.gw[:col_count] <= cut)))
    hidden_cols = np.flatnonzero((card.gw[:col_count] > cut) & (card.gw[:col_count] > 0))
    shown = np.stack([cells])
    apply_blank(shown, hidden_events([card], [cut]))
    panels = []
    for title, picture in (
        ("Shown", image_of(shown[0])),
        ("Actual", image_of(cells)),
        ("Picture", image_of(painted)),
    ):
        if title == "Shown":
            subtitle = "future event squares are gray"
        else:
            ink = ink_of(cells_of(picture), np.ones((ROWS, COLS), dtype=bool))
            goals = int(ink[:, :3, hidden_cols].any(axis=1)[rows].sum()) if len(hidden_cols) else 0
            assists = int(ink[:, 3:, hidden_cols].any(axis=1)[rows].sum()) if len(hidden_cols) else 0
            subtitle = f"{goals} goal rows and {assists} assist rows in this crop"
        panels.append(_panel(picture, card, rows, col_count, title, subtitle, shown_cols))
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
    sheet.save(OUT / "picture-card-arsenal.png")


def execute() -> dict:
    started = time.perf_counter()
    device = torch.device("cpu")
    print("rendering card pictures", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    odds = load_odds()
    train_cells = np.stack([render_cells(card, odds) for card in train])
    test_cells = np.stack([render_cells(card, odds) for card in test])
    print(f"train {len(train)} test {len(test)} pictures", flush=True)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    torch.manual_seed(SEED)
    model = PicturePainter().to(device)
    print("training picture", flush=True)
    train_picture(model, train, train_cells, np.random.default_rng(SEED), device, epochs=40)
    lines = [
        "# Picture painter",
        "",
        "The training set is the rendered card. Each square is one cell.",
        "A future goal or assist square is gray. The painter reads the whole",
        "card once and writes a color on every future event square. Training",
        "is the average absolute difference between that picture and the real",
        "card. A bar is counted where the painted color is closer to the goal",
        "or assist color than to white or to the gray blank. The other squad",
        "is not an input.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    compared: list[str] = []
    faults: list[str] = []
    partners: list[str] = []
    runs: list[str] = []
    strings: list[str] = []
    errors: list[str] = []
    chances: dict[float, dict] = {}
    arsenal = next(i for i, card in enumerate(test) if card.team == "Arsenal")
    kept = None
    for revealed in FRACTIONS:
        cuts = [cut_at_fraction(card, revealed) for card in test]
        keep = arsenal if revealed == 0.2 else None
        painted_g, painted_a, image, ink, fault, color_error = paint_picture(
            model, test, test_cells, cuts, device, keep
        )
        if image is not None:
            kept = image
        present_hidden = hidden_events(test, cuts)[:, :, 0, :]
        actual_ink = ink_of(test_cells, np.stack([card.present for card in test]))
        chances[revealed] = {
            "painted": minute_rates(ink, minutes, present_hidden),
            "actual": minute_rates(actual_ink & present_hidden[:, :, None, :], minutes, present_hidden),
        }
        errors.append(f"| {revealed:.0%} | {color_error:.4f} |")
        faults.append(
            f"| {revealed:.0%} | {fault['goal_filled']} | {fault['goal_bad']} | "
            f"{fault['assist_filled']} | {fault['assist_bad']} |"
        )
        for channel, painted, actual in (("goals", painted_g, labels), ("assists", painted_a, assist_labels)):
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, channel, totals))
            compared.append(_compare(revealed, channel, totals))
            print(
                f"picture {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} score {balanced_score(totals)['score']:.3f}",
                flush=True,
            )
        goal_pair = assist_partners(painted_g, painted_a, minutes, test, cuts)
        actual_pair = assist_partners(labels, assist_labels, minutes, test, cuts)
        partners.append(f"| {revealed:.0%} | {_rate(*actual_pair)} | {_rate(*goal_pair)} |")
        runs.append(
            f"| {revealed:.0%} | {_rate(*form_runs(labels, test, cuts))} | {_rate(*form_runs(painted_g, test, cuts))} |"
        )
        actual_string = string_density(labels, minutes, test, cuts)
        painted_string = string_density(painted_g, minutes, test, cuts)
        strings.append(
            f"| {revealed:.0%} | {actual_string['peak']:.0f} | {painted_string['peak']:.0f} | "
            f"{actual_string['typical']:.0f} | {painted_string['typical']:.0f} | "
            f"{actual_string['share']:.0%} | {painted_string['share']:.0%} |"
        )
    lines.extend(
        [
            "",
            "The rank is the hit rate times the bar-count match. The gate is the",
            "21:47 UK full painter. The remainder score is the previous painter,",
            "which drew a count and then picked weeks. This run does not replace",
            "the gate.",
            "",
            "| revealed | channel | actual bars | gate bars | picture bars | gate hits | picture hits | gate score | remainder score | picture score |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *compared,
            "",
            "Color error is the average absolute difference on the gray squares,",
            "on the same 0 to 1 scale as the picture.",
            "",
            "| revealed | color error |",
            "|---|---:|",
            *errors,
            "",
            "The busiest string is the most goal appearances one player has in the",
            "hidden window. The typical string is the median of that count across",
            "clubs. The regular share is the fraction of a regular's played weeks",
            "that contain a goal. A regular plays at least half the hidden weeks.",
            "",
            "| revealed | actual busiest | painted busiest | actual typical | painted typical | actual regular share | painted regular share |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *strings,
            "",
            "A goal is paired when another player with minutes in that gameweek",
            "has an assist. A run is a goal appearance with another goal by the",
            "same player on a neighboring hidden fixture.",
            "",
            "| revealed | actual goals with an assist beside them | painted |",
            "|---|---|---|",
            *partners,
            "",
            "| revealed | actual goals in a run | painted |",
            "|---|---|---|",
            *runs,
            "",
            "With the future event squares left gray, the share of appearances",
            "that receive an event square. The squares are listed from the bottom.",
            "",
            "| source | minutes | 1st square | 2nd square | 3rd square | any square |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for revealed in FRACTIONS:
        for source in ("actual", "painted"):
            item = chances[revealed][source]
            for channel in ("goal", "assist"):
                for label in ("zero", "played"):
                    slots = item[f"{channel}_{label}_slots"]
                    rate = item[f"{channel}_{label}"]
                    lines.append(
                        f"| {revealed:.0%} {source} {channel}s | {label} | "
                        f"{slots[0]:.3f} | {slots[1]:.3f} | {slots[2]:.3f} | {rate:.3f} |"
                    )
    lines.extend(
        [
            "",
            "A painted stack skips a lower square when a higher square is filled",
            "and a square below it is empty.",
            "",
            "| revealed | goal stacks | skipping | assist stacks | skipping |",
            "|---|---:|---:|---:|---:|",
            *faults,
            "",
        ]
    )
    path = ROOT_REPORTS / "flashcard_picture_trial.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    cut = cut_at_fraction(test[arsenal], 0.2)
    _examples(test[arsenal], test_cells[arsenal], kept, cut)
    print(f"picture painter finished in {time.perf_counter() - started:.0f}s", flush=True)
    return {"report": path, "picture": OUT / "picture-card-arsenal.png"}


def main() -> None:
    execute()


if __name__ == "__main__":
    main()
