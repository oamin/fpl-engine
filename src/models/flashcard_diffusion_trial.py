"""Restore the gray remainder from noise, as one picture.

The visible card stays in place. The future event squares start as noise.
A denoiser reads the whole card and proposes the clean colors. Each step
moves the noise toward that picture and adds a little fresh noise, for
32 steps. The last picture is the painted card. A bar is read where the
color is closer to the goal or assist color than to white or to gray.
The stored gate stays the 21:47 UK full painter.
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
from src.models.flashcard_picture_trial import (
    REMAINDER_SCORE,
    PicturePainter,
    event_targets,
    stamp_colors,
)
from src.models.flashcard_remainder_trial import (
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
STEPS = 32
EPOCHS = 80
PICTURE_SCORE = 0.0


def _sinusoid(t: torch.Tensor, width: int = 32) -> torch.Tensor:
    half = width // 2
    freq = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    angle = t.float()[:, None] * freq[None]
    return torch.cat((angle.sin(), angle.cos()), dim=-1)


class Denoiser(PicturePainter):
    """Read a noisy card and propose the clean color of every event square."""

    def __init__(self, steps: int = STEPS) -> None:
        super().__init__()
        beta = torch.linspace(1e-4, 0.15, steps)
        alpha = 1.0 - beta
        self.time_proj = nn.Sequential(nn.Linear(32, D), nn.GELU(), nn.Linear(D, D))
        self.to_color = nn.Linear(D, EVENT_SLOTS * 3)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("bar", torch.cumprod(alpha, dim=0))
        nn.init.zeros_(self.to_color.weight)
        nn.init.constant_(self.to_color.bias, 1.0)

    def forward(self, patches: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tokens = self.place(self.tokens_of(patches))
        tokens = tokens + self.time_proj(_sinusoid(t))[:, None, None, :]
        batch = tokens.shape[0]
        raw = self.to_color(tokens).view(batch, ROWS, COLS, EVENT_SLOTS, 3)
        return raw.permute(0, 1, 3, 2, 4).clamp(0, 1)


def overwrite_events(patches: torch.Tensor, colors: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
    """Put event-square colors into the card. A true mask entry is replaced."""
    out = patches.clone()
    tiled = colors.permute(0, 4, 1, 2, 3).repeat(1, SQ * SQ, 1, 1, 1)
    rows = torch.as_tensor(EVENT_ROW, device=patches.device)
    for slot in range(EVENT_SLOTS):
        current = out[:, :, rows[:, slot], :]
        mask = where[:, :, slot].unsqueeze(1)
        out[:, :, rows[:, slot], :] = torch.where(mask, tiled[:, :, :, slot, :], current)
    return out


def _mix(x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, model: Denoiser, generator: torch.Generator) -> torch.Tensor:
    """One reverse step. The last step keeps the proposed picture."""
    if int(t[0]) == 0:
        return x0
    step = int(t[0])
    beta = model.beta[step]
    bar = model.bar[step]
    previous = model.bar[step - 1]
    coef_x0 = previous.sqrt() * beta / (1 - bar)
    coef_xt = model.alpha[step].sqrt() * (1 - previous) / (1 - bar)
    mean = coef_x0 * x0 + coef_xt * x_t
    var = beta * (1 - previous) / (1 - bar)
    noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
    return mean + var.sqrt() * noise


def q_sample(model: Denoiser, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    bar = model.bar[t].view(-1, 1, 1, 1, 1)
    return bar.sqrt() * x0 + (1.0 - bar).sqrt() * eps


def train_denoiser(
    model: Denoiser,
    cards: list,
    cells: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
    epochs: int = EPOCHS,
    hide=apply_blank,
) -> None:
    clean = event_targets(cells).mean(axis=(-3, -2))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    steps = int(model.beta.shape[0])
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            indexes = order[start : start + BATCH]
            batch_cards = [cards[int(i)] for i in indexes]
            cuts = [sample_cut(card, rng) for card in batch_cards]
            hidden_np = hidden_events(batch_cards, cuts)
            blank = np.array(cells[indexes], copy=True)
            hide(blank, hidden_np)
            hidden = torch.from_numpy(hidden_np).to(device)
            x0 = torch.from_numpy(clean[indexes]).to(device)
            t = torch.randint(0, steps, (len(indexes),), device=device)
            eps = torch.randn(x0.shape, device=device)
            shown = overwrite_events(patches_of(blank).to(device), q_sample(model, x0, t, eps), hidden)
            opt.zero_grad()
            pred = model(shown, t)
            err = (pred - x0).pow(2).mean(dim=-1)
            weight = hidden.float()
            loss = (err * weight).sum() / weight.sum().clamp_min(1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        print(f"diffusion epoch {epoch + 1} picture error {total / max(seen, 1):.4f}", flush=True)


@torch.no_grad()
def restore(model: Denoiser, patches: torch.Tensor, hidden: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Walk the gray squares from noise to a picture. The visible card stays."""
    model.eval()
    colors = torch.randn(
        (patches.shape[0], ROWS, EVENT_SLOTS, COLS, 3),
        generator=generator,
        device=patches.device,
        dtype=patches.dtype,
    )
    steps = int(model.beta.shape[0])
    for step in range(steps - 1, -1, -1):
        t = torch.full((patches.shape[0],), step, device=patches.device, dtype=torch.long)
        shown = overwrite_events(patches, colors, hidden)
        colors = _mix(colors, model(shown, t), t, model, generator)
    return colors.clamp(0, 1)


@torch.no_grad()
def paint_diffusion(
    model: Denoiser,
    cards: list,
    cells: np.ndarray,
    cuts: list[int],
    device: torch.device,
    generator: torch.Generator,
    keep: int | None = None,
    hide=apply_blank,
    read=ink_of,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict[str, int]]:
    model.eval()
    images = np.stack([cells[i] for i in range(len(cards))])
    hidden_np = hidden_events(cards, cuts)
    hide(images, hidden_np)
    colors = np.zeros((len(cards), ROWS, EVENT_SLOTS, COLS, 3), dtype=np.float32)
    for start in range(0, len(cards), BATCH):
        stop = min(start + BATCH, len(cards))
        patches = patches_of(images[start:stop]).to(device)
        hidden = torch.from_numpy(hidden_np[start:stop]).to(device)
        colors[start:stop] = restore(model, patches, hidden, generator).cpu().numpy()
    stamped = stamp_colors(images, colors, hidden_np)
    present = hidden_np[:, :, 0, :]
    ink = read(stamped, present) & hidden_np
    goals = ink[:, :, :3].sum(axis=2).astype(np.int64)
    assists = ink[:, :, 3:].sum(axis=2).astype(np.int64)
    kept = stamped[keep] if keep is not None else None
    return goals, assists, kept, ink, stack_faults(ink)


def _compare(revealed: float, channel: str, totals: dict[str, int]) -> str:
    gate = balanced_score(GATE[(channel, revealed)])
    painted = balanced_score(totals)
    return (
        f"| {revealed:.0%} | {channel} | {totals['actual']} | {GATE[(channel, revealed)]['painted']} | "
        f"{totals['painted']} | {GATE[(channel, revealed)]['exact']} | {totals['exact']} | "
        f"{gate['score']:.3f} | {REMAINDER_SCORE[(channel, revealed)]:.3f} | {PICTURE_SCORE:.3f} | "
        f"{painted['score']:.3f} |"
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
        ("Restored", image_of(painted)),
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
    sheet.save(OUT / "diffusion-card-arsenal.png")


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
    model = Denoiser().to(device)
    print("training diffusion", flush=True)
    train_denoiser(model, train, train_cells, np.random.default_rng(SEED), device)
    lines = [
        "# Diffusion painter",
        "",
        "The training set is the rendered card. Each square is one cell.",
        "A future goal or assist square is gray. Those squares are noised,",
        "and the painter restores the clean picture in 32 steps. The visible",
        "card, including the minute stack, stays in place at every step.",
        "A bar is counted where the restored color is closer to the goal or",
        "assist color than to white or to the gray blank. One finished card",
        "is one draw. The other squad is not an input.",
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
    chances: dict[float, dict] = {}
    arsenal = next(i for i, card in enumerate(test) if card.team == "Arsenal")
    kept = None
    for revealed in FRACTIONS:
        cuts = [cut_at_fraction(card, revealed) for card in test]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED)
        keep = arsenal if revealed == 0.2 else None
        painted_g, painted_a, image, ink, fault = paint_diffusion(
            model, test, test_cells, cuts, device, generator, keep
        )
        if image is not None:
            kept = image
        present_hidden = hidden_events(test, cuts)[:, :, 0, :]
        actual_ink = ink_of(test_cells, np.stack([card.present for card in test]))
        chances[revealed] = {
            "painted": minute_rates(ink, minutes, present_hidden),
            "actual": minute_rates(actual_ink & present_hidden[:, :, None, :], minutes, present_hidden),
        }
        faults.append(
            f"| {revealed:.0%} | {fault['goal_filled']} | {fault['goal_bad']} | "
            f"{fault['assist_filled']} | {fault['assist_bad']} |"
        )
        for channel, painted, actual in (("goals", painted_g, labels), ("assists", painted_a, assist_labels)):
            totals = tally_cards(test, actual, painted, 0, cuts)
            lines.append(_row(revealed, channel, totals))
            compared.append(_compare(revealed, channel, totals))
            print(
                f"diffusion {revealed:.0%} {channel} painted {totals['painted']} "
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
            "21:47 UK full painter. The remainder score drew a count and then",
            "picked weeks. The picture score matched the colors in one pass and",
            "painted no bars. This run does not replace the gate.",
            "",
            "| revealed | channel | actual bars | gate bars | restored bars | gate hits | restored hits | gate score | remainder score | picture score | restored score |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *compared,
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
    path = ROOT_REPORTS / "flashcard_diffusion_trial.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    cut = cut_at_fraction(test[arsenal], 0.2)
    _examples(test[arsenal], test_cells[arsenal], kept, cut)
    print(f"diffusion painter finished in {time.perf_counter() - started:.0f}s", flush=True)
    return {"report": path, "picture": OUT / "diffusion-card-arsenal.png"}


def main() -> None:
    execute()


if __name__ == "__main__":
    main()
