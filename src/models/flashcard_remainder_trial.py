"""One paint for the whole gray remainder of the card.

The training set is the rendered card. A future goal or assist square is gray.
Odds, venue, the opponent mark, the minute stack, and the position color stay
visible, including on a future gameweek. The painter reads that picture once,
draws one code for the entire gray region, and paints every gray square from
that code. The codes are learned from the training pictures. The stored gate
stays the 21:47 UK full painter.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.models.flashcard_card_score import tally_cards
from src.models.flashcard_image_trial import (
    COLS_SHOWN,
    EVENT_SLOTS,
    WHITE,
    _GATE,
    _crop_rows,
    _panel,
    apply_bits,
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
from src.models.flashcard_squad_trial import (
    BATCH,
    COLS,
    D,
    EPOCHS,
    HEADS,
    LR,
    ROWS,
    SEED,
    build_cards,
    build_league,
    cut_at_fraction,
    sample_cut,
)
from src.models.flashcard_square_trial import HEADER_H, PLAYER_H, SQ
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUT = Path("/opt/cursor/artifacts")
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
HEADER_CELLS = HEADER_H // SQ
PLAYER_CELLS = PLAYER_H // SQ
CODES = 32
WARMUP = 10
BETA = 0.1
POS_WEIGHT = 5.0
CELL_ROWS = HEADER_CELLS + ROWS * PLAYER_CELLS


class _Mix(nn.Module):
    """Players in one gameweek meet, then the weeks of one player meet."""

    def __init__(self) -> None:
        super().__init__()
        self.player = nn.MultiheadAttention(D, HEADS, dropout=0.0, batch_first=True)
        self.week = nn.MultiheadAttention(D, HEADS, dropout=0.0, batch_first=True)
        self.norm_player = nn.LayerNorm(D)
        self.norm_week = nn.LayerNorm(D)
        self.norm_ff = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, rows, columns, width = tokens.shape
        players = tokens.permute(0, 2, 1, 3).reshape(batch * columns, rows, width)
        mixed, _ = self.player(players, players, players, need_weights=False)
        players = self.norm_player(players + mixed)
        tokens = players.reshape(batch, columns, rows, width).permute(0, 2, 1, 3)
        weeks = tokens.reshape(batch * rows, columns, width)
        mixed, _ = self.week(weeks, weeks, weeks, need_weights=False)
        weeks = self.norm_week(weeks + mixed)
        tokens = weeks.reshape(batch, rows, columns, width)
        return self.norm_ff(tokens + self.ff(tokens))


class RemainderPainter(nn.Module):
    """Read the card and paint the gray remainder from one code."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Conv2d(SQ * SQ * 3, D, kernel_size=1)
        self.collapse = nn.Linear(PLAYER_CELLS * D, D)
        self.header = nn.Linear(HEADER_CELLS * D, D)
        self.read = _Mix()
        self.place = _Mix()
        self.query_full = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.query_seen = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.pool_full = nn.MultiheadAttention(D, HEADS, dropout=0.0, batch_first=True)
        self.pool_seen = nn.MultiheadAttention(D, HEADS, dropout=0.0, batch_first=True)
        self.full_head = nn.Linear(D, CODES)
        self.seen_head = nn.Linear(D, CODES)
        self.book = nn.Embedding(CODES, D)
        self.from_code = nn.Linear(D, D)
        self.scale = nn.Linear(D, D)
        self.paint = nn.Linear(D, EVENT_SLOTS)
        self.row_mark = nn.Parameter(torch.randn(1, ROWS, 1, D) * 0.02)
        self.column_mark = nn.Parameter(torch.randn(1, 1, COLS, D) * 0.02)
        nn.init.normal_(self.book.weight, std=0.1)
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.constant_(self.paint.bias, -2.0)

    def tokens_of(self, patches: torch.Tensor) -> torch.Tensor:
        local = F.gelu(self.embed(patches))
        batch, width, _, columns = local.shape
        body = local[:, :, HEADER_CELLS:].reshape(batch, width, ROWS, PLAYER_CELLS, columns)
        flat = body.permute(0, 2, 4, 3, 1).reshape(batch, ROWS, columns, PLAYER_CELLS * width)
        players = self.collapse(flat)
        head = local[:, :, :HEADER_CELLS].permute(0, 3, 2, 1).reshape(batch, columns, HEADER_CELLS * width)
        tokens = players + self.header(head)[:, None, :, :] + self.row_mark + self.column_mark
        return self.read(tokens)

    def _logits(self, tokens: torch.Tensor, query: torch.Tensor, pool: nn.MultiheadAttention, head: nn.Linear) -> torch.Tensor:
        batch = tokens.shape[0]
        flat = tokens.reshape(batch, ROWS * COLS, D)
        pooled, _ = pool(query.expand(batch, -1, -1), flat, flat, need_weights=False)
        return head(pooled[:, 0])

    def full_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._logits(tokens, self.query_full, self.pool_full, self.full_head)

    def seen_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._logits(tokens, self.query_seen, self.pool_seen, self.seen_head)

    def pick(self, logits: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """One code for the card. The forward value is that one code."""
        if self.training and torch.is_grad_enabled():
            noise = torch.rand(logits.shape, device=logits.device, dtype=logits.dtype).clamp(1e-6, 1 - 1e-6)
            soft = torch.softmax(logits - torch.log(-torch.log(noise)), dim=-1)
            hard = torch.zeros_like(soft).scatter(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
            choice = hard + soft - soft.detach()
        else:
            probs = torch.softmax(logits, dim=-1)
            index = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            choice = F.one_hot(index, CODES).to(dtype=probs.dtype)
        return choice @ self.book.weight

    def decode(self, tokens: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        shift = self.from_code(code)[:, None, None, :]
        scale = self.scale(code)[:, None, None, :]
        placed = self.place(tokens * (1 + scale) + shift)
        return self.paint(placed).permute(0, 1, 3, 2)

    def forward(self, full: torch.Tensor, seen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_tokens = self.tokens_of(full)
        seen_tokens = self.tokens_of(seen)
        full_logits = self.full_logits(full_tokens)
        seen_logits = self.seen_logits(seen_tokens)
        paint = self.decode(seen_tokens, self.pick(full_logits, None))
        return paint, full_logits, seen_logits


def _kl(full_logits: torch.Tensor, seen_logits: torch.Tensor) -> torch.Tensor:
    full = torch.log_softmax(full_logits, dim=-1)
    seen = torch.log_softmax(seen_logits, dim=-1)
    return torch.sum(torch.exp(full) * (full - seen), dim=-1).mean()


def _reconstruction(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = torch.tensor(POS_WEIGHT, device=logits.device)
    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=weight)
    return (per * mask).sum() / mask.sum().clamp_min(1)


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


def train_remainder(
    model: RemainderPainter,
    cards: list,
    cells: np.ndarray,
    ink: np.ndarray,
    rng: np.random.Generator,
    device: torch.device,
    epochs: int = EPOCHS,
) -> None:
    opt = _optimizer(model)
    model.train()
    target_rate = float(ink.mean())
    for epoch in range(epochs):
        beta = BETA * min(1.0, epoch / WARMUP)
        total = 0.0
        seen = 0
        painted_rate = 0.0
        prior_rate = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            indexes = order[start : start + BATCH]
            cuts = [sample_cut(cards[int(i)], rng) for i in indexes]
            full = np.stack([cells[int(i)] for i in indexes])
            target = torch.from_numpy(np.stack([ink[int(i)] for i in indexes])).to(device).float()
            hidden = torch.from_numpy(hidden_events([cards[int(i)] for i in indexes], cuts)).to(device)
            blank = np.array(full, copy=True)
            apply_blank(blank, hidden.cpu().numpy())
            opt.zero_grad()
            logits, full_logits, seen_logits = model(patches_of(full).to(device), patches_of(blank).to(device))
            loss = _reconstruction(logits, target, hidden.float()) + beta * _kl(full_logits, seen_logits)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
            covered = hidden.float().sum().clamp_min(1)
            painted_rate = float(((logits.detach() > 0) & hidden).float().sum() / covered)
            if start + BATCH >= len(cards):
                model.eval()
                with torch.no_grad():
                    seen_tokens = model.tokens_of(patches_of(blank).to(device))
                    code = model.pick(model.seen_logits(seen_tokens), None)
                    prior_bits = model.decode(seen_tokens, code) > 0
                    prior_rate = float((prior_bits & hidden).float().sum() / covered)
                model.train()
        print(
            f"remainder epoch {epoch + 1} loss {total / max(seen, 1):.3f} beta {beta:.3f} "
            f"ink {target_rate:.4f} train {painted_rate:.4f} draw {prior_rate:.4f}",
            flush=True,
        )


def _counts(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    goals = ink[:, :, :3].sum(axis=2).astype(np.int64)
    assists = ink[:, :, 3:].sum(axis=2).astype(np.int64)
    return goals, assists


@torch.no_grad()
def paint_remainder(
    model: RemainderPainter,
    cards: list,
    cells: np.ndarray,
    cuts: list[int],
    device: torch.device,
    generator: torch.Generator,
    keep: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict[str, int]]:
    """One code per card, then every gray square in one paint."""
    model.eval()
    images = np.stack([cells[i] for i in range(len(cards))])
    hidden = hidden_events(cards, cuts)
    apply_blank(images, hidden)
    chosen = np.zeros_like(hidden)
    for start in range(0, len(cards), BATCH):
        stop = min(start + BATCH, len(cards))
        tokens = model.tokens_of(patches_of(images[start:stop]).to(device))
        code = model.pick(model.seen_logits(tokens), generator)
        bits = model.decode(tokens, code) > 0
        chosen[start:stop] = bits.cpu().numpy()
    apply_bits(images, chosen, hidden)
    ink = ink_of(images, hidden[:, :, 0, :]) & hidden
    goals, assists = _counts(ink)
    kept = images[keep] if keep is not None else None
    return goals, assists, kept, ink, stack_faults(ink)


def _appearance_rate(mask: np.ndarray, ink: np.ndarray, slots: slice) -> float:
    marks = ink[:, :, slots].any(axis=2)
    return float(marks[mask].mean()) if bool(mask.any()) else 0.0


def _slot_rate(mask: np.ndarray, ink: np.ndarray, slot: int) -> float:
    return float(ink[:, :, slot][mask].mean()) if bool(mask.any()) else 0.0


def minute_rates(ink: np.ndarray, minutes: np.ndarray, present: np.ndarray) -> dict[str, float | list[float]]:
    """Event squares on a zero-minute appearance and on a played one."""
    zero = present & (minutes <= 0)
    played = present & (minutes > 0)
    out: dict[str, float | list[float]] = {}
    for channel, band, bottom in (("goal", slice(0, 3), (2, 1, 0)), ("assist", slice(3, 6), (5, 4, 3))):
        out[channel + "_zero"] = _appearance_rate(zero, ink, band)
        out[channel + "_played"] = _appearance_rate(played, ink, band)
        for label, mask in (("zero", zero), ("played", played)):
            out[f"{channel}_{label}_slots"] = [_slot_rate(mask, ink, slot) for slot in bottom]
    return out


def assist_partners(
    goals: np.ndarray,
    assists: np.ndarray,
    minutes: np.ndarray,
    cards: list,
    cuts: list[int],
) -> tuple[int, int]:
    """Goals whose gameweek also paints an assist on another player with minutes."""
    paired = 0
    total = 0
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        for col in np.flatnonzero(hidden):
            played = minutes[index, :, col] > 0
            partners = (assists[index, :, col] > 0) & played
            for row in np.flatnonzero((goals[index, :, col] > 0) & played):
                total += 1
                others = partners.copy()
                others[row] = False
                paired += int(bool(others.any()))
    return paired, total


def form_runs(goals: np.ndarray, cards: list, cuts: list[int]) -> tuple[int, int]:
    """Goal appearances with another goal by the same player on a neighboring fixture."""
    beside = 0
    total = 0
    for index, card in enumerate(cards):
        hidden = np.flatnonzero((card.gw > cuts[index]) & (card.gw > 0))
        if len(hidden) == 0:
            continue
        marks = goals[index, : card.n_players][:, hidden] > 0
        total += int(marks.sum())
        if len(hidden) < 2:
            continue
        neighbor = marks[:, :-1] & marks[:, 1:]
        touched = np.zeros_like(marks)
        touched[:, :-1] |= neighbor
        touched[:, 1:] |= neighbor
        beside += int(touched.sum())
    return beside, total


def _row(revealed: float, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | {channel} | {totals['actual']} | {totals['painted']} | "
        f"{totals['exact']} | {totals['actual_within']} | {totals['painted_within']} |"
    )


def _compare(revealed: float, channel: str, totals: dict[str, int]) -> str:
    exact, within = _GATE[(channel, revealed)]
    return (
        f"| {revealed:.0%} | {channel} | {totals['actual']} | {totals['painted']} | {totals['exact']} | {exact} | "
        f"{totals['exact'] - exact:+d} | {totals['actual_within']} | {within} | "
        f"{totals['actual_within'] - within:+d} |"
    )


def _rate(hit: int, total: int) -> str:
    if total == 0:
        return "0/0"
    return f"{hit}/{total} ({hit / total:.0%})"


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
        ("Remainder", image_of(painted)),
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
    sheet.save(OUT / "remainder-card-arsenal.png")


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
    present = np.stack([card.present for card in train])
    train_ink = ink_of(train_cells, present)
    print(f"train {len(train)} test {len(test)} pictures", flush=True)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    torch.manual_seed(SEED)
    model = RemainderPainter().to(device)
    print("training remainder", flush=True)
    train_remainder(model, train, train_cells, train_ink, np.random.default_rng(SEED), device, epochs=40)
    lines = [
        "# Remainder painter",
        "",
        "The training set is the rendered card. Each square is one cell.",
        "A future goal or assist square is gray. The painter reads the whole",
        "card once, draws one code for the gray remainder, and paints every",
        "gray square from that code and the visible card. Odds, venue, the",
        "opponent mark, the minute stack, and the position color stay on the",
        "picture, including on a future gameweek. The code is one of the",
        "remainder patterns learned from the training pictures. Each finished",
        "card is one draw at temperature 1. The other squad is not an input.",
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
    chances: dict[float, dict] = {}
    arsenal = next(i for i, card in enumerate(test) if card.team == "Arsenal")
    kept = None
    for revealed in FRACTIONS:
        cuts = [cut_at_fraction(card, revealed) for card in test]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED)
        keep = arsenal if revealed == 0.2 else None
        painted_g, painted_a, image, ink, fault = paint_remainder(
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
                f"remainder {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
        goal_pair = assist_partners(painted_g, painted_a, minutes, test, cuts)
        actual_pair = assist_partners(labels, assist_labels, minutes, test, cuts)
        partners.append(
            f"| {revealed:.0%} | {_rate(*actual_pair)} | {_rate(*goal_pair)} |"
        )
        actual_run = form_runs(labels, test, cuts)
        painted_run = form_runs(painted_g, test, cuts)
        runs.append(f"| {revealed:.0%} | {_rate(*actual_run)} | {_rate(*painted_run)} |")
        print(
            f"remainder {revealed:.0%} assist beside a goal {goal_pair[0]}/{goal_pair[1]} "
            f"actual {actual_pair[0]}/{actual_pair[1]}",
            flush=True,
        )
    lines.extend(
        [
            "",
            "One code is drawn for the whole remainder. The paint of every future",
            "event square is fixed by that code and the visible card. The stored",
            "gate is the 21:47 UK full painter. This run does not replace it.",
            "",
            "| revealed | channel | actual bars | painted bars | same appearance | gate | difference | within 1 GW | gate | difference |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *compared,
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
    path = ROOT_REPORTS / "flashcard_remainder_trial.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    cut = cut_at_fraction(test[arsenal], 0.2)
    _examples(test[arsenal], test_cells[arsenal], kept, cut)
    print(f"remainder painter finished in {time.perf_counter() - started:.0f}s", flush=True)
    return {"report": path, "picture": OUT / "remainder-card-arsenal.png"}


def main() -> None:
    execute()


if __name__ == "__main__":
    main()
