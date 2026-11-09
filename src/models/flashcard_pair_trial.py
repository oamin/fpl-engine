"""Paired sequence trial.

Each club is read with the other squad's recent gameweeks beside it. The
match week and the two before it come from the other card, lined up by
gameweek. Minutes stay visible. Goals and assists after the cut stay masked
on both sides.

The sequence reader is the one trained here. A hidden week is scored on that
club's own players.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    BATCH,
    COLS,
    CUT_GW,
    D,
    EPOCHS,
    LR,
    MASK_ID,
    POS,
    ROWS,
    SEED,
    SquadCard,
    bar_correspondence,
    build_cards,
    build_league,
    capture_table,
    cut_at_fraction,
    positive_by_position,
    sample_cut,
    squad_index,
)

WINDOW = 3
SLOTS = WINDOW * 2
ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"


def pair_view(
    card: SquadCard, cut_gw: int, squads: dict[tuple[str, str], SquadCard]
) -> dict[str, np.ndarray]:
    """Our season, plus the other squad at this gameweek and the two before it."""
    hidden = (card.gw > cut_gw) & (card.gw > 0)
    goal_token = np.full((ROWS, COLS), MASK_ID, dtype=np.int64)
    assist_token = np.full((ROWS, COLS), MASK_ID, dtype=np.int64)
    visible = card.present & ~hidden.reshape(1, COLS)
    goal_token[visible] = card.goals[visible]
    assist_token[visible] = card.assists[visible]
    minutes = np.zeros((ROWS, COLS, SLOTS), dtype=np.float32)
    present = np.zeros((ROWS, COLS, SLOTS), dtype=np.float32)
    position = np.full((ROWS, COLS, SLOTS), -1, dtype=np.int64)
    opp_goal = np.full((ROWS, COLS, SLOTS), MASK_ID, dtype=np.int64)
    opp_assist = np.full((ROWS, COLS, SLOTS), MASK_ID, dtype=np.int64)
    pad = np.ones((ROWS, COLS, SLOTS), dtype=bool)
    for col in range(COLS):
        if card.gw[col] <= 0 or not card.opponent[col]:
            continue
        other = squads.get((card.season, card.opponent[col]))
        if other is None:
            continue
        week = int(card.gw[col])
        wanted = {w for w in range(week - WINDOW + 1, week + 1) if w > 0}
        srcs = [i for i in range(other.n_fixtures) if int(other.gw[i]) in wanted]
        srcs.sort(key=lambda i: int(other.kickoff[i]))
        for slot, src in enumerate(srcs[:SLOTS]):
            position[:, col, slot] = other.position
            minutes[:, col, slot] = other.minutes[:, src]
            present[:, col, slot] = other.present[:, src].astype(np.float32)
            pad[:, col, slot] = other.position < 0
            if int(other.gw[src]) <= cut_gw:
                named = other.position >= 0
                played = other.present[:, src]
                opp_goal[named & played, col, slot] = other.goals[named & played, src]
                opp_assist[named & played, col, slot] = other.assists[named & played, src]
                opp_goal[named & ~played, col, slot] = 0
                opp_assist[named & ~played, col, slot] = 0
    return {
        "minutes": card.minutes.astype(np.float32),
        "present": card.present.astype(np.float32),
        "position": card.position.astype(np.int64),
        "home": card.home.astype(np.float32),
        "goal_token": goal_token,
        "assist_token": assist_token,
        "goal_label": card.goals.astype(np.int64),
        "assist_label": card.assists.astype(np.int64),
        "loss": (card.present & hidden.reshape(1, COLS)).astype(np.float32),
        "row_pad": card.position < 0,
        "column_on": (card.gw > 0).astype(np.float32),
        "gw": card.gw.astype(np.int64),
        "opp_minutes": minutes,
        "opp_present": present,
        "opp_position": position,
        "opp_goal_token": opp_goal,
        "opp_assist_token": opp_assist,
        "opp_pad": pad,
    }


def _stack(views: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    floats = {
        "minutes", "present", "home", "loss", "column_on",
        "opp_minutes", "opp_present",
    }
    longs = {
        "position", "goal_token", "assist_token", "goal_label", "assist_label", "gw",
        "opp_position", "opp_goal_token", "opp_assist_token",
    }
    for key in floats:
        batch[key] = torch.from_numpy(np.stack([view[key] for view in views]).astype(np.float32))
    for key in longs:
        batch[key] = torch.from_numpy(np.stack([view[key] for view in views]).astype(np.int64))
    batch["row_pad"] = torch.from_numpy(np.stack([view["row_pad"] for view in views]))
    batch["opp_pad"] = torch.from_numpy(np.stack([view["opp_pad"] for view in views]))
    return batch


class PairSequence(nn.Module):
    """Read our squad, then the other squad's last three gameweeks."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.cell = nn.Linear(2, D)
        self.position = nn.Embedding(5, D)
        self.goal_in = nn.Embedding(5, D)
        self.assist_in = nn.Embedding(5, D)
        self.home = nn.Linear(1, D)
        self.gw = nn.Embedding(COLS + 1, D)
        self.row = nn.MultiheadAttention(D, 4, dropout=dropout, batch_first=True)
        self.col = nn.MultiheadAttention(D, 4, dropout=dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(D, 4, dropout=dropout, batch_first=True)
        self.norm_row = nn.LayerNorm(D)
        self.norm_col = nn.LayerNorm(D)
        self.norm_cross = nn.LayerNorm(D)
        self.goal = nn.Linear(D, 4)
        self.assist = nn.Linear(D, 4)
        self.pos_goal = nn.Embedding(5, 4)
        self.pos_assist = nn.Embedding(5, 4)
        self.pos_goal_min = nn.Embedding(5, 4)
        self.pos_assist_min = nn.Embedding(5, 4)
        for layer in (self.goal, self.assist):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for table in (self.pos_goal, self.pos_assist, self.pos_goal_min, self.pos_assist_min):
            nn.init.zeros_(table.weight)
        nn.init.zeros_(self.cross.out_proj.weight)
        nn.init.zeros_(self.cross.out_proj.bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        cells = self._cells(
            batch["minutes"], batch["present"], batch["position"], batch["row_pad"],
            batch["goal_token"], batch["assist_token"],
        )
        cells = cells + self.home(batch["home"][:, None, :, None])
        week = batch["gw"].clamp(0, COLS)
        cells = cells + self.gw(week)[:, None, :, :]
        batch_size, _rows, cols, _dim = cells.shape
        row_view = cells.permute(0, 2, 1, 3).reshape(batch_size * cols, ROWS, D)
        row_pad = batch["row_pad"][:, None, :].expand(batch_size, cols, ROWS).reshape(batch_size * cols, ROWS)
        mixed, _ = self.row(row_view, row_view, row_view, key_padding_mask=row_pad, need_weights=False)
        row_view = self.norm_row(row_view + mixed)
        col_view = row_view.reshape(batch_size, cols, ROWS, D).permute(0, 2, 1, 3).reshape(batch_size * ROWS, cols, D)
        col_pad = (batch["column_on"] < 0.5)[:, None, :].expand(batch_size, ROWS, cols)
        col_pad = col_pad.reshape(batch_size * ROWS, cols)
        mixed, _ = self.col(col_view, col_view, col_view, key_padding_mask=col_pad, need_weights=False)
        col_view = self.norm_col(col_view + mixed)
        ours = col_view.reshape(batch_size, ROWS, cols, D).permute(0, 2, 1, 3).reshape(batch_size * cols, ROWS, D)
        other = self._cells(
            batch["opp_minutes"], batch["opp_present"], batch["opp_position"], batch["opp_pad"],
            batch["opp_goal_token"], batch["opp_assist_token"],
        )
        other = other.permute(0, 2, 1, 3, 4).reshape(batch_size * cols, ROWS * SLOTS, D)
        pad = batch["opp_pad"].permute(0, 2, 1, 3).reshape(batch_size * cols, ROWS * SLOTS)
        empty = pad.all(dim=-1)
        safe = pad.clone()
        safe[empty, 0] = False
        mixed, _ = self.cross(ours, other, other, key_padding_mask=safe, need_weights=False)
        mixed = mixed.masked_fill(empty[:, None, None], 0.0)
        out = self.norm_cross(ours + mixed).reshape(batch_size, cols, ROWS, D).permute(0, 2, 1, 3)
        pos = torch.where(batch["position"] < 0, torch.full_like(batch["position"], 4), batch["position"])
        played = (batch["minutes"] / 90.0).unsqueeze(-1)
        goal = self.goal(out) + self.pos_goal(pos)[:, :, None, :] + self.pos_goal_min(pos)[:, :, None, :] * played
        assist = (
            self.assist(out)
            + self.pos_assist(pos)[:, :, None, :]
            + self.pos_assist_min(pos)[:, :, None, :] * played
        )
        return goal, assist

    def _cells(self, minutes, present, position, pad, goal_token, assist_token):
        stats = torch.stack([minutes / 90.0, present], dim=-1)
        pos = torch.where(position < 0, torch.full_like(position, 4), position)
        pos = torch.where(pad, torch.full_like(pos, 4), pos)
        pos_emb = self.position(pos)
        cells = self.cell(stats)
        if pos_emb.ndim + 1 == cells.ndim:
            pos_emb = pos_emb.unsqueeze(-2)
        return cells + pos_emb + self.goal_in(goal_token) + self.assist_in(assist_token)


def _nll(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1), reduction="none").reshape(labels.shape)
    return (flat * mask).sum() / mask.sum().clamp_min(1.0)


def train_pair(
    model: PairSequence,
    cards: list[SquadCard],
    squads: dict[tuple[str, str], SquadCard],
    rng: np.random.Generator,
    epochs: int = EPOCHS,
    lr: float = LR,
    batch_size: int = BATCH,
) -> list[float]:
    fast = [p for name, p in model.named_parameters() if "pos_" in name or "position" in name]
    rest = [p for p in model.parameters() if id(p) not in {id(item) for item in fast}]
    opt = torch.optim.AdamW(
        [
            {"params": rest, "lr": lr, "weight_decay": 1e-4},
            {"params": fast, "lr": lr * 10, "weight_decay": 0.0},
        ]
    )
    history = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), batch_size):
            pick = [cards[int(i)] for i in order[start : start + batch_size]]
            views = [pair_view(card, sample_cut(card, rng), squads) for card in pick]
            batch = _stack(views)
            opt.zero_grad()
            goal_logits, assist_logits = model(batch)
            mask = batch["loss"]
            loss = 0.5 * (
                _nll(goal_logits, batch["goal_label"], mask) + _nll(assist_logits, batch["assist_label"], mask)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(mask.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        epoch_nll = total / max(seen, 1.0)
        history.append(epoch_nll)
        print(f"pair epoch {epoch + 1} train nll {epoch_nll:.3f}", flush=True)
    return history


@torch.no_grad()
def predict_pair(
    model: PairSequence,
    cards: list[SquadCard],
    squads: dict[tuple[str, str], SquadCard],
    cuts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    goals = np.zeros((len(cards), ROWS, COLS, 4), dtype=np.float64)
    assists = np.zeros_like(goals)
    for start in range(0, len(cards), BATCH):
        stop = min(start + BATCH, len(cards))
        views = [pair_view(cards[i], cuts[i], squads) for i in range(start, stop)]
        goal_logits, assist_logits = model(_stack(views))
        goals[start:stop] = F.softmax(goal_logits, dim=-1).numpy()
        assists[start:stop] = F.softmax(assist_logits, dim=-1).numpy()
    return goals, assists


def _report(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    print("loading paired cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    squads = squad_index(train + test)
    print(f"train {len(train)} test {len(test)}", flush=True)
    torch.manual_seed(SEED)
    model = PairSequence(dropout=0.1)
    train_pair(model, train, squad_index(train), np.random.default_rng(SEED), EPOCHS)
    print("scoring", flush=True)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    fixed = [CUT_GW] * len(test)
    goal_probs, assist_probs = predict_pair(model, test, squads, fixed)
    capture = capture_table(test, {"pair": 1.0 - goal_probs[:, :, :, 0]}, CUT_GW)
    bar_rng = np.random.default_rng(SEED)
    lines = [
        "# Paired sequence trial",
        "",
        "Each hidden week is read with the other squad's match week and the two",
        "gameweeks before it. Minutes stay visible. Goals and assists after the",
        "cut stay masked on both cards.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        f"Cards: train {len(train)}, test {len(test)}.",
        f"Gameweek-{CUT_GW} top 10/20/50 on all appearances: "
        + ", ".join(f"{value:.1%}" for value in capture["pair"]),
        f"Players in that ranking: {capture['n_players']}.",
        "",
        "## Remainders from gameweek 1",
        "",
        "Both clubs are scored. A bar is an appearance of at least one event.",
        "Drawn bars are the mean of 20 samples.",
        "",
        "| revealed | channel | actual bars | drawn bars | covered | on a real week | GK | FWD |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for revealed in (0.8, 0.2):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        probs_g, probs_a = predict_pair(model, test, squads, cuts)
        rates = positive_by_position(probs_g, test, 0, cuts)
        for channel, probs, channel_labels in (
            ("goals", probs_g, labels),
            ("assists", probs_a, assist_labels),
        ):
            bars = bar_correspondence(test, probs, channel_labels, 0, bar_rng, cuts=cuts)
            lines.append(
                f"| {revealed:.0%} | {channel} | {bars['actual_bars']} | {bars['drawn_bars']:.1f} | "
                f"{bars['actual_covered']:.1%} | {bars['drawn_on_actual']:.1%} | "
                f"{rates[POS['GKP']]:.1%} | {rates[POS['FWD']]:.1%} |"
            )
            print(
                f"reveal {revealed:.0%} {channel} actual {bars['actual_bars']} "
                f"drawn {bars['drawn_bars']:.1f} covered {bars['actual_covered']:.3f} "
                f"on_week {bars['drawn_on_actual']:.3f} gk {rates[0]:.3f} fwd {rates[3]:.3f}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    lines.extend(["", f"Wall clock: {seconds:.0f} seconds on CPU.", ""])
    _report(lines, ROOT_REPORTS / "flashcard_pair_trial.md")
    print(f"done in {seconds:.0f}s capture10 {capture['pair'][0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
