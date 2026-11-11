"""Three ways to paint the rest of the card, on the squares themselves.

MaskGIT reads the whole grid and fills the masked goal and assist squares
in a few passes. The column raster paints one gameweek at a time, left to
right. VAR paints a coarse map and then each finer map. A finished card is
one draw at temperature 1. The stored gate stays the 21:47 UK full painter.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from src.models.flashcard_card_score import tally_cards
from src.models.flashcard_odds_trial import column_odds, load_odds
from src.models.flashcard_squad_trial import (
    BATCH,
    COLS,
    D,
    EPOCHS,
    HEADS,
    LR,
    MASK_ID,
    ROWS,
    SEED,
    _nll,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    sample_cut,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
SCALES = (5, 10, 20, 40)
MASKGIT_STEPS = 8
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


def cosine_ratio(progress: float) -> float:
    """Share of target squares that stay masked. 0 is the start, 1 is the end."""
    return float(np.cos(np.clip(progress, 0.0, 1.0) * np.pi / 2.0))


def max_pool(values: np.ndarray, side: int) -> np.ndarray:
    """Max inside each block. `side` divides both grid axes."""
    block = ROWS // side
    folded = np.asarray(values).reshape(values.shape[0], side, block, side, block)
    return folded.max(axis=2).max(axis=3)


def any_pool(mask: np.ndarray, side: int) -> np.ndarray:
    block = ROWS // side
    folded = np.asarray(mask).reshape(mask.shape[0], side, block, side, block)
    return folded.any(axis=2).any(axis=3)


def repeat_up(values: np.ndarray) -> np.ndarray:
    """Double a square map along both axes."""
    return np.repeat(np.repeat(values, 2, axis=1), 2, axis=2)


def context_tokens(labels: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """True counts on revealed squares. Hidden squares carry the mask id."""
    context = np.array(labels, copy=True)
    context[hidden] = MASK_ID
    return context


def still_masked(confidence: np.ndarray, unknown: np.ndarray, step: int, steps: int, original: int) -> np.ndarray:
    """Keep the least sure of the still-open squares masked.

    `confidence` and `unknown` are flat over one card. `original` is how many
    squares were open at the start.
    """
    stay = int(np.floor(cosine_ratio((step + 1) / steps) * original))
    out = np.array(unknown, copy=True)
    open_idx = np.flatnonzero(unknown)
    if len(open_idx) == 0 or stay >= len(open_idx):
        return out
    order = open_idx[np.argsort(confidence[open_idx])]
    commit = order if stay <= 0 else order[stay:]
    out[commit] = False
    return out


def _column_features(card, odds: np.ndarray, cut: int) -> np.ndarray:
    missing = np.isnan(odds).any(axis=1)
    filled = np.nan_to_num(odds, nan=0.0)
    visible = ((card.gw > 0) & (card.gw <= cut)).astype(np.float32)
    home = card.home.astype(np.float32)
    return np.concatenate(
        [
            home[:, None],
            filled.astype(np.float32),
            missing.astype(np.float32)[:, None],
            visible[:, None],
        ],
        axis=1,
    ).astype(np.float32)


def pack_card(card, cut: int, clubs: dict[str, int], odds_table: dict) -> dict[str, np.ndarray]:
    """One cut of one club. Future goal and assist squares are masked."""
    hidden_cols = (card.gw > cut) & (card.gw > 0)
    hidden = card.present & hidden_cols.reshape(1, -1)
    position = np.where(card.position < 0, 4, card.position).astype(np.int64)
    opponent = np.array([clubs.get(name, 0) if name else 0 for name in card.opponent], dtype=np.int64)
    return {
        "goals": card.goals.astype(np.int64),
        "assists": card.assists.astype(np.int64),
        "hidden": hidden,
        "present": card.present.astype(bool),
        "position": position,
        "minutes": card.minutes.astype(np.float32),
        "opponent": opponent,
        "column_on": (card.gw > 0),
        "features": _column_features(card, column_odds(card, odds_table), cut),
        "row_pad": card.position < 0,
    }


def _stack(packs: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key in packs[0]:
        value = np.stack([pack[key] for pack in packs])
        tensor = torch.from_numpy(value)
        out[key] = tensor.to(device)
    return out


class TokenGrid(nn.Module):
    """Read a grid of goal and assist squares.

    Column attention is causal when `causal` is set, so a gameweek can see
    the weeks to its left and not the weeks to its right.
    """

    def __init__(self, n_clubs: int, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.goal_tok = nn.Embedding(MASK_ID + 1, D)
        self.assist_tok = nn.Embedding(MASK_ID + 1, D)
        self.parent_goal = nn.Embedding(MASK_ID + 1, D)
        self.parent_assist = nn.Embedding(MASK_ID + 1, D)
        self.position = nn.Embedding(5, D)
        self.opponent = nn.Embedding(n_clubs, D)
        self.minute = nn.Linear(1, D)
        self.column = nn.Linear(6, D)
        self.row = nn.MultiheadAttention(D, HEADS, dropout=0.1, batch_first=True)
        self.col = nn.MultiheadAttention(D, HEADS, dropout=0.1, batch_first=True)
        self.norm_row = nn.LayerNorm(D)
        self.norm_col = nn.LayerNorm(D)
        self.goal = nn.Linear(D, 4)
        self.assist = nn.Linear(D, 4)

    def forward(
        self,
        goal_tok: torch.Tensor,
        assist_tok: torch.Tensor,
        parent_goal: torch.Tensor,
        parent_assist: torch.Tensor,
        position: torch.Tensor,
        minutes: torch.Tensor,
        opponent: torch.Tensor,
        features: torch.Tensor,
        row_pad: torch.Tensor,
        col_pad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, rows, cols = goal_tok.shape
        cells = (
            self.goal_tok(goal_tok)
            + self.assist_tok(assist_tok)
            + self.parent_goal(parent_goal)
            + self.parent_assist(parent_assist)
            + self.position(position)[:, :, None, :]
            + self.minute((minutes / 90.0).unsqueeze(-1))
            + self.opponent(opponent)[:, None, :, :]
            + self.column(features)[:, None, :, :]
        )
        row_view = cells.permute(0, 2, 1, 3).reshape(batch * cols, rows, D)
        row_mask = row_pad[:, None, :].expand(batch, cols, rows).reshape(batch * cols, rows)
        mixed, _ = self.row(row_view, row_view, row_view, key_padding_mask=row_mask, need_weights=False)
        row_view = self.norm_row(row_view + mixed)
        col_view = row_view.reshape(batch, cols, rows, D).permute(0, 2, 1, 3).reshape(batch * rows, cols, D)
        col_mask = col_pad[:, None, :].expand(batch, rows, cols).reshape(batch * rows, cols)
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(torch.ones(cols, cols, dtype=torch.bool, device=cells.device), diagonal=1)
        mixed, _ = self.col(
            col_view, col_view, col_view, key_padding_mask=col_mask, attn_mask=attn_mask, need_weights=False
        )
        out = self.norm_col(col_view + mixed).reshape(batch, rows, cols, D)
        return self.goal(out), self.assist(out)


def _blank_parent(goal_tok: torch.Tensor) -> torch.Tensor:
    return torch.full_like(goal_tok, MASK_ID)


def _draw(logits: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """One temperature-1 draw, and the probability of the drawn class."""
    probs = torch.softmax(logits, dim=-1)
    flat = probs.reshape(-1, 4)
    choice = torch.multinomial(flat, 1, generator=generator).reshape(probs.shape[:-1])
    picked = probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
    return choice, picked


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


def _maskgit_inputs(batch: dict[str, torch.Tensor], rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hide a cosine share of the future squares. Loss sits on the hidden share."""
    hidden = batch["hidden"]
    ratio = cosine_ratio(float(rng.random()))
    drop = torch.from_numpy(rng.random(hidden.shape) < ratio).to(hidden.device) & hidden
    if not bool(drop.any()):
        drop = hidden
    goal = torch.where(drop, torch.full_like(batch["goals"], MASK_ID), batch["goals"])
    assist = torch.where(drop, torch.full_like(batch["assists"], MASK_ID), batch["assists"])
    revealed = batch["present"] & ~batch["hidden"]
    goal = torch.where(revealed, batch["goals"], goal)
    assist = torch.where(revealed, batch["assists"], assist)
    goal = torch.where(batch["present"], goal, torch.zeros_like(goal))
    assist = torch.where(batch["present"], assist, torch.zeros_like(assist))
    return goal, assist, drop


def train_maskgit(model: TokenGrid, cards: list, clubs: dict, odds: dict, rng: np.random.Generator, device: torch.device) -> None:
    opt = _optimizer(model)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            packs = [pack_card(card, sample_cut(card, rng), clubs, odds) for card in pick]
            batch = _stack(packs, device)
            goal_tok, assist_tok, drop = _maskgit_inputs(batch, rng)
            opt.zero_grad()
            goal_logits, assist_logits = model(
                goal_tok, assist_tok, _blank_parent(goal_tok), _blank_parent(assist_tok),
                batch["position"], batch["minutes"], batch["opponent"], batch["features"],
                batch["row_pad"], ~batch["column_on"],
            )
            loss = 0.5 * (
                _nll(goal_logits, batch["goals"], drop.float()) + _nll(assist_logits, batch["assists"], drop.float())
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(drop.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        print(f"maskgit epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f}", flush=True)


@torch.no_grad()
def paint_maskgit(model: TokenGrid, cards: list, clubs: dict, odds: dict, cuts: list[int], device: torch.device, generator: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    goals = np.zeros((len(cards), ROWS, COLS), dtype=np.int64)
    assists = np.zeros_like(goals)
    for start in range(0, len(cards), BATCH):
        pick = cards[start : start + BATCH]
        these = cuts[start : start + BATCH]
        packs = [pack_card(card, cut, clubs, odds) for card, cut in zip(pick, these, strict=True)]
        batch = _stack(packs, device)
        hidden = batch["hidden"]
        original = hidden.flatten(1).sum(dim=1)
        goal = context_from_batch(batch, "goals")
        assist = context_from_batch(batch, "assists")
        unknown = hidden.clone()
        for step in range(MASKGIT_STEPS):
            goal_logits, assist_logits = model(
                goal, assist, _blank_parent(goal), _blank_parent(assist),
                batch["position"], batch["minutes"], batch["opponent"], batch["features"],
                batch["row_pad"], ~batch["column_on"],
            )
            drawn_g, conf_g = _draw(goal_logits, generator)
            drawn_a, conf_a = _draw(assist_logits, generator)
            confidence = ((conf_g + conf_a) * 0.5).cpu().numpy()
            unknown_np = unknown.cpu().numpy()
            goal_np = goal.cpu().numpy()
            assist_np = assist.cpu().numpy()
            drawn_g_np = drawn_g.cpu().numpy()
            drawn_a_np = drawn_a.cpu().numpy()
            for index in range(len(pick)):
                flat_unknown = unknown_np[index].reshape(-1)
                stay = still_masked(
                    confidence[index].reshape(-1), flat_unknown, step, MASKGIT_STEPS, int(original[index])
                )
                commit = flat_unknown & ~stay
                goal_flat = goal_np[index].reshape(-1)
                assist_flat = assist_np[index].reshape(-1)
                goal_flat[commit] = drawn_g_np[index].reshape(-1)[commit]
                assist_flat[commit] = drawn_a_np[index].reshape(-1)[commit]
                unknown_np[index] = stay.reshape(ROWS, COLS)
            goal = torch.from_numpy(goal_np).to(device)
            assist = torch.from_numpy(assist_np).to(device)
            unknown = torch.from_numpy(unknown_np).to(device)
        goals[start : start + len(pick)] = _painted(goal.cpu().numpy(), hidden.cpu().numpy())
        assists[start : start + len(pick)] = _painted(assist.cpu().numpy(), hidden.cpu().numpy())
    return goals, assists


def context_from_batch(batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    revealed = batch["present"] & ~batch["hidden"]
    context = torch.where(revealed, batch[key], torch.full_like(batch[key], MASK_ID))
    return torch.where(batch["present"], context, torch.zeros_like(context))


def _painted(tokens: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    painted = np.zeros(tokens.shape, dtype=np.int64)
    painted[hidden] = tokens[hidden]
    return painted


def train_column(model: TokenGrid, cards: list, clubs: dict, odds: dict, rng: np.random.Generator, device: torch.device) -> None:
    opt = _optimizer(model)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            packs = [pack_card(card, sample_cut(card, rng), clubs, odds) for card in pick]
            batch = _stack(packs, device)
            goal = context_from_batch(batch, "goals")
            assist = context_from_batch(batch, "assists")
            opt.zero_grad()
            goal_logits, assist_logits = model(
                goal, assist, _blank_parent(goal), _blank_parent(assist),
                batch["position"], batch["minutes"], batch["opponent"], batch["features"],
                batch["row_pad"], ~batch["column_on"],
            )
            drop = batch["hidden"].float()
            loss = 0.5 * (
                _nll(goal_logits, batch["goals"], drop) + _nll(assist_logits, batch["assists"], drop)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(drop.sum().detach())
            total += float(loss.detach()) * weight
            seen += weight
        print(f"column epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f}", flush=True)


@torch.no_grad()
def paint_column(model: TokenGrid, cards: list, clubs: dict, odds: dict, cuts: list[int], device: torch.device, generator: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    goals = np.zeros((len(cards), ROWS, COLS), dtype=np.int64)
    assists = np.zeros_like(goals)
    for start in range(0, len(cards), BATCH):
        pick = cards[start : start + BATCH]
        these = cuts[start : start + BATCH]
        packs = [pack_card(card, cut, clubs, odds) for card, cut in zip(pick, these, strict=True)]
        batch = _stack(packs, device)
        goal = context_from_batch(batch, "goals")
        assist = context_from_batch(batch, "assists")
        hidden = batch["hidden"]
        for col in range(COLS):
            if not bool(hidden[:, :, col].any()):
                continue
            goal_logits, assist_logits = model(
                goal, assist, _blank_parent(goal), _blank_parent(assist),
                batch["position"], batch["minutes"], batch["opponent"], batch["features"],
                batch["row_pad"], ~batch["column_on"],
            )
            drawn_g, _ = _draw(goal_logits[:, :, col], generator)
            drawn_a, _ = _draw(assist_logits[:, :, col], generator)
            take = hidden[:, :, col]
            goal[:, :, col] = torch.where(take, drawn_g, goal[:, :, col])
            assist[:, :, col] = torch.where(take, drawn_a, assist[:, :, col])
        goals[start : start + len(pick)] = _painted(goal.cpu().numpy(), hidden.cpu().numpy())
        assists[start : start + len(pick)] = _painted(assist.cpu().numpy(), hidden.cpu().numpy())
    return goals, assists


def _pool_pack(pack: dict[str, np.ndarray], side: int, parent_goal: np.ndarray | None, parent_assist: np.ndarray | None) -> dict[str, np.ndarray]:
    """One card, pooled to `side` by `side`. A block is hidden if any square in it is."""
    block = ROWS // side
    goals = pack["goals"][None]
    assists = pack["assists"][None]
    hidden = any_pool(pack["hidden"][None], side)[0]
    present = any_pool(pack["present"][None], side)[0]
    goal_labels = max_pool(goals, side)[0]
    assist_labels = max_pool(assists, side)[0]
    if parent_goal is None:
        parent_g = np.full((side, side), MASK_ID, dtype=np.int64)
        parent_a = np.full((side, side), MASK_ID, dtype=np.int64)
    else:
        parent_g = repeat_up(parent_goal[None])[0]
        parent_a = repeat_up(parent_assist[None])[0]
    position = pack["position"].reshape(side, block).max(axis=1)
    minutes = pack["minutes"].reshape(side, block, side, block).max(axis=1).max(axis=2)
    opponent = pack["opponent"].reshape(side, block)[:, 0]
    features = pack["features"].reshape(side, block, pack["features"].shape[-1]).mean(axis=1)
    column_on = pack["column_on"].reshape(side, block).any(axis=1)
    row_pad = position < 0
    goal_tok = context_tokens(goal_labels, hidden)
    assist_tok = context_tokens(assist_labels, hidden)
    goal_tok = np.where(present, goal_tok, 0)
    assist_tok = np.where(present, assist_tok, 0)
    return {
        "goals": goal_labels.astype(np.int64),
        "assists": assist_labels.astype(np.int64),
        "hidden": hidden,
        "goal_tok": goal_tok.astype(np.int64),
        "assist_tok": assist_tok.astype(np.int64),
        "parent_goal": parent_g.astype(np.int64),
        "parent_assist": parent_a.astype(np.int64),
        "position": position.astype(np.int64),
        "minutes": minutes.astype(np.float32),
        "opponent": opponent.astype(np.int64),
        "features": features.astype(np.float32),
        "row_pad": row_pad,
        "column_on": column_on,
    }


def _loss_scale(model: TokenGrid, packs: list[dict[str, np.ndarray]], device: torch.device) -> torch.Tensor:
    batch = _stack(packs, device)
    goal_logits, assist_logits = model(
        batch["goal_tok"], batch["assist_tok"], batch["parent_goal"], batch["parent_assist"],
        batch["position"], batch["minutes"], batch["opponent"], batch["features"],
        batch["row_pad"], ~batch["column_on"],
    )
    drop = batch["hidden"].float()
    return 0.5 * (
        _nll(goal_logits, batch["goals"], drop) + _nll(assist_logits, batch["assists"], drop)
    )


def train_var(model: TokenGrid, cards: list, clubs: dict, odds: dict, rng: np.random.Generator, device: torch.device) -> None:
    opt = _optimizer(model)
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            full = [pack_card(card, sample_cut(card, rng), clubs, odds) for card in pick]
            opt.zero_grad()
            loss = torch.zeros((), device=device)
            weight = 0.0
            for side in SCALES:
                parent_g = None if side == SCALES[0] else max_pool(np.stack([pack["goals"] for pack in full]), side // 2)
                parent_a = None if side == SCALES[0] else max_pool(np.stack([pack["assists"] for pack in full]), side // 2)
                pooled = []
                for index, pack in enumerate(full):
                    goal_parent = None if parent_g is None else parent_g[index]
                    assist_parent = None if parent_a is None else parent_a[index]
                    pooled.append(_pool_pack(pack, side, goal_parent, assist_parent))
                scale_loss = _loss_scale(model, pooled, device)
                loss = loss + scale_loss
                weight += float(torch.stack([torch.from_numpy(item["hidden"]) for item in pooled]).sum())
            (loss / len(SCALES)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) / len(SCALES) * max(weight, 1.0)
            seen += weight
        print(f"var epoch {epoch + 1} train nll {total / max(seen, 1.0):.3f}", flush=True)


@torch.no_grad()
def paint_var(model: TokenGrid, cards: list, clubs: dict, odds: dict, cuts: list[int], device: torch.device, generator: torch.Generator) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    goals = np.zeros((len(cards), ROWS, COLS), dtype=np.int64)
    assists = np.zeros_like(goals)
    for start in range(0, len(cards), BATCH):
        pick = cards[start : start + BATCH]
        these = cuts[start : start + BATCH]
        full = [pack_card(card, cut, clubs, odds) for card, cut in zip(pick, these, strict=True)]
        parent_g = None
        parent_a = None
        sampled_g = None
        sampled_a = None
        for side in SCALES:
            pooled = [
                _pool_pack(
                    pack,
                    side,
                    None if parent_g is None else parent_g[index],
                    None if parent_a is None else parent_a[index],
                )
                for index, pack in enumerate(full)
            ]
            batch = _stack(pooled, device)
            goal_logits, assist_logits = model(
                batch["goal_tok"], batch["assist_tok"], batch["parent_goal"], batch["parent_assist"],
                batch["position"], batch["minutes"], batch["opponent"], batch["features"],
                batch["row_pad"], ~batch["column_on"],
            )
            drawn_g, _ = _draw(goal_logits, generator)
            drawn_a, _ = _draw(assist_logits, generator)
            hidden = batch["hidden"]
            sampled_g = torch.where(hidden, drawn_g, batch["goals"]).cpu().numpy()
            sampled_a = torch.where(hidden, drawn_a, batch["assists"]).cpu().numpy()
            parent_g = sampled_g
            parent_a = sampled_a
        goals[start : start + len(pick)] = _painted(sampled_g, np.stack([pack["hidden"] for pack in full]))
        assists[start : start + len(pick)] = _painted(sampled_a, np.stack([pack["hidden"] for pack in full]))
    return goals, assists


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


def main() -> None:
    started = time.perf_counter()
    device = torch.device("cpu")
    print("loading generator cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    odds = load_odds()
    n_clubs = max(clubs.values(), default=0) + 1
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    lines = [
        "# Card generators",
        "",
        "Three painters train on the squares of the card. The known prefix",
        "keeps its goal and assist squares. Odds, venue, opponent, minutes,",
        "and position stay visible, including on a future gameweek. Each",
        "finished card is one draw at temperature 1. The other squad is not",
        "an input.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    compared: dict[str, list[str]] = {}
    runners = (
        ("maskgit", False, train_maskgit, paint_maskgit),
        ("column", True, train_column, paint_column),
        ("var", False, train_var, paint_var),
    )
    for name, causal, train_fn, paint_fn in runners:
        print(f"training {name}", flush=True)
        torch.manual_seed(SEED)
        model = TokenGrid(n_clubs, causal=causal).to(device)
        train_fn(model, train, clubs, odds, np.random.default_rng(SEED), device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(SEED)
        rows: list[str] = []
        for revealed in FRACTIONS:
            cuts = [cut_at_fraction(card, revealed) for card in test]
            painted_g, painted_a = paint_fn(model, test, clubs, odds, cuts, device, generator)
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
    seconds = time.perf_counter() - started
    lines.extend(
        [
            "",
            "MaskGIT fills the open squares in eight passes, keeping the sure",
            "ones each time. The column raster paints the next gameweek from",
            "the weeks on its left. VAR paints scales of 5, 10, 20, and 40,",
            "and a coarse square is the highest count in its block. The stored",
            "gate is the 21:47 UK full painter. These runs do not replace it.",
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
    lines.extend([f"Wall clock: {seconds:.0f} seconds on CPU.", ""])
    path = ROOT_REPORTS / "flashcard_generator_trial.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
