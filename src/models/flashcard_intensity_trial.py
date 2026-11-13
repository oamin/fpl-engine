"""Trial owned event totals before placing bars.

Proposal 1: each player owns a rate μ, then the weeks share that mass.
Proposal 2 is built in: presence chance is separate from bar height.
Proposal 3: each fixture owns a rate μ, then the squad shares that mass.
Proposal 4: train against a straight-through round of the row count.

Assists keep the old four-way head. The card is the 72-pixel legacy header.
Two inits are scored: native seed 0, and the body draw left by a 76-pixel header.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flashcard_card_score import _chance_line, _paint, tally_cards
from src.models.flashcard_legacy_card import HEADER_H as LEGACY_HEADER
from src.models.flashcard_legacy_card import compose_legacy
from src.models.flashcard_odds_trial import load_odds
from src.models.flashcard_squad_trial import (
    BATCH,
    CELL_W,
    COLS,
    D,
    EPOCHS,
    HEADS,
    ROWS,
    SEED,
    _nll,
    _stack,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    sample_cut,
    squad_index,
)
from src.models.flashcard_square_trial import PLAYER_H
from src.models.flashcard_strip_trial import (
    StripVision,
    _batch_images,
    _optimizer,
    _square_bands,
    strip_features,
    train_stripped,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
OUT = ROOT_REPORTS / "intensity_trial.md"
WIDE_HEADER = LEGACY_HEADER + CELL_W


class IntensityReader(nn.Module):
    """Picture reader with an owned goal rate, then a share across the mask."""

    def __init__(self, header_h: int, mode: str, dropout: float = 0.1, player_h: int = PLAYER_H) -> None:
        super().__init__()
        if mode not in {"row", "row_ste", "fixture"}:
            raise ValueError(f"unknown intensity mode {mode}")
        self.mode = mode
        self.header_h = int(header_h)
        self.player_h = int(player_h)
        self.header = nn.Linear(3 * self.header_h * CELL_W, D)
        self.band_h = _square_bands(self.player_h)
        self.bands = nn.ModuleList(nn.Linear(3 * height * CELL_W, D) for height in self.band_h)
        self.band_norm = nn.ModuleList(nn.LayerNorm(D) for _ in self.band_h)
        self.header_norm = nn.LayerNorm(D)
        self.row = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.col = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_row = nn.LayerNorm(D)
        self.norm_col = nn.LayerNorm(D)
        self.col_pos = nn.Parameter(torch.zeros(1, 1, COLS, D))
        nn.init.normal_(self.col_pos, std=0.02)
        self.goal_score = nn.Linear(D, 1)
        self.goal_height = nn.Linear(D, 3)
        self.goal_mu = nn.Linear(D, 1)
        self.skip_score = nn.Linear(D, 1)
        self.skip_height = nn.Linear(D, 3)
        self.assist = nn.Linear(D, 4)
        self.skip_assist = nn.Linear(D, 4)
        for layer in (
            self.goal_score,
            self.goal_height,
            self.goal_mu,
            self.skip_score,
            self.skip_height,
            self.assist,
            self.skip_assist,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode(
        self,
        images: torch.Tensor,
        row_pad: torch.Tensor,
        column_on: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        out = col_view.reshape(batch, ROWS, COLS, D)
        return out, cells, header_tok

    def forward(
        self,
        images: torch.Tensor,
        row_pad: torch.Tensor,
        column_on: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out, cells, header_tok = self.encode(images, row_pad, column_on)
        score = (self.goal_score(out) + self.skip_score(cells)).squeeze(-1)
        height = self.goal_height(out) + self.skip_height(cells)
        assist = self.assist(out) + self.skip_assist(cells)
        mask = loss_mask if loss_mask is not None else column_on[:, None, :].expand(-1, ROWS, -1)
        mask = mask * (~row_pad)[:, :, None].to(mask.dtype)
        if self.mode == "fixture":
            pooled = header_tok
            live = column_on > 0.5
            mu = F.softplus(self.goal_mu(pooled).squeeze(-1))
            mu = mu * live.to(mu.dtype)
            logits = score.masked_fill(mask <= 0, -1e9)
            share = torch.softmax(logits.transpose(1, 2), dim=-1).transpose(1, 2)
            share = torch.where(mask > 0, share, torch.zeros_like(share))
            presence = (mu[:, None, :] * share).clamp(0.0, 1.0)
            row_mu = presence.sum(dim=-1)
        else:
            live = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            pooled = (out * mask.unsqueeze(-1)).sum(dim=2) / live
            mu = F.softplus(self.goal_mu(pooled).squeeze(-1))
            mu = mu * (mask.sum(dim=-1) > 0).to(mu.dtype)
            logits = score.masked_fill(mask <= 0, -1e9)
            share = torch.softmax(logits, dim=-1)
            share = torch.where(mask > 0, share, torch.zeros_like(share))
            presence = (mu.unsqueeze(-1) * share).clamp(0.0, 1.0)
            row_mu = mu
        height_p = torch.softmax(height, dim=-1)
        goal = torch.zeros(out.shape[0], ROWS, COLS, 4, device=out.device, dtype=out.dtype)
        goal[..., 0] = 1.0 - presence
        goal[..., 1:] = presence.unsqueeze(-1) * height_p
        return {
            "goal_logits": goal.clamp_min(1e-8).log(),
            "assist_logits": assist,
            "presence": presence,
            "row_mu": row_mu,
            "height_logits": height,
            "goal_probs": goal,
        }


def intensity_goal_loss(
    pack: dict[str, torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    ste: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Poisson on the owned rate, presence NLL, and height on bars."""
    presence = pack["presence"]
    row_mu = pack["row_mu"]
    actual = ((labels > 0).to(presence.dtype) * mask).sum(dim=-1)
    live = mask.sum(dim=-1) > 0
    rate = row_mu.clamp_min(1e-8)
    if ste:
        soft = presence.sum(dim=-1)
        painted = soft + (soft.round() - soft).detach()
        count_term = (painted - actual).square()
    else:
        count_term = rate - actual * rate.log()
    count_loss = (count_term * live).sum() / live.sum().clamp_min(1.0)
    target = (labels > 0).to(presence.dtype)
    p = presence.clamp(1e-6, 1.0 - 1e-6)
    cell = -(target * p.log() + (1.0 - target) * (1.0 - p).log())
    cell_loss = (cell * mask).sum() / mask.sum().clamp_min(1.0)
    hit = mask.bool() & (labels > 0)
    if bool(hit.any()):
        height_target = labels.clamp(min=1, max=3).long() - 1
        height_loss = F.cross_entropy(pack["height_logits"][hit], height_target[hit])
    else:
        height_loss = presence.new_zeros(())
    return cell_loss + count_loss + height_loss, count_loss.detach()


def _copy_shifted_body(narrow: nn.Module, wide: nn.Module) -> int:
    copied = 0
    wide_params = dict(wide.named_parameters())
    with torch.no_grad():
        for name, param in narrow.named_parameters():
            other = wide_params.get(name)
            if other is None or param.shape != other.shape:
                continue
            param.copy_(other)
            copied += int(param.numel())
    return copied


def build_reader(mode: str, shifted: bool) -> nn.Module:
    torch.manual_seed(SEED)
    if shifted:
        wide = IntensityReader(WIDE_HEADER, mode=mode, player_h=PLAYER_H)
        torch.manual_seed(SEED)
        model = IntensityReader(LEGACY_HEADER, mode=mode, player_h=PLAYER_H)
        copied = _copy_shifted_body(model, wide)
        print(f"{mode} shifted body copied {copied}", flush=True)
        return model
    return IntensityReader(LEGACY_HEADER, mode=mode, player_h=PLAYER_H)


def build_baseline(shifted: bool) -> nn.Module:
    torch.manual_seed(SEED)
    if shifted:
        wide = StripVision(WIDE_HEADER, dropout=0.1, player_h=PLAYER_H)
        torch.manual_seed(SEED)
        model = StripVision(LEGACY_HEADER, dropout=0.1, player_h=PLAYER_H)
        copied = _copy_shifted_body(model, wide)
        print(f"baseline shifted body copied {copied}", flush=True)
        return model
    return StripVision(LEGACY_HEADER, dropout=0.1, player_h=PLAYER_H)


def train_intensity(
    model: IntensityReader,
    cards: list,
    league: dict,
    rng: np.random.Generator,
    clubs: dict,
    odds: dict,
    name: str,
) -> None:
    opt = _optimizer(model)
    squads = squad_index(cards)
    ste = model.mode == "row_ste"
    model.train()
    for epoch in range(EPOCHS):
        cell_total = 0.0
        count_total = 0.0
        seen = 0.0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [sample_cut(card, rng) for card in pick]
            views = [
                strip_features(apply_cut(card, league, cut, clubs, squads))
                for card, cut in zip(pick, cuts, strict=True)
            ]
            batch = _stack(views)
            batch["image"] = (
                torch.from_numpy(_batch_images(views, pick, odds, compose_legacy))
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
            )
            opt.zero_grad()
            pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
            goal_loss, count_loss = intensity_goal_loss(pack, batch["goal_label"], batch["loss"], ste)
            assist_loss = _nll(pack["assist_logits"], batch["assist_label"], batch["loss"])
            loss = 0.5 * (goal_loss + assist_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            weight = float(batch["loss"].sum().detach())
            cell_total += float(goal_loss.detach()) * weight
            count_total += float(count_loss) * weight
            seen += weight
        print(
            f"{name} epoch {epoch + 1} goal {cell_total / max(seen, 1.0):.3f} "
            f"count {count_total / max(seen, 1.0):.3f}",
            flush=True,
        )


def paint_intensity(
    model: nn.Module,
    cards: list,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict,
    cuts: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    views = [
        strip_features(apply_cut(card, league, cut, clubs, squads))
        for card, cut in zip(cards, cuts, strict=True)
    ]
    batch = _stack(views)
    batch["image"] = (
        torch.from_numpy(_batch_images(views, cards, odds, compose_legacy))
        .permute(0, 3, 1, 2)
        .float()
        .div(255.0)
    )
    model.eval()
    with torch.no_grad():
        if isinstance(model, IntensityReader):
            pack = model(batch["image"], batch["row_pad"], batch["column_on"], batch["loss"])
            goal = pack["goal_probs"].cpu().numpy()
            assist = F.softmax(pack["assist_logits"], dim=-1).cpu().numpy()
            row_mu = pack["row_mu"].cpu().numpy()
        else:
            goal_logits, assist_logits = model(batch["image"], batch["row_pad"], batch["column_on"])
            goal = F.softmax(goal_logits, dim=-1).cpu().numpy()
            assist = F.softmax(assist_logits, dim=-1).cpu().numpy()
            row_mu = (1.0 - goal[..., 0]) * batch["loss"].numpy()
            row_mu = row_mu.sum(axis=-1)
    return goal, assist, row_mu


def _position_order(cards: list, goal_p: np.ndarray, cuts: list[int]) -> str:
    line = _chance_line(cards, goal_p, cuts, "goal")
    return line.replace("goal chance: ", "")


def score_run(
    model: nn.Module,
    label: str,
    test: list,
    league: dict,
    clubs: dict,
    squads: dict,
    odds: dict,
) -> dict:
    cuts = [cut_at_fraction(card, 0.2) for card in test]
    labels = np.stack([card.goals for card in test])
    minutes = np.stack([card.minutes for card in test])
    goal_p, _assist_p, row_mu = paint_intensity(model, test, league, clubs, squads, odds, cuts)
    paint = _paint(goal_p, minutes, test, cuts)
    goals = tally_cards(test, labels, paint, 0, cuts)
    soft = 0.0
    mid = np.zeros(2)
    for index, card in enumerate(test):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        soft += float(((1.0 - goal_p[index, ..., 0]) * mask).sum())
        live = mask.sum(axis=1) > 0
        vals = row_mu[index, live]
        mid[0] += float(((vals > 0) & (vals < 0.5)).sum())
        mid[1] += float(live.sum())
    order = _position_order(test, goal_p, cuts)
    result = {
        "label": label,
        "painted": int(goals["painted"]),
        "exact": int(goals["exact"]),
        "soft": round(soft, 1),
        "rows_below_half": int(mid[0]),
        "live_rows": int(mid[1]),
        "chance": order,
        "gkp": order.split(",")[0].strip(),
    }
    print(
        f"{label} painted {result['painted']} soft {result['soft']} "
        f"exact {result['exact']} below_half {result['rows_below_half']}/{result['live_rows']} "
        f"{result['chance']}",
        flush=True,
    )
    return result


def pass_gate(native: dict, shifted: dict, actual: int = 730) -> bool:
    gap = abs(native["painted"] - shifted["painted"])
    near = abs(native["painted"] - actual) <= 80 and abs(shifted["painted"] - actual) <= 80
    return gap <= 50 and near


def main() -> None:
    started = time.perf_counter()
    print("loading intensity trial", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(train + test)
    odds = load_odds()
    rows: list[dict] = []

    for shifted, tag in ((False, "native"), (True, "shifted")):
        model = build_baseline(shifted)
        train_stripped(
            model,
            train,
            league,
            np.random.default_rng(SEED),
            clubs,
            odds,
            f"baseline-{tag}",
            compose_legacy,
        )
        rows.append(score_run(model, f"baseline/{tag}", test, league, clubs, squads, odds))

    modes = (
        ("row", "1_row_intensity"),
        ("fixture", "3_fixture_intensity"),
        ("row_ste", "1plus4_row_ste"),
    )
    for mode, title in modes:
        pair: list[dict] = []
        for shifted, tag in ((False, "native"), (True, "shifted")):
            model = build_reader(mode, shifted)
            train_intensity(
                model,
                train,
                league,
                np.random.default_rng(SEED),
                clubs,
                odds,
                f"{title}-{tag}",
            )
            result = score_run(model, f"{title}/{tag}", test, league, clubs, squads, odds)
            rows.append(result)
            pair.append(result)
        print(
            f"gate {title}: gap {abs(pair[0]['painted'] - pair[1]['painted'])} "
            f"pass {pass_gate(pair[0], pair[1])}",
            flush=True,
        )

    lines = [
        "# Intensity allocation trial",
        "",
        "Card: 72-pixel odds / venue / opponent. Goals only change head. Assists stay four-way.",
        "Native init is seed 0. Shifted init copies the body from a 76-pixel header draw.",
        "Pass: painted gap ≤ 50 and both within 80 of 730 actual goals at the 20% cut.",
        "",
        "| method | init | soft Σ | painted | exact | rows in (0,0.5) | goal chance by position |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label'].split('/')[0]} | {row['label'].split('/')[1]} | {row['soft']} | "
            f"{row['painted']} | {row['exact']} | {row['rows_below_half']}/{row['live_rows']} | "
            f"{row['chance']} |"
        )
    lines.extend(
        [
            "",
            f"Wall clock: {time.perf_counter() - started:.0f} seconds on CPU.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT_REPORTS / "intensity_trial.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
