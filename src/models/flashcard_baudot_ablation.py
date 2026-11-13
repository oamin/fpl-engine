"""One-by-one Baudot mixer loss/train ablations.

Arms (sequential suggestions from the inflation review):
  1. soft_dice_20      — current baseline
  2. tversky_20        — Tversky β>α (penalise blank false paints)
  3. tversky_long      — Tversky + 50 epochs, cosine LR, early-stop on score
  4. tversky_long_pos  — (3) + equal per-position Tversky terms

Holdout table uses tally_row + balanced_score (familiar columns).
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.flashcard_baudot_row_trial import (
    ART,
    BATCH,
    CUT_FRACS,
    EVENT_BITS,
    GOAL_BITS,
    LR,
    MAX_PLAYERS,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    BlockTokenMixer,
    BaudotCard,
    _batch,
    _player_pad,
    build_baudot_cards,
    build_name_codes,
    cut_at_fraction,
    evaluate_mixer,
    load_odds,
    build_league,
    build_cards,
    fit_clubs,
)
from src.models.flashcard_card_score import balanced_score

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_ablation_table.md"
ART.mkdir(parents=True, exist_ok=True)

TVERSKY_ALPHA = 0.3  # FN weight
TVERSKY_BETA = 0.7  # FP weight
LONG_EPOCHS = 50
EARLY_PATIENCE = 8
BCE_W = 0.05


def soft_dice(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
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


def tversky(
    pred: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    alpha: float = TVERSKY_ALPHA,
    beta: float = TVERSKY_BETA,
) -> torch.Tensor:
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


def tversky_per_position(
    pred: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    position: torch.Tensor,
    alpha: float = TVERSKY_ALPHA,
    beta: float = TVERSKY_BETA,
) -> torch.Tensor:
    """Equal-weight Tversky on each role's remainder picture."""
    terms = []
    for pos in range(4):
        mask = live & (position[:, :, None, None] == pos)
        if bool(mask.any()):
            terms.append(tversky(pred, target, mask, alpha=alpha, beta=beta))
    if not terms:
        return tversky(pred, target, live, alpha=alpha, beta=beta)
    return torch.stack(terms).mean()


def _positions_batch(cards: list[BaudotCard]) -> torch.Tensor:
    out = np.full((len(cards), MAX_PLAYERS), -1, dtype=np.int64)
    for i, card in enumerate(cards):
        out[i, : card.n_players] = card.position[: card.n_players]
    return torch.from_numpy(out)


def _holdout_mean_score(model: BlockTokenMixer, cards: list[BaudotCard], device: torch.device) -> float:
    stats = evaluate_mixer(model, cards, device)
    scores = [row["score"] for row in stats["rows"]]
    return float(np.mean(scores)) if scores else 0.0


def train_arm(
    name: str,
    cards: list[BaudotCard],
    test: list[BaudotCard],
    device: torch.device,
    *,
    loss_kind: str,
    epochs: int,
    cosine: bool,
    early_stop: bool,
    per_position: bool,
) -> tuple[BlockTokenMixer, dict]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = BlockTokenMixer(width=cards[0].width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = None
    if cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: list[float] = []
    val_scores: list[float] = []
    best_state = None
    best_score = -1.0
    bad = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            fracs = [float(rng.choice(CUT_FRACS)) for _ in pick]
            cuts = [cut_at_fraction(card, frac) for card, frac in zip(pick, fracs, strict=True)]
            bits, row_on, rem_on, pad, present, target = _batch(pick, cuts, device)
            position = _positions_batch(pick).to(device)
            live = (
                present[:, :, :, None]
                & rem_on[:, None, :, None].bool()
                & (~pad)[:, :, None, None]
            )
            opt.zero_grad()
            pred = model(bits, row_on, pad)
            if loss_kind == "soft_dice":
                loss = soft_dice(pred, target, live)
            elif per_position:
                loss = tversky_per_position(pred, target, live, position)
            else:
                loss = tversky(pred, target, live)
            bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
            bce = (bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0)
            loss = loss + BCE_W * bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        if sched is not None:
            sched.step()
        avg = total / max(seen, 1)
        history.append(avg)

        score = _holdout_mean_score(model, test, device)
        val_scores.append(score)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(
            f"{name} epoch {epoch + 1}/{epochs} loss {avg:.4f} holdout_score {score:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if early_stop and bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate_mixer(model, test, device)
    # position split at 20% goals for DEF diagnosis
    pos_split = _position_split(model, test, device, frac=0.2, channel="goals")
    meta = {
        "name": name,
        "loss_kind": loss_kind,
        "epochs_run": len(history),
        "cosine": cosine,
        "early_stop": early_stop,
        "per_position": per_position,
        "best_holdout_mean_score": best_score,
        "final_holdout_mean_score": float(np.mean([r["score"] for r in holdout["rows"]])),
        "seconds": round(time.time() - t0, 1),
        "history": history,
        "val_scores": val_scores,
        "holdout": holdout,
        "goals_20_by_pos": pos_split,
    }
    return model, meta


@torch.no_grad()
def _position_split(
    model: BlockTokenMixer,
    cards: list[BaudotCard],
    device: torch.device,
    frac: float,
    channel: str,
) -> dict[str, dict[str, int]]:
    from src.models.flashcard_baudot_row_trial import remainder_view, _channel_slice

    POS = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}
    sl = _channel_slice(channel)
    out = {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS.values()}
    model.eval()
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
            pos = POS.get(int(card.position[row]))
            if pos is None:
                continue
            for gw in range(cut, card.n_fixtures):
                if not view["present"][row, gw]:
                    continue
                ag = int(view["target"][row, gw, sl].sum())
                pg = int((pred[row, gw, sl] >= 0.5).sum())
                ae, pe = int(ag > 0), int(pg > 0)
                out[pos]["actual"] += ae
                out[pos]["painted"] += pe
                out[pos]["hits"] += int(ae and pe)
    return out


def _row_line(arm: str, row: dict) -> str:
    return (
        f"| {arm} | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
        f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} |"
    )


def write_table(metas: list[dict]) -> Path:
    lines = [
        "# Baudot mixer ablations (one-by-one)",
        "",
        "Full past + one-shot remainder FOV. Metrics: `tally_row` + `balanced_score`.",
        "",
        "| arm | loss | train | per-pos | epochs | best mean score | seconds |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for m in metas:
        train = "50ep cosine+early" if m["cosine"] else "20ep constant"
        lines.append(
            f"| {m['name']} | {m['loss_kind']} | {train} | {m['per_position']} | "
            f"{m['epochs_run']} | {m['best_holdout_mean_score']:.3f} | {m['seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Holdout table",
            "",
            "| arm | revealed | channel | actual events | painted events | hits | actual ±1 | painted ±1 | score |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for m in metas:
        for row in m["holdout"]["rows"]:
            lines.append(_row_line(m["name"], row))
    lines.extend(
        [
            "",
            "## Goals at 20% cut by position (actual / painted / hits)",
            "",
            "| arm | GKP a/p/h | DEF a/p/h | MID a/p/h | FWD a/p/h |",
            "|---|---|---|---|---|",
        ]
    )
    for m in metas:
        g = m["goals_20_by_pos"]
        cells = []
        for pos in ("GKP", "DEF", "MID", "FWD"):
            s = g[pos]
            cells.append(f"{s['actual']}/{s['painted']}/{s['hits']}")
        lines.append(f"| {m['name']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Arms: (1) soft_dice baseline → (2) Tversky β=0.7 α=0.3 → "
            "(3) longer cosine+early-stop → (4) + equal per-position Tversky.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    (ART / "baudot_ablation_table.md").write_text("\n".join(lines), encoding="utf-8")
    return OUT


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    print("loading cards", flush=True)
    league = build_league(seasons)
    odds = load_odds()
    codes = build_name_codes(seasons)
    squad = build_cards(seasons, league)
    clubs = fit_clubs([c for c in squad if c.season in TRAIN_SEASONS])
    for card in squad:
        clubs.setdefault(card.team, len(clubs) + 1)
        for opp in card.opponent:
            if opp:
                clubs.setdefault(opp, len(clubs) + 1)
    baudot = build_baudot_cards(squad, clubs, codes, odds)
    train = [c for c in baudot if c.season in TRAIN_SEASONS]
    test = [c for c in baudot if c.season == TEST_SEASON]

    arms = [
        dict(name="1_soft_dice_20", loss_kind="soft_dice", epochs=20, cosine=False, early_stop=False, per_position=False),
        dict(name="2_tversky_20", loss_kind="tversky", epochs=20, cosine=False, early_stop=False, per_position=False),
        dict(name="3_tversky_long", loss_kind="tversky", epochs=LONG_EPOCHS, cosine=True, early_stop=True, per_position=False),
        dict(
            name="4_tversky_long_pos",
            loss_kind="tversky",
            epochs=LONG_EPOCHS,
            cosine=True,
            early_stop=True,
            per_position=True,
        ),
    ]
    metas = []
    for arm in arms:
        print(f"\n=== {arm['name']} ===", flush=True)
        _model, meta = train_arm(cards=train, test=test, device=device, **arm)
        del _model
        metas.append(meta)

    path = write_table(metas)
    slim = []
    for m in metas:
        slim.append(
            {
                "name": m["name"],
                "epochs_run": m["epochs_run"],
                "best_holdout_mean_score": m["best_holdout_mean_score"],
                "seconds": m["seconds"],
                "rows": m["holdout"]["rows"],
                "goals_20_by_pos": m["goals_20_by_pos"],
                "gkp_by_cut": m["holdout"]["gkp_painted_goals_by_cut"],
            }
        )
    (ROOT / "REPORT_STATE.json").write_text(
        json.dumps({"trial": "baudot_ablation", "arms": slim, "report": str(path)}, indent=2),
        encoding="utf-8",
    )
    (ART / "baudot_ablation.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
