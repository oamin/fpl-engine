"""Train the opacity card with a proper score.

The card is the Baudot row mixer. Event dots keep their probability.
Two training losses, pre-registered, same FOV and seed:

  log_loss  — negative log likelihood of the remainder event bits.
              This is the player-GW likelihood, and it is the default train loss.
  brier     — mean squared residual of the same bits.

Early-stop each arm on its own holdout mean (lower is better).
The Tversky-trained reading of this card is the baseline:
holdout log loss about 0.078, Brier about 0.0195, from the residual-card run.
The published reading is the opacity of the event dots. A solid true dot
predicted at 0.20 contributes 0.20 of sureness and leaves a shortfall of 0.80.
Precision is sureness on true dots over all sureness. Recall is sureness on
true dots over the count of solid dots. There is no 0.5 cutoff.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.flashcard_baudot_residual_card import live_pairs, residual_scores, soft_events
from src.models.flashcard_baudot_row_trial import (
    ART,
    BATCH,
    CUT_FRACS,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GOAL_BITS,
    LR,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    BlockTokenMixer,
    _batch,
    _player_pad,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    event_target,
    fit_clubs,
    load_odds,
    train_mixer,
)
ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_opacity_loss.md"
ARMS = ("log_loss", "brier")


def _mean_key(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else 0.0


def proper_table(model, cards, device) -> list[dict]:
    rows = []
    for frac in CUT_FRACS:
        ys, ps = [], []
        for card in cards:
            cut = cut_at_fraction(card, frac)
            soft = soft_events(model, card, cut, device)
            y, p = live_pairs(card, soft, cut)
            ys.append(y)
            ps.append(p)
        row = residual_scores(np.concatenate(ys), np.concatenate(ps))
        row["revealed"] = f"{frac:.0%}"
        rows.append(row)
    return rows


def _sureness(y: np.ndarray, p: np.ndarray) -> dict:
    """Precision and recall from opacity, with no threshold.

    A true dot predicted at 0.20 adds 0.20 to sureness on true events and is
    short of the solid 1 by 0.80. A blank dot predicted at 0.20 adds 0.20 to
    sureness on false events. Precision = on_true / (on_true + on_false).
    Recall = on_true / solid dots, the average opacity on the true dots.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    solid = float(y.sum())
    blank = float((1.0 - y).sum())
    on_true = float((p * y).sum())
    on_false = float((p * (1.0 - y)).sum())
    painted = on_true + on_false
    precision = on_true / painted if painted > 0 else 0.0
    recall = on_true / solid if solid > 0 else 0.0
    if solid + painted == 0:
        score = 1.0
    else:
        f1 = (2.0 * on_true) / (solid + painted)
        volume = 0.0 if solid == 0 or painted == 0 else min(solid, painted) / max(solid, painted)
        score = f1 * volume
    pc = np.clip(p, 1e-4, 1.0 - 1e-4)
    log_loss = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc))) if len(y) else 0.0
    brier = float(np.mean((y - p) ** 2)) if len(y) else 0.0
    return {
        "n": int(len(y)),
        "solid": solid,
        "blank": blank,
        "on_true": on_true,
        "on_false": on_false,
        "precision": precision,
        "recall": recall,
        "score": score,
        "log_loss": log_loss,
        "brier": brier,
        "mean_true": recall,
        "mean_false": on_false / blank if blank > 0 else 0.0,
    }


def _dot_pairs(card, soft: np.ndarray, cut: int, index: int | slice) -> tuple[np.ndarray, np.ndarray]:
    pad = _player_pad(card)
    ys, ps = [], []
    for gw in range(cut, card.n_fixtures):
        target = event_target(card, gw)
        for row in range(card.n_players):
            if pad[row] or not card.present[row, gw]:
                continue
            ys.append(np.atleast_1d(target[row, index]))
            ps.append(np.atleast_1d(soft[row, gw, index]))
    if not ys:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ys), np.concatenate(ps)


def opacity_tables(model, cards, device) -> dict[str, list[dict]]:
    """Sureness on the event-marker dot, and on every unary event dot."""
    marker, dots = [], []
    for frac in CUT_FRACS:
        revealed = f"{frac:.0%}"
        for channel, mark, sl in (
            ("goals", 0, slice(0, GOAL_BITS)),
            ("assists", GOAL_BITS, slice(GOAL_BITS, EVENT_BITS)),
        ):
            my, mp, dy, dp = [], [], [], []
            for card in cards:
                cut = cut_at_fraction(card, frac)
                soft = soft_events(model, card, cut, device)
                y, p = _dot_pairs(card, soft, cut, mark)
                my.append(y)
                mp.append(p)
                y, p = _dot_pairs(card, soft, cut, sl)
                dy.append(y)
                dp.append(p)
            for bucket, ys, ps in (
                (marker, my, mp),
                (dots, dy, dp),
            ):
                row = _sureness(np.concatenate(ys), np.concatenate(ps))
                row["revealed"] = revealed
                row["channel"] = channel
                bucket.append(row)
    return {"marker": marker, "dots": dots}


def arm_loss(name: str, pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    mask = live.to(pred.dtype)
    if name == "log_loss":
        nll = F.binary_cross_entropy(pred, target, reduction="none")
        return (nll * mask).sum() / mask.sum().clamp_min(1.0)
    resid = (pred - target) ** 2
    return (resid * mask).sum() / mask.sum().clamp_min(1.0)


def train_arm(name: str, model, cards, test, device) -> dict:
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best = float("inf")
    bad = 0
    t0 = time.time()
    epochs_run = 0
    key = name
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [cut_at_fraction(card, float(rng.choice(CUT_FRACS))) for card in pick]
            bits, row_on, rem_on, pad, present, target = _batch(pick, cuts, device)
            live = present[:, :, :, None] & rem_on[:, None, :, None].bool() & (~pad)[:, :, None, None]
            opt.zero_grad()
            pred = model(bits, row_on, pad)
            loss = arm_loss(name, pred, target, live)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        proper = proper_table(model, test, device)
        score = _mean_key(proper, key)
        improved = score < best - 1e-5
        if improved:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} train {total / max(seen, 1):.4f} "
            f"holdout {key} {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    proper = proper_table(model, test, device)
    reading = opacity_tables(model, test, device)
    return {
        "name": name,
        "epochs_run": epochs_run,
        "best": best,
        "seconds": round(time.time() - t0, 1),
        "proper": proper,
        "marker": reading["marker"],
        "dots": reading["dots"],
    }


def _fmt(row: dict) -> str:
    return (
        f"| {row['arm']} | {row['revealed']} | {row['channel']} | {row['log_loss']:.4f} | {row['brier']:.4f} | "
        f"{row['solid']:.0f} | {row['on_true']:.1f} | {row['on_false']:.1f} | "
        f"{row['mean_false']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['score']:.3f} |"
    )


def _ordered(rows: list[dict], arm: str) -> list[dict]:
    out = []
    for row in sorted(rows, key=lambda r: (r["channel"] != "goals", r["revealed"])):
        item = dict(row)
        item["arm"] = arm
        out.append(item)
    return out


def write_report(arms: list[dict]) -> Path:
    header = (
        "| arm | revealed | channel | log loss | Brier | solid dots | on true | on false | "
        "mean on false | precision | recall | score |"
    )
    rule = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Opacity card — sureness on the event dots",
        "",
        "A true event is a solid dot. The model paints that dot with opacity equal to its probability. "
        "Predicted opacity 0.20 on a solid 1 is 0.20 of the way there and 0.80 short. "
        "On true is the sum of that opacity over solid dots. On false is the sum over blank dots. "
        "Precision = on true / (on true + on false). Recall = on true / solid dots, "
        "which is the average opacity on a true dot. Mean on false is the average opacity on a blank dot. "
        "The score is the harmonic mean of precision and recall, times the ratio of the smaller "
        "opacity mass to the larger of (solid dots, total opacity). No probability is thresholded.",
        "",
        "The event marker is the first dot of the channel: solid when the player has at least one "
        "goal or assist in that gameweek. Log loss and Brier on each row are that same dot.",
        "",
        "## Event marker",
        "",
        header,
        rule,
    ]
    for meta in arms:
        for row in _ordered(meta["marker"], meta["name"]):
            lines.append(_fmt(row))
    lines.extend(
        [
            "",
            "## Every event dot",
            "",
            "The other two dots in the channel are the unary squares for two and for three. "
            "A brace is two solid goal dots. The same sureness sums cover every square the loss trains.",
            "",
            header,
            rule,
        ]
    )
    for meta in arms:
        for row in _ordered(meta["dots"], meta["name"]):
            lines.append(_fmt(row))
    lines.append("")
    log_arm = next(m for m in arms if m["name"] == "log_loss")
    brier_arm = next(m for m in arms if m["name"] == "brier")
    tv = next(m for m in arms if m["name"] == "tversky")

    def _mean_recall(meta: dict) -> float:
        return _mean_key(meta["marker"], "recall")

    def _mean_precision(meta: dict) -> float:
        return _mean_key(meta["marker"], "precision")

    lines.extend(
        [
            "## Call",
            "",
            "Event-marker means. "
            f"Tversky precision **{_mean_precision(tv):.3f}**, recall **{_mean_recall(tv):.3f}**. "
            f"Log loss precision **{_mean_precision(log_arm):.3f}**, recall **{_mean_recall(log_arm):.3f}**. "
            f"Brier precision **{_mean_precision(brier_arm):.3f}**, recall **{_mean_recall(brier_arm):.3f}**. "
            f"Training loss on every remainder event bit: log loss **{_mean_key(log_arm['proper'], 'log_loss'):.4f}**, "
            f"Brier arm **{_mean_key(brier_arm['proper'], 'brier'):.4f}**.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    print("loading", flush=True)
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
    print(f"cards train {len(train)} test {len(test)} device {device}", flush=True)
    results = []
    torch.manual_seed(SEED)
    model = BlockTokenMixer(width=train[0].width)
    t0 = time.time()
    trained = train_mixer(model, train, np.random.default_rng(SEED), device, test=test)
    reading = opacity_tables(model, test, device)
    proper = proper_table(model, test, device)
    results.append(
        {
            "name": "tversky",
            "epochs_run": trained["epochs_run"],
            "seconds": round(time.time() - t0, 1),
            "proper": proper,
            "marker": reading["marker"],
            "dots": reading["dots"],
        }
    )
    for name in ARMS:
        torch.manual_seed(SEED)
        model = BlockTokenMixer(width=train[0].width)
        results.append(train_arm(name, model, train, test, device))
    path = write_report(results)
    state = {
        "trial": "baudot_opacity_loss",
        "default_card": "opacity",
        "default_train_loss": "log_loss",
        "reading": "sureness on event dots, no threshold",
        "arms": [
            {
                "name": meta["name"],
                "epochs_run": meta["epochs_run"],
                "seconds": meta["seconds"],
                "mean_log_loss": _mean_key(meta["proper"], "log_loss"),
                "mean_brier": _mean_key(meta["proper"], "brier"),
                "marker_precision": _mean_key(meta["marker"], "precision"),
                "marker_recall": _mean_key(meta["marker"], "recall"),
                "marker_score": _mean_key(meta["marker"], "score"),
                "proper": meta["proper"],
                "marker": meta["marker"],
                "dots": meta["dots"],
            }
            for meta in results
        ],
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (ART / "baudot_opacity_loss.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (REPORTS / "baudot_opacity_loss.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
