"""Train the opacity card with a proper score.

The card is the Baudot row mixer. Event dots keep their probability.
Two training losses, pre-registered, same FOV and seed:

  log_loss  — negative log likelihood of the remainder event bits.
              This is the player-GW likelihood, and it is the default train loss.
  brier     — mean squared residual of the same bits.

Early-stop each arm on its own holdout mean (lower is better).
The Tversky-trained reading of this card is the baseline:
holdout log loss about 0.078, Brier about 0.0195, from the residual-card run.
The 0.5 tally is reported beside the residual. It is not the selection metric.
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
    LR,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    BlockTokenMixer,
    _batch,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    evaluate_mixer,
    fit_clubs,
    load_odds,
)
ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_opacity_loss.md"
ARMS = ("log_loss", "brier")
# Tversky-trained opacity reading, seed 0, reports/baudot_residual_card.json.
TVERSKY_PROPER = (
    {"revealed": "20%", "n": 94134, "brier": 0.019691726192831993, "log_loss": 0.07942351698875427, "mae": 0.03063645027577877},
    {"revealed": "40%", "n": 72822, "brier": 0.019398488104343414, "log_loss": 0.07750938087701797, "mae": 0.0303444005548954},
    {"revealed": "60%", "n": 48066, "brier": 0.019361218437552452, "log_loss": 0.07718615978956223, "mae": 0.030382143333554268},
    {"revealed": "80%", "n": 25680, "brier": 0.019409965723752975, "log_loss": 0.07808905839920044, "mae": 0.030261578038334846},
)


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
    tally = evaluate_mixer(model, test, device)
    return {
        "name": name,
        "epochs_run": epochs_run,
        "best": best,
        "seconds": round(time.time() - t0, 1),
        "proper": proper,
        "tally_mean": float(np.mean([r["score"] for r in tally["rows"]])),
        "tally": tally["rows"],
    }


def write_report(arms: list[dict]) -> Path:
    base_ll = _mean_key(list(TVERSKY_PROPER), "log_loss")
    base_br = _mean_key(list(TVERSKY_PROPER), "brier")
    lines = [
        "# Opacity card — log loss and Brier",
        "",
        "The default card is the Baudot row card. Event dots keep the probability. "
        "The training loss is the residual of those probabilities against the true bits. "
        "Log loss is the likelihood. Brier is the mean squared residual. "
        "The opponent-card cross-read is not in this run.",
        "",
        f"Tversky-trained baseline on this card: mean log loss **{base_ll:.4f}**, mean Brier **{base_br:.4f}**.",
        "",
        "| arm | epochs | mean log loss | Δ log | mean Brier | Δ Brier | 0.5 tally | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| tversky | 20 | {base_ll:.4f} | +0.0000 | {base_br:.4f} | +0.0000 | 0.218 | — |",
    ]
    for meta in arms:
        ll = _mean_key(meta["proper"], "log_loss")
        br = _mean_key(meta["proper"], "brier")
        lines.append(
            f"| {meta['name']} | {meta['epochs_run']} | {ll:.4f} | {ll - base_ll:+.4f} | "
            f"{br:.4f} | {br - base_br:+.4f} | {meta['tally_mean']:.3f} | {meta['seconds']} |"
        )
    lines.extend(["", "## Holdout residual", "", "| arm | revealed | bits | log loss | Brier | |residual| |", "|---|---|---:|---:|---:|---:|"])
    for row in TVERSKY_PROPER:
        lines.append(
            f"| tversky | {row['revealed']} | {row['n']} | {row['log_loss']:.4f} | {row['brier']:.4f} | {row['mae']:.4f} |"
        )
    for meta in arms:
        for row in meta["proper"]:
            lines.append(
                f"| {meta['name']} | {row['revealed']} | {row['n']} | {row['log_loss']:.4f} | {row['brier']:.4f} | {row['mae']:.4f} |"
            )
    lines.extend(["", "## 0.5 tally (not the selection metric)", ""])
    for meta in arms:
        lines.append(f"### {meta['name']}  mean {meta['tally_mean']:.3f}")
        lines.append("")
        lines.append("| revealed | channel | actual | painted | hits | score |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in meta["tally"]:
            lines.append(
                f"| {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | {row['hits']} | {row['score']:.3f} |"
            )
        lines.append("")
    log_arm = next(m for m in arms if m["name"] == "log_loss")
    brier_arm = next(m for m in arms if m["name"] == "brier")
    ll = _mean_key(log_arm["proper"], "log_loss")
    br = _mean_key(brier_arm["proper"], "brier")
    lines.extend(
        [
            "## Call",
            "",
            f"Log loss is the training loss of the opacity card. Its holdout mean is **{ll:.4f}** "
            f"(Tversky-trained {base_ll:.4f}). "
            f"The Brier arm's mean squared residual is **{br:.4f}** (Tversky-trained {base_br:.4f}). "
            "The 0.5 tally is recorded beside those scores and does not choose the checkpoint.",
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
    for name in ARMS:
        torch.manual_seed(SEED)
        model = BlockTokenMixer(width=train[0].width)
        results.append(train_arm(name, model, train, test, device))
    path = write_report(results)
    state = {
        "trial": "baudot_opacity_loss",
        "default_card": "opacity",
        "default_train_loss": "log_loss",
        "tversky_mean_log_loss": _mean_key(list(TVERSKY_PROPER), "log_loss"),
        "tversky_mean_brier": _mean_key(list(TVERSKY_PROPER), "brier"),
        "arms": [
            {
                "name": meta["name"],
                "epochs_run": meta["epochs_run"],
                "seconds": meta["seconds"],
                "mean_log_loss": _mean_key(meta["proper"], "log_loss"),
                "mean_brier": _mean_key(meta["proper"], "brier"),
                "tally_mean": meta["tally_mean"],
                "proper": meta["proper"],
                "tally": meta["tally"],
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
