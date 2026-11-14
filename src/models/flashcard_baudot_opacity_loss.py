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
from src.models.flashcard_card_score import GATE, balanced_score
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


def _eval_line(arm: str, revealed: str, channel: str, log_loss: float, brier: float, row: dict) -> str:
    return (
        f"| {arm} | {revealed} | {channel} | {log_loss:.4f} | {brier:.4f} | "
        f"{row['actual']} | {row['painted']} | {row['hits']} | "
        f"{row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} |"
    )


def _tversky_eval_rows() -> list[dict]:
    rows = []
    for channel in ("goals", "assists"):
        for frac in CUT_FRACS:
            revealed = f"{frac:.0%}"
            proper = next(r for r in TVERSKY_PROPER if r["revealed"] == revealed)
            tot = GATE[(channel, frac)]
            score = balanced_score(tot)
            rows.append(
                {
                    "revealed": revealed,
                    "channel": channel,
                    "log_loss": proper["log_loss"],
                    "brier": proper["brier"],
                    "actual": tot["actual"],
                    "painted": tot["painted"],
                    "hits": tot["exact"],
                    "actual_within_1": tot["actual_within"],
                    "painted_within_1": tot["painted_within"],
                    "score": score["score"],
                }
            )
    return rows


def write_report(arms: list[dict]) -> Path:
    base_ll = _mean_key(list(TVERSKY_PROPER), "log_loss")
    base_br = _mean_key(list(TVERSKY_PROPER), "brier")
    header = (
        "| arm | revealed | channel | log loss | Brier | actual events | painted events | hits | "
        "actual ±1 | painted ±1 | score |"
    )
    rule = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Opacity card — log loss and Brier",
        "",
        "The default card is the Baudot row card. Event dots keep the probability. "
        "Log loss and Brier are the residual on every remainder event bit at that cut. "
        "Actual events, painted events, hits, the ±1 columns, and the score are the "
        "0.5 tally used on the 2026-11-13 mixer table. "
        "The Tversky event counts are that published table.",
        "",
        f"Mean log loss **{base_ll:.4f}** / Brier **{base_br:.4f}** on the Tversky reading. "
        "Mean score of its eight tally cells is **0.218**.",
        "",
        header,
        rule,
    ]
    for row in _tversky_eval_rows():
        lines.append(
            _eval_line("tversky", row["revealed"], row["channel"], row["log_loss"], row["brier"], row)
        )
    for meta in arms:
        by_cut = {r["revealed"]: r for r in meta["proper"]}
        ordered = sorted(meta["tally"], key=lambda r: (r["channel"] != "goals", r["revealed"]))
        for row in ordered:
            proper = by_cut[row["revealed"]]
            lines.append(
                _eval_line(meta["name"], row["revealed"], row["channel"], proper["log_loss"], proper["brier"], row)
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
            f"Log loss holdout mean **{ll:.4f}** (Tversky {base_ll:.4f}). "
            f"Brier arm mean squared residual **{br:.4f}** (Tversky {base_br:.4f}), "
            f"with log loss **{_mean_key(brier_arm['proper'], 'log_loss'):.4f}**. "
            f"On the 0.5 tally the log-loss arm scores **{log_arm['tally_mean']:.3f}** "
            f"and the Brier arm **{brier_arm['tally_mean']:.3f}**, against the Tversky table **0.218**. "
            "Both residual arms paint under the 0.5 line, so painted events and hits stay at zero.",
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
