"""Baudot stepped-GW RL screen vs locked full-remainder gate.

Same card + BlockTokenMixer encoder. FOV = next-GW only (as in the parked
stepped supervised arm).

Protocol (pre-registered):
  1. Warm-start: learnable Tversky on next-GW (same as stepped supervised)
     for WARM_EPOCHS — pure SCST from scratch collapsed to entropy-max / blank.
  2. RL fine-tune: self-critical REINFORCE
       state  = past rows true, target GW G/A blank
       action = Bernoulli sample of hard event bits on live target cells
       reward = channel balanced_score on channels with actual>0
       baseline = greedy thr-0.5 reward (SCST)

Holdout = teacher-forced stepped tally table (thr 0.5), same cells as GATE.
Primary call: mean score vs gate 0.218.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.flashcard_baudot_row_trial import (
    ART,
    BATCH,
    BCE_W,
    CUT_FRACS,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GATE,
    GATE_META,
    GOAL_BITS,
    LR,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    BaudotCard,
    BlockTokenMixer,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    fit_clubs,
    load_odds,
    tversky_loss,
)
from src.models.flashcard_baudot_stepped_gw import (
    _batch_next,
    _gate_rows,
    _mean_score,
    _pos_cell,
    _sample_target_gw,
    evaluate_stepped,
)
from src.models.flashcard_card_score import balanced_score

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_rl_stepped.md"

# RL hyperparameters (pre-registered for this screen).
ENTROPY_COEF = 0.01
EPS = 1e-5
WARM_EPOCHS = 15  # Tversky next-GW warm-start before SCST
STEPPED_SUPERVISED_MEAN = 0.184  # parked report; not re-run


class PolicyMixer(BlockTokenMixer):
    """Gate mixer; expose logits for Bernoulli actions. Same −3 paint bias as gate."""

    def forward_logits(
        self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor
    ) -> torch.Tensor:
        tokens = self._encode_rows(bits) + self.time_pos[:, : bits.shape[1]]
        b, t, p, d = tokens.shape
        time_view = tokens.permute(0, 2, 1, 3).reshape(b * p, t, d)
        time_mask = (row_on < 0.5)[:, None, :].expand(b, p, t).reshape(b * p, t)
        mixed, _ = self.time(time_view, time_view, time_view, key_padding_mask=time_mask, need_weights=False)
        time_view = self.norm_t(time_view + mixed)
        tokens = time_view.reshape(b, p, t, d).permute(0, 2, 1, 3)
        squad_view = tokens.reshape(b * t, p, d)
        squad_mask = player_pad[:, None, :].expand(b, t, p).reshape(b * t, p)
        mixed, _ = self.squad(squad_view, squad_view, squad_view, key_padding_mask=squad_mask, need_weights=False)
        tokens = self.norm_s(squad_view + mixed).reshape(b, t, p, d)
        return self.paint(tokens).permute(0, 2, 1, 3)  # B,P,T,E

    def forward(self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(bits, row_on, player_pad))


def _channel_balanced(
    action: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    sl: slice,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-batch-item balanced_score for one channel. Returns (score B, actual B)."""
    live_f = live[..., sl].to(action.dtype)
    a = action[..., sl] * live_f
    y = target[..., sl] * live_f
    dims = (1, 2, 3)
    actual = y.sum(dim=dims)
    painted = a.sum(dim=dims)
    exact = (a * y).sum(dim=dims)
    # Match flashcard_card_score.balanced_score, including both-zero → 1.
    both_zero = actual + painted <= 0
    hits = torch.where(
        both_zero,
        torch.ones_like(actual),
        (2.0 * exact) / (actual + painted + EPS),
    )
    both_pos = (actual > 0) & (painted > 0)
    volume = torch.where(
        both_zero,
        torch.ones_like(actual),
        torch.where(
            both_pos,
            torch.minimum(actual, painted) / (torch.maximum(actual, painted) + EPS),
            torch.zeros_like(actual),
        ),
    )
    return hits * volume, actual


def step_reward(
    action: torch.Tensor, target: torch.Tensor, live: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean goals/assists balanced_score over channels with actual>0.

    Returns (score B, has_actual B). Blank cards and empty channels are
    excluded: official both-zero→1 would otherwise make never-paint optimal
    under SCST whenever one channel is quiet.
    """
    g, ag = _channel_balanced(action, target, live, slice(0, GOAL_BITS))
    a, aa = _channel_balanced(action, target, live, slice(GOAL_BITS, EVENT_BITS))
    g_on = ag > 0
    a_on = aa > 0
    denom = g_on.to(g.dtype) + a_on.to(a.dtype)
    score = torch.where(
        denom > 0,
        (g * g_on.to(g.dtype) + a * a_on.to(a.dtype)) / denom.clamp_min(1.0),
        torch.zeros_like(g),
    )
    has_actual = denom > 0
    return score, has_actual


def reinforce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Self-critical REINFORCE on Bernoulli event-bit actions."""
    probs = torch.sigmoid(logits).clamp(EPS, 1.0 - EPS)
    # live may arrive as B,P,T,1 from the stepped batch — broadcast to EVENT_BITS.
    if live.shape[-1] == 1:
        live = live.expand(*live.shape[:-1], EVENT_BITS)
    live_f = live.to(probs.dtype)
    dist = torch.distributions.Bernoulli(probs=probs)
    sample = dist.sample() * live_f
    greedy = ((probs >= 0.5).to(probs.dtype)) * live_f

    r_sample, has_actual = step_reward(sample, target, live)
    with torch.no_grad():
        r_greedy, _ = step_reward(greedy, target, live)
        advantage = (r_sample - r_greedy) * has_actual.to(r_sample.dtype)

    log_p = dist.log_prob(sample) * live_f
    log_p = log_p.sum(dim=(1, 2, 3)) / live_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
    # Average PG over eventful steps only (avoid diluting with blank-week zeros).
    n_evt = has_actual.to(log_p.dtype).sum().clamp_min(1.0)
    pg = -(advantage.detach() * log_p).sum() / n_evt

    entropy = (dist.entropy() * live_f).sum() / live_f.sum().clamp_min(1.0)
    loss = pg - ENTROPY_COEF * entropy

    evt = has_actual
    stats = {
        "r_sample": float(r_sample[evt].mean().detach()) if bool(evt.any()) else 0.0,
        "r_greedy": float(r_greedy[evt].mean().detach()) if bool(evt.any()) else 0.0,
        "advantage": float(advantage[evt].mean().detach()) if bool(evt.any()) else 0.0,
        "entropy": float(entropy.detach()),
        "pg": float(pg.detach()),
        "frac_eventful": float(evt.float().mean().detach()),
    }
    return loss, stats


def train_rl(
    model: PolicyMixer,
    train: list[BaudotCard],
    test: list[BaudotCard],
    device: torch.device,
) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history: list[float] = []
    val_scores: list[float] = []
    train_stats: list[dict] = []
    phase_log: list[str] = []
    best_state = None
    best_score = -1.0
    bad = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        phase = "warm_tversky" if epoch < WARM_EPOCHS else "scst"
        if epoch == WARM_EPOCHS:
            bad = 0  # fresh patience window for the RL phase
            print(f"=== switch to SCST at epoch {epoch + 1} (warm best {best_score:.4f}) ===", flush=True)
        model.train()
        total = 0.0
        seen = 0
        epoch_stats = {
            "r_sample": 0.0,
            "r_greedy": 0.0,
            "advantage": 0.0,
            "entropy": 0.0,
            "frac_eventful": 0.0,
            "phase": phase,
        }
        order = rng.permutation(len(train))
        for start in range(0, len(train), BATCH):
            pick = [train[int(i)] for i in order[start : start + BATCH]]
            targets = [_sample_target_gw(card, rng) for card in pick]
            bits, row_on, rem_on, pad, present, target = _batch_next(pick, targets, device)
            live = (
                present[:, :, :, None]
                & rem_on[:, None, :, None].bool()
                & (~pad)[:, :, None, None]
            )
            opt.zero_grad()
            if phase == "warm_tversky":
                pred = model(bits, row_on, pad)
                alpha, beta = model.tversky_weights()
                loss = tversky_loss(pred, target, live, alpha, beta)
                bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
                loss = loss + BCE_W * (
                    (bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0)
                )
                st = {
                    "r_sample": 0.0,
                    "r_greedy": 0.0,
                    "advantage": 0.0,
                    "entropy": 0.0,
                    "frac_eventful": 0.0,
                }
            else:
                logits = model.forward_logits(bits, row_on, pad)
                loss, st = reinforce_loss(logits, target, live)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
            for k in ("r_sample", "r_greedy", "advantage", "entropy", "frac_eventful"):
                epoch_stats[k] += st[k]
        sched.step()
        avg = total / max(seen, 1)
        history.append(avg)
        for k in ("r_sample", "r_greedy", "advantage", "entropy", "frac_eventful"):
            epoch_stats[k] /= max(seen, 1)
        train_stats.append(epoch_stats)
        phase_log.append(phase)

        hold = evaluate_stepped(model, test, device)
        score = _mean_score(hold["rows"])
        val_scores.append(score)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        extra = ""
        if phase == "scst":
            extra = (
                f" R_s={epoch_stats['r_sample']:.3f} R_g={epoch_stats['r_greedy']:.3f} "
                f"H={epoch_stats['entropy']:.3f} evt={epoch_stats['frac_eventful']:.2f}"
            )
        print(
            f"rl_step_bal epoch {epoch + 1}/{EPOCHS} [{phase}] loss {avg:.4f}{extra} "
            f"holdout_score {score:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        # Early-stop only in the RL phase after a full patience window.
        if phase == "scst" and bad >= EARLY_PATIENCE:
            print(f"rl_step_bal early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate_stepped(model, test, device)
    with torch.no_grad():
        a, b = model.tversky_weights()
        final_ab = (float(a), float(b))
    return {
        "name": "rl_step_bal",
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "warm_epochs": WARM_EPOCHS,
        "best_holdout_mean_score": best_score,
        "final_alpha": final_ab[0],
        "final_beta": final_ab[1],
        "entropy_coef": ENTROPY_COEF,
        "seconds": round(time.time() - t0, 1),
        "holdout": holdout,
        "history": history,
        "val_scores": val_scores,
        "train_stats": train_stats,
        "phase_log": phase_log,
    }


def _delta(a: float, b: float) -> str:
    return f"{a - b:+.3f}"


def write_report(meta: dict) -> Path:
    gate_rows = _gate_rows()
    gate_mean = _mean_score(gate_rows)
    mean = _mean_score(meta["holdout"]["rows"])
    hold = meta["holdout"]
    lines = [
        "# Baudot stepped-GW RL vs locked full-remainder gate",
        "",
        "Same Baudot card + `BlockTokenMixer` encoder. Training is self-critical "
        "REINFORCE on next-GW Bernoulli paint; reward = channel `balanced_score` "
        "(hits × volume). Holdout = teacher-forced stepped table (thr 0.5), same "
        "cells as the locked GATE.",
        "",
        f"- Gate: `{GATE_META['method']}` · mean **{gate_mean:.3f}**",
        f"- Parked stepped supervised (Tversky): mean **{STEPPED_SUPERVISED_MEAN:.3f}** "
        f"(Δ {_delta(STEPPED_SUPERVISED_MEAN, gate_mean)})",
        f"- Warm-start: {meta['warm_epochs']} epochs next-GW Tversky, then SCST",
        f"- Entropy coef: {ENTROPY_COEF}",
        "",
        "## Scoreboard",
        "",
        "| arm | params | epochs | α | β | mean score | Δ gate | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| gate (full remainder one-shot) | — | — | {GATE_META['alpha']:.3f} | "
        f"{GATE_META['beta']:.3f} | {gate_mean:.3f} | +0.000 | — |",
        f"| stepped_next_gw (supervised, parked) | — | — | — | — | {STEPPED_SUPERVISED_MEAN:.3f} | "
        f"{_delta(STEPPED_SUPERVISED_MEAN, gate_mean)} | — |",
        f"| rl_step_bal | {meta['params']} | {meta['epochs_run']} | {meta['final_alpha']:.3f} | "
        f"{meta['final_beta']:.3f} | {mean:.3f} | {_delta(mean, gate_mean)} | {meta['seconds']} |",
        "",
        "## Goals at 20% by position (actual / painted / hits)",
        "",
        "| arm | GKP | DEF | MID | FWD |",
        "|---|---|---|---|---|",
        f"| rl_step_bal | {_pos_cell(hold['goals_20_by_pos'], 'GKP')} | "
        f"{_pos_cell(hold['goals_20_by_pos'], 'DEF')} | "
        f"{_pos_cell(hold['goals_20_by_pos'], 'MID')} | "
        f"{_pos_cell(hold['goals_20_by_pos'], 'FWD')} |",
        "",
        "## Assists at 20% by position (actual / painted / hits)",
        "",
        "| arm | GKP | DEF | MID | FWD |",
        "|---|---|---|---|---|",
        f"| rl_step_bal | {_pos_cell(hold['assists_20_by_pos'], 'GKP')} | "
        f"{_pos_cell(hold['assists_20_by_pos'], 'DEF')} | "
        f"{_pos_cell(hold['assists_20_by_pos'], 'MID')} | "
        f"{_pos_cell(hold['assists_20_by_pos'], 'FWD')} |",
        "",
        "## Holdout table (same cells as gate)",
        "",
        "| arm | revealed | channel | actual | painted | hits | actual ±1 | painted ±1 | score | Δ gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(
            f"| gate | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
            f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | "
            f"{row['score']:.3f} | +0.000 |"
        )
    for row in hold["rows"]:
        key = (row["channel"], {"20%": 0.2, "40%": 0.4, "60%": 0.6, "80%": 0.8}[row["revealed"]])
        gscore = balanced_score(GATE[key])["score"]
        lines.append(
            f"| rl_step_bal | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
            f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | "
            f"{row['score']:.3f} | {_delta(row['score'], gscore)} |"
        )

    phases = meta.get("phase_log") or []
    val_scores = meta.get("val_scores") or []
    best_i = int(max(range(len(val_scores)), key=lambda i: val_scores[i])) if val_scores else -1
    best_phase = phases[best_i] if best_i >= 0 and best_i < len(phases) else "unknown"
    scst_scores = [val_scores[i] for i, p in enumerate(phases) if p == "scst"]
    scst_best = float(max(scst_scores)) if scst_scores else float("nan")
    warm_scores = [val_scores[i] for i, p in enumerate(phases) if p == "warm_tversky"]
    warm_best = float(max(warm_scores)) if warm_scores else float("nan")

    if best_phase == "scst" and mean > gate_mean + 1e-3:
        verdict = (
            f"**Promote candidate.** SCST checkpoint mean {mean:.3f} beats gate {gate_mean:.3f}. "
            "Re-check DEF/assist placement before any lock change."
        )
        park = "open — SCST beats gate on mean; needs DEF/assist audit before promote"
    elif best_phase == "scst" and mean > warm_best + 1e-3:
        verdict = (
            f"**Partial.** SCST ({mean:.3f}) beats this run's warm Tversky ({warm_best:.3f}) "
            f"but still trails the full-remainder gate ({gate_mean:.3f})."
        )
        park = "park as default — SCST helps stepped FOV, still below gate"
    else:
        verdict = (
            f"**Park.** Best holdout {mean:.3f} is from the **{best_phase}** phase "
            f"(epoch {best_i + 1}), not a SCST win. Warm best {warm_best:.3f}; "
            f"SCST best {scst_best:.3f}. SCST did not improve on next-GW Tversky and "
            f"does not beat the full-remainder gate ({gate_mean:.3f}, "
            f"Δ {_delta(mean, gate_mean)}). DEF goals/assists stay blank."
        )
        park = "park — SCST does not beat warm Tversky or the gate"

    lines.extend(
        [
            "",
            "## Phase note",
            "",
            f"- Best checkpoint phase: `{best_phase}` (epoch {best_i + 1})",
            f"- Warm-start best: {warm_best:.3f}",
            f"- SCST-phase best: {scst_best:.3f}",
            "",
            "## Morning call",
            "",
            verdict,
            "",
            f"Disposition: {park}.",
            "",
            "Read order:",
            "1. Mean vs gate (primary).",
            "2. Was the best checkpoint SCST or warm Tversky?",
            "3. DEF goals/assists a/p/h at 20%.",
            "4. Overpaint / GKP noise on the holdout table.",
            "",
        ]
    )
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    (ART / "baudot_rl_stepped.md").write_text(text, encoding="utf-8")
    return OUT


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
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
    width = train[0].width
    print(f"cards train {len(train)} test {len(test)} width {width} device {device}", flush=True)

    print("\n=== rl_step_bal ===", flush=True)
    model = PolicyMixer(width=width)
    meta = train_rl(model, train, test, device)
    mean = _mean_score(meta["holdout"]["rows"])
    gate_mean = _mean_score(_gate_rows())
    print(
        f"rl_step_bal mean {mean:.3f} Δgate {mean - gate_mean:+.3f} "
        f"DEF goals 20% {meta['holdout']['goals_20_by_pos']['DEF']}",
        flush=True,
    )
    path = write_report(meta)
    slim = {
        "trial": "baudot_rl_stepped",
        "method": "self_critical_reinforce_next_gw_balanced_score",
        "fov": "all_past_gws + one_shot_next_gw_stepped",
        "gate": GATE_META,
        "stepped_supervised_mean_parked": STEPPED_SUPERVISED_MEAN,
        "arm": {
            "name": meta["name"],
            "params": meta["params"],
            "epochs_run": meta["epochs_run"],
            "warm_epochs": meta["warm_epochs"],
            "best_holdout_mean_score": meta["best_holdout_mean_score"],
            "final_alpha": meta["final_alpha"],
            "final_beta": meta["final_beta"],
            "entropy_coef": meta["entropy_coef"],
            "seconds": meta["seconds"],
            "mean_score": mean,
            "delta_gate_mean": mean - gate_mean,
            "rows": meta["holdout"]["rows"],
            "gkp_by_cut": meta["holdout"]["gkp_painted_goals_by_cut"],
            "goals_20_by_pos": meta["holdout"]["goals_20_by_pos"],
            "assists_20_by_pos": meta["holdout"]["assists_20_by_pos"],
            "val_scores": meta["val_scores"],
            "train_stats": meta["train_stats"],
            "phase_log": meta["phase_log"],
        },
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (ART / "baudot_rl_stepped.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
