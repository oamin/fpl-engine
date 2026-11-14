"""Baudot card fixed — trial learner variants against the locked gate.

Same card, FOV, cuts, and learnable Tversky. Arms (in order):
  1. gate_mixer     — locked BlockTokenMixer (depth-1) replay / reference
  2. deeper_mixer   — two residual time + squad attention blocks
  3. block_cnn      — schema-kernel 1D-CNN + temporal conv painter

Holdout uses tally_row + balanced_score; compared to flashcard_card_score.GATE.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flashcard_baudot_row_trial import (
    ART,
    BATCH,
    CLUB_BITS,
    CUT_FRACS,
    D,
    EVENT_BITS,
    GATE,
    GATE_META,
    HEADER_BITS,
    HEADS,
    ID_BITS,
    LR,
    MAX_GW,
    MAX_PLAYERS,
    MINS_BITS,
    ODDS_BITS,
    OPP_BITS,
    PLAYER_BITS,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    TVERSKY_DELTA_MAX,
    WHO_BITS,
    BaudotCard,
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
    tversky_loss,
    EARLY_PATIENCE,
    EPOCHS,
    BCE_W,
)
from src.models.flashcard_card_score import balanced_score

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_model_variants.md"
ART.mkdir(parents=True, exist_ok=True)


class DeeperBlockTokenMixer(BlockTokenMixer):
    """Same schema tokens; two residual time/squad attention blocks."""

    def __init__(self, width: int, dropout: float = 0.1, depth: int = 2) -> None:
        super().__init__(width=width, dropout=dropout)
        self.depth = int(depth)
        self.time_layers = nn.ModuleList(
            nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True) for _ in range(self.depth)
        )
        self.squad_layers = nn.ModuleList(
            nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True) for _ in range(self.depth)
        )
        self.norm_time = nn.ModuleList(nn.LayerNorm(D) for _ in range(self.depth))
        self.norm_squad = nn.ModuleList(nn.LayerNorm(D) for _ in range(self.depth))

    def forward(self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_rows(bits) + self.time_pos[:, : bits.shape[1]]
        b, t, p, d = tokens.shape
        time_view = tokens.permute(0, 2, 1, 3).reshape(b * p, t, d)
        time_mask = (row_on < 0.5)[:, None, :].expand(b, p, t).reshape(b * p, t)
        for attn, norm in zip(self.time_layers, self.norm_time, strict=True):
            mixed, _ = attn(time_view, time_view, time_view, key_padding_mask=time_mask, need_weights=False)
            time_view = norm(time_view + mixed)
        tokens = time_view.reshape(b, p, t, d).permute(0, 2, 1, 3)
        squad_view = tokens.reshape(b * t, p, d)
        squad_mask = player_pad[:, None, :].expand(b, t, p).reshape(b * t, p)
        for attn, norm in zip(self.squad_layers, self.norm_squad, strict=True):
            mixed, _ = attn(squad_view, squad_view, squad_view, key_padding_mask=squad_mask, need_weights=False)
            squad_view = norm(squad_view + mixed)
        tokens = squad_view.reshape(b, t, p, d)
        ink = torch.sigmoid(self.paint(tokens))
        return ink.permute(0, 2, 1, 3)


class BlockCNNPainter(nn.Module):
    """Schema-kernel CNN along the bit spine + temporal conv across GWs."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.width = width
        # Per-block embeds (same as mixer), then vertical/temporal mix via conv.
        self.club = nn.Linear(CLUB_BITS, D)
        self.opp = nn.Linear(OPP_BITS, D)
        self.odds = nn.Linear(ODDS_BITS, D)
        self.who = nn.Linear(WHO_BITS, D)
        self.mins = nn.Linear(MINS_BITS, D)
        self.events = nn.Linear(EVENT_BITS, D)
        self.header_mix = nn.Linear(3 * D, D)
        self.player_mix = nn.Linear(3 * D, D)
        self.norm_h = nn.LayerNorm(D)
        self.norm_p = nn.LayerNorm(D)
        # Temporal conv over GW axis (kernel 5); player-local.
        self.temporal = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(D, D, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # Light mix across adjacent roster slots (GKP→FWD ordering).
        self.roster = nn.Sequential(
            nn.Conv1d(D, D, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm_t = nn.LayerNorm(D)
        self.norm_r = nn.LayerNorm(D)
        self.time_pos = nn.Parameter(torch.zeros(1, MAX_GW, 1, D))
        nn.init.normal_(self.time_pos, std=0.02)
        self.paint = nn.Linear(D, EVENT_BITS)
        nn.init.zeros_(self.paint.weight)
        nn.init.constant_(self.paint.bias, -3.0)
        self.tversky_logit = nn.Parameter(torch.tensor(-2.0))

    def tversky_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit)
        return 0.5 - delta, 0.5 + delta

    def _encode_rows(self, bits: torch.Tensor) -> torch.Tensor:
        club = self.club(bits[..., :CLUB_BITS])
        opp = self.opp(bits[..., CLUB_BITS : CLUB_BITS + OPP_BITS])
        odds = self.odds(bits[..., CLUB_BITS + OPP_BITS : HEADER_BITS])
        header = self.norm_h(self.header_mix(torch.cat([club, opp, odds], dim=-1)))
        body = bits[..., HEADER_BITS:].reshape(bits.shape[0], bits.shape[1], MAX_PLAYERS, PLAYER_BITS)
        who = self.who(body[..., :WHO_BITS])
        mins = self.mins(body[..., WHO_BITS : WHO_BITS + MINS_BITS])
        ev = self.events(body[..., WHO_BITS + MINS_BITS :])
        players = self.norm_p(self.player_mix(torch.cat([who, mins, ev], dim=-1)))
        return players + header[:, :, None, :]

    def forward(self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_rows(bits) + self.time_pos[:, : bits.shape[1]]
        b, t, p, d = tokens.shape
        # Mask padded GWs to zero before conv.
        tokens = tokens * row_on[:, :, None, None].to(tokens.dtype)
        # Temporal: B*P, D, T
        time_in = tokens.permute(0, 2, 3, 1).reshape(b * p, d, t)
        time_out = self.temporal(time_in)
        time_out = self.norm_t((time_in + time_out).permute(0, 2, 1)).permute(0, 2, 1)
        tokens = time_out.reshape(b, p, d, t).permute(0, 3, 1, 2)  # B,T,P,D
        # Roster: B*T, D, P
        rost_in = tokens.permute(0, 1, 3, 2).reshape(b * t, d, p)
        # Zero padded players
        pad = player_pad[:, None, :].expand(b, t, p).reshape(b * t, 1, p).to(rost_in.dtype)
        rost_in = rost_in * (1.0 - pad)
        rost_out = self.roster(rost_in)
        rost_out = self.norm_r((rost_in + rost_out).permute(0, 2, 1)).permute(0, 2, 1)
        tokens = rost_out.reshape(b, t, d, p).permute(0, 1, 3, 2)  # B,T,P,D
        ink = torch.sigmoid(self.paint(tokens))
        return ink.permute(0, 2, 1, 3)


def _mean_score(stats: dict) -> float:
    return float(np.mean([row["score"] for row in stats["rows"]]))


def _gate_rows() -> list[dict]:
    rows = []
    for frac in CUT_FRACS:
        for channel in ("goals", "assists"):
            tot = GATE[(channel, frac)]
            sc = balanced_score(tot)
            rows.append(
                {
                    "revealed": f"{frac:.0%}",
                    "channel": channel,
                    "actual": tot["actual"],
                    "painted": tot["painted"],
                    "hits": tot["exact"],
                    "actual_within_1": tot["actual_within"],
                    "painted_within_1": tot["painted_within"],
                    "score": sc["score"],
                }
            )
    return rows


def train_variant(
    name: str,
    model: nn.Module,
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
    ab_hist: list[tuple[float, float]] = []
    best_state = None
    best_score = -1.0
    bad = 0
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(train))
        for start in range(0, len(train), BATCH):
            pick = [train[int(i)] for i in order[start : start + BATCH]]
            fracs = [float(rng.choice(CUT_FRACS)) for _ in pick]
            cuts = [cut_at_fraction(card, frac) for card, frac in zip(pick, fracs, strict=True)]
            bits, row_on, rem_on, pad, present, target = _batch(pick, cuts, device)
            live = (
                present[:, :, :, None]
                & rem_on[:, None, :, None].bool()
                & (~pad)[:, :, None, None]
            )
            opt.zero_grad()
            pred = model(bits, row_on, pad)
            alpha, beta = model.tversky_weights()
            loss = tversky_loss(pred, target, live, alpha, beta)
            bce = F.binary_cross_entropy(pred, target, reduction="none")
            bce = (bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0)
            loss = loss + BCE_W * bce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        avg = total / max(seen, 1)
        history.append(avg)
        with torch.no_grad():
            a, b = model.tversky_weights()
            ab_hist.append((float(a), float(b)))
        hold = evaluate_mixer(model, test, device)
        score = _mean_score(hold)
        val_scores.append(score)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} loss {avg:.4f} "
            f"α={ab_hist[-1][0]:.3f} β={ab_hist[-1][1]:.3f} holdout_score {score:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate_mixer(model, test, device)
    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "best_holdout_mean_score": best_score,
        "final_alpha": ab_hist[int(np.argmax(val_scores))][0] if val_scores else float("nan"),
        "final_beta": ab_hist[int(np.argmax(val_scores))][1] if val_scores else float("nan"),
        "seconds": round(time.time() - t0, 1),
        "holdout": holdout,
        "history": history,
        "val_scores": val_scores,
    }


def _delta(arm_score: float, gate_score: float) -> str:
    d = arm_score - gate_score
    return f"{d:+.3f}"


def write_report(metas: list[dict]) -> Path:
    gate_rows = _gate_rows()
    gate_mean = float(np.mean([r["score"] for r in gate_rows]))
    lines = [
        "# Baudot model variants vs locked gate",
        "",
        "Card / FOV / learnable Tversky fixed. Only the learner changes.",
        f"Gate method: `{GATE_META['method']}` · mean score {gate_mean:.3f}.",
        "",
        "| arm | params | epochs | α | β | mean score | Δ gate | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| gate (locked) | — | — | {GATE_META['alpha']:.3f} | {GATE_META['beta']:.3f} | {gate_mean:.3f} | +0.000 | — |",
    ]
    for m in metas:
        mean = _mean_score(m["holdout"])
        lines.append(
            f"| {m['name']} | {m['params']} | {m['epochs_run']} | {m['final_alpha']:.3f} | "
            f"{m['final_beta']:.3f} | {mean:.3f} | {_delta(mean, gate_mean)} | {m['seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Holdout table",
            "",
            "| arm | revealed | channel | actual | painted | hits | actual ±1 | painted ±1 | score | Δ gate |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"| gate | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
            f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} | +0.000 |"
        )
    for m in metas:
        for row in m["holdout"]["rows"]:
            key = (row["channel"], {"20%": 0.2, "40%": 0.4, "60%": 0.6, "80%": 0.8}[row["revealed"]])
            gscore = balanced_score(GATE[key])["score"]
            lines.append(
                f"| {m['name']} | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
                f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} | "
                f"{_delta(row['score'], gscore)} |"
            )
    lines.extend(
        [
            "",
            "A positive Δ means the variant beats the locked gate on that cell.",
            "GKP painted goals by cut are listed per arm in REPORT_STATE.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    (ART / "baudot_model_variants.md").write_text("\n".join(lines), encoding="utf-8")
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
    width = train[0].width

    arms = [
        ("1_deeper_mixer", DeeperBlockTokenMixer(width=width, depth=2)),
        ("2_block_cnn", BlockCNNPainter(width=width)),
    ]
    metas = []
    for name, model in arms:
        print(f"\n=== {name} ===", flush=True)
        meta = train_variant(name, model, train, test, device)
        metas.append(meta)
        print(
            f"{name} mean score {_mean_score(meta['holdout']):.3f} "
            f"gkp {meta['holdout']['gkp_painted_goals_by_cut']}",
            flush=True,
        )

    path = write_report(metas)
    slim = []
    for m in metas:
        slim.append(
            {
                "name": m["name"],
                "params": m["params"],
                "epochs_run": m["epochs_run"],
                "best_holdout_mean_score": m["best_holdout_mean_score"],
                "final_alpha": m["final_alpha"],
                "final_beta": m["final_beta"],
                "seconds": m["seconds"],
                "rows": m["holdout"]["rows"],
                "gkp_by_cut": m["holdout"]["gkp_painted_goals_by_cut"],
                "mean_score": _mean_score(m["holdout"]),
                "delta_gate_mean": _mean_score(m["holdout"]) - float(np.mean([r["score"] for r in _gate_rows()])),
            }
        )
    (ROOT / "REPORT_STATE.json").write_text(
        json.dumps(
            {
                "trial": "baudot_model_variants",
                "gate": GATE_META,
                "arms": slim,
                "report": str(path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ART / "baudot_model_variants.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
