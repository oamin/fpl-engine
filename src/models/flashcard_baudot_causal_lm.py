"""Baudot causal block-LM — next event-block given a 1D club-season stream.

Collapse each club-season into a causal sequence of schema blocks:
  [HDR_gw, PL0_gw, PL1_gw, …]_gw=0 … then the next GW, …

HDR = club‖opp‖odds.  PL = who‖mins‖events.
Past GW event blocks stay visible. Remainder event blocks are MASKED in the
input; the model predicts those event bits only (event-space next-block).

Tiny GPT-style stack (causal transformer), learnable Tversky on event blocks.
Same Baudot card as the locked gate. Judged on GATE tallies + DEF a/p/h.

This is the disciplined version of “LLM next-dot”: block tokens, not bit soup;
event-space loss only; full-past + masked-remainder FOV (gate-comparable).
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
    BCE_W,
    CLUB_BITS,
    CUT_FRACS,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GATE,
    GATE_META,
    GOAL_BITS,
    HEADER_BITS,
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
    _channel_slice,
    _player_pad,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    event_target,
    fit_clubs,
    load_odds,
    tversky_loss,
)
from src.models.flashcard_card_score import balanced_score, tally_row

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_causal_lm.md"
POS_NAME = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}

# Sequence geometry: one HDR + one token per roster slot, per GW.
TOKENS_PER_GW = 1 + MAX_PLAYERS  # 41
MAX_SEQ = MAX_GW * TOKENS_PER_GW  # 1640
HDR_DIM = HEADER_BITS  # 28
PL_DIM = PLAYER_BITS  # 27

# Small causal stack (LLM-shaped, domain-sized).
D_LM = 96
N_LAYER = 4
N_HEAD = 4
DROPOUT = 0.1
LM_BATCH = 4  # long sequences


def _causal_mask(n: int, device: torch.device) -> torch.Tensor:
    """True where attention is forbidden (PyTorch flex / src mask style for MHA)."""
    return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)


class CausalBlockLM(nn.Module):
    """GPT-style causal transformer over Baudot header/player blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.hdr_in = nn.Linear(HDR_DIM, D_LM)
        self.pl_in = nn.Linear(PL_DIM, D_LM)
        self.kind = nn.Embedding(2, D_LM)  # 0=hdr, 1=player
        self.pos = nn.Embedding(MAX_SEQ, D_LM)
        self.gw_pos = nn.Embedding(MAX_GW, D_LM)
        layer = nn.TransformerEncoderLayer(
            d_model=D_LM,
            nhead=N_HEAD,
            dim_feedforward=D_LM * 4,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=N_LAYER)
        self.norm = nn.LayerNorm(D_LM)
        self.evt_head = nn.Linear(D_LM, EVENT_BITS)
        nn.init.zeros_(self.evt_head.weight)
        nn.init.constant_(self.evt_head.bias, -3.0)
        self.tversky_logit = nn.Parameter(torch.tensor(-2.0))

    def tversky_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        delta = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit)
        return 0.5 - delta, 0.5 + delta

    def forward(
        self,
        hdr: torch.Tensor,
        players: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """hdr: B,G,H  players: B,G,P,Pl  → event logits B,G,P,E (pre-sigmoid).

        Builds the causal 1D stream GW-major: HDR, PL0..PL{P-1}, next GW, …
        """
        b, g, p, _ = players.shape
        device = players.device
        # Token stream B, G*(1+P), D
        hdr_tok = self.hdr_in(hdr) + self.kind(torch.zeros(b, g, dtype=torch.long, device=device))
        hdr_tok = hdr_tok + self.gw_pos(torch.arange(g, device=device))[None, :, :]
        pl_tok = self.pl_in(players) + self.kind(torch.ones(b, g, p, dtype=torch.long, device=device))
        pl_tok = pl_tok + self.gw_pos(torch.arange(g, device=device))[None, :, None, :]
        # Interleave: for each gw, hdr then players
        pieces = []
        for gi in range(g):
            pieces.append(hdr_tok[:, gi : gi + 1, :])
            pieces.append(pl_tok[:, gi, :, :])
        tok = torch.cat(pieces, dim=1)  # B, G*(1+P), D
        n = tok.shape[1]
        idx = torch.arange(n, device=device)
        tok = tok + self.pos(idx)[None, :, :]
        # Pad mask: True = ignore. seq_len is n_fixtures * TOKENS_PER_GW
        pad = idx[None, :] >= seq_len[:, None]
        causal = _causal_mask(n, device)
        out = self.blocks(tok, mask=causal, src_key_padding_mask=pad)
        out = self.norm(out)
        # Gather player token outputs: positions 1..P within each GW block
        # layout index for gw gi, player pi: gi * TOKENS_PER_GW + 1 + pi
        gather = []
        for gi in range(g):
            base = gi * TOKENS_PER_GW + 1
            gather.append(out[:, base : base + p, :])
        pl_out = torch.stack(gather, dim=1)  # B,G,P,D
        return self.evt_head(pl_out)  # B,G,P,E logits


def _pack_card(card: BaudotCard, cut: int | None) -> dict[str, np.ndarray]:
    """Pack bits into hdr/players; mask event bits for gw >= cut (None = no mask)."""
    g, p = MAX_GW, MAX_PLAYERS
    hdr = np.zeros((g, HDR_DIM), dtype=np.float32)
    players = np.zeros((g, p, PL_DIM), dtype=np.float32)
    target = np.zeros((g, p, EVENT_BITS), dtype=np.float32)
    evt_live = np.zeros((g, p), dtype=bool)
    for gw in range(card.n_fixtures):
        row = card.bits[gw]
        hdr[gw] = row[:HEADER_BITS]
        body = row[HEADER_BITS:].reshape(MAX_PLAYERS, PLAYER_BITS)
        players[gw] = body
        target[gw] = event_target(card, gw)
        if cut is not None and gw >= cut:
            # Mask event region in the input stream (who+mins kept).
            players[gw, :, WHO_BITS + MINS_BITS :] = 0.0
            for row_i in range(card.n_players):
                if card.present[row_i, gw] and int(card.position[row_i]) >= 0 and int(card.player_id[row_i]) > 0:
                    evt_live[gw, row_i] = True
        elif cut is None:
            for row_i in range(card.n_players):
                if card.present[row_i, gw] and int(card.position[row_i]) >= 0 and int(card.player_id[row_i]) > 0:
                    evt_live[gw, row_i] = True
    seq_len = int(card.n_fixtures) * TOKENS_PER_GW
    return {
        "hdr": hdr,
        "players": players,
        "target": target,
        "evt_live": evt_live,
        "seq_len": np.array(seq_len, dtype=np.int64),
        "n_fixtures": np.array(card.n_fixtures, dtype=np.int64),
    }


def _batch_pack(cards: list[BaudotCard], cuts: list[int | None], device: torch.device):
    packs = [_pack_card(card, cut) for card, cut in zip(cards, cuts, strict=True)]
    return (
        torch.from_numpy(np.stack([p["hdr"] for p in packs])).to(device),
        torch.from_numpy(np.stack([p["players"] for p in packs])).to(device),
        torch.from_numpy(np.stack([p["target"] for p in packs])).to(device),
        torch.from_numpy(np.stack([p["evt_live"] for p in packs])).to(device),
        torch.from_numpy(np.stack([p["seq_len"] for p in packs])).to(device),
    )


def _mean_score(rows: list[dict]) -> float:
    return float(np.mean([r["score"] for r in rows]))


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


@torch.no_grad()
def evaluate(model: CausalBlockLM, cards: list[BaudotCard], device: torch.device) -> dict:
    model.eval()
    rows: list[dict] = []
    pos = {
        "goals_20": {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()},
        "assists_20": {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()},
    }
    for frac in CUT_FRACS:
        for channel in ("goals", "assists"):
            totals = {key: 0 for key in ("actual", "painted", "exact", "actual_within", "painted_within")}
            sl = _channel_slice(channel)
            for card in cards:
                cut = cut_at_fraction(card, frac)
                hdr, players, target, evt_live, seq_len = _batch_pack([card], [cut], device)
                logits = model(hdr, players, seq_len)[0]  # G,P,E
                pred = torch.sigmoid(logits).cpu().numpy()
                pad = _player_pad(card)
                for row in range(card.n_players):
                    if pad[row]:
                        continue
                    cols = [
                        gw
                        for gw in range(cut, card.n_fixtures)
                        if card.present[row, gw]
                    ]
                    if not cols:
                        continue
                    actual = np.array(
                        [int(event_target(card, gw)[row, sl].sum()) for gw in cols],
                        dtype=np.int64,
                    )
                    painted = np.array(
                        [int((pred[gw, row, sl] >= 0.5).sum()) for gw in cols],
                        dtype=np.int64,
                    )
                    part = tally_row(card.gw[cols], actual, painted)
                    for key in totals:
                        totals[key] += part[key]
                    if frac == 0.2:
                        pname = POS_NAME.get(int(card.position[row]))
                        if pname is not None:
                            bucket = pos["goals_20" if channel == "goals" else "assists_20"]
                            for a, pnt in zip(actual, painted, strict=True):
                                ae, pe = int(a > 0), int(pnt > 0)
                                bucket[pname]["actual"] += ae
                                bucket[pname]["painted"] += pe
                                bucket[pname]["hits"] += int(ae and pe)
            score = balanced_score(totals)
            rows.append(
                {
                    "revealed": f"{frac:.0%}",
                    "channel": channel,
                    "actual": totals["actual"],
                    "painted": totals["painted"],
                    "hits": totals["exact"],
                    "actual_within_1": totals["actual_within"],
                    "painted_within_1": totals["painted_within"],
                    "score": score["score"],
                }
            )
    return {"rows": rows, "position": pos}


def train_lm(
    model: CausalBlockLM,
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
        for start in range(0, len(train), LM_BATCH):
            pick = [train[int(i)] for i in order[start : start + LM_BATCH]]
            cuts = [cut_at_fraction(card, float(rng.choice(CUT_FRACS))) for card in pick]
            hdr, players, target, evt_live, seq_len = _batch_pack(pick, cuts, device)
            live = evt_live[:, :, :, None]  # B,G,P,1
            # reshape to B,P,G,E for tversky_loss which expects B,P,T,E
            opt.zero_grad()
            logits = model(hdr, players, seq_len)  # B,G,P,E
            pred = torch.sigmoid(logits).permute(0, 2, 1, 3)  # B,P,G,E
            tgt = target.permute(0, 2, 1, 3)
            live_bp = live.permute(0, 2, 1, 3)
            alpha, beta = model.tversky_weights()
            loss = tversky_loss(pred, tgt, live_bp, alpha, beta)
            bce = F.binary_cross_entropy(pred, tgt, reduction="none")
            bce = (bce * live_bp.to(pred.dtype)).sum() / live_bp.to(pred.dtype).sum().clamp_min(1.0)
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
        hold = evaluate(model, test, device)
        score = _mean_score(hold["rows"])
        val_scores.append(score)
        improved = score > best_score + 1e-4
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(
            f"causal_lm epoch {epoch + 1}/{EPOCHS} loss {avg:.4f} "
            f"α={ab_hist[-1][0]:.3f} β={ab_hist[-1][1]:.3f} holdout_score {score:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"causal_lm early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate(model, test, device)
    return {
        "name": "causal_block_lm",
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "best_holdout_mean_score": best_score,
        "final_alpha": ab_hist[int(np.argmax(val_scores))][0] if val_scores else float("nan"),
        "final_beta": ab_hist[int(np.argmax(val_scores))][1] if val_scores else float("nan"),
        "seconds": round(time.time() - t0, 1),
        "holdout": holdout,
        "d_model": D_LM,
        "n_layer": N_LAYER,
        "tokens_per_gw": TOKENS_PER_GW,
    }


def _delta(a: float, b: float) -> str:
    return f"{a - b:+.3f}"


def _pos_cell(d: dict, pos: str) -> str:
    x = d[pos]
    return f"{x['actual']}/{x['painted']}/{x['hits']}"


def write_report(meta: dict) -> Path:
    gate_rows = _gate_rows()
    gate_mean = _mean_score(gate_rows)
    mean = _mean_score(meta["holdout"]["rows"])
    hold = meta["holdout"]
    lines = [
        "# Baudot causal block-LM vs locked gate",
        "",
        "1D club-season stream of schema blocks (HDR + player blocks per GW).",
        "Causal transformer; loss only on **masked remainder event blocks**.",
        "Same Baudot card and learnable Tversky; FOV = full past + masked remainder events.",
        "",
        f"Model: {meta['n_layer']} layers, d={meta['d_model']}, {meta['tokens_per_gw']} tokens/GW.",
        f"Gate: `{GATE_META['method']}` · mean {gate_mean:.3f}.",
        "",
        "| arm | params | epochs | α | β | mean score | Δ gate | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| gate (bidirectional mixer) | — | — | {GATE_META['alpha']:.3f} | {GATE_META['beta']:.3f} | "
        f"{gate_mean:.3f} | +0.000 | — |",
        f"| causal_block_lm | {meta['params']} | {meta['epochs_run']} | {meta['final_alpha']:.3f} | "
        f"{meta['final_beta']:.3f} | {mean:.3f} | {_delta(mean, gate_mean)} | {meta['seconds']} |",
        "",
        "## Goals at 20% by position (actual / painted / hits)",
        "",
        "| arm | GKP | DEF | MID | FWD |",
        "|---|---|---|---|---|",
        f"| causal_block_lm | {_pos_cell(hold['position']['goals_20'], 'GKP')} | "
        f"{_pos_cell(hold['position']['goals_20'], 'DEF')} | "
        f"{_pos_cell(hold['position']['goals_20'], 'MID')} | "
        f"{_pos_cell(hold['position']['goals_20'], 'FWD')} |",
        "",
        "## Assists at 20% by position (actual / painted / hits)",
        "",
        "| arm | GKP | DEF | MID | FWD |",
        "|---|---|---|---|---|",
        f"| causal_block_lm | {_pos_cell(hold['position']['assists_20'], 'GKP')} | "
        f"{_pos_cell(hold['position']['assists_20'], 'DEF')} | "
        f"{_pos_cell(hold['position']['assists_20'], 'MID')} | "
        f"{_pos_cell(hold['position']['assists_20'], 'FWD')} |",
        "",
        "## Holdout table",
        "",
        "| arm | revealed | channel | actual | painted | hits | actual ±1 | painted ±1 | score | Δ gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in gate_rows:
        lines.append(
            f"| gate | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
            f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} | +0.000 |"
        )
    for row in hold["rows"]:
        key = (row["channel"], {"20%": 0.2, "40%": 0.4, "60%": 0.6, "80%": 0.8}[row["revealed"]])
        gscore = balanced_score(GATE[key])["score"]
        lines.append(
            f"| causal_block_lm | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
            f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} | "
            f"{_delta(row['score'], gscore)} |"
        )
    lines.extend(
        [
            "",
            "Primary call: mean vs gate, and whether DEF goals/assists leave 0/0.",
            "",
        ]
    )
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    (ART / "baudot_causal_lm.md").write_text(text, encoding="utf-8")
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

    print(
        f"\n=== causal_block_lm d={D_LM} layers={N_LAYER} "
        f"seq≈{MAX_SEQ} device={device} ===",
        flush=True,
    )
    model = CausalBlockLM()
    meta = train_lm(model, train, test, device)
    print(
        f"causal_block_lm mean {_mean_score(meta['holdout']['rows']):.3f} "
        f"DEF goals {meta['holdout']['position']['goals_20']['DEF']}",
        flush=True,
    )
    path = write_report(meta)
    gate_mean = _mean_score(_gate_rows())
    slim = {
        "trial": "baudot_causal_lm",
        "fov": "all_past_gws + masked_remainder_event_blocks_causal",
        "gate": GATE_META,
        "arm": {
            "name": meta["name"],
            "params": meta["params"],
            "d_model": meta["d_model"],
            "n_layer": meta["n_layer"],
            "tokens_per_gw": meta["tokens_per_gw"],
            "epochs_run": meta["epochs_run"],
            "mean_score": _mean_score(meta["holdout"]["rows"]),
            "delta_gate_mean": _mean_score(meta["holdout"]["rows"]) - gate_mean,
            "final_alpha": meta["final_alpha"],
            "final_beta": meta["final_beta"],
            "seconds": meta["seconds"],
            "rows": meta["holdout"]["rows"],
            "position": meta["holdout"]["position"],
        },
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (ART / "baudot_causal_lm.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
