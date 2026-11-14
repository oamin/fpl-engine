"""Baudot card fixed — role paint heads vs locked gate (DEF goals / assists).

Gate failure under diagnosis: shared paint Linear never paints DEF goals
(~103 actual / 0 painted at 20% under the mixer family). Equal per-position
Tversky forced DEF volume but not hits, and reopened GKP noise — parked.

This trial keeps card, FOV, and learnable Tversky. Arms (in order):
  1. role_paint       — same mixer encoder; four role-specific paint heads,
                        role read from the card's role bits (not a sidecar).
  2. role_paint_dual  — (1) + separate learnable Tversky δ for goals vs assists.
  3. def_goal_aux     — (1) + auxiliary FN-heavy Tversky on DEF goals only
                        (λ=0.35, α=0.7, β=0.3). Not equal per-position.

Primary diagnostic: goals/assists by position (a/p/h), especially DEF goals.
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
    D,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GATE,
    GATE_META,
    GOAL_BITS,
    HEADER_BITS,
    HEADS,
    LR,
    MAX_GW,
    MAX_PLAYERS,
    MINS_BITS,
    ODDS_BITS,
    OPP_BITS,
    PLAYER_BITS,
    REPORTS,
    ROLE_BITS,
    ROLE_CODE,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    TVERSKY_DELTA_MAX,
    WHO_BITS,
    BaudotCard,
    BlockTokenMixer,
    _batch,
    _channel_slice,
    _player_pad,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    evaluate_mixer,
    fit_clubs,
    load_odds,
    remainder_view,
    tversky_loss,
)
from src.models.flashcard_card_score import balanced_score

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_role_paint.md"
POS_NAME = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}


def _role_codes(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """4×4 matrix of ROLE_CODE bit patterns, rows ordered GKP..FWD."""
    rows = [ROLE_CODE[i] for i in range(4)]
    return torch.tensor(rows, device=device, dtype=dtype)


def role_ids_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """Read role id (0..3) from each player's role bits on the card. Shape B,P."""
    body = bits[..., HEADER_BITS:].reshape(bits.shape[0], bits.shape[1], MAX_PLAYERS, PLAYER_BITS)
    # Role is constant across GWs on these cards; use GW0 pattern.
    role_bits = body[:, 0, :, :ROLE_BITS]  # B,P,4
    codes = _role_codes(bits.device, role_bits.dtype)  # 4,4
    ham = (role_bits[:, :, None, :] - codes[None, None, :, :]).abs().sum(-1)  # B,P,4
    return ham.argmin(dim=-1).clamp(0, 3)


class RolePaintMixer(BlockTokenMixer):
    """Shared time/squad mixer; one paint Linear per role, selected from role bits."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__(width=width, dropout=dropout)
        del self.paint
        self.paint_roles = nn.ModuleList(nn.Linear(D, EVENT_BITS) for _ in range(4))
        for pos, head in enumerate(self.paint_roles):
            nn.init.zeros_(head.weight)
            # Mild role prior: GKP goals strongly off; DEF goals less suppressed.
            bias = torch.full((EVENT_BITS,), -3.0)
            if pos == 0:
                bias[:GOAL_BITS] = -6.0
            elif pos == 1:
                bias[:GOAL_BITS] = -2.0
            nn.init.constant_(head.bias, 0.0)
            with torch.no_grad():
                head.bias.copy_(bias)

    def forward(self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
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
        role = role_ids_from_bits(bits)  # B,P
        # Stack role heads then gather.
        stack = torch.stack([head(tokens) for head in self.paint_roles], dim=0)  # 4,B,T,P,E
        idx = role[None, :, None, :, None].expand(1, b, t, p, EVENT_BITS)
        logits = stack.gather(0, idx).squeeze(0)
        ink = torch.sigmoid(logits)
        return ink.permute(0, 2, 1, 3)


class RolePaintDualTversky(RolePaintMixer):
    """Role paint heads + separate learnable δ for goal vs assist channels."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__(width=width, dropout=dropout)
        del self.tversky_logit
        self.tversky_logit_g = nn.Parameter(torch.tensor(-2.0))
        self.tversky_logit_a = nn.Parameter(torch.tensor(-2.0))

    def tversky_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean (α, β) for logging; training uses channel-specific weights."""
        dg = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit_g)
        da = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit_a)
        alpha = 0.5 - 0.5 * (dg + da)
        beta = 0.5 + 0.5 * (dg + da)
        return alpha, beta

    def channel_tversky_weights(self) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        dg = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit_g)
        da = TVERSKY_DELTA_MAX * torch.sigmoid(self.tversky_logit_a)
        return (0.5 - dg, 0.5 + dg), (0.5 - da, 0.5 + da)


def dual_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    alpha_g: torch.Tensor,
    beta_g: torch.Tensor,
    alpha_a: torch.Tensor,
    beta_a: torch.Tensor,
) -> torch.Tensor:
    live = live.to(pred.dtype)
    pred = pred * live
    target = target * live
    terms = []
    for sl, alpha, beta in (
        (slice(0, GOAL_BITS), alpha_g, beta_g),
        (slice(GOAL_BITS, EVENT_BITS), alpha_a, beta_a),
    ):
        p = pred[..., sl]
        y = target[..., sl]
        tp = (p * y).sum()
        fp = (p * (1.0 - y)).sum()
        fn = ((1.0 - p) * y).sum()
        terms.append(1.0 - (tp + 1e-5) / (tp + alpha * fn + beta * fp + 1e-5))
    return 0.5 * (terms[0] + terms[1])


DEF_AUX_LAMBDA = 0.35
DEF_AUX_ALPHA = 0.7  # FN-heavy on DEF goals only
DEF_AUX_BETA = 0.3


def def_goal_aux_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    is_def: torch.Tensor,
) -> torch.Tensor:
    """FN-heavy Tversky on DEF goal bits only. is_def: B,P bool."""
    mask = live & is_def[:, :, None, None]
    p = pred[..., :GOAL_BITS] * mask.to(pred.dtype)
    y = target[..., :GOAL_BITS] * mask.to(pred.dtype)
    tp = (p * y).sum()
    fp = (p * (1.0 - y)).sum()
    fn = ((1.0 - p) * y).sum()
    return 1.0 - (tp + 1e-5) / (tp + DEF_AUX_ALPHA * fn + DEF_AUX_BETA * fp + 1e-5)


def _def_mask_batch(cards: list[BaudotCard], device: torch.device) -> torch.Tensor:
    out = torch.zeros(len(cards), MAX_PLAYERS, dtype=torch.bool, device=device)
    for i, card in enumerate(cards):
        for row in range(card.n_players):
            if int(card.position[row]) == 1:
                out[i, row] = True
    return out


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


@torch.no_grad()
def position_split(
    model: nn.Module,
    cards: list[BaudotCard],
    device: torch.device,
    frac: float,
    channel: str,
) -> dict[str, dict[str, int]]:
    """Event-week counts by role: actual / painted / same-week hits."""
    sl = _channel_slice(channel)
    out = {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()}
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
            pos = POS_NAME.get(int(card.position[row]))
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


def train_arm(
    name: str,
    model: nn.Module,
    train: list[BaudotCard],
    test: list[BaudotCard],
    device: torch.device,
    dual: bool = False,
    def_aux: bool = False,
) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history: list[float] = []
    val_scores: list[float] = []
    ab_hist: list[dict] = []
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
            if dual:
                (ag, bg), (aa, ba) = model.channel_tversky_weights()
                loss = dual_tversky_loss(pred, target, live, ag, bg, aa, ba)
                ab_hist_entry = {
                    "alpha_g": float(ag.detach()),
                    "beta_g": float(bg.detach()),
                    "alpha_a": float(aa.detach()),
                    "beta_a": float(ba.detach()),
                }
            else:
                alpha, beta = model.tversky_weights()
                loss = tversky_loss(pred, target, live, alpha, beta)
                ab_hist_entry = {"alpha": float(alpha.detach()), "beta": float(beta.detach())}
            if def_aux:
                is_def = _def_mask_batch(pick, device)
                loss = loss + DEF_AUX_LAMBDA * def_goal_aux_loss(pred, target, live, is_def)
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
        ab_hist.append(ab_hist_entry)
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
        if dual:
            ab = ab_hist[-1]
            ab_s = (
                f"αg={ab['alpha_g']:.3f} βg={ab['beta_g']:.3f} "
                f"αa={ab['alpha_a']:.3f} βa={ab['beta_a']:.3f}"
            )
        else:
            ab_s = f"α={ab_hist[-1]['alpha']:.3f} β={ab_hist[-1]['beta']:.3f}"
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} loss {avg:.4f} {ab_s} "
            f"holdout_score {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate_mixer(model, test, device)
    best_ab = ab_hist[int(np.argmax(val_scores))] if val_scores else {}
    pos = {
        "goals_20": position_split(model, test, device, 0.2, "goals"),
        "assists_20": position_split(model, test, device, 0.2, "assists"),
        "goals_40": position_split(model, test, device, 0.4, "goals"),
        "assists_40": position_split(model, test, device, 0.4, "assists"),
    }
    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "best_holdout_mean_score": best_score,
        "final_ab": best_ab,
        "seconds": round(time.time() - t0, 1),
        "holdout": holdout,
        "position": pos,
        "history": history,
        "val_scores": val_scores,
    }


def _delta(arm_score: float, gate_score: float) -> str:
    return f"{arm_score - gate_score:+.3f}"


def _pos_cell(d: dict[str, dict[str, int]], pos: str) -> str:
    x = d[pos]
    return f"{x['actual']}/{x['painted']}/{x['hits']}"


def write_report(metas: list[dict], gate_pos: dict) -> Path:
    gate_rows = _gate_rows()
    gate_mean = float(np.mean([r["score"] for r in gate_rows]))
    lines = [
        "# Baudot role paint vs locked gate (DEF goals / assists)",
        "",
        "Card / FOV fixed. Shared paint head → role-specific paint heads (role from card bits).",
        "Equal per-position Tversky stays parked (forced DEF volume, no hits, GKP noise).",
        f"Gate method: `{GATE_META['method']}` · mean score {gate_mean:.3f}.",
        "",
        "| arm | params | epochs | mean score | Δ gate | seconds |",
        "|---|---:|---:|---:|---:|---:|",
        f"| gate (locked) | — | — | {gate_mean:.3f} | +0.000 | — |",
    ]
    for m in metas:
        mean = _mean_score(m["holdout"])
        lines.append(
            f"| {m['name']} | {m['params']} | {m['epochs_run']} | {mean:.3f} | "
            f"{_delta(mean, gate_mean)} | {m['seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Goals at 20% by position (actual / painted / hits)",
            "",
            "| arm | GKP | DEF | MID | FWD |",
            "|---|---|---|---|---|",
            f"| gate_replay | {_pos_cell(gate_pos['goals_20'], 'GKP')} | {_pos_cell(gate_pos['goals_20'], 'DEF')} | "
            f"{_pos_cell(gate_pos['goals_20'], 'MID')} | {_pos_cell(gate_pos['goals_20'], 'FWD')} |",
        ]
    )
    for m in metas:
        g = m["position"]["goals_20"]
        lines.append(
            f"| {m['name']} | {_pos_cell(g, 'GKP')} | {_pos_cell(g, 'DEF')} | "
            f"{_pos_cell(g, 'MID')} | {_pos_cell(g, 'FWD')} |"
        )
    lines.extend(
        [
            "",
            "## Assists at 20% by position (actual / painted / hits)",
            "",
            "| arm | GKP | DEF | MID | FWD |",
            "|---|---|---|---|---|",
            f"| gate_replay | {_pos_cell(gate_pos['assists_20'], 'GKP')} | {_pos_cell(gate_pos['assists_20'], 'DEF')} | "
            f"{_pos_cell(gate_pos['assists_20'], 'MID')} | {_pos_cell(gate_pos['assists_20'], 'FWD')} |",
        ]
    )
    for m in metas:
        a = m["position"]["assists_20"]
        lines.append(
            f"| {m['name']} | {_pos_cell(a, 'GKP')} | {_pos_cell(a, 'DEF')} | "
            f"{_pos_cell(a, 'MID')} | {_pos_cell(a, 'FWD')} |"
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
            "Primary call: does DEF goals leave 0/0, and do assist hits rise without "
            "GKP goal noise or a mean-score collapse vs the locked gate.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    (ART / "baudot_role_paint.md").write_text("\n".join(lines), encoding="utf-8")
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

    print("\n=== gate_replay (position baseline) ===", flush=True)
    gate_model = BlockTokenMixer(width=width)
    gate_meta = train_arm("gate_replay", gate_model, train, test, device, dual=False)
    gate_pos = gate_meta["position"]
    print(
        f"gate_replay mean {_mean_score(gate_meta['holdout']):.3f} "
        f"DEF goals 20% {gate_pos['goals_20']['DEF']}",
        flush=True,
    )

    arms: list[tuple[str, nn.Module, bool, bool]] = [
        ("1_role_paint", RolePaintMixer(width=width), False, False),
        ("2_role_paint_dual", RolePaintDualTversky(width=width), True, False),
        ("3_def_goal_aux", RolePaintMixer(width=width), False, True),
    ]
    metas = []
    for name, model, dual, def_aux in arms:
        print(f"\n=== {name} ===", flush=True)
        meta = train_arm(name, model, train, test, device, dual=dual, def_aux=def_aux)
        metas.append(meta)
        print(
            f"{name} mean {_mean_score(meta['holdout']):.3f} "
            f"DEF goals 20% {meta['position']['goals_20']['DEF']} "
            f"assists 20% DEF {meta['position']['assists_20']['DEF']}",
            flush=True,
        )

    path = write_report(metas, gate_pos)
    slim = {
        "trial": "baudot_role_paint",
        "gate": GATE_META,
        "gate_replay": {
            "mean_score": _mean_score(gate_meta["holdout"]),
            "position": gate_pos,
            "rows": gate_meta["holdout"]["rows"],
        },
        "arms": [
            {
                "name": m["name"],
                "params": m["params"],
                "epochs_run": m["epochs_run"],
                "best_holdout_mean_score": m["best_holdout_mean_score"],
                "final_ab": m["final_ab"],
                "seconds": m["seconds"],
                "mean_score": _mean_score(m["holdout"]),
                "delta_gate_mean": _mean_score(m["holdout"])
                - float(np.mean([r["score"] for r in _gate_rows()])),
                "rows": m["holdout"]["rows"],
                "position": m["position"],
            }
            for m in metas
        ],
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (ART / "baudot_role_paint.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
