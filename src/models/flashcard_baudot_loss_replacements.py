"""Baudot loss replacements vs locked learnable Tversky gate.

Same card, BlockTokenMixer, FOV (full past + one-shot remainder), seed.
Only the training objective changes.

Arms (in order):
  1. listwise_bal  — batch listwise rank on presence logits (bit-0) + smooth
                     balanced_score + light thermometer-tail BCE
  2. calib_theta   — proper BCE on σ(z−θ) + learned channel thresholds θ_G/θ_A
                     via smooth balanced_score (ranking vs cut decoupled)

Primary morning table: mean vs gate, DEF/assist a/p/h at 20%.
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
    CUT_FRACS,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GATE,
    GATE_META,
    GOAL_BITS,
    LR,
    MAX_GW,
    MAX_PLAYERS,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
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
    event_target,
    fit_clubs,
    load_odds,
    remainder_view,
)
from src.models.flashcard_card_score import balanced_score, tally_row

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_loss_replacements.md"
POS_NAME = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}

# Loss-1 hyperparameters (overnight memo).
LIST_GAMMA = 4.0
LAM_BAL = 0.75
LAM_TAIL = 0.25
BAL_EPS = 1.0

# Loss-2 hyperparameters.
THETA_INIT = 2.0
PAINT_BIAS_INIT = -1.0
SHARP_TAU = 5.0


class LogitMixer(BlockTokenMixer):
    """Same mixer; expose pre-sigmoid logits for ranking losses."""

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
        return self.paint(tokens).permute(0, 2, 1, 3)  # B,P,T,E logits

    def forward(self, bits: torch.Tensor, row_on: torch.Tensor, player_pad: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(bits, row_on, player_pad))


class CalibThetaMixer(LogitMixer):
    """Logit mixer + learned presence thresholds per channel."""

    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__(width=width, dropout=dropout)
        nn.init.constant_(self.paint.bias, PAINT_BIAS_INIT)
        self.theta_g = nn.Parameter(torch.tensor(THETA_INIT))
        self.theta_a = nn.Parameter(torch.tensor(THETA_INIT))

    def channel_theta(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.theta_g, self.theta_a

    def presence_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """σ(z − θ) on bit-0 of each channel; other bits σ(z). Shape B,P,T,E."""
        out = torch.sigmoid(logits)
        tg, ta = self.theta_g, self.theta_a
        out = out.clone()
        out[..., 0] = torch.sigmoid(logits[..., 0] - tg)
        out[..., GOAL_BITS] = torch.sigmoid(logits[..., GOAL_BITS] - ta)
        return out


def _smooth_balanced(p: torch.Tensor, y: torch.Tensor, live: torch.Tensor, eps: float = BAL_EPS) -> torch.Tensor:
    """1 − soft F1 × volume on one presence channel. p,y,live: same shape."""
    m = live.to(p.dtype)
    p = p * m
    y = y * m
    tp = (p * y).sum()
    a = y.sum()
    painted = p.sum()
    f1 = (2.0 * tp + eps) / (a + painted + eps)
    vol = (torch.minimum(a, painted) + eps) / (torch.maximum(a, painted) + eps)
    return 1.0 - f1 * vol


def _tail_bce(logits: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """BCE on thermometer bits k=1,2 for goals and assists."""
    m = live.to(logits.dtype)
    terms = []
    for base in (0, GOAL_BITS):
        for k in (1, 2):
            if base + k >= EVENT_BITS:
                continue
            p = torch.sigmoid(logits[..., base + k])
            y = target[..., base + k]
            bce = F.binary_cross_entropy(p, y, reduction="none")
            terms.append((bce * m).sum() / m.sum().clamp_min(1.0))
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def listwise_bal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    gamma: float = LIST_GAMMA,
    lam_bal: float = LAM_BAL,
    lam_tail: float = LAM_TAIL,
) -> torch.Tensor:
    """Batch listwise on bit-0 + smooth BalScore + tail BCE."""
    m = live[..., 0]  # B,P,T
    list_terms = []
    bal_terms = []
    for bit in (0, GOAL_BITS):
        z = logits[..., bit]
        y = target[..., bit]
        z_flat = z[m]
        y_flat = y[m]
        if float(y_flat.sum()) > 0:
            log_r = F.log_softmax(z_flat / gamma, dim=0)
            y_tilde = y_flat / y_flat.sum().clamp_min(1.0)
            list_terms.append(-(y_tilde * log_r).sum())
        else:
            list_terms.append(z.new_zeros(()))
        p = torch.sigmoid(z)
        bal_terms.append(_smooth_balanced(p, y, m))
    loss_list = torch.stack(list_terms).mean()
    loss_bal = torch.stack(bal_terms).mean()
    loss_tail = _tail_bce(logits, target, live[..., 0])
    return loss_list + lam_bal * loss_bal + lam_tail * loss_tail


def calib_theta_loss(
    model: CalibThetaMixer,
    logits: torch.Tensor,
    target: torch.Tensor,
    live: torch.Tensor,
    lam_tail: float = LAM_TAIL,
) -> torch.Tensor:
    """Proper BCE on σ(z−θ) + BalScore on sharp soft decisions for θ."""
    m = live[..., 0]
    tg, ta = model.channel_theta()
    # Calibration path: network sees detached θ.
    q_g = torch.sigmoid(logits[..., 0] - tg.detach())
    q_a = torch.sigmoid(logits[..., GOAL_BITS] - ta.detach())
    y_g = target[..., 0]
    y_a = target[..., GOAL_BITS]
    bce_g = F.binary_cross_entropy(q_g, y_g, reduction="none")
    bce_a = F.binary_cross_entropy(q_a, y_a, reduction="none")
    loss_cal = ((bce_g * m).sum() + (bce_a * m).sum()) / (2.0 * m.sum().clamp_min(1.0))
    # Threshold path: θ trainable; logits detached.
    z_g = logits[..., 0].detach()
    z_a = logits[..., GOAL_BITS].detach()
    d_g = torch.sigmoid(SHARP_TAU * (z_g - tg))
    d_a = torch.sigmoid(SHARP_TAU * (z_a - ta))
    loss_bal = 0.5 * (
        _smooth_balanced(d_g, y_g, m) + _smooth_balanced(d_a, y_a, m)
    )
    loss_tail = _tail_bce(logits, target, m)
    return loss_cal + loss_bal + lam_tail * loss_tail


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
def position_split(
    model: nn.Module,
    cards: list[BaudotCard],
    device: torch.device,
    frac: float,
    channel: str,
    use_theta: bool = False,
) -> dict[str, dict[str, int]]:
    sl = _channel_slice(channel)
    out = {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()}
    model.eval()
    for card in cards:
        cut = cut_at_fraction(card, frac)
        view = remainder_view(card, cut)
        pad = _player_pad(card)
        bits = torch.from_numpy(view["bits"][None]).to(device)
        row_on = torch.from_numpy(view["row_on"][None]).to(device)
        pad_t = torch.from_numpy(pad[None]).to(device)
        if use_theta:
            assert isinstance(model, CalibThetaMixer)
            logits = model.forward_logits(bits, row_on, pad_t)[0]
            pred = model.presence_probs(logits[None])[0].cpu().numpy()
        else:
            pred = model(bits, row_on, pad_t)[0].cpu().numpy()
        for row in range(card.n_players):
            if pad[row]:
                continue
            pos = POS_NAME.get(int(card.position[row]))
            if pos is None:
                continue
            for gw in range(cut, card.n_fixtures):
                if not view["present"][row, gw]:
                    continue
                ag = int(event_target(card, gw)[row, sl].sum() > 0)
                pg = int((pred[row, gw, sl] >= 0.5).sum() > 0)
                out[pos]["actual"] += ag
                out[pos]["painted"] += pg
                out[pos]["hits"] += int(ag and pg)
    return out


@torch.no_grad()
def evaluate_arm(
    model: nn.Module,
    cards: list[BaudotCard],
    device: torch.device,
    use_theta: bool = False,
) -> dict:
    """Holdout tallies; for calib_theta apply presence via σ(z−θ)."""
    if not use_theta:
        base = evaluate_mixer(model, cards, device)
        pos = {
            "goals_20": position_split(model, cards, device, 0.2, "goals", False),
            "assists_20": position_split(model, cards, device, 0.2, "assists", False),
        }
        return {"rows": base["rows"], "position": pos, "gkp_by_cut": base["gkp_painted_goals_by_cut"]}

    # Custom eval with theta-shifted presence.
    model.eval()
    assert isinstance(model, CalibThetaMixer)
    rows: list[dict] = []
    gkp_by_cut: dict[str, int] = {}
    for frac in CUT_FRACS:
        for channel in ("goals", "assists"):
            totals = {key: 0 for key in ("actual", "painted", "exact", "actual_within", "painted_within")}
            gkp_painted = 0
            sl = _channel_slice(channel)
            for card in cards:
                cut = cut_at_fraction(card, frac)
                view = remainder_view(card, cut)
                pad = _player_pad(card)
                logits = model.forward_logits(
                    torch.from_numpy(view["bits"][None]).to(device),
                    torch.from_numpy(view["row_on"][None]).to(device),
                    torch.from_numpy(pad[None]).to(device),
                )[0]
                pred = model.presence_probs(logits[None])[0].cpu().numpy()
                for row in range(card.n_players):
                    if pad[row]:
                        continue
                    cols = [gw for gw in range(cut, card.n_fixtures) if view["present"][row, gw]]
                    if not cols:
                        continue
                    actual = np.array(
                        [int(event_target(card, gw)[row, sl].sum()) for gw in cols],
                        dtype=np.int64,
                    )
                    painted = np.array(
                        [int((pred[row, gw, sl] >= 0.5).sum()) for gw in cols],
                        dtype=np.int64,
                    )
                    part = tally_row(card.gw[cols], actual, painted)
                    for key in totals:
                        totals[key] += part[key]
                    if channel == "goals" and int(card.position[row]) == 0:
                        gkp_painted += int((painted > 0).sum())
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
                    "gkp_painted_events": gkp_painted,
                }
            )
        gkp_by_cut[f"{frac:.0%}"] = next(
            r["gkp_painted_events"] for r in rows if r["revealed"] == f"{frac:.0%}" and r["channel"] == "goals"
        )
    pos = {
        "goals_20": position_split(model, cards, device, 0.2, "goals", True),
        "assists_20": position_split(model, cards, device, 0.2, "assists", True),
    }
    return {"rows": rows, "position": pos, "gkp_by_cut": gkp_by_cut}


def train_arm(
    name: str,
    model: nn.Module,
    train: list[BaudotCard],
    test: list[BaudotCard],
    device: torch.device,
    kind: str,
) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    history: list[float] = []
    val_scores: list[float] = []
    best_state = None
    best_score = -1.0
    bad = 0
    t0 = time.time()
    use_theta = kind == "calib_theta"
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
            assert isinstance(model, LogitMixer)
            logits = model.forward_logits(bits, row_on, pad)
            if kind == "listwise_bal":
                loss = listwise_bal_loss(logits, target, live)
            else:
                assert isinstance(model, CalibThetaMixer)
                loss = calib_theta_loss(model, logits, target, live)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        avg = total / max(seen, 1)
        history.append(avg)
        hold = evaluate_arm(model, test, device, use_theta=use_theta)
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
        if use_theta and isinstance(model, CalibThetaMixer):
            extra = (
                f" θg={float(model.theta_g.detach()):.2f} "
                f"θa={float(model.theta_a.detach()):.2f}"
            )
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} loss {avg:.4f} holdout_score {score:.4f}"
            f"{extra}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate_arm(model, test, device, use_theta=use_theta)
    meta = {
        "name": name,
        "kind": kind,
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "best_holdout_mean_score": best_score,
        "seconds": round(time.time() - t0, 1),
        "holdout": holdout,
    }
    if use_theta and isinstance(model, CalibThetaMixer):
        meta["theta_g"] = float(model.theta_g.detach())
        meta["theta_a"] = float(model.theta_a.detach())
    return meta


def _delta(a: float, b: float) -> str:
    return f"{a - b:+.3f}"


def _pos_cell(d: dict, pos: str) -> str:
    x = d[pos]
    return f"{x['actual']}/{x['painted']}/{x['hits']}"


def write_report(metas: list[dict]) -> Path:
    gate_rows = _gate_rows()
    gate_mean = _mean_score(gate_rows)
    # Autopsy baseline for DEF blank under Tversky gate.
    gate_pos = {
        "goals": {"GKP": "0/0/0", "DEF": "103/0/0", "MID": "394/~370/83", "FWD": "233/~331/111"},
        "assists": {"GKP": "4/0/0", "DEF": "181/0/0", "MID": "434/~363/65", "FWD": "88/~330/49"},
    }
    lines = [
        "# Baudot loss replacements vs locked Tversky gate",
        "",
        "Morning table. Same card, mixer, FOV, seed. Only the loss changes.",
        "",
        f"Gate: `{GATE_META['method']}` · α={GATE_META['alpha']:.3f} β={GATE_META['beta']:.3f} · mean **{gate_mean:.3f}**.",
        "",
        "| arm | loss | params | epochs | mean score | Δ gate | seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
        f"| gate | learnable Tversky β≥α | — | — | {gate_mean:.3f} | +0.000 | — |",
    ]
    for m in metas:
        mean = _mean_score(m["holdout"]["rows"])
        lines.append(
            f"| {m['name']} | {m['kind']} | {m['params']} | {m['epochs_run']} | "
            f"{mean:.3f} | {_delta(mean, gate_mean)} | {m['seconds']} |"
        )
    lines.extend(
        [
            "",
            "## Goals at 20% by position (actual / painted / hits)",
            "",
            "| arm | GKP | DEF | MID | FWD |",
            "|---|---|---|---|---|",
            f"| gate (autopsy) | {gate_pos['goals']['GKP']} | {gate_pos['goals']['DEF']} | "
            f"{gate_pos['goals']['MID']} | {gate_pos['goals']['FWD']} |",
        ]
    )
    for m in metas:
        g = m["holdout"]["position"]["goals_20"]
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
            f"| gate (autopsy) | {gate_pos['assists']['GKP']} | {gate_pos['assists']['DEF']} | "
            f"{gate_pos['assists']['MID']} | {gate_pos['assists']['FWD']} |",
        ]
    )
    for m in metas:
        a = m["holdout"]["position"]["assists_20"]
        lines.append(
            f"| {m['name']} | {_pos_cell(a, 'GKP')} | {_pos_cell(a, 'DEF')} | "
            f"{_pos_cell(a, 'MID')} | {_pos_cell(a, 'FWD')} |"
        )
    lines.extend(
        [
            "",
            "## Holdout table (familiar metrics)",
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
    # Optional theta dump
    for m in metas:
        if "theta_g" in m:
            lines.extend(
                [
                    "",
                    f"### {m['name']} learned thresholds",
                    "",
                    f"- θ_G = {m['theta_g']:.3f} · θ_A = {m['theta_a']:.3f}",
                ]
            )
    lines.extend(
        [
            "",
            "## How to read in the morning",
            "",
            "1. Did DEF goals/assists leave `0/0` with non-trivial hits (not just volume)?",
            "2. Did FWD assist overpaint drop vs gate autopsy (~330 painted / 88 actual)?",
            "3. Did mean score hold near 0.218 without GKP noise?",
            "4. Prefer an arm only if (1)–(3) improve together; volume-only DEF paint is a fail.",
            "",
        ]
    )
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    (ART / "baudot_loss_replacements.md").write_text(text, encoding="utf-8")
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
        ("1_listwise_bal", LogitMixer(width=width), "listwise_bal"),
        ("2_calib_theta", CalibThetaMixer(width=width), "calib_theta"),
    ]
    metas = []
    for name, model, kind in arms:
        print(f"\n=== {name} ===", flush=True)
        meta = train_arm(name, model, train, test, device, kind=kind)
        metas.append(meta)
        print(
            f"{name} mean {_mean_score(meta['holdout']['rows']):.3f} "
            f"DEF goals {meta['holdout']['position']['goals_20']['DEF']} "
            f"DEF assists {meta['holdout']['position']['assists_20']['DEF']}",
            flush=True,
        )

    path = write_report(metas)
    gate_mean = _mean_score(_gate_rows())
    slim = {
        "trial": "baudot_loss_replacements",
        "gate": GATE_META,
        "arms": [
            {
                "name": m["name"],
                "kind": m["kind"],
                "params": m["params"],
                "epochs_run": m["epochs_run"],
                "mean_score": _mean_score(m["holdout"]["rows"]),
                "delta_gate_mean": _mean_score(m["holdout"]["rows"]) - gate_mean,
                "seconds": m["seconds"],
                "rows": m["holdout"]["rows"],
                "position": m["holdout"]["position"],
                **{k: m[k] for k in ("theta_g", "theta_a") if k in m},
            }
            for m in metas
        ],
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (ART / "baudot_loss_replacements.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
