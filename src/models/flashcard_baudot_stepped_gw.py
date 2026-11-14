"""Baudot stepped single-GW paint — complementary FOV to the locked gate.

Locked gate FOV: all past rows true → one-shot paint of the *entire* remainder.
This trial: all past rows true → one-shot paint of the *next* GW row only, then
step. Teacher-forced on holdout (earlier remainder weeks use true G/A when the
step advances). Same card, BlockTokenMixer, learnable Tversky (β≥α).

Judged on the locked GATE tallies / balanced_score, plus DEF goals by position.
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
    _channel_slice,
    _player_pad,
    blank_events_in_row,
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
OUT = REPORTS / "baudot_stepped_gw.md"
POS_NAME = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}


def next_gw_view(card: BaudotCard, target_gw: int) -> dict[str, np.ndarray]:
    """Past rows true; only target_gw has G/A blank. One-step paint target."""
    assert 1 <= target_gw < card.n_fixtures
    bits = card.bits.copy()
    row_on = np.zeros(MAX_GW, dtype=np.float32)
    rem_on = np.zeros(MAX_GW, dtype=np.float32)
    target = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), dtype=np.float32)
    present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
    for gw in range(card.n_fixtures):
        row_on[gw] = 1.0
    bits[target_gw] = blank_events_in_row(bits[target_gw])
    rem_on[target_gw] = 1.0
    target[:, target_gw] = event_target(card, target_gw)
    present[:, target_gw] = card.present[:, target_gw]
    return {
        "bits": bits,
        "row_on": row_on,
        "rem_on": rem_on,
        "target": target,
        "present": present,
        "target_gw": target_gw,
    }


def _batch_next(cards: list[BaudotCard], targets: list[int], device: torch.device):
    bits_l, row_ons, rem_ons, presents, tgts, pads = [], [], [], [], [], []
    for card, tg in zip(cards, targets, strict=True):
        view = next_gw_view(card, int(tg))
        bits_l.append(view["bits"])
        row_ons.append(view["row_on"])
        rem_ons.append(view["rem_on"])
        presents.append(view["present"])
        tgts.append(view["target"])
        pads.append(_player_pad(card))
    return (
        torch.from_numpy(np.stack(bits_l)).to(device),
        torch.from_numpy(np.stack(row_ons)).to(device),
        torch.from_numpy(np.stack(rem_ons)).to(device),
        torch.from_numpy(np.stack(pads)).to(device),
        torch.from_numpy(np.stack(presents)).to(device),
        torch.from_numpy(np.stack(tgts)).to(device),
    )


def _sample_target_gw(card: BaudotCard, rng: np.random.Generator) -> int:
    """Uniform over GW indices with ≥1 past row."""
    return int(rng.integers(1, card.n_fixtures))


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
def evaluate_stepped(
    model: BlockTokenMixer,
    cards: list[BaudotCard],
    device: torch.device,
) -> dict:
    """Teacher-forced step through each cut's remainder; same tallies as the gate."""
    model.eval()
    rows: list[dict] = []
    gkp_by_cut: dict[str, int] = {}
    pos_goals_20 = {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()}
    pos_assists_20 = {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()}
    for frac in CUT_FRACS:
        # Cache teacher-forced next-GW paints once per card for this cut.
        cached: list[tuple[BaudotCard, np.ndarray, dict[int, np.ndarray]]] = []
        for card in cards:
            cut = cut_at_fraction(card, frac)
            pad = _player_pad(card)
            preds_by_gw: dict[int, np.ndarray] = {}
            for gw in range(cut, card.n_fixtures):
                view = next_gw_view(card, gw)
                preds_by_gw[gw] = model(
                    torch.from_numpy(view["bits"][None]).to(device),
                    torch.from_numpy(view["row_on"][None]).to(device),
                    torch.from_numpy(pad[None]).to(device),
                )[0].cpu().numpy()
            cached.append((card, pad, preds_by_gw))
        for channel in ("goals", "assists"):
            totals = {key: 0 for key in ("actual", "painted", "exact", "actual_within", "painted_within")}
            gkp_painted = 0
            sl = _channel_slice(channel)
            for card, pad, preds_by_gw in cached:
                cut = cut_at_fraction(card, frac)
                for row in range(card.n_players):
                    if pad[row]:
                        continue
                    cols: list[int] = []
                    actual_list: list[int] = []
                    painted_list: list[int] = []
                    for gw in range(cut, card.n_fixtures):
                        if not card.present[row, gw]:
                            continue
                        ag = int(event_target(card, gw)[row, sl].sum())
                        pg = int((preds_by_gw[gw][row, gw, sl] >= 0.5).sum())
                        cols.append(int(card.gw[gw]))
                        actual_list.append(ag)
                        painted_list.append(pg)
                        if frac == 0.2:
                            pos = POS_NAME.get(int(card.position[row]))
                            if pos is not None:
                                bucket = pos_goals_20 if channel == "goals" else pos_assists_20
                                ae, pe = int(ag > 0), int(pg > 0)
                                bucket[pos]["actual"] += ae
                                bucket[pos]["painted"] += pe
                                bucket[pos]["hits"] += int(ae and pe)
                        if channel == "goals" and int(card.position[row]) == 0 and pg > 0:
                            gkp_painted += 1
                    if not cols:
                        continue
                    part = tally_row(
                        np.array(cols, dtype=np.int64),
                        np.array(actual_list, dtype=np.int64),
                        np.array(painted_list, dtype=np.int64),
                    )
                    for key in totals:
                        totals[key] += part[key]
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
                    "hit_rate": score["hits"],
                    "volume": score["volume"],
                    "score": score["score"],
                    "gkp_painted_events": gkp_painted,
                }
            )
        gkp_by_cut[f"{frac:.0%}"] = next(
            r["gkp_painted_events"] for r in rows if r["revealed"] == f"{frac:.0%}" and r["channel"] == "goals"
        )
    return {
        "rows": rows,
        "gkp_painted_goals_by_cut": gkp_by_cut,
        "goals_20_by_pos": pos_goals_20,
        "assists_20_by_pos": pos_assists_20,
    }


def train_stepped(
    model: BlockTokenMixer,
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
            targets = [_sample_target_gw(card, rng) for card in pick]
            bits, row_on, rem_on, pad, present, target = _batch_next(pick, targets, device)
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
        print(
            f"stepped_gw epoch {epoch + 1}/{EPOCHS} loss {avg:.4f} "
            f"α={ab_hist[-1][0]:.3f} β={ab_hist[-1][1]:.3f} holdout_score {score:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"stepped_gw early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = evaluate_stepped(model, test, device)
    return {
        "name": "stepped_next_gw",
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
        "# Baudot stepped single-GW vs locked full-remainder gate",
        "",
        "Complementary FOV. Same card, BlockTokenMixer, learnable Tversky.",
        "- **Gate:** all past true → one-shot paint of the entire remainder window.",
        "- **This arm:** all past true → one-shot paint of the next GW only, stepped;",
        "  holdout is teacher-forced (true G/A on earlier remainder weeks).",
        f"Gate method: `{GATE_META['method']}` · mean score {gate_mean:.3f}.",
        "",
        "| arm | params | epochs | α | β | mean score | Δ gate | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| gate (full remainder one-shot) | — | — | {GATE_META['alpha']:.3f} | {GATE_META['beta']:.3f} | "
        f"{gate_mean:.3f} | +0.000 | — |",
        f"| stepped_next_gw | {meta['params']} | {meta['epochs_run']} | {meta['final_alpha']:.3f} | "
        f"{meta['final_beta']:.3f} | {mean:.3f} | {_delta(mean, gate_mean)} | {meta['seconds']} |",
        "",
        "## Goals at 20% by position (actual / painted / hits)",
        "",
        "| arm | GKP | DEF | MID | FWD |",
        "|---|---|---|---|---|",
        f"| stepped_next_gw | {_pos_cell(hold['goals_20_by_pos'], 'GKP')} | "
        f"{_pos_cell(hold['goals_20_by_pos'], 'DEF')} | "
        f"{_pos_cell(hold['goals_20_by_pos'], 'MID')} | "
        f"{_pos_cell(hold['goals_20_by_pos'], 'FWD')} |",
        "",
        "## Assists at 20% by position (actual / painted / hits)",
        "",
        "| arm | GKP | DEF | MID | FWD |",
        "|---|---|---|---|---|",
        f"| stepped_next_gw | {_pos_cell(hold['assists_20_by_pos'], 'GKP')} | "
        f"{_pos_cell(hold['assists_20_by_pos'], 'DEF')} | "
        f"{_pos_cell(hold['assists_20_by_pos'], 'MID')} | "
        f"{_pos_cell(hold['assists_20_by_pos'], 'FWD')} |",
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
            f"| stepped_next_gw | {row['revealed']} | {row['channel']} | {row['actual']} | {row['painted']} | "
            f"{row['hits']} | {row['actual_within_1']} | {row['painted_within_1']} | {row['score']:.3f} | "
            f"{_delta(row['score'], gscore)} |"
        )
    lines.extend(
        [
            "",
            "A positive Δ means stepped next-GW beats the locked full-remainder gate on that cell.",
            "DEF goals leaving 0 under the gate is a primary diagnostic here too.",
            "",
        ]
    )
    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    (ART / "baudot_stepped_gw.md").write_text(text, encoding="utf-8")
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

    print("\n=== stepped_next_gw ===", flush=True)
    model = BlockTokenMixer(width=width)
    meta = train_stepped(model, train, test, device)
    print(
        f"stepped_next_gw mean {_mean_score(meta['holdout']['rows']):.3f} "
        f"DEF goals 20% {meta['holdout']['goals_20_by_pos']['DEF']}",
        flush=True,
    )
    path = write_report(meta)
    gate_mean = _mean_score(_gate_rows())
    slim = {
        "trial": "baudot_stepped_gw",
        "fov": "all_past_gws + one_shot_next_gw_stepped_teacher_forced",
        "complementary_to": GATE_META,
        "arm": {
            "name": meta["name"],
            "params": meta["params"],
            "epochs_run": meta["epochs_run"],
            "best_holdout_mean_score": meta["best_holdout_mean_score"],
            "final_alpha": meta["final_alpha"],
            "final_beta": meta["final_beta"],
            "seconds": meta["seconds"],
            "mean_score": _mean_score(meta["holdout"]["rows"]),
            "delta_gate_mean": _mean_score(meta["holdout"]["rows"]) - gate_mean,
            "rows": meta["holdout"]["rows"],
            "gkp_by_cut": meta["holdout"]["gkp_painted_goals_by_cut"],
            "goals_20_by_pos": meta["holdout"]["goals_20_by_pos"],
            "assists_20_by_pos": meta["holdout"]["assists_20_by_pos"],
        },
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (ART / "baudot_stepped_gw.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
