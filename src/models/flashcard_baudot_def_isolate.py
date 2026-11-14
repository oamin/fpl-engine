"""Baudot DEF isolate — train/test positions independently.

Question: can we isolate DEF on the card and teach only their events?
Prior role-paint kept the full roster and only changed heads / aux loss;
DEF soft ink stayed at the paint floor (~0.006). This trial changes the
picture itself.

Arms (same mixer / learnable Tversky / FOV / seed as the locked gate):
  1. def_loss_mask   — full card still visible to the encoder; live Tversky
                       window restricted to DEF player-GW cells only.
  2. def_card_isolate — pad out non-DEF slots (squad attn ignores them) and
                       zero their player blocks on every GW so the picture
                       only carries DEF who/mins/events + shared header.

Early-stop and primary score are DEF-only balanced means (not the full
MID/FWD-dominated gate mean). Soft ink on DEF is reported like the autopsy.
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
    HEADER_BITS,
    LR,
    MAX_PLAYERS,
    PLAYER_BITS,
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
    event_target,
    fit_clubs,
    load_odds,
    remainder_view,
    tversky_loss,
)
from src.models.flashcard_card_score import balanced_score, tally_row

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_def_isolate.md"
POS_NAME = {0: "GKP", 1: "DEF", 2: "MID", 3: "FWD"}
TARGET_POS = 1  # DEF
SOFT_THRS = (0.05, 0.1, 0.25, 0.5)


def _pos_mask(card: BaudotCard, pos: int = TARGET_POS) -> np.ndarray:
    mask = np.zeros(MAX_PLAYERS, dtype=bool)
    for row in range(card.n_players):
        if int(card.position[row]) == pos and int(card.player_id[row]) > 0:
            mask[row] = True
    return mask


def _isolate_pad(card: BaudotCard, pos: int = TARGET_POS) -> np.ndarray:
    """Pad everything that is not the target position (plus empty slots)."""
    pad = _player_pad(card).copy()
    keep = _pos_mask(card, pos)
    pad |= ~keep
    return pad


def _zero_non_pos_bits(bits: np.ndarray, card: BaudotCard, pos: int = TARGET_POS) -> np.ndarray:
    """Blank non-target player blocks on the Baudot row picture."""
    out = bits.copy()
    keep = _pos_mask(card, pos)
    for row in range(MAX_PLAYERS):
        if keep[row]:
            continue
        lo = HEADER_BITS + row * PLAYER_BITS
        hi = lo + PLAYER_BITS
        out[:, lo:hi] = 0.0
    return out


def _batch_arm(
    cards: list[BaudotCard],
    cuts: list[int],
    device: torch.device,
    kind: str,
):
    bits_l, row_ons, rem_ons, presents, targets, pads, pos_masks = [], [], [], [], [], [], []
    for card, cut in zip(cards, cuts, strict=True):
        view = remainder_view(card, cut)
        bits = view["bits"]
        if kind == "def_card_isolate":
            bits = _zero_non_pos_bits(bits, card)
            pad = _isolate_pad(card)
        else:
            pad = _player_pad(card)
        bits_l.append(bits)
        row_ons.append(view["row_on"])
        rem_ons.append(view["rem_on"])
        presents.append(view["present"])
        targets.append(view["target"])
        pads.append(pad)
        pos_masks.append(_pos_mask(card))
    return (
        torch.from_numpy(np.stack(bits_l)).to(device),
        torch.from_numpy(np.stack(row_ons)).to(device),
        torch.from_numpy(np.stack(rem_ons)).to(device),
        torch.from_numpy(np.stack(pads)).to(device),
        torch.from_numpy(np.stack(presents)).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
        torch.from_numpy(np.stack(pos_masks)).to(device),
    )


def _fmt_aph(d: dict[str, int]) -> str:
    return f"{d['actual']}/{d['painted']}/{d['hits']}"


@torch.no_grad()
def position_split(
    model: BlockTokenMixer,
    cards: list[BaudotCard],
    device: torch.device,
    frac: float,
    channel: str,
    kind: str,
) -> dict[str, dict[str, int]]:
    sl = _channel_slice(channel)
    out = {name: {"actual": 0, "painted": 0, "hits": 0} for name in POS_NAME.values()}
    model.eval()
    for card in cards:
        cut = cut_at_fraction(card, frac)
        view = remainder_view(card, cut)
        bits = view["bits"]
        if kind == "def_card_isolate":
            bits = _zero_non_pos_bits(bits, card)
            pad = _isolate_pad(card)
        else:
            pad = _player_pad(card)
        pred = model(
            torch.from_numpy(bits[None]).to(device),
            torch.from_numpy(view["row_on"][None]).to(device),
            torch.from_numpy(pad[None]).to(device),
        )[0].cpu().numpy()
        for row in range(card.n_players):
            pos = POS_NAME.get(int(card.position[row]))
            if pos is None:
                continue
            # Card-isolate pads non-DEF → paint forced 0; still tally truth actuals.
            for gw in range(cut, card.n_fixtures):
                if not view["present"][row, gw]:
                    continue
                ag = int(event_target(card, gw)[row, sl].sum() > 0)
                if pad[row]:
                    pg = 0
                else:
                    pg = int((pred[row, gw, sl] >= 0.5).sum() > 0)
                out[pos]["actual"] += ag
                out[pos]["painted"] += pg
                out[pos]["hits"] += int(ag and pg)
    return out


@torch.no_grad()
def evaluate_def(
    model: BlockTokenMixer,
    cards: list[BaudotCard],
    device: torch.device,
    kind: str,
) -> dict:
    """DEF-only tally_row + balanced_score across cuts; plus full position split."""
    model.eval()
    rows: list[dict] = []
    for frac in CUT_FRACS:
        for channel in ("goals", "assists"):
            totals = {key: 0 for key in ("actual", "painted", "exact", "actual_within", "painted_within")}
            sl = _channel_slice(channel)
            for card in cards:
                cut = cut_at_fraction(card, frac)
                view = remainder_view(card, cut)
                bits = view["bits"]
                if kind == "def_card_isolate":
                    bits = _zero_non_pos_bits(bits, card)
                    pad = _isolate_pad(card)
                else:
                    # loss-mask arm: evaluate DEF paint on the full card
                    pad = _player_pad(card)
                keep = _pos_mask(card)
                pred = model(
                    torch.from_numpy(bits[None]).to(device),
                    torch.from_numpy(view["row_on"][None]).to(device),
                    torch.from_numpy(pad[None]).to(device),
                )[0].cpu().numpy()
                for row in range(card.n_players):
                    if not keep[row]:
                        continue
                    cols = [gw for gw in range(cut, card.n_fixtures) if view["present"][row, gw]]
                    if not cols:
                        continue
                    actual = np.array(
                        [int(view["target"][row, gw, sl].sum()) for gw in cols], dtype=np.int64
                    )
                    painted = np.array(
                        [int((pred[row, gw, sl] >= 0.5).sum()) for gw in cols], dtype=np.int64
                    )
                    part = tally_row(card.gw[cols], actual, painted)
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
                }
            )
    pos = {
        "goals_20": position_split(model, cards, device, 0.2, "goals", kind),
        "assists_20": position_split(model, cards, device, 0.2, "assists", kind),
    }
    mean_score = float(np.mean([r["score"] for r in rows])) if rows else 0.0
    return {"rows": rows, "position": pos, "def_mean_score": mean_score}


@torch.no_grad()
def soft_ink_def(
    model: BlockTokenMixer,
    cards: list[BaudotCard],
    device: torch.device,
    kind: str,
    frac: float = 0.2,
) -> dict:
    """Autopsy-style soft ink on DEF remainder cells at one cut."""
    model.eval()
    channels = {
        "goals": {"event": [], "blank": [], "max_event": [], "painted": 0, "hits": 0, "actual": 0},
        "assists": {"event": [], "blank": [], "max_event": [], "painted": 0, "hits": 0, "actual": 0},
    }
    for card in cards:
        cut = cut_at_fraction(card, frac)
        view = remainder_view(card, cut)
        bits = view["bits"]
        if kind == "def_card_isolate":
            bits = _zero_non_pos_bits(bits, card)
            pad = _isolate_pad(card)
        else:
            pad = _player_pad(card)
        keep = _pos_mask(card)
        pred = model(
            torch.from_numpy(bits[None]).to(device),
            torch.from_numpy(view["row_on"][None]).to(device),
            torch.from_numpy(pad[None]).to(device),
        )[0].cpu().numpy()
        for row in range(card.n_players):
            if not keep[row]:
                continue
            for gw in range(cut, card.n_fixtures):
                if not view["present"][row, gw]:
                    continue
                for channel, sl in (("goals", slice(0, GOAL_BITS)), ("assists", slice(GOAL_BITS, EVENT_BITS))):
                    soft = pred[row, gw, sl]
                    mean_s = float(soft.mean())
                    max_s = float(soft.max())
                    is_event = int(view["target"][row, gw, sl].sum() > 0)
                    painted = int((soft >= 0.5).sum() > 0)
                    bucket = channels[channel]
                    if is_event:
                        bucket["event"].append(mean_s)
                        bucket["max_event"].append(max_s)
                        bucket["actual"] += 1
                        bucket["hits"] += int(painted)
                    else:
                        bucket["blank"].append(mean_s)
                    bucket["painted"] += painted
    out: dict = {}
    for channel, bucket in channels.items():
        ev = np.asarray(bucket["event"], dtype=np.float64)
        bl = np.asarray(bucket["blank"], dtype=np.float64)
        mx = np.asarray(bucket["max_event"], dtype=np.float64)
        thr_hits = {}
        for thr in SOFT_THRS:
            thr_hits[str(thr)] = int((mx >= thr).sum()) if mx.size else 0
        out[channel] = {
            "n_event": int(ev.size),
            "n_blank": int(bl.size),
            "mean_soft_event": float(ev.mean()) if ev.size else None,
            "mean_soft_blank": float(bl.mean()) if bl.size else None,
            "mean_max_event": float(mx.mean()) if mx.size else None,
            "pct_max_ge_0_5": float((mx >= 0.5).mean()) if mx.size else 0.0,
            "n_max_ge_thr": thr_hits,
            "painted_weeks": bucket["painted"],
            "hits": bucket["hits"],
            "actual_weeks": bucket["actual"],
        }
    return out


def train_arm(
    name: str,
    kind: str,
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
    ab_hist: list[tuple[float, float]] = []
    val_scores: list[float] = []
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
            bits, row_on, rem_on, pad, present, target, pos_m = _batch_arm(pick, cuts, device, kind)
            live = (
                present[:, :, :, None]
                & rem_on[:, None, :, None].bool()
                & (~pad)[:, :, None, None]
                & pos_m[:, :, None, None]
            )
            # Both arms restrict loss to DEF; card-isolate also pads non-DEF.
            opt.zero_grad()
            pred = model(bits, row_on, pad)
            alpha, beta = model.tversky_weights()
            loss = tversky_loss(pred, target, live, alpha, beta)
            bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
            loss = loss + BCE_W * (
                (bce * live.to(pred.dtype)).sum() / live.to(pred.dtype).sum().clamp_min(1.0)
            )
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
        hold = evaluate_def(model, test, device, kind)
        score = float(hold["def_mean_score"])
        val_scores.append(score)
        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} loss {avg:.4f} "
            f"α={ab_hist[-1][0]:.3f} β={ab_hist[-1][1]:.3f} "
            f"def_mean {score:.4f} "
            f"DEF goals {_fmt_aph(hold['position']['goals_20']['DEF'])} "
            f"assists {_fmt_aph(hold['position']['assists_20']['DEF'])}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hold = evaluate_def(model, test, device, kind)
    soft = soft_ink_def(model, test, device, kind)
    with torch.no_grad():
        a, b = model.tversky_weights()
        final_ab = (float(a), float(b))
    return {
        "name": name,
        "kind": kind,
        "params": sum(p.numel() for p in model.parameters()),
        "epochs_run": len(history),
        "final_alpha": final_ab[0],
        "final_beta": final_ab[1],
        "best_def_mean_score": best_score,
        "def_mean_score": hold["def_mean_score"],
        "seconds": round(time.time() - t0, 1),
        "history": history,
        "ab_history": ab_hist,
        "val_scores": val_scores,
        "rows": hold["rows"],
        "position": hold["position"],
        "soft_ink_20": soft,
    }


def write_report(arms: list[dict], train_n: int, test_n: int) -> Path:
    gate_pos = {
        "goals": {"GKP": "0/0/0", "DEF": "103/0/0", "MID": "394/~370/83", "FWD": "233/~331/111"},
        "assists": {"GKP": "4/0/0", "DEF": "181/0/0", "MID": "434/~363/65", "FWD": "88/~330/49"},
    }
    gate_soft = {
        "goals": {"mean_event": 0.0063, "mean_blank": 0.0062, "mean_max": 0.0098, "pct_ge_05": 0.0},
        "assists": {"mean_event": 0.0067, "mean_blank": 0.0065, "mean_max": 0.0104, "pct_ge_05": 0.0},
    }
    lines = [
        "# Baudot DEF isolate — train/test positions independently",
        "",
        "Same mixer / learnable Tversky / FOV / seed as the locked gate. "
        "Only the **player window** on the card and in the loss changes.",
        "",
        f"- Train cards: {train_n} · holdout: {test_n} ({TEST_SEASON})",
        f"- Gate: `{GATE_META['module']}` · locked mean {float(np.mean([balanced_score(GATE[k])['score'] for k in GATE])):.3f}",
        "- Early-stop metric: **DEF-only** mean balanced score (not full roster mean).",
        "",
        "## Arms",
        "",
        "| arm | picture | loss window |",
        "|---|---|---|",
        "| `1_def_loss_mask` | full roster (MID/FWD visible) | DEF cells only |",
        "| `2_def_card_isolate` | non-DEF player blocks zeroed + padded out of squad attn | DEF cells only |",
        "",
        "## DEF scoreboard (primary)",
        "",
        "| arm | DEF mean score | DEF goals 20% a/p/h | DEF assists 20% a/p/h | α | β | epochs |",
        "|---|---:|---|---|---:|---:|---:|",
        f"| gate (autopsy) | — | {gate_pos['goals']['DEF']} | {gate_pos['assists']['DEF']} | "
        f"{GATE_META['alpha']:.3f} | {GATE_META['beta']:.3f} | — |",
    ]
    for m in arms:
        g = m["position"]["goals_20"]["DEF"]
        a = m["position"]["assists_20"]["DEF"]
        lines.append(
            f"| {m['name']} | {m['def_mean_score']:.3f} | {_fmt_aph(g)} | {_fmt_aph(a)} | "
            f"{m['final_alpha']:.3f} | {m['final_beta']:.3f} | {m['epochs_run']} |"
        )

    lines.extend(
        [
            "",
            "## Soft ink on DEF (20% remainder)",
            "",
            "| arm | channel | mean soft event | mean soft blank | mean max event | % max≥0.5 | n max≥0.05 |",
            "|---|---|---:|---:|---:|---:|---:|",
            f"| gate (autopsy) | goals | {gate_soft['goals']['mean_event']:.4f} | "
            f"{gate_soft['goals']['mean_blank']:.4f} | {gate_soft['goals']['mean_max']:.4f} | "
            f"{gate_soft['goals']['pct_ge_05']:.3f} | 0 |",
            f"| gate (autopsy) | assists | {gate_soft['assists']['mean_event']:.4f} | "
            f"{gate_soft['assists']['mean_blank']:.4f} | {gate_soft['assists']['mean_max']:.4f} | "
            f"{gate_soft['assists']['pct_ge_05']:.3f} | 0 |",
        ]
    )
    for m in arms:
        for channel in ("goals", "assists"):
            s = m["soft_ink_20"][channel]
            me = "—" if s["mean_soft_event"] is None else f"{s['mean_soft_event']:.4f}"
            mb = "—" if s["mean_soft_blank"] is None else f"{s['mean_soft_blank']:.4f}"
            mm = "—" if s["mean_max_event"] is None else f"{s['mean_max_event']:.4f}"
            lines.append(
                f"| {m['name']} | {channel} | {me} | {mb} | {mm} | "
                f"{s['pct_max_ge_0_5']:.3f} | {s['n_max_ge_thr'].get('0.05', 0)} |"
            )

    lines.extend(
        [
            "",
            "## Full position split at 20% (actual / painted / hits)",
            "",
            "### Goals",
            "",
            "| arm | GKP | DEF | MID | FWD |",
            "|---|---|---|---|---|",
            f"| gate (autopsy) | {gate_pos['goals']['GKP']} | {gate_pos['goals']['DEF']} | "
            f"{gate_pos['goals']['MID']} | {gate_pos['goals']['FWD']} |",
        ]
    )
    for m in arms:
        g = m["position"]["goals_20"]
        lines.append(
            f"| {m['name']} | {_fmt_aph(g['GKP'])} | {_fmt_aph(g['DEF'])} | "
            f"{_fmt_aph(g['MID'])} | {_fmt_aph(g['FWD'])} |"
        )
    lines.extend(
        [
            "",
            "### Assists",
            "",
            "| arm | GKP | DEF | MID | FWD |",
            "|---|---|---|---|---|",
            f"| gate (autopsy) | {gate_pos['assists']['GKP']} | {gate_pos['assists']['DEF']} | "
            f"{gate_pos['assists']['MID']} | {gate_pos['assists']['FWD']} |",
        ]
    )
    for m in arms:
        a = m["position"]["assists_20"]
        lines.append(
            f"| {m['name']} | {_fmt_aph(a['GKP'])} | {_fmt_aph(a['DEF'])} | "
            f"{_fmt_aph(a['MID'])} | {_fmt_aph(a['FWD'])} |"
        )

    lines.extend(
        [
            "",
            "## DEF-only cut table (best arm by DEF mean)",
            "",
        ]
    )
    best = max(arms, key=lambda m: m["def_mean_score"]) if arms else None
    if best is not None:
        lines.extend(
            [
                f"Best arm: `{best['name']}` (DEF mean {best['def_mean_score']:.3f}).",
                "",
                "| revealed | channel | actual | painted | hits | score |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for r in best["rows"]:
            lines.append(
                f"| {r['revealed']} | {r['channel']} | {r['actual']} | {r['painted']} | "
                f"{r['hits']} | {r['score']:.3f} |"
            )

    # Verdict
    any_hits = any(
        m["position"]["goals_20"]["DEF"]["hits"] > 0 or m["position"]["assists_20"]["DEF"]["hits"] > 0
        for m in arms
    )
    soft_lift = any(
        (m["soft_ink_20"]["goals"]["mean_max_event"] or 0) >= 0.05
        or (m["soft_ink_20"]["assists"]["mean_max_event"] or 0) >= 0.05
        for m in arms
    )
    if any_hits and soft_lift:
        verdict = (
            "**Partial / open.** Isolating DEF moves soft ink off the floor and lands "
            "non-trivial hits. Not a gate replacement (full-roster mean is undefined under "
            "DEF-only paint); keep as a rare-event diagnostic and decide whether to merge "
            "a DEF specialist into the shared painter."
        )
        park = "diagnostic keep — not promoted over the locked full-roster gate"
    elif soft_lift and not any_hits:
        verdict = (
            "**Volume / ranking fail.** Soft ink rises above the autopsy floor but "
            "thresholded hits stay dead or placement fails. Isolation alone does not "
            "solve DEF; new dots or a ranking grain still needed."
        )
        park = "park — isolation insufficient"
    else:
        verdict = (
            "**No lift.** Even with MID/FWD removed from the picture and the loss, "
            "DEF soft ink stays near the paint bias. The collapse is not explained by "
            "MID/FWD drowning the Tversky sum alone."
        )
        park = "park — isolation does not unfreeze DEF"

    lines.extend(
        [
            "",
            "## Morning call",
            "",
            verdict,
            "",
            f"Disposition: {park}.",
            "",
            "Read order:",
            "1. Does DEF soft max leave the ~0.01 floor (n max≥0.05 > 0)?",
            "2. Do DEF goals/assists leave `0/0` with real hits (not volume-only)?",
            "3. Is `2_def_card_isolate` better than `1_def_loss_mask`? "
            "(If yes, the shared picture comparison was part of the collapse.)",
            "4. Do not promote over the locked gate on DEF-only mean; a specialist "
            "would need a merge plan that preserves MID/FWD.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    print("loading league + odds + name codes", flush=True)
    league = build_league(seasons)
    odds_table = load_odds()
    name_codes = build_name_codes(seasons)
    squad_all = build_cards(seasons, league)
    clubs = fit_clubs([c for c in squad_all if c.season in TRAIN_SEASONS])
    for card in squad_all:
        clubs.setdefault(card.team, len(clubs) + 1)
        for opp in card.opponent:
            if opp:
                clubs.setdefault(opp, len(clubs) + 1)
    baudot_all = build_baudot_cards(squad_all, clubs, name_codes, odds_table)
    train = [c for c in baudot_all if c.season in TRAIN_SEASONS]
    test = [c for c in baudot_all if c.season == TEST_SEASON]
    width = train[0].width
    print(f"cards train {len(train)} test {len(test)} width {width}", flush=True)

    # Sanity: how many DEF slots per card?
    def_counts = [int(_pos_mask(c).sum()) for c in train]
    print(
        f"DEF slots/card train mean {np.mean(def_counts):.1f} "
        f"min {min(def_counts)} max {max(def_counts)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arms_spec = [
        ("1_def_loss_mask", "def_loss_mask"),
        ("2_def_card_isolate", "def_card_isolate"),
    ]
    results: list[dict] = []
    for name, kind in arms_spec:
        print(f"\n=== {name} ===", flush=True)
        model = BlockTokenMixer(width=width)
        meta = train_arm(name, kind, model, train, test, device)
        print(
            f"{name} done def_mean={meta['def_mean_score']:.3f} "
            f"DEF goals {_fmt_aph(meta['position']['goals_20']['DEF'])} "
            f"soft_max_g={meta['soft_ink_20']['goals']['mean_max_event']}",
            flush=True,
        )
        results.append(meta)

    report = write_report(results, len(train), len(test))
    state = {
        "trial": "baudot_def_isolate",
        "gate": GATE_META,
        "arms": results,
        "seconds": round(time.time() - t0, 1),
        "report": str(report),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (ART / "baudot_def_isolate.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"report {report}", flush=True)
    print(f"total seconds {state['seconds']}", flush=True)


if __name__ == "__main__":
    main()
