"""Stepped and horizon row opacity on the Baudot card.

The gray of a dot is the model's probability. The loss on a dot is the
squared gap between that gray and the solid truth, (opacity − truth)².

Three arms, seed 0, same mixer, same card bits. Both windows are rows.

  stepped   — one gameweek, one shot, then step. Past rows stay true.
              Events from the painted week onward are blank.
  horizon   — one shot of every week beyond a cutoff. The loss is the
              squared gap on every live remainder dot. No column isolation.
  combined  — the two squared gaps, equal weight

Header and minutes stay on every row. During training the player slots are
shuffled so the mixer has to read the player-ID bits, not the seat index.

The holdout reads 1, 2, 4 and 8 weeks ahead.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models.flashcard_baudot_row_trial import (
    ART,
    BATCH,
    CUT_FRACS,
    EARLY_PATIENCE,
    EPOCHS,
    EVENT_BITS,
    GOAL_BITS,
    HEADER_BITS,
    LR,
    MAX_GW,
    MAX_PLAYERS,
    PLAYER_BITS,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    WHO_BITS,
    MINS_BITS,
    BlockTokenMixer,
    _batch,
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
    remainder_view,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_stepped_overlap.md"
ARMS = ("stepped", "horizon", "combined")
DISTANCES = (1, 2, 4, 8)


def _square_mask(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = live.to(pred.dtype)[..., None].expand_as(pred)
    return (pred - target) ** 2, mask


def row_gap(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Mean squared gap over every live dot in the scored window."""
    err, mask = _square_mask(pred, target, live)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def remainder_gap(
    pred: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
    pad: torch.Tensor,
) -> torch.Tensor:
    """All live remainder dots, equal weight. No per-player column isolation."""
    live = present & ~pad[..., None]
    return row_gap(pred, target, live)


def _take(pred, target, present, pad, cuts: list[int]):
    batch = pred.shape[0]
    index = torch.arange(batch, device=pred.device)
    choose = torch.tensor(cuts, device=pred.device, dtype=torch.long)
    live = present[index, :, choose] & ~pad
    return pred[index, :, choose], target[index, :, choose], live


def shuffle_player_slots(
    bits: torch.Tensor,
    present: torch.Tensor,
    target: torch.Tensor,
    pad: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute player seats on each card. The ID bits move with the player."""
    batch = bits.shape[0]
    out_bits = bits.clone()
    out_present = present.clone()
    out_target = target.clone()
    out_pad = pad.clone()
    for b in range(batch):
        order = rng.permutation(MAX_PLAYERS)
        body = bits[b, :, HEADER_BITS:].reshape(-1, MAX_PLAYERS, PLAYER_BITS)
        shuffled = body[:, order, :]
        out_bits[b, :, HEADER_BITS:] = shuffled.reshape(bits.shape[1], MAX_PLAYERS * PLAYER_BITS)
        out_present[b] = present[b, order]
        out_target[b] = target[b, order]
        out_pad[b] = pad[b, order]
    return out_bits, out_present, out_target, out_pad


def step_row_view(card, target_gw: int) -> dict:
    """One row, one shot. Past rows stay true. Events from this week on are blank.

    Header and minutes stay on every row, including the week being painted.
    """
    bits = card.bits.copy()
    row_on = np.zeros(MAX_GW, dtype=np.float32)
    target = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), dtype=np.float32)
    present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
    for gw in range(card.n_fixtures):
        row_on[gw] = 1.0
        if gw >= target_gw:
            bits[gw] = blank_events_in_row(bits[gw])
    target[:, target_gw] = event_target(card, target_gw)
    present[:, target_gw] = card.present[:, target_gw]
    return {"bits": bits, "row_on": row_on, "target": target, "present": present}


@torch.no_grad()
def _run(model, seen, pad, device) -> np.ndarray:
    model.eval()
    return model(
        torch.from_numpy(seen["bits"][None]).to(device),
        torch.from_numpy(seen["row_on"][None]).to(device),
        torch.from_numpy(pad[None]).to(device),
    )[0].detach().cpu().numpy()


def assert_step_is_blank(card) -> None:
    """The painted row keeps its true labels off the input. The row before stays true."""
    cut = cut_at_fraction(card, 0.2)
    view = remainder_view(card, cut)
    if not np.array_equal(view["bits"][cut - 1], card.bits[cut - 1]):
        raise AssertionError("past row was rewritten")
    painted = view["bits"][cut]
    for row in range(MAX_PLAYERS):
        base = HEADER_BITS + row * PLAYER_BITS + WHO_BITS + MINS_BITS
        if float(painted[base : base + EVENT_BITS].sum()) != 0.0:
            raise AssertionError("painted row still carries goals or assists")
    if int(view["target"][:, cut].sum()) <= 0:
        raise AssertionError("painted row has no event labels")
    step = step_row_view(card, cut)
    if not np.array_equal(step["bits"][cut - 1, :HEADER_BITS], card.bits[cut - 1, :HEADER_BITS]):
        raise AssertionError("the step rewrote a past header")
    painted = step["bits"][cut]
    if not np.array_equal(painted[:HEADER_BITS], card.bits[cut, :HEADER_BITS]):
        raise AssertionError("the painted week lost its header")
    later = min(cut + 1, card.n_fixtures - 1)
    for row in range(MAX_PLAYERS):
        base = HEADER_BITS + row * PLAYER_BITS
        mins = slice(base + WHO_BITS, base + WHO_BITS + MINS_BITS)
        events = slice(base + WHO_BITS + MINS_BITS, base + PLAYER_BITS)
        if not np.array_equal(painted[mins], card.bits[cut, mins]):
            raise AssertionError("the painted week lost its minutes")
        if float(step["bits"][later, events].sum()) != 0.0:
            raise AssertionError("a later week still shows its goals")


def _numpy_gap(pred: np.ndarray, target: np.ndarray, live: np.ndarray, position: np.ndarray) -> dict:
    """Squared gap on one week. pred/target (P, E), live (P,)."""
    played = np.flatnonzero(live)
    if len(played) == 0:
        return {"row": 0.0, "on_solid": 0.0, "on_blank": 0.0, "gkp": 0.0, "solid_n": 0}
    dots_p = pred[played].reshape(-1)
    dots_y = target[played].reshape(-1)
    gap = (dots_p - dots_y) ** 2
    solid = dots_y > 0.5
    gkp = played[position[played] == 0]
    gkp_p = float(pred[gkp, 0].mean()) if len(gkp) else 0.0
    return {
        "row": float(gap.mean()),
        "on_solid": float(dots_p[solid].mean()) if solid.any() else 0.0,
        "on_blank": float(dots_p[~solid].mean()) if (~solid).any() else 0.0,
        "gkp": gkp_p,
        "solid_n": int(solid.sum()),
    }


def _numpy_window(
    pred: np.ndarray,
    target: np.ndarray,
    present: np.ndarray,
    pad: np.ndarray,
    position: np.ndarray,
) -> dict:
    """Mean squared gap over every live remainder dot. pred/target (P, T, E)."""
    live = present & ~pad[:, None]
    if not live.any():
        return {"column": 0.0, "on_solid": 0.0, "on_blank": 0.0, "gkp": 0.0}
    dots_p = pred[live].reshape(-1)
    dots_y = target[live].reshape(-1)
    solid = dots_y > 0.5
    gkp_mask = live & (position[:, None] == 0)
    gkp_p = float(pred[gkp_mask][:, 0].mean()) if gkp_mask.any() else 0.0
    return {
        "column": float(((dots_p - dots_y) ** 2).mean()),
        "on_solid": float(dots_p[solid].mean()) if solid.any() else 0.0,
        "on_blank": float(dots_p[~solid].mean()) if (~solid).any() else 0.0,
        "gkp": gkp_p,
    }


def _marker_slice(channel: str) -> slice:
    if channel == "goals":
        return slice(0, 1)
    if channel == "assists":
        return slice(GOAL_BITS, GOAL_BITS + 1)
    return slice(0, EVENT_BITS)


def _batch_step(cards, targets, device):
    bits_l, row_ons, presents, targets_a, pads = [], [], [], [], []
    for card, target_gw in zip(cards, targets, strict=True):
        view = step_row_view(card, int(target_gw))
        bits_l.append(view["bits"])
        row_ons.append(view["row_on"])
        presents.append(view["present"])
        targets_a.append(view["target"])
        pads.append(_player_pad(card))
    return (
        torch.from_numpy(np.stack(bits_l)).to(device),
        torch.from_numpy(np.stack(row_ons)).to(device),
        torch.from_numpy(np.stack(pads)).to(device),
        torch.from_numpy(np.stack(presents)).to(device),
        torch.from_numpy(np.stack(targets_a)).to(device),
    )


@torch.no_grad()
def read_arm(model, cards, device, kind: str, stitch: bool = True) -> dict:
    """Stepped kind steps one week. Horizon kind paints the remainder in one shot."""
    model.eval()
    rows = []
    for frac in CUT_FRACS:
        bucket = {d: [] for d in DISTANCES}
        markers = {channel: [] for channel in ("goals", "assists")}
        for card in cards:
            cut = cut_at_fraction(card, frac)
            pad = _player_pad(card)
            if kind == "horizon":
                seen = remainder_view(card, cut)
                pred = _run(model, seen, pad, device)
                target = seen["target"]
                present = seen["present"]
            else:
                pred = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), np.float32)
                target = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_BITS), np.float32)
                present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
                if stitch:
                    weeks = range(cut, card.n_fixtures)
                else:
                    weeks = [cut + d - 1 for d in DISTANCES if cut + d - 1 < card.n_fixtures]
                for gw in weeks:
                    one = _run(model, step_row_view(card, gw), pad, device)
                    pred[:, gw] = one[:, gw]
                    target[:, gw] = event_target(card, gw)
                    present[:, gw] = card.present[:, gw]
            strip = _numpy_window(pred, target, present, pad, card.position)
            strip["revealed"] = f"{frac:.0%}"
            strip["kind"] = "strip"
            rows.append(strip)
            for distance in DISTANCES:
                row = cut + distance - 1
                if row >= card.n_fixtures:
                    continue
                live = present[:, row] & ~pad
                stats = _numpy_gap(pred[:, row], target[:, row], live, card.position)
                stats["distance"] = distance
                bucket[distance].append(stats)
                if distance == 1:
                    for channel in ("goals", "assists"):
                        sl = _marker_slice(channel)
                        part = _numpy_gap(pred[:, row, sl], target[:, row, sl], live, card.position)
                        part["channel"] = channel
                        markers[channel].append(part)
        for distance, stats in bucket.items():
            if not stats:
                continue
            rows.append(
                {
                    "revealed": f"{frac:.0%}",
                    "distance": distance,
                    "n": len(stats),
                    "row": float(np.mean([s["row"] for s in stats])),
                    "on_solid": float(np.mean([s["on_solid"] for s in stats])),
                    "on_blank": float(np.mean([s["on_blank"] for s in stats])),
                    "gkp": float(np.mean([s["gkp"] for s in stats])),
                }
            )
        for channel, stats in markers.items():
            if not stats:
                continue
            rows.append(
                {
                    "revealed": f"{frac:.0%}",
                    "distance": 1,
                    "channel": channel,
                    "row": float(np.mean([s["row"] for s in stats])),
                    "on_solid": float(np.mean([s["on_solid"] for s in stats if s["solid_n"] > 0] or [0.0])),
                    "on_blank": float(np.mean([s["on_blank"] for s in stats])),
                    "gkp": float(np.mean([s["gkp"] for s in stats])),
                }
            )
    return {"rows": rows}


def _mean_distance(reading: dict, distance: int, key: str) -> float:
    picked = [
        r[key]
        for r in reading["rows"]
        if r.get("distance") == distance and "channel" not in r and r.get("kind") != "strip"
    ]
    return float(np.mean(picked)) if picked else 0.0


def _mean_strip(reading: dict, key: str) -> float:
    picked = [r[key] for r in reading["rows"] if r.get("kind") == "strip"]
    return float(np.mean(picked)) if picked else 0.0


def _score_pair(name, model, test, device, stitch: bool):
    if name == "horizon":
        reading = read_arm(model, test, device, "horizon", stitch=True)
        form = reading
    elif name == "stepped":
        reading = read_arm(model, test, device, "stepped", stitch=stitch)
        form = reading
    else:
        reading = read_arm(model, test, device, "stepped", stitch=stitch)
        form = read_arm(model, test, device, "horizon", stitch=True)
    next_week = _mean_distance(reading, 1, "row")
    future = _mean_strip(form, "column")
    if name == "horizon":
        score = future
    elif name == "stepped":
        score = next_week
    else:
        score = 0.5 * (next_week + future)
    return reading, form, next_week, future, score


def train_arm(name: str, model, cards, test, device) -> dict:
    """Stepped paints one gameweek. Horizon paints the remainder as rows."""
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best = float("inf")
    bad = 0
    t0 = time.time()
    epochs_run = 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [int(rng.integers(1, card.n_fixtures)) for card in pick]
            opt.zero_grad()
            loss = None
            if name in ("stepped", "combined"):
                bits, row_on, pad, present, target = _batch_step(pick, cuts, device)
                bits, present, target, pad = shuffle_player_slots(bits, present, target, pad, rng)
                pred = model(bits, row_on, pad)
                week_pred, week_y, week_live = _take(pred, target, present, pad, cuts)
                loss = row_gap(week_pred, week_y, week_live)
            if name in ("horizon", "combined"):
                bits, row_on, _rem, pad, present, target = _batch(pick, cuts, device)
                bits, present, target, pad = shuffle_player_slots(bits, present, target, pad, rng)
                pred = model(bits, row_on, pad)
                horizon_loss = remainder_gap(pred, target, present, pad)
                loss = horizon_loss if loss is None else 0.5 * (loss + horizon_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        reading, _form, next_week, future, score = _score_pair(name, model, test, device, stitch=False)
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
            f"next {next_week:.4f} horizon {future:.4f} "
            f"gkp {_mean_distance(reading, 1, 'gkp'):.3f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 12:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    reading, form, _next, _future, _score = _score_pair(name, model, test, device, stitch=True)
    return {
        "name": name,
        "epochs_run": epochs_run,
        "best": best,
        "seconds": round(time.time() - t0, 1),
        "view": name,
        "reading": reading,
        "form": form,
    }


def _fmt_horizon(meta: dict) -> list[str]:
    lines = []
    for distance in DISTANCES:
        gap = _mean_distance(meta["reading"], distance, "row")
        solid = _mean_distance(meta["reading"], distance, "on_solid")
        blank = _mean_distance(meta["reading"], distance, "on_blank")
        gkp = _mean_distance(meta["reading"], distance, "gkp")
        lines.append(f"| {meta['name']} | {distance} | {gap:.4f} | {solid:.3f} | {blank:.3f} | {gkp:.3f} |")
    return lines


def write_report(arms: list[dict]) -> Path:
    lines = [
        "# Stepped and horizon row opacity",
        "",
        "The loss on a dot is the squared gap, (opacity − truth)². "
        "A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04.",
        "",
        "Both arms paint rows. The stepped arm paints one gameweek: past rows stay true, "
        "and events from that week onward are blank. "
        "The horizon arm paints every week beyond a cutoff in one shot, and scores every live "
        "remainder dot equally. There is no column isolation. "
        "Header and minutes stay on every row. Player seats are shuffled in training.",
        "",
        "Combined gives the one-week gap and the remainder gap equal weight.",
        "",
        "## Next week and further",
        "",
        "Stepped: one gameweek, stepped. Horizon: the remainder beyond the cutoff, one shot.",
        "",
        "| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for meta in arms:
        lines.extend(_fmt_horizon(meta))
    lines.extend(
        [
            "",
            "## Remainder window",
            "",
            "A week eight ahead is read from the one-shot remainder. Events between the cutoff and that week stay blank. Header and minutes of those weeks stay.",
            "",
            "| arm | remainder gap | opacity on solid | opacity on blank | GKP goal opacity |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for meta in arms:
        lines.append(
            f"| {meta['name']} | {_mean_strip(meta['form'], 'column'):.4f} | "
            f"{_mean_strip(meta['form'], 'on_solid'):.3f} | "
            f"{_mean_strip(meta['form'], 'on_blank'):.3f} | "
            f"{_mean_strip(meta['form'], 'gkp'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Event marker, one week ahead",
            "",
            "The first goal dot and the first assist dot.",
            "",
            "| arm | revealed | channel | squared gap | opacity on solid | opacity on blank | GKP |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for meta in arms:
        markers = [r for r in meta["reading"]["rows"] if "channel" in r]
        markers.sort(key=lambda r: (r["channel"] != "goals", r["revealed"]))
        for row in markers:
            lines.append(
                f"| {meta['name']} | {row['revealed']} | {row['channel']} | {row['row']:.4f} | "
                f"{row['on_solid']:.3f} | {row['on_blank']:.3f} | {row['gkp']:.3f} |"
            )
    lines.append("")

    def _arm(name: str) -> dict:
        return next(m for m in arms if m["name"] == name)

    stepped, horizon, both = _arm("stepped"), _arm("horizon"), _arm("combined")
    lines.extend(
        [
            "## Call",
            "",
            "Next-week squared gap: "
            f"stepped **{_mean_distance(stepped['reading'], 1, 'row'):.4f}**, "
            f"horizon **{_mean_distance(horizon['reading'], 1, 'row'):.4f}**, "
            f"combined **{_mean_distance(both['reading'], 1, 'row'):.4f}**. "
            "Remainder gap, every live dot: "
            f"stepped **{_mean_strip(stepped['form'], 'column'):.4f}**, "
            f"horizon **{_mean_strip(horizon['form'], 'column'):.4f}**, "
            f"combined **{_mean_strip(both['form'], 'column'):.4f}**. "
            "Goalkeeper goal opacity on the next week: "
            f"stepped **{_mean_distance(stepped['reading'], 1, 'gkp'):.3f}**, "
            f"horizon **{_mean_distance(horizon['reading'], 1, 'gkp'):.3f}**, "
            f"combined **{_mean_distance(both['reading'], 1, 'gkp'):.3f}**. "
            "The stepped card steps one gameweek at a time. "
            "The horizon card is one shot of the weeks beyond the cutoff.",
            "",
            "Example cards: `reports/baudot_step_stepped_arsenal_2025-26.png`, "
            "`reports/baudot_step_horizon_arsenal_2025-26.png`, "
            "`reports/baudot_step_product_arsenal_2025-26.png`, "
            "`reports/baudot_step_product_stretched_arsenal_2025-26.png`.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def _check_gap_formula() -> None:
    """A solid dot at 0.20 costs 0.64. One week matches the row. Two weeks do not."""
    pred = torch.zeros(1, 1, EVENT_BITS)
    target = torch.zeros(1, 1, EVENT_BITS)
    pred[0, 0, 0] = 0.2
    target[0, 0, 0] = 1.0
    live = torch.ones(1, 1, dtype=torch.bool)
    got = float(row_gap(pred, target, live))
    if abs(got - (0.64 / EVENT_BITS)) > 1e-5:
        raise AssertionError(f"one solid week {got}")
    week = torch.zeros(1, 2, 2, EVENT_BITS)
    truth = torch.zeros(1, 2, 2, EVENT_BITS)
    present = torch.ones(1, 2, 2, dtype=torch.bool)
    pad = torch.zeros(1, 2, dtype=torch.bool)
    week[0, 0, 0, 0] = 0.2
    truth[0, 0, 0, 0] = 1.0
    same = float(remainder_gap(week[:, :, :1], truth[:, :, :1], present[:, :, :1], pad))
    # two live players × one week × EVENT_BITS dots; one solid at 0.20 costs 0.64
    if abs(same - (0.64 / (2 * EVENT_BITS))) > 1e-5:
        raise AssertionError(f"one-week remainder {same}")
    longer = torch.zeros(1, 2, 2, EVENT_BITS)
    longer_y = torch.zeros(1, 2, 2, EVENT_BITS)
    present2 = torch.ones(1, 2, 2, dtype=torch.bool)
    longer[0, 0, 0, 0] = 0.2
    longer_y[0, 0, 0, 0] = 1.0
    longer[0, 0, 1, 0] = 0.8
    strip_t = remainder_gap(longer, longer_y, present2, pad)
    expected = (0.64 + 0.64) / (2 * 2 * EVENT_BITS)
    if abs(float(strip_t.detach()) - expected) > 1e-5:
        raise AssertionError(f"two-week remainder {float(strip_t.detach())}")


def _check_shuffle() -> None:
    bits = torch.zeros(1, 2, HEADER_BITS + MAX_PLAYERS * PLAYER_BITS)
    present = torch.zeros(1, MAX_PLAYERS, 2, dtype=torch.bool)
    target = torch.zeros(1, MAX_PLAYERS, 2, EVENT_BITS)
    pad = torch.ones(1, MAX_PLAYERS, dtype=torch.bool)
    pad[0, :2] = False
    present[0, 0, :] = True
    present[0, 1, :] = True
    target[0, 0, 0, 0] = 1.0
    bits[0, :, HEADER_BITS : HEADER_BITS + WHO_BITS] = 1.0
    bits[0, :, HEADER_BITS + PLAYER_BITS : HEADER_BITS + PLAYER_BITS + WHO_BITS] = 0.5
    # Force a swap of the two live seats.
    class _Swap:
        def permutation(self, n):
            order = np.arange(n)
            order[0], order[1] = 1, 0
            return order

    out_bits, out_present, out_target, out_pad = shuffle_player_slots(bits, present, target, pad, _Swap())
    who0 = out_bits[0, 0, HEADER_BITS : HEADER_BITS + WHO_BITS]
    who1 = out_bits[0, 0, HEADER_BITS + PLAYER_BITS : HEADER_BITS + PLAYER_BITS + WHO_BITS]
    if not torch.allclose(who0, torch.full_like(who0, 0.5)):
        raise AssertionError("seat 0 did not receive the other player's ID bits")
    if not torch.allclose(who1, torch.ones_like(who1)):
        raise AssertionError("seat 1 did not receive the first player's ID bits")
    if float(out_target[0, 0, 0, 0]) != 0.0 or float(out_target[0, 1, 0, 0]) != 1.0:
        raise AssertionError("target did not follow the shuffled seat")
    if bool(out_pad[0, 0]) or bool(out_pad[0, 1]):
        raise AssertionError("pad did not follow the shuffled seat")


def main() -> None:
    _check_gap_formula()
    _check_shuffle()
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
    assert_step_is_blank(test[0])
    print(f"cards train {len(train)} test {len(test)} device {device}", flush=True)
    results = []
    for name in ARMS:
        torch.manual_seed(SEED)
        model = BlockTokenMixer(width=train[0].width)
        results.append(train_arm(name, model, train, test, device))
        torch.save(model.state_dict(), ART / f"baudot_step_gap_{name}.pt")
    path = write_report(results)
    state = {
        "trial": "baudot_stepped_overlap",
        "score": "mean squared gap between opacity and the solid dot",
        "loss": "squared gap; stepped is one gameweek; horizon is the remainder as rows; seats shuffled",
        "arms": [
            {
                "name": meta["name"],
                "epochs_run": meta["epochs_run"],
                "seconds": meta["seconds"],
                "best": meta["best"],
                "row_d1": _mean_distance(meta["reading"], 1, "row"),
                "strip": _mean_strip(meta["form"], "column"),
                "on_blank_d1": _mean_distance(meta["reading"], 1, "on_blank"),
                "on_solid_d1": _mean_distance(meta["reading"], 1, "on_solid"),
                "gkp_d1": _mean_distance(meta["reading"], 1, "gkp"),
                "reading": meta["reading"]["rows"],
            }
            for meta in results
        ],
        "report": str(path),
    }
    (ROOT / "REPORT_STATE.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (ART / "baudot_stepped_overlap.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (REPORTS / "baudot_stepped_overlap.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
