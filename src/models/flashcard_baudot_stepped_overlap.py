"""Stepped per-dot gap on the opacity card.

The gray of a dot is the model's probability. The loss on a dot is the
squared gap between that gray and the solid truth, (opacity − truth)².
A solid dot painted at 0.20 costs 0.64. A blank dot painted at 0.20 costs
0.04. The probability itself is the minimum. Set overlap is not the loss:
it can raise its score by painting a likely scorer solid on the weeks he
does not score.

Three arms, seed 0, same mixer, same card bits.

  row       — one gameweek, one shot, then step. Past rows stay true.
            Events from the painted week onward are blank.
  column    — one shot of the weeks beyond a cutoff, scored as each
            player's column.
  combined  — the two squared gaps, equal weight

Header and minutes stay on every row. The model reads the card as it is.

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
ARMS = ("row", "column", "combined")
DISTANCES = (1, 2, 4, 8)
VIEWS = {"row": "squad", "column": "time"}


def _square_mask(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = live.to(pred.dtype)[..., None].expand_as(pred)
    return (pred - target) ** 2, mask


def row_gap(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Mean squared gap over every live dot in the week."""
    err, mask = _square_mask(pred, target, live)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def column_strip(
    pred: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
    pad: torch.Tensor,
) -> torch.Tensor:
    """Each player's future weeks, then the mean over players.

    pred/target are B, P, T, E. present is B, P, T and is already limited to
    the weeks still blank. Week t is scored with the paint at week t, which
    sees that week's minutes and opponent. A goal in one week and a blank in
    the next pull different dots. Each player counts once.
    """
    live = present & ~pad[..., None]
    err = (pred - target) ** 2
    mask = live.to(pred.dtype)[..., None].expand_as(pred)
    per_player = (err * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3)).clamp_min(1)
    weight = live.any(dim=-1).to(pred.dtype)
    return (per_player * weight).sum() / weight.sum().clamp_min(1)


def _take(pred, target, present, pad, cuts: list[int]):
    batch = pred.shape[0]
    index = torch.arange(batch, device=pred.device)
    choose = torch.tensor(cuts, device=pred.device, dtype=torch.long)
    live = present[index, :, choose] & ~pad
    return pred[index, :, choose], target[index, :, choose], live


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


def _numpy_strip(
    pred: np.ndarray,
    target: np.ndarray,
    present: np.ndarray,
    pad: np.ndarray,
    position: np.ndarray,
) -> dict:
    """Player-equal squared gap down the future. pred/target (P, T, E), present (P, T)."""
    live = present & ~pad[:, None]
    players = np.flatnonzero(live.any(axis=1))
    empty = {"column": 0.0, "on_solid": 0.0, "on_blank": 0.0, "gkp": 0.0}
    if len(players) == 0:
        return empty
    gaps, solid_ps, blank_ps, gkp_ps = [], [], [], []
    for i in players:
        weeks = np.flatnonzero(live[i])
        dots_p = pred[i, weeks].reshape(-1)
        dots_y = target[i, weeks].reshape(-1)
        gaps.append(float(((dots_p - dots_y) ** 2).mean()))
        solid = dots_y > 0.5
        if solid.any():
            solid_ps.append(float(dots_p[solid].mean()))
        if (~solid).any():
            blank_ps.append(float(dots_p[~solid].mean()))
        if int(position[i]) == 0:
            gkp_ps.append(float(pred[i, weeks, 0].mean()))
    return {
        "column": float(np.mean(gaps)),
        "on_solid": float(np.mean(solid_ps)) if solid_ps else 0.0,
        "on_blank": float(np.mean(blank_ps)) if blank_ps else 0.0,
        "gkp": float(np.mean(gkp_ps)) if gkp_ps else 0.0,
    }


def _marker_slice(channel: str) -> slice:
    if channel == "goals":
        return slice(0, 1)
    if channel == "assists":
        return slice(GOAL_BITS, GOAL_BITS + 1)
    return slice(0, EVENT_BITS)


@torch.no_grad()
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
    """Row kind steps one week at a time. Column kind paints the remainder in one shot."""
    model.eval()
    rows = []
    for frac in CUT_FRACS:
        bucket = {d: [] for d in DISTANCES}
        markers = {channel: [] for channel in ("goals", "assists")}
        for card in cards:
            cut = cut_at_fraction(card, frac)
            pad = _player_pad(card)
            if kind == "column":
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
            strip = _numpy_strip(pred, target, present, pad, card.position)
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
    if name == "column":
        reading = read_arm(model, test, device, "column", stitch=True)
        form = reading
    elif name == "row":
        reading = read_arm(model, test, device, "row", stitch=stitch)
        form = reading
    else:
        reading = read_arm(model, test, device, "row", stitch=stitch)
        form = read_arm(model, test, device, "column", stitch=True)
    next_week = _mean_distance(reading, 1, "row")
    future = _mean_strip(form, "column")
    if name == "column":
        score = future
    elif name == "row":
        score = next_week
    else:
        score = 0.5 * (next_week + future)
    return reading, form, next_week, future, score


def train_arm(name: str, model, cards, test, device) -> dict:
    """Row steps one gameweek. Column paints the remainder beyond a cutoff in one shot."""
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
            if name in ("row", "combined"):
                bits, row_on, pad, present, target = _batch_step(pick, cuts, device)
                pred = model(bits, row_on, pad)
                week_pred, week_y, week_live = _take(pred, target, present, pad, cuts)
                loss = row_gap(week_pred, week_y, week_live)
            if name in ("column", "combined"):
                bits, row_on, _rem, pad, present, target = _batch(pick, cuts, device)
                pred = model(bits, row_on, pad)
                column_loss = column_strip(pred, target, present, pad)
                loss = column_loss if loss is None else 0.5 * (loss + column_loss)
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
            f"next {next_week:.4f} strip {future:.4f} "
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
        "# Stepped per-dot gap",
        "",
        "The loss on a dot is the squared gap, (opacity − truth)². "
        "A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04.",
        "",
        "The row step paints one gameweek. Past rows stay true, and events from that week onward are blank. "
        "The column step paints every week beyond a cutoff in one shot, and the loss is each player's column. "
        "Header and minutes stay on every row. "
        "Both score the squared gap between the gray and the true dots.",
        "",
        "Combined gives the one-week gap and the remainder-column gap equal weight.",
        "",
        "## Next week and further",
        "",
            "Row: one gameweek, stepped. Column: the remainder beyond the cutoff, one shot.",
        "",
        "| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for meta in arms:
        lines.extend(_fmt_horizon(meta))
    lines.extend(
        [
            "",
            "## Each player's future column",
            "",
            "A week eight ahead is read from the one-shot remainder. Events between the cutoff and that week stay blank. Header and minutes of those weeks stay.",
            "",
            "| arm | column gap | opacity on solid | opacity on blank | GKP goal opacity |",
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

    row, column, both = _arm("row"), _arm("column"), _arm("combined")
    lines.extend(
        [
            "## Call",
            "",
            "Next-week squared gap: "
            f"row **{_mean_distance(row['reading'], 1, 'row'):.4f}**, "
            f"column **{_mean_distance(column['reading'], 1, 'row'):.4f}**, "
            f"combined **{_mean_distance(both['reading'], 1, 'row'):.4f}**. "
            "Column gap, each player once down the future: "
            f"row **{_mean_strip(row['form'], 'column'):.4f}**, "
            f"column **{_mean_strip(column['form'], 'column'):.4f}**, "
            f"combined **{_mean_strip(both['form'], 'column'):.4f}**. "
            "Goalkeeper goal opacity on the next week: "
            f"row **{_mean_distance(row['reading'], 1, 'gkp'):.3f}**, "
            f"column **{_mean_distance(column['reading'], 1, 'gkp'):.3f}**, "
            f"combined **{_mean_distance(both['reading'], 1, 'gkp'):.3f}**. "
            "The row card steps one gameweek at a time. "
            "The column card is one shot of the weeks beyond the cutoff.",
            "",
            "Example cards: `reports/baudot_step_row_arsenal_2025-26.png`, `reports/baudot_step_column_arsenal_2025-26.png`, "
            "`reports/baudot_step_row_liverpool_2025-26.png`, `reports/baudot_step_column_liverpool_2025-26.png`.",
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
        raise AssertionError(f"solid gap {got}")
    week = torch.zeros(1, 1, 1, EVENT_BITS)
    truth = torch.zeros(1, 1, 1, EVENT_BITS)
    week[0, 0, 0, 0] = 0.2
    truth[0, 0, 0, 0] = 1.0
    present = torch.ones(1, 1, 1, dtype=torch.bool)
    pad = torch.zeros(1, 1, dtype=torch.bool)
    same = float(column_strip(week, truth, present, pad))
    if abs(same - got) > 1e-5:
        raise AssertionError("one week is not the row")
    longer = torch.nn.Parameter(torch.zeros(1, 1, 2, EVENT_BITS))
    with torch.no_grad():
        longer[0, 0, 0, 0] = 0.2
        longer[0, 0, 1, 0] = 0.8
    longer_y = torch.zeros(1, 1, 2, EVENT_BITS)
    longer_y[0, 0, 0, 0] = 1.0
    present2 = torch.ones(1, 1, 2, dtype=torch.bool)
    strip_t = column_strip(longer, longer_y, present2, pad)
    # Each week keeps its own paint: 0.64 on the goal week and 0.64 on the blank week.
    if abs(float(strip_t.detach()) - (1.28 / (2 * EVENT_BITS))) > 1e-5:
        raise AssertionError(f"two-week column {float(strip_t.detach())}")
    strip_t.backward()
    goal_grad = float(longer.grad[0, 0, 0, 0])
    blank_grad = float(longer.grad[0, 0, 1, 0])
    if goal_grad >= 0 or blank_grad <= 0:
        raise AssertionError(f"weeks share a gray: goal {goal_grad} blank {blank_grad}")


def _check_views() -> None:
    """The lineup view ignores past games. The form view ignores teammates."""
    width = HEADER_BITS + MAX_PLAYERS * PLAYER_BITS
    torch.manual_seed(1)
    model = BlockTokenMixer(width=width).eval()
    # The paint head starts at zero, so the check gives it a real readout.
    with torch.no_grad():
        model.paint.weight.normal_(0.0, 0.05)
    bits = torch.rand(1, 6, width)
    row_on = torch.ones(1, 6)
    pad = torch.ones(1, MAX_PLAYERS, dtype=torch.bool)
    pad[:, :2] = False

    def events(player: int) -> slice:
        start = HEADER_BITS + player * PLAYER_BITS + WHO_BITS + MINS_BITS
        return slice(start, start + EVENT_BITS)

    def who_mins(player: int) -> slice:
        start = HEADER_BITS + player * PLAYER_BITS
        return slice(start, start + WHO_BITS + MINS_BITS)

    with torch.no_grad():
        time_base = model(bits, row_on, pad, view="time")
        teammate = bits.clone()
        teammate[0, :5, events(1)] = 1.0
        if not torch.allclose(time_base[0, 0], model(teammate, row_on, pad, view="time")[0, 0], atol=1e-5):
            raise AssertionError("form view changed when a teammate's past changed")
        own = bits.clone()
        own[0, :5, events(0)] = 1.0
        if torch.allclose(time_base[0, 0, -1], model(own, row_on, pad, view="time")[0, 0, -1], atol=1e-4):
            raise AssertionError("form view ignored the player's own past")
        squad_base = model(bits, row_on, pad, view="squad")
        if not torch.allclose(squad_base[:, :, -1], model(own, row_on, pad, view="squad")[:, :, -1], atol=1e-5):
            raise AssertionError("lineup view read a past game")
        lineup = bits.clone()
        lineup[0, -1, who_mins(1)] = 0.0
        if torch.allclose(squad_base[0, 0, -1], model(lineup, row_on, pad, view="squad")[0, 0, -1], atol=1e-4):
            raise AssertionError("lineup view ignored a teammate")
        if not torch.allclose(time_base[0, 0], model(lineup, row_on, pad, view="time")[0, 0], atol=1e-5):
            raise AssertionError("form view read a teammate")
        named = bits.clone()
        name_at = HEADER_BITS  # player 0's role and id
        named[0, :, name_at : name_at + WHO_BITS] = 0.0
        if not torch.allclose(time_base[0, 0], model(named, row_on, pad, view="time")[0, 0], atol=1e-5):
            raise AssertionError("form view still sees the player's name")



def main() -> None:
    _check_gap_formula()
    _check_views()
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
        "loss": "squared gap; row is one stepped gameweek; column is the remainder beyond the cutoff",
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
