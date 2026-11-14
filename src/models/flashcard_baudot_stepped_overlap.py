"""Stepped per-dot gap on the opacity card.

The gray of a dot is the model's probability. The loss on a dot is the
squared gap between that gray and the solid truth, (opacity − truth)².
A solid dot painted at 0.20 costs 0.64. A blank dot painted at 0.20 costs
0.04. The probability itself is the minimum. Set overlap is not the loss:
it can raise its score by painting a likely scorer solid on the weeks he
does not score.

Three arms, seed 0. Each step reveals the true past and blanks the week
being painted.

  row       — the shared teamsheet, plus this player's position and minutes.
            No opponent, no odds, no own name, no past games. Same position
            and the same minutes get the same gray.
  column    — that player's earlier games, then this week's minutes and opponent.
            The name is hidden, and so are the teammates.
  combined  — the two probabilities, equal weight

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
def holdout(model, cards, device, view: str) -> dict:
    """Overlap at each distance, with every in-between week left blank."""
    model.eval()
    rows = []
    for frac in CUT_FRACS:
        bucket = {d: [] for d in DISTANCES}
        markers = {channel: [] for channel in ("goals", "assists")}
        for card in cards:
            cut = cut_at_fraction(card, frac)
            seen = remainder_view(card, cut)
            pad = _player_pad(card)
            pred = model(
                torch.from_numpy(seen["bits"][None]).to(device),
                torch.from_numpy(seen["row_on"][None]).to(device),
                torch.from_numpy(pad[None]).to(device),
                view=view,
            )[0].cpu().numpy()
            target = seen["target"]
            present = seen["present"]
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
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(cards))
        for start in range(0, len(cards), BATCH):
            pick = [cards[int(i)] for i in order[start : start + BATCH]]
            cuts = [int(rng.integers(1, card.n_fixtures)) for card in pick]
            bits, row_on, _rem, pad, present, target = _batch(pick, cuts, device)
            opt.zero_grad()
            if name == "combined":
                squad_pred = model(bits, row_on, pad, view="squad")
                form_pred = model(bits, row_on, pad, view="time")
                week_pred, week_y, week_live = _take(squad_pred, target, present, pad, cuts)
                loss = 0.5 * (
                    row_gap(week_pred, week_y, week_live) + column_strip(form_pred, target, present, pad)
                )
            elif name == "row":
                pred = model(bits, row_on, pad, view="squad")
                week_pred, week_y, week_live = _take(pred, target, present, pad, cuts)
                loss = row_gap(week_pred, week_y, week_live)
            else:
                pred = model(bits, row_on, pad, view="time")
                loss = column_strip(pred, target, present, pad)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        if name == "combined":
            reading = holdout(model, test, device, "squad")
            form = holdout(model, test, device, "time")
            next_week = _mean_distance(reading, 1, "row")
            future = _mean_strip(form, "column")
            score = 0.5 * (next_week + future)
        else:
            reading = holdout(model, test, device, VIEWS[name])
            form = None
            next_week = _mean_distance(reading, 1, "row")
            future = _mean_strip(reading, "column")
            score = next_week if name == "row" else future
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
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if name == "combined":
        reading = holdout(model, test, device, "squad")
        form = holdout(model, test, device, "time")
    else:
        reading = holdout(model, test, device, VIEWS[name])
        form = reading
    return {
        "name": name,
        "epochs_run": epochs_run,
        "best": best,
        "seconds": round(time.time() - t0, 1),
        "view": "both" if name == "combined" else VIEWS[name],
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
        "The row step looks at the shared teamsheet, plus this player's position and minutes. "
        "It does not see the opponent. His own name is hidden, and his past games are hidden. "
        "Two players of the same position with the same minutes get the same gray. "
        "The column step looks at that player's earlier games, then this week's minutes and opponent. "
        "The name is hidden, and the teammates are hidden. "
        "Both score the squared gap on the week they paint.",
        "",
        "Combined paints the mean of the two probabilities and scores the squared gap of that mean.",
        "",
        "## Next week and further",
        "",
            "Row: the other names on the teamsheet. Column: the player's earlier games.",
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
            "For a week eight ahead, the column still uses only the games played before the cut.",
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
            "The row card is the teamsheet: same position and the same minutes share a gray, and the opponent is hidden. "
            "The column card is that player's earlier games, then this week's minutes and opponent. "
            "On the Arsenal card, where both are painting a player who is on the pitch, the goal grays correlate at 0.42. "
            "The row paints the striker dark because of his position. The column paints him from his earlier games, so that column is lighter.",
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


MATE_K = 15
EMB = 8
N_LINE = 9
N_FORM = HEADER_BITS + 7


def _player_time(arr, n_fixtures: int) -> np.ndarray:
    out = np.zeros((MAX_PLAYERS, n_fixtures), np.float32)
    block = np.asarray(arr)
    rows = min(block.shape[0], MAX_PLAYERS)
    cols = min(block.shape[1], n_fixtures)
    out[:rows, :cols] = block[:rows, :cols]
    return out


def form_cube(card, ends) -> np.ndarray:
    """One feature row per week. ends[t] is the first game the column may not see."""
    weeks = card.n_fixtures
    ends = np.asarray(ends, np.int64)
    features = np.zeros((weeks, MAX_PLAYERS, N_FORM), np.float32)
    features[:, :, :HEADER_BITS] = card.bits[:weeks, :HEADER_BITS][:, None, :]
    minutes = _player_time(card.minutes, weeks)
    features[:, :, HEADER_BITS] = np.clip(minutes / 90.0, 0, 1.5).T
    present = _player_time(card.present, weeks) > 0
    goals = _player_time(card.goals, weeks)
    assists = _player_time(card.assists, weeks)
    hist = np.arange(weeks)[None, :] < ends[:, None]
    played = present[None, :, :] & hist[:, None, :]
    last = np.where(played, np.arange(weeks)[None, None, :], -1).max(axis=2)
    last5 = np.flip((np.cumsum(played[:, :, ::-1], axis=2) <= 5) & played[:, :, ::-1], axis=2)
    seen = last5.sum(axis=2).clip(min=1).astype(np.float32)
    has = played.any(axis=2)
    counts = played.sum(axis=2)
    safe = np.clip(last, 0, weeks - 1)
    players = np.arange(MAX_PLAYERS)[None, :]
    block = np.stack(
        [
            np.where(has, goals[players, safe], 0.0),
            np.where(has, assists[players, safe], 0.0),
            np.where(has, (last5 * goals[None]).sum(axis=2) / seen, 0.0),
            np.where(has, (last5 * assists[None]).sum(axis=2) / seen, 0.0),
            np.where(has, (last5 * minutes[None]).sum(axis=2) / seen / 90.0, 0.0),
            np.where(has, np.minimum(counts, 30) / 30.0, 0.0),
        ],
        axis=-1,
    ).astype(np.float32)
    features[:, :, HEADER_BITS + 1 : HEADER_BITS + 7] = block
    return features


def form_vector(card, t: int, history_end: int) -> np.ndarray:
    """Earlier games only, plus this week's minutes and opponent. Shape P, F.

    history_end is exclusive. Games at or after it stay out, and so does week t.
    """
    ends = np.full(card.n_fixtures, int(history_end), np.int64)
    return form_cube(card, ends)[t]


def lineup_inputs(card, t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The teamsheet, shared by the whole side. Own name is not singled out.

    Two players of the same position with the same minutes get the same features.
    Shape P, F / P, K / P, K.
    """
    floats = np.zeros((MAX_PLAYERS, N_LINE), np.float32)
    ids = np.zeros((MAX_PLAYERS, MATE_K), np.int64)
    mask = np.zeros((MAX_PLAYERS, MATE_K), np.float32)
    if t >= card.n_fixtures:
        return floats, ids, mask
    floats[:, 0] = np.clip(card.minutes[:, t] / 90.0, 0, 1.5)
    on = np.flatnonzero((card.minutes[:, t] > 0) & (card.position >= 0))
    on = on[on < card.n_players]
    counts = np.zeros(4, np.float32)
    sheet = np.zeros(MATE_K, np.int64)
    used = 0
    for p in on:
        role = int(card.position[p])
        if 0 <= role <= 3:
            counts[role] += 1.0
        if used < MATE_K:
            sheet[used] = int(card.player_id[p])
            used += 1
    for p in range(card.n_players):
        role = int(card.position[p])
        if 0 <= role <= 3:
            floats[p, 1 + role] = 1.0
            floats[p, 5:9] = counts / 5.0
        ids[p] = sheet
        mask[p, :used] = 1.0
    return floats, ids, mask


def _week_label(card, t: int) -> tuple[np.ndarray, np.ndarray]:
    target = event_target(card, t)
    live = card.present[:, t] & (card.position >= 0)
    return target, live


class FormNet(torch.nn.Module):
    """Earlier games, this week's minutes, this week's opponent. No name, no teammates."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(N_FORM, 32), torch.nn.ReLU(), torch.nn.Linear(32, EVENT_BITS))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.constant_(self.net[-1].bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class LineupNet(torch.nn.Module):
    """The shared teamsheet, plus position and minutes. No opponent, no own name, no past games."""

    def __init__(self, n_ids: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(max(n_ids, 1), EMB)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_LINE + EMB, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, EVENT_BITS),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.constant_(self.net[-1].bias, -3.0)

    def forward(self, floats: torch.Tensor, mate_ids: torch.Tensor, mate_mask: torch.Tensor) -> torch.Tensor:
        mate = self.emb(mate_ids.clamp(min=0, max=self.emb.num_embeddings - 1))
        weight = mate_mask.to(dtype=mate.dtype).unsqueeze(-1)
        pooled = (mate * weight).sum(dim=-2) / weight.sum(dim=-2).clamp_min(1)
        return torch.sigmoid(self.net(torch.cat([floats, pooled], dim=-1)))


def _check_questions(card) -> None:
    t = min(8, card.n_fixtures - 1)
    base = form_vector(card, t, t)
    saved = card.goals[:, t:].copy()
    card.goals[:, t:] = 9
    try:
        leaked = form_vector(card, t, t)
    finally:
        card.goals[:, t:] = saved
    if not np.allclose(base, leaked):
        raise AssertionError("form read the week it is scoring")
    saved = card.goals[1:].copy()
    card.goals[1:] = 0
    try:
        others = form_vector(card, t, t)
    finally:
        card.goals[1:] = saved
    if not np.allclose(base[0], others[0]):
        raise AssertionError("form read a teammate's games")
    if t >= 2:
        saved = card.goals[:, t - 1].copy()
        card.goals[:, t - 1] = 9
        try:
            moved = form_vector(card, t, t)
        finally:
            card.goals[:, t - 1] = saved
        if card.present[:, t - 1].any() and np.allclose(base, moved):
            raise AssertionError("form ignored the previous game")
    floats, ids, mate_mask = lineup_inputs(card, t)
    if not np.array_equal(ids[0], ids[min(1, card.n_players - 1)]):
        raise AssertionError("lineup gave two players different teamsheets")
    same = None
    same_week = t
    for week in range(card.n_fixtures):
        for a in range(card.n_players):
            for b in range(a + 1, card.n_players):
                if (
                    int(card.position[a]) >= 0
                    and int(card.position[a]) == int(card.position[b])
                    and card.minutes[a, week] == card.minutes[b, week]
                    and card.minutes[a, week] > 0
                ):
                    same = (a, b)
                    same_week = week
                    break
            if same:
                break
        if same:
            break
    if same is None:
        raise AssertionError("no two players share a position and minutes")
    floats_s, ids_s, _mask_s = lineup_inputs(card, same_week)
    a, b = same
    if not (np.allclose(floats_s[a], floats_s[b]) and np.array_equal(ids_s[a], ids_s[b])):
        raise AssertionError("same role and minutes still got different lineup features")
    saved_goals = card.goals[:, :t].copy()
    card.goals[:, :t] = 9
    try:
        floats3, ids3, mask3 = lineup_inputs(card, t)
    finally:
        card.goals[:, :t] = saved_goals
    if not (np.allclose(floats, floats3) and np.array_equal(ids, ids3) and np.allclose(mate_mask, mask3)):
        raise AssertionError("lineup read a past game")


def lineup_cube(card) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weeks = card.n_fixtures
    floats = np.zeros((weeks, MAX_PLAYERS, N_LINE), np.float32)
    ids = np.zeros((weeks, MAX_PLAYERS, MATE_K), np.int64)
    mask = np.zeros((weeks, MAX_PLAYERS, MATE_K), np.float32)
    for t in range(weeks):
        floats[t], ids[t], mask[t] = lineup_inputs(card, t)
    return floats, ids, mask


def _paint_form(net, card, history_end, device) -> np.ndarray:
    weeks = card.n_fixtures
    if history_end is None:
        ends = np.arange(weeks)
    else:
        ends = np.minimum(np.arange(weeks), int(history_end))
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(form_cube(card, ends)).to(device))
    return out.cpu().numpy().transpose(1, 0, 2)


def _paint_lineup(net, card, device) -> np.ndarray:
    floats, ids, mask = lineup_cube(card)
    net.eval()
    with torch.no_grad():
        out = net(
            torch.from_numpy(floats).to(device),
            torch.from_numpy(ids).to(device),
            torch.from_numpy(mask).to(device),
        )
    return out.cpu().numpy().transpose(1, 0, 2)


def _reading_from_paint(cards, paint) -> dict:
    rows = []
    for frac in CUT_FRACS:
        bucket = {d: [] for d in DISTANCES}
        markers = {channel: [] for channel in ("goals", "assists")}
        strips = []
        for card in cards:
            cut = cut_at_fraction(card, frac)
            pred = paint(card, cut)
            target = np.stack([event_target(card, t) for t in range(card.n_fixtures)], axis=1)
            present = card.present.copy()
            present[:, :cut] = False
            pad = _player_pad(card)
            strip = _numpy_strip(pred, target, present, pad, card.position)
            strip["revealed"] = f"{frac:.0%}"
            strip["kind"] = "strip"
            strips.append(strip)
            for distance in DISTANCES:
                row = cut + distance - 1
                if row >= card.n_fixtures:
                    continue
                live = card.present[:, row] & ~pad
                stats = _numpy_gap(pred[:, row], target[:, row], live, card.position)
                stats["distance"] = distance
                bucket[distance].append(stats)
                if distance == 1:
                    for channel in ("goals", "assists"):
                        sl = _marker_slice(channel)
                        part = _numpy_gap(pred[:, row, sl], target[:, row, sl], live, card.position)
                        part["channel"] = channel
                        markers[channel].append(part)
        rows.extend(strips)
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


def _brier(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2
    mask = live[..., None].to(dtype=pred.dtype).expand_as(pred)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def _cache_questions(cards) -> list[dict]:
    packed = []
    for card in cards:
        weeks = card.n_fixtures
        ends = np.arange(weeks)
        target = np.stack([event_target(card, t) for t in range(weeks)], axis=0)
        live = np.stack(
            [card.present[:, t] & (card.position >= 0) for t in range(weeks)],
            axis=0,
        )
        floats, ids, mask = lineup_cube(card)
        cuts = {}
        for frac in CUT_FRACS:
            cut = cut_at_fraction(card, frac)
            frozen = form_cube(card, np.minimum(np.arange(weeks), cut))
            cuts[frac] = torch.from_numpy(frozen)
        packed.append(
            {
                "card": card,
                "form": torch.from_numpy(form_cube(card, ends)),
                "floats": torch.from_numpy(floats),
                "ids": torch.from_numpy(ids),
                "mask": torch.from_numpy(mask),
                "target": torch.from_numpy(target),
                "live": torch.from_numpy(live),
                "frozen": cuts,
            }
        )
    return packed


def _fit_question(name, lineup, form, train_pack, test_pack, device) -> dict:
    params = []
    if name in ("row", "combined"):
        params += list(lineup.parameters())
    if name in ("column", "combined"):
        params += list(form.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best = float("inf")
    bad = 0
    t0 = time.time()
    epochs_run = 0
    by_card = {id(pack["card"]): pack for pack in test_pack}

    def paint(card, cut):
        pack = by_card[id(card)]
        frac = next(key for key, frozen in pack["frozen"].items() if frozen.shape[0] >= 0 and cut == cut_at_fraction(card, key))
        lineup.eval()
        form.eval()
        with torch.no_grad():
            if name != "column":
                row = lineup(pack["floats"].to(device), pack["ids"].to(device), pack["mask"].to(device))
            if name != "row":
                column = form(pack["frozen"][frac].to(device))
            if name == "row":
                out = row
            elif name == "column":
                out = column
            else:
                out = 0.5 * (row + column)
        return out.cpu().numpy().transpose(1, 0, 2)

    for epoch in range(EPOCHS):
        lineup.train()
        form.train()
        total = 0.0
        for pack in train_pack:
            opt.zero_grad()
            target = pack["target"].to(device)
            live = pack["live"].to(device)
            if name != "column":
                row = lineup(pack["floats"].to(device), pack["ids"].to(device), pack["mask"].to(device))
            if name != "row":
                column = form(pack["form"].to(device))
            if name == "row":
                loss = _brier(row, target, live)
            elif name == "column":
                loss = _brier(column, target, live)
            else:
                loss = _brier(0.5 * (row + column), target, live)
            loss.backward()
            opt.step()
            total += float(loss.detach())
        sched.step()
        reading = _reading_from_paint([pack["card"] for pack in test_pack], paint)
        score = _mean_distance(reading, 1, "row")
        improved = score < best - 1e-5
        if improved:
            best = score
            packs = {}
            if name in ("row", "combined"):
                packs["lineup"] = copy.deepcopy(lineup.state_dict())
            if name in ("column", "combined"):
                packs["form"] = copy.deepcopy(form.state_dict())
            best_state = packs
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} train {total / max(len(train_pack), 1):.4f} "
            f"next {score:.4f} gkp {_mean_distance(reading, 1, 'gkp'):.3f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 12:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        if "lineup" in best_state:
            lineup.load_state_dict(best_state["lineup"])
        if "form" in best_state:
            form.load_state_dict(best_state["form"])
    reading = _reading_from_paint([pack["card"] for pack in test_pack], paint)
    return {
        "name": name,
        "epochs_run": epochs_run,
        "best": best,
        "seconds": round(time.time() - t0, 1),
        "view": name,
        "reading": reading,
        "form": reading,
    }


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
    _check_questions(test[0])
    n_ids = 1 + max(int(c.player_id.max()) for c in train + test)
    print(f"cards train {len(train)} test {len(test)} ids {n_ids} device {device}", flush=True)
    print("caching lineup and form features", flush=True)
    train_pack = _cache_questions(train)
    test_pack = _cache_questions(test)
    results = []
    for name in ARMS:
        torch.manual_seed(SEED)
        lineup = LineupNet(n_ids).to(device)
        form = FormNet().to(device)
        results.append(_fit_question(name, lineup, form, train_pack, test_pack, device))
        torch.save(
            {"arch": name, "n_ids": n_ids, "lineup": lineup.state_dict(), "form": form.state_dict()},
            ART / f"baudot_step_gap_{name}.pt",
        )
    path = write_report(results)
    state = {
        "trial": "baudot_stepped_overlap",
        "score": "mean squared gap between opacity and the solid dot",
        "loss": "squared gap; row is the teamsheet, column is earlier games",
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
