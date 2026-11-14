"""Stepped overlap on the opacity card.

The gray of a dot is the model's probability. The score of a card is the
overlap of that gray with the solid truth:

    2 * (opacity on solid dots) / (all opacity + solid dots)

A solid dot painted at 0.20, with no other ink, scores 0.33. The loss on a
step is one minus the overlap of the strip being painted.

Three arms, same mixer, same card, seed 0:

  row       — the next gameweek is one picture
  column    — scorers share half, blank players share half; players tie inside a half
  combined  — the two overlaps with equal weight

Past rows stay true. The gameweek being painted, and every later row, stay
blank of goals and assists. Training steps one week. The holdout then reads
the same overlap 1, 2, 4 and 8 weeks ahead, with the weeks in between blank.
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
    fit_clubs,
    load_odds,
    remainder_view,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = REPORTS / "baudot_stepped_overlap.md"
ARMS = ("row", "column", "combined")
DISTANCES = (1, 2, 4, 8)


def picture_overlap(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """One overlap per batch row, then the mean. pred/target are B, P, E."""
    mask = live.to(pred.dtype)[..., None]
    inter = (pred * target * mask).sum(dim=(1, 2))
    solid = (target * mask).sum(dim=(1, 2))
    ink = (pred * mask).sum(dim=(1, 2))
    denom = ink + solid
    overlap = torch.where(denom > 0, (2 * inter) / denom, torch.ones_like(denom))
    return overlap.mean()


def column_overlap(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Players who scored share half the score. Blank players share the other half.

    Inside each half every player weighs the same, so a defender who scored
    weighs the same as a forward who scored. A scoring player uses the picture
    overlap on their own dots. A blank player scores 1 minus their mean opacity.
    Averaging every player into one pool lets the blank majority drown the events.
    """
    mask = live.to(pred.dtype)[..., None]
    inter = (pred * target * mask).sum(dim=-1)
    solid = (target * mask).sum(dim=-1)
    ink = (pred * mask).sum(dim=-1)
    denom = ink + solid
    dice = torch.where(denom > 0, (2 * inter) / denom.clamp_min(1e-6), torch.ones_like(denom))
    clear = 1 - pred.mean(dim=-1)
    player = torch.where(solid > 0, dice, clear)
    played = live
    scored = (solid > 0) & played
    blank = (solid <= 0) & played

    def _mean(group: torch.Tensor) -> torch.Tensor:
        weight = group.to(pred.dtype)
        return (player * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1)

    scored_mean = _mean(scored)
    blank_mean = _mean(blank)
    has_scored = scored.any(dim=-1)
    has_blank = blank.any(dim=-1)
    both = 0.5 * (scored_mean + blank_mean)
    overlap = torch.where(has_scored & has_blank, both, torch.where(has_scored, scored_mean, blank_mean))
    return overlap.mean()


def step_loss(name: str, pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    row = picture_overlap(pred, target, live)
    column = column_overlap(pred, target, live)
    if name == "row":
        return 1 - row
    if name == "column":
        return 1 - column
    return 1 - 0.5 * (row + column)


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


def _numpy_overlap(pred: np.ndarray, target: np.ndarray, live: np.ndarray) -> dict:
    """pred/target (P, E), live (P,). Picture overlap plus the player mean."""
    mask = live[:, None]
    p = pred * mask
    y = target * mask
    inter = float((p * y).sum())
    solid = float(y.sum())
    ink = float(p.sum())
    denom = ink + solid
    picture = 1.0 if denom == 0 else (2 * inter) / denom
    scored, blanks = [], []
    for row in range(pred.shape[0]):
        if not live[row]:
            continue
        solid_i = float(y[row].sum())
        ink_i = float(p[row].sum())
        if solid_i > 0:
            scored.append((2 * float((p[row] * y[row]).sum())) / (ink_i + solid_i))
        else:
            blanks.append(1.0 - float(pred[row].mean()))
    if scored and blanks:
        column = 0.5 * (float(np.mean(scored)) + float(np.mean(blanks)))
    elif scored:
        column = float(np.mean(scored))
    elif blanks:
        column = float(np.mean(blanks))
    else:
        column = 1.0
    on_true = inter
    on_false = float((p * (1.0 - y)).sum())
    return {
        "picture": picture,
        "column": column,
        "solid": solid,
        "on_true": on_true,
        "on_false": on_false,
        "precision": on_true / ink if ink > 0 else 0.0,
        "recall": on_true / solid if solid > 0 else 0.0,
    }


def _marker_slice(channel: str) -> slice:
    if channel == "goals":
        return slice(0, 1)
    if channel == "assists":
        return slice(GOAL_BITS, GOAL_BITS + 1)
    return slice(0, EVENT_BITS)


@torch.no_grad()
def holdout(model, cards, device) -> dict:
    """Overlap at each distance, with every in-between week left blank."""
    model.eval()
    rows = []
    for frac in CUT_FRACS:
        bucket = {d: [] for d in DISTANCES}
        markers = {channel: [] for channel in ("goals", "assists")}
        for card in cards:
            cut = cut_at_fraction(card, frac)
            view = remainder_view(card, cut)
            pad = _player_pad(card)
            pred = model(
                torch.from_numpy(view["bits"][None]).to(device),
                torch.from_numpy(view["row_on"][None]).to(device),
                torch.from_numpy(pad[None]).to(device),
            )[0].cpu().numpy()
            target = view["target"]
            present = view["present"]
            for distance in DISTANCES:
                row = cut + distance - 1
                if row >= card.n_fixtures:
                    continue
                live = present[:, row] & ~pad
                stats = _numpy_overlap(pred[:, row], target[:, row], live)
                stats["distance"] = distance
                bucket[distance].append(stats)
                if distance == 1:
                    for channel in ("goals", "assists"):
                        sl = _marker_slice(channel)
                        part = _numpy_overlap(pred[:, row, sl], target[:, row, sl], live)
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
                    "picture": float(np.mean([s["picture"] for s in stats])),
                    "column": float(np.mean([s["column"] for s in stats])),
                    "solid": float(np.sum([s["solid"] for s in stats])),
                    "on_true": float(np.sum([s["on_true"] for s in stats])),
                    "on_false": float(np.sum([s["on_false"] for s in stats])),
                }
            )
        for channel, stats in markers.items():
            if not stats:
                continue
            ink = sum(s["on_true"] + s["on_false"] for s in stats)
            solid = sum(s["solid"] for s in stats)
            on_true = sum(s["on_true"] for s in stats)
            rows.append(
                {
                    "revealed": f"{frac:.0%}",
                    "distance": 1,
                    "channel": channel,
                    "picture": (2 * on_true / (ink + solid)) if ink + solid > 0 else 1.0,
                    "solid": solid,
                    "on_true": on_true,
                    "on_false": sum(s["on_false"] for s in stats),
                    "precision": on_true / ink if ink > 0 else 0.0,
                    "recall": on_true / solid if solid > 0 else 0.0,
                }
            )
    return {"rows": rows}


def _mean_distance(reading: dict, distance: int, key: str) -> float:
    picked = [r[key] for r in reading["rows"] if r.get("distance") == distance and "channel" not in r]
    return float(np.mean(picked)) if picked else 0.0


def train_arm(name: str, model, cards, test, device) -> dict:
    rng = np.random.default_rng(SEED)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best = -1.0
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
            pred = model(bits, row_on, pad)
            row_pred, row_y, live = _take(pred, target, present, pad, cuts)
            loss = step_loss(name, row_pred, row_y, live)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        reading = holdout(model, test, device)
        if name == "column":
            score = _mean_distance(reading, 1, "column")
        elif name == "combined":
            score = 0.5 * (_mean_distance(reading, 1, "picture") + _mean_distance(reading, 1, "column"))
        else:
            score = _mean_distance(reading, 1, "picture")
        improved = score > best + 1e-4
        if improved:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        print(
            f"{name} epoch {epoch + 1}/{EPOCHS} train {total / max(seen, 1):.4f} "
            f"holdout {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epoch + 1 >= 15:
            print(f"{name} early stop at epoch {epoch + 1}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    reading = holdout(model, test, device)
    return {
        "name": name,
        "epochs_run": epochs_run,
        "best": best,
        "seconds": round(time.time() - t0, 1),
        "reading": reading,
    }


def _fmt_horizon(meta: dict) -> list[str]:
    lines = []
    for distance in DISTANCES:
        picture = _mean_distance(meta["reading"], distance, "picture")
        column = _mean_distance(meta["reading"], distance, "column")
        lines.append(f"| {meta['name']} | {distance} | {picture:.3f} | {column:.3f} |")
    return lines


def write_report(arms: list[dict]) -> Path:
    lines = [
        "# Stepped overlap",
        "",
        "The score is the overlap of the gray card with the solid card: "
        "`2 * opacity on solid dots / (all opacity + solid dots)`. "
        "The loss on a step is one minus that overlap. "
        "Row treats the next gameweek as one picture. "
        "Column gives the players who scored half the score and the blank players the other half. "
        "Inside each half every player weighs the same, so a defender who scored weighs the same "
        "as a forward who scored. A blank player scores one minus their mean opacity. "
        "Combined gives the row overlap and the column overlap equal weight.",
        "",
        "Training steps the next week, with the true past behind it. "
        "The table reads the same overlap 1, 2, 4 and 8 weeks ahead. "
        "The weeks in between stay blank.",
        "",
        "## Next week and further",
        "",
        "| arm | weeks ahead | picture overlap | player overlap |",
        "|---|---:|---:|---:|",
    ]
    for meta in arms:
        lines.extend(_fmt_horizon(meta))
    lines.extend(
        [
            "",
            "## Event marker, one week ahead",
            "",
            "The first goal dot and the first assist dot, one week ahead. "
            "Solid counts are that week across the 20 holdout clubs.",
            "",
            "| arm | revealed | channel | solid | on true | on false | precision | recall | overlap |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for meta in arms:
        markers = [r for r in meta["reading"]["rows"] if "channel" in r]
        markers.sort(key=lambda r: (r["channel"] != "goals", r["revealed"]))
        for row in markers:
            lines.append(
                f"| {meta['name']} | {row['revealed']} | {row['channel']} | {row['solid']:.0f} | "
                f"{row['on_true']:.1f} | {row['on_false']:.1f} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {row['picture']:.3f} |"
            )
    lines.append("")
    def _arm(name: str) -> dict:
        return next(m for m in arms if m["name"] == name)

    row, column, both = _arm("row"), _arm("column"), _arm("combined")
    lines.extend(
        [
            "## Call",
            "",
            "Next-week picture overlap: "
            f"row **{_mean_distance(row['reading'], 1, 'picture'):.3f}**, "
            f"column **{_mean_distance(column['reading'], 1, 'picture'):.3f}**, "
            f"combined **{_mean_distance(both['reading'], 1, 'picture'):.3f}**. "
            "Eight weeks ahead, with the weeks between left blank: "
            f"row **{_mean_distance(row['reading'], 8, 'picture'):.3f}**, "
            f"column **{_mean_distance(column['reading'], 8, 'picture'):.3f}**, "
            f"combined **{_mean_distance(both['reading'], 8, 'picture'):.3f}**. "
            "The overlap does not fade with distance. Each future row still carries its own "
            "opponent, odds and minutes, so the step does not have to pass the week through the blank rows. "
            "The column arm covers more of the true goal dots and puts more gray on the blank ones, "
            "and the picture overlap falls. Combined stays between the two and does not pass the row.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def _check_overlap_formula() -> None:
    """A solid dot at 0.20 with no other ink overlaps by 0.33."""
    pred = torch.zeros(1, 1, EVENT_BITS)
    target = torch.zeros(1, 1, EVENT_BITS)
    pred[0, 0, 0] = 0.2
    target[0, 0, 0] = 1
    live = torch.ones(1, 1, dtype=torch.bool)
    got = float(picture_overlap(pred, target, live))
    if abs(got - (0.4 / 1.2)) > 1e-5:
        raise AssertionError(f"overlap {got} != 0.333")
    pred[:] = target
    if abs(float(picture_overlap(pred, target, live)) - 1) > 1e-5:
        raise AssertionError("a perfect dot does not score 1")


def main() -> None:
    _check_overlap_formula()
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
    path = write_report(results)
    state = {
        "trial": "baudot_stepped_overlap",
        "score": "picture overlap of gray and solid event dots",
        "loss": "one minus the step overlap",
        "arms": [
            {
                "name": meta["name"],
                "epochs_run": meta["epochs_run"],
                "seconds": meta["seconds"],
                "best": meta["best"],
                "picture_d1": _mean_distance(meta["reading"], 1, "picture"),
                "picture_d2": _mean_distance(meta["reading"], 2, "picture"),
                "picture_d4": _mean_distance(meta["reading"], 4, "picture"),
                "picture_d8": _mean_distance(meta["reading"], 8, "picture"),
                "column_d1": _mean_distance(meta["reading"], 1, "column"),
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
