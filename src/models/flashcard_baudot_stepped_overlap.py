"""Stepped per-dot gap on the opacity card.

The gray of a dot is the model's probability. The loss on a dot is the
squared gap between that gray and the solid truth, (opacity − truth)².
A solid dot painted at 0.20 costs 0.64. A blank dot painted at 0.20 costs
0.04. The probability itself is the minimum. Set overlap is not the loss:
it can raise its score by painting a likely scorer solid on the weeks he
does not score.

Three arms, same mixer, same card, seed 0. Each step reveals the true past
and blanks the week being painted.

  row       — mean squared gap over the dots of that week
  column    — each player's own mean squared gap, then the mean over players
  combined  — the two gaps with equal weight

The holdout reads the same gap 1, 2, 4 and 8 weeks ahead.
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


def _square_mask(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = live.to(pred.dtype)[..., None].expand_as(pred)
    return (pred - target) ** 2, mask


def row_gap(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Mean squared gap over every live dot in the week."""
    err, mask = _square_mask(pred, target, live)
    return (err * mask).sum() / mask.sum().clamp_min(1)


def column_gap(pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """Each player's mean squared gap, then the mean over the players who played.

    A goalkeeper's six dots weigh the same as a forward's six. Filling a
    goalkeeper's squares costs that player a full miss.
    """
    err, mask = _square_mask(pred, target, live)
    per_player = (err * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    weight = live.to(pred.dtype)
    return (per_player * weight).sum() / weight.sum().clamp_min(1)


def step_loss(name: str, pred: torch.Tensor, target: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    row = row_gap(pred, target, live)
    column = column_gap(pred, target, live)
    if name == "row":
        return row
    if name == "column":
        return column
    return 0.5 * (row + column)


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
        return {"row": 0.0, "column": 0.0, "on_solid": 0.0, "on_blank": 0.0, "gkp": 0.0, "solid_n": 0}
    dots_p = pred[played].reshape(-1)
    dots_y = target[played].reshape(-1)
    gap = (dots_p - dots_y) ** 2
    solid = dots_y > 0.5
    per_player = ((pred[played] - target[played]) ** 2).mean(axis=1)
    gkp = played[position[played] == 0]
    gkp_p = float(pred[gkp, 0].mean()) if len(gkp) else 0.0
    return {
        "row": float(gap.mean()),
        "column": float(per_player.mean()),
        "on_solid": float(dots_p[solid].mean()) if solid.any() else 0.0,
        "on_blank": float(dots_p[~solid].mean()) if (~solid).any() else 0.0,
        "gkp": gkp_p,
        "solid_n": int(solid.sum()),
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
                    "column": float(np.mean([s["column"] for s in stats])),
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
    picked = [r[key] for r in reading["rows"] if r.get("distance") == distance and "channel" not in r]
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
            score = 0.5 * (_mean_distance(reading, 1, "row") + _mean_distance(reading, 1, "column"))
        else:
            score = _mean_distance(reading, 1, "row")
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
        "The set-overlap score is withdrawn. It can improve by painting a likely scorer "
        "solid on the weeks he does not score, which filled the attacker columns and, "
        "on the column arm, the goalkeeper squares.",
        "",
        "The loss is now the squared gap on each dot, (opacity − truth)². "
        "A solid dot at 0.20 costs 0.64. A blank dot at 0.20 costs 0.04. "
        "The minimum is the probability of the dot. "
        "Row averages the dots of the week. Column averages each player first, "
        "so a goalkeeper's squares weigh the same as a forward's. "
        "Combined gives the two averages equal weight.",
        "",
        "Opacity on a solid dot should sit near the chance of that event. "
        "Opacity on a blank dot, and on a goalkeeper's goal dot, should sit near zero.",
        "",
        "## Next week and further",
        "",
        "| arm | weeks ahead | squared gap | opacity on solid | opacity on blank | GKP goal opacity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for meta in arms:
        lines.extend(_fmt_horizon(meta))
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
            f"column **{_mean_distance(column['reading'], 1, 'column'):.4f}**, "
            f"combined **{0.5 * (_mean_distance(both['reading'], 1, 'row') + _mean_distance(both['reading'], 1, 'column')):.4f}**. "
            f"Blank-dot opacity **{_mean_distance(row['reading'], 1, 'on_blank'):.3f}**. "
            f"Opacity on a solid dot **{_mean_distance(row['reading'], 1, 'on_solid'):.3f}**. "
            f"Goalkeeper goal opacity **{_mean_distance(row['reading'], 1, 'gkp'):.3f}**. "
            "The three arms match. Every player has six event dots, so the mean over dots "
            "and the mean over players are the same number, and the combined arm is that number again. "
            "The column split only changed the old overlap ratio. "
            "A true goal marker at the 20% cut is "
            f"**{next(r['on_solid'] for r in row['reading']['rows'] if r.get('channel') == 'goals' and r['revealed'] == '20%'):.3f}** "
            "opaque, and a blank goal marker is "
            f"**{next(r['on_blank'] for r in row['reading']['rows'] if r.get('channel') == 'goals' and r['revealed'] == '20%'):.3f}**.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def _check_gap_formula() -> None:
    """A solid dot at 0.20 costs 0.64. A blank dot at 0.80 costs 0.64. A perfect dot costs 0."""
    pred = torch.zeros(1, 1, EVENT_BITS)
    target = torch.zeros(1, 1, EVENT_BITS)
    pred[0, 0, 0] = 0.2
    target[0, 0, 0] = 1.0
    live = torch.ones(1, 1, dtype=torch.bool)
    # five blank bits at 0 plus one solid bit at 0.2: mean square = 0.64 / 6
    got = float(row_gap(pred, target, live))
    if abs(got - (0.64 / EVENT_BITS)) > 1e-5:
        raise AssertionError(f"solid gap {got}")
    pred = torch.zeros(1, 1, EVENT_BITS)
    pred[0, 0, 0] = 0.8
    target = torch.zeros(1, 1, EVENT_BITS)
    got = float(row_gap(pred, target, live))
    if abs(got - (0.64 / EVENT_BITS)) > 1e-5:
        raise AssertionError(f"blank gap {got}")
    pred = target.clone()
    if float(row_gap(pred, target, live)) > 1e-6:
        raise AssertionError("a matching card is not a zero gap")


def main() -> None:
    _check_gap_formula()
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
        "loss": "squared per-dot gap on the stepped week",
        "arms": [
            {
                "name": meta["name"],
                "epochs_run": meta["epochs_run"],
                "seconds": meta["seconds"],
                "best": meta["best"],
                "row_d1": _mean_distance(meta["reading"], 1, "row"),
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
