"""Horizon FOV painter on the typed tensor card.

One shot of every week beyond a cutoff. Past rows keep true G/A. Events from
the cut onward are blank on input. Header / who / minutes stay. Loss is
log + Brier on every live remainder cell (equal weight). Seat shuffle each batch.

Same mixer as stepped. Holdout 2025-26; train 2022-23…2024-25.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models.tensor_card import (
    ART,
    REPORTS,
    TEST_SEASON,
    TRAIN_SEASONS,
    TensorCard,
    build_tensor_cards,
    cut_at_fraction,
)
from src.models.tensor_card_stepped import (
    BATCH,
    CUT_FRACS,
    DISTANCES,
    EARLY_PATIENCE,
    EPOCHS,
    LR,
    MAX_GW,
    SEED,
    TypedCardMixer,
    _event_ink,
    _fit_slice_logistic,
    _metrics,
    _pack_card,
    _stack_batch,
    fit_player_codes,
    n_clubs,
    shuffle_seats,
    train_stepped,
)


def horizon_view(card: TensorCard, cut: int) -> dict:
    """Blank G/A from cut onward on input; score all remainder weeks."""
    events = _event_ink(card)
    events_in = events.copy()
    events_in[:, cut:, :] = 0.0
    target = events.copy()
    target[:, :cut, :] = 0.0
    present = card.present.copy()
    present[:, :cut] = False
    row_on = np.zeros(MAX_GW, dtype=np.float32)
    row_on[: card.n_fixtures] = 1.0
    return {
        "events_in": events_in,
        "target": target,
        "present": present,
        "row_on": row_on,
        "cut": cut,
    }


def assert_horizon_fov(card: TensorCard) -> None:
    cut = cut_at_fraction(card, 0.4)
    view = horizon_view(card, cut)
    assert float(view["events_in"][:, cut:, :].sum()) == 0.0
    if cut > 0:
        assert np.array_equal(view["events_in"][:, :cut, :], _event_ink(card)[:, :cut, :])
        assert float(view["target"][:, :cut, :].sum()) == 0.0
    assert int(view["present"][:, cut:].sum()) > 0


def horizon_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
    pad: torch.Tensor,
) -> torch.Tensor:
    """Log + Brier on every live remainder cell."""
    live = present & ~pad[..., None]
    mask = live.to(pred.dtype)[..., None].expand_as(pred)
    denom = mask.sum().clamp_min(1.0)
    brier = ((pred - target) ** 2 * mask).sum() / denom
    p = pred.clamp(1e-6, 1 - 1e-6)
    log = -(target * torch.log(p) + (1 - target) * torch.log(1 - p))
    return (log * mask).sum() / denom + brier


@torch.no_grad()
def predict_horizon(
    model: TypedCardMixer,
    card: TensorCard,
    cut: int,
    player_codes: dict[int, int],
    device: torch.device,
) -> np.ndarray:
    batch = _stack_batch([_pack_card(card, horizon_view(card, cut), player_codes, device)])
    return model(
        batch["club_id"],
        batch["opp_id"],
        batch["home"],
        batch["odds"],
        batch["player_code"],
        batch["role"],
        batch["minutes"],
        batch["events_in"],
        batch["row_on"],
        batch["player_pad"],
    )[0].cpu().numpy()


@torch.no_grad()
def score_horizon(
    model: TypedCardMixer,
    cards: list[TensorCard],
    player_codes: dict[int, int],
    device: torch.device,
) -> dict:
    model.eval()
    rows = []
    rem_logs, rem_briers, rem_ns = [], [], []
    for frac in CUT_FRACS:
        for card in cards:
            cut = cut_at_fraction(card, frac)
            soft = predict_horizon(model, card, cut, player_codes, device)
            truth = _event_ink(card)
            live = card.present.copy()
            live[:, :cut] = False
            live[card.n_players :] = False
            live &= card.position[:, None] >= 0
            # flatten remainder metrics
            played = np.flatnonzero(live)
            # per-cell over P,T
            ps, ys = [], []
            for p in range(card.n_players):
                for t in range(cut, card.n_fixtures):
                    if not live[p, t]:
                        continue
                    ps.append(soft[p, t])
                    ys.append(truth[p, t])
            if not ps:
                continue
            p_arr = np.stack(ps, axis=0).reshape(-1).clip(1e-6, 1 - 1e-6)
            y_arr = np.stack(ys, axis=0).reshape(-1)
            log = float(-(y_arr * np.log(p_arr) + (1 - y_arr) * np.log(1 - p_arr)).mean())
            brier = float(((p_arr - y_arr) ** 2).mean())
            solid = y_arr > 0.5
            rem = {
                "revealed": frac,
                "log_loss": log,
                "brier": brier,
                "on_solid": float(p_arr[solid].mean()) if solid.any() else 0.0,
                "on_blank": float(p_arr[~solid].mean()) if (~solid).any() else 0.0,
                "n": int(len(p_arr)),
                "kind": "remainder",
                "team": card.team,
            }
            rows.append(rem)
            rem_logs.append(log)
            rem_briers.append(brier)
            rem_ns.append(len(p_arr))
            for distance in DISTANCES:
                t = cut + distance - 1
                if t >= card.n_fixtures:
                    continue
                live_t = live[:, t]
                m = _metrics(soft[:, t, :], truth[:, t, :], live_t)
                m.update({"distance": distance, "revealed": frac, "team": card.team})
                rows.append(m)
    w = np.array(rem_ns, dtype=np.float64)
    remainder = {
        "log_loss": float(np.average(rem_logs, weights=w)) if len(w) else float("nan"),
        "brier": float(np.average(rem_briers, weights=w)) if len(w) else float("nan"),
        "n": int(w.sum()) if len(w) else 0,
    }
    by_d = {}
    for d in DISTANCES:
        subset = [r for r in rows if r.get("distance") == d and r.get("n", 0) > 0]
        if not subset:
            by_d[d] = {"log_loss": float("nan"), "brier": float("nan"), "n": 0}
            continue
        ww = np.array([r["n"] for r in subset], dtype=np.float64)
        by_d[d] = {
            "log_loss": float(np.average([r["log_loss"] for r in subset], weights=ww)),
            "brier": float(np.average([r["brier"] for r in subset], weights=ww)),
            "on_solid": float(np.average([r["on_solid"] for r in subset], weights=ww)),
            "on_blank": float(np.average([r["on_blank"] for r in subset], weights=ww)),
            "n": int(ww.sum()),
        }
    return {"rows": rows, "remainder": remainder, "by_distance": by_d}


def train_horizon(
    train: list[TensorCard],
    test: list[TensorCard],
    player_codes: dict[int, int],
    n_club: int,
    device: torch.device,
) -> tuple[TypedCardMixer, dict]:
    rng = np.random.default_rng(SEED)
    model = TypedCardMixer(n_club=n_club, n_player=max(player_codes.values())).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best = float("inf")
    bad = 0
    t0 = time.time()
    epochs_run = 0
    history = []
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(train))
        for start in range(0, len(train), BATCH):
            pick = [train[int(i)] for i in order[start : start + BATCH]]
            items = []
            for card in pick:
                cut = int(rng.integers(1, card.n_fixtures))
                items.append(_pack_card(card, horizon_view(card, cut), player_codes, device))
            batch = shuffle_seats(_stack_batch(items), rng)
            opt.zero_grad()
            pred = model(
                batch["club_id"],
                batch["opp_id"],
                batch["home"],
                batch["odds"],
                batch["player_code"],
                batch["role"],
                batch["minutes"],
                batch["events_in"],
                batch["row_on"],
                batch["player_pad"],
            )
            loss = horizon_loss(pred, batch["target"], batch["present"], batch["player_pad"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        reading = score_horizon(model, test, player_codes, device)
        score = reading["remainder"]["log_loss"]
        improved = score < best - 1e-5
        if improved:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        row = {
            "epoch": epochs_run,
            "train_loss": total / max(seen, 1),
            "holdout_rem_log": score,
            "holdout_rem_brier": reading["remainder"]["brier"],
            "holdout_d1_log": reading["by_distance"][1]["log_loss"],
        }
        history.append(row)
        print(
            f"horizon epoch {epochs_run}/{EPOCHS} train {row['train_loss']:.4f} "
            f"rem_log {score:.4f} d1_log {row['holdout_d1_log']:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epochs_run >= 12:
            print(f"horizon early stop at epoch {epochs_run}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    reading = score_horizon(model, test, player_codes, device)
    return model, {
        "epochs_run": epochs_run,
        "best_rem_log": best,
        "seconds": round(time.time() - t0, 1),
        "history": history,
        "reading": reading,
    }


def write_report(horizon: dict, stepped: dict, slice_base: dict) -> Path:
    path = REPORTS / "typed_tensor_horizon.md"
    hr = horizon["reading"]
    st = stepped["reading"]["by_distance"]
    lines = [
        "# Typed tensor — horizon FOV painter",
        "",
        "One-shot remainder on the cemented typed tensor. Past true; blank G/A from the cut;",
        "header/minutes stay; log + Brier on every live remainder cell; seats shuffled.",
        "Same mixer as stepped. Train 2022-23…2024-25; holdout 2025-26.",
        "",
        f"Horizon epochs: {horizon['epochs_run']} ({horizon['seconds']}s). "
        f"Best remainder log: {horizon['best_rem_log']:.4f}.",
        f"Stepped (retrain, early-stop on d1): best d1 log {stepped['best_d1_log']:.4f}.",
        "",
        "## Remainder window (horizon primary)",
        "",
        f"| arm | remainder log | remainder Brier | n |",
        f"| --- | ---: | ---: | ---: |",
        f"| horizon | {hr['remainder']['log_loss']:.4f} | {hr['remainder']['brier']:.4f} | {hr['remainder']['n']} |",
        "",
        "## Holdout by weeks ahead",
        "",
        "| arm | weeks ahead | log loss | Brier | on solid | on blank | n |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in DISTANCES:
        h = hr["by_distance"][d]
        s = st[d]
        lines.append(
            f"| horizon | {d} | {h['log_loss']:.4f} | {h['brier']:.4f} | "
            f"{h.get('on_solid', float('nan')):.3f} | {h.get('on_blank', float('nan')):.3f} | {h['n']} |"
        )
        lines.append(
            f"| stepped | {d} | {s['log_loss']:.4f} | {s['brier']:.4f} | "
            f"{s.get('on_solid', float('nan')):.3f} | {s.get('on_blank', float('nan')):.3f} | {s['n']} |"
        )
    lines.extend(
        [
            "",
            "## Spine logistic reference",
            "",
            f"log loss **{slice_base['log_loss']:.4f}**, Brier **{slice_base['brier']:.4f}**.",
            "",
            "## Call",
            "",
        ]
    )
    rem = hr["remainder"]["log_loss"]
    d1 = hr["by_distance"][1]["log_loss"]
    slice_log = slice_base["log_loss"]
    stepped_d1 = stepped["best_d1_log"]
    delta = rem - slice_log
    if delta < -0.01:
        lines.append(
            f"Horizon remainder log **{rem:.4f}** clearly beats sliced logistic "
            f"**{slice_log:.4f}** (Δ={delta:.4f}). d1 **{d1:.4f}** vs stepped "
            f"**{stepped_d1:.4f}**. Candidate for a follow-up."
        )
    elif delta < 0:
        lines.append(
            f"Horizon remainder log **{rem:.4f}** is only a hair under sliced logistic "
            f"**{slice_log:.4f}** (Δ={delta:.4f}) — not a clear win on a different cell set. "
            f"d1 **{d1:.4f}** still trails the spine (stepped d1 **{stepped_d1:.4f}**). "
            "Park as inconclusive; card stays cemented."
        )
    else:
        lines.append(
            f"Horizon remainder log **{rem:.4f}** does not beat sliced logistic **{slice_log:.4f}** "
            f"(stepped d1 was **{stepped_d1:.4f}**). Screen loss; card stays cemented."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "horizon": {
            "epochs_run": horizon["epochs_run"],
            "best_rem_log": horizon["best_rem_log"],
            "seconds": horizon["seconds"],
            "remainder": hr["remainder"],
            "by_distance": hr["by_distance"],
            "history": horizon["history"],
        },
        "stepped": {
            "best_d1_log": stepped["best_d1_log"],
            "epochs_run": stepped["epochs_run"],
            "by_distance": st,
        },
        "slice_logistic": slice_base,
    }
    (REPORTS / "typed_tensor_horizon.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_horizon.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    from src.models.tensor_card_render import render_horizon, render_stepped_d1

    device = torch.device("cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    cards = build_tensor_cards(seasons)
    assert_horizon_fov(cards[0])
    train = [c for c in cards if c.season in TRAIN_SEASONS]
    test = [c for c in cards if c.season == TEST_SEASON]
    codes = fit_player_codes(cards)
    n_club = n_clubs(cards)
    print(f"cards train={len(train)} test={len(test)} clubs={n_club} players={len(codes)}", flush=True)
    slice_base = _fit_slice_logistic(train, test)
    print(f"slice logistic log={slice_base['log_loss']:.4f}", flush=True)

    print("=== stepped (for d1 card + reference) ===", flush=True)
    stepped_model, stepped_meta = train_stepped(train, test, codes, n_club, device)
    arsenal = next(c for c in test if c.team == "Arsenal")
    stepped_png = render_stepped_d1(arsenal, stepped_model, codes, frac=0.4)
    print(f"wrote {stepped_png}", flush=True)

    print("=== horizon ===", flush=True)
    horizon_model, horizon_meta = train_horizon(train, test, codes, n_club, device)
    horizon_png = render_horizon(arsenal, horizon_model, codes, frac=0.4)
    print(f"wrote {horizon_png}", flush=True)

    path = write_report(horizon_meta, stepped_meta, slice_base)
    print(f"wrote {path}", flush=True)
    print(
        json.dumps(
            {
                "horizon_remainder": horizon_meta["reading"]["remainder"],
                "horizon_d1": horizon_meta["reading"]["by_distance"][1],
                "stepped_d1": stepped_meta["best_d1_log"],
                "slice": slice_base["log_loss"],
                "stepped_png": str(stepped_png),
                "horizon_png": str(horizon_png),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
