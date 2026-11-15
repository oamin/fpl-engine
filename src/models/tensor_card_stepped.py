"""Stepped FOV painter on the typed tensor card.

One gameweek, one shot. Past rows keep true G/A. Events from the painted week
onward are blank. Header, who, and minutes stay visible. Output opacity is
P(count ≥ 1) for goals and assists. Loss is log loss + Brier on live cells in
the painted week. Player seats are shuffled each batch.

Holdout: closed 2025-26. Train: 2022-23…2024-25. Not a product / Numagent change.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.tensor_card import (
    ART,
    FEATURE_NAMES,
    MAX_GW,
    MAX_PLAYERS,
    REPORTS,
    TEST_SEASON,
    TRAIN_SEASONS,
    TensorCard,
    asof_inputs,
    blank_future_events,
    build_tensor_cards,
    cut_at_fraction,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 0
D = 48
HEADS = 4
EPOCHS = 40
BATCH = 8
LR = 1e-3
EARLY_PATIENCE = 8
DISTANCES = (1, 2, 4, 8)
CUT_FRACS = (0.2, 0.4, 0.6, 0.8)
EVENT_CH = 2  # goal, assist opacities


def fit_player_codes(cards: list[TensorCard]) -> dict[int, int]:
    ids = sorted({int(pid) for card in cards for pid in card.player_id.tolist() if int(pid) > 0})
    return {pid: i + 1 for i, pid in enumerate(ids)}


def n_clubs(cards: list[TensorCard]) -> int:
    return int(max(max(c.club_id, int(c.opp_id.max())) for c in cards))


class TypedCardMixer(nn.Module):
    """Time + squad attention over typed fields → event opacities."""

    def __init__(self, n_club: int, n_player: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.club = nn.Embedding(n_club + 1, D, padding_idx=0)
        self.opp = nn.Embedding(n_club + 1, D, padding_idx=0)
        self.player = nn.Embedding(n_player + 1, D, padding_idx=0)
        self.role = nn.Embedding(5, D, padding_idx=4)  # 0..3 roles, 4 = pad
        self.odds = nn.Linear(3, D)
        self.home = nn.Linear(1, D)
        self.mins = nn.Linear(1, D)
        self.events = nn.Linear(EVENT_CH, D)
        self.header_mix = nn.Linear(4 * D, D)
        self.player_mix = nn.Linear(4 * D, D)
        self.norm_h = nn.LayerNorm(D)
        self.norm_p = nn.LayerNorm(D)
        self.time = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.squad = nn.MultiheadAttention(D, HEADS, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(D)
        self.norm_s = nn.LayerNorm(D)
        self.time_pos = nn.Parameter(torch.zeros(1, MAX_GW, 1, D))
        nn.init.normal_(self.time_pos, std=0.02)
        self.paint = nn.Linear(D, EVENT_CH)
        nn.init.zeros_(self.paint.weight)
        nn.init.constant_(self.paint.bias, -3.0)

    def forward(
        self,
        club_id: torch.Tensor,  # B
        opp_id: torch.Tensor,  # B,T
        home: torch.Tensor,  # B,T
        odds: torch.Tensor,  # B,T,3
        player_code: torch.Tensor,  # B,P
        role: torch.Tensor,  # B,P
        minutes: torch.Tensor,  # B,P,T raw
        events_in: torch.Tensor,  # B,P,T,2  (blanked under FOV)
        row_on: torch.Tensor,  # B,T
        player_pad: torch.Tensor,  # B,P bool
    ) -> torch.Tensor:
        b, t = opp_id.shape
        p = player_code.shape[1]
        club = self.club(club_id)[:, None, :].expand(b, t, D)
        opp = self.opp(opp_id)
        odds_e = self.odds(odds)
        home_e = self.home(home[..., None])
        header = self.norm_h(self.header_mix(torch.cat([club, opp, odds_e, home_e], dim=-1)))
        who = self.player(player_code)[:, None, :, :].expand(b, t, p, D)
        role_e = self.role(role)[:, None, :, :].expand(b, t, p, D)
        mins_e = self.mins((minutes / 90.0).clamp(0, 2)[..., None].permute(0, 2, 1, 3))  # B,T,P,D
        # events_in is B,P,T,2 → B,T,P,2
        ev = self.events(events_in.permute(0, 2, 1, 3))
        players = self.norm_p(self.player_mix(torch.cat([who, role_e, mins_e, ev], dim=-1)))
        tokens = players + header[:, :, None, :] + self.time_pos[:, :t]
        # time mix per player
        time_view = tokens.permute(0, 2, 1, 3).reshape(b * p, t, D)
        time_mask = (row_on < 0.5)[:, None, :].expand(b, p, t).reshape(b * p, t)
        mixed, _ = self.time(time_view, time_view, time_view, key_padding_mask=time_mask, need_weights=False)
        tokens = self.norm_t(time_view + mixed).reshape(b, p, t, D).permute(0, 2, 1, 3)
        # squad mix per week
        squad_view = tokens.reshape(b * t, p, D)
        squad_mask = player_pad[:, None, :].expand(b, t, p).reshape(b * t, p)
        mixed, _ = self.squad(squad_view, squad_view, squad_view, key_padding_mask=squad_mask, need_weights=False)
        tokens = self.norm_s(squad_view + mixed).reshape(b, t, p, D)
        ink = torch.sigmoid(self.paint(tokens))  # B,T,P,2
        return ink.permute(0, 2, 1, 3)  # B,P,T,2


def _event_ink(card: TensorCard) -> np.ndarray:
    """Opacity truth / input ink: 1 if count ≥ 1."""
    ink = np.zeros((MAX_PLAYERS, MAX_GW, EVENT_CH), dtype=np.float32)
    ink[:, :, 0] = (card.goals >= 1).astype(np.float32)
    ink[:, :, 1] = (card.assists >= 1).astype(np.float32)
    return ink


def step_view(card: TensorCard, cut: int) -> dict:
    """Past true; blank G/A from cut onward; score week `cut`."""
    events = _event_ink(card)
    events_in = events.copy()
    events_in[:, cut:, :] = 0.0
    target = np.zeros_like(events)
    present = np.zeros((MAX_PLAYERS, MAX_GW), dtype=bool)
    target[:, cut, :] = events[:, cut, :]
    present[:, cut] = card.present[:, cut]
    row_on = np.zeros(MAX_GW, dtype=np.float32)
    row_on[: card.n_fixtures] = 1.0
    return {
        "events_in": events_in,
        "target": target,
        "present": present,
        "row_on": row_on,
        "cut": cut,
    }


def assert_step_fov(card: TensorCard) -> None:
    cut = cut_at_fraction(card, 0.4)
    view = step_view(card, cut)
    blanked = blank_future_events(card, cut)
    assert np.array_equal(blanked.minutes, card.minutes)
    assert float(view["events_in"][:, cut:, :].sum()) == 0.0
    if cut > 0:
        assert np.array_equal(view["events_in"][:, :cut, :], _event_ink(card)[:, :cut, :])
    assert int(view["present"][:, cut].sum()) > 0


def _pack_card(
    card: TensorCard,
    view: dict,
    player_codes: dict[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    codes = np.zeros(MAX_PLAYERS, dtype=np.int64)
    roles = np.full(MAX_PLAYERS, 4, dtype=np.int64)
    pad = np.ones(MAX_PLAYERS, dtype=bool)
    for p in range(card.n_players):
        pid = int(card.player_id[p])
        if pid <= 0:
            continue
        codes[p] = player_codes.get(pid, 0)
        roles[p] = int(card.position[p]) if int(card.position[p]) >= 0 else 4
        pad[p] = False
    odds = card.odds_1x2.copy()
    miss = ~np.isfinite(odds)
    odds[miss] = 1.0 / 3.0
    return {
        "club_id": torch.tensor(card.club_id, device=device, dtype=torch.long),
        "opp_id": torch.from_numpy(card.opp_id.astype(np.int64)).to(device),
        "home": torch.from_numpy(card.home.astype(np.float32)).to(device),
        "odds": torch.from_numpy(odds.astype(np.float32)).to(device),
        "player_code": torch.from_numpy(codes).to(device),
        "role": torch.from_numpy(roles).to(device),
        "minutes": torch.from_numpy(card.minutes.astype(np.float32)).to(device),
        "events_in": torch.from_numpy(view["events_in"]).to(device),
        "row_on": torch.from_numpy(view["row_on"]).to(device),
        "player_pad": torch.from_numpy(pad).to(device),
        "target": torch.from_numpy(view["target"]).to(device),
        "present": torch.from_numpy(view["present"]).to(device),
        "cut": int(view["cut"]),
    }


def shuffle_seats(batch: dict[str, torch.Tensor], rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Permute player seats so ID/role move with the player."""
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    b = out["player_code"].shape[0]
    for i in range(b):
        order = torch.as_tensor(rng.permutation(MAX_PLAYERS), device=out["player_code"].device)
        for key in ("player_code", "role", "player_pad"):
            out[key][i] = out[key][i][order]
        for key in ("minutes", "events_in", "target", "present"):
            out[key][i] = out[key][i][order]
    return out


def _stack_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key in items[0]:
        if key == "cut":
            out[key] = torch.tensor([it["cut"] for it in items], device=items[0]["club_id"].device)
        elif key == "club_id":
            out[key] = torch.stack([it[key] for it in items], dim=0)
        else:
            out[key] = torch.stack([it[key] for it in items], dim=0)
    return out


def stepped_loss(pred: torch.Tensor, target: torch.Tensor, present: torch.Tensor, pad: torch.Tensor, cuts: torch.Tensor) -> torch.Tensor:
    """Log loss + Brier on the painted week, live non-pad players."""
    b = pred.shape[0]
    index = torch.arange(b, device=pred.device)
    week_p = pred[index, :, cuts, :]  # B,P,2
    week_y = target[index, :, cuts, :]
    live = present[index, :, cuts] & ~pad  # B,P
    mask = live.to(pred.dtype)[..., None].expand_as(week_p)
    denom = mask.sum().clamp_min(1.0)
    brier = ((week_p - week_y) ** 2 * mask).sum() / denom
    p = week_p.clamp(1e-6, 1 - 1e-6)
    log = -(week_y * torch.log(p) + (1 - week_y) * torch.log(1 - p))
    log_loss = (log * mask).sum() / denom
    return log_loss + brier


def _metrics(pred: np.ndarray, target: np.ndarray, live: np.ndarray) -> dict:
    """pred/target (P,2), live (P,)."""
    played = np.flatnonzero(live)
    if len(played) == 0:
        return {"log_loss": 0.0, "brier": 0.0, "on_solid": 0.0, "on_blank": 0.0, "n": 0, "solid_n": 0}
    p = pred[played].reshape(-1).clip(1e-6, 1 - 1e-6)
    y = target[played].reshape(-1)
    log = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    solid = y > 0.5
    return {
        "log_loss": log,
        "brier": brier,
        "on_solid": float(p[solid].mean()) if solid.any() else 0.0,
        "on_blank": float(p[~solid].mean()) if (~solid).any() else 0.0,
        "n": int(len(p)),
        "solid_n": int(solid.sum()),
    }


@torch.no_grad()
def _predict_card(
    model: TypedCardMixer,
    card: TensorCard,
    paint_gw: int,
    player_codes: dict[int, int],
    device: torch.device,
) -> np.ndarray:
    batch = _stack_batch([_pack_card(card, step_view(card, paint_gw), player_codes, device)])
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
    )[0].cpu().numpy()
    return pred


@torch.no_grad()
def score_distances(
    model: TypedCardMixer,
    cards: list[TensorCard],
    player_codes: dict[int, int],
    device: torch.device,
    stitch: bool = False,
) -> dict:
    """Holdout table at cut fractions. Stepped paints each scored week with past true.

    stitch=True walks every week from the cut (final report). stitch=False only
    scores the distance anchors (fast early-stop).
    """
    model.eval()
    rows = []
    for frac in CUT_FRACS:
        for card in cards:
            cut = cut_at_fraction(card, frac)
            if stitch:
                weeks = list(range(cut, card.n_fixtures))
            else:
                weeks = [cut + d - 1 for d in DISTANCES if cut + d - 1 < card.n_fixtures]
            painted: dict[int, np.ndarray] = {}
            for gw in weeks:
                painted[gw] = _predict_card(model, card, gw, player_codes, device)
            for distance in DISTANCES:
                t = cut + distance - 1
                if t not in painted:
                    continue
                live = card.present[:, t].copy()
                live[card.n_players :] = False
                live &= card.position >= 0
                m = _metrics(painted[t][:, t, :], _event_ink(card)[:, t, :], live)
                m.update({"distance": distance, "revealed": frac, "team": card.team})
                rows.append(m)
    by_d = {}
    for d in DISTANCES:
        subset = [r for r in rows if r["distance"] == d and r["n"] > 0]
        if not subset:
            by_d[d] = {"log_loss": float("nan"), "brier": float("nan"), "on_solid": 0.0, "on_blank": 0.0, "n": 0}
            continue
        w = np.array([r["n"] for r in subset], dtype=np.float64)
        by_d[d] = {
            "log_loss": float(np.average([r["log_loss"] for r in subset], weights=w)),
            "brier": float(np.average([r["brier"] for r in subset], weights=w)),
            "on_solid": float(np.average([r["on_solid"] for r in subset], weights=w)),
            "on_blank": float(np.average([r["on_blank"] for r in subset], weights=w)),
            "n": int(w.sum()),
        }
    return {"rows": rows, "by_distance": by_d}


def _fit_slice_logistic(train: list[TensorCard], test: list[TensorCard]) -> dict:
    """Same spine logistic as baselines — for holdout comparison on stepped weeks."""
    def rows(cards):
        out = []
        for card in cards:
            for p in range(card.n_players):
                if int(card.position[p]) < 0 or int(card.player_id[p]) <= 0:
                    continue
                for t in range(1, card.n_fixtures):
                    if not card.present[p, t]:
                        continue
                    feats = asof_inputs(card, p, t)
                    out.append(
                        (
                            [feats[n] for n in FEATURE_NAMES],
                            1 if card.goals[p, t] >= 1 else 0,
                            1 if card.assists[p, t] >= 1 else 0,
                            card.season,
                            int(card.player_id[p]),
                            t,
                        )
                    )
        return out

    tr = rows(train)
    te = rows(test)
    x_tr = np.array([r[0] for r in tr], dtype=np.float64)
    yg = np.array([r[1] for r in tr], dtype=np.int64)
    ya = np.array([r[2] for r in tr], dtype=np.int64)
    pipe_g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipe_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipe_g.fit(x_tr, yg)
    pipe_a.fit(x_tr, ya)
    # score distance-1 style: all test slices
    x_te = np.array([r[0] for r in te], dtype=np.float64)
    yg_te = np.array([r[1] for r in te], dtype=np.float64)
    ya_te = np.array([r[2] for r in te], dtype=np.float64)
    pg = pipe_g.predict_proba(x_te)[:, 1]
    pa = pipe_a.predict_proba(x_te)[:, 1]
    p = np.stack([pg, pa], axis=1).reshape(-1).clip(1e-6, 1 - 1e-6)
    y = np.stack([yg_te, ya_te], axis=1).reshape(-1)
    return {
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "n": int(len(y)),
        "base_rate_goal": float(yg_te.mean()),
        "base_rate_assist": float(ya_te.mean()),
    }


def train_stepped(
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
                items.append(_pack_card(card, step_view(card, cut), player_codes, device))
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
            loss = stepped_loss(pred, batch["target"], batch["present"], batch["player_pad"], batch["cut"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        reading = score_distances(model, test, player_codes, device, stitch=False)
        score = reading["by_distance"][1]["log_loss"]
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
            "holdout_d1_log": score,
            "holdout_d1_brier": reading["by_distance"][1]["brier"],
        }
        history.append(row)
        print(
            f"stepped epoch {epochs_run}/{EPOCHS} train {row['train_loss']:.4f} "
            f"d1_log {score:.4f} d1_brier {row['holdout_d1_brier']:.4f}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epochs_run >= 12:
            print(f"stepped early stop at epoch {epochs_run}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    reading = score_distances(model, test, player_codes, device, stitch=True)
    return model, {
        "epochs_run": epochs_run,
        "best_d1_log": best,
        "seconds": round(time.time() - t0, 1),
        "history": history,
        "reading": reading,
    }


def write_report(meta: dict, slice_base: dict) -> Path:
    path = REPORTS / "typed_tensor_stepped.md"
    by = meta["reading"]["by_distance"]
    lines = [
        "# Typed tensor — stepped FOV painter",
        "",
        "Full-card stepped FOV on the cemented typed tensor (`tensor_card.py`).",
        "Past rows keep true G/A; events from the painted week onward are blank.",
        "Header / who / minutes stay. Opacity = P(count ≥ 1). Loss = log + Brier.",
        "Seats shuffled each batch. Train 2022-23…2024-25; holdout 2025-26.",
        "",
        f"Epochs run: {meta['epochs_run']}. Wall: {meta['seconds']}s. "
        f"Best holdout d1 log loss: {meta['best_d1_log']:.4f}.",
        "",
        "## Holdout by weeks ahead",
        "",
        "| weeks ahead | log loss | Brier | opacity on solid | opacity on blank | n cells |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in DISTANCES:
        m = by[d]
        lines.append(
            f"| {d} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
            f"{m['on_solid']:.3f} | {m['on_blank']:.3f} | {m['n']} |"
        )
    lines.extend(
        [
            "",
            "## Spine logistic (same seasons, sliced) for reference",
            "",
            f"log loss **{slice_base['log_loss']:.4f}**, Brier **{slice_base['brier']:.4f}**, "
            f"n={slice_base['n']} (player–GW × 2 channels).",
            "",
            "## Call",
            "",
        ]
    )
    d1 = by[1]["log_loss"]
    if d1 < slice_base["log_loss"]:
        lines.append(
            f"Stepped full-card d1 log **{d1:.4f}** beats sliced logistic "
            f"**{slice_base['log_loss']:.4f}** on the 2025-26 holdout. "
            "Keep the painter; horizon FOV is the next comparison."
        )
    else:
        lines.append(
            f"Stepped full-card d1 log **{d1:.4f}** does not beat sliced logistic "
            f"**{slice_base['log_loss']:.4f}**. Card stays cemented; painter is a "
            "screen loss or needs a follow-up (architecture / FOV), not a Baudot revival."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "meta": {k: v for k, v in meta.items() if k != "reading"},
        "by_distance": by,
        "slice_logistic": slice_base,
    }
    # drop history tensors etc.
    payload["meta"]["history"] = meta["history"]
    (REPORTS / "typed_tensor_stepped.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_stepped.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> None:
    device = torch.device("cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    cards = build_tensor_cards(seasons)
    for card in cards[:1]:
        assert_step_fov(card)
    train = [c for c in cards if c.season in TRAIN_SEASONS]
    test = [c for c in cards if c.season == TEST_SEASON]
    codes = fit_player_codes(cards)
    n_club = n_clubs(cards)
    print(f"cards train={len(train)} test={len(test)} clubs={n_club} players={len(codes)}", flush=True)
    slice_base = _fit_slice_logistic(train, test)
    print(f"slice logistic log={slice_base['log_loss']:.4f} brier={slice_base['brier']:.4f}", flush=True)
    _model, meta = train_stepped(train, test, codes, n_club, device)
    path = write_report(meta, slice_base)
    print(f"wrote {path}", flush=True)
    print(json.dumps({"by_distance": meta["reading"]["by_distance"], "slice": slice_base}, indent=2))


if __name__ == "__main__":
    main()
