"""Fixed horizon/stepped mixer vs rich zero-gated spine (4-season LOSO).

Pre-registered before totals were read:
  reports/typed_tensor_horizon_fix.md

Control: rich zero-gated SliceLogistic.
Candidate: TypedCardMixer with Co-PI fixes —
  - season-scoped player codes (no cross-season element collision)
  - full card.present as time-attention mask
  - structural m=0 zero-gate on paint output
  - pure binary log-loss (no Brier mix)
  - early-stop on train-internal validation (never holdout)

Not a Numagent comparison. No Odds API.
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
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.gates import CLOSED_SEASONS, cluster_interval, run_asof_audit
from src.models.tensor_card import (
    ART,
    MAX_GW,
    MAX_PLAYERS,
    REPORTS,
    TensorCard,
    build_tensor_cards,
    iter_slices,
)
from src.models.tensor_card_rich_baseline import (
    CANDIDATE_FEATURES,
    MATERIALITY,
    _fixture_over25,
    load_over25_table,
    rich_extras,
)
from src.models.tensor_card_stepped import (
    BATCH,
    D,
    EARLY_PATIENCE,
    EVENT_CH,
    HEADS,
    LR,
    SEED,
    n_clubs,
    step_view,
)

EPOCHS = 30
VAL_FRAC = 0.2
P_FLOOR = 1e-6
N_BOOT = 1000
CONTRAST = "horizon_fix_minus_rich"
CONTROL_FEATURES = CANDIDATE_FEATURES


def fit_player_codes_season(cards: list[TensorCard]) -> dict[tuple[str, int], int]:
    """Unique embedding id per (season, FPL element) — no cross-season collision."""
    keys = sorted(
        {
            (card.season, int(pid))
            for card in cards
            for pid in card.player_id.tolist()
            if int(pid) > 0
        }
    )
    return {key: i + 1 for i, key in enumerate(keys)}


class FixedTypedCardMixer(nn.Module):
    """Time + squad attention with present mask + minutes zero-gate."""

    def __init__(self, n_club: int, n_player: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.club = nn.Embedding(n_club + 1, D, padding_idx=0)
        self.opp = nn.Embedding(n_club + 1, D, padding_idx=0)
        self.player = nn.Embedding(n_player + 1, D, padding_idx=0)
        self.role = nn.Embedding(5, D, padding_idx=4)
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
        club_id: torch.Tensor,
        opp_id: torch.Tensor,
        home: torch.Tensor,
        odds: torch.Tensor,
        player_code: torch.Tensor,
        role: torch.Tensor,
        minutes: torch.Tensor,
        events_in: torch.Tensor,
        row_on: torch.Tensor,
        player_pad: torch.Tensor,
        cell_present: torch.Tensor,
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
        mins_e = self.mins((minutes / 90.0).clamp(0, 2)[..., None].permute(0, 2, 1, 3))
        ev = self.events(events_in.permute(0, 2, 1, 3))
        players = self.norm_p(self.player_mix(torch.cat([who, role_e, mins_e, ev], dim=-1)))
        tokens = players + header[:, :, None, :] + self.time_pos[:, :t]

        time_view = tokens.permute(0, 2, 1, 3).reshape(b * p, t, D)
        # Mask padded weeks OR player-weeks with no sheet row.
        week_off = (row_on < 0.5)[:, None, :].expand(b, p, t)
        absent = ~cell_present
        time_mask = (week_off | absent).reshape(b * p, t)
        # MHA forbids all-True key masks; force first week visible for pad seats.
        all_masked = time_mask.all(dim=1)
        if bool(all_masked.any()):
            time_mask = time_mask.clone()
            time_mask[all_masked, 0] = False
        mixed, _ = self.time(
            time_view, time_view, time_view, key_padding_mask=time_mask, need_weights=False
        )
        tokens = self.norm_t(time_view + mixed).reshape(b, p, t, D).permute(0, 2, 1, 3)

        squad_view = tokens.reshape(b * t, p, D)
        squad_mask = player_pad[:, None, :].expand(b, t, p).reshape(b * t, p)
        mixed, _ = self.squad(
            squad_view, squad_view, squad_view, key_padding_mask=squad_mask, need_weights=False
        )
        tokens = self.norm_s(squad_view + mixed).reshape(b, t, p, D)
        logits = self.paint(tokens).permute(0, 2, 1, 3)  # B,P,T,2
        ink = torch.sigmoid(logits)
        active = (minutes > 0) & cell_present
        floor = torch.full_like(ink, P_FLOOR)
        return torch.where(active[..., None], ink, floor)


def _pack_card(
    card: TensorCard,
    view: dict,
    player_codes: dict[tuple[str, int], int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    codes = np.zeros(MAX_PLAYERS, dtype=np.int64)
    roles = np.full(MAX_PLAYERS, 4, dtype=np.int64)
    pad = np.ones(MAX_PLAYERS, dtype=bool)
    for p in range(card.n_players):
        pid = int(card.player_id[p])
        if pid <= 0:
            continue
        codes[p] = player_codes.get((card.season, pid), 0)
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
        "cell_present": torch.from_numpy(card.present.astype(bool)).to(device),
        "cut": int(view["cut"]),
    }


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


def _shuffle_seats(batch: dict[str, torch.Tensor], rng: np.random.Generator) -> dict[str, torch.Tensor]:
    out = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    b = out["player_code"].shape[0]
    for i in range(b):
        order = torch.as_tensor(rng.permutation(MAX_PLAYERS), device=out["player_code"].device)
        for key in ("player_code", "role", "player_pad"):
            out[key][i] = out[key][i][order]
        for key in ("minutes", "events_in", "target", "present", "cell_present"):
            out[key][i] = out[key][i][order]
    return out


def paint_log_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    present: torch.Tensor,
    pad: torch.Tensor,
    cuts: torch.Tensor,
) -> torch.Tensor:
    """Pure log-loss on the painted week (live non-pad seats)."""
    b = pred.shape[0]
    index = torch.arange(b, device=pred.device)
    week_p = pred[index, :, cuts, :]
    week_y = target[index, :, cuts, :]
    live = present[index, :, cuts] & ~pad
    mask = live.to(pred.dtype)[..., None].expand_as(week_p)
    denom = mask.sum().clamp_min(1.0)
    p = week_p.clamp(P_FLOOR, 1.0 - P_FLOOR)
    log = -(week_y * torch.log(p) + (1.0 - week_y) * torch.log(1.0 - p))
    return (log * mask).sum() / denom


@torch.no_grad()
def _val_log_loss(
    model: FixedTypedCardMixer,
    cards: list[TensorCard],
    player_codes: dict[tuple[str, int], int],
    device: torch.device,
    rng: np.random.Generator,
    n_cuts: int = 2,
) -> float:
    model.eval()
    losses = []
    for card in cards:
        for _ in range(n_cuts):
            cut = int(rng.integers(1, card.n_fixtures))
            batch = _stack_batch([_pack_card(card, step_view(card, cut), player_codes, device)])
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
                batch["cell_present"],
            )
            losses.append(
                float(paint_log_loss(pred, batch["target"], batch["present"], batch["player_pad"], batch["cut"]))
            )
    return float(np.mean(losses)) if losses else float("inf")


def train_fixed_mixer(
    train_cards: list[TensorCard],
    player_codes: dict[tuple[str, int], int],
    n_club: int,
    device: torch.device,
    seed: int = SEED,
) -> tuple[FixedTypedCardMixer, dict]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_cards))
    n_val = max(1, int(round(VAL_FRAC * len(train_cards))))
    val_idx = set(int(i) for i in order[:n_val])
    fit = [train_cards[i] for i in range(len(train_cards)) if i not in val_idx]
    val = [train_cards[i] for i in range(len(train_cards)) if i in val_idx]
    if not fit:
        fit, val = train_cards, train_cards[:1]

    model = FixedTypedCardMixer(n_club=n_club, n_player=max(player_codes.values())).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best = float("inf")
    bad = 0
    t0 = time.time()
    history = []
    epochs_run = 0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        perm = rng.permutation(len(fit))
        for start in range(0, len(fit), BATCH):
            pick = [fit[int(i)] for i in perm[start : start + BATCH]]
            items = []
            for card in pick:
                cut = int(rng.integers(1, card.n_fixtures))
                items.append(_pack_card(card, step_view(card, cut), player_codes, device))
            batch = _shuffle_seats(_stack_batch(items), rng)
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
                batch["cell_present"],
            )
            loss = paint_log_loss(
                pred, batch["target"], batch["present"], batch["player_pad"], batch["cut"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()
        val_loss = _val_log_loss(model, val, player_codes, device, rng)
        improved = val_loss < best - 1e-5
        if improved:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        history.append(
            {
                "epoch": epochs_run,
                "train_loss": total / max(seen, 1),
                "val_loss": val_loss,
            }
        )
        if bad >= EARLY_PATIENCE and epochs_run >= 10:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "epochs_run": epochs_run,
        "best_val_log": best,
        "seconds": round(time.time() - t0, 1),
        "n_fit": len(fit),
        "n_val": len(val),
        "history": history,
    }


@torch.no_grad()
def predict_week(
    model: FixedTypedCardMixer,
    card: TensorCard,
    t: int,
    player_codes: dict[tuple[str, int], int],
    device: torch.device,
) -> np.ndarray:
    """Return (P, 2) opacities for week t with past true G/A."""
    model.eval()
    batch = _stack_batch([_pack_card(card, step_view(card, t), player_codes, device)])
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
        batch["cell_present"],
    )[0].cpu().numpy()
    return pred[:, t, :]


def _collect_rich(cards: list[TensorCard], over_table: dict) -> list[dict]:
    rows: list[dict] = []
    for card in cards:
        for row in iter_slices(
            card,
            min_history=1,
            minutes_mode="actual",
            include_mates=False,
            history_mode="present",
        ):
            pids = card.player_id[: card.n_players]
            matches = np.flatnonzero(pids == int(row["player_id"]))
            if len(matches) != 1:
                continue
            p = int(matches[0])
            t = int(row["t"])
            over = _fixture_over25(card, t, over_table)
            feats = dict(row["features"])
            feats.update(rich_extras(card, p, t, over))
            rows.append(
                {
                    **row,
                    "features": feats,
                    "card_season": card.season,
                    "card_team": card.team,
                    "seat": p,
                }
            )
    return rows


def _feat_matrix(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    return np.array([[r["features"][name] for name in names] for r in rows], dtype=np.float64)


def _minutes(rows: list[dict]) -> np.ndarray:
    return np.array([r["features"]["minutes"] for r in rows], dtype=np.float64)


def _binary(rows: list[dict], channel: str) -> np.ndarray:
    key = "y_goal" if channel == "goal" else "y_assist"
    return np.array([r[key] for r in rows], dtype=np.int64)


def _clip_p(p: np.ndarray) -> np.ndarray:
    return np.clip(p, P_FLOOR, 1.0 - P_FLOOR)


def _row_nll(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = _clip_p(p)
    y = y.astype(np.float64)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _fit_logistic(x_tr: np.ndarray, y_tr: np.ndarray) -> object:
    if len(y_tr) == 0 or y_tr.min() == y_tr.max():
        rate = float(y_tr.mean()) if len(y_tr) else P_FLOOR

        class _Const:
            def predict_proba_pos(self, x: np.ndarray) -> np.ndarray:
                return np.full(len(x), rate, dtype=np.float64)

        return _Const()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipe.fit(x_tr, y_tr)

    class _Wrap:
        def __init__(self, model: object) -> None:
            self.model = model

        def predict_proba_pos(self, x: np.ndarray) -> np.ndarray:
            return self.model.predict_proba(x)[:, 1]

    return _Wrap(pipe)


def _predict_gated(model: object, x: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    out = np.full(len(minutes), P_FLOOR, dtype=np.float64)
    active = minutes > 0
    if np.any(active):
        out[active] = _clip_p(model.predict_proba_pos(x[active]))
    return out


def _score_binary(y: np.ndarray, p: np.ndarray) -> dict:
    p = _clip_p(p)
    if len(y) == 0:
        return {
            "n": 0,
            "log_loss": float("nan"),
            "brier": float("nan"),
            "cal_slope": float("nan"),
        }
    if y.min() == y.max() or float(np.std(p)) < 1e-12:
        slope = float("nan")
    else:
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        cal = LogisticRegression(max_iter=500, solver="lbfgs")
        cal.fit(logit, y)
        slope = float(cal.coef_[0, 0])
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "cal_slope": slope,
    }


def _subset_metrics(pred: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    metrics = {}
    for arm, prefix in (("control", "p_ctrl"), ("candidate", "p_cand")):
        channel_scores = {}
        ps, ys = [], []
        for channel in ("goal", "assist"):
            y = pred[f"y_{channel}"][mask]
            p = pred[f"{prefix}_{channel}"][mask]
            channel_scores[channel] = _score_binary(y, p)
            ps.append(p)
            ys.append(y)
        channel_scores["pooled"] = _score_binary(np.concatenate(ys), np.concatenate(ps))
        metrics[arm] = channel_scores
    return metrics


def _gw_deltas(test_rows: list[dict], pred: dict[str, np.ndarray]) -> np.ndarray:
    gws = np.array([int(r["gw"]) for r in test_rows], dtype=np.int64)
    deltas = []
    for gw in sorted(set(gws.tolist())):
        m = gws == gw
        nll_cand = np.concatenate(
            [_row_nll(pred[f"y_{ch}"][m], pred[f"p_cand_{ch}"][m]) for ch in ("goal", "assist")]
        )
        nll_ctrl = np.concatenate(
            [_row_nll(pred[f"y_{ch}"][m], pred[f"p_ctrl_{ch}"][m]) for ch in ("goal", "assist")]
        )
        deltas.append(float(nll_cand.mean() - nll_ctrl.mean()))
    return np.asarray(deltas, dtype=float)


def _predict_fold(
    train_rows: list[dict],
    test_rows: list[dict],
    test_cards: list[TensorCard],
    model: FixedTypedCardMixer,
    player_codes: dict[tuple[str, int], int],
    device: torch.device,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_rows))
    train_rows = [train_rows[i] for i in order]
    minutes_tr = _minutes(train_rows)
    minutes_te = _minutes(test_rows)
    active_tr = minutes_tr > 0
    x_ctrl_tr = _feat_matrix(train_rows, CONTROL_FEATURES)
    x_ctrl_te = _feat_matrix(test_rows, CONTROL_FEATURES)

    # Cache mixer week paints per (season, team, t)
    paint_cache: dict[tuple[str, str, int], np.ndarray] = {}
    card_index = {(c.season, c.team): c for c in test_cards}

    p_goal = np.full(len(test_rows), P_FLOOR, dtype=np.float64)
    p_assist = np.full(len(test_rows), P_FLOOR, dtype=np.float64)
    for i, row in enumerate(test_rows):
        key = (row["card_season"], row["card_team"], int(row["t"]))
        if key not in paint_cache:
            card = card_index[(row["card_season"], row["card_team"])]
            paint_cache[key] = predict_week(model, card, int(row["t"]), player_codes, device)
        soft = paint_cache[key]
        seat = int(row["seat"])
        p_goal[i] = float(soft[seat, 0])
        p_assist[i] = float(soft[seat, 1])

    out: dict[str, np.ndarray] = {
        "minutes": minutes_te,
        "p_cand_goal": _clip_p(p_goal),
        "p_cand_assist": _clip_p(p_assist),
    }
    for channel in ("goal", "assist"):
        y_tr = _binary(train_rows, channel)
        y_te = _binary(test_rows, channel).astype(np.float64)
        ctrl = _fit_logistic(x_ctrl_tr[active_tr], y_tr[active_tr])
        out[f"p_ctrl_{channel}"] = _predict_gated(ctrl, x_ctrl_te, minutes_te)
        out[f"y_{channel}"] = y_te
    return out


def run(seed: int = SEED, device: torch.device | None = None) -> dict:
    device = device or torch.device("cpu")
    over_table = load_over25_table(CLOSED_SEASONS)
    cards = build_tensor_cards(CLOSED_SEASONS)
    by_season: dict[str, list] = {s: [] for s in CLOSED_SEASONS}
    for card in cards:
        by_season[card.season].append(card)

    deltas_by_season: dict[str, np.ndarray] = {}
    deltas_active_by_season: dict[str, np.ndarray] = {}
    fold_rows: list[dict] = []
    n_pred = 0
    train_meta: list[dict] = []

    for holdout in CLOSED_SEASONS:
        train_cards = [c for s, cs in by_season.items() if s != holdout for c in cs]
        test_cards = by_season[holdout]
        codes = fit_player_codes_season(train_cards + test_cards)
        n_club = n_clubs(train_cards + test_cards)
        model, meta = train_fixed_mixer(train_cards, codes, n_club, device, seed=seed)
        train_meta.append({"holdout": holdout, **{k: meta[k] for k in ("epochs_run", "best_val_log", "seconds", "n_fit", "n_val")}})
        print(
            f"holdout {holdout}: epochs={meta['epochs_run']} val={meta['best_val_log']:.4f} "
            f"({meta['seconds']}s)",
            flush=True,
        )

        train_rows = _collect_rich(train_cards, over_table)
        test_rows = _collect_rich(test_cards, over_table)
        pred = _predict_fold(train_rows, test_rows, test_cards, model, codes, device, seed)
        deltas_by_season[holdout] = _gw_deltas(test_rows, pred)

        minutes = pred["minutes"]
        active_idx = np.flatnonzero(minutes > 0)
        if active_idx.size:
            active_rows = [test_rows[i] for i in active_idx]
            active_pred = {
                "minutes": minutes[active_idx],
                **{
                    k: v[active_idx]
                    for k, v in pred.items()
                    if k != "minutes" and isinstance(v, np.ndarray)
                },
            }
            deltas_active_by_season[holdout] = _gw_deltas(active_rows, active_pred)
        else:
            deltas_active_by_season[holdout] = np.asarray([0.0], dtype=float)

        all_mask = np.ones(len(test_rows), dtype=bool)
        active_mask = minutes > 0
        fold_rows.append(
            {
                "holdout": holdout,
                "n_train": len(train_rows),
                "n_test": len(test_rows),
                "n_active": int(active_mask.sum()),
                "all": _subset_metrics(pred, all_mask),
                "active": _subset_metrics(pred, active_mask),
                "mean_gw_delta": float(deltas_by_season[holdout].mean()),
                "mean_gw_delta_active": float(deltas_active_by_season[holdout].mean()),
            }
        )
        n_pred += len(test_rows)

    interval = cluster_interval(deltas_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed)
    interval_active = cluster_interval(
        deltas_active_by_season, seasons=CLOSED_SEASONS, n_boot=N_BOOT, seed=seed
    )
    clears = float(interval["hi"]) < 0.0 and float(interval_active["hi"]) < 0.0
    material = float(interval["mean"]) <= MATERIALITY and float(interval_active["mean"]) <= MATERIALITY
    folds_neg = all(f["mean_gw_delta"] < 0.0 for f in fold_rows)
    if clears and material:
        decision = "promote_horizon_fix"
        promote = True
    elif clears and (folds_neg or float(interval["mean"]) < 0.0):
        decision = "soft_promote_horizon_fix"
        promote = False
    elif float(interval["lo"]) > 0.0:
        decision = "keep_rich_baseline"
        promote = False
    else:
        decision = "inconclusive_keep_rich_baseline"
        promote = False

    audit = run_asof_audit()
    return {
        "task": "typed_tensor_horizon_fix",
        "contrast": CONTRAST,
        "decision": decision,
        "promote": promote,
        "materiality": MATERIALITY,
        "clears_zero": clears,
        "meets_materiality": material,
        "n_pred_rows": n_pred,
        "folds": fold_rows,
        "train_meta": train_meta,
        "interval": interval,
        "interval_active_minutes": interval_active,
        "asof_audit_passed": bool(audit.get("passed")),
        "asof_audit_failures": list(audit.get("failures") or []),
        "seed": seed,
        "n_boot": N_BOOT,
        "epochs": EPOCHS,
        "val_frac": VAL_FRAC,
    }


def _fmt(v: float, nd: int = 5) -> str:
    if v != v:
        return "—"
    return f"{v:+.{nd}f}" if nd >= 4 else f"{v:.{nd}f}"


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_horizon_fix.md"
    iv = results["interval"]
    iva = results["interval_active_minutes"]
    decision = results["decision"]
    lines = [
        "# Typed tensor — horizon painter fixes vs rich spine",
        "",
        "**Pre-registered before totals were read.** Not a Numagent / `score_xp_v2` comparison.",
        "Product on main remains Numagent 2.0.",
        "",
        "Co-PI follow-up after confounded market/role× screen: fix the typed-card mixer",
        "defects, then judge on the same player–GW likelihood protocol as the rich",
        "zero-gated spine.",
        "",
        "## Comparison",
        "",
        "| Arm | Spec |",
        "| --- | --- |",
        "| **Control** | Rich zero-gated SliceLogistic (mates off; present-row history). |",
        (
            "| **Candidate** | Fixed `TypedCardMixer`: season-scoped player codes, full `present` "
            "time-mask, structural `m=0 → p≈1e-6` zero-gate, **pure** binary log-loss, "
            "early-stop on train-internal val (never holdout). Paint one week with past true G/A. |"
        ),
        "",
        "## Protocol",
        "",
        "- LOSO on 2022-23…2025-26. `2026-27` untouched.",
        "- Same `iter_slices` player–GW cells as rich baseline (`min_history=1`, actual minutes).",
        f"- Primary: `Δ = {CONTRAST}` via `cluster_interval` "
        f"(n_boot={results['n_boot']}, seed={results['seed']}).",
        f"- Promote if hi &lt; 0 on all and active **and** mean Δ ≤ {results['materiality']}.",
        "- Soft-promote if hi &lt; 0 on both but mean above materiality.",
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
        (
            f"Interval Δ log-loss (horizon_fix − rich): mean {_fmt(iv['mean'])}, "
            f"95% [{_fmt(iv['lo'])}, {_fmt(iv['hi'])}]."
        ),
        (
            f"Active minutes subset: mean {_fmt(iva['mean'])}, "
            f"95% [{_fmt(iva['lo'])}, {_fmt(iva['hi'])}]."
        ),
        "",
        f"Clears zero={results['clears_zero']}. Meets materiality={results['meets_materiality']}.",
        "",
        "Gameweeks per season: "
        + ", ".join(f"{s}={iv['n_gws'].get(s, 0)}" for s in CLOSED_SEASONS)
        + ".",
        "",
        f"Product as-of audit passed={results['asof_audit_passed']}"
        + (
            f" (failures: {results['asof_audit_failures']})"
            if results["asof_audit_failures"]
            else "."
        ),
        "",
        "## Train meta (per LOSO fold)",
        "",
        "| holdout | epochs | best val log | seconds | n_fit | n_val |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["train_meta"]:
        lines.append(
            f"| {row['holdout']} | {row['epochs_run']} | {row['best_val_log']:.4f} | "
            f"{row['seconds']} | {row['n_fit']} | {row['n_val']} |"
        )
    lines += [
        "",
        "## Fold holdouts (pooled log loss)",
        "",
        "| holdout | n_test | n_active | rich LL | fixed-mixer LL | Δ mean GW | active Δ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in results["folds"]:
        c = fold["all"]["control"]["pooled"]["log_loss"]
        k = fold["all"]["candidate"]["pooled"]["log_loss"]
        lines.append(
            f"| {fold['holdout']} | {fold['n_test']} | {fold['n_active']} | "
            f"{c:.4f} | {k:.4f} | {fold['mean_gw_delta']:+.5f} | "
            f"{fold['mean_gw_delta_active']:+.5f} |"
        )

    lines += [
        "",
        "## Per-channel (n-weighted across LOSO folds)",
        "",
        "| arm | channel | scope | log loss | Brier | cal slope | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for scope in ("all", "active"):
        for arm in ("control", "candidate"):
            for channel in ("goal", "assist", "pooled"):
                ns, lls, brs, cals = [], [], [], []
                for fold in results["folds"]:
                    m = fold[scope][arm][channel]
                    if m["n"] <= 0 or m["log_loss"] != m["log_loss"]:
                        continue
                    ns.append(m["n"])
                    lls.append(m["log_loss"])
                    brs.append(m["brier"])
                    cals.append(m["cal_slope"])
                if not ns:
                    continue
                w = np.asarray(ns, dtype=float)
                cal = (
                    float(np.average(cals, weights=w))
                    if np.isfinite(cals).all()
                    else float("nan")
                )
                lines.append(
                    f"| {arm} | {channel} | {scope} | "
                    f"{float(np.average(lls, weights=w)):.4f} | "
                    f"{float(np.average(brs, weights=w)):.4f} | "
                    f"{_fmt(cal, 3)} | {int(w.sum())} |"
                )

    lines += ["", "## Call", ""]
    if decision == "promote_horizon_fix":
        lines.append(
            "**promote_horizon_fix.** Fixed mixer clears zero and materiality vs rich "
            "spine. Adopt fixed mixer as FOV/slice candidate path."
        )
    elif decision == "soft_promote_horizon_fix":
        lines.append(
            "**soft_promote_horizon_fix.** Clears zero but misses materiality. Prefer "
            "fixed mixer for further FOV work; keep rich logistic as slice default."
        )
    elif decision == "keep_rich_baseline":
        lines.append(
            "**keep_rich_baseline.** Fixed mixer loses. Keep rich zero-gated spine; "
            "horizon FOV arm stays exam-only."
        )
    else:
        lines.append(
            "**inconclusive_keep_rich_baseline.** Interval covers zero. Keep rich "
            "zero-gated spine; fixed mixer does not earn promotion."
        )
    lines += [
        "",
        "Reproduce: `PYTHONPATH=. python3 -m src.models.tensor_card_horizon_fix`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

    def _clean(obj):
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return _clean(obj.tolist())
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    payload = _clean(results)
    (REPORTS / "typed_tensor_horizon_fix.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_horizon_fix.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return path


def main() -> None:
    results = run()
    path = write_report(results)
    print(
        json.dumps(
            {
                "decision": results["decision"],
                "interval": results["interval"],
                "interval_active_minutes": results["interval_active_minutes"],
                "promote": results["promote"],
                "clears_zero": results["clears_zero"],
                "meets_materiality": results["meets_materiality"],
                "train_meta": results["train_meta"],
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
