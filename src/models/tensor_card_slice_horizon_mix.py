"""Mix sliced logistic spine with horizon FOV card.

Arms (pre-registered):
  1. blend — p = α p_horizon + (1-α) p_slice; α chosen on train remainders
  2. stack — logistic on [logit_h, logit_s] fit on train remainders
  3. residual — p = σ(logit(p_slice) + δ_card); δ from horizon-FOV mixer

Holdout 2025-26 remainder log/Brier. Card stays cemented; not a product change.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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
    build_tensor_cards,
    cut_at_fraction,
)
from src.models.tensor_card_horizon import (
    assert_horizon_fov,
    horizon_loss,
    horizon_view,
    predict_horizon,
    train_horizon,
)
from src.models.tensor_card_stepped import (
    BATCH,
    CUT_FRACS,
    EARLY_PATIENCE,
    EPOCHS,
    LR,
    SEED,
    TypedCardMixer,
    _event_ink,
    _pack_card,
    _stack_batch,
    fit_player_codes,
    n_clubs,
)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _score_pairs(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    y = y.astype(np.float64)
    return {
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "n": int(len(y)),
        "on_solid": float(p[y > 0.5].mean()) if (y > 0.5).any() else 0.0,
        "on_blank": float(p[y <= 0.5].mean()) if (y <= 0.5).any() else 0.0,
    }


@dataclass
class SliceLogistic:
    pipe_g: object
    pipe_a: object

    @classmethod
    def fit(cls, cards: list[TensorCard]) -> "SliceLogistic":
        xs, yg, ya = [], [], []
        for card in cards:
            for p in range(card.n_players):
                if int(card.position[p]) < 0 or int(card.player_id[p]) <= 0:
                    continue
                for t in range(1, card.n_fixtures):
                    if not card.present[p, t]:
                        continue
                    feats = asof_inputs(card, p, t)
                    xs.append([feats[n] for n in FEATURE_NAMES])
                    yg.append(1 if card.goals[p, t] >= 1 else 0)
                    ya.append(1 if card.assists[p, t] >= 1 else 0)
        x = np.asarray(xs, dtype=np.float64)
        pipe_g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
        pipe_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
        pipe_g.fit(x, np.asarray(yg))
        pipe_a.fit(x, np.asarray(ya))
        return cls(pipe_g=pipe_g, pipe_a=pipe_a)

    def predict_card(self, card: TensorCard) -> np.ndarray:
        """Soft opacities (P, T, 2); 0 where no as-of row."""
        out = np.zeros((MAX_PLAYERS, MAX_GW, 2), dtype=np.float32)
        rows, index = [], []
        for p in range(card.n_players):
            if int(card.position[p]) < 0 or int(card.player_id[p]) <= 0:
                continue
            for t in range(1, card.n_fixtures):
                if not card.present[p, t]:
                    continue
                feats = asof_inputs(card, p, t)
                rows.append([feats[n] for n in FEATURE_NAMES])
                index.append((p, t))
        if not rows:
            return out
        x = np.asarray(rows, dtype=np.float64)
        pg = self.pipe_g.predict_proba(x)[:, 1]
        pa = self.pipe_a.predict_proba(x)[:, 1]
        for i, (p, t) in enumerate(index):
            out[p, t, 0] = pg[i]
            out[p, t, 1] = pa[i]
        return out

    def spine_metrics(self, cards: list[TensorCard]) -> dict:
        ps, ys = [], []
        for card in cards:
            soft = self.predict_card(card)
            truth = _event_ink(card)
            for p in range(card.n_players):
                if int(card.position[p]) < 0:
                    continue
                for t in range(1, card.n_fixtures):
                    if not card.present[p, t]:
                        continue
                    ps.append(soft[p, t])
                    ys.append(truth[p, t])
        p = np.stack(ps, axis=0).reshape(-1)
        y = np.stack(ys, axis=0).reshape(-1)
        return _score_pairs(p, y)


def collect_remainder_pairs(
    cards: list[TensorCard],
    slice_model: SliceLogistic,
    horizon_model: TypedCardMixer | None,
    player_codes: dict[int, int] | None,
    device: torch.device | None,
    soft_h_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened (p_h, p_s, y) on cut-fraction remainders."""
    ph, ps, yy = [], [], []
    for frac in CUT_FRACS:
        for card in cards:
            cut = cut_at_fraction(card, frac)
            soft_s = slice_model.predict_card(card)
            if soft_h_fn is not None:
                soft_h = soft_h_fn(card, cut)
            else:
                assert horizon_model is not None and player_codes is not None and device is not None
                soft_h = predict_horizon(horizon_model, card, cut, player_codes, device)
            truth = _event_ink(card)
            for p in range(card.n_players):
                if int(card.position[p]) < 0:
                    continue
                for t in range(cut, card.n_fixtures):
                    if not card.present[p, t]:
                        continue
                    # slice needs t>=1; if missing, fall back to base rate later via clip
                    ph.append(soft_h[p, t])
                    ps.append(soft_s[p, t])
                    yy.append(truth[p, t])
    return (
        np.stack(ph, axis=0).reshape(-1).astype(np.float64),
        np.stack(ps, axis=0).reshape(-1).astype(np.float64),
        np.stack(yy, axis=0).reshape(-1).astype(np.float64),
    )


def screen_blends(ph: np.ndarray, ps: np.ndarray, y: np.ndarray) -> dict:
    rows = []
    for a in ALPHAS:
        p = a * ph + (1.0 - a) * ps
        m = _score_pairs(p, y)
        m["alpha"] = a
        rows.append(m)
    best = min(rows, key=lambda r: r["log_loss"])
    return {"grid": rows, "best_alpha": best["alpha"], "best": best}


def fit_stack(ph_tr: np.ndarray, ps_tr: np.ndarray, y_tr: np.ndarray):
    x = np.stack([_logit(ph_tr), _logit(ps_tr)], axis=1)
    if y_tr.min() == y_tr.max():
        return None
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    pipe.fit(x, y_tr.astype(int))
    return pipe


def predict_stack(pipe, ph: np.ndarray, ps: np.ndarray) -> np.ndarray:
    x = np.stack([_logit(ph), _logit(ps)], axis=1)
    return pipe.predict_proba(x)[:, 1]


def train_residual(
    train: list[TensorCard],
    test: list[TensorCard],
    slice_model: SliceLogistic,
    player_codes: dict[int, int],
    n_club: int,
    device: torch.device,
) -> tuple[TypedCardMixer, dict]:
    """Horizon FOV mixer learns δ; p = σ(logit(p_slice) + δ)."""
    rng = np.random.default_rng(SEED)
    model = TypedCardMixer(n_club=n_club, n_player=max(player_codes.values())).to(device)
    # residual starts near 0 (paint bias was -3 for absolute; re-init near 0)
    nn_init = model.paint
    torch.nn.init.zeros_(nn_init.weight)
    torch.nn.init.zeros_(nn_init.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_state = None
    best = float("inf")
    bad = 0
    t0 = time.time()
    history = []
    epochs_run = 0

    def pack_with_slice(card: TensorCard, cut: int):
        view = horizon_view(card, cut)
        packed = _pack_card(card, view, player_codes, device)
        soft_s = slice_model.predict_card(card)
        packed["slice_p"] = torch.from_numpy(soft_s.astype(np.float32)).to(device)
        return packed

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        seen = 0
        order = rng.permutation(len(train))
        for start in range(0, len(train), BATCH):
            pick = [train[int(i)] for i in order[start : start + BATCH]]
            items = []
            slices = []
            for card in pick:
                cut = int(rng.integers(1, card.n_fixtures))
                packed = pack_with_slice(card, cut)
                slices.append(packed.pop("slice_p"))
                items.append(packed)
            batch = _stack_batch(items)
            slice_stack = torch.stack(slices, dim=0)
            out_slice = slice_stack.clone()
            batch_shuffled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            for i in range(len(pick)):
                order_p = torch.as_tensor(rng.permutation(MAX_PLAYERS), device=device)
                for key in ("player_code", "role", "player_pad"):
                    batch_shuffled[key][i] = batch[key][i][order_p]
                for key in ("minutes", "events_in", "target", "present"):
                    batch_shuffled[key][i] = batch[key][i][order_p]
                out_slice[i] = slice_stack[i][order_p]
            opt.zero_grad()
            delta = model.forward_logits(
                batch_shuffled["club_id"],
                batch_shuffled["opp_id"],
                batch_shuffled["home"],
                batch_shuffled["odds"],
                batch_shuffled["player_code"],
                batch_shuffled["role"],
                batch_shuffled["minutes"],
                batch_shuffled["events_in"],
                batch_shuffled["row_on"],
                batch_shuffled["player_pad"],
            )
            missing = out_slice.sum(dim=-1, keepdim=True) <= 0
            prior = torch.full_like(out_slice, 0.05)
            slice_p = torch.where(missing.expand_as(out_slice), prior, out_slice).clamp(1e-6, 1 - 1e-6)
            slice_logit = torch.log(slice_p / (1 - slice_p))
            pred = torch.sigmoid(slice_logit + delta)
            loss = horizon_loss(pred, batch_shuffled["target"], batch_shuffled["present"], batch_shuffled["player_pad"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
            seen += 1
        sched.step()

        def soft_fn(card, cut):
            return predict_residual(model, slice_model, card, cut, player_codes, device)

        ph, ps, y = collect_remainder_pairs(test, slice_model, None, None, None, soft_h_fn=soft_fn)
        # here ph is already mixed residual output
        score = _score_pairs(ph, y)["log_loss"]
        improved = score < best - 1e-5
        if improved:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        epochs_run = epoch + 1
        history.append({"epoch": epochs_run, "train_loss": total / max(seen, 1), "holdout_rem_log": score})
        print(
            f"residual epoch {epochs_run}/{EPOCHS} train {total / max(seen, 1):.4f} "
            f"rem_log {score:.4f}{' *' if improved else ''}",
            flush=True,
        )
        if bad >= EARLY_PATIENCE and epochs_run >= 12:
            print(f"residual early stop at epoch {epochs_run}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "epochs_run": epochs_run,
        "best_rem_log": best,
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }


@torch.no_grad()
def predict_residual(
    model: TypedCardMixer,
    slice_model: SliceLogistic,
    card: TensorCard,
    cut: int,
    player_codes: dict[int, int],
    device: torch.device,
) -> np.ndarray:
    batch = _stack_batch([_pack_card(card, horizon_view(card, cut), player_codes, device)])
    delta = model.forward_logits(
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
    soft_s = slice_model.predict_card(card)
    soft_s = np.where(soft_s <= 0, 0.05, soft_s)
    return _sigmoid(_logit(soft_s) + delta).astype(np.float32)


def write_report(results: dict) -> Path:
    path = REPORTS / "typed_tensor_slice_horizon_mix.md"
    lines = [
        "# Typed tensor — slice × horizon mix",
        "",
        "Mix the sliced logistic spine with the horizon FOV card on remainder cells.",
        "Train seasons 2022-23…2024-25; holdout 2025-26. Cut fractions 20/40/60/80%.",
        "",
        "## Remainder holdout",
        "",
        "| arm | log loss | Brier | on solid | on blank | n | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("slice", "horizon", "blend", "stack", "residual"):
        m = results[name]
        note = m.get("note", "")
        lines.append(
            f"| {name} | {m['log_loss']:.4f} | {m['brier']:.4f} | "
            f"{m.get('on_solid', float('nan')):.3f} | {m.get('on_blank', float('nan')):.3f} | "
            f"{m['n']} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Blend grid (holdout; α = weight on horizon)",
            "",
            "| α | log loss | Brier |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in results["blend_grid"]:
        lines.append(f"| {row['alpha']:.2f} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines.extend(["", "## Call", "", results["call"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "typed_tensor_slice_horizon_mix.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "typed_tensor_slice_horizon_mix.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


def main() -> None:
    device = torch.device("cpu")
    seasons = TRAIN_SEASONS + (TEST_SEASON,)
    cards = build_tensor_cards(seasons)
    assert_horizon_fov(cards[0])
    train = [c for c in cards if c.season in TRAIN_SEASONS]
    test = [c for c in cards if c.season == TEST_SEASON]
    codes = fit_player_codes(cards)
    n_club = n_clubs(cards)
    print(f"cards train={len(train)} test={len(test)}", flush=True)

    slice_model = SliceLogistic.fit(train)
    spine = slice_model.spine_metrics(test)
    print(f"slice spine log={spine['log_loss']:.4f}", flush=True)

    print("=== horizon ===", flush=True)
    horizon_model, horizon_meta = train_horizon(train, test, codes, n_club, device)
    ph_tr, ps_tr, y_tr = collect_remainder_pairs(train, slice_model, horizon_model, codes, device)
    ph_te, ps_te, y_te = collect_remainder_pairs(test, slice_model, horizon_model, codes, device)

    slice_rem = _score_pairs(ps_te, y_te)
    slice_rem["note"] = "slice on remainder cells"
    horizon_rem = _score_pairs(ph_te, y_te)
    horizon_rem["note"] = f"epochs={horizon_meta['epochs_run']}"

    # α on train remainders
    train_blend = screen_blends(ph_tr, ps_tr, y_tr)
    a_star = train_blend["best_alpha"]
    hold_blend_grid = screen_blends(ph_te, ps_te, y_te)
    blend_hold = _score_pairs(a_star * ph_te + (1 - a_star) * ps_te, y_te)
    blend_hold["note"] = f"α={a_star:.2f} chosen on train"
    blend_hold["alpha"] = a_star

    pipe = fit_stack(ph_tr, ps_tr, y_tr)
    if pipe is None:
        stack_hold = dict(slice_rem)
        stack_hold["note"] = "degenerate"
    else:
        stack_hold = _score_pairs(predict_stack(pipe, ph_te, ps_te), y_te)
        stack_hold["note"] = "logit stack fit on train"

    print("=== residual mix ===", flush=True)
    residual_model, residual_meta = train_residual(train, test, slice_model, codes, n_club, device)

    def soft_res(card, cut):
        return predict_residual(residual_model, slice_model, card, cut, codes, device)

    ph_res, _ps, y_res = collect_remainder_pairs(test, slice_model, None, None, None, soft_h_fn=soft_res)
    residual_hold = _score_pairs(ph_res, y_res)
    residual_hold["note"] = f"epochs={residual_meta['epochs_run']}"

    arms = {
        "slice": slice_rem,
        "horizon": horizon_rem,
        "blend": blend_hold,
        "stack": stack_hold,
        "residual": residual_hold,
    }
    best_name = min(arms, key=lambda k: arms[k]["log_loss"])
    best = arms[best_name]
    if best["log_loss"] < min(slice_rem["log_loss"], horizon_rem["log_loss"]) - 0.005:
        call = (
            f"**{best_name}** wins remainder log **{best['log_loss']:.4f}** vs slice "
            f"{slice_rem['log_loss']:.4f} / horizon {horizon_rem['log_loss']:.4f}. Candidate mix."
        )
    elif best["log_loss"] < min(slice_rem["log_loss"], horizon_rem["log_loss"]):
        call = (
            f"**{best_name}** is hairline-best at **{best['log_loss']:.4f}** "
            f"(slice {slice_rem['log_loss']:.4f}, horizon {horizon_rem['log_loss']:.4f}). "
            "Inconclusive — do not promote."
        )
    else:
        call = (
            f"No mix beats both parents on remainder "
            f"(best {best_name}={best['log_loss']:.4f}; slice {slice_rem['log_loss']:.4f}; "
            f"horizon {horizon_rem['log_loss']:.4f}). Park mixes."
        )

    results = {
        **arms,
        "blend_grid": hold_blend_grid["grid"],
        "train_best_alpha": a_star,
        "spine_slice_all_rows": spine,
        "horizon_meta": {"epochs": horizon_meta["epochs_run"], "best": horizon_meta["best_rem_log"]},
        "residual_meta": residual_meta,
        "call": call,
        "winner": best_name,
    }
    path = write_report(results)
    print(call, flush=True)
    print(f"wrote {path}", flush=True)

    # Card: slice remainder | horizon | residual mix (the learned mix)
    arsenal = next(c for c in test if c.team == "Arsenal")
    cut = cut_at_fraction(arsenal, 0.4)
    soft_h = predict_horizon(horizon_model, arsenal, cut, codes, device)
    soft_s = slice_model.predict_card(arsenal)
    soft_s_rem = soft_s.copy()
    soft_s_rem[:, :cut, :] = 0.0  # show slice only on remainder for the panel
    soft_res = predict_residual(residual_model, slice_model, arsenal, cut, codes, device)
    from PIL import Image, ImageDraw

    from src.models.tensor_card_render import BG, INK, MUTED, _draw_panel, _font, _regulars

    players = _regulars(arsenal)
    panel_w = 28 + len(players) * 2 * 15 + 20
    panel_h = 40 + arsenal.n_fixtures * 15 + 30
    gutter = 36
    width = 24 + 3 * panel_w + 2 * gutter
    height = 70 + panel_h + 48
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.text(
        (24, 16),
        f"Slice×horizon mix — {arsenal.team} {arsenal.season} · cut 40%",
        fill=INK,
        font=_font(16),
    )
    draw.text(
        (24, 40),
        f"Train blend picked α={a_star:.2f}. Residual = σ(logit(slice) + card δ).",
        fill=MUTED,
        font=_font(12),
    )
    origins = [(48, 88), (48 + panel_w + gutter, 88), (48 + 2 * (panel_w + gutter), 88)]
    _draw_panel(draw, origins[0], arsenal, players, "paint_remainder", soft_s_rem, cut, "1. Slice on rem.")
    _draw_panel(draw, origins[1], arsenal, players, "paint_remainder", soft_h, cut, "2. Horizon")
    _draw_panel(draw, origins[2], arsenal, players, "paint_remainder", soft_res, cut, "3. Residual mix")
    out = REPORTS / "typed_tensor_slice_horizon_mix_arsenal_2025-26.png"
    img.save(out)
    ART.mkdir(parents=True, exist_ok=True)
    img.save(ART / out.name)
    print(f"wrote {out}", flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "note"} if isinstance(v, dict) and "log_loss" in v else v for k, v in results.items() if k in arms or k in ("train_best_alpha", "winner", "call")}, indent=2))


if __name__ == "__main__":
    main()
