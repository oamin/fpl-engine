"""True card, opacity card, and residual card for one club-season.

The model card keeps the probability as the gray of a full-size dot.
The residual is truth minus that probability, on the remainder event bits.
Mean squared residual is the Brier score. Mean negative log likelihood of
the same bits is the log score. Both are computed on the live remainder
only: past rows are revealed, so their residual is zero by construction.

Picture: 2026-11-13 mixer, seed 0, Arsenal 2025-26, cut at 20%.
Each player has two dots: first goal bit, first assist bit
(the unary bit that means at least one).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.models.flashcard_baudot_row_trial import (
    ART,
    BG,
    CUT_FRACS,
    GATE,
    GOAL_BITS,
    GRID,
    INK,
    MUTED,
    OFF,
    ON,
    PLATE,
    REPORTS,
    SEED,
    TEST_SEASON,
    TRAIN_SEASONS,
    BlockTokenMixer,
    _font,
    _player_pad,
    build_baudot_cards,
    build_cards,
    build_league,
    build_name_codes,
    cut_at_fraction,
    evaluate_mixer,
    event_target,
    fit_clubs,
    load_odds,
    remainder_view,
    train_mixer,
)
from src.models.flashcard_card_score import balanced_score

MISS = (176, 48, 42)
FALSE = (36, 78, 168)
EPS = 1e-4


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round((1.0 - t) * x + t * y)) for x, y in zip(a, b))


def soft_events(model, card, cut: int, device: torch.device) -> np.ndarray:
    """P, T, E probabilities for one card at one cut."""
    view = remainder_view(card, cut)
    pad = _player_pad(card)
    model.eval()
    with torch.no_grad():
        return model(
            torch.from_numpy(view["bits"][None]).to(device),
            torch.from_numpy(view["row_on"][None]).to(device),
            torch.from_numpy(pad[None]).to(device),
        )[0].cpu().numpy()


def live_pairs(card, soft: np.ndarray, cut: int) -> tuple[np.ndarray, np.ndarray]:
    """Truth and probability for every remainder event bit on a player who played."""
    ys, ps = [], []
    pad = _player_pad(card)
    for gw in range(cut, card.n_fixtures):
        target = event_target(card, gw)
        for row in range(card.n_players):
            if pad[row] or not card.present[row, gw]:
                continue
            ys.append(target[row])
            ps.append(soft[row, gw])
    if not ys:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(ys), np.concatenate(ps)


def residual_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Brier is the mean squared residual. Log score is the mean negative log likelihood."""
    if len(y) == 0:
        return {"n": 0, "brier": 0.0, "log_loss": 0.0, "mae": 0.0}
    p = np.clip(p, EPS, 1.0 - EPS)
    resid = y - p
    return {
        "n": int(len(y)),
        "brier": float(np.mean(resid**2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "mae": float(np.mean(np.abs(resid))),
    }


def holdout_scores(model, cards, device) -> list[dict]:
    rows = []
    for frac in CUT_FRACS:
        ys, ps = [], []
        for card in cards:
            cut = cut_at_fraction(card, frac)
            soft = soft_events(model, card, cut, device)
            y, p = live_pairs(card, soft, cut)
            ys.append(y)
            ps.append(p)
        score = residual_scores(np.concatenate(ys), np.concatenate(ps))
        score["revealed"] = f"{frac:.0%}"
        rows.append(score)
    return rows


def _rot_name(text: str, size: int = 11) -> Image.Image:
    font = _font(size)
    tmp = Image.new("RGB", (140, size + 6), PLATE)
    d = ImageDraw.Draw(tmp)
    d.text((2, 0), text, fill=INK, font=font)
    return tmp.rotate(90, expand=True, fillcolor=PLATE)


def render_trio(card, soft: np.ndarray, cut: int, scores: dict, path: Path) -> Path:
    """Three panels: truth, opacity, signed residual. Remainder and revealed rows."""
    n_g = card.n_fixtures
    players = [i for i in range(min(12, card.n_players)) if not _player_pad(card)[i]]
    channels = (("G", 0), ("A", GOAL_BITS))
    DOT, GAP = 11, 3
    step = DOT + GAP
    cols = len(players) * len(channels)
    LEFT, TOP = 44, 78
    panel_w = LEFT + cols * step + 18
    names_h = 156
    panel_h = TOP + n_g * step + names_h
    gap = 28
    W = 3 * panel_w + 2 * gap + 32
    paste_y = 64
    H = paste_y + panel_h + 72
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((16, 12), f"{card.team} {card.season}  ·  truth, opacity, residual", fill=INK, font=_font(18))
    d.text(
        (16, 38),
        f"G = at least one goal, A = at least one assist. Line is the 20% cut (row {cut}). "
        "Above the line the bits are revealed, so the residual is blank.",
        fill=MUTED,
        font=_font(11),
    )
    foot = (
        f"Arsenal remainder  ·  Brier {scores['brier']:.4f}  ·  log loss {scores['log_loss']:.4f}  ·  "
        f"|residual| {scores['mae']:.4f}  ·  {scores['n']} event bits"
    )
    d.text((16, H - 24), foot, fill=INK, font=_font(12))

    titles = (
        "True card",
        "Model card  ·  opacity = probability",
        "Difference  ·  truth − model",
    )
    subs = (
        "Black = event. Light = none.",
        "Darker gray = a higher probability.",
        "Red = missed event. Blue = extra ink. Light = the cards agree.",
    )

    def dot(panel, r, c, fill):
        x = LEFT + c * step
        y = TOP + r * step
        ImageDraw.Draw(panel).rectangle((x, y, x + DOT - 1, y + DOT - 1), fill=fill)

    panels = []
    for title, sub in zip(titles, subs, strict=True):
        panel = Image.new("RGB", (panel_w, panel_h), BG)
        pd = ImageDraw.Draw(panel)
        pd.text((12, 8), title, fill=INK, font=_font(15))
        pd.text((12, 30), sub, fill=MUTED, font=_font(10))
        pd.rounded_rectangle(
            (LEFT - 8, TOP - 8, LEFT + cols * step + 4, TOP + n_g * step + 4),
            radius=6,
            fill=PLATE,
            outline=GRID,
        )
        for r in range(n_g):
            pd.text((6, TOP + r * step), str(int(card.gw[r])), fill=MUTED, font=_font(8))
        if 0 < cut < n_g:
            y = TOP + cut * step - 2
            pd.line((LEFT - 6, y, LEFT + cols * step + 2, y), fill=MISS, width=2)
        for i, row in enumerate(players):
            name = str(card.names[row]).split()[-1][:12]
            name_img = _rot_name(name)
            x = LEFT + i * len(channels) * step
            panel.paste(name_img, (x, TOP + n_g * step + 8))
            for k, _ in enumerate(channels):
                pd.text((x + k * step, TOP - 22), channels[k][0], fill=MUTED, font=_font(9))
        panels.append(panel)

    truth_p, opacity_p, resid_p = panels
    for r in range(n_g):
        target = event_target(card, r)
        for i, row in enumerate(players):
            for k, (_, bit) in enumerate(channels):
                c = i * len(channels) + k
                played = bool(card.present[row, r]) and not _player_pad(card)[row]
                y = float(target[row, bit]) if played else 0.0
                p = float(soft[row, r, bit]) if played and r >= cut else y
                dot(truth_p, r, c, ON if y >= 0.5 else OFF)
                if not played:
                    dot(opacity_p, r, c, OFF)
                    dot(resid_p, r, c, OFF)
                    continue
                if r < cut:
                    dot(opacity_p, r, c, ON if y >= 0.5 else OFF)
                    dot(resid_p, r, c, OFF)
                    continue
                dot(opacity_p, r, c, blend(OFF, ON, p))
                resid = y - p
                if resid >= 0:
                    dot(resid_p, r, c, blend(OFF, MISS, resid))
                else:
                    dot(resid_p, r, c, blend(OFF, FALSE, -resid))

    for i, panel in enumerate(panels):
        img.paste(panel, (16 + i * (panel_w + gap), paste_y))
    scale_y = paste_y + panel_h + 4
    draw_scale(img, 16 + (panel_w + gap), scale_y, "opacity", (0, 0.25, 0.5, 0.75, 1), lambda t: blend(OFF, ON, t))
    draw_scale(
        img,
        16 + 2 * (panel_w + gap),
        scale_y,
        "truth − model",
        (-1, -0.5, 0, 0.5, 1),
        lambda t: blend(OFF, FALSE, -t) if t < 0 else blend(OFF, MISS, t),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def draw_scale(img: Image.Image, x: int, y: int, caption: str, values, key) -> None:
    d = ImageDraw.Draw(img)
    d.text((x, y), caption, fill=MUTED, font=_font(9))
    x0 = x
    y0 = y + 14
    for val in values:
        label = f"{val:g}" if isinstance(val, float) else str(val)
        d.rectangle((x0, y0, x0 + 12, y0 + 12), fill=key(float(val)), outline=GRID)
        d.text((x0 + 14, y0), label, fill=MUTED, font=_font(8))
        x0 += 42


def write_note(arsenal: dict, holdout: list[dict], gate_mean: float, path: Path) -> None:
    lines = [
        "# Residual of the true card and the opacity card",
        "",
        "The painted dot keeps the model probability as its gray. "
        "The difference card is truth minus that probability. "
        "Past rows are revealed, so they cancel and the residual lives on the remainder event bits.",
        "",
        "Mean squared residual is the Brier score. "
        "Mean negative log likelihood of the same bits is the log score. "
        "Both are strictly proper for a probability. "
        "The log score is the likelihood: a dark dot on a blank week costs more than the same mistake costs under Brier. "
        "Brier is the residual itself, squared and averaged.",
        "",
        f"Model: 2026-11-13 mixer, seed 0. Thresholded gate mean on this retrain **{gate_mean:.3f}**.",
        "",
        f"Arsenal 2025-26 remainder at 20%, {arsenal['n']} event bits: "
        f"Brier **{arsenal['brier']:.4f}**, log loss **{arsenal['log_loss']:.4f}**, "
        f"mean absolute residual **{arsenal['mae']:.4f}**.",
        "",
        "Picture: `reports/baudot_residual_arsenal_2025-26.png`.",
        "",
        "## Holdout remainder, all event bits",
        "",
        "| revealed | bits | Brier | log loss | mean |residual| |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in holdout:
        lines.append(
            f"| {row['revealed']} | {row['n']} | {row['brier']:.4f} | {row['log_loss']:.4f} | {row['mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Lower is better on both scores. This is a reading of the existing mixer, not a new trained comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
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
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = BlockTokenMixer(width=train[0].width).to(device)
    print(f"training mixer device {device}", flush=True)
    train_mixer(model, train, rng, device, test=test)
    stats = evaluate_mixer(model, test, device)
    gate_mean = float(np.mean([row["score"] for row in stats["rows"]]))
    print(f"retrained gate mean {gate_mean:.4f}", flush=True)
    arsenal = next(c for c in test if c.team == "Arsenal")
    cut = cut_at_fraction(arsenal, 0.2)
    soft = soft_events(model, arsenal, cut, device)
    y, p = live_pairs(arsenal, soft, cut)
    arsenal_scores = residual_scores(y, p)
    print("arsenal", arsenal_scores, flush=True)
    holdout = holdout_scores(model, test, device)
    for row in holdout:
        print("holdout", row, flush=True)
    out = ART / "baudot_residual_arsenal_2025-26.png"
    render_trio(arsenal, soft, cut, arsenal_scores, out)
    render_trio(arsenal, soft, cut, arsenal_scores, REPORTS / "baudot_residual_arsenal_2025-26.png")
    write_note(arsenal_scores, holdout, gate_mean, REPORTS / "baudot_residual_card.md")
    state = {
        "trial": "baudot_residual_card",
        "model": "baudot_row_mixer_learnable_tversky",
        "gate_mean_retrain": gate_mean,
        "published_gate_mean": float(np.mean([balanced_score(GATE[k])["score"] for k in GATE])),
        "arsenal_20": arsenal_scores,
        "holdout": holdout,
        "picture": str(out),
    }
    (ART / "baudot_residual_card.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (REPORTS / "baudot_residual_card.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
