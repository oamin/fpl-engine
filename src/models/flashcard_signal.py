"""Which input the square-card reader uses when it paints bars.

The full painter is trained as usual. Each signal is then removed at the
test cut, and the painted card is scored again. A second check keeps the
model's fixture budgets and changes only the ranking.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.flashcard_calibrated_trial import integer_budgets
from src.models.flashcard_card_score import _paint, tally_cards
from src.models.flashcard_odds_trial import load_odds
from src.models.flashcard_squad_trial import (
    SEED,
    _by_club,
    _by_position,
    _stack,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    squad_index,
)
from src.models.flashcard_square_trial import (
    EVENT_H,
    HEADER_H,
    MINUTE_H,
    ODDS_H,
    PLAYER_H,
    compose_square,
)
from src.models.flashcard_strip_trial import StripVision, _batch_images, strip_features, train_stripped
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"

MINUTE_BAND = (0, MINUTE_H)
GOAL_BAND = (MINUTE_H, MINUTE_H + EVENT_H)
ASSIST_BAND = (MINUTE_H + EVENT_H, MINUTE_H + 2 * EVENT_H)
POS_BAND = (MINUTE_H + 2 * EVENT_H, PLAYER_H)


def blank_rows(images: torch.Tensor, y0: int, y1: int) -> None:
    """Paint a horizontal band white. White is the empty mark."""
    images[:, :, y0:y1, :] = 1.0


def blank_player_band(images: torch.Tensor, band: tuple[int, int]) -> None:
    """Paint one row of every player cell white."""
    start, stop = band
    y = HEADER_H
    while y + PLAYER_H <= images.shape[2]:
        blank_rows(images, y + start, y + stop)
        y += PLAYER_H


def place_ranked(key: np.ndarray, budgets: np.ndarray, mask: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    """Give each fixture's budget to the rows with the largest key and some minutes."""
    drawn = np.zeros(minutes.shape, dtype=np.int64)
    for col in range(minutes.shape[1]):
        budget = int(budgets[col])
        if budget <= 0:
            continue
        rows = np.flatnonzero(mask[:, col] & (minutes[:, col] > 0))
        if len(rows) == 0:
            continue
        order = rows[np.argsort(-key[rows, col], kind="stable")]
        drawn[order[: min(budget, len(order))], col] = 1
    return drawn


def _forward(model, batch: dict, images: torch.Tensor):
    return model(
        images,
        batch["row_pad"],
        batch["column_on"],
        batch["position"],
        batch["minutes"],
        batch["matchup"],
        batch["club"],
        batch["opponent"],
        None,
    )


def goal_parts(model: StripVision, batch: dict) -> dict[str, torch.Tensor]:
    """Additive goal-logit terms that do not pass through the picture."""
    couple = model.couple
    pos = torch.where(batch["position"] < 0, torch.full_like(batch["position"], 4), batch["position"])
    played = (batch["minutes"] / 90.0).unsqueeze(-1)
    return {
        "position": couple.pos_goal(pos)[:, :, None, :],
        "minutes": couple.pos_goal_min(pos)[:, :, None, :] * played,
        "club": _by_club(couple.club_goal, couple.club_goal_min, batch["club"], pos, played),
        "opponent": _by_club(couple.opp_goal, couple.opp_goal_min, batch["opponent"], pos, played),
        "form": _by_position(couple.form_goal, couple.form_goal_min, batch["matchup"], pos, played),
    }


def _hidden(card, cut: int) -> np.ndarray:
    return (card.gw > cut) & (card.gw > 0)


def _prior_goals(card, cut: int) -> np.ndarray:
    shown = (card.gw <= cut) & (card.gw > 0)
    return (card.goals[:, shown] * card.present[:, shown]).sum(axis=1)


def _clone(batch: dict) -> dict:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def main() -> None:
    started = time.perf_counter()
    print("loading signal cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(train + test)
    odds = load_odds()
    n_clubs = max(clubs.values(), default=0) + 1
    torch.manual_seed(SEED)
    model = StripVision(HEADER_H, dropout=0.1, n_clubs=n_clubs, player_h=PLAYER_H)
    train_stripped(model, train, league, np.random.default_rng(SEED), clubs, odds, "signal", compose_square)

    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    revealed = 0.2
    cuts = [cut_at_fraction(card, revealed) for card in test]
    views = [
        strip_features(apply_cut(card, league, cut, clubs, squads))
        for card, cut in zip(test, cuts, strict=True)
    ]
    batch = _stack(views)
    images = torch.from_numpy(_batch_images(views, test, odds, compose_square)).permute(0, 3, 1, 2).float().div(255.0)

    def score(name: str, images_now: torch.Tensor, batch_now: dict):
        model.eval()
        with torch.no_grad():
            goal_logits, assist_logits = _forward(model, batch_now, images_now)
        goal_p = F.softmax(goal_logits, dim=-1).numpy()
        assist_p = F.softmax(assist_logits, dim=-1).numpy()
        goals = tally_cards(test, labels, _paint(goal_p, minutes, test, cuts), 0, cuts)
        assists = tally_cards(test, assist_labels, _paint(assist_p, minutes, test, cuts), 0, cuts)
        print(
            f"{name} goals {goals['painted']}/{goals['exact']}/{goals['actual_within']} "
            f"assists {assists['painted']}/{assists['exact']}/{assists['actual_within']}",
            flush=True,
        )
        return goal_p, goal_logits, goals, assists

    rows = []
    goal_p, goal_logits, goals, assists = score("intact", images, batch)
    rows.append(("intact", goals, assists))
    with torch.no_grad():
        parts = {name: term.expand_as(goal_logits) for name, term in goal_parts(model, batch).items()}
        parts["picture"] = goal_logits - sum(parts.values())

    def run(name: str, edit) -> None:
        images_now = images.clone()
        batch_now = _clone(batch)
        edit(images_now, batch_now)
        _p, _logits, g, a = score(name, images_now, batch_now)
        rows.append((name, g, a))

    run("white odds", lambda img, _b: blank_rows(img, 0, ODDS_H))
    run("white venue", lambda img, _b: blank_rows(img, ODDS_H, HEADER_H))
    run("white minutes", lambda img, _b: blank_player_band(img, MINUTE_BAND))
    run("white goals", lambda img, _b: blank_player_band(img, GOAL_BAND))
    run("white assists", lambda img, _b: blank_player_band(img, ASSIST_BAND))
    run("white position color", lambda img, _b: blank_player_band(img, POS_BAND))
    run("white picture", lambda img, _b: img.fill_(1.0))
    run("zero numeric minutes", lambda _img, b: b["minutes"].zero_())
    run("zero position id", lambda _img, b: b["position"].fill_(-1))
    run("zero matchup", lambda _img, b: b["matchup"].zero_())
    run("zero opponent id", lambda _img, b: b["opponent"].zero_())

    budgets = []
    for i, (card, cut) in enumerate(zip(test, cuts)):
        mask = card.present & _hidden(card, cut).reshape(1, -1) & (card.minutes > 0)
        budgets.append(integer_budgets(np.where(mask, 1.0 - goal_p[i, ..., 0], 0.0).sum(axis=0)))
    prior = np.stack([
        np.broadcast_to(_prior_goals(card, cut)[:, None], card.goals.shape).copy()
        for card, cut in zip(test, cuts)
    ])
    position = np.stack([np.broadcast_to(card.position[:, None], card.goals.shape).copy() for card in test])
    rank_rows = []
    for name, key in (
        ("minutes", minutes),
        ("prior goals", prior.astype(np.float64) + minutes / 1000.0),
        ("position then minutes", position.astype(np.float64) + minutes / 1000.0),
    ):
        painted = np.stack([
            place_ranked(
                key[i],
                budgets[i],
                test[i].present & _hidden(test[i], cuts[i]).reshape(1, -1),
                minutes[i],
            )
            for i in range(len(test))
        ])
        totals = tally_cards(test, labels, painted, 0, cuts)
        rank_rows.append((name, totals))
        print(f"rank {name} goals {totals['painted']}/{totals['exact']}/{totals['actual_within']}", flush=True)

    def own_budget(score: np.ndarray) -> np.ndarray:
        score = np.clip(score, 0.0, 0.99)
        probs = np.zeros(score.shape + (4,), dtype=np.float64)
        probs[..., 0] = 1.0 - score
        probs[..., 1] = score
        return _paint(probs, minutes, test, cuts)

    pos_rate = np.zeros_like(minutes)
    for code, rate in ((1, 0.04), (2, 0.10), (3, 0.18)):
        pos_rate[position == code] = rate
    own_rows = []
    for name, score in (
        ("minutes / 90", minutes / 90.0),
        ("prior goals", np.clip(prior, 0, 3) / 3.0),
        ("position rate", pos_rate),
    ):
        totals = tally_cards(test, labels, own_budget(score), 0, cuts)
        own_rows.append((name, totals))
        print(f"own {name} goals {totals['painted']}/{totals['exact']}/{totals['actual_within']}", flush=True)

    by_pos = []
    chances = []
    for code, label in ((0, "GK"), (1, "DEF"), (2, "MID"), (3, "FWD")):
        chosen = []
        for i, card in enumerate(test):
            mask = (
                card.present
                & _hidden(card, cuts[i]).reshape(1, -1)
                & (card.minutes > 0)
                & (card.position[:, None] == code)
            )
            if mask.any():
                chosen.append(1.0 - goal_p[i, mask, 0])
        values = np.concatenate(chosen) if chosen else np.array([0.0])
        chances.append(values)
        by_pos.append((label, float(values.mean()) if chosen else 0.0, int(values.size if chosen else 0)))
    mean_chance = float(np.concatenate([c for c in chances if c.size]).mean())

    term_size = {}
    for name, term in parts.items():
        values = []
        for i, card in enumerate(test):
            mask = torch.from_numpy(
                card.present & _hidden(card, cuts[i]).reshape(1, -1) & (card.minutes > 0)
            )
            if bool(mask.any()):
                values.append(term[i, mask, 0].abs().mean())
        term_size[name] = float(torch.stack(values).mean()) if values else 0.0

    lines = [
        "# What the square card uses",
        "",
        "The full painter is the gate reader, trained with seed 0. The test",
        "cut is 20% revealed on 2025-26. Each row removes one signal and paints",
        "again. White is the empty mark. The placement still requires minutes,",
        "so a player on zero minutes cannot receive a bar in any of these rows.",
        "",
        "| signal removed | goal bars painted | goal same appearance | goal within 1 GW | assist same appearance | assist within 1 GW |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, g, a in rows:
        lines.append(
            f"| {name} | {g['painted']} | {g['exact']} | {g['actual_within']} | {a['exact']} | {a['actual_within']} |"
        )
    lines.extend(
        [
            "",
            "The next rows keep the intact model's fixture budgets and change",
            "only who receives them. Minutes still have to be above zero.",
            "",
            "| ranking | goal bars painted | goal same appearance | goal within 1 GW |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, totals in rank_rows:
        lines.append(f"| {name} | {totals['painted']} | {totals['exact']} | {totals['actual_within']} |")
    lines.extend(
        [
            "",
            "These rows choose their own fixture budgets. A forward is 0.18,",
            "a midfielder 0.10, a defender 0.04, and a goalkeeper 0.",
            "",
            "| signal | goal bars painted | goal same appearance | goal within 1 GW |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, totals in own_rows:
        lines.append(f"| {name} | {totals['painted']} | {totals['exact']} | {totals['actual_within']} |")
    lines.extend(
        [
            "",
            f"On a hidden appearance with minutes, the mean chance of a goal bar is {mean_chance:.1%}.",
            "",
            "| position | mean goal-bar chance | appearances |",
            "|---|---:|---:|",
        ]
    )
    for label, rate, count in by_pos:
        lines.append(f"| {label} | {rate:.1%} | {count} |")
    lines.extend(
        [
            "",
            "Mean absolute contribution to the zero-class logit on those appearances.",
            "The picture term is whatever remains after the additive terms.",
            "",
            "| term | mean absolute logit |",
            "|---|---:|",
        ]
    )
    for name, size in term_size.items():
        lines.append(f"| {name} | {size:.3f} |")
    seconds = time.perf_counter() - started
    lines.extend(["", f"Wall clock: {seconds:.0f} seconds on CPU.", ""])
    path = ROOT_REPORTS / "flashcard_signal.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
