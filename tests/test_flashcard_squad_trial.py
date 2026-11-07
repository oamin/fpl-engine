"""Squad cards hide the whole segment, and a one-pass model can read minutes."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_squad_trial import (
    COLS,
    GOALS_C,
    IMG_H,
    IMG_W,
    MASK_C,
    MASK_ID,
    ROWS,
    SquadCard,
    SquadSequence,
    apply_cut,
    render_grid,
    train_model,
)


def _league() -> dict:
    fixtures = []
    for gw in range(1, 31):
        fixtures.append(
            {
                "fixture": gw,
                "gw": gw,
                "kickoff": gw,
                "gc": 0.0 if gw <= 19 else 5.0,
                "played60": {1} if gw <= 19 else {2},
                "played1": {1} if gw <= 19 else {2},
            }
        )
    return {"toy": {"Opp": {"fixtures": fixtures, "by_id": {fx["fixture"]: fx for fx in fixtures}}}}


def _card() -> SquadCard:
    gw = np.zeros(COLS, dtype=np.int64)
    kickoff = np.zeros(COLS, dtype=np.int64)
    home = np.zeros(COLS, dtype=np.int64)
    fixture_id = np.zeros(COLS, dtype=np.int64)
    opponents = [""] * COLS
    for col, week in enumerate(range(1, 31)):
        gw[col] = week
        kickoff[col] = week
        home[col] = week % 2
        fixture_id[col] = week
        opponents[col] = "Opp"
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    assists = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    minutes[0, :30] = 90
    goals[0, :30] = 3
    present[0, :30] = True
    position = np.full(ROWS, -1, dtype=np.int64)
    position[0] = 2
    names = ["Player"] + ["spare"] * (ROWS - 1)
    return SquadCard(
        season="toy",
        team="Home",
        names=tuple(names),
        position=position,
        gw=gw,
        kickoff=kickoff,
        home=home,
        opponent=tuple(opponents),
        fixture_id=fixture_id,
        minutes=minutes,
        goals=goals,
        assists=assists,
        present=present,
        n_players=1,
        n_fixtures=30,
    )


class SquadCutTests(unittest.TestCase):
    def test_hidden_goals_are_masked_for_both_readers(self) -> None:
        view = apply_cut(_card(), _league(), 19)
        self.assertTrue(np.all(view["goal_token"][0, :19] == 3))
        self.assertTrue(np.all(view["goal_token"][0, 19:30] == MASK_ID))
        self.assertTrue(np.all(view["goal_label"][0, 19:30] == 3))
        image = render_grid(view)
        self.assertEqual(image.shape, (IMG_H, IMG_W, 3))
        hidden_band = image[HEADER_GOALS, 19 * 4 : 20 * 4]
        self.assertTrue(np.all(hidden_band == MASK_C))
        visible_band = image[HEADER_GOALS : HEADER_GOALS + 4, 0:4]
        self.assertTrue(np.any(np.all(visible_band == GOALS_C, axis=-1)))

    def test_changing_a_hidden_goal_does_not_change_the_input(self) -> None:
        card = _card()
        other = _card()
        other.goals = other.goals.copy()
        other.goals[0, 25] = 0
        left = apply_cut(card, _league(), 19)
        right = apply_cut(other, _league(), 19)
        self.assertTrue(np.array_equal(left["goal_token"], right["goal_token"]))
        self.assertTrue(np.array_equal(render_grid(left), render_grid(right)))
        self.assertEqual(int(left["goal_label"][0, 25]), 3)
        self.assertEqual(int(right["goal_label"][0, 25]), 0)

    def test_hidden_header_ignores_later_conceded_goals(self) -> None:
        view = apply_cut(_card(), _league(), 19)
        self.assertLess(view["concede"][24], 1.0)
        self.assertEqual(view["concede_missing"][24], 0.0)
        self.assertEqual(view["regulars"][24], 0.0)
        self.assertGreater(view["regulars"][5], 0.5)

    def test_sequence_model_reads_a_benched_teammate(self) -> None:
        rng = np.random.default_rng(0)
        cards = [_coupled_card(rng) for _ in range(16)]
        league = {"toy": {"Opp": {"fixtures": [], "by_id": {}}}}
        torch.manual_seed(0)
        model = SquadSequence(dropout=0.0)
        train_model(
            model, cards, league, np.random.default_rng(0), epochs=30, kind="sequence", lr=2e-3, batch_size=4
        )
        model.eval()
        out_probs = []
        in_probs = []
        with torch.no_grad():
            for card in cards:
                logits, _ = model(_one(apply_cut(card, league, 6)))
                prob = torch.softmax(logits, dim=-1)[0, 1].numpy()
                hidden = np.where(card.gw > 6)[0]
                star_out = card.minutes[0, hidden] < 1
                out_probs.extend(prob[hidden][star_out, 1].tolist())
                in_probs.extend(prob[hidden][~star_out, 1].tolist())
        self.assertGreater(float(np.mean(out_probs)), 0.7)
        self.assertLess(float(np.mean(in_probs)), 0.3)


HEADER_GOALS = 12 + 4


def _coupled_card(rng: np.random.Generator) -> SquadCard:
    """Row 1 scores exactly when row 0 has no minutes."""
    n = 12
    gw = np.zeros(COLS, dtype=np.int64)
    kickoff = np.zeros(COLS, dtype=np.int64)
    gw[:n] = np.arange(1, n + 1)
    kickoff[:n] = gw[:n]
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    star = rng.random(n) < 0.5
    minutes[0, :n] = np.where(star, 90, 0)
    minutes[1, :n] = 90
    present[0, :n] = True
    present[1, :n] = True
    goals[1, :n] = (minutes[0, :n] < 1).astype(np.int64)
    position = np.full(ROWS, -1, dtype=np.int64)
    position[:2] = 2
    return SquadCard(
        season="toy",
        team="Home",
        names=tuple(["Star", "Other"] + ["spare"] * (ROWS - 2)),
        position=position,
        gw=gw,
        kickoff=kickoff,
        home=np.zeros(COLS, dtype=np.int64),
        opponent=tuple([""] * COLS),
        fixture_id=np.zeros(COLS, dtype=np.int64),
        minutes=minutes,
        goals=goals,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=2,
        n_fixtures=n,
    )


def _one(view: dict) -> dict:
    from src.models.flashcard_squad_trial import _stack

    return _stack([view])
