"""Squad cards hide the whole segment, and a one-pass model can read minutes."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_squad_trial import (
    BAND_H,
    CELL_W,
    COLS,
    GOALS_C,
    HEADER_H,
    IMG_H,
    IMG_W,
    MASK_C,
    MASK_ID,
    MINUTES_C,
    PLAYER_H,
    POS_C,
    ROWS,
    SquadCard,
    SquadSequence,
    POS,
    SquadVision,
    apply_cut,
    bar_correspondence,
    render_grid,
    sample_bars,
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
        goal_y = HEADER_H + 2 * BAND_H
        hidden_band = image[goal_y, 19 * CELL_W : 20 * CELL_W]
        self.assertTrue(np.all(hidden_band == MASK_C))
        visible_band = image[goal_y : goal_y + BAND_H, 0:CELL_W]
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

    def test_picture_uses_the_minute_count_position_and_finer_concede(self) -> None:
        card = _card()
        card.minutes = card.minutes.copy()
        card.present = card.present.copy()
        card.position = card.position.copy()
        card.minutes[0, 0] = 59
        card.minutes[0, 1] = 60
        card.position[1] = 3
        card.minutes[1, 0] = 90
        card.present[1, 0] = True
        view = apply_cut(card, _league(), 19)
        view["concede"] = view["concede"].copy()
        view["concede_missing"] = view["concede_missing"].copy()
        view["concede"][0] = 0.74
        view["concede"][1] = 1.12
        view["concede_missing"][0] = 0.0
        view["concede_missing"][1] = 0.0
        image = render_grid(view)
        minutes_band = image[HEADER_H : HEADER_H + 2 * BAND_H]
        self.assertFalse(np.array_equal(minutes_band[:, 0:CELL_W], minutes_band[:, CELL_W : 2 * CELL_W]))
        self.assertTrue(np.any(np.all(minutes_band[:, 0:CELL_W] == MINUTES_C, axis=-1)))
        self.assertTrue(np.any(np.all(minutes_band[:, CELL_W : 2 * CELL_W] == MINUTES_C, axis=-1)))
        pos_y = HEADER_H + 4 * BAND_H
        self.assertTrue(np.array_equal(image[pos_y, 0], np.array(POS_C[2])))
        self.assertTrue(np.array_equal(image[pos_y + PLAYER_H, 0], np.array(POS_C[3])))
        concede = image[BAND_H : BAND_H * 2]
        self.assertFalse(np.array_equal(concede[:, 0:CELL_W], concede[:, CELL_W : 2 * CELL_W]))
        torch.manual_seed(0)
        batch = _one(view)
        goals, assists = SquadVision(dropout=0.0)(
            batch["image"], batch["row_pad"], batch["column_on"], batch["position"], batch["minutes"]
        )
        self.assertEqual(tuple(goals.shape), (1, ROWS, COLS, 4))
        self.assertEqual(tuple(assists.shape), (1, ROWS, COLS, 4))

    def test_matching_bars_overlap_on_the_same_weeks(self) -> None:
        card = _card()
        labels = card.goals[None, :, :]
        probs = np.zeros((1, ROWS, COLS, 4), dtype=np.float64)
        probs[:, :, :, 0] = 1.0
        hidden = (card.gw > 19) & card.present[0]
        probs[0, 0, hidden] = 0.0
        probs[0, 0, hidden, 3] = 1.0
        matched = bar_correspondence([card], probs, labels, 19, np.random.default_rng(0), draws=5)
        self.assertEqual(matched["actual_bars"], int(hidden.sum()))
        self.assertEqual(matched["drawn_bars"], float(hidden.sum()))
        self.assertEqual(matched["actual_covered"], 1.0)
        self.assertEqual(matched["drawn_on_actual"], 1.0)
        blank = np.zeros_like(probs)
        blank[:, :, :, 0] = 1.0
        missed = bar_correspondence([card], blank, labels, 19, np.random.default_rng(0), draws=3)
        self.assertEqual(missed["drawn_bars"], 0.0)
        self.assertEqual(missed["actual_covered"], 0.0)
        certain = np.zeros((2, 4))
        certain[:, 1] = 1.0
        self.assertTrue(np.all(sample_bars(certain, np.random.default_rng(0)) == 1))

    def test_position_is_learned_for_a_goalkeeper_and_a_forward(self) -> None:
        cards = [_role_card() for _ in range(8)]
        league = {"toy": {"Opp": {"fixtures": [], "by_id": {}}}}
        torch.manual_seed(0)
        model = SquadSequence(dropout=0.0)
        train_model(model, cards, league, np.random.default_rng(0), epochs=20, kind="sequence", lr=2e-3, batch_size=4)
        model.eval()
        with torch.no_grad():
            logits, _ = model(_one(apply_cut(cards[0], league, 3)))
        prob = torch.softmax(logits, dim=-1)[0]
        hidden = np.where(cards[0].gw > 3)[0]
        self.assertLess(float(prob[0, hidden, 1:].sum(-1).mean()), 0.2)
        self.assertGreater(float(prob[1, hidden, 1:].sum(-1).mean()), 0.7)


def _role_card() -> SquadCard:
    """A goalkeeper never scores. A forward scores every week. Minutes match."""
    n = 6
    gw = np.zeros(COLS, dtype=np.int64)
    gw[:n] = np.arange(1, n + 1)
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    minutes[:2, :n] = 90
    present[:2, :n] = True
    goals[1, :n] = 1
    position = np.full(ROWS, -1, dtype=np.int64)
    position[0] = POS["GKP"]
    position[1] = POS["FWD"]
    return SquadCard(
        season="toy",
        team="Home",
        names=tuple(["Keeper", "Forward"] + ["spare"] * (ROWS - 2)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
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
