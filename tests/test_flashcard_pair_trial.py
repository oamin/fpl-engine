"""The other squad is read by gameweek, and a later goal stays masked."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_pair_trial import SLOTS, PairSequence, _stack, pair_view, train_pair
from src.models.flashcard_squad_trial import COLS, MASK_ID, POS, ROWS, SquadCard, squad_index


def _card(
    team: str, opponent: str, minutes_row: np.ndarray, goals_row: np.ndarray, season: str = "toy"
) -> SquadCard:
    n = len(minutes_row)
    gw = np.zeros(COLS, dtype=np.int64)
    gw[:n] = np.arange(1, n + 1)
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    position = np.full(ROWS, -1, dtype=np.int64)
    minutes[0, :n] = minutes_row
    goals[0, :n] = goals_row
    present[0, :n] = True
    position[0] = POS["FWD"]
    return SquadCard(
        season=season,
        team=team,
        names=tuple(["Player"] + ["spare"] * (ROWS - 1)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
        home=np.zeros(COLS, dtype=np.int64),
        opponent=tuple([opponent] * n + [""] * (COLS - n)),
        fixture_id=gw.copy(),
        minutes=minutes,
        goals=goals,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=1,
        n_fixtures=n,
    )


class PairWindowTests(unittest.TestCase):
    def test_previous_weeks_are_visible_and_later_goals_stay_masked(self) -> None:
        home = _card("Home", "Opp", np.full(6, 90.0), np.zeros(6))
        opp = _card("Opp", "Home", np.array([90, 0, 90, 0, 90, 0], dtype=float), np.array([1, 2, 3, 1, 2, 3]))
        view = pair_view(home, 3, squad_index([home, opp]))
        # Gameweek 4 is hidden. Its window holds gameweeks 2, 3 and 4.
        week = 3
        filled = np.where(~view["opp_pad"][0, week])[0]
        self.assertGreaterEqual(len(filled), 3)
        played = view["opp_minutes"][0, week, filled] >= 90
        self.assertEqual(int(view["opp_goal_token"][0, week, filled][played][0]), 3)
        self.assertTrue(np.any(view["opp_goal_token"][0, week, filled] == MASK_ID))
        shown = np.where(~view["opp_pad"][0, 2])[0]
        self.assertIn(3, set(int(view["opp_goal_token"][0, 2, slot]) for slot in shown))
        opp.goals = opp.goals.copy()
        opp.goals[0, 3] = 0
        again = pair_view(home, 3, squad_index([home, opp]))
        self.assertTrue(np.array_equal(view["opp_goal_token"], again["opp_goal_token"]))

    def test_a_squad_reads_the_other_squads_previous_week(self) -> None:
        cards = []
        rng = np.random.default_rng(0)
        for index in range(12):
            opp_minutes = rng.integers(0, 2, size=8) * 90
            goals = np.zeros(8, dtype=int)
            goals[2:] = (opp_minutes[1:-1] < 1).astype(int)
            season = f"toy-{index}"
            cards.append(_card("Home", "Opp", np.full(8, 90.0), goals, season))
            cards.append(_card("Opp", "Home", opp_minutes.astype(float), np.zeros(8, dtype=int), season))
        torch.manual_seed(0)
        model = PairSequence(dropout=0.0)
        train_pair(model, cards, squad_index(cards), np.random.default_rng(0), epochs=25, lr=2e-3, batch_size=4)
        model.eval()
        home = next(card for card in cards if card.team == "Home" and card.season == "toy-0")
        opp = next(card for card in cards if card.team == "Opp" and card.season == "toy-0")
        with torch.no_grad():
            logits, _ = model(_stack([pair_view(home, 2, squad_index([home, opp]))]))
        prob = torch.softmax(logits, dim=-1)[0, 0, 2:8, 1:].sum(-1).numpy()
        hidden_goals = home.goals[0, 2:8] > 0
        self.assertGreater(float(prob[hidden_goals].mean()), 0.6)
        self.assertLess(float(prob[~hidden_goals].mean()), 0.4)
        self.assertEqual(pair_view(home, 2, squad_index([home, opp]))["opp_minutes"].shape[-1], SLOTS)
