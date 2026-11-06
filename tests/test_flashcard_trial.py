"""Flash-card inputs: the column being scored is not an input."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.models.flashcard_trial import (
    BLANK,
    FlashGRU,
    Seq,
    _opponent_names,
    choose_cut,
    evaluate_gru,
    fit_team_index,
    shift_events,
    train_model,
)


class FlashcardInputTests(unittest.TestCase):
    def test_opponent_is_the_other_club_on_the_fixture(self) -> None:
        frame = pd.DataFrame(
            {
                "fixture": [1, 1, 1, 2],
                "team": ["Arsenal", "Arsenal", "Wolves", "Brentford"],
            }
        )
        names = _opponent_names(frame).tolist()
        self.assertEqual(names, ["Wolves", "Wolves", "Arsenal", "UNK"])

    def test_current_bar_is_not_the_input_at_that_column(self) -> None:
        events = np.arange(1, 21, dtype=np.int64).reshape(1, 5, 4)
        cut = np.array([2])
        length = np.array([5])
        teacher = shift_events(events, cut, length, "teacher")
        blank = shift_events(events, cut, length, "blank")
        self.assertTrue(np.all(teacher[0, 0] == BLANK))
        self.assertTrue(np.all(teacher[0, 1:] == events[0, :-1]))
        self.assertTrue(np.all(blank[0, 3:] == BLANK))
        self.assertTrue(np.all(blank[0, 2] == events[0, 1]))

    def test_cut_leaves_a_remainder(self) -> None:
        self.assertEqual(choose_cut(20, 0.5), 10)
        self.assertEqual(choose_cut(12, 0.1), 4)
        self.assertEqual(choose_cut(12, 0.95), 8)

    def test_gru_learns_a_minutes_rule(self) -> None:
        """90-minute columns are goal height 1. Zero-minute columns are height 0."""
        rng = np.random.default_rng(0)
        sequences = []
        for i in range(32):
            minutes = rng.choice(np.array([0, 90], dtype=np.float32), size=16)
            events = np.zeros((16, 4), dtype=np.int64)
            events[:, 0] = (minutes >= 60).astype(np.int64)
            sequences.append(
                Seq(
                    season="toy",
                    element=i,
                    name="Toy",
                    minutes=minutes,
                    home=np.zeros(16, dtype=np.int64),
                    team=np.array(["A"] * 16, dtype=object),
                    opponent=np.array(["B"] * 16, dtype=object),
                    position=np.array(["MID"] * 16, dtype=object),
                    gw=np.arange(1, 17),
                    events=events,
                )
            )
        index = fit_team_index(sequences)
        torch.manual_seed(0)
        model = FlashGRU(n_teams=len(index) + 1)
        train_model(
            model,
            sequences,
            index,
            np.random.default_rng(0),
            torch.device("cpu"),
            epochs=40,
        )
        probe = sequences[0]
        # Make the probe's remainder a known mix of played and benched weeks.
        probe.minutes[:] = np.array([90, 0] * 8, dtype=np.float32)
        probe.events[:, 0] = (probe.minutes >= 60).astype(np.int64)
        scored = evaluate_gru(model, [probe], index, 0.5, torch.device("cpu"))
        self.assertGreater(scored["p_positive_played"]["goals"], 0.8)


if __name__ == "__main__":
    unittest.main()
