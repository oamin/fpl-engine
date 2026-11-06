"""The standard card hides the remainder, and a column model can read minutes."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_trial import Seq
from src.models.flashcard_visual_trial import (
    HEIGHT,
    INK,
    MASK,
    SLOTS,
    WIDTH,
    ColumnVision,
    collapse_repeated_rows,
    column_probabilities,
    mask_remainder,
    render_full,
    train_vision,
)


def _seq(
    minutes: list[float],
    goals: list[int],
    gw: list[int] | None = None,
    opponent: list[str] | None = None,
    home: list[int] | None = None,
) -> Seq:
    width = len(minutes)
    events = np.zeros((width, 4), dtype=np.int64)
    events[:, 0] = np.asarray(goals)
    return Seq(
        season="toy",
        element=1,
        name="Toy",
        minutes=np.asarray(minutes, dtype=np.float32),
        home=np.zeros(width, dtype=np.int64) if home is None else np.asarray(home),
        team=np.array(["Arsenal"] * width, dtype=object),
        opponent=np.array(["Wolves"] * width if opponent is None else opponent, dtype=object),
        position=np.array(["MID"] * width, dtype=object),
        gw=np.arange(1, width + 1) if gw is None else np.asarray(gw),
        events=events,
    )


class CardRenderTests(unittest.TestCase):
    def test_every_card_is_the_same_size(self) -> None:
        image = render_full(_seq([90, 0], [1, 0]))
        self.assertEqual(image.shape, (HEIGHT, WIDTH, 3))
        self.assertEqual((HEIGHT, WIDTH), (75, 400))
        self.assertLessEqual(2, SLOTS)

    def test_hidden_goals_do_not_change_the_input(self) -> None:
        shown = _seq([90] * 8, [3, 0, 0, 0, 0, 0, 0, 0])
        hidden = _seq([90] * 8, [3, 0, 0, 0, 3, 3, 3, 3])
        left = mask_remainder(render_full(shown), cut=4, length=8)
        right = mask_remainder(render_full(hidden), cut=4, length=8)
        self.assertTrue(np.array_equal(left, right))
        # The visible goal still changes the picture.
        other = _seq([90] * 8, [0, 0, 0, 0, 3, 3, 3, 3])
        changed = mask_remainder(render_full(other), cut=4, length=8)
        self.assertFalse(np.array_equal(left, changed))

    def test_gray_is_hidden_and_a_zero_is_not_gray(self) -> None:
        seq = _seq([90] * 4, [0, 3, 0, 3])
        image = mask_remainder(render_full(seq), cut=2, length=4)
        goal_ink = np.array(INK["goals"])
        mask = np.array(MASK)
        # Column 1 is a visible hat-trick. Column 3 is hidden.
        visible = image[:, 10:18]
        hidden = image[:, 30:38]
        self.assertTrue(np.any(np.all(visible == goal_ink, axis=-1)))
        self.assertFalse(np.any(np.all(hidden == goal_ink, axis=-1)))
        self.assertTrue(np.any(np.all(hidden == mask, axis=-1)))
        self.assertFalse(np.any(np.all(visible == mask, axis=-1)))

    def test_a_bench_column_is_not_an_unused_slot(self) -> None:
        seq = _seq([0.0, 90.0], [0, 1])
        image = render_full(seq)
        bench = image[:, 0:8]
        unused = image[:, 20:28]
        self.assertGreater(int(np.all(bench == 255, axis=-1).sum()), 20)
        self.assertFalse(np.array_equal(bench, unused))

    def test_repeated_log_line_drops_and_a_double_stays(self) -> None:
        repeated = _seq([10, 10, 70], [1, 1, 0], gw=[6, 6, 7], opponent=["Leeds", "Leeds", "Wolves"])
        collapsed = collapse_repeated_rows(repeated)
        self.assertEqual(len(collapsed.minutes), 2)
        self.assertEqual(collapsed.gw.tolist(), [6, 7])
        double = _seq([80, 70], [1, 0], gw=[33, 33], opponent=["Leeds", "Newcastle"])
        self.assertEqual(len(collapse_repeated_rows(double).minutes), 2)


class ColumnVisionTests(unittest.TestCase):
    def test_model_reads_a_full_minutes_bar_as_a_goal(self) -> None:
        rng = np.random.default_rng(0)
        sequences = []
        for element in range(24):
            minutes = rng.choice(np.array([0, 90], dtype=np.float32), size=12)
            sequences.append(_seq(minutes.tolist(), (minutes >= 60).astype(int).tolist()))
            sequences[-1].element = element
        full = [render_full(seq) for seq in sequences]
        torch.manual_seed(0)
        model = ColumnVision()
        train_vision(model, sequences, full, np.random.default_rng(0), epochs=12)
        probs = column_probabilities(model, sequences[:4], full[:4], 0.5)
        goal_p = probs[0]
        high = []
        low = []
        for row, seq in enumerate(sequences[:4]):
            cut = 6
            for slot in range(cut, len(seq.minutes)):
                chance = float(goal_p[row, slot, 1])
                if seq.minutes[slot] >= 60:
                    high.append(chance)
                else:
                    low.append(chance)
        self.assertGreater(float(np.mean(high)), 0.6)
        self.assertLess(float(np.mean(low)), 0.25)
