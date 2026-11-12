"""Whole-card counts: same row, summed, with a one-gameweek window."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_card_score import (
    _paint,
    balanced_score,
    paint_horizon,
    tally_cards,
    tally_row,
    window_count,
)
from src.models.flashcard_squad_trial import cut_at_fraction, sample_cut


class _Season:
    def __init__(self) -> None:
        self.n_fixtures = 10
        self.gw = np.arange(1, 11)


class _Card:
    def __init__(self, gw: np.ndarray, present: np.ndarray) -> None:
        self.gw = gw
        self.present = present
        self.n_players = present.shape[0]


class CardScoreTests(unittest.TestCase):
    def test_training_cuts_are_the_four_breaks(self) -> None:
        card = _Season()
        seen = {sample_cut(card, np.random.default_rng(seed)) for seed in range(40)}
        expected = {cut_at_fraction(card, fraction) for fraction in (0.2, 0.4, 0.6, 0.8)}
        self.assertEqual(seen, expected)

    def test_one_week_away_counts_in_the_window_only(self) -> None:
        gameweeks = np.array([24, 25, 26])
        exact = tally_row(gameweeks, np.array([1, 0, 0]), np.array([0, 1, 0]))
        self.assertEqual(exact["exact"], 0)
        self.assertEqual(exact["actual_within"], 1)
        self.assertEqual(exact["painted_within"], 1)
        far = tally_row(gameweeks, np.array([1, 0, 0]), np.array([0, 0, 1]))
        self.assertEqual(far["actual_within"], 0)
        self.assertEqual(far["painted_within"], 0)

    def test_the_same_appearance_is_inside_the_window(self) -> None:
        hit = tally_row(np.array([24, 25]), np.array([1, 0]), np.array([1, 0]))
        self.assertEqual(hit["exact"], 1)
        self.assertEqual(hit["actual_within"], 1)

    def test_a_bar_on_another_row_does_not_count(self) -> None:
        gw = np.array([24, 25, 0])
        present = np.ones((2, 3), dtype=bool)
        labels = np.zeros((1, 2, 3))
        labels[0, 0, 0] = 1
        painted = np.zeros((1, 2, 3), dtype=np.int64)
        painted[0, 1, 0] = 1
        totals = tally_cards([_Card(gw, present)], labels, painted, 19)
        self.assertEqual(totals["actual"], 1)
        self.assertEqual(totals["painted"], 1)
        self.assertEqual(totals["exact"], 0)
        self.assertEqual(totals["actual_within"], 0)

    def test_rows_add_up(self) -> None:
        gw = np.array([24, 25])
        present = np.ones((2, 2), dtype=bool)
        labels = np.zeros((1, 2, 2))
        labels[0, 0, 0] = 1
        labels[0, 1, 1] = 1
        painted = np.zeros((1, 2, 2), dtype=np.int64)
        painted[0, 0, 0] = 1
        painted[0, 1, 1] = 1
        totals = tally_cards([_Card(gw, present)], labels, painted, 19)
        self.assertEqual(totals["actual"], 2)
        self.assertEqual(totals["exact"], 2)
        self.assertEqual(totals["actual_within"], 2)

    def test_twice_the_bars_scores_below_a_matched_card(self) -> None:
        matched = balanced_score(
            {"actual": 100, "painted": 100, "exact": 40, "actual_within": 50, "painted_within": 50}
        )
        doubled = balanced_score(
            {"actual": 100, "painted": 200, "exact": 80, "actual_within": 90, "painted_within": 100}
        )
        halved = balanced_score(
            {"actual": 100, "painted": 50, "exact": 40, "actual_within": 40, "painted_within": 40}
        )
        self.assertAlmostEqual(matched["volume"], 1.0)
        self.assertAlmostEqual(doubled["volume"], 0.5)
        self.assertAlmostEqual(halved["volume"], 0.5)
        self.assertGreater(matched["score"], doubled["score"])
        self.assertGreater(matched["score"], halved["score"])
        self.assertAlmostEqual(matched["score"], 2 * 40 / 200)

    def test_an_empty_window_is_a_large_error(self) -> None:
        labels = torch.zeros(1, 1, 4)
        labels[0, 0, 0] = 1
        labels[0, 0, 3] = 2
        mask = torch.ones(1, 1, 4)
        matched = torch.zeros(1, 1, 4, 4)
        matched[..., 0] = 0.5
        matched[..., 1] = 0.5
        empty = torch.zeros(1, 1, 4, 4)
        empty[..., 0] = 1
        self.assertAlmostEqual(float(window_count(matched, labels, mask)), 0.0)
        self.assertAlmostEqual(float(window_count(empty, labels, mask)), 1.0)

    def test_a_matching_total_on_the_wrong_rows_is_an_error(self) -> None:
        labels = torch.zeros(1, 2, 4)
        labels[0, 1, 0] = 1
        labels[0, 1, 1] = 1
        mask = torch.ones(1, 2, 4)
        swapped = torch.zeros(1, 2, 4, 4)
        swapped[..., 0] = 1.0
        swapped[0, 0, :, 0] = 0.5
        swapped[0, 0, :, 1] = 0.5
        self.assertAlmostEqual(float(window_count(swapped, labels, mask)), 1.0)

    def test_the_window_places_its_expected_bars(self) -> None:
        probs = np.zeros((5, 4))
        probs[..., 0] = 0.95
        probs[..., 1] = 0.05
        probs[2, 0] = 0.20
        probs[2, 2] = 0.80
        mask = np.ones(5, dtype=bool)
        painted = paint_horizon(probs, mask)
        self.assertEqual(int((painted > 0).sum()), 1)
        self.assertEqual(int(painted[2]), 2)
        self.assertTrue(np.all(painted[:2] == 0))
        card = _paint(
            probs.reshape(1, 1, 5, 4),
            np.zeros((1, 1, 5)),
            [_Card(np.array([5, 6, 7, 8, 9]), mask.reshape(1, 5))],
            [0],
        )
        self.assertEqual(int((card > 0).sum()), 1)

    def test_each_row_paints_its_own_count(self) -> None:
        probs = np.zeros((1, 2, 1, 4))
        probs[0, 0, 0] = (0.05, 0.15, 0.70, 0.10)
        probs[0, 1, 0] = (0.40, 0.60, 0.00, 0.00)
        minutes = np.zeros((1, 2, 1))
        present = np.ones((2, 1), dtype=bool)
        painted = _paint(probs, minutes, [_Card(np.array([5]), present)], [0])
        self.assertEqual(int(painted[0, 0, 0]), 2)
        self.assertEqual(int(painted[0, 1, 0]), 1)
