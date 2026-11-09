"""Whole-card counts: same row, summed, with a one-gameweek window."""

from __future__ import annotations

import unittest

import numpy as np

from src.models.flashcard_card_score import tally_cards, tally_row
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
