"""Two-path what-if. No season climb and no network."""

from __future__ import annotations

import inspect
import unittest

from src.live.whatif import PathView, Person, WhatIf, render, visible_weeks


def _person(name: str, starting: bool = True, captain: bool = False) -> Person:
    return Person(
        name=name,
        position="MID",
        club="ARS",
        price=50,
        score=4.0,
        starting=starting,
        captain=captain,
    )


def _view(xi: float, bench: float) -> PathView:
    return PathView(
        bank=15,
        hits=0,
        sells=(),
        buys=(),
        squad=(_person("A", captain=True), _person("B", starting=False)),
        outlook=((6, xi, bench), (7, xi, bench)),
    )


class VisibleWeeksTest(unittest.TestCase):
    def test_a_copied_week_stays_out_of_the_search(self) -> None:
        scores = {
            6: {"p": 1.0},
            7: {"p": 2.0},
            8: {"p": 2.0},
        }
        clubs = {6: {"ars"}, 7: {"ars"}, 8: {"ars"}}
        found = visible_weeks((6, 7), scores, clubs, {"p"})
        self.assertEqual(found["future_gws"], [6, 7])
        self.assertEqual(set(found["score_by_gw"]), {6, 7})
        self.assertNotIn(8, found["clubs"])


class RenderTest(unittest.TestCase):
    def test_the_note_is_the_two_priced_weeks(self) -> None:
        note = WhatIf(
            team="ojaminFC",
            minutes_hash="abc",
            line_weeks=(6, 7),
            wildcard=_view(20.0, 3.0),
            hold=_view(14.0, 1.0),
        )
        text = render(note)
        self.assertIn("GW6 XI", text)
        self.assertIn("GW7 XI", text)
        self.assertIn("6.00", text)
        self.assertNotIn("GW8", text)
        self.assertNotIn("play the wildcard", text.lower())
        self.assertNotIn("79", text)

    def test_the_module_does_not_refresh_lines(self) -> None:
        import src.live.whatif as whatif

        source = inspect.getsource(whatif)
        self.assertNotIn("refresh_lines", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("half_plan_scores.csv", source)
