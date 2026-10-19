"""LLM contextual-tag + expected-lineup writer. No network, no scorer."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import src.live.minutes_llm as minutes_llm
from src.live.minutes_llm import (
    PlayerCase,
    apply_completion,
    build_prompt,
    parse_response,
    required_cases,
    validate,
)


def _case(
    pid: int,
    *,
    status: str = "a",
    chance: float | None = None,
    name: str | None = None,
    last: float | None = 60.0,
    history: tuple[float, ...] = (),
    club: str = "Arsenal",
    position: str = "MID",
    focus: bool = True,
) -> PlayerCase:
    return PlayerCase(
        player_id=pid,
        web_name=name or f"P{pid}",
        position=position,
        club=club,
        status=status,
        chance=chance,
        news="",
        last_minutes=last,
        minutes_history=history,
        focus=focus,
    )


def _tag(pid: int, tag: str = "none", note: str = "") -> dict:
    return {"player_id": pid, "tag": tag, "note": note}


def _bootstrap(elements: list[dict]) -> dict:
    return {
        "teams": [{"id": 1, "name": "Arsenal"}],
        "elements": elements,
    }


def _element(
    pid: int,
    status: str = "a",
    chance: object = None,
    *,
    etype: int = 3,
) -> dict:
    return {
        "id": pid,
        "web_name": f"P{pid}",
        "element_type": etype,
        "team": 1,
        "status": status,
        "chance_of_playing_next_round": chance,
        "news": "",
    }


class RuleTest(unittest.TestCase):
    def test_the_prompt_allows_lineups(self) -> None:
        prompt = build_prompt([_case(1)], 6)
        self.assertIn("expected lineups", prompt.lower())
        self.assertIn('"lineups"', prompt)
        self.assertIn("Do not use ask", prompt)
        self.assertNotIn('"xmi"', prompt)

    def test_available_players_with_no_flag_are_left_out_without_expand(self) -> None:
        bootstrap = _bootstrap(
            [
                _element(1, "a", None),
                _element(2, "i", 0),
                _element(3, "d", 75),
                _element(4, "a", 100),
                _element(8, "a", None),
            ]
        )
        cases = required_cases(
            bootstrap, owned={8}, last_minutes={}, expand_club_squads=False
        )
        ids = {case.player_id for case in cases}
        self.assertEqual(ids, {2, 3, 8})
        self.assertTrue(next(case for case in cases if case.player_id == 2).must_be_zero)

    def test_expand_adds_club_squad_mates(self) -> None:
        bootstrap = _bootstrap(
            [
                _element(1, "a", None),
                _element(2, "d", 75),
                _element(8, "a", None),
            ]
        )
        cases = required_cases(bootstrap, owned={8}, last_minutes={})
        ids = {case.player_id for case in cases}
        self.assertEqual(ids, {1, 2, 8})
        self.assertFalse(next(c for c in cases if c.player_id == 1).focus)

    def test_bare_array_and_object_both_parse(self) -> None:
        bare = parse_response('[{"player_id": 1, "tag": "none"}]')
        self.assertEqual(len(bare.players), 1)
        self.assertEqual(bare.lineups, ())
        obj = parse_response(
            json.dumps(
                {
                    "players": [{"player_id": 1, "tag": "none"}],
                    "lineups": [{"club": "Arsenal", "xi": list(range(1, 12)), "bench": [12]}],
                }
            )
        )
        self.assertEqual(len(obj.lineups), 1)
        self.assertEqual(obj.lineups[0].club, "Arsenal")


class GateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = [
            _case(1, history=(90.0, 90.0, 45.0)),
            _case(2, status="s", last=0.0),
            _case(3, chance=0.0, name="Out", last=0.0),
        ]

    def _apply(self, payload: object) -> tuple[minutes_llm.MinutesResult, Path]:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        csv_path = root / "xmi.csv"
        report = root / "minutes.md"
        text = payload if isinstance(payload, str) else json.dumps(payload)
        result = apply_completion(text, self.cases, gw=6, csv_path=csv_path, report_path=report)
        return result, csv_path

    def test_none_writes_rolling_default(self) -> None:
        result, csv_path = self._apply([_tag(1, "none"), _tag(2, "injured"), _tag(3, "injured")])
        self.assertTrue(result.ok)
        text = csv_path.read_text(encoding="utf-8")
        self.assertIn("1,6,75,rolling avg,rolling", text)
        self.assertIn("2,6,0,injured,override:hard_flag", text)

    def test_injury_override_scales_chance(self) -> None:
        cases = [_case(1, chance=50.0, history=(80.0, 80.0, 80.0))]
        result = validate(cases, parse_response(json.dumps([_tag(1, "injured")])))
        self.assertTrue(result.ok)
        self.assertAlmostEqual(float(result.rows[0]["xmi"]), 40.0)
        self.assertEqual(result.rows[0]["source"], "override:injured")

    def test_expected_lineup_sets_xi_bench_out(self) -> None:
        # 11 outfield-shaped ids: one GKP + 10 MID, plus bench and out
        cases = [_case(1, position="GKP", history=(90.0, 90.0, 90.0))]
        cases += [_case(i, history=(70.0, 70.0, 70.0)) for i in range(2, 12)]
        cases.append(_case(12, history=(20.0, 20.0, 20.0)))  # bench
        cases.append(_case(13, history=(60.0, 60.0, 60.0)))  # out
        payload = {
            "players": [_tag(i, "none") for i in range(1, 14)],
            "lineups": [
                {
                    "club": "Arsenal",
                    "xi": list(range(1, 12)),
                    "bench": [12],
                }
            ],
        }
        result = validate(cases, parse_response(json.dumps(payload)))
        self.assertTrue(result.ok, result.errors)
        by_id = {r["player_id"]: r for r in result.rows}
        self.assertEqual(by_id[1]["tag"], "lineup_xi")
        self.assertEqual(by_id[1]["source"], "lineup:xi")
        self.assertGreaterEqual(float(by_id[1]["xmi"]), 65.0)
        self.assertEqual(by_id[12]["tag"], "lineup_bench")
        self.assertEqual(float(by_id[12]["xmi"]), 15.0)
        self.assertEqual(by_id[13]["tag"], "lineup_out")
        self.assertEqual(float(by_id[13]["xmi"]), 0.0)

    def test_injured_beats_lineup_xi(self) -> None:
        cases = [_case(1, position="GKP", history=(90.0, 90.0, 90.0))]
        cases += [_case(i, history=(70.0, 70.0, 70.0)) for i in range(2, 12)]
        payload = {
            "players": [_tag(1, "injured")] + [_tag(i, "none") for i in range(2, 12)],
            "lineups": [{"club": "Arsenal", "xi": list(range(1, 12)), "bench": []}],
        }
        result = validate(cases, parse_response(json.dumps(payload)))
        self.assertTrue(result.ok, result.errors)
        row = next(r for r in result.rows if r["player_id"] == 1)
        self.assertEqual(float(row["xmi"]), 0.0)
        self.assertEqual(row["source"], "override:injured")

    def test_lineup_xi_beats_low_rolling_after_injury(self) -> None:
        """Returning starter: rolling ~15 from injury blanks, but predicted XI → starter floor.

        Without a lineup, tag=none would keep the depressed rolling mean. A predicted
        XI must not inherit that — firm_starter floor (≥65) applies.
        """
        # Last three GWs: cameo / blank / blank → rolling mean 15
        low_history = (45.0, 0.0, 0.0)
        cases = [_case(1, position="GKP", history=(90.0, 90.0, 90.0))]
        cases.append(
            _case(2, name="Returner", history=low_history, last=0.0)
        )
        cases += [_case(i, history=(70.0, 70.0, 70.0)) for i in range(3, 12)]

        # Baseline: no lineup → rolling default
        rolling_only = validate(
            cases, parse_response(json.dumps([_tag(i, "none") for i in range(1, 12)]))
        )
        self.assertTrue(rolling_only.ok, rolling_only.errors)
        roll_row = next(r for r in rolling_only.rows if r["player_id"] == 2)
        self.assertAlmostEqual(float(roll_row["xmi"]), 15.0)
        self.assertEqual(roll_row["source"], "rolling")

        # Predicted XI including the returner
        payload = {
            "players": [_tag(i, "none") for i in range(1, 12)],
            "lineups": [{"club": "Arsenal", "xi": list(range(1, 12)), "bench": []}],
        }
        result = validate(cases, parse_response(json.dumps(payload)))
        self.assertTrue(result.ok, result.errors)
        xi_row = next(r for r in result.rows if r["player_id"] == 2)
        self.assertEqual(xi_row["tag"], "lineup_xi")
        self.assertEqual(xi_row["source"], "lineup:xi")
        self.assertGreaterEqual(float(xi_row["xmi"]), 65.0)
        self.assertGreater(float(xi_row["xmi"]), float(roll_row["xmi"]))

    def test_lineup_needs_one_gkp(self) -> None:
        cases = [_case(i, history=(70.0,)) for i in range(1, 12)]  # all MID
        payload = {
            "players": [_tag(i) for i in range(1, 12)],
            "lineups": [{"club": "Arsenal", "xi": list(range(1, 12))}],
        }
        result = validate(cases, parse_response(json.dumps(payload)))
        self.assertFalse(result.ok)
        self.assertTrue(any("exactly one GKP" in e for e in result.errors))

    def test_legacy_ask_becomes_none_rolling(self) -> None:
        result, csv_path = self._apply(
            [
                {"player_id": 1, "tag": "ask", "note": "unclear"},
                _tag(2, "injured"),
                _tag(3, "injured"),
            ]
        )
        self.assertTrue(result.ok)
        row = next(r for r in result.rows if r["player_id"] == 1)
        self.assertEqual(row["source"], "rolling")

    def test_prose_writes_no_sheet(self) -> None:
        result, csv_path = self._apply("The squad should all start.")
        self.assertFalse(result.ok)
        self.assertFalse(csv_path.exists())


class IsolationTest(unittest.TestCase):
    def test_the_writer_does_not_plan_a_chip_or_call_the_odds_api(self) -> None:
        source = inspect.getsource(minutes_llm)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("the-odds-api", source)


if __name__ == "__main__":
    unittest.main()
