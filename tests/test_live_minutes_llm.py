"""LLM minutes writer. No network, no scorer, no Odds API."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from src.live.minutes_llm import (
    PlayerCase,
    apply_completion,
    build_prompt,
    parse_response,
    required_cases,
    validate,
)
import src.live.minutes_llm as minutes_llm


def _case(
    pid: int,
    *,
    status: str = "a",
    chance: float | None = None,
    name: str | None = None,
) -> PlayerCase:
    return PlayerCase(
        player_id=pid,
        web_name=name or f"P{pid}",
        position="MID",
        club="Arsenal",
        status=status,
        chance=chance,
        news="",
        last_minutes=60.0,
    )


def _answer(pid: int, xmi: float = 70.0, source: str = "llm", note: str = "") -> dict:
    return {"player_id": pid, "xmi": xmi, "source": source, "note": note}


def _bootstrap(elements: list[dict]) -> dict:
    return {
        "teams": [{"id": 1, "name": "Arsenal"}],
        "elements": elements,
    }


def _element(pid: int, status: str = "a", chance: object = None) -> dict:
    return {
        "id": pid,
        "web_name": f"P{pid}",
        "element_type": 3,
        "team": 1,
        "status": status,
        "chance_of_playing_next_round": chance,
        "news": "",
    }


class RuleTest(unittest.TestCase):
    def test_the_prompt_treats_last_minutes_as_context(self) -> None:
        prompt = build_prompt([_case(1)], 6)
        self.assertIn("not the default", prompt)
        self.assertNotIn("rolling", prompt.lower())
        self.assertIn("1 | P1 | MID | Arsenal | a |  | 60 |", prompt)

    def test_available_players_with_no_flag_are_left_out(self) -> None:
        bootstrap = _bootstrap(
            [
                _element(1, "a", None),
                _element(2, "i", 0),
                _element(3, "d", 75),
                _element(4, "a", 100),
                _element(8, "a", None),
            ]
        )
        cases = required_cases(bootstrap, owned={8}, last_minutes={})
        ids = {case.player_id for case in cases}
        self.assertEqual(ids, {2, 3, 8})
        self.assertTrue(next(case for case in cases if case.player_id == 2).must_be_zero)

    def test_a_fenced_array_parses_and_prose_does_not(self) -> None:
        parsed = parse_response("```json\n[{\"player_id\": 1}]\n```")
        self.assertEqual(parsed, [{"player_id": 1}])
        with self.assertRaises(minutes_llm.MinutesLLMError):
            parse_response("Everyone plays 90 minutes.")
        with self.assertRaises(minutes_llm.MinutesLLMError):
            parse_response("```json\n[1, 2\n```")


class GateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = [_case(1), _case(2, status="s"), _case(3, chance=0.0, name="Out")]

    def _apply(self, payload: object) -> tuple[minutes_llm.MinutesResult, Path]:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        csv_path = root / "xmi.csv"
        report = root / "minutes.md"
        text = payload if isinstance(payload, str) else json.dumps(payload)
        result = apply_completion(text, self.cases, gw=6, csv_path=csv_path, report_path=report)
        return result, csv_path

    def test_a_complete_answer_writes_the_sheet(self) -> None:
        result, csv_path = self._apply(
            [_answer(1, 80), _answer(2, 0), _answer(3, 0)]
        )
        self.assertTrue(result.ok)
        text = csv_path.read_text(encoding="utf-8")
        self.assertIn("player_id,gw,xmi,source", text)
        self.assertIn("1,6,80,llm", text)
        self.assertIn("2,6,0,llm", text)

    def test_prose_writes_no_sheet(self) -> None:
        result, csv_path = self._apply("The squad should all start.")
        self.assertFalse(result.ok)
        self.assertFalse(csv_path.exists())

    def test_a_missing_id_writes_no_sheet(self) -> None:
        result, csv_path = self._apply([_answer(1), _answer(2, 0)])
        self.assertFalse(result.ok)
        self.assertFalse(csv_path.exists())
        self.assertTrue(any("missing player_id 3" in error for error in result.errors))

    def test_a_duplicate_id_writes_no_sheet(self) -> None:
        result, csv_path = self._apply(
            [_answer(1), _answer(1, 10), _answer(2, 0), _answer(3, 0)]
        )
        self.assertFalse(result.ok)
        self.assertFalse(csv_path.exists())
        self.assertTrue(any("listed 2 times" in error for error in result.errors))

    def test_minutes_outside_0_to_90_are_rejected(self) -> None:
        for bad in (-5, 95, "NaN"):
            result, csv_path = self._apply(
                [_answer(1, bad), _answer(2, 0), _answer(3, 0)]  # type: ignore[arg-type]
            )
            self.assertFalse(result.ok)
            self.assertFalse(csv_path.exists())

    def test_an_ask_is_not_stored_as_zero(self) -> None:
        result, csv_path = self._apply(
            [
                _answer(1),
                _answer(2, 0),
                {"player_id": 3, "xmi": 0, "source": "ask", "note": "hamstring, no date"},
            ]
        )
        self.assertFalse(result.ok)
        self.assertFalse(csv_path.exists())
        self.assertEqual(result.asks[0]["note"], "hamstring, no date")
        self.assertNotIn(3, {row["player_id"] for row in result.rows})

    def test_a_player_who_cannot_play_cannot_be_given_minutes(self) -> None:
        result, csv_path = self._apply([_answer(1), _answer(2, 90), _answer(3, 0)])
        self.assertFalse(result.ok)
        self.assertFalse(csv_path.exists())
        self.assertTrue(any("cannot play" in error for error in result.errors))
        chance, chance_path = self._apply([_answer(1), _answer(2, 0), _answer(3, 45)])
        self.assertFalse(chance.ok)
        self.assertFalse(chance_path.exists())


class IsolationTest(unittest.TestCase):
    def test_the_writer_does_not_plan_a_chip_or_call_the_odds_api(self) -> None:
        source = inspect.getsource(minutes_llm)
        self.assertNotIn("plan_half", source)
        self.assertNotIn("price_half", source)
        self.assertNotIn("the-odds-api", source)
        self.assertNotIn("deadline.run", source)

    def test_the_stored_snapshot_matches_the_required_set(self) -> None:
        from src.live.deadline import ENTRY_PATH, final_players

        bootstrap = json.loads(Path("/workspace/data/live/bootstrap.json").read_text(encoding="utf-8"))
        entry = json.loads(Path(ENTRY_PATH).read_text(encoding="utf-8"))
        owned = {int(player["id"]) for player in final_players(entry)}
        cases = required_cases(bootstrap, owned, {})
        ids = {case.player_id for case in cases}
        self.assertEqual(len(cases), 227)
        self.assertIn(6, ids)
        self.assertTrue(owned <= ids)
        saliba = next(case for case in cases if case.player_id == 6)
        self.assertTrue(saliba.must_be_zero)
        result = validate(cases[:2], [])
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
