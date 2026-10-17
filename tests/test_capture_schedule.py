"""Capture windows use the response Date header and keep the raw bootstrap."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.eval.capture_schedule import (
    captured_at_from_date_header,
    choose_slot,
    missing_capture_gws,
    run,
)
from src.eval.predictions import capture_official_ep
from src.live.deadline import SEASON


class _Response:
    def __init__(self, payload: dict, date: str | None) -> None:
        self._body = json.dumps(payload).encode()
        self.headers = {} if date is None else {"Date": date}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def _payload(deadline: str) -> dict:
    return {
        "events": [
            {
                "id": 5,
                "deadline_time": "2026-10-03T10:00:00Z",
                "is_current": True,
                "is_next": False,
                "finished": True,
            },
            {
                "id": 6,
                "deadline_time": deadline,
                "is_current": False,
                "is_next": True,
                "finished": False,
            },
        ],
        "elements": [{"id": 7, "ep_next": 4.5, "ep_this": 9.9}],
    }


class CaptureScheduleTest(unittest.TestCase):
    def test_a_missing_date_header_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            captured_at_from_date_header(None)
        with self.assertRaises(RuntimeError):
            captured_at_from_date_header("  ")

    def test_windows_are_the_locked_offsets(self) -> None:
        deadline = datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc)
        self.assertEqual(choose_slot(datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc), deadline), "t24")
        self.assertEqual(choose_slot(datetime(2026, 10, 9, 6, 0, tzinfo=timezone.utc), deadline), "t24")
        self.assertIsNone(choose_slot(datetime(2026, 10, 9, 14, 1, tzinfo=timezone.utc), deadline))
        self.assertIsNone(choose_slot(datetime(2026, 10, 9, 5, 59, tzinfo=timezone.utc), deadline))
        self.assertEqual(choose_slot(datetime(2026, 10, 10, 9, 45, tzinfo=timezone.utc), deadline), "t1")
        self.assertEqual(choose_slot(datetime(2026, 10, 10, 7, 0, tzinfo=timezone.utc), deadline), "t1")
        self.assertIsNone(choose_slot(datetime(2026, 10, 10, 9, 46, tzinfo=timezone.utc), deadline))
        self.assertIsNone(choose_slot(datetime(2026, 10, 10, 6, 59, tzinfo=timezone.utc), deadline))
        self.assertIsNone(choose_slot(datetime(2026, 10, 7, 8, 19, tzinfo=timezone.utc), deadline))

    def test_a_closed_window_without_a_file_fails(self) -> None:
        payload = _payload("2026-10-10T10:00:00Z")

        def opener(*_args: object, **_kwargs: object) -> _Response:
            return _Response(payload, "Sat, 10 Oct 2026 09:50:00 GMT")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError) as raised:
                run(opener, root)
            self.assertIn("6", str(raised.exception))
            directory = root / "data" / "predictions" / SEASON / "gw06"
            directory.mkdir(parents=True)
            (directory / "official_existing.csv").write_text("captured\n", encoding="utf-8")
            self.assertEqual(run(opener, root), 0)

    def test_gameweeks_before_six_are_not_required(self) -> None:
        deadlines = {
            "5": "2026-10-03T10:00:00Z",
            "6": "2026-10-10T10:00:00Z",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            after = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
            self.assertEqual(missing_capture_gws(root, after, deadlines), [6])
            still_open = datetime(2026, 10, 7, 8, 19, tzinfo=timezone.utc)
            self.assertEqual(missing_capture_gws(root, still_open, deadlines), [])

    def test_outside_the_window_writes_nothing(self) -> None:
        payload = _payload("2026-10-10T10:00:00Z")

        def opener(*_args: object, **_kwargs: object) -> _Response:
            return _Response(payload, "Wed, 07 Oct 2026 08:19:00 GMT")

        with tempfile.TemporaryDirectory() as tmp:
            code = run(opener, Path(tmp))
            self.assertEqual(code, 0)
            self.assertEqual(list(Path(tmp).rglob("*")), [])

    def test_the_date_header_is_the_capture_clock(self) -> None:
        payload = _payload("2026-10-10T10:00:00Z")

        def opener(*_args: object, **_kwargs: object) -> _Response:
            return _Response(payload, "Wed, 09 Oct 2026 10:00:00 GMT")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run(opener, root), 0)
            official = list(root.rglob("official_*.csv"))
            raw = list(root.rglob("bootstrap_*.json"))
            self.assertEqual(len(official), 1)
            self.assertEqual(len(raw), 1)
            self.assertIn("20261009T100000Z", official[0].name)
            self.assertIn("t24", official[0].name)
            frame = official[0].read_text(encoding="utf-8")
            self.assertIn("2026-10-09T10:00:00Z", frame)
            self.assertIn("4.5", frame)
            self.assertNotIn("9.9", frame)
            stored = json.loads(raw[0].read_text(encoding="utf-8"))
            self.assertEqual(stored["elements"][0]["ep_this"], 9.9)
            self.assertEqual(run(opener, root), 0)
            self.assertEqual(len(list(root.rglob("official_*.csv"))), 1)

    def test_the_manual_capture_does_not_read_the_local_clock(self) -> None:
        source = inspect.getsource(capture_official_ep)
        self.assertNotIn("datetime.now", source)

    def test_force_write_stays_out_of_the_decision_slot(self) -> None:
        payload = _payload("2026-10-10T10:00:00Z")

        def opener(*_args: object, **_kwargs: object) -> _Response:
            return _Response(payload, "Wed, 07 Oct 2026 08:19:00 GMT")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(run(opener, root, force=True), 0)
            official = list(root.rglob("official_*.csv"))
            self.assertEqual(len(official), 1)
            self.assertEqual(list(root.rglob("slot_*.json")), [])
            with self.assertRaises(RuntimeError):
                run(opener, None, force=True)

    def test_fail_after_write_leaves_the_file_and_raises(self) -> None:
        payload = _payload("2026-10-10T10:00:00Z")

        def opener(*_args: object, **_kwargs: object) -> _Response:
            return _Response(payload, "Wed, 07 Oct 2026 08:19:00 GMT")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError) as raised:
                run(opener, root, force=True, fail_after_write=True)
            self.assertIn("forced failure after the scratch write", str(raised.exception))
            self.assertEqual(len(list(root.rglob("official_*.csv"))), 1)

    def test_an_engine_failure_still_writes_the_official_slot(self) -> None:
        from unittest.mock import patch

        from src.eval.capture_schedule import write_snapshot

        payload = _payload("2026-10-10T10:00:00Z")
        payload["elements"] = [{"id": i, "ep_next": 1.5, "ep_this": 0.0} for i in range(1, 16)]
        payload["teams"] = [{"id": 1, "name": "Arsenal"}]
        captured = datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The live files are absent on CI. Point the checks at a file that exists
            # so the patched engine is the thing that fails.
            present = root / "present.txt"
            present.write_text("x", encoding="utf-8")
            with (
                patch("src.live.t1_inputs.minutes_sheet", return_value=present),
                patch("src.live.t1_inputs.exchange_sheet", return_value=present),
                patch("src.live.deadline.ENTRY_PATH", present),
                patch("src.live.deadline.LOG_PATH", present),
                patch("src.live.deadline.ODDS_PATH", present),
                patch("src.live.deadline.FIXTURES_PATH", present),
                patch(
                    "src.eval.predictions.export_deadline_scores",
                    side_effect=RuntimeError("deadline is not priced"),
                ),
            ):
                write_snapshot(payload, captured, gw=6, slot="t24", root=root, pair_engine=True)
            self.assertEqual(len(list(root.rglob("slot_t24.json"))), 1)
            self.assertEqual(len(list(root.rglob("official_*.csv"))), 1)
            errors = list(root.rglob("shadow_error_*.txt"))
            self.assertEqual(len(errors), 1)
            self.assertIn("deadline is not priced", errors[0].read_text(encoding="utf-8"))

    def test_the_capture_does_not_call_the_odds_api(self) -> None:
        import src.eval.capture_schedule as schedule

        source = inspect.getsource(schedule)
        self.assertNotIn("fetch_odds_api", source)
        self.assertNotIn("ODDS_API_KEY", source)


if __name__ == "__main__":
    unittest.main()
