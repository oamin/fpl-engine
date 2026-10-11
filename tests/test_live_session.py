"""Numagent live session. No network and no official ledger."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.live.freeze_week import sha256_file
from src.live.news_packets import content_sha256
from src.live.session import (
    FROZEN_LINES,
    Lineup,
    NumagentPick,
    SessionError,
    assert_exchange_folder,
    book_is_ready,
    compile_minutes,
    lineup_points,
    lock_from_pick,
    resolve_exchange,
    run_numagent,
    score_deadline,
    write_step_scores,
)
from src.live.t1_inputs import FROZEN_MINUTES


def _bootstrap() -> dict:
    def element(pid: int, name: str) -> dict:
        return {
            "id": pid,
            "web_name": name,
            "element_type": 3,
            "team": 1,
            "status": "a",
            "news": "",
            "chance_of_playing_next_round": None,
        }

    return {
        "teams": [{"id": 1, "name": "Arsenal"}],
        "elements": [element(1, "One"), element(2, "Two"), element(3, "Three")],
    }


def _packet(folder: Path) -> None:
    headline = "Two is out"
    body = "Two has an injury and will miss the match."
    url = "https://www.bbc.co.uk/sport/football/100"
    published = "2026-10-16T12:00:00Z"
    raw = {
        "packet_id": "bbc_sport:gw02:two-out",
        "source": "bbc_sport",
        "url": url,
        "published_at_utc": published,
        "club": "Arsenal",
        "player_ids": [2],
        "headline": headline,
        "body": body,
        "sha256": content_sha256(
            headline=headline, body=body, url=url, published_at_utc=published
        ),
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "two.json").write_text(json.dumps(raw), encoding="utf-8")


def _ready(folder: Path, marker: str = "t1") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gw_lines.csv").write_text(f"gw,marker\n7,{marker}\n", encoding="utf-8")
    (folder / "betfair_to_score.json").write_text("{}", encoding="utf-8")
    (folder / "outrights_ranks.json").write_text("{}", encoding="utf-8")


def _lineup() -> Lineup:
    ids = tuple(f"2026-27:{i}" for i in range(1, 16))
    return Lineup(ids, ids[:11], ids[0], ids[1], ids[11:])


def _pick() -> NumagentPick:
    lineup = _lineup()
    return NumagentPick(
        chip=None,
        decision=lineup,
        hold_starting_11=lineup.starting_11,
        hold_captain="2026-27:2",
        ft1_starting_11=lineup.starting_11,
        ft1_captain="2026-27:3",
        ft1_move="hold",
        decision_xi_c_score_xp=30.0,
        hold_xi_c_score_xp=20.0,
        ft1_xi_c_score_xp=25.0,
        step_scores={7: {"2026-27:1": 1.0}, 8: {"2026-27:1": 2.0}},
    )


def _official(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["player_id,gw,official_xp,source_field,event_role,captured_at,deadline"]
    values = {1: 5.0, 2: 1.0, 3: 3.0}
    for i in range(1, 16):
        rows.append(
            f"2026-27:{i},7,{values.get(i, 1.0)},ep_next,next,2026-10-16T18:00:00Z,2026-10-17T10:00:00Z"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class MinutesCompile(unittest.TestCase):
    def _layout(self, root: Path) -> tuple[Path, Path, Path]:
        gw = root / "data" / "predictions" / "2026-27" / "gw02"
        _packet(gw / "news_packets")
        note = gw / "string_sources"
        note.mkdir(parents=True)
        (note / "src_one.json").write_text(
            '{"headline": "One has an injury and is ruled out."}\n',
            encoding="utf-8",
        )
        history = root / "history"
        history.mkdir()
        (history / "gw01.csv").write_text(
            "player_id,gw,minutes\n1,1,90\n2,1,90\n",
            encoding="utf-8",
        )
        boot = root / "bootstrap.json"
        boot.write_text(json.dumps(_bootstrap()), encoding="utf-8")
        return gw, history, boot

    def test_packets_set_minutes_and_the_dossier_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _gw, history, boot = self._layout(root)
            sheet = compile_minutes(
                2,
                deadline_utc="2026-10-17T10:00:00Z",
                root=root,
                bootstrap_path=boot,
                history_dir=history,
                ledger=root / "ledger.jsonl",
            )
            with sheet.open(encoding="utf-8") as handle:
                rows = {int(row["player_id"]): row for row in csv.DictReader(handle)}
        self.assertEqual(float(rows[1]["xmi"]), 90.0)
        self.assertEqual(rows[1]["tag"], "rolling avg")
        self.assertEqual(float(rows[2]["xmi"]), 0.0)
        self.assertEqual(rows[2]["tag"], "injured")
        self.assertEqual(float(rows[3]["xmi"]), 0.0)
        self.assertEqual(list(rows[1]), ["player_id", "gw", "xmi", "tag"])

    def test_a_second_write_leaves_the_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _gw, history, boot = self._layout(root)
            sheet = compile_minutes(
                2,
                deadline_utc="2026-10-17T10:00:00Z",
                root=root,
                bootstrap_path=boot,
                history_dir=history,
                ledger=root / "ledger.jsonl",
            )
            before = sheet.read_bytes()
            with self.assertRaises(SessionError):
                compile_minutes(
                    2,
                    deadline_utc="2026-10-17T10:00:00Z",
                    root=root,
                    bootstrap_path=boot,
                    history_dir=history,
                    ledger=root / "ledger.jsonl",
                )
            self.assertEqual(sheet.read_bytes(), before)

    def test_a_locked_gameweek_is_not_recompiled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _gw, history, boot = self._layout(root)
            ledger = root / "ledger.jsonl"
            ledger.write_text('{"gw": 2}\n', encoding="utf-8")
            with self.assertRaises(SessionError):
                compile_minutes(
                    2,
                    deadline_utc="2026-10-17T10:00:00Z",
                    root=root,
                    bootstrap_path=boot,
                    history_dir=history,
                    ledger=ledger,
                )
            self.assertFalse((root / "data" / "predictions" / "2026-27" / "gw02" / "xmi_t1.csv").exists())

    def test_a_string_sources_path_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _gw, history, boot = self._layout(root)
            with self.assertRaises(SessionError):
                compile_minutes(
                    2,
                    deadline_utc="2026-10-17T10:00:00Z",
                    root=root,
                    bootstrap_path=boot,
                    history_dir=history,
                    ledger=root / "ledger.jsonl",
                    sources_path=root / "data" / "predictions" / "2026-27" / "gw02" / "string_sources",
                )
            self.assertFalse((root / "data" / "predictions" / "2026-27" / "gw02" / "xmi_t1.csv").exists())

    def test_the_frozen_minutes_file_is_not_the_sheet(self) -> None:
        self.assertTrue(FROZEN_MINUTES.name.endswith(".csv"))
        self.assertNotEqual(FROZEN_MINUTES.name, "xmi_t1.csv")


class ExchangeBook(unittest.TestCase):
    def test_a_ready_t1_book_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            folder_t1 = root / "data" / "predictions" / "2026-27" / "gw07" / "betfair_t1"
            older = root / "data" / "predictions" / "2026-27" / "gw07" / "betfair_20261008"
            _ready(folder_t1, "fresh")
            _ready(older, "stale")
            calls: list[Path] = []

            def pull(path: Path) -> int:
                calls.append(path)
                return 0

            with mock.patch(
                "src.live.betfair_props.discover_betfair_artifacts",
                side_effect=AssertionError("fallback"),
            ):
                found = resolve_exchange(7, root=root, pull=pull)
            self.assertEqual(found, folder_t1)
            self.assertEqual(calls, [])
            self.assertIn("fresh", (found / "gw_lines.csv").read_text(encoding="utf-8"))

    def test_a_missing_book_pulls_only_into_betfair_t1(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            older = root / "data" / "predictions" / "2026-27" / "gw07" / "betfair_20261008"
            _ready(older, "stale")

            def pull(path: Path) -> int:
                self.assertEqual(path.name, "betfair_t1")
                self.assertNotEqual(path, older)
                _ready(path, "pulled")
                return 0

            found = resolve_exchange(7, root=root, pull=pull)
            self.assertEqual(found.name, "betfair_t1")
            self.assertIn("pulled", (found / "gw_lines.csv").read_text(encoding="utf-8"))
            self.assertIn("stale", (older / "gw_lines.csv").read_text(encoding="utf-8"))

    def test_a_failed_pull_removes_a_partial_book(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def pull(path: Path) -> int:
                path.mkdir(parents=True, exist_ok=True)
                (path / "gw_lines.csv").write_text("partial\n", encoding="utf-8")
                return 2

            with self.assertRaises(SessionError):
                resolve_exchange(7, root=root, pull=pull)
            target = root / "data" / "predictions" / "2026-27" / "gw07" / "betfair_t1"
            self.assertFalse((target / "gw_lines.csv").exists())
            self.assertFalse(book_is_ready(target))

    def test_live_and_older_folders_are_refused(self) -> None:
        with self.assertRaises(SessionError):
            assert_exchange_folder(FROZEN_LINES.parent)
        with self.assertRaises(SessionError):
            assert_exchange_folder(Path("gw07") / "betfair_20261008")

    def test_the_score_call_keeps_choice_empty(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            book = Path(folder) / "betfair_t1"
            _ready(book)
            with mock.patch("src.live.scorer.price_half", return_value="priced") as price:
                with mock.patch(
                    "src.live.betfair_props.discover_betfair_artifacts",
                    side_effect=AssertionError("fallback"),
                ):
                    found = score_deadline(
                        gw=7,
                        logs=None,  # type: ignore[arg-type]
                        odds=None,  # type: ignore[arg-type]
                        fixtures=[],
                        bootstrap={},
                        state=None,
                        minutes={},
                        played=None,
                        artifacts_dir=book,
                        choice=None,
                    )
            self.assertEqual(found, "priced")
            self.assertIsNone(price.call_args.kwargs["choice"])
            self.assertEqual(price.call_args.kwargs["artifacts_dir"], book)
            with self.assertRaises(SessionError):
                score_deadline(
                    gw=7,
                    logs=None,  # type: ignore[arg-type]
                    odds=None,  # type: ignore[arg-type]
                    fixtures=[],
                    bootstrap={},
                    state=None,
                    minutes={},
                    played=None,
                    artifacts_dir=book,
                    choice={"2026-27:1": 1.0},
                )


class ScoresAndLock(unittest.TestCase):
    def test_every_week_is_stored_and_a_second_stamp_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = write_step_scores(
                7,
                {7: {"2026-27:1": 1.5}, 8: {"2026-27:1": 2.5}},
                stamp="20261017T090000Z",
                root=root,
            )
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["column"], "score_xp")
            self.assertEqual(rows[1]["column"], "forecast_xp")
            self.assertEqual(rows[1]["gw"], "8")
            with self.assertRaises(FileExistsError):
                write_step_scores(
                    7,
                    {7: {"2026-27:1": 9.0}},
                    stamp="20261017T090000Z",
                    root=root,
                )

    def test_lineup_points_count_the_captain_twice(self) -> None:
        scores = {"2026-27:1": 5.0, "2026-27:2": 1.0}
        self.assertEqual(lineup_points(["2026-27:1", "2026-27:2"], "2026-27:1", scores), 11.0)

    def _lock_inputs(self, root: Path) -> dict:
        official = root / "data" / "predictions" / "2026-27" / "gw07" / "official_20261016T180000Z_t1.csv"
        _official(official)
        minutes = root / "xmi_t1.csv"
        minutes.write_text("player_id,gw,xmi,tag\n1,7,90,rolling avg\n", encoding="utf-8")
        lines = root / "gw_lines.csv"
        lines.write_text("gw\n7\n", encoding="utf-8")
        scores = root / "scores.csv"
        scores.write_text("player_id,gw,score,column\n", encoding="utf-8")
        return {
            "official_path": official,
            "minutes_path": minutes,
            "lines_path": lines,
            "scores_path": scores,
            "ledger": root / "week_freeze.jsonl",
            "root": root,
        }

    def test_a_late_lock_writes_no_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = self._lock_inputs(root)
            with self.assertRaises(SessionError):
                lock_from_pick(
                    _pick(),
                    gw=7,
                    deadline_utc="2026-10-17T10:00:00Z",
                    frozen_at_utc="2026-10-17T10:00:00Z",
                    **kwargs,
                )
            self.assertFalse(kwargs["ledger"].exists())

    def test_a_second_gameweek_row_leaves_the_ledger_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kwargs = self._lock_inputs(root)
            lock_from_pick(
                _pick(),
                gw=7,
                deadline_utc="2026-10-17T10:00:00Z",
                frozen_at_utc="2026-10-17T09:00:00Z",
                **kwargs,
            )
            before = kwargs["ledger"].read_bytes()
            row = json.loads(before.decode("utf-8"))
            self.assertEqual(row["forecast_expected_points"]["decision_xi_c_ep_next"], 22.0)
            self.assertEqual(row["forecast_expected_points"]["hold_xi_c_ep_next"], 18.0)
            self.assertEqual(row["forecast_expected_points"]["ft1_xi_c_ep_next"], 20.0)
            self.assertEqual(row["forecast_expected_points"]["decision_xi_c_score_xp"], 30.0)
            self.assertEqual(row["forecast_expected_points"]["hold_xi_c_score_xp"], 20.0)
            self.assertEqual(row["forecast_expected_points"]["ft1_xi_c_score_xp"], 25.0)
            self.assertIsNone(row["realised"]["evaluated_at_utc"])
            self.assertEqual(row["provenance"]["official_sha256"], sha256_file(kwargs["official_path"]))
            from src.live.freeze_week import ProvenanceViolationError

            with self.assertRaises(ProvenanceViolationError):
                lock_from_pick(
                    _pick(),
                    gw=7,
                    deadline_utc="2026-10-17T10:00:00Z",
                    frozen_at_utc="2026-10-17T09:30:00Z",
                    **kwargs,
                )
            self.assertEqual(kwargs["ledger"].read_bytes(), before)

    def test_a_run_stores_the_manifest_without_locking(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gw = root / "data" / "predictions" / "2026-27" / "gw07"
            _ready(gw / "betfair_t1")
            (gw / "xmi_t1.csv").parent.mkdir(parents=True, exist_ok=True)
            (gw / "xmi_t1.csv").write_text("player_id,gw,xmi,tag\n1,7,90,rolling avg\n", encoding="utf-8")
            boot = root / "bootstrap.json"
            boot.write_text("{}", encoding="utf-8")

            def score_fn(**_kwargs: object) -> NumagentPick:
                return _pick()

            payload = run_numagent(
                7,
                deadline_utc="2026-10-17T10:00:00Z",
                root=root,
                lock=False,
                ledger=root / "week_freeze.jsonl",
                pull=lambda _path: 1,
                score_fn=score_fn,
                stamp="20261017T090000Z",
                bootstrap_path=boot,
            )
            self.assertFalse(payload["locked"])
            self.assertIsNone(payload["realised"])
            self.assertFalse((root / "week_freeze.jsonl").exists())
            stored = json.loads((gw / "session_numagent.json").read_text(encoding="utf-8"))
            self.assertFalse(stored["locked"])
            self.assertIsNone(stored["realised"])
            text = (gw / "20261017T090000Z.csv").read_text(encoding="utf-8")
            self.assertIn("forecast_xp", text)
            self.assertIn("score_xp", text)

    def test_a_late_run_writes_no_ledger_row(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gw = root / "data" / "predictions" / "2026-27" / "gw07"
            _ready(gw / "betfair_t1")
            gw.mkdir(parents=True, exist_ok=True)
            (gw / "xmi_t1.csv").write_text("player_id,gw,xmi,tag\n1,7,90,rolling avg\n", encoding="utf-8")
            boot = root / "bootstrap.json"
            boot.write_text("{}", encoding="utf-8")
            ledger = root / "week_freeze.jsonl"
            with self.assertRaises(SessionError):
                run_numagent(
                    7,
                    deadline_utc="2026-10-17T10:00:00Z",
                    root=root,
                    lock=True,
                    frozen_at_utc="2026-10-17T10:00:00Z",
                    ledger=ledger,
                    pull=lambda _path: 1,
                    score_fn=lambda **_kwargs: _pick(),
                    stamp="20261017T090100Z",
                    bootstrap_path=boot,
                )
            self.assertFalse(ledger.exists())


if __name__ == "__main__":
    unittest.main()
