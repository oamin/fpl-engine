"""Live score_xp and forecast_xp read xmi_t1 and the T-1 Exchange sheet."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.live.lines import LINES_PATH
from src.live.t1_inputs import (
    FROZEN_LINES,
    FROZEN_MINUTES,
    LiveScoreInputError,
    exchange_sheet,
    minutes_sheet,
    refuse_frozen,
    require_score_inputs,
)


class SheetPaths(unittest.TestCase):
    def test_the_compiled_sheet_is_not_the_frozen_minutes_file(self) -> None:
        path = minutes_sheet(6)
        self.assertEqual(path.name, "xmi_t1.csv")
        self.assertNotEqual(path.resolve(), FROZEN_MINUTES)
        self.assertTrue(path.is_file())

    def test_the_exchange_sheet_is_the_t1_folder(self) -> None:
        path = exchange_sheet(6)
        self.assertEqual(path.parent.name, "betfair_t1")
        self.assertEqual(path.name, "gw_lines.csv")
        self.assertNotEqual(path.resolve(), FROZEN_LINES)

    def test_the_stored_sheet_is_the_live_book(self) -> None:
        from src.live.t1_inputs import live_score_book

        book = live_score_book(6, None)
        self.assertEqual(book, exchange_sheet(6))
        self.assertTrue(book.is_file())
        earlier = Path("data/predictions/2026-27/gw06/betfair_20261008/gw_lines.csv")
        self.assertNotEqual(book.resolve(), earlier.resolve())

    def test_a_missing_t1_sheet_does_not_open_the_earlier_pull(self) -> None:
        from src.live.t1_inputs import live_score_book

        missing = Path("/tmp/betfair_t1/gw_lines.csv")
        earlier = Path("data/predictions/2026-27/gw06/betfair_20261008/gw_lines.csv")
        self.assertTrue(earlier.is_file())
        with mock.patch("src.live.t1_inputs.exchange_sheet", return_value=missing):
            self.assertIsNone(live_score_book(6, None))

    def test_a_missing_t1_sheet_is_an_error(self) -> None:
        missing = Path("/tmp/betfair_t1/gw_lines.csv")
        with mock.patch("src.live.t1_inputs.exchange_sheet", return_value=missing):
            with self.assertRaises(LiveScoreInputError) as raised:
                require_score_inputs(6)
        self.assertIn("betfair_t1", str(raised.exception))
        self.assertIn("not used", str(raised.exception))

    def test_frozen_paths_are_refused(self) -> None:
        with self.assertRaises(LiveScoreInputError):
            refuse_frozen(FROZEN_MINUTES)
        with self.assertRaises(LiveScoreInputError):
            refuse_frozen(LINES_PATH)


class DiscoverPrefersT1(unittest.TestCase):
    def test_betfair_t1_beats_a_newer_sibling(self) -> None:
        from src.live import betfair_props as bp

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gw_dir = root / "gw06"
            older = gw_dir / "betfair_t1"
            newer = gw_dir / "betfair_later"
            older.mkdir(parents=True)
            newer.mkdir()
            for path in (older, newer):
                (path / "betfair_meta.json").write_text("{}", encoding="utf-8")
            (newer / "gw_lines.csv").write_text("gw\n", encoding="utf-8")
            with mock.patch.object(bp, "PREDICTIONS", root):
                found = bp.discover_betfair_artifacts(6)
            self.assertEqual(found, older)

    def test_an_unnamed_book_is_the_t1_sheet_when_that_file_exists(self) -> None:
        from src.live import betfair_props as bp

        with tempfile.TemporaryDirectory() as folder:
            sheet = Path(folder) / "gw_lines.csv"
            sheet.write_text("gw\n6\n", encoding="utf-8")
            with mock.patch("src.live.t1_inputs.exchange_sheet", return_value=sheet):
                found = bp.resolve_live_book(6, None)
                named_frozen = bp.resolve_live_book(6, LINES_PATH)
            self.assertEqual(found, sheet)
            self.assertEqual(named_frozen, sheet)


class ExportReadsTheUpdatedFiles(unittest.TestCase):
    def test_export_stops_before_any_older_sheet(self) -> None:
        from src.eval.predictions import export_deadline_scores

        missing = Path("/tmp/betfair_t1/gw_lines.csv")
        with (
            mock.patch("src.live.t1_inputs.exchange_sheet", return_value=missing),
            mock.patch("src.live.deadline.load_odds_frame") as odds,
        ):
            with self.assertRaises(LiveScoreInputError) as raised:
                export_deadline_scores()
            odds.assert_not_called()
        text = str(raised.exception)
        self.assertIn("betfair_t1", text)
        self.assertNotIn("betfair_20261008", text)
        self.assertNotIn("xmi_gw6", text)

    def test_export_reads_the_paths_it_is_given(self) -> None:
        from src.eval.predictions import export_deadline_scores
        from src.live.deadline import load_minutes, load_odds_frame

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            minutes = root / "xmi_t1.csv"
            minutes.write_text("player_id,gw,xmi\n411,6,45\n", encoding="utf-8")
            sheet_dir = root / "betfair_t1"
            sheet_dir.mkdir()
            sheet = sheet_dir / "gw_lines.csv"
            sheet.write_text(
                "Date,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,gw\n",
                encoding="utf-8",
            )
            seen: dict[str, object] = {}
            real_odds = load_odds_frame
            real_minutes = load_minutes

            def spy_odds(odds_path: Path, live_path: Path | None):
                seen["lines"] = Path(live_path)  # type: ignore[arg-type]
                return real_odds(odds_path, live_path)

            def spy_minutes(path: Path, gw: int):
                seen["minutes"] = Path(path)
                return real_minutes(path, gw)

            def stop(**kwargs: object) -> None:
                seen["artifacts"] = kwargs.get("artifacts_dir")
                seen["minute_map"] = kwargs.get("minutes")
                raise RuntimeError("stop after inputs")

            with (
                mock.patch("src.eval.predictions.load_odds_frame", spy_odds),
                mock.patch("src.eval.predictions.load_minutes", spy_minutes),
                mock.patch("src.eval.predictions.readiness", return_value=(True, ())),
                mock.patch("src.eval.predictions.price_half", side_effect=stop),
            ):
                with self.assertRaises(RuntimeError):
                    export_deadline_scores(minutes_path=minutes, lines_path=sheet)
            self.assertEqual(seen["lines"], sheet)
            self.assertEqual(seen["minutes"], minutes)
            self.assertEqual(seen["artifacts"], sheet_dir)
            minute_map = seen["minute_map"]
            self.assertIsInstance(minute_map, dict)
            self.assertEqual(minute_map.get("2026-27:411"), 45.0)

    def test_the_forecast_folder_is_the_sheet_folder(self) -> None:
        from src.live.deadline import DeadlineError, collect

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            minutes = root / "xmi_t1.csv"
            minutes.write_text("player_id,gw,xmi\n411,6,45\n", encoding="utf-8")
            sheet_dir = root / "betfair_t1"
            sheet_dir.mkdir()
            sheet = sheet_dir / "gw_lines.csv"
            sheet.write_text(
                "Date,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,source,gw\n"
                "10/10/2026,Arsenal,Leeds,1.4,5.0,9.0,betfair,6\n",
                encoding="utf-8",
            )
            seen: dict[str, object] = {}

            def stop(**kwargs: object) -> None:
                seen["artifacts"] = kwargs.get("artifacts_dir")
                seen["minutes"] = kwargs.get("minutes")
                raise DeadlineError("stop after inputs")

            with (
                mock.patch("src.live.deadline.readiness", return_value=(True, ())),
                mock.patch("src.live.scorer.load_ep_next", return_value={}),
                mock.patch("src.live.scorer.price_half", side_effect=stop),
            ):
                with self.assertRaises(DeadlineError):
                    collect(
                        minutes_path=minutes,
                        live_path=sheet,
                        trial_path=root / "no-meta.json",
                    )
            self.assertEqual(seen["artifacts"], sheet_dir)
            minutes_map = seen["minutes"]
            self.assertIsInstance(minutes_map, dict)
            self.assertEqual(minutes_map.get("2026-27:411"), 45.0)


if __name__ == "__main__":
    unittest.main()
