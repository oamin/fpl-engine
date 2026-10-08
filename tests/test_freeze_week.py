"""Weekly freeze ledger: schema lock and immutable forecasts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.live import freeze_week as fw


def _ids(prefix: str, n: int) -> list[str]:
    return [f"2026-27:{prefix}{i}" for i in range(n)]


def _valid_row(**overrides: object) -> dict:
    squad = _ids("p", 15)
    xi = squad[:11]
    bench = squad[11:]
    row = {
        "gw": 6,
        "deadline_utc": "2026-10-10T10:00:00Z",
        "frozen_at_utc": "2026-10-10T09:00:00Z",
        "provenance": {
            "official_capture_file": "official_x.csv",
            "official_sha256": "a" * 64,
            "minutes_sha256": "b" * 64,
            "odds_source": "betfair",
            "odds_manifest_sha256": "c" * 64,
            "pool_forecast_sha256": "d" * 64,
        },
        "decision": {
            "chip_played": "wildcard",
            "squad_15": squad,
            "starting_11": xi,
            "captain": xi[0],
            "vice_captain": xi[1],
            "bench_order": bench,
        },
        "counterfactuals": {
            "hold_starting_11": xi,
            "hold_captain": xi[0],
            "ft1_starting_11": xi,
            "ft1_captain": xi[0],
            "ft1_move": "sell A buy B",
        },
        "forecast_expected_points": {
            "decision_xi_c_ep_next": 100.0,
            "decision_xi_c_score_xp": 99.0,
            "hold_xi_c_ep_next": 60.0,
            "ft1_xi_c_ep_next": 80.0,
        },
        "realised": fw.empty_realised(),
    }
    row.update(overrides)
    return row


class FreezeWeek(unittest.TestCase):
    def test_bar_matches_decision_spec(self) -> None:
        bar = fw.bar_definition()
        self.assertEqual(bar["start_gw"], 6)
        self.assertEqual(bar["end_gw"], 25)
        self.assertEqual(bar["n_weeks"], 20)
        self.assertEqual(bar["first_read_gw"], 26)

    def test_validate_and_append(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "week_freeze.jsonl"
            row = fw.build_freeze_row(
                gw=6,
                deadline_utc="2026-10-10T10:00:00Z",
                frozen_at_utc="2026-10-10T09:00:00Z",
                provenance=_valid_row()["provenance"],
                decision=_valid_row()["decision"],
                counterfactuals=_valid_row()["counterfactuals"],
                forecast_expected_points=_valid_row()["forecast_expected_points"],
            )
            fw.write_deadline_freeze(row, path)
            loaded = fw.read_ledger(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["gw"], 6)
            with self.assertRaises(fw.ProvenanceViolationError):
                fw.write_deadline_freeze(row, path)

    def test_attach_realised_keeps_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "week_freeze.jsonl"
            row = _valid_row()
            fw.validate_freeze_row(row, require_null_realised=True)
            fw.write_deadline_freeze(row, path)
            filled = fw.attach_realised_outcomes(
                6,
                {
                    "evaluated_at_utc": "2026-10-13T12:00:00Z",
                    "actual_decision_xi_pts": 70.0,
                    "actual_hold_xi_pts": 55.0,
                    "actual_ft1_xi_pts": 60.0,
                    "net_gain_vs_hold": 15.0,
                    "net_gain_vs_ft1": 10.0,
                    "pool_spearman_rho": 0.2,
                    "clean_sheet_brier": 0.1,
                },
                path,
            )
            self.assertEqual(filled["forecast_expected_points"]["hold_xi_c_ep_next"], 60.0)
            self.assertEqual(filled["realised"]["net_gain_vs_ft1"], 10.0)
            with self.assertRaises(fw.ProvenanceViolationError):
                fw.attach_realised_outcomes(
                    6,
                    filled["realised"],
                    path,
                )

    def test_bad_squad_size_rejected(self) -> None:
        row = _valid_row()
        row["decision"]["squad_15"] = row["decision"]["squad_15"][:14]
        with self.assertRaises(fw.FreezeError):
            fw.validate_freeze_row(row)

    def test_sha256_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.txt"
            path.write_text("hello", encoding="utf-8")
            digest = fw.sha256_file(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(fw.sha256_json({"b": 1, "a": 2}), fw.sha256_json({"a": 2, "b": 1}))


if __name__ == "__main__":
    unittest.main()
