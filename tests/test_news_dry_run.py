"""Ownership parsing and the differential gate. No price_half call."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from src.live.news_dry_run import (
    OWN_LIMIT,
    apply_scores,
    differential_frame,
    is_differential,
    parse_ownership,
)
from src.str_agent.differential import judge_differential, prepare_differential

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "data" / "live" / "bootstrap.json"


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("2026-27:1", "GKP", 14.9, eligible=True, minutes=90, n_prior=5, score=4.0),
            _row("2026-27:2", "GKP", 15.0, eligible=True, minutes=90, n_prior=5, score=6.0),
            _row("2026-27:3", "DEF", None, eligible=True, minutes=90, n_prior=5, score=5.0),
            _row("2026-27:4", "DEF", 1.0, eligible=False, minutes=10, n_prior=5, score=1.0),
            _row("2026-27:5", "MID", 9.0, eligible=True, minutes=44.9, n_prior=5, score=3.0),
            _row("2026-27:6", "FWD", 0.4, eligible=True, minutes=60, n_prior=2, score=7.0),
            _row("2026-27:7", "FWD", 0.4, eligible=True, minutes=60, n_prior=3, score=2.5),
        ]
    )


def _row(
    player_id: str,
    position: str,
    ownership: float | None,
    *,
    eligible: bool,
    minutes: float,
    n_prior: int,
    score: float,
) -> dict:
    return {
        "player_id": player_id,
        "position": position,
        "team_norm": "arsenal",
        "value": 50,
        "eligible": eligible,
        "minutes": minutes,
        "n_prior": n_prior,
        "ownership": ownership,
        "name": player_id,
        "score_xp": score,
    }


class OwnershipTest(unittest.TestCase):
    def test_unparseable_ownership_is_missing(self) -> None:
        self.assertIsNone(parse_ownership(None))
        self.assertIsNone(parse_ownership(""))
        self.assertIsNone(parse_ownership("  "))
        self.assertIsNone(parse_ownership("n/a"))
        self.assertIsNone(parse_ownership(True))
        self.assertEqual(parse_ownership("14.9"), 14.9)
        self.assertEqual(parse_ownership("15%"), 15.0)
        self.assertEqual(parse_ownership(0), 0.0)

    def test_the_cut_is_strict(self) -> None:
        self.assertTrue(is_differential(14.999))
        self.assertFalse(is_differential(15.0))
        self.assertFalse(is_differential(None))

    def test_differential_frame_drops_the_owned_bypass(self) -> None:
        kept = set(differential_frame(_pool())["player_id"])
        self.assertEqual(kept, {"2026-27:1", "2026-27:7"})

    def test_a_missing_score_stays_missing(self) -> None:
        scored = apply_scores(_pool(), {"2026-27:1": 3.2})
        row = scored.loc[scored["player_id"] == "2026-27:2"].iloc[0]
        self.assertTrue(pd.isna(row["score_xp"]))
        hit = scored.loc[scored["player_id"] == "2026-27:1"].iloc[0]
        self.assertEqual(float(hit["score_xp"]), 3.2)

    def test_haaland_is_not_a_differential(self) -> None:
        bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
        haaland = next(row for row in bootstrap["elements"] if row["web_name"] == "Haaland")
        percent = parse_ownership(haaland["selected_by_percent"])
        self.assertIsNotNone(percent)
        assert percent is not None
        self.assertGreaterEqual(percent, OWN_LIMIT)
        self.assertFalse(is_differential(percent))


class DifferentialContextTest(unittest.TestCase):
    def test_context_lists_only_the_low_owned(self) -> None:
        pack = prepare_differential()
        self.assertNotIn("score_xp", pack.context)
        self.assertNotIn("ep_next", pack.context)
        bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
        haaland = next(row for row in bootstrap["elements"] if row["web_name"] == "Haaland")
        self.assertNotIn(str(haaland["id"]), pack.directory)
        self.assertGreaterEqual(len(pack.directory), 15)
        for percent in pack.ownership.values():
            self.assertLess(percent, OWN_LIMIT)

    def test_a_high_owned_name_fails_closed(self) -> None:
        pack = prepare_differential()
        low = next(iter(pack.directory))
        decision = {
            "chip_played": None,
            "squad_15": [low],
            "starting_11": [low],
            "captain": low,
            "vice_captain": low,
            "bench_order": [],
            "transfers_in": ["411"],
            "transfers_out": [],
        }
        judged = judge_differential(pack, decision)
        self.assertFalse(judged["is_legal"])
        self.assertTrue(any("fresh squad" in error for error in judged["errors"]))
