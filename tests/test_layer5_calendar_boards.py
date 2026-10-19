"""Calendar boards: strength fill when FD missing; as-of hist."""

from __future__ import annotations

import unittest

import pandas as pd

from src.strategy.layer5_calendar_boards import (
    _bookie_allowed,
    _pots_from_strength,
    project_week_calendar,
)


class CalendarBoards(unittest.TestCase):
    def test_pots_from_strength_home_away_and_clip(self) -> None:
        strengths = {
            "Arsenal": {"att": 2.0, "defn": 0.8, "n": 10},
            "Burnley": {"att": 0.9, "defn": 1.8, "n": 10},
        }
        xg_h, xgc_h, xcs_h = _pots_from_strength(
            strengths, "Arsenal", opp="Burnley", home=True
        )
        xg_a, xgc_a, _ = _pots_from_strength(
            strengths, "Arsenal", opp="Burnley", home=False
        )
        self.assertGreaterEqual(xg_h, 0.4)
        self.assertLessEqual(xg_h, 3.0)
        self.assertGreaterEqual(xgc_h, 0.4)
        self.assertLessEqual(xgc_h, 3.0)
        self.assertGreaterEqual(xcs_h, 0.05)
        self.assertLessEqual(xcs_h, 0.55)
        self.assertGreater(xg_h, xg_a)
        self.assertLess(xgc_h, xgc_a)

    def test_project_week_strength_fill_when_fd_empty(self) -> None:
        players = [
            (101, "A1", "Arsenal", "MID"),
            (102, "A2", "Arsenal", "DEF"),
            (201, "B1", "Burnley", "MID"),
            (202, "B2", "Burnley", "DEF"),
        ]
        hist_rows = []
        for gw in (1, 2, 3):
            for el, name, team, pos in players:
                hist_rows.append(
                    {
                        "element": el,
                        "name": name,
                        "position": pos,
                        "team": team,
                        "team_n": team,
                        "GW": gw,
                        "fixture": 100 + gw,
                        "minutes": 90.0,
                        "goals_scored": 1.0 if team == "Arsenal" else 0.0,
                        "assists": 0.0,
                        "bonus": 0.0,
                        "yellow_cards": 0.0,
                        "red_cards": 0.0,
                        "saves": 0.0,
                        "defensive_contribution": 5.0,
                        "was_home": team == "Arsenal",
                        "team_h_score": 2.0,
                        "team_a_score": 0.0,
                        "date_iso": pd.Timestamp("2025-08-01").date(),
                    }
                )
        hist = pd.DataFrame(hist_rows)
        d = pd.Timestamp("2025-09-01").date()
        merged_rows = list(hist.to_dict("records"))
        for el, name, team, pos in players:
            merged_rows.append(
                {
                    "element": el,
                    "name": name,
                    "position": pos,
                    "team": team,
                    "team_n": team,
                    "GW": 5,
                    "fixture": 500,
                    "minutes": 0.0,
                    "goals_scored": 0.0,
                    "assists": 0.0,
                    "bonus": 0.0,
                    "yellow_cards": 0.0,
                    "red_cards": 0.0,
                    "saves": 0.0,
                    "defensive_contribution": 0.0,
                    "was_home": team == "Arsenal",
                    "team_h_score": float("nan"),
                    "team_a_score": float("nan"),
                    "date_iso": d,
                }
            )
        merged = pd.DataFrame(merged_rows)
        fd = pd.DataFrame(columns=["team_n", "date_iso", "xg", "xcs", "xgc"])

        board, n_fd, n_fill, src = project_week_calendar(
            hist,
            merged,
            fd,
            decision_gw=4,
            target_gw=5,
            alpha_g=0.96,
            alpha_a=0.90,
        )
        self.assertFalse(board.empty)
        self.assertEqual(n_fd, 0)
        self.assertGreaterEqual(n_fill, 1)
        self.assertEqual(src, "strength_fill")
        self.assertIn("score_xp_v2", board.columns)
        self.assertEqual(set(board["element"]), {101, 102, 201, 202})

    def test_bookie_window_matches_l4_depth(self) -> None:
        self.assertTrue(_bookie_allowed(12, 12))
        self.assertTrue(_bookie_allowed(12, 13))
        self.assertFalse(_bookie_allowed(12, 14))  # t+2: beyond BOOKIE_DEPTH=2
        self.assertFalse(_bookie_allowed(12, 15))
        self.assertFalse(_bookie_allowed(12, 19))

    def test_beyond_t2_ignores_fd_even_when_present(self) -> None:
        """t+2+ must be strength_fill even if FD rows exist for that fixture."""
        players = [
            (101, "A1", "Arsenal", "MID"),
            (102, "A2", "Arsenal", "DEF"),
            (201, "B1", "Burnley", "MID"),
            (202, "B2", "Burnley", "DEF"),
        ]
        hist_rows = []
        for gw in (1, 2, 3):
            for el, name, team, pos in players:
                hist_rows.append(
                    {
                        "element": el,
                        "name": name,
                        "position": pos,
                        "team": team,
                        "team_n": team,
                        "GW": gw,
                        "fixture": 100 + gw,
                        "minutes": 90.0,
                        "goals_scored": 1.0 if team == "Arsenal" else 0.0,
                        "assists": 0.0,
                        "bonus": 0.0,
                        "yellow_cards": 0.0,
                        "red_cards": 0.0,
                        "saves": 0.0,
                        "defensive_contribution": 5.0,
                        "was_home": team == "Arsenal",
                        "team_h_score": 2.0,
                        "team_a_score": 0.0,
                        "date_iso": pd.Timestamp("2025-08-01").date(),
                    }
                )
        hist = pd.DataFrame(hist_rows)
        d = pd.Timestamp("2025-09-20").date()
        merged_rows = list(hist.to_dict("records"))
        for el, name, team, pos in players:
            merged_rows.append(
                {
                    "element": el,
                    "name": name,
                    "position": pos,
                    "team": team,
                    "team_n": team,
                    "GW": 8,
                    "fixture": 800,
                    "minutes": 0.0,
                    "goals_scored": 0.0,
                    "assists": 0.0,
                    "bonus": 0.0,
                    "yellow_cards": 0.0,
                    "red_cards": 0.0,
                    "saves": 0.0,
                    "defensive_contribution": 0.0,
                    "was_home": team == "Arsenal",
                    "team_h_score": float("nan"),
                    "team_a_score": float("nan"),
                    "date_iso": d,
                }
            )
        merged = pd.DataFrame(merged_rows)
        # Tempting FD rows — must be ignored at decision=5, target=8 (beyond t+2)
        fd = pd.DataFrame(
            [
                {"team_n": "Arsenal", "date_iso": d, "xg": 9.0, "xcs": 0.9, "xgc": 0.1},
                {"team_n": "Burnley", "date_iso": d, "xg": 0.1, "xcs": 0.05, "xgc": 3.0},
            ]
        )
        board, n_fd, n_fill, src = project_week_calendar(
            hist,
            merged,
            fd,
            decision_gw=5,
            target_gw=8,
            alpha_g=0.96,
            alpha_a=0.90,
        )
        self.assertFalse(board.empty)
        self.assertEqual(n_fd, 0)
        self.assertGreaterEqual(n_fill, 1)
        self.assertEqual(src, "strength_fill")


if __name__ == "__main__":
    unittest.main()
