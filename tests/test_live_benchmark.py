"""Priors for the 2026/27 benchmark must not use later weeks of that season."""

from __future__ import annotations

import unittest

import pandas as pd

from src.live.benchmark import assign_prior_id, stamp_unmatched_priors
from src.models.season_climb_ft import SquadState, choose_transfers, run_ft_season
from src.models.xp_engine import add_player_priors
from src.teams import norm_team


def _row(**kwargs: object) -> dict:
    base = {
        "player_id": "p",
        "gw": 1,
        "date": "2026-08-21",
        "position": "GKP",
        "team_norm": "arsenal",
        "fixture_id": "2026-08-21:arsenal:chelsea",
        "minutes": 90.0,
        "total_points": 6.0,
        "xG": 0.0,
        "xA": 0.0,
        "defcon_hit": 0.0,
    }
    base.update(kwargs)
    return base


class BenchmarkPriorTest(unittest.TestCase):
    def test_promoted_club_names(self) -> None:
        self.assertEqual(norm_team("Hull City"), "hull")
        self.assertEqual(norm_team("Hull"), "hull")
        self.assertEqual(norm_team("Coventry City"), "coventry")
        self.assertEqual(norm_team("Coventry"), "coventry")

    def test_prior_id_uses_opta_code_not_web_name(self) -> None:
        codes = {111: 28, 222: 90}
        self.assertEqual(assign_prior_id("7", 111, codes), "2026-27:28")
        self.assertEqual(assign_prior_id("8", 999, codes), "2025-26:8")
        self.assertEqual(assign_prior_id("9", None, codes), "2025-26:9")

    def test_fill_from_ignores_later_weeks(self) -> None:
        preseason = pd.DataFrame(
            [_row(player_id="old", gw=-1, date="2026-05-01", minutes=30.0)]
        )
        frame = pd.DataFrame(
            [
                _row(player_id="old", gw=-1, date="2026-05-01", minutes=30.0),
                _row(player_id="new", gw=1, date="2026-08-21", minutes=10.0),
                _row(
                    player_id="new",
                    gw=2,
                    date="2026-08-28",
                    minutes=90.0,
                    fixture_id="2026-08-28:arsenal:chelsea",
                ),
            ]
        )
        filled = add_player_priors(frame, fill_from=preseason)
        debut = filled.loc[filled["player_id"] == "new"].sort_values("gw")
        self.assertEqual(float(debut["xmi"].iloc[0]), 30.0)
        leaked = add_player_priors(frame)
        leaked_xmi = float(
            leaked.loc[leaked["player_id"] == "new"].sort_values("gw")["xmi"].iloc[0]
        )
        self.assertNotEqual(leaked_xmi, 30.0)

    def test_unmatched_entry_uses_matched_preseason_prior(self) -> None:
        frame = pd.DataFrame(
            [
                _row(
                    player_id="2026-27:1",
                    gw=-2,
                    date="2026-05-01",
                    minutes=90.0,
                    xG=0.4,
                ),
                _row(
                    player_id="2026-27:1",
                    gw=1,
                    date="2026-08-21",
                    minutes=90.0,
                    xmi=80.0,
                    exp_points=5.0,
                    roll3_points=5.0,
                    exp_xG=0.3,
                    exp_xA=0.1,
                    exp_defcon_hit=0.0,
                    exp_team_xg=1.5,
                    exp_team_xa=1.0,
                ),
                _row(
                    player_id="2026-27:2",
                    gw=1,
                    date="2026-08-21",
                    minutes=90.0,
                    xmi=99.0,
                    exp_points=99.0,
                    roll3_points=99.0,
                    exp_xG=9.0,
                    exp_xA=9.0,
                    exp_defcon_hit=1.0,
                    exp_team_xg=1.5,
                    exp_team_xa=1.0,
                    fixture_id="2026-08-21:chelsea:arsenal",
                    team_norm="chelsea",
                ),
            ]
        )
        stamped, n_debut = stamp_unmatched_priors(frame)
        self.assertEqual(n_debut, 1)
        debut = stamped.loc[stamped["player_id"] == "2026-27:2"].iloc[0]
        self.assertEqual(float(debut["xmi"]), 80.0)
        self.assertEqual(float(debut["exp_xG"]), 0.3)
        self.assertAlmostEqual(float(debut["share_xG"]), 0.3 / 1.5)

    def test_over_club_cap_hold_is_not_kept(self) -> None:
        rows = []
        n = 0

        def add(pos: str, club: str, score: float, owned: bool) -> str:
            nonlocal n
            n += 1
            pid = f"p{n}"
            rows.append(
                {
                    "player_id": pid,
                    "position": pos,
                    "team_norm": club,
                    "value": 45,
                    "score_xp": score,
                    "eligible": True,
                    "total_points": 0.0,
                    "minutes": 90.0,
                }
            )
            return pid

        owned = []
        owned += [add("GKP", "a", 3.0, True), add("GKP", "b", 2.0, True)]
        for i, club in enumerate("cdefg"):
            owned.append(add("DEF", club, 4.0 - 0.1 * i, True))
        for i in range(4):
            owned.append(add("MID", "city", 6.0 - 0.1 * i, True))
        owned.append(add("MID", "other", 3.0, True))
        for i, club in enumerate("hij"):
            owned.append(add("FWD", club, 4.0 - 0.1 * i, True))
        spare = add("MID", "leeds", 0.1, False)
        pool = pd.DataFrame(rows)
        state = SquadState(purchase={pid: 45 for pid in owned}, bank=100, ft=1)
        new_state, n_tx, _hits = choose_transfers(
            state,
            pool,
            "score_xp",
            1,
            [1],
            {1: set(pool["player_id"])},
        )
        clubs = pool.set_index("player_id")["team_norm"]
        city = sum(1 for pid in new_state.ids() if clubs[pid] == "city")
        self.assertEqual(n_tx, 1)
        self.assertLessEqual(city, 3)
        self.assertIn(spare, new_state.ids())

    def test_opening_squad_is_kept_for_the_first_week(self) -> None:
        spec = (
            [("GKP", "a"), ("GKP", "b")]
            + [("DEF", club) for club in "cdefg"]
            + [("MID", club) for club in "hijkl"]
            + [("FWD", club) for club in "mno"]
        )
        rows = []
        ids = []
        for i, (pos, club) in enumerate(spec, start=1):
            pid = f"p{i}"
            ids.append(pid)
            for gw in (1, 2):
                rows.append(
                    {
                        "gw": gw,
                        "player_id": pid,
                        "player_name": pid,
                        "position": pos,
                        "team": club,
                        "team_norm": club,
                        "value": 50,
                        "eligible": True,
                        "score_xp": 5.0,
                        "total_points": 2.0,
                        "minutes": 90.0,
                    }
                )
        frame = pd.DataFrame(rows)
        opening = SquadState(purchase={pid: 50 for pid in ids}, bank=250, ft=0)
        weekly = run_ft_season(
            frame, {"xp": "score_xp"}, [1, 2], roster=frame, opening=opening
        )
        self.assertEqual(list(weekly["gw"]), [1, 2])
        self.assertEqual(int(weekly.loc[weekly["gw"] == 1, "n_transfers"].iloc[0]), 0)
        self.assertEqual(int(weekly.loc[weekly["gw"] == 2, "n_transfers"].iloc[0]), 0)
