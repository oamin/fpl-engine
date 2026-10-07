"""Live scorer. The formula is xp_on_pot. No network and no Odds API call."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.live.deadline import (
    BOOTSTRAP_PATH,
    ENTRY_PATH,
    LOG_PATH,
    bench_for_transfers,
    final_players,
    player_key,
    run,
)
from src.live.scorer import (
    build_pool,
    copy_note,
    deadline_shares,
    plan_deadline,
    price_half,
    priced_gameweeks,
    roster_from_bootstrap,
    score_on_line,
    score_steps,
    scores_for_horizon,
)
from src.models.open_horizon import side_pot, xp_on_pot
from src.models.season_climb_ft import SquadState
from src.models.xp_engine import compute_xp


def _pot(home: float, draw: float, away: float) -> dict[str, float]:
    return side_pot(home, draw, away, 1.8, 2.0, is_home=True)


class FormulaTest(unittest.TestCase):
    def test_xp_on_pot_matches_compute_xp_for_one_midfielder(self) -> None:
        pot = {
            "lam_scored": 1.6,
            "lam_assist": 1.2,
            "e_total": 2.8,
            "p_cs_mkt": 0.30,
        }
        frame = pd.DataFrame(
            [
                {
                    "player_id": "2026-27:1",
                    "gw": 6,
                    "date": "2026-10-10",
                    "position": "MID",
                    "xmi": 90.0,
                    "share_xG": 0.25,
                    "share_xA": 0.10,
                    "exp_defcon_hit": 0.4,
                    "value": 80.0,
                    **pot,
                }
            ]
        )
        scored = compute_xp(frame)
        direct = xp_on_pot(
            position="MID",
            xmi=90.0,
            share_xg=0.25,
            share_xa=0.10,
            exp_defcon_hit=0.4,
            fwd_goal_scale=float(scored["fwd_goal_scale"].iloc[0]),
            pot=pot,
        )
        self.assertAlmostEqual(float(scored["xp"].iloc[0]), direct, places=6)
        self.assertAlmostEqual(float(scored["fwd_goal_scale"].iloc[0]), 1.0, places=6)

    def test_minutes_and_the_pot_both_move_the_score(self) -> None:
        pot = _pot(1.4, 4.5, 7.0)
        other = _pot(3.2, 3.4, 2.2)
        quiet = score_on_line(
            position="MID",
            xmi=0.0,
            share_xg=0.3,
            share_xa=0.1,
            exp_defcon_hit=0.2,
            pot=pot,
        )
        played = score_on_line(
            position="MID",
            xmi=90.0,
            share_xg=0.3,
            share_xa=0.1,
            exp_defcon_hit=0.2,
            pot=pot,
        )
        away = score_on_line(
            position="MID",
            xmi=90.0,
            share_xg=0.3,
            share_xa=0.1,
            exp_defcon_hit=0.2,
            pot=other,
        )
        self.assertLess(quiet, played)
        self.assertNotAlmostEqual(played, away, places=6)


class ShareTest(unittest.TestCase):
    def test_the_stub_keeps_the_earlier_share_and_drops_its_own_zero(self) -> None:
        rows = []
        for gw, xg in ((1, 0.4), (2, 0.2)):
            rows.append(
                {
                    "date": f"2026-08-{20 + gw:02d}",
                    "gw": gw,
                    "player_id": 7,
                    "team_norm": "arsenal",
                    "position": "MID",
                    "minutes": 90.0,
                    "xG": xg,
                    "xA": 0.1,
                    "total_points": 5.0,
                    "defcon_raw": 0.0,
                }
            )
        shares = deadline_shares(pd.DataFrame(rows), 6)
        self.assertEqual(len(shares), 1)
        row = shares.iloc[0]
        self.assertEqual(row["player_id"], player_key(7))
        self.assertEqual(int(row["n_prior"]), 2)
        self.assertAlmostEqual(float(row["exp_xG"]), 0.3, places=6)
        self.assertAlmostEqual(float(row["share_xG"]), 1.0, places=6)

    def test_live_minutes_replace_the_rolling_prior(self) -> None:
        pool = pd.DataFrame(
            [
                {
                    "player_id": "2026-27:7",
                    "position": "MID",
                    "team_norm": "arsenal",
                    "minutes": 0.0,
                    "xmi": 90.0,
                    "share_xG": 0.4,
                    "share_xA": 0.1,
                    "exp_defcon_hit": 0.0,
                }
            ]
        )
        pot = _pot(1.5, 4.0, 6.0)
        scores = score_steps(pool, {(6, "arsenal"): [pot, _pot(1.1, 8.0, 15.0)]}, [6])
        played = score_on_line(
            position="MID",
            xmi=90.0,
            share_xg=0.4,
            share_xa=0.1,
            exp_defcon_hit=0.0,
            pot=pot,
        )
        benched = score_on_line(
            position="MID",
            xmi=0.0,
            share_xg=0.4,
            share_xa=0.1,
            exp_defcon_hit=0.0,
            pot=pot,
        )
        self.assertAlmostEqual(scores[6]["2026-27:7"], benched, places=6)
        self.assertNotAlmostEqual(scores[6]["2026-27:7"], played, places=6)


class HorizonTest(unittest.TestCase):
    def test_a_missing_line_repeats_the_last_priced_week(self) -> None:
        line = {6: {"a": 1.0}, 7: {"a": 2.0}}
        steps, copies = scores_for_horizon(line, [6, 7, 8])
        self.assertEqual(steps[8], steps[7])
        self.assertEqual(copies, [(8, 7)])
        self.assertIn("GW8 repeats GW7 and has no 1X2 of its own", copy_note(copies))

    def test_the_walk_stops_at_the_first_club_week_without_a_line(self) -> None:
        names = {1: "Arsenal", 2: "Chelsea"}
        fixtures = []
        for gw in (6, 7, 8, 9):
            fixtures.append(
                {
                    "event": gw,
                    "kickoff_time": "2026-10-10T15:00:00Z",
                    "team_h": 1,
                    "team_a": 2,
                }
            )
        pots = {
            (6, "arsenal"): [{}],
            (6, "chelsea"): [{}],
            (7, "arsenal"): [{}],
            (7, "chelsea"): [{}],
            (9, "arsenal"): [{}],
            (9, "chelsea"): [{}],
        }
        self.assertEqual(priced_gameweeks(fixtures, pots, names, 6, 19), [6, 7])


class PlanTest(unittest.TestCase):
    def test_a_higher_bench_this_week_is_bench_boost_and_not_a_climb(self) -> None:
        high = ["g1", "d1", "d2", "d3", "m1", "m2", "m3", "m4", "m5", "f1", "f2"]
        bench = ["g2", "d4", "d5", "f3"]
        positions = {
            "g1": "GKP",
            "g2": "GKP",
            "d1": "DEF",
            "d2": "DEF",
            "d3": "DEF",
            "d4": "DEF",
            "d5": "DEF",
            "m1": "MID",
            "m2": "MID",
            "m3": "MID",
            "m4": "MID",
            "m5": "MID",
            "f1": "FWD",
            "f2": "FWD",
            "f3": "FWD",
        }
        rows = []
        for index, pid in enumerate(high + bench):
            rows.append(
                {
                    "player_id": pid,
                    "position": positions[pid],
                    "team_norm": f"c{index:02d}",
                    "value": 50,
                    "eligible": True,
                    "minutes": 90.0,
                    "share_xG": 0.0,
                    "share_xA": 0.0,
                    "exp_defcon_hit": 0.0,
                    "total_points": 0.0,
                    "score_xp": 0.0,
                }
            )
        pool = pd.DataFrame(rows)
        clubs = {gw: {f"c{i:02d}" for i in range(15)} for gw in range(6, 20)}

        def week_scores(bench_score: float) -> dict[str, float]:
            scores = {pid: 50.0 for pid in high}
            scores.update({pid: bench_score for pid in bench})
            return scores

        steps = {6: week_scores(8.0), 7: week_scores(1.0), 8: week_scores(1.0)}
        state = SquadState(purchase={pid: 50 for pid in high + bench}, bank=0, ft=1)
        plan, _weeks = plan_deadline(6, state, pool, steps, clubs, played={1: "triple_captain"})
        self.assertEqual(plan.chip, "bench_boost")
        self.assertEqual(bench_for_transfers(plan, 6), 6)

    def test_the_live_choice_follows_ep_next(self) -> None:
        from src.live.scorer import ScorerError, live_choice

        high = ["g1", "d1", "d2", "d3", "m1", "m2", "m3", "m4", "m5", "f1", "f2"]
        bench = ["g2", "d4", "d5", "f3"]
        positions = {
            "g1": "GKP",
            "g2": "GKP",
            "d1": "DEF",
            "d2": "DEF",
            "d3": "DEF",
            "d4": "DEF",
            "d5": "DEF",
            "m1": "MID",
            "m2": "MID",
            "m3": "MID",
            "m4": "MID",
            "m5": "MID",
            "f1": "FWD",
            "f2": "FWD",
            "f3": "FWD",
        }
        rows = []
        for index, pid in enumerate(high + bench):
            rows.append(
                {
                    "player_id": pid,
                    "position": positions[pid],
                    "team_norm": f"c{index:02d}",
                    "value": 50,
                    "eligible": True,
                    "minutes": 90.0,
                    "share_xG": 0.0,
                    "share_xA": 0.0,
                    "exp_defcon_hit": 0.0,
                    "total_points": 0.0,
                    "score_xp": 50.0 if pid in high else 8.0,
                }
            )
        pool = pd.DataFrame(rows)
        clubs = {gw: {f"c{i:02d}" for i in range(15)} for gw in range(6, 20)}
        engine = {pid: 50.0 for pid in high}
        engine.update({pid: 8.0 for pid in bench})
        steps = {6: engine, 7: engine, 8: engine}
        choice = {pid: 50.0 for pid in high}
        choice.update({pid: 0.0 for pid in bench})
        state = SquadState(purchase={pid: 50 for pid in high + bench}, bank=0, ft=1)
        _plan, weeks = plan_deadline(
            6, state, pool, steps, clubs, played={1: "triple_captain"}, choice=choice
        )
        by_gw = {int(row.gw): row for row in weeks}
        self.assertEqual(by_gw[6].held.bench_xp, 0.0)
        self.assertGreater(by_gw[6].held.xi_xp, 0.0)
        self.assertEqual(by_gw[7].held.xi_xp, 0.0)
        chosen = live_choice({"a", "b"}, {"a", "b", "c"}, {"a": 1.0, "b": 10.0})
        self.assertEqual(max(("a", "b"), key=chosen.get), "b")
        self.assertNotIn("c", chosen)
        with self.assertRaises(ScorerError):
            plan_deadline(
                6, state, pool, steps, clubs, played={1: "triple_captain"}, choice={"g2": 1.0}
            )

    def test_the_paired_comparison_ignores_which_score_drives(self) -> None:
        from src.live.scorer import paired_live_rows

        frame = pd.DataFrame(
            {
                "player_id": ["a", "b", "c", "d"],
                "gw": [6, 6, 6, 6],
                "score_xp": [1.0, None, 3.0, 4.0],
                "ep_next": [2.0, 5.0, None, 1.5],
                "choice_field": ["ep_next", "ep_next", "ep_next", "ep_next"],
            }
        )
        kept = paired_live_rows(frame)
        self.assertEqual(kept["player_id"].tolist(), ["a", "d"])
        self.assertEqual(kept["score_xp_minus_ep_next"].tolist(), [-1.0, 2.5])
        relabelled = frame.copy()
        relabelled["choice_field"] = "score_xp"
        again = paired_live_rows(relabelled)
        self.assertEqual(
            kept["score_xp_minus_ep_next"].tolist(),
            again["score_xp_minus_ep_next"].tolist(),
        )

    def test_a_mixed_stamp_is_not_a_decision_pair(self) -> None:
        from src.live.scorer import ScorerError, write_shadow_log

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "engine.csv"
            official = root / "official.csv"
            engine.write_text(
                "player_id,gw,score,created_at\na,6,1.0,2026-10-07T07:04:50Z\n",
                encoding="utf-8",
            )
            official.write_text(
                "player_id,gw,official_xp,captured_at\na,6,2.0,2026-10-07T08:04:06Z\n",
                encoding="utf-8",
            )
            with self.assertRaises(ScorerError):
                write_shadow_log(root / "shadow.csv", engine, official)
            official.write_text(
                "player_id,gw,official_xp,captured_at\na,6,2.0,2026-10-07T07:04:50Z\n",
                encoding="utf-8",
            )
            same = write_shadow_log(root / "shadow_ok.csv", engine, official)
            self.assertEqual(same["captured_at"].tolist(), ["2026-10-07T07:04:50Z"])
            self.assertEqual(float(same["score_xp"].iloc[0]) - float(same["ep_next"].iloc[0]), -1.0)


def _world() -> tuple[dict, list, pd.DataFrame, pd.DataFrame]:
    positions = (
        [(1, "GKP"), (2, "GKP")]
        + [(i, "DEF") for i in range(3, 8)]
        + [(i, "MID") for i in range(8, 13)]
        + [(i, "FWD") for i in range(13, 16)]
    )
    types = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    bootstrap = {
        "teams": [{"id": i, "name": f"Club{i}"} for i in range(1, 16)],
        "elements": [
            {
                "id": pid,
                "element_type": types[pos],
                "team": pid,
                "now_cost": 50,
            }
            for pid, pos in positions
        ],
    }
    pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
    fixtures = []
    for gw, day in ((6, "2026-10-10"), (7, "2026-10-17"), (8, "2026-10-24")):
        for home, away in pairs:
            fixtures.append(
                {
                    "event": gw,
                    "kickoff_time": f"{day}T15:00:00Z",
                    "team_h": home,
                    "team_a": away,
                }
            )
    odds_rows = []
    for day in ("10/10/2026", "17/10/2026"):
        for home, away in pairs:
            odds_rows.append(
                {
                    "Date": day,
                    "HomeTeam": f"Club{home}",
                    "AwayTeam": f"Club{away}",
                    "AvgH": 1.7,
                    "AvgD": 3.8,
                    "AvgA": 5.0,
                    "Avg>2.5": 1.9,
                    "Avg<2.5": 1.9,
                }
            )
    logs = []
    for pid, pos in positions:
        for gw in (1, 2, 3):
            logs.append(
                {
                    "date": f"2026-08-{20 + gw:02d}",
                    "gw": gw,
                    "player_id": pid,
                    "team_norm": f"club{pid}",
                    "position": pos,
                    "minutes": 90.0,
                    "xG": 0.2,
                    "xA": 0.05,
                    "total_points": 2.0,
                    "defcon_raw": 0.0,
                }
            )
    return bootstrap, fixtures, pd.DataFrame(odds_rows), pd.DataFrame(logs)


class PriceHalfTest(unittest.TestCase):
    def test_two_priced_weeks_copy_onto_the_third_slot(self) -> None:
        bootstrap, fixtures, odds, logs = _world()
        owned = [player_key(i) for i in range(1, 16)]
        state = SquadState(purchase={pid: 50 for pid in owned}, bank=0, ft=1)
        minutes = {pid: 90.0 for pid in owned}
        result = price_half(
            gw=6,
            logs=logs,
            odds=odds,
            fixtures=fixtures,
            bootstrap=bootstrap,
            state=state,
            minutes=minutes,
            played={1: "triple_captain"},
        )
        self.assertEqual(result.line_weeks, (6, 7))
        self.assertEqual(result.horizon_weeks, (6, 7, 8))
        self.assertEqual(result.step_scores[8], result.step_scores[7])
        self.assertIn("GW8 repeats GW7 and has no 1X2 of its own", result.copy_note)
        self.assertIsNotNone(result.plan)

        zeroed = dict(minutes)
        zeroed[player_key(8)] = 0.0
        quieter = price_half(
            gw=6,
            logs=logs,
            odds=odds,
            fixtures=fixtures,
            bootstrap=bootstrap,
            state=state,
            minutes=zeroed,
            played={1: "triple_captain"},
        )
        self.assertLess(
            quieter.step_scores[6][player_key(8)],
            result.step_scores[6][player_key(8)],
        )


class StoredRunTest(unittest.TestCase):
    def test_no_minutes_file_leaves_the_scorer_unrun(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "live_deadline_gw6.md"
            log = run(report_path=dest)
            text = dest.read_text(encoding="utf-8")
        self.assertIn("The scorer is ready and was not run", text)
        self.assertIn("missing_minutes", log.reasons)
        self.assertFalse(log.scorer_ran)
        self.assertIsNone(log.chip)
        self.assertEqual(log.priced_weeks, ())
        self.assertNotIn("327", text)

    def test_the_real_log_can_form_a_pool_without_a_plan(self) -> None:
        logs = pd.read_csv(LOG_PATH)
        shares = deadline_shares(logs, 6)
        bootstrap = json.loads(BOOTSTRAP_PATH.read_text(encoding="utf-8"))
        entry = json.loads(ENTRY_PATH.read_text(encoding="utf-8"))
        roster = roster_from_bootstrap(bootstrap)
        owned = {player_key(int(player["id"])) for player in final_players(entry)}
        minutes = {pid: 90.0 for pid in owned}
        pool = build_pool(roster, shares, minutes, owned)
        self.assertTrue(owned <= set(pool["player_id"]))
        self.assertGreaterEqual(int(pool["eligible"].sum()), 15)
        self.assertGreater(float(shares["share_xG"].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
