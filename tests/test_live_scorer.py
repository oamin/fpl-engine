"""Live scorer. The formula is xp_on_pot. No network and no Odds API call."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.live.deadline import (
    BOOTSTRAP_PATH,
    ENTRY_PATH,
    LOG_PATH,
    DeadlineError,
    Holding,
    bench_for_transfers,
    collect,
    final_players,
    player_key,
    render,
    run,
    submitted_line,
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
from src.models.forecast_xp import side_pot, xp_on_pot
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

    def test_ep_next_is_the_number_that_prices_the_eleven(self) -> None:
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
                    "score_xp": 1.0,
                }
            )
        pool = pd.DataFrame(rows)
        clubs = {gw: {f"c{i:02d}" for i in range(15)} for gw in range(6, 20)}
        engine = {pid: 1.0 for pid in high + bench}
        steps = {6: engine, 7: engine, 8: engine}
        choice = {pid: 20.0 for pid in high}
        choice.update({pid: 0.0 for pid in bench})
        state = SquadState(purchase={pid: 50 for pid in high + bench}, bank=0, ft=1)
        _plan, weeks = plan_deadline(
            6, state, pool, steps, clubs, played={1: "triple_captain"}, choice=choice
        )
        by_gw = {int(row.gw): row for row in weeks}
        self.assertEqual(by_gw[6].held.xi_xp, 240.0)
        self.assertEqual(by_gw[6].held.bench_xp, 0.0)
        self.assertEqual(by_gw[7].held.xi_xp, 0.0)
        _engine_plan, engine_weeks = plan_deadline(
            6, state, pool, steps, clubs, played={1: "triple_captain"}
        )
        engine_by = {int(row.gw): row for row in engine_weeks}
        self.assertEqual(engine_by[6].held.xi_xp, 12.0)


def _official(path: Path, rows: list[tuple[str, float]], field: str = "ep_next") -> None:
    lines = ["player_id,gw,official_xp,source_field,captured_at"]
    for pid, xp in rows:
        lines.append(f"{pid},6,{xp},{field},2026-10-10T09:00:00Z")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class DecisionCaptureTest(unittest.TestCase):
    def test_the_live_path_reads_only_the_captured_ep_next(self) -> None:
        source = inspect.getsource(collect)
        self.assertIn("load_ep_next", source)
        self.assertIn("choice=choice", source)
        self.assertNotIn("score_xp", source)
        self.assertNotIn("score_steps", source)
        self.assertNotIn("export_deadline_scores", source)
        note = inspect.getsource(render)
        self.assertIn("not the decision pair", note)
        self.assertIn("does not count as a chip rule", note)
        self.assertIn("No manual override is recorded.", note)
        with self.assertRaises(DeadlineError):
            collect(dry_run=False, decision_file=Path("x.csv"))
        with self.assertRaises(DeadlineError):
            collect(dry_run=True)

    def test_the_decision_file_is_the_t1_slot(self) -> None:
        from src.live.scorer import ScorerError, decision_capture_file, load_ep_next

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            early = root / "official_20261009T100000Z.csv"
            later = root / "official_20261010T090000Z.csv"
            _official(early, [("a", 4.0), ("b", 1.0)])
            _official(later, [("a", 9.0), ("b", 9.0)])
            (root / "slot_t24.json").write_text(
                json.dumps({"slot": "t24", "official": early.name}),
                encoding="utf-8",
            )
            with self.assertRaises(ScorerError):
                decision_capture_file(6, root)
            (root / "slot_t1.json").write_text(
                json.dumps({"gw": 6, "slot": "t1", "official": early.name}),
                encoding="utf-8",
            )
            chosen = load_ep_next(6, root)
            self.assertEqual(chosen, {"a": 4.0, "b": 1.0})
            _official(later, [("a", 0.0), ("b", 0.0)], field="score_xp")
            with self.assertRaises(ScorerError):
                load_ep_next(6, path=later)
            zeros = root / "zeros.csv"
            _official(zeros, [("a", 0.0), ("b", 0.0)])
            with self.assertRaises(ScorerError):
                load_ep_next(6, path=zeros)
            named = load_ep_next(6, path=early)
            self.assertEqual(named["a"], 4.0)

    def test_the_submitted_line_is_the_owned_fifteen(self) -> None:
        roles = (
            [("g1", "GKP"), ("g2", "GKP")]
            + [(f"d{i}", "DEF") for i in range(1, 6)]
            + [(f"m{i}", "MID") for i in range(1, 6)]
            + [(f"f{i}", "FWD") for i in range(1, 4)]
        )
        holdings = [
            Holding(
                element=index,
                key=pid,
                name=pid,
                position=pos,
                team=f"c{index:02d}",
                purchase=50,
                purchase_source="gw1",
                current=50,
                formula_sell=50,
                selling=50,
                selling_source="formula",
            )
            for index, (pid, pos) in enumerate(roles, start=1)
        ]
        high = {"g1", "d1", "d2", "d3", "m1", "m2", "m3", "m4", "m5", "f1", "f2"}
        choice = {row.key: (20.0 if row.key in high else 0.0) for row in holdings}
        note = submitted_line(holdings, choice, 15)
        self.assertIn("current 15, legal", note)
        self.assertIn("Formation 1-3-5-2", note)
        self.assertIn("g1 (C)", note)
        self.assertIn("bank 15 tenths", note)
        self.assertIn("The chip is the logged judgement.", note)
        broken = list(holdings)
        broken[0] = Holding(
            element=broken[0].element,
            key=broken[0].key,
            name=broken[0].name,
            position=broken[0].position,
            team=broken[1].team,
            purchase=50,
            purchase_source="gw1",
            current=50,
            formula_sell=50,
            selling=50,
            selling_source="formula",
        )
        # Four from one club: g1 joins g2's club, and two more are pointed at it.
        broken[2] = Holding(
            element=broken[2].element,
            key=broken[2].key,
            name=broken[2].name,
            position=broken[2].position,
            team=broken[1].team,
            purchase=50,
            purchase_source="gw1",
            current=50,
            formula_sell=50,
            selling=50,
            selling_source="formula",
        )
        broken[3] = Holding(
            element=broken[3].element,
            key=broken[3].key,
            name=broken[3].name,
            position=broken[3].position,
            team=broken[1].team,
            purchase=50,
            purchase_source="gw1",
            current=50,
            formula_sell=50,
            selling=50,
            selling_source="formula",
        )
        self.assertIn("not legal", submitted_line(broken, choice, 15))


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
                "web_name": f"Player{pid}",
                "first_name": "P",
                "second_name": f"Player{pid}",
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
            artifacts_dir=Path("/tmp/no-betfair-artifacts"),
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

    def test_to_score_and_outrights_enter_score_and_forecast(self) -> None:
        """Betfair anytime rates move GW6 score_xp; outrights move unpriced GW8."""
        bootstrap, fixtures, odds, logs = _world()
        owned = [player_key(i) for i in range(1, 16)]
        state = SquadState(purchase={pid: 50 for pid in owned}, bank=0, ft=1)
        minutes = {pid: 90.0 for pid in owned}
        baseline = price_half(
            gw=6,
            logs=logs,
            odds=odds,
            fixtures=fixtures,
            bootstrap=bootstrap,
            state=state,
            minutes=minutes,
            played={1: "triple_captain"},
        )
        mid = next(e for e in bootstrap["elements"] if int(e["id"]) == 8)
        from src.teams import norm_team

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # A low anytime rate for element 8. His own rate is the allocation.
            (root / "betfair_to_score.json").write_text(
                json.dumps(
                    [
                        {
                            "runner": str(mid["web_name"]),
                            "mu_raw": 0.2,
                            "matched": 5000,
                            "p_mid": 0.18,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            # Extreme strengths so GW8 forecast differs from a GW7 copy.
            ranks = []
            for team in bootstrap["teams"]:
                name = str(team["name"])
                tid = int(team["id"])
                ranks.append(
                    {
                        "club": name,
                        "club_norm": norm_team(name),
                        "p_win": 0.9 if tid == 1 else 0.01,
                        "p_top6": 0.95 if tid == 1 else 0.05,
                        "p_rel": 0.01 if tid != 2 else 0.8,
                        "E_rank": 1.5 if tid == 1 else (19.0 if tid == 2 else 12.0),
                        "strength": 0.9 if tid == 1 else (-0.9 if tid == 2 else 0.0),
                    }
                )
            (root / "outrights_ranks.json").write_text(
                json.dumps(ranks), encoding="utf-8"
            )
            with_book = price_half(
                gw=6,
                logs=logs,
                odds=odds,
                fixtures=fixtures,
                bootstrap=bootstrap,
                state=state,
                minutes=minutes,
                played={1: "triple_captain"},
                artifacts_dir=root,
            )
        pid = player_key(8)
        self.assertLess(
            with_book.step_scores[6][pid],
            baseline.step_scores[6][pid],
        )
        self.assertEqual(with_book.step_scores[7][pid], baseline.step_scores[7][pid])
        other = player_key(9)
        self.assertEqual(with_book.step_scores[6][other], baseline.step_scores[6][other])
        # Unpriced GW8 should not merely copy GW7 when outrights supply pots.
        self.assertNotEqual(with_book.step_scores[8], with_book.step_scores[7])
        self.assertEqual(with_book.copy_note, "")


class StoredRunTest(unittest.TestCase):
    def test_no_minutes_file_leaves_the_scorer_unrun(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "live_deadline_gw6.md"
            with mock.patch(
                "src.live.t1_inputs.exchange_sheet",
                return_value=Path("/tmp/no-such-betfair-t1-gw_lines.csv"),
            ):
                log = run(report_path=dest)
            text = dest.read_text(encoding="utf-8")
        self.assertTrue(str(log.minutes_file).endswith("xmi_t1.csv"))
        self.assertIn("missing_opening_line", log.reasons)
        self.assertNotIn("missing_minutes", log.reasons)
        self.assertIn("no 1X2", text)
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
