"""Squad cards hide the whole segment, and a one-pass model can read minutes."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from src.models.flashcard_squad_trial import (
    BAND_H,
    CELL_W,
    COLS,
    CONCEDE_C,
    GOALS_C,
    HEADER_H,
    IMG_H,
    IMG_W,
    MASK_C,
    MASK_ID,
    MINUTES_C,
    PLAYER_H,
    POS_C,
    ROWS,
    SquadCard,
    SquadSequence,
    POS,
    SquadVision,
    OWN_HIST_C,
    OPP_HIST_C,
    apply_cut,
    bar_correspondence,
    cut_at_fraction,
    fit_clubs,
    prior_table,
    render_grid,
    sample_bars,
    squad_index,
    train_model,
    _stack,
)


def _league() -> dict:
    fixtures = []
    for gw in range(1, 31):
        fixtures.append(
            {
                "fixture": gw,
                "gw": gw,
                "kickoff": gw,
                "gc": 0.0 if gw <= 19 else 5.0,
                "played60": {1} if gw <= 19 else {2},
                "played1": {1} if gw <= 19 else {2},
            }
        )
    return {"toy": {"Opp": {"fixtures": fixtures, "by_id": {fx["fixture"]: fx for fx in fixtures}}}}


def _card() -> SquadCard:
    gw = np.zeros(COLS, dtype=np.int64)
    kickoff = np.zeros(COLS, dtype=np.int64)
    home = np.zeros(COLS, dtype=np.int64)
    fixture_id = np.zeros(COLS, dtype=np.int64)
    opponents = [""] * COLS
    for col, week in enumerate(range(1, 31)):
        gw[col] = week
        kickoff[col] = week
        home[col] = week % 2
        fixture_id[col] = week
        opponents[col] = "Opp"
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    assists = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    minutes[0, :30] = 90
    goals[0, :30] = 3
    present[0, :30] = True
    position = np.full(ROWS, -1, dtype=np.int64)
    position[0] = 2
    names = ["Player"] + ["spare"] * (ROWS - 1)
    return SquadCard(
        season="toy",
        team="Home",
        names=tuple(names),
        position=position,
        gw=gw,
        kickoff=kickoff,
        home=home,
        opponent=tuple(opponents),
        fixture_id=fixture_id,
        minutes=minutes,
        goals=goals,
        assists=assists,
        present=present,
        n_players=1,
        n_fixtures=30,
    )


class SquadCutTests(unittest.TestCase):
    def test_hidden_goals_are_masked_for_both_readers(self) -> None:
        view = apply_cut(_card(), _league(), 19)
        self.assertTrue(np.all(view["goal_token"][0, :19] == 3))
        self.assertTrue(np.all(view["goal_token"][0, 19:30] == MASK_ID))
        self.assertTrue(np.all(view["goal_label"][0, 19:30] == 3))
        image = render_grid(view)
        self.assertEqual(image.shape, (IMG_H, IMG_W, 3))
        goal_y = HEADER_H + 2 * BAND_H
        hidden_band = image[goal_y, 19 * CELL_W : 20 * CELL_W]
        self.assertTrue(np.all(hidden_band == MASK_C))
        visible_band = image[goal_y : goal_y + BAND_H, 0:CELL_W]
        self.assertTrue(np.any(np.all(visible_band == GOALS_C, axis=-1)))

    def test_changing_a_hidden_goal_does_not_change_the_input(self) -> None:
        card = _card()
        other = _card()
        other.goals = other.goals.copy()
        other.goals[0, 25] = 0
        left = apply_cut(card, _league(), 19)
        right = apply_cut(other, _league(), 19)
        self.assertTrue(np.array_equal(left["goal_token"], right["goal_token"]))
        self.assertTrue(np.array_equal(render_grid(left), render_grid(right)))
        self.assertEqual(int(left["goal_label"][0, 25]), 3)
        self.assertEqual(int(right["goal_label"][0, 25]), 0)

    def test_hidden_header_ignores_later_conceded_goals(self) -> None:
        view = apply_cut(_card(), _league(), 19)
        self.assertLess(view["concede"][24], 1.0)
        self.assertEqual(view["concede_missing"][24], 0.0)
        self.assertEqual(view["regulars"][24], 0.0)
        self.assertGreater(view["regulars"][5], 0.5)

    def test_sequence_model_reads_a_benched_teammate(self) -> None:
        rng = np.random.default_rng(0)
        cards = [_coupled_card(rng) for _ in range(16)]
        league = {"toy": {"Opp": {"fixtures": [], "by_id": {}}}}
        torch.manual_seed(0)
        model = SquadSequence(dropout=0.0)
        train_model(
            model, cards, league, np.random.default_rng(0), epochs=30, kind="sequence", lr=2e-3, batch_size=4
        )
        model.eval()
        out_probs = []
        in_probs = []
        with torch.no_grad():
            for card in cards:
                logits, _ = model(_one(apply_cut(card, league, 6)))
                prob = torch.softmax(logits, dim=-1)[0, 1].numpy()
                hidden = np.where(card.gw > 6)[0]
                star_out = card.minutes[0, hidden] < 1
                out_probs.extend(prob[hidden][star_out, 1].tolist())
                in_probs.extend(prob[hidden][~star_out, 1].tolist())
        self.assertGreater(float(np.mean(out_probs)), 0.7)
        self.assertLess(float(np.mean(in_probs)), 0.3)

    def test_picture_uses_the_minute_count_position_and_finer_concede(self) -> None:
        card = _card()
        card.minutes = card.minutes.copy()
        card.present = card.present.copy()
        card.position = card.position.copy()
        card.minutes[0, 0] = 59
        card.minutes[0, 1] = 60
        card.position[1] = 3
        card.minutes[1, 0] = 90
        card.present[1, 0] = True
        view = apply_cut(card, _league(), 19)
        view["concede"] = view["concede"].copy()
        view["concede_missing"] = view["concede_missing"].copy()
        view["concede"][0] = 0.74
        view["concede"][1] = 1.12
        view["concede_missing"][0] = 0.0
        view["concede_missing"][1] = 0.0
        image = render_grid(view)
        self.assertEqual(image.shape[0], ROWS * PLAYER_H)
        minutes_band = image[HEADER_H : HEADER_H + 2 * BAND_H]
        self.assertFalse(np.array_equal(minutes_band[:, 0:CELL_W], minutes_band[:, CELL_W : 2 * CELL_W]))
        self.assertTrue(np.any(np.all(minutes_band[:, 0:CELL_W] == MINUTES_C, axis=-1)))
        self.assertTrue(np.any(np.all(minutes_band[:, CELL_W : 2 * CELL_W] == MINUTES_C, axis=-1)))
        pos_y = HEADER_H + 4 * BAND_H
        self.assertTrue(np.array_equal(image[pos_y, 0], np.array(POS_C[2])))
        self.assertTrue(np.array_equal(image[pos_y + PLAYER_H, 0], np.array(POS_C[3])))
        for color in (OWN_HIST_C, OPP_HIST_C, CONCEDE_C):
            self.assertFalse(np.any(np.all(image == color, axis=-1)))
        torch.manual_seed(0)
        batch = _one(view)
        goals, assists = SquadVision(dropout=0.0)(
            batch["image"],
            batch["row_pad"],
            batch["column_on"],
            batch["position"],
            batch["minutes"],
            batch["matchup"],
            batch["club"],
            batch["opponent"],
        )
        self.assertEqual(tuple(goals.shape), (1, ROWS, COLS, 4))
        self.assertEqual(tuple(assists.shape), (1, ROWS, COLS, 4))

    def test_matching_bars_overlap_on_the_same_weeks(self) -> None:
        card = _card()
        labels = card.goals[None, :, :]
        probs = np.zeros((1, ROWS, COLS, 4), dtype=np.float64)
        probs[:, :, :, 0] = 1.0
        hidden = (card.gw > 19) & card.present[0]
        probs[0, 0, hidden] = 0.0
        probs[0, 0, hidden, 3] = 1.0
        matched = bar_correspondence([card], probs, labels, 19, np.random.default_rng(0), draws=5)
        self.assertEqual(matched["actual_bars"], int(hidden.sum()))
        self.assertEqual(matched["drawn_bars"], float(hidden.sum()))
        self.assertEqual(matched["actual_covered"], 1.0)
        self.assertEqual(matched["drawn_on_actual"], 1.0)
        blank = np.zeros_like(probs)
        blank[:, :, :, 0] = 1.0
        missed = bar_correspondence([card], blank, labels, 19, np.random.default_rng(0), draws=3)
        self.assertEqual(missed["drawn_bars"], 0.0)
        self.assertEqual(missed["actual_covered"], 0.0)
        certain = np.zeros((2, 4))
        certain[:, 1] = 1.0
        self.assertTrue(np.all(sample_bars(certain, np.random.default_rng(0)) == 1))

    def test_position_is_learned_for_a_goalkeeper_and_a_forward(self) -> None:
        cards = [_role_card() for _ in range(8)]
        league = {"toy": {"Opp": {"fixtures": [], "by_id": {}}}}
        torch.manual_seed(0)
        model = SquadSequence(dropout=0.0)
        train_model(model, cards, league, np.random.default_rng(0), epochs=20, kind="sequence", lr=2e-3, batch_size=4)
        model.eval()
        with torch.no_grad():
            logits, _ = model(_one(apply_cut(cards[0], league, 3)))
        prob = torch.softmax(logits, dim=-1)[0]
        hidden = np.where(cards[0].gw > 3)[0]
        self.assertLess(float(prob[0, hidden, 1:].sum(-1).mean()), 0.2)
        self.assertGreater(float(prob[1, hidden, 1:].sum(-1).mean()), 0.7)

    def test_history_uses_earlier_seasons_only(self) -> None:
        league = _two_season_league()
        prior = prior_table(league)
        self.assertAlmostEqual(prior[("2023-24", "Strong")][0], 2.5)
        self.assertAlmostEqual(prior[("2023-24", "Weak")][1], 2.2)
        self.assertNotIn(("2022-23", "Strong"), prior)
        self.assertNotIn(("2023-24", "Promoted"), prior)
        card = _history_card("Strong", "Weak", goals=1)
        early = apply_cut(card, league, 0)
        late = apply_cut(card, league, 4)
        self.assertAlmostEqual(float(early["own_hist"][3]), 2.5)
        self.assertAlmostEqual(float(early["opp_hist"][3]), 2.2, places=5)
        self.assertEqual(float(early["own_form_missing"][3]), 1.0)
        self.assertEqual(float(early["own_hist_missing"][3]), 0.0)
        self.assertGreater(float(late["own_form"][5]), 2.0)
        self.assertLess(float(late["own_form"][5]), 3.0)
        promoted = _history_card("Promoted", "Strong", goals=0)
        view = apply_cut(promoted, league, 4)
        self.assertEqual(float(view["own_hist_missing"][3]), 1.0)
        self.assertAlmostEqual(float(view["opp_hist"][3]), 0.4, places=5)
        clubs = fit_clubs([card])
        self.assertEqual(int(apply_cut(card, league, 4, clubs)["club"]), clubs["Strong"])
        self.assertEqual(int(apply_cut(promoted, league, 4, clubs)["club"]), 0)

    def test_a_promoted_side_is_not_given_the_strong_club_rate(self) -> None:
        league = _two_season_league()
        strong = [_history_card("Strong", "Weak", goals=1) for _ in range(8)]
        weak = [_history_card("Weak", "Strong", goals=0) for _ in range(8)]
        cards = strong + weak
        clubs = fit_clubs(cards)
        torch.manual_seed(0)
        model = SquadSequence(dropout=0.0, n_clubs=max(clubs.values()) + 1)
        train_model(
            model,
            cards,
            league,
            np.random.default_rng(0),
            epochs=25,
            kind="sequence",
            lr=2e-3,
            batch_size=4,
            clubs=clubs,
        )
        model.eval()
        with torch.no_grad():
            high, _ = model(_one(apply_cut(strong[0], league, 0, clubs)))
            low, _ = model(_one(apply_cut(weak[0], league, 0, clubs)))
        high_p = torch.softmax(high, dim=-1)[0, 0, :8, 1:].sum(-1).mean()
        low_p = torch.softmax(low, dim=-1)[0, 0, :8, 1:].sum(-1).mean()
        self.assertGreater(float(high_p), 0.6)
        self.assertLess(float(low_p), 0.3)
        image = render_grid(apply_cut(strong[0], league, 0, clubs))
        self.assertFalse(np.any(np.all(image == OWN_HIST_C, axis=-1)))
        self.assertFalse(np.any(np.all(image == OPP_HIST_C, axis=-1)))

    def test_the_other_squad_keeps_hidden_events_masked(self) -> None:
        home, opp = _facing("2023-24", defenders=90, goals=0)
        opp.goals = opp.goals.copy()
        opp.goals[0, 0] = 2
        opp.goals[0, 4] = 3
        league = _flat_league()
        view = apply_cut(home, league, 1, squads=squad_index([home, opp]))
        self.assertEqual(int(view["opp_position"][0, 0]), POS["DEF"])
        self.assertEqual(float(view["opp_minutes"][0, 0]), 90.0)
        self.assertEqual(int(view["opp_goal_token"][0, 0]), 2)
        self.assertEqual(int(view["opp_goal_token"][0, 4]), MASK_ID)
        self.assertTrue(bool(view["opp_pad"][4, 0]))
        opp.goals[0, 4] = 1
        again = apply_cut(home, league, 1, squads=squad_index([home, opp]))
        self.assertTrue(np.array_equal(view["opp_goal_token"], again["opp_goal_token"]))
        self.assertEqual(int(again["goal_label"][0, 4]), 0)

    def test_a_squad_reads_the_other_squad(self) -> None:
        open_pair = [_facing("2022-23", defenders=0, goals=1) for _ in range(6)]
        shut_pair = [_facing("2023-24", defenders=90, goals=0) for _ in range(6)]
        cards = [card for pair in open_pair + shut_pair for card in pair]
        league = _flat_league()
        clubs = fit_clubs(cards)
        index = squad_index(cards)
        torch.manual_seed(0)
        model = SquadSequence(dropout=0.0, n_clubs=max(clubs.values()) + 1)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        rng = np.random.default_rng(0)
        model.train()
        for _ in range(40):
            order = rng.permutation(len(cards))
            for start in range(0, len(cards), 4):
                pick = [cards[int(i)] for i in order[start : start + 4]]
                batch = _stack([apply_cut(card, league, 1, clubs, index) for card in pick])
                opt.zero_grad()
                goal_logits, _ = model(batch)
                loss = torch.nn.functional.cross_entropy(
                    goal_logits.reshape(-1, 4), batch["goal_label"].reshape(-1), reduction="none"
                ).reshape(batch["goal_label"].shape)
                mask = batch["loss"]
                (loss * mask).sum().div(mask.sum().clamp_min(1)).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            open_p = torch.softmax(model(_stack([apply_cut(open_pair[0][0], league, 1, clubs, index)]))[0], dim=-1)
            shut_p = torch.softmax(model(_stack([apply_cut(shut_pair[0][0], league, 1, clubs, index)]))[0], dim=-1)
        hidden = slice(1, 6)
        self.assertGreater(float(open_p[0, 0, hidden, 1:].sum(-1).mean()), 0.6)
        self.assertLess(float(shut_p[0, 0, hidden, 1:].sum(-1).mean()), 0.4)

    def test_a_hidden_column_does_not_see_a_later_absence(self) -> None:
        home, opp = _absent_scorer()
        league = _scorer_league()
        index = squad_index([home, opp])
        shown = apply_cut(home, league, 6, squads=index)
        hidden = apply_cut(home, league, 3, squads=index)
        self.assertEqual(shown["matchup"].shape[-1], 14)
        self.assertAlmostEqual(float(shown["opp_form"][5]), 1.2)
        self.assertAlmostEqual(float(hidden["opp_form"][5]), 0.0)
        self.assertAlmostEqual(float(shown["key_play"][5]), 1.0 / 3.0)
        self.assertAlmostEqual(float(hidden["key_play"][5]), 1.0)
        self.assertEqual(float(hidden["key_missing"][5]), 0.0)

    def test_reveal_starts_at_gameweek_one(self) -> None:
        card = _card()
        self.assertEqual(cut_at_fraction(card, 0.5), 15)
        self.assertEqual(cut_at_fraction(card, 0.8), 24)
        self.assertLess(cut_at_fraction(card, 0.2), cut_at_fraction(card, 0.8))


def _fixture_block(gf: float, gc: float, n: int) -> dict:
    fixtures = []
    for gw in range(1, n + 1):
        fixtures.append(
            {
                "fixture": gw,
                "gw": gw,
                "kickoff": gw,
                "gc": gc,
                "gf": gf,
                "played60": set(),
                "played1": set(),
            }
        )
    return {"fixtures": fixtures, "by_id": {fx["fixture"]: fx for fx in fixtures}}


def _two_season_league() -> dict:
    return {
        "2022-23": {
            "Strong": _fixture_block(2.5, 0.4, 10),
            "Weak": _fixture_block(0.4, 2.2, 10),
        },
        "2023-24": {
            "Strong": _fixture_block(2.5, 0.4, 8),
            "Weak": _fixture_block(0.4, 2.2, 8),
        },
    }


def _history_card(team: str, opponent: str, goals: int) -> SquadCard:
    n = 8
    gw = np.zeros(COLS, dtype=np.int64)
    gw[:n] = np.arange(1, n + 1)
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    scored = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    minutes[0, :n] = 90
    present[0, :n] = True
    scored[0, :n] = goals
    position = np.full(ROWS, -1, dtype=np.int64)
    position[0] = POS["FWD"]
    return SquadCard(
        season="2023-24",
        team=team,
        names=tuple(["Forward"] + ["spare"] * (ROWS - 1)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
        home=np.ones(COLS, dtype=np.int64),
        opponent=tuple([opponent] * n + [""] * (COLS - n)),
        fixture_id=gw.copy(),
        minutes=minutes,
        goals=scored,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=1,
        n_fixtures=n,
    )


def _role_card() -> SquadCard:
    """A goalkeeper never scores. A forward scores every week. Minutes match."""
    n = 6
    gw = np.zeros(COLS, dtype=np.int64)
    gw[:n] = np.arange(1, n + 1)
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    minutes[:2, :n] = 90
    present[:2, :n] = True
    goals[1, :n] = 1
    position = np.full(ROWS, -1, dtype=np.int64)
    position[0] = POS["GKP"]
    position[1] = POS["FWD"]
    return SquadCard(
        season="toy",
        team="Home",
        names=tuple(["Keeper", "Forward"] + ["spare"] * (ROWS - 2)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
        home=np.zeros(COLS, dtype=np.int64),
        opponent=tuple([""] * COLS),
        fixture_id=np.zeros(COLS, dtype=np.int64),
        minutes=minutes,
        goals=goals,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=2,
        n_fixtures=n,
    )


def _coupled_card(rng: np.random.Generator) -> SquadCard:
    """Row 1 scores exactly when row 0 has no minutes."""
    n = 12
    gw = np.zeros(COLS, dtype=np.int64)
    kickoff = np.zeros(COLS, dtype=np.int64)
    gw[:n] = np.arange(1, n + 1)
    kickoff[:n] = gw[:n]
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    star = rng.random(n) < 0.5
    minutes[0, :n] = np.where(star, 90, 0)
    minutes[1, :n] = 90
    present[0, :n] = True
    present[1, :n] = True
    goals[1, :n] = (minutes[0, :n] < 1).astype(np.int64)
    position = np.full(ROWS, -1, dtype=np.int64)
    position[:2] = 2
    return SquadCard(
        season="toy",
        team="Home",
        names=tuple(["Star", "Other"] + ["spare"] * (ROWS - 2)),
        position=position,
        gw=gw,
        kickoff=kickoff,
        home=np.zeros(COLS, dtype=np.int64),
        opponent=tuple([""] * COLS),
        fixture_id=np.zeros(COLS, dtype=np.int64),
        minutes=minutes,
        goals=goals,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=2,
        n_fixtures=n,
    )


def _flat_league() -> dict:
    return {
        season: {"Home": _fixture_block(1.0, 1.0, 6), "Opp": _fixture_block(1.0, 1.0, 6)}
        for season in ("2021-22", "2022-23", "2023-24")
    }


def _facing(season: str, defenders: int, goals: int) -> tuple[SquadCard, SquadCard]:
    """Home scores on hidden weeks only when the other squad's defenders do not play."""
    n = 6

    def shell() -> tuple:
        gw = np.zeros(COLS, dtype=np.int64)
        gw[:n] = np.arange(1, n + 1)
        minutes = np.zeros((ROWS, COLS), dtype=np.float32)
        scored = np.zeros((ROWS, COLS), dtype=np.int64)
        present = np.zeros((ROWS, COLS), dtype=bool)
        position = np.full(ROWS, -1, dtype=np.int64)
        return gw, minutes, scored, present, position

    gw, minutes, scored, present, position = shell()
    minutes[0, :n] = 90
    present[0, :n] = True
    scored[0, 1:n] = goals
    position[0] = POS["FWD"]
    home = SquadCard(
        season=season,
        team="Home",
        names=tuple(["Forward"] + ["spare"] * (ROWS - 1)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
        home=np.ones(COLS, dtype=np.int64),
        opponent=tuple(["Opp"] * n + [""] * (COLS - n)),
        fixture_id=gw.copy(),
        minutes=minutes,
        goals=scored,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=1,
        n_fixtures=n,
    )
    gw, minutes, scored, present, position = shell()
    minutes[:4, :n] = defenders
    present[:4, :n] = True
    position[:4] = POS["DEF"]
    opp = SquadCard(
        season=season,
        team="Opp",
        names=tuple(["Defender"] * 4 + ["spare"] * (ROWS - 4)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
        home=np.zeros(COLS, dtype=np.int64),
        opponent=tuple(["Home"] * n + [""] * (COLS - n)),
        fixture_id=gw.copy(),
        minutes=minutes,
        goals=scored,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=4,
        n_fixtures=n,
    )
    return home, opp


def _scorer_league() -> dict:
    fixtures = []
    for gw, goals in enumerate((0.0, 0.0, 0.0, 3.0, 3.0, 3.0), start=1):
        fixtures.append(
            {
                "fixture": gw,
                "gw": gw,
                "kickoff": gw,
                "gc": 1.0,
                "gf": goals,
                "played60": set(),
                "played1": {1},
            }
        )
    block = {"fixtures": fixtures, "by_id": {fx["fixture"]: fx for fx in fixtures}}
    return {"2023-24": {"Home": block, "Opp": block}}


def _absent_scorer() -> tuple[SquadCard, SquadCard]:
    """The opponent's scorer plays the first three weeks and misses the next three."""
    n = 6
    gw = np.zeros(COLS, dtype=np.int64)
    gw[:n] = np.arange(1, n + 1)
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    position = np.full(ROWS, -1, dtype=np.int64)
    minutes[0, :n] = 90
    present[0, :n] = True
    position[0] = POS["FWD"]
    home = SquadCard(
        season="2023-24",
        team="Home",
        names=tuple(["Forward"] + ["spare"] * (ROWS - 1)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
        home=np.ones(COLS, dtype=np.int64),
        opponent=tuple(["Opp"] * n + [""] * (COLS - n)),
        fixture_id=gw.copy(),
        minutes=minutes,
        goals=goals,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=1,
        n_fixtures=n,
    )
    minutes = np.zeros((ROWS, COLS), dtype=np.float32)
    goals = np.zeros((ROWS, COLS), dtype=np.int64)
    present = np.zeros((ROWS, COLS), dtype=bool)
    position = np.full(ROWS, -1, dtype=np.int64)
    minutes[0, :3] = 90
    present[0, :n] = True
    goals[0, :3] = 1
    position[0] = POS["FWD"]
    opp = SquadCard(
        season="2023-24",
        team="Opp",
        names=tuple(["Scorer"] + ["spare"] * (ROWS - 1)),
        position=position,
        gw=gw,
        kickoff=gw.copy(),
        home=np.zeros(COLS, dtype=np.int64),
        opponent=tuple(["Home"] * n + [""] * (COLS - n)),
        fixture_id=gw.copy(),
        minutes=minutes,
        goals=goals,
        assists=np.zeros((ROWS, COLS), dtype=np.int64),
        present=present,
        n_players=1,
        n_fixtures=n,
    )
    return home, opp


def _one(view: dict) -> dict:
    return _stack([view])
