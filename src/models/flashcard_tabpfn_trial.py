"""Pretrained tabular reader for the same card the vision gate paints.

Each row is one cell of that card. The cell is the player slot, the fixture
slot, the position, the minutes, the sheet flag, the venue, and the club's
recent scoring rate. Goal and assist tokens are the counts already on the
card, or the mask where the cut has hidden them. The same fixture then
contributes its forty squad rows and forty opponent rows, as position and
minutes. Club and opponent are fixed codes, the same number in every season.

Concede, regulars, own history, and opponent history stay out. TabPFN is a
classifier pretrained on synthetic tables. It is not trained on these seasons.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.flashcard_calibrated_trial import place_expected
from src.models.flashcard_card_score import tally_cards
from src.models.flashcard_squad_trial import (
    COLS,
    MASK_ID,
    ROWS,
    SEED,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    squad_index,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
CONTEXT = 4_000
# Permanent club codes. A club keeps its number in every season. A club that
# arrives later takes the next free number; existing numbers stay put.
CLUB_CODE = {
    "Arsenal": 1,
    "Aston Villa": 2,
    "Bournemouth": 3,
    "Brentford": 4,
    "Brighton": 5,
    "Burnley": 6,
    "Chelsea": 7,
    "Crystal Palace": 8,
    "Everton": 9,
    "Fulham": 10,
    "Ipswich": 11,
    "Leeds": 12,
    "Leicester": 13,
    "Liverpool": 14,
    "Luton": 15,
    "Man City": 16,
    "Man Utd": 17,
    "Newcastle": 18,
    "Nott'm Forest": 19,
    "Sheffield Utd": 20,
    "Southampton": 21,
    "Spurs": 22,
    "Sunderland": 23,
    "West Ham": 24,
    "Wolves": 25,
}


def club_code(name: str) -> int:
    """The club's permanent code. An unseen name is 0."""
    return int(CLUB_CODE.get(str(name), 0))


def season_code(season: str) -> int:
    """The opening year of the season, so two seasons of one club stay distinct."""
    head = str(season)[:4]
    return int(head) if head.isdigit() else 0


def _indexed(prefix: str, field: str) -> tuple[str, ...]:
    return tuple(f"{prefix}_{field}_{index:02d}" for index in range(ROWS))


MATE_MINUTES = _indexed("mate", "minutes")
MATE_POSITION = _indexed("mate", "position")
OPP_MINUTES = _indexed("opp", "minutes")
OPP_POSITION = _indexed("opp", "position")
WIDE = MATE_MINUTES + MATE_POSITION + OPP_MINUTES + OPP_POSITION
CELL = (
    "row_slot",
    "col_slot",
    "side",
    "position",
    "minutes",
    "present",
    "home",
    "own_form",
    "own_form_missing",
    "season",
    "club",
    "opponent",
    "goal_visible",
    "assist_visible",
)
FEATURES = CELL + WIDE
POSITION_COLUMNS = ("position",) + MATE_POSITION + OPP_POSITION
CATEGORICAL = (
    "row_slot",
    "side",
    "position",
    "season",
    "club",
    "opponent",
    "goal_visible",
    "assist_visible",
) + MATE_POSITION + OPP_POSITION


def _column_block(view: dict[str, np.ndarray], col: int) -> np.ndarray:
    """Position and minutes down the squad and the opposition at one fixture."""
    return np.concatenate(
        [
            view["minutes"][:, col],
            view["position"],
            view["opp_minutes"][:, col],
            view["opp_position"][:, col],
        ]
    )


def table_at(cards: list, league: dict, squads: dict, cuts: list[int]) -> pd.DataFrame:
    """One row per visible cell. Hidden own appearances are included, with masked tokens."""
    rows: list[dict[str, float]] = []
    wide: list[np.ndarray] = []
    for index, (card, cut) in enumerate(zip(cards, cuts, strict=True)):
        view = apply_cut(card, league, cut, {}, squads)
        season = season_code(card.season)
        own_club = club_code(card.team)
        for col in range(COLS):
            if int(card.gw[col]) <= 0:
                continue
            hidden = int(card.gw[col]) > cut
            block = _column_block(view, col)
            opponent = club_code(card.opponent[col]) if card.opponent[col] else 0
            home = float(card.home[col])
            form = float(view["own_form"][col])
            form_missing = float(view["own_form_missing"][col])
            for player in range(ROWS):
                if int(card.position[player]) < 0 or not bool(card.present[player, col]):
                    continue
                rows.append(
                    {
                        "card_index": index,
                        "player": player,
                        "fixture": col,
                        "row_slot": player,
                        "col_slot": col,
                        "side": 0,
                        "position": int(card.position[player]),
                        "minutes": float(card.minutes[player, col]),
                        "present": 1.0,
                        "home": home,
                        "own_form": form,
                        "own_form_missing": form_missing,
                        "season": season,
                        "club": own_club,
                        "opponent": opponent,
                        "goal_visible": int(view["goal_token"][player, col]),
                        "assist_visible": int(view["assist_token"][player, col]),
                        "y_goal": int(card.goals[player, col]),
                        "y_assist": int(card.assists[player, col]),
                        "hidden": int(hidden),
                    }
                )
                wide.append(block)
            if hidden:
                continue
            for player in range(ROWS):
                if int(view["opp_position"][player, col]) < 0 or float(view["opp_present"][player, col]) < 0.5:
                    continue
                if int(view["opp_goal_token"][player, col]) >= MASK_ID or int(view["opp_assist_token"][player, col]) >= MASK_ID:
                    continue
                rows.append(
                    {
                        "card_index": index,
                        "player": player,
                        "fixture": col,
                        "row_slot": player,
                        "col_slot": col,
                        "side": 1,
                        "position": int(view["opp_position"][player, col]),
                        "minutes": float(view["opp_minutes"][player, col]),
                        "present": 1.0,
                        "home": home,
                        "own_form": 0.0,
                        "own_form_missing": 1.0,
                        "season": season,
                        "club": opponent,
                        "opponent": own_club,
                        "goal_visible": int(view["opp_goal_token"][player, col]),
                        "assist_visible": int(view["opp_assist_token"][player, col]),
                        "y_goal": int(view["opp_goal_token"][player, col]),
                        "y_assist": int(view["opp_assist_token"][player, col]),
                        "hidden": 0,
                    }
                )
                wide.append(block)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    extra = pd.DataFrame(np.vstack(wide), columns=list(WIDE))
    return pd.concat([frame, extra], axis=1)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Model matrix. A padded position uses the same spare index as the vision reader."""
    out = frame.loc[:, list(FEATURES)].copy()
    for name in POSITION_COLUMNS:
        out[name] = np.where(out[name].to_numpy() < 0, 4, out[name].to_numpy()).astype(np.int64)
    return out


def align_probs(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map classifier columns onto counts 0, 1, 2, and 3."""
    out = np.zeros((len(proba), 4), dtype=np.float64)
    for slot, label in enumerate(classes):
        if 0 <= int(label) <= 3:
            out[:, int(label)] = proba[:, slot]
    total = out.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return out / total


def take_cards(blocks: list[pd.DataFrame], rng: np.random.Generator, limit: int) -> pd.DataFrame:
    """Whole cards, in a random order, until the context is full."""
    order = rng.permutation(len(blocks))
    kept: list[pd.DataFrame] = []
    total = 0
    for index in order:
        block = blocks[int(index)]
        if block.empty:
            continue
        if kept and total + len(block) > limit:
            continue
        kept.append(block)
        total += len(block)
        if total >= limit:
            break
    if not kept:
        return blocks[0].iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)


def _blocks(frame: pd.DataFrame) -> list[pd.DataFrame]:
    if frame.empty:
        return []
    return [block for _, block in frame.groupby("card_index", sort=False)]


def _classifier():
    import os

    from tabpfn import TabPFNClassifier

    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    return TabPFNClassifier(
        device="cpu",
        n_estimators=1,
        fit_mode="low_memory",
        ignore_pretraining_limits=True,
        categorical_features_indices=[FEATURES.index(name) for name in CATEGORICAL],
        random_state=SEED,
        n_jobs=1,
    )


def _predict_proba(model, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.zeros((0, 4))
    return np.asarray(model.predict_proba(features(frame)))


def _scatter(frame: pd.DataFrame, probs: np.ndarray, n_cards: int) -> np.ndarray:
    painted = np.zeros((n_cards, ROWS, COLS, 4), dtype=np.float64)
    painted[..., 0] = 1.0
    if len(frame) == 0:
        return painted
    cards = frame["card_index"].to_numpy()
    players = frame["player"].to_numpy()
    fixtures = frame["fixture"].to_numpy()
    painted[cards, players, fixtures] = probs
    return painted


def _place(probs: np.ndarray, cards: list, cuts: list[int]) -> np.ndarray:
    painted = np.zeros((len(cards), ROWS, COLS), dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        painted[index] = place_expected(probs[index], card.minutes, mask)
    return painted


def _line(revealed: float, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | tabpfn | {channel} | {totals['actual']} | {totals['painted']} | "
        f"{totals['exact']} | {totals['actual_within']} | {totals['painted_within']} |"
    )


def _queries(block: pd.DataFrame) -> pd.DataFrame:
    own = block[(block["side"] == 0) & (block["hidden"] == 1)]
    return own.reset_index(drop=True)


def _context(block: pd.DataFrame) -> pd.DataFrame:
    return block[block["hidden"] == 0].reset_index(drop=True)


def main() -> None:
    started = time.perf_counter()
    print("loading tabular cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    squads = squad_index(train + test)
    frames = []
    for fraction in FRACTIONS:
        frame = table_at(train, league, squads, [cut_at_fraction(card, fraction) for card in train])
        frame["card_index"] = frame["card_index"] + len(frames) * len(train)
        frames.append(frame)
        print(f"train cells at {fraction:.0%}: {len(frame)}", flush=True)
    pool = _blocks(pd.concat(frames, ignore_index=True))
    rng = np.random.default_rng(SEED)
    background = take_cards(pool, rng, CONTEXT)
    print(
        f"context cells {len(background)} goal events {(background['y_goal'] > 0).sum()}",
        flush=True,
    )
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    goal_at_20: dict[str, int] | None = None
    lines = [
        "# Pretrained tabular reader",
        "",
        "Each row is one cell of the stripped card. The cell carries the player",
        "slot, the fixture slot, position, minutes, the sheet flag, venue, and",
        "the club's recent scoring rate. Goal and assist tokens are the painted",
        "counts, or the mask on a hidden cell. That fixture's forty squad rows",
        "and forty opponent rows are on the same row, as position and minutes.",
        "Club and opponent use a fixed code, the same number in every season.",
        "Concede, regulars, own history, and opponent history are absent.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "The gate, at 20% revealed, puts 190 goal bars on the same appearance",
        "and 258 within one gameweek.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for revealed in FRACTIONS:
        cuts = [cut_at_fraction(card, revealed) for card in test]
        print(f"scoring {revealed:.0%}", flush=True)
        frame = table_at(test, league, squads, cuts)
        goal_probs = np.zeros((len(test), ROWS, COLS, 4), dtype=np.float64)
        assist_probs = np.zeros_like(goal_probs)
        goal_probs[..., 0] = 1.0
        assist_probs[..., 0] = 1.0
        for card_index, block in frame.groupby("card_index", sort=True):
            query = _queries(block)
            if query.empty:
                continue
            context = pd.concat([background, _context(block)], ignore_index=True)
            print(f"  card {int(card_index) + 1}/{len(test)} context {len(context)} query {len(query)}", flush=True)
            goal_model = _classifier()
            goal_model.fit(features(context), context["y_goal"].to_numpy())
            assist_model = _classifier()
            assist_model.fit(features(context), context["y_assist"].to_numpy())
            goal_p = align_probs(_predict_proba(goal_model, query), goal_model.classes_)
            assist_p = align_probs(_predict_proba(assist_model, query), assist_model.classes_)
            goal_probs[query["card_index"], query["player"], query["fixture"]] = goal_p
            assist_probs[query["card_index"], query["player"], query["fixture"]] = assist_p
        for channel, probs, actual in (
            ("goals", goal_probs, labels),
            ("assists", assist_probs, assist_labels),
        ):
            totals = tally_cards(test, actual, _place(probs, test, cuts), 0, cuts)
            lines.append(_line(revealed, channel, totals))
            if revealed == 0.2 and channel == "goals":
                goal_at_20 = totals
            print(
                f"tabpfn {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    if goal_at_20 is None:
        raise RuntimeError("the 20% goal cut did not score")
    lines.extend(
        [
            "",
            "At 20% revealed this reader puts "
            f"{goal_at_20['exact']} goal bars on the same appearance",
            f"and {goal_at_20['actual_within']} within one gameweek. It paints "
            f"{goal_at_20['painted']} goal bars against {goal_at_20['actual']} actual.",
            "The gate puts 190 and 258. The gate stays the stripped card measured at 15:53.",
            "",
            f"Wall clock: {seconds:.0f} seconds on CPU.",
            "",
        ]
    )
    path = ROOT_REPORTS / "flashcard_tabpfn_trial.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
