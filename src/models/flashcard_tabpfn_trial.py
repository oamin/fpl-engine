"""Pretrained tabular reader for the same card the vision gate paints.

Each row is one player in one fixture after the cut. The columns are the
facts the stripped card already shows: position, minutes, venue, the club's
recent scoring rate, that player's revealed goals and assists, and who is
on the pitch. Concede, regulars, and the two history rows stay out.

TabPFN is a classifier pretrained on synthetic tables. It is not trained on
these seasons. Two classifiers, one for goals and one for assists, read the
training rows and return a probability for 0, 1, 2, and 3. Bars are placed
with the same fixture budget as the gate.
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
    ROWS,
    SEED,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
    fit_clubs,
    squad_index,
)
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
CATEGORICAL = ("position", "club", "opponent")
COLUMNS = (
    "position",
    "minutes",
    "home",
    "own_form",
    "own_form_missing",
    "games",
    "goals_so_far",
    "assists_so_far",
    "minutes_so_far",
    "goals_per_90",
    "assists_per_90",
    "team_goals_so_far",
    "mates_playing",
    "opp_playing",
    "club",
    "opponent",
)


def _prefix(card, cut: int) -> dict[str, np.ndarray]:
    """Totals from fixtures the cut has already revealed."""
    visible = (card.gw > 0) & (card.gw <= cut)
    present = card.present[:, visible]
    games = present.sum(axis=1).astype(np.float64)
    minutes = np.where(present, card.minutes[:, visible], 0.0).sum(axis=1)
    goals = np.where(present, card.goals[:, visible], 0).sum(axis=1).astype(np.float64)
    assists = np.where(present, card.assists[:, visible], 0).sum(axis=1).astype(np.float64)
    return {
        "games": games,
        "minutes": minutes,
        "goals": goals,
        "assists": assists,
        "team_goals": float(goals.sum()),
    }


def _per_90(total: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    out = np.zeros_like(total, dtype=np.float64)
    played = minutes > 0
    out[played] = total[played] / (minutes[played] / 90.0)
    return out


def table_at(cards: list, league: dict, clubs: dict, squads: dict, cuts: list[int]) -> pd.DataFrame:
    """One row per hidden appearance. Labels are the capped event counts."""
    rows: list[dict[str, float]] = []
    for index, (card, cut) in enumerate(zip(cards, cuts, strict=True)):
        view = apply_cut(card, league, cut, clubs, squads)
        prefix = _prefix(card, cut)
        goals_per_90 = _per_90(prefix["goals"], prefix["minutes"])
        assists_per_90 = _per_90(prefix["assists"], prefix["minutes"])
        hidden = (card.gw > cut) & (card.gw > 0)
        mates = (card.minutes > 0).sum(axis=0)
        opp_playing = (view["opp_minutes"] > 0).sum(axis=0)
        club_id = int(clubs.get(card.team, 0))
        for player in range(card.n_players):
            if int(card.position[player]) < 0:
                continue
            for col in np.flatnonzero(hidden & card.present[player]):
                opponent = card.opponent[int(col)]
                rows.append(
                    {
                        "card_index": index,
                        "player": player,
                        "fixture": int(col),
                        "position": int(card.position[player]),
                        "minutes": float(card.minutes[player, col]),
                        "home": float(card.home[col]),
                        "own_form": float(view["own_form"][col]),
                        "own_form_missing": float(view["own_form_missing"][col]),
                        "games": float(prefix["games"][player]),
                        "goals_so_far": float(prefix["goals"][player]),
                        "assists_so_far": float(prefix["assists"][player]),
                        "minutes_so_far": float(prefix["minutes"][player]),
                        "goals_per_90": float(goals_per_90[player]),
                        "assists_per_90": float(assists_per_90[player]),
                        "team_goals_so_far": prefix["team_goals"],
                        "mates_playing": float(mates[col]),
                        "opp_playing": float(opp_playing[col]),
                        "club": club_id,
                        "opponent": int(clubs.get(opponent, 0)) if opponent else 0,
                        "y_goal": int(card.goals[player, col]),
                        "y_assist": int(card.assists[player, col]),
                    }
                )
    return pd.DataFrame(rows)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, list(COLUMNS)].copy()


def align_probs(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map classifier columns onto counts 0, 1, 2, and 3."""
    out = np.zeros((len(proba), 4), dtype=np.float64)
    for slot, label in enumerate(classes):
        out[:, int(label)] = proba[:, slot]
    total = out.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return out / total


CONTEXT = 5_000


def stratified_sample(frame: pd.DataFrame, label: str, rng: np.random.Generator) -> pd.DataFrame:
    """A sample that keeps each event count's share. The context is 5,000 rows."""
    if len(frame) <= CONTEXT:
        return frame
    labels = frame[label].to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    raw = counts / counts.sum() * CONTEXT
    take = np.floor(raw).astype(np.int64)
    leftover = CONTEXT - int(take.sum())
    if leftover > 0:
        for index in np.argsort(-(raw - take))[:leftover]:
            take[index] += 1
    parts = []
    for label_value, size in zip(classes, take, strict=True):
        pool = frame.index[labels == label_value].to_numpy()
        chosen = rng.choice(pool, size=int(min(size, len(pool))), replace=False)
        parts.append(frame.loc[chosen])
    return pd.concat(parts, ignore_index=True)


def _classifier():
    import os

    from tabpfn import TabPFNClassifier

    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    return TabPFNClassifier(
        device="cpu",
        n_estimators=2,
        fit_mode="low_memory",
        ignore_pretraining_limits=True,
        categorical_features_indices=[COLUMNS.index(name) for name in CATEGORICAL],
        random_state=SEED,
        n_jobs=1,
    )


def _predict_proba(model, frame: pd.DataFrame) -> np.ndarray:
    matrix = features(frame)
    parts = []
    for start in range(0, len(matrix), 1000):
        parts.append(model.predict_proba(matrix.iloc[start : start + 1000]))
        print(f"  predicted {min(start + 1000, len(matrix))}/{len(matrix)}", flush=True)
    return np.vstack(parts)


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


def predict_cut(
    goal_model,
    assist_model,
    cards: list,
    league: dict,
    clubs: dict,
    squads: dict,
    cuts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    frame = table_at(cards, league, clubs, squads, cuts)
    goal_p = align_probs(_predict_proba(goal_model, frame), goal_model.classes_)
    assist_p = align_probs(_predict_proba(assist_model, frame), assist_model.classes_)
    return _scatter(frame, goal_p, len(cards)), _scatter(frame, assist_p, len(cards))


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


def main() -> None:
    started = time.perf_counter()
    print("loading tabular cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    squads = squad_index(train + test)
    frames = [
        table_at(train, league, clubs, squads, [cut_at_fraction(card, fraction) for card in train])
        for fraction in FRACTIONS
    ]
    train_frame = pd.concat(frames, ignore_index=True)
    print(
        f"train rows {len(train_frame)} goal events {(train_frame['y_goal'] > 0).sum()} "
        f"assist events {(train_frame['y_assist'] > 0).sum()}",
        flush=True,
    )
    rng = np.random.default_rng(SEED)
    goal_rows = stratified_sample(train_frame, "y_goal", rng)
    assist_rows = stratified_sample(train_frame, "y_assist", rng)
    print(f"goal context {len(goal_rows)} assist context {len(assist_rows)}", flush=True)
    print("fitting goal classifier", flush=True)
    goal_model = _classifier()
    goal_model.fit(features(goal_rows), goal_rows["y_goal"].to_numpy())
    print("fitting assist classifier", flush=True)
    assist_model = _classifier()
    assist_model.fit(features(assist_rows), assist_rows["y_assist"].to_numpy())
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    goal_at_20: dict[str, int] | None = None
    lines = [
        "# Pretrained tabular reader",
        "",
        "TabPFN v2 is a classifier pretrained on synthetic tables. Its public",
        "checkpoint is the one used here. Each target gets a stratified sample",
        "of 5,000 rows from 2022-23, 2023-24, and 2024-25, at the 20%, 40%,",
        "60%, and 80% cuts, so the share of 0, 1, 2, and 3 stays as it is.",
        "Concede, regulars, own history, and opponent history are not columns.",
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
        goal_p, assist_p = predict_cut(goal_model, assist_model, test, league, clubs, squads, cuts)
        for channel, probs, actual in (
            ("goals", goal_p, labels),
            ("assists", assist_p, assist_labels),
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
