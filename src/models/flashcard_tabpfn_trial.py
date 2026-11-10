"""Pretrained tabular reader for the confirmed squad card.

Each row is one player in one fixture. The columns are that card: the team,
the opponent, home or away, and the Shin de-vigged home, draw, and away
probabilities, then this player's minutes, goals, and assists across the
window. The squad's minutes and positions in the fixture being scored sit on
the same row. A hidden goal or assist is the mask. A column that has been
chopped off the stepped card is empty.

The one-shot pass matches the full painter. Every fixture stays. A cut hides
later goals and assists and leaves minutes, venue, opponents, and odds in
place. The loss rows are the hidden appearances. The stepped pass matches the
stepped painter. The true past stays, the next gameweek keeps minutes, venue,
and odds, the opponent name is blank that week, and every later column is
removed. The loss rows are only that week.

TabPFN is a classifier pretrained on synthetic tables. It is not trained on
these seasons. Concede, regulars, own history, and opponent history stay out.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.flashcard_calibrated_trial import place_expected
from src.models.flashcard_card_score import tally_cards
from src.models.flashcard_odds_trial import column_odds, load_odds
from src.models.flashcard_squad_trial import (
    COLS,
    ROWS,
    SEED,
    apply_cut,
    build_cards,
    build_league,
    cut_at_fraction,
)
from src.models.flashcard_step_trial import chop_future
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
FRACTIONS = (0.2, 0.4, 0.6, 0.8)
CONTEXT = 3_000
BLOCK_CAP = 48
POS_NAME = {0: "GK", 1: "DEF", 2: "MID", 3: "FWD"}
MISSING = "na"


def _slot(index: int) -> str:
    return f"{index:02d}"


def _timeline() -> tuple[str, ...]:
    names: list[str] = []
    for index in range(COLS):
        slot = _slot(index)
        names.extend(
            (
                f"opp_{slot}",
                f"home_{slot}",
                f"odds_h_{slot}",
                f"odds_d_{slot}",
                f"odds_a_{slot}",
                f"min_{slot}",
                f"goal_{slot}",
                f"assist_{slot}",
            )
        )
    return tuple(names)


def _mates() -> tuple[str, ...]:
    names: list[str] = []
    for index in range(ROWS):
        slot = _slot(index)
        names.extend((f"mate_min_{slot}", f"mate_pos_{slot}"))
    return tuple(names)


TIMELINE = _timeline()
MATES = _mates()
FEATURES = ("team", "position", "target") + TIMELINE + MATES
CATEGORICAL = (
    ("team", "position")
    + tuple(name for name in TIMELINE if name.startswith(("opp_", "goal_", "assist_")))
    + tuple(name for name in MATES if name.startswith("mate_pos_"))
)


def position_name(code: int) -> str:
    """The chart's position word. Keepers are GK."""
    return POS_NAME.get(int(code), MISSING)


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


def _token(value: int) -> str:
    return str(int(value))


def _odds_visible(gameweeks: np.ndarray, mode: str, target: int) -> np.ndarray:
    show = gameweeks > 0
    if mode == "step":
        show = show & (gameweeks <= target)
    return show


def _limit_rows(frame: pd.DataFrame, rng: np.random.Generator, cap: int) -> pd.DataFrame:
    """A random slice of the loss rows. The share of events stays as it is."""
    if len(frame) <= cap:
        return frame.reset_index(drop=True)
    pick = rng.choice(frame.index.to_numpy(), size=cap, replace=False)
    return frame.loc[np.sort(pick)].reset_index(drop=True)


def match_probs(card, table: dict | None) -> np.ndarray:
    """Home, draw, and away Shin probabilities. Missing prices are NaN."""
    if table is None:
        return np.full((COLS, 3), np.nan, dtype=np.float64)
    return column_odds(card, table)


def card_rows(
    card,
    league: dict,
    squads: dict,
    probs: np.ndarray,
    mode: str,
    cut: int = 0,
    target: int = 0,
    card_index: int = 0,
) -> pd.DataFrame:
    """Loss rows for one card. One-shot uses `cut`. Stepped uses `target`."""
    if mode == "step":
        view = apply_cut(card, league, target - 1, {}, squads)
        chop_future(view, card.gw, target)
        active = card.present & (card.gw == target).reshape(1, -1)
    else:
        view = apply_cut(card, league, cut, {}, squads)
        hidden = (card.gw > cut) & (card.gw > 0)
        active = card.present & hidden.reshape(1, -1)
    visible_odds = _odds_visible(card.gw, mode, target)
    rows: list[dict[str, object]] = []
    for player in range(ROWS):
        if int(card.position[player]) < 0:
            continue
        fixtures = np.flatnonzero(active[player])
        if len(fixtures) == 0:
            continue
        base = _player_grid(card, view, int(player), probs, visible_odds, mode, target)
        for fixture in fixtures:
            row = dict(base)
            row["target"] = int(fixture)
            row["card_index"] = card_index
            row["player"] = int(player)
            row["fixture"] = int(fixture)
            row["y_goal"] = int(card.goals[player, fixture])
            row["y_assist"] = int(card.assists[player, fixture])
            _mates_at(row, card, view, int(fixture))
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=list(FEATURES) + ["card_index", "player", "fixture", "y_goal", "y_assist"])
    return pd.DataFrame(rows)


def _player_grid(
    card,
    view: dict[str, np.ndarray],
    player: int,
    probs: np.ndarray,
    visible_odds: np.ndarray,
    mode: str,
    target: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "team": card.team,
        "position": position_name(int(card.position[player])),
    }
    for slot in range(COLS):
        name = _slot(slot)
        on = float(view["column_on"][slot]) > 0.5
        if not on:
            row[f"opp_{name}"] = MISSING
            row[f"home_{name}"] = np.nan
            row[f"odds_h_{name}"] = np.nan
            row[f"odds_d_{name}"] = np.nan
            row[f"odds_a_{name}"] = np.nan
            row[f"min_{name}"] = np.nan
            row[f"goal_{name}"] = MISSING
            row[f"assist_{name}"] = MISSING
            continue
        opponent = card.opponent[slot]
        if mode == "step" and int(card.gw[slot]) == target:
            opponent = ""
        row[f"opp_{name}"] = opponent if opponent else MISSING
        row[f"home_{name}"] = float(card.home[slot])
        if visible_odds[slot] and np.isfinite(probs[slot, 0]):
            row[f"odds_h_{name}"] = float(probs[slot, 0])
            row[f"odds_d_{name}"] = float(probs[slot, 1])
            row[f"odds_a_{name}"] = float(probs[slot, 2])
        else:
            row[f"odds_h_{name}"] = np.nan
            row[f"odds_d_{name}"] = np.nan
            row[f"odds_a_{name}"] = np.nan
        if bool(card.present[player, slot]):
            row[f"min_{name}"] = float(card.minutes[player, slot])
            row[f"goal_{name}"] = _token(int(view["goal_token"][player, slot]))
            row[f"assist_{name}"] = _token(int(view["assist_token"][player, slot]))
        else:
            row[f"min_{name}"] = np.nan
            row[f"goal_{name}"] = MISSING
            row[f"assist_{name}"] = MISSING
    return row


def _mates_at(row: dict[str, object], card, view: dict[str, np.ndarray], fixture: int) -> None:
    on = float(view["column_on"][fixture]) > 0.5
    for player in range(ROWS):
        name = _slot(player)
        if not on or int(card.position[player]) < 0 or not bool(card.present[player, fixture]):
            row[f"mate_min_{name}"] = np.nan
            row[f"mate_pos_{name}"] = MISSING
            continue
        row[f"mate_min_{name}"] = float(card.minutes[player, fixture])
        row[f"mate_pos_{name}"] = position_name(int(card.position[player]))


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Model matrix. Labels and bookkeeping stay off it."""
    return frame.loc[:, list(FEATURES)].copy()


def _classifier():
    import os

    from tabpfn import TabPFNClassifier

    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    return TabPFNClassifier(
        device="cpu",
        n_estimators=1,
        fit_mode="fit_with_cache",
        ignore_pretraining_limits=True,
        categorical_features_indices=[FEATURES.index(name) for name in CATEGORICAL],
        random_state=SEED,
        n_jobs=1,
    )


def _fit_channel(context: pd.DataFrame, label: str):
    model = _classifier()
    model.fit(features(context), context[label].to_numpy())
    return model


def _predict(model, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    parts = []
    for start in range(0, len(frame), 1024):
        part = frame.iloc[start : start + 1024]
        proba = np.asarray(model.predict_proba(features(part)))
        parts.append(align_probs(proba, model.classes_))
    return np.vstack(parts)


def _scatter(frame: pd.DataFrame, probs: np.ndarray, n_cards: int) -> np.ndarray:
    painted = np.zeros((n_cards, ROWS, COLS, 4), dtype=np.float64)
    painted[..., 0] = 1.0
    if len(frame) == 0:
        return painted
    painted[
        frame["card_index"].to_numpy(),
        frame["player"].to_numpy(),
        frame["fixture"].to_numpy(),
    ] = probs
    return painted


def _place(probs: np.ndarray, cards: list, cuts: list[int]) -> np.ndarray:
    painted = np.zeros((len(cards), ROWS, COLS), dtype=np.int64)
    for index, card in enumerate(cards):
        hidden = (card.gw > cuts[index]) & (card.gw > 0)
        mask = card.present & hidden.reshape(1, -1)
        painted[index] = place_expected(probs[index], card.minutes, mask)
    return painted


def _line(revealed: float, name: str, channel: str, totals: dict[str, int]) -> str:
    return (
        f"| {revealed:.0%} | {name} | {channel} | {totals['actual']} | {totals['painted']} | "
        f"{totals['exact']} | {totals['actual_within']} | {totals['painted_within']} |"
    )


def _one_shot_context(cards: list, league: dict, squads: dict, odds: dict, rng: np.random.Generator) -> pd.DataFrame:
    frames = []
    for index, card in enumerate(cards):
        probs = match_probs(card, odds)
        for fraction in FRACTIONS:
            block = card_rows(
                card,
                league,
                squads,
                probs,
                "shot",
                cut=cut_at_fraction(card, fraction),
                card_index=index,
            )
            block = _limit_rows(block, rng, BLOCK_CAP)
            if block.empty:
                continue
            block["card_index"] = len(frames)
            frames.append(block)
        if (index + 1) % 20 == 0:
            print(f"one-shot train cards {index + 1}/{len(cards)}", flush=True)
    context = take_cards(frames, rng, CONTEXT)
    print(f"one-shot context rows {len(context)} cards {context['card_index'].nunique()}", flush=True)
    return context


def _step_context(cards: list, league: dict, squads: dict, odds: dict, rng: np.random.Generator) -> pd.DataFrame:
    tasks: list[tuple[int, int]] = []
    for index, card in enumerate(cards):
        seen: set[int] = set()
        for gw in card.gw:
            number = int(gw)
            if number > 0 and number not in seen:
                seen.add(number)
                tasks.append((index, number))
    order = rng.permutation(len(tasks))
    frames = []
    for slot in order:
        index, gw = tasks[int(slot)]
        card = cards[index]
        block = card_rows(
            card,
            league,
            squads,
            match_probs(card, odds),
            "step",
            target=gw,
            card_index=len(frames),
        )
        if block.empty:
            continue
        frames.append(block)
        if sum(len(frame) for frame in frames) >= CONTEXT:
            break
    context = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"step context rows {len(context)} tasks {len(frames)}", flush=True)
    return context


def _prepared_shots(cards: list, league: dict, squads: dict, odds: dict) -> list[tuple[float, list[int], pd.DataFrame]]:
    prepared = []
    for revealed in FRACTIONS:
        cuts = [cut_at_fraction(card, revealed) for card in cards]
        frames = [
            card_rows(card, league, squads, match_probs(card, odds), "shot", cut=cut, card_index=index)
            for index, (card, cut) in enumerate(zip(cards, cuts, strict=True))
        ]
        frame = pd.concat([block for block in frames if not block.empty], ignore_index=True)
        prepared.append((revealed, cuts, frame))
    return prepared


def _score_shot(name: str, context: pd.DataFrame, cards: list, league: dict, squads: dict, odds: dict, lines: list[str]) -> dict[str, int] | None:
    labels = np.stack([card.goals for card in cards])
    assist_labels = np.stack([card.assists for card in cards])
    prepared = _prepared_shots(cards, league, squads, odds)
    saved: dict[tuple[str, float], np.ndarray] = {}
    for label, channel in (("y_goal", "goals"), ("y_assist", "assists")):
        print(f"fitting {name} {channel}", flush=True)
        model = _fit_channel(context, label)
        for revealed, _cuts, frame in prepared:
            print(f"predict {name} {channel} {revealed:.0%} rows {len(frame)}", flush=True)
            saved[(channel, revealed)] = _scatter(frame, _predict(model, frame), len(cards))
        del model
        gc.collect()
    goal_at_20 = None
    actuals = {"goals": labels, "assists": assist_labels}
    for revealed, cuts, _frame in prepared:
        for channel in ("goals", "assists"):
            totals = tally_cards(cards, actuals[channel], _place(saved[(channel, revealed)], cards, cuts), 0, cuts)
            lines.append(_line(revealed, name, channel, totals))
            if revealed == 0.2 and channel == "goals":
                goal_at_20 = totals
            print(
                f"{name} {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )
    return goal_at_20


def _score_steps(name: str, context: pd.DataFrame, cards: list, league: dict, squads: dict, odds: dict, lines: list[str]) -> None:
    labels = np.stack([card.goals for card in cards])
    assist_labels = np.stack([card.assists for card in cards])
    by_gw: dict[int, list[int]] = {}
    for index, card in enumerate(cards):
        seen: set[int] = set()
        for gw in card.gw:
            number = int(gw)
            if number > 0 and number not in seen:
                seen.add(number)
                by_gw.setdefault(number, []).append(index)
    prepared = []
    for gw, indexes in sorted(by_gw.items()):
        frames = [
            card_rows(
                cards[index],
                league,
                squads,
                match_probs(cards[index], odds),
                "step",
                target=gw,
                card_index=index,
            )
            for index in indexes
        ]
        frame = pd.concat([block for block in frames if not block.empty], ignore_index=True)
        prepared.append((gw, indexes, frame))
    painted = {
        "goals": np.zeros((len(cards), ROWS, COLS), dtype=np.int64),
        "assists": np.zeros((len(cards), ROWS, COLS), dtype=np.int64),
    }
    for label, channel in (("y_goal", "goals"), ("y_assist", "assists")):
        print(f"fitting {name} {channel}", flush=True)
        model = _fit_channel(context, label)
        for gw, indexes, frame in prepared:
            print(f"predict {name} {channel} gw {gw} rows {len(frame)}", flush=True)
            probs = _scatter(frame, _predict(model, frame), len(cards))
            for slot in indexes:
                card = cards[slot]
                mask = card.present & (card.gw == gw).reshape(1, -1)
                painted[channel][slot] += place_expected(probs[slot], card.minutes, mask)
        del model
        gc.collect()
    actuals = {"goals": labels, "assists": assist_labels}
    for revealed in FRACTIONS:
        cuts = [cut_at_fraction(card, revealed) for card in cards]
        for channel in ("goals", "assists"):
            totals = tally_cards(cards, actuals[channel], painted[channel], 0, cuts)
            lines.append(_line(revealed, name, channel, totals))
            print(
                f"{name} {revealed:.0%} {channel} painted {totals['painted']} "
                f"exact {totals['exact']} within {totals['actual_within']}",
                flush=True,
            )


def _verdict(goal_at_20: dict[str, int] | None) -> list[str]:
    if goal_at_20 is None:
        raise RuntimeError("the 20% goal cut did not score")
    exact = goal_at_20["exact"]
    within = goal_at_20["actual_within"]
    painted = goal_at_20["painted"]
    actual = goal_at_20["actual"]
    beats = exact > 190 and within > 258
    lines = [
        "",
        "At 20% revealed the one-shot reader puts "
        f"{exact} goal bars on the same appearance",
        f"and {within} within one gameweek. It paints {painted} goal bars against {actual} actual.",
        "The gate puts 190 and 258.",
    ]
    if beats:
        lines.append("Both counts are ahead of the gate.")
    else:
        lines.append("The gate stays the stripped card measured at 15:53.")
    return lines


def _write(lines: list[str]) -> None:
    path = ROOT_REPORTS / "flashcard_tabpfn_trial.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    print("loading tabular cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    squads: dict = {}
    odds = load_odds()
    lines = [
        "# Tabular squad card",
        "",
        "Each row is one player in one fixture of the confirmed card. The",
        "features are the header across the window and that player's minutes,",
        "goals, and assists. The header is the team, the opponent, home or away,",
        "and the Shin de-vigged home, draw, and away probabilities. The squad's",
        "minutes and positions in the fixture being scored are on the same row.",
        "A hidden event is the mask. Concede, regulars, own history, and",
        "opponent history stay out.",
        "",
        "The one-shot reader keeps every fixture. A cut hides later goals and",
        "assists and leaves minutes, venue, opponents, and odds in place.",
        "The stepped reader keeps the true past, keeps minutes, venue, and odds",
        "on the next gameweek, blanks that week's opponent, and drops every",
        "later column. Its loss is only that week.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        "",
        "The gate, at 20% revealed, puts 190 goal bars on the same appearance",
        "and 258 within one gameweek.",
        "",
        "| revealed | model | channel | actual bars | painted bars | same appearance | actual within 1 GW | painted within 1 GW |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    shot = _one_shot_context(train, league, squads, odds, np.random.default_rng(SEED))
    goal_at_20 = _score_shot("one-shot", shot, test, league, squads, odds, lines)
    _write(lines)
    step = _step_context(train, league, squads, odds, np.random.default_rng(SEED + 1))
    _score_steps("stepped", step, test, league, squads, odds, lines)
    seconds = time.perf_counter() - started
    lines.extend(_verdict(goal_at_20))
    lines.extend(["", f"Wall clock: {seconds:.0f} seconds on CPU.", ""])
    _write(lines)
    print(f"done in {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
