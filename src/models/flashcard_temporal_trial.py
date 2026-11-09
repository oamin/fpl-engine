"""How far a highlighted week sits from the week the goal arrived.

Exact overlap treats a spike one gameweek early as a total miss. This run
keeps the locked vision reader and measures two softer views.

The first is the top 10% of a player's own hidden appearances, the same
highlights as the capture score. Each hidden goal week is counted by its
distance in gameweeks to the nearest highlight. Distance 0 is that score.
A same-week other fixture, one gameweek, and two gameweeks are separate.

The second moves probability mass. For each player the actual goals and the
predicted expected goals are normalized on the gameweek axis, and the
1D Wasserstein distance is the mean gameweeks the mass must travel.
A uniform spread and a permutation of the same mass are the comparisons.
A permutation keeps the spikes and destroys the timing.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from src.models.flashcard_calibrated_trial import place_expected
from src.models.flashcard_trial import TEST_SEASON, TRAIN_SEASONS
from src.models.flashcard_squad_trial import (
    CUT_GW,
    EPOCHS,
    SEED,
    SquadSequence,
    SquadVision,
    build_cards,
    build_league,
    capture_table,
    cut_at_fraction,
    fit_clubs,
    predict,
    squad_index,
    train_model,
)

ROOT_REPORTS = Path(__file__).resolve().parents[2] / "reports"
EXACT, SAME_GW, ONE, TWO, REST = range(5)


def earth_mover(positions: np.ndarray, mass_a: np.ndarray, mass_b: np.ndarray) -> float:
    """Gameweeks of transport between two masses on the same fixtures. Empty mass is undefined."""
    left = np.asarray(mass_a, dtype=np.float64)
    right = np.asarray(mass_b, dtype=np.float64)
    if left.sum() <= 0 or right.sum() <= 0:
        return float("nan")
    spots = np.asarray(positions, dtype=np.float64)
    return float(wasserstein_distance(spots, spots, left, right))


def categorize(gameweeks: np.ndarray, goal_index: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Bucket each goal appearance: exact, the other fixture that week, 1, 2, or farther."""
    chosen = np.flatnonzero(selected)
    chosen_gw = gameweeks[chosen]
    buckets = np.empty(len(goal_index), dtype=np.int64)
    for slot, index in enumerate(goal_index):
        if selected[index]:
            buckets[slot] = EXACT
            continue
        if len(chosen_gw) == 0:
            buckets[slot] = REST
            continue
        distance = int(np.abs(int(gameweeks[index]) - chosen_gw).min())
        if distance == 0:
            buckets[slot] = SAME_GW
        elif distance == 1:
            buckets[slot] = ONE
        elif distance == 2:
            buckets[slot] = TWO
        else:
            buckets[slot] = REST
    return buckets


def _shares(groups: list[np.ndarray]) -> dict[str, float]:
    share = np.zeros(5)
    counts = np.zeros(5)
    for buckets in groups:
        bincount = np.bincount(buckets, minlength=5).astype(np.float64)
        share += bincount / bincount.sum()
        counts += bincount
    share /= max(len(groups), 1)
    miss = 1.0 - float(share[EXACT])
    near = float(share[SAME_GW] + share[ONE])
    wide = near + float(share[TWO])
    return {
        "exact": float(share[EXACT]),
        "same_gw": float(share[SAME_GW]),
        "one": float(share[ONE]),
        "two": float(share[TWO]),
        "rest": float(share[REST]),
        "within1": float(share[EXACT] + share[SAME_GW] + share[ONE]),
        "within2": float(share[EXACT] + share[SAME_GW] + share[ONE] + share[TWO]),
        "miss_within1": near / miss if miss else 0.0,
        "miss_within2": wide / miss if miss else 0.0,
        "events": float(counts.sum()),
        "players": float(len(groups)),
    }


def rank_histogram(rows: list[dict], score_key: str, frac: float, seeds: int = 20) -> dict[str, float]:
    """Player-weighted distances from goal weeks to a top slice of one score."""
    groups: list[np.ndarray] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for row in rows:
            score = np.asarray(row[score_key], dtype=np.float64)
            order = np.argsort(-(score + rng.random(len(score)) * 1e-6))
            take = max(1, int(round(frac * len(score))))
            selected = np.zeros(len(score), dtype=bool)
            selected[order[:take]] = True
            groups.append(categorize(row["gw"], np.flatnonzero(row["actual"] > 0), selected))
    summary = _shares(groups)
    summary["events"] /= seeds
    summary["players"] /= seeds
    return summary


def bar_histogram(rows: list[dict], seeds: int = 20) -> tuple[dict[str, float], dict[str, float]]:
    """Distances to the ranked bars, and to the same number of bars on random weeks."""
    placed: list[np.ndarray] = []
    random_groups: list[np.ndarray] = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for row in rows:
            selected = np.asarray(row["drawn"], dtype=bool)
            goals = np.flatnonzero(row["actual"] > 0)
            if seed == 0:
                placed.append(categorize(row["gw"], goals, selected))
            pool = np.flatnonzero(row["minutes"] > 0)
            draw = np.zeros(len(row["gw"]), dtype=bool)
            count = int(selected.sum())
            if count > 0 and len(pool) > 0:
                draw[rng.choice(pool, size=min(count, len(pool)), replace=False)] = True
            random_groups.append(categorize(row["gw"], goals, draw))
    shuffled = _shares(random_groups)
    shuffled["events"] /= seeds
    shuffled["players"] /= seeds
    return _shares(placed), shuffled


def mass_near(gameweeks: np.ndarray, mass: np.ndarray, actual: np.ndarray, window: int) -> float:
    """Share of `mass` sitting within `window` gameweeks of an actual goal."""
    weight = np.asarray(mass, dtype=np.float64)
    if weight.sum() <= 0:
        return float("nan")
    goals = gameweeks[np.asarray(actual) > 0]
    close = np.array([np.min(np.abs(goals - spot)) <= window for spot in gameweeks])
    return float(weight[close].sum() / weight.sum())


def _finite_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": float("nan"), "median": float("nan"), "within1": float("nan"), "within2": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "within1": float(np.mean(arr <= 1.0)),
        "within2": float(np.mean(arr <= 2.0)),
        "n": int(len(arr)),
    }


def transport(rows: list[dict], mass_key: str, seeds: int = 20) -> dict[str, dict[str, float]]:
    """Wasserstein and nearby mass for the model, a flat spread, and a permutation."""
    model_w: list[float] = []
    flat_w: list[float] = []
    mixed_w: list[float] = []
    model_near = np.zeros(3)
    flat_near = np.zeros(3)
    mixed_near = np.zeros(3)
    counted = 0
    for index, row in enumerate(rows):
        actual = row["actual"]
        mass = np.asarray(row[mass_key], dtype=np.float64)
        flat = np.ones(len(mass))
        model_w.append(earth_mover(row["gw"], actual, mass))
        flat_w.append(earth_mover(row["gw"], actual, flat))
        rng = np.random.default_rng(index)
        mixed_draw = []
        mixed_window = np.zeros(3)
        for _ in range(seeds):
            shuffled = mass.copy()
            rng.shuffle(shuffled)
            mixed_draw.append(earth_mover(row["gw"], actual, shuffled))
            for window in range(3):
                mixed_window[window] += mass_near(row["gw"], shuffled, actual, window)
        mixed_w.append(float(np.nanmean(mixed_draw)))
        mixed_window /= seeds
        if not np.isfinite(model_w[-1]):
            continue
        counted += 1
        for window in range(3):
            model_near[window] += mass_near(row["gw"], mass, actual, window)
            flat_near[window] += mass_near(row["gw"], flat, actual, window)
            mixed_near[window] += mixed_window[window]
    scale = max(counted, 1)
    packed = {
        "model": _finite_stats(model_w),
        "uniform": _finite_stats(flat_w),
        "permuted": _finite_stats(mixed_w),
    }
    for name, total in (("model", model_near), ("uniform", flat_near), ("permuted", mixed_near)):
        packed[name]["on"] = float(total[0] / scale)
        packed[name]["near1"] = float(total[1] / scale)
        packed[name]["near2"] = float(total[2] / scale)
    return packed


def player_rows(
    cards: list,
    labels: np.ndarray,
    expected: np.ndarray,
    chance: np.ndarray,
    minutes: np.ndarray,
    drawn: np.ndarray,
    cut_gw: int,
    cuts: list[int] | None = None,
    min_appearances: int = 8,
) -> list[dict]:
    """Hidden appearances for players with at least one actual bar. Same pool as the capture exam at 8."""
    rows = []
    for index, card in enumerate(cards):
        cut = cuts[index] if cuts is not None else cut_gw
        hidden = (card.gw > cut) & (card.gw > 0)
        for row in range(card.n_players):
            cols = np.where(hidden & card.present[row])[0]
            if len(cols) < min_appearances:
                continue
            actual = labels[index, row, cols].astype(np.float64)
            if int((actual > 0).sum()) < 1:
                continue
            rows.append(
                {
                    "gw": card.gw[cols].astype(np.int64),
                    "actual": actual,
                    "minutes": minutes[index, row, cols].astype(np.float64),
                    "expected": expected[index, row, cols].astype(np.float64),
                    "chance": chance[index, row, cols].astype(np.float64),
                    "drawn": drawn[index, row, cols].astype(bool),
                }
            )
    return rows


def _place_all(probs: np.ndarray, minutes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    drawn = np.zeros(minutes.shape, dtype=np.int64)
    for index in range(len(minutes)):
        drawn[index] = place_expected(probs[index], minutes[index], mask[index])
    return drawn


def _mask(cards: list, cut_gw: int, cuts: list[int] | None = None) -> np.ndarray:
    masks = []
    for index, card in enumerate(cards):
        cut = cuts[index] if cuts is not None else cut_gw
        hidden = (card.gw > cut) & (card.gw > 0)
        masks.append(card.present & hidden.reshape(1, -1))
    return np.stack(masks)


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _gw(value: float) -> str:
    return f"{value:.2f}"


def _hist_line(name: str, summary: dict[str, float]) -> str:
    return (
        f"| {name} | {_pct(summary['exact'])} | {_pct(summary['same_gw'])} | {_pct(summary['one'])} | "
        f"{_pct(summary['two'])} | {_pct(summary['rest'])} | {_pct(summary['within1'])} | "
        f"{_pct(summary['miss_within1'])} | {_pct(summary['miss_within2'])} |"
    )


def _mass_line(name: str, stats: dict[str, float]) -> str:
    return (
        f"| {name} | {_gw(stats['mean'])} | {_gw(stats['median'])} | {_pct(stats['within1'])} | "
        f"{_pct(stats['within2'])} | {_pct(stats['on'])} | {_pct(stats['near1'])} | {_pct(stats['near2'])} |"
    )


def _report(lines: list[str]) -> None:
    path = ROOT_REPORTS / "flashcard_temporal_trial.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _window(
    cards: list,
    labels: np.ndarray,
    probs: np.ndarray,
    minutes: np.ndarray,
    cut_gw: int,
    cuts: list[int] | None,
    min_appearances: int,
) -> tuple[list[dict], dict[str, float], dict[str, dict[str, float]]]:
    chance = 1.0 - probs[:, :, :, 0]
    expected = probs @ np.arange(4)
    mask = _mask(cards, cut_gw, cuts)
    drawn = _place_all(probs, minutes, mask)
    rows = player_rows(
        cards, labels, expected, chance, minutes, drawn, cut_gw, cuts, min_appearances
    )
    vision = rank_histogram(rows, "chance", 0.1)
    return rows, vision, transport(rows, "expected")


def main() -> None:
    started = time.perf_counter()
    print("loading temporal cards", flush=True)
    league = build_league(TRAIN_SEASONS + (TEST_SEASON,))
    train = build_cards(TRAIN_SEASONS, league)
    test = build_cards((TEST_SEASON,), league)
    clubs = fit_clubs(train)
    test_squads = squad_index(test)
    n_clubs = max(clubs.values(), default=0) + 1
    print(f"train {len(train)} test {len(test)}", flush=True)
    torch.manual_seed(SEED)
    sequence = SquadSequence(dropout=0.1, n_clubs=n_clubs)
    vision = SquadVision(dropout=0.1, n_clubs=n_clubs)
    train_model(sequence, train, league, np.random.default_rng(SEED), EPOCHS, "sequence", clubs=clubs)
    train_model(vision, train, league, np.random.default_rng(SEED), EPOCHS, "vision", clubs=clubs)
    print("scoring", flush=True)
    goal_p, assist_p = predict(vision, test, league, CUT_GW, "vision", clubs, test_squads)
    capture = capture_table(test, {"vision": 1.0 - goal_p[:, :, :, 0]}, CUT_GW)
    labels = np.stack([card.goals for card in test])
    assist_labels = np.stack([card.assists for card in test])
    minutes = np.stack([card.minutes for card in test])
    rows, top10, mass = _window(test, labels, goal_p, minutes, CUT_GW, None, 8)
    minutes_top = rank_histogram(rows, "minutes", 0.1)
    random_top = rank_histogram([{**row, "flat": np.zeros(len(row["gw"]))} for row in rows], "flat", 0.1)
    top20 = rank_histogram(rows, "chance", 0.2)
    top50 = rank_histogram(rows, "chance", 0.5)
    placed, shuffled = bar_histogram(rows)
    bars = transport(rows, "chance")
    _, assist_top, assist_mass = _window(test, assist_labels, assist_p, minutes, CUT_GW, None, 8)
    appearances = float(np.mean([len(row["gw"]) for row in rows])) if rows else 0.0
    lines = [
        "# Temporal error",
        "",
        "The reader is the locked vision model. A goal week counts as exact only",
        "when that appearance is inside the highlighted set. The next columns",
        "are the other fixture in the same gameweek, one gameweek away, and two",
        "gameweeks away. The shares are the mean across players, which is the",
        "weighting of the capture score. Distance 0 in that weighting is the",
        "top-10% score.",
        "",
        "Wasserstein distance is computed per player on the gameweek axis, after",
        "normalizing the actual goals and the predicted mass. It is the mean",
        "number of gameweeks the mass has to move. Uniform spreads the same",
        "total evenly. Permuted keeps the predicted spikes and shuffles which",
        "week they land on.",
        "",
        f"Train: {', '.join(TRAIN_SEASONS)}. Test: {TEST_SEASON}.",
        f"Gameweek-{CUT_GW} top 10/20/50: "
        + ", ".join(f"{value:.1%}" for value in capture["vision"]),
        f"Players: {capture['n_players']}. Mean hidden appearances: {appearances:.1f}.",
        "Locked default: 24.7%, 41.5%, 80.1%.",
        "",
        "## Top slice of each player's weeks",
        "",
        "Gameweeks 20–38. A player needs eight hidden appearances and one goal.",
        "",
        "| highlights | exact | same GW | 1 GW | 2 GW | 3+ GW | within 1 | misses within 1 | misses within 2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _hist_line("vision top 10%", top10),
        _hist_line("minutes top 10%", minutes_top),
        _hist_line("random top 10%", random_top),
        _hist_line("vision top 20%", top20),
        _hist_line("vision top 50%", top50),
        "",
        "## Probability mass",
        "",
        "Goal counts against expected goals, then goal bars against the probability of a bar.",
        "",
        "| mass | mean GW | median GW | players ≤ 1 | players ≤ 2 | mass on a goal GW | mass within 1 | mass within 2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _mass_line("expected goals", mass["model"]),
        _mass_line("uniform", mass["uniform"]),
        _mass_line("permuted expected goals", mass["permuted"]),
        _mass_line("bar probability", bars["model"]),
        _mass_line("uniform bars", bars["uniform"]),
        _mass_line("permuted bar probability", bars["permuted"]),
        "",
        "## Ranked bars",
        "",
        "The fixture budget from the calibrated decoder. Random draws the same number",
        "of bars on that player's weeks with minutes.",
        "",
        "| bars | exact | same GW | 1 GW | 2 GW | 3+ GW | within 1 | misses within 1 | misses within 2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _hist_line("ranked", placed),
        _hist_line("same count, random weeks", shuffled),
        "",
        "## Assists",
        "",
        f"Players with eight hidden appearances and one assist: {assist_top['players']:.0f}.",
        f"Vision top 10% exact {_pct(assist_top['exact'])}, within 1 {_pct(assist_top['within1'])}, "
        f"and {_pct(assist_top['miss_within1'])} of the misses are within 1.",
        f"Expected-assist transport: model {_gw(assist_mass['model']['mean'])} gameweeks, "
        f"uniform {_gw(assist_mass['uniform']['mean'])}, "
        f"permuted {_gw(assist_mass['permuted']['mean'])}.",
        "",
        "## Remainders from gameweek 1",
        "",
        "| revealed | players | exact | within 1 | misses within 1 | random within 1 | model GW | uniform GW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for revealed, minimum in ((0.8, 4), (0.2, 8)):
        cuts = [cut_at_fraction(card, revealed) for card in test]
        probs = np.zeros_like(goal_p)
        for index, card in enumerate(test):
            one, _assist = predict(vision, [card], league, cuts[index], "vision", clubs, test_squads)
            probs[index] = one[0]
        window_rows, window_top, window_mass = _window(test, labels, probs, minutes, 0, cuts, minimum)
        flat_rows = [{**row, "flat": np.zeros(len(row["gw"]))} for row in window_rows]
        window_random = rank_histogram(flat_rows, "flat", 0.1)
        lines.append(
            f"| {revealed:.0%} | {window_top['players']:.0f} | {_pct(window_top['exact'])} | "
            f"{_pct(window_top['within1'])} | {_pct(window_top['miss_within1'])} | "
            f"{_pct(window_random['within1'])} | {_gw(window_mass['model']['mean'])} | "
            f"{_gw(window_mass['uniform']['mean'])} |"
        )
        print(
            f"reveal {revealed:.0%} exact {window_top['exact']:.3f} "
            f"within1 {window_top['within1']:.3f} mover {window_mass['model']['mean']:.2f}",
            flush=True,
        )
    lines.extend(
        [
            "",
            "The 80% row keeps a player with four hidden appearances, because that hidden",
            "run is short. The 20% row keeps the eight-appearance rule.",
            "",
            "## What the distances say",
            "",
            "On gameweeks 20–38 the top 10% is exact for "
            f"{_pct(top10['exact'])} of a player's goal weeks. Another "
            f"{_pct(top10['same_gw'])} are the other fixture in that gameweek and "
            f"{_pct(top10['one'])} are one gameweek away. Of the weeks outside the top 10%, "
            f"{_pct(top10['miss_within1'])} sit within one gameweek and "
            f"{_pct(top10['miss_within2'])} sit within two. The largest share is farther out: "
            f"{_pct(top10['rest'])} are three or more gameweeks from the nearest highlighted week.",
            "",
            "A random top 10% already leaves "
            f"{_pct(random_top['miss_within1'])} of its misses within one gameweek. "
            f"Minutes leaves {_pct(minutes_top['miss_within1'])}. Vision leaves "
            f"{_pct(top10['miss_within1'])}. The exact week is where vision is ahead of minutes "
            f"({_pct(top10['exact'])} against {_pct(minutes_top['exact'])}). The misses that remain "
            "are spread about as widely as the misses of a minutes ranking.",
            "",
            "The probability mass agrees. Matching expected goals to the actual goals costs "
            f"{_gw(mass['model']['mean'])} gameweeks on average. A flat spread costs "
            f"{_gw(mass['uniform']['mean'])} and a permutation of the same spikes costs "
            f"{_gw(mass['permuted']['mean'])}. For {_pct(mass['model']['within1'])} of players "
            "the move is one gameweek or less. Expected goals and the probability of a bar "
            "have the same mean transport, so a brace is not what moves the mass.",
            "",
            "The ranked fixture bars are thinner. They land on "
            f"{_pct(placed['exact'])} of a player's goal weeks. Of the other weeks, "
            f"{_pct(placed['miss_within1'])} are within one gameweek of that player's bar, against "
            f"{_pct(shuffled['miss_within1'])} when the same number of bars is placed at random. "
            f"{_pct(placed['rest'])} are three or more gameweeks away. This average counts a player "
            "who was given no bar as a miss on every goal week, which is why it sits below the "
            "earlier event-weighted recall.",
            "",
        ]
    )
    seconds = time.perf_counter() - started
    lines.extend(["The capture gate stays at 24.7%.", f"Wall clock: {seconds:.0f} seconds on CPU.", ""])
    _report(lines)
    print(
        f"done in {seconds:.0f}s capture10 {capture['vision'][0]:.3f} "
        f"exact {top10['exact']:.3f} within1 {top10['within1']:.3f} "
        f"miss1 {top10['miss_within1']:.3f} mover {mass['model']['mean']:.2f} "
        f"uniform {mass['uniform']['mean']:.2f} permuted {mass['permuted']['mean']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
